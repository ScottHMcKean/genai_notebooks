# Databricks notebook source
# MAGIC %md
# MAGIC # VS Bench -- 04: Latency Benchmark
# MAGIC
# MAGIC For each `(scale, model, backend)`: warmup + `BENCH_QUERIES` queries.
# MAGIC Records `embed_ms`, `search_ms`, `total_rtt_ms` per query + pg BUFFERS.
# MAGIC Aggregates p50/p90/p99 and flags pgvector disk spill.

# COMMAND ----------

# MAGIC %pip install --quiet \
# MAGIC   databricks-vectorsearch>=0.44.0 databricks-sdk>=0.49.0 \
# MAGIC   "psycopg[binary,pool]>=3.2.0"
# MAGIC %restart_python

# COMMAND ----------

import datetime as dt
import os, sys
from time import perf_counter

import pandas as pd
import psycopg
from databricks.sdk import WorkspaceClient
from mlflow.deployments import get_deploy_client

sys.path.insert(0, os.getcwd())
import config as C

w = WorkspaceClient()
api = w.api_client
mc = get_deploy_client("databricks")

BRANCH_PATH = f"projects/{C.LAKEBASE_PROJECT}/branches/{C.LAKEBASE_BRANCH}"
ENDPOINT_PATH = f"{BRANCH_PATH}/endpoints/{C.LAKEBASE_ENDPOINT}"

print(f"scales={list(C.active_scales())} models={C.MODELS} "
      f"queries={C.BENCH_QUERIES} warmup={C.BENCH_WARMUP} topk={C.BENCH_TOPK}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Queries (held-out rows, id >= 900_000)

# COMMAND ----------

queries = (spark.table(f"{C.CATALOG}.{C.SCHEMA}.dbpedia_source")
           .where("id >= 900000").orderBy("id").limit(C.BENCH_QUERIES)
           .select("text").toPandas()["text"].tolist())
print(f"Loaded {len(queries)} queries")

# COMMAND ----------

# MAGIC %md
# MAGIC ## pg connection

# COMMAND ----------

eps_resp = api.do("GET", f"/api/2.0/postgres/{BRANCH_PATH}/endpoints")
eps = eps_resp.get("endpoints", []) if isinstance(eps_resp, dict) else (eps_resp or [])
host = eps[0]["status"]["hosts"]["host"]
tok = api.do("POST", "/api/2.0/postgres/credentials",
             body={"endpoint": ENDPOINT_PATH})["token"]
email = w.current_user.me().user_name
pg = psycopg.connect(host=host, port=5432, dbname=C.LAKEBASE_DB, user=email,
                     password=tok, sslmode="require", autocommit=True)
with pg.cursor() as cur:
    cur.execute(f"SET hnsw.ef_search = {C.HNSW_EF_SEARCH}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## One call per operation -- keep the shapes honest

# COMMAND ----------

def embed(model, text):
    ep = C.PPT_MODEL if model == "ppt" else C.PT_ENDPOINT
    t0 = perf_counter()
    if model == "ppt":
        resp = mc.predict(endpoint=ep, inputs={"input": [text]})
        ms = (perf_counter() - t0) * 1000
        return resp["data"][0]["embedding"], ms, int(resp["usage"]["prompt_tokens"])
    resp = mc.predict(endpoint=ep, inputs={"dataframe_records": [{"text": text}]})
    ms = (perf_counter() - t0) * 1000
    vec = resp["predictions"][0] if isinstance(resp, dict) and "predictions" in resp else resp[0]
    return vec, ms, max(1, len(text) // 4)  # char/4 ~ token count for PT

def search_vs(scale, model, vec):
    t0 = perf_counter()
    w.vector_search_indexes.query_index(
        index_name=C.vs_index_name(scale, model),
        query_vector=vec, columns=["id"], num_results=C.BENCH_TOPK,
    )
    return {"search_ms": (perf_counter() - t0) * 1000}

def search_pg(scale, model, vec):
    tbl = C.pg_table_name(scale, model)
    lit = "[" + ",".join(f"{float(x):.6f}" for x in vec) + "]"
    with pg.cursor() as cur:
        t0 = perf_counter()
        cur.execute(
            f"SELECT id FROM {tbl} ORDER BY embedding <=> %s::vector LIMIT {C.BENCH_TOPK}",
            (lit,),
        )
        cur.fetchall()
        wall_ms = (perf_counter() - t0) * 1000

        cur.execute(
            f"EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) "
            f"SELECT id FROM {tbl} ORDER BY embedding <=> %s::vector LIMIT {C.BENCH_TOPK}",
            (lit,),
        )
        plan = cur.fetchone()[0][0]

    def _sum(p, key):
        return int(p.get(key, 0) or 0) + sum(_sum(c, key) for c in (p.get("Plans") or []))
    root = plan.get("Plan", {})
    return {
        "search_ms": wall_ms,
        "pg_exec_ms": float(plan.get("Execution Time", 0.0)),
        "pg_shared_hit": _sum(root, "Shared Hit Blocks"),
        "pg_shared_read": _sum(root, "Shared Read Blocks"),
    }

BACKENDS = {"vs": search_vs, "pg": search_pg}

# COMMAND ----------

# MAGIC %md
# MAGIC ## Run warmup + benchmark

# COMMAND ----------

rows = []
for scale, model in C.combos():
    for backend, search in BACKENDS.items():
        print(f"  {scale}/{model}/{backend}: warmup + bench")
        for q in queries[:C.BENCH_WARMUP]:
            vec, _, _ = embed(model, q)
            search(scale, model, vec)
        for i, q in enumerate(queries):
            vec, embed_ms, tokens = embed(model, q)
            srch = search(scale, model, vec)
            rows.append({
                "scale": scale, "model": model, "backend": backend,
                "query_idx": i, "tokens": tokens, "embed_ms": embed_ms,
                "tokens_per_sec": tokens / (embed_ms / 1000) if embed_ms else None,
                "total_rtt_ms": embed_ms + srch["search_ms"], **srch,
            })

results = pd.DataFrame(rows)
stamp = dt.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
out_tbl = f"{C.CATALOG}.{C.SCHEMA}.vs_bench_results_{stamp}"
spark.createDataFrame(results).write.mode("overwrite").saveAsTable(out_tbl)
print(f"{len(results):,} rows -> {out_tbl}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Aggregate

# COMMAND ----------

def q(p): return lambda s: s.quantile(p)

agg = (results.groupby(["scale", "backend", "model"])
       .agg(embed_p50=("embed_ms", q(0.50)), embed_p99=("embed_ms", q(0.99)),
            search_p50=("search_ms", q(0.50)), search_p90=("search_ms", q(0.90)),
            search_p99=("search_ms", q(0.99)),
            rtt_p50=("total_rtt_ms", q(0.50)), rtt_p99=("total_rtt_ms", q(0.99)),
            tps_mean=("tokens_per_sec", "mean"))
       .reset_index().round(1))
print(agg.to_string(index=False))
try:
    display(agg)  # noqa: F821
except NameError:
    pass

# COMMAND ----------

# MAGIC %md
# MAGIC ## pgvector spill check

# COMMAND ----------

pg_rows = results[results["backend"] == "pg"]
if not pg_rows.empty:
    spill = (pg_rows.groupby(["scale", "model"])
             .agg(shared_hit_mean=("pg_shared_hit", "mean"),
                  shared_read_mean=("pg_shared_read", "mean"),
                  shared_read_max=("pg_shared_read", "max"))
             .reset_index().round(2))
    spill["spilling"] = spill["shared_read_mean"] > 1.0
    print(spill.to_string(index=False))
    if spill["spilling"].any():
        print("WARNING: HNSW spilling to disk. Scale Lakebase up or reduce scale.")
