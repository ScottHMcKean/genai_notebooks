# Databricks notebook source
# MAGIC %md
# MAGIC # 04 · Benchmark an existing embedding endpoint (tokens/s)
# MAGIC
# MAGIC Standalone throughput benchmark, decoupled from deployment (endpoint spin-up can exceed the
# MAGIC notebook `create_and_wait` timeout). Point it at any READY embedding Model Serving endpoint that
# MAGIC takes `{"dataframe_records": [{"text": ...}], "params": {"batch_size": N}}` and it sweeps request
# MAGIC batch size × client concurrency, computing tokens/s from the corpus's ground-truth token counts.
# MAGIC
# MAGIC > Runs on **serverless CPU** — it only queries the endpoint over HTTP. Set `endpoint`, `gpu`, `path`.

# COMMAND ----------

# MAGIC %pip install --quiet --upgrade "mlflow>=3.1" "databricks-sdk>=0.102.0" "transformers>=4.48"
# MAGIC %restart_python

# COMMAND ----------

dbutils.widgets.text("endpoint", "shm-bioclin-mbert-st-a10", "endpoint name")
dbutils.widgets.dropdown("gpu", "A10", ["A10", "T4"], "GPU label")
dbutils.widgets.text("path", "st_pyfunc_serving", "path label for results")
# Cap input length to N tokens (client-side). Needed for the vLLM endpoint, which REJECTS inputs
# longer than max_model_len (512) rather than truncating like sentence-transformers/TEI. Leave blank
# to send full-length inputs.
dbutils.widgets.text("max_tokens", "", "truncate inputs to N tokens (blank=off)")

# COMMAND ----------

CATALOG, SCHEMA = "shm_skunkworks_catalog", "genai"
CORPUS_TABLE = f"{CATALOG}.{SCHEMA}.pubmedqa_corpus"
RESULTS_TABLE = f"{CATALOG}.{SCHEMA}.embedding_bench_results"

ENDPOINT_NAME = dbutils.widgets.get("endpoint")
GPU = dbutils.widgets.get("gpu")
PATH = dbutils.widgets.get("path")
print(f"endpoint={ENDPOINT_NAME} | gpu={GPU} | path={PATH}")

# COMMAND ----------

# Wait for the endpoint to be READY (poll up to ~50 min) — decoupled from the deploy notebook.
import time
from databricks.sdk import WorkspaceClient
w = WorkspaceClient()
deadline = time.time() + 3000
while time.time() < deadline:
    e = w.serving_endpoints.get(ENDPOINT_NAME)
    ready = e.state.ready.value if e.state and e.state.ready else "?"
    cu = e.state.config_update.value if e.state and e.state.config_update else "?"
    print(time.strftime("%T"), "ready=", ready, "config_update=", cu)
    if ready == "READY":
        break
    time.sleep(30)
assert ready == "READY", f"endpoint not READY: {ready}/{cu}"

# COMMAND ----------

import time, json, numpy as np, pandas as pd
from mlflow.deployments import get_deploy_client
from concurrent.futures import ThreadPoolExecutor

client = get_deploy_client("databricks")

corpus_pdf = spark.table(CORPUS_TABLE).select("text", "n_tokens").toPandas()
TEXTS = corpus_pdf["text"].tolist()[:4000]
TOK_BY_TEXT = dict(zip(corpus_pdf["text"], corpus_pdf["n_tokens"]))

# Optional client-side truncation to N tokens (for vLLM, which rejects > max_model_len).
MAX_TOKENS = dbutils.widgets.get("max_tokens").strip()
if MAX_TOKENS:
    N = int(MAX_TOKENS)
    from transformers import AutoTokenizer
    SNAPSHOT_DIR = f"/Volumes/{CATALOG}/{SCHEMA}/huggingface/bioclinical_modernbert"
    tok = AutoTokenizer.from_pretrained(SNAPSHOT_DIR)
    def _trunc(t):
        ids = tok.encode(t, add_special_tokens=True, truncation=True, max_length=N)
        return tok.decode(ids, skip_special_tokens=True)
    orig_tokens = dict(TOK_BY_TEXT)
    TEXTS = [_trunc(t) for t in TEXTS]
    # tokens/s should reflect tokens ACTUALLY processed = min(orig, N).
    TOK_BY_TEXT = {t: min(int(orig_tokens[o]), N) for o, t in zip(corpus_pdf["text"].tolist()[:len(TEXTS)], TEXTS)}
    print(f"truncated inputs to {N} tokens")
print("bench rows:", len(TEXTS))

def embed_batch(texts):
    t0 = time.perf_counter()
    client.predict(endpoint=ENDPOINT_NAME,
                   inputs={"dataframe_records": [{"text": t} for t in texts], "params": {"batch_size": 64}})
    return time.perf_counter() - t0

print("warmup:", round(embed_batch(TEXTS[:8]), 3), "s")

# COMMAND ----------

def run_benchmark(texts, batch_size, concurrency):
    batches = [texts[i:i+batch_size] for i in range(0, len(texts), batch_size)]
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        lat = list(ex.map(embed_batch, batches))
    wall = time.perf_counter() - t0
    total_tokens = int(sum(TOK_BY_TEXT[t] for t in texts))
    lat = np.array(lat)
    return {"batch_size": batch_size, "concurrency": concurrency, "rows": len(texts),
            "total_tokens": total_tokens, "wall_s": round(wall, 2),
            "tokens_per_s": round(total_tokens / wall, 1), "rows_per_s": round(len(texts) / wall, 1),
            "p50_ms": round(float(np.percentile(lat, 50))*1000, 1),
            "p95_ms": round(float(np.percentile(lat, 95))*1000, 1),
            "p99_ms": round(float(np.percentile(lat, 99))*1000, 1)}

results = []
for bs in [16, 32, 64]:
    for conc in [1, 4, 8]:
        r = run_benchmark(TEXTS, bs, conc)
        print(r)
        results.append(r)

bench_df = pd.DataFrame(results).sort_values("tokens_per_s", ascending=False)
display(bench_df)

# COMMAND ----------

from datetime import datetime, timezone
rows = [{
    "run_ts": datetime.now(timezone.utc), "path": PATH, "gpu": GPU,
    "detail": f"endpoint:{ENDPOINT_NAME}", "batch_size": int(r["batch_size"]),
    "concurrency": int(r["concurrency"]), "rows": int(r["rows"]), "total_tokens": int(r["total_tokens"]),
    "wall_s": float(r["wall_s"]), "tokens_per_s": float(r["tokens_per_s"]), "rows_per_s": float(r["rows_per_s"]),
    "p50_ms": float(r["p50_ms"]), "p95_ms": float(r["p95_ms"]), "p99_ms": float(r["p99_ms"]),
} for r in results]
spark.createDataFrame(pd.DataFrame(rows)).write.mode("append").saveAsTable(RESULTS_TABLE)

best = bench_df.iloc[0]
headline = (f"[{PATH} / {GPU}] peak {best.tokens_per_s:,.0f} tokens/s ({best.rows_per_s:,.0f} rows/s) "
            f"at batch_size={int(best.batch_size)}, concurrency={int(best.concurrency)}; "
            f"p50 {best.p50_ms}ms p95 {best.p95_ms}ms")
print(headline)
dbutils.notebook.exit(json.dumps({"endpoint": ENDPOINT_NAME, "gpu": GPU,
                                  "peak_tokens_per_s": float(best.tokens_per_s), "headline": headline}))
