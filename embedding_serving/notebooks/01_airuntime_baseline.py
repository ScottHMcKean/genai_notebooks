# Databricks notebook source
# /// script
# [tool.databricks.environment]
# base_environment = "databricks_ai_v5"
# ///
# MAGIC %md
# MAGIC # 01 · AI Runtime baseline — raw encode tokens/s
# MAGIC
# MAGIC The **runtime ceiling**: load `NeuML/bioclinical-modernbert-base-embeddings` directly on the GPU
# MAGIC with sentence-transformers and measure how fast it can encode the PubMedQA corpus, with **no
# MAGIC serving/network overhead**. This is the number every serving path (vLLM, TEI) is measured against.
# MAGIC
# MAGIC Runs on both GPUs:
# MAGIC - **A10** — serverless GPU (this notebook's `databricks_ai_v5` env, `GPU_1xA10`); gets Flash-Attention 2.
# MAGIC - **T4** — attach to a classic single-node `g4dn.xlarge` cluster; falls back to SDPA (Turing has no FA2).
# MAGIC
# MAGIC Set the `gpu` widget to label the run correctly. Attention impl is auto-detected (FA2 if the GPU +
# MAGIC `flash-attn` support it, else SDPA).

# COMMAND ----------

# MAGIC %pip install --quiet --upgrade "sentence-transformers>=3.0" "transformers>=4.48" "mlflow>=3.1"
# MAGIC %restart_python

# COMMAND ----------

dbutils.widgets.dropdown("gpu", "A10", ["A10", "T4"], "GPU label")

# COMMAND ----------

# MAGIC %sh nvidia-smi

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration

# COMMAND ----------

CATALOG, SCHEMA = "shm_skunkworks_catalog", "genai"
SNAPSHOT_DIR = f"/Volumes/{CATALOG}/{SCHEMA}/huggingface/bioclinical_modernbert"
CORPUS_TABLE = f"{CATALOG}.{SCHEMA}.pubmedqa_corpus"
RESULTS_TABLE = f"{CATALOG}.{SCHEMA}.embedding_bench_results"

GPU = dbutils.widgets.get("gpu")
MAX_SEQ_LEN = 512          # cap sequence length for a stable, comparable throughput number
BATCH_SIZES = [16, 32, 64, 128]

print(f"gpu={GPU} | model={SNAPSHOT_DIR}")

# COMMAND ----------

import time, json
import numpy as np
import pandas as pd
import torch

print("cuda:", torch.cuda.is_available(), "| device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")
print("capability:", torch.cuda.get_device_capability(0) if torch.cuda.is_available() else None)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Pick the best attention implementation the GPU supports
# MAGIC ModernBERT prefers Flash-Attention 2 (Ampere SM 8.0+). On T4 (SM 7.5) FA2 is unavailable, so we
# MAGIC fall back to PyTorch SDPA. We probe once and record which impl actually loaded.

# COMMAND ----------

def pick_attn_impl():
    cap = torch.cuda.get_device_capability(0)[0] if torch.cuda.is_available() else 0
    if cap >= 8:  # Ampere or newer
        try:
            import flash_attn  # noqa: F401
            return "flash_attention_2"
        except Exception:
            return "sdpa"
    return "sdpa"  # Turing (T4) and below

ATTN_IMPL = pick_attn_impl()
print("attn_implementation:", ATTN_IMPL)

# COMMAND ----------

from sentence_transformers import SentenceTransformer

try:
    model = SentenceTransformer(
        SNAPSHOT_DIR, device="cuda",
        model_kwargs={"attn_implementation": ATTN_IMPL, "torch_dtype": torch.bfloat16 if GPU == "A10" else torch.float16},
    )
except Exception as e:
    print(f"load with {ATTN_IMPL} failed ({e}); retrying with sdpa/fp32")
    ATTN_IMPL = "sdpa"
    model = SentenceTransformer(SNAPSHOT_DIR, device="cuda", model_kwargs={"attn_implementation": "sdpa"})

model.max_seq_length = MAX_SEQ_LEN
EMBED_DIMS = model.get_sentence_embedding_dimension()
print("loaded | embed dims:", EMBED_DIMS, "| max_seq_len:", model.max_seq_length)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load the corpus + its precomputed token counts

# COMMAND ----------

corpus_pdf = spark.table(CORPUS_TABLE).select("text", "n_tokens").toPandas()
TEXTS = corpus_pdf["text"].tolist()
TOTAL_TOKENS = int(corpus_pdf["n_tokens"].sum())
print(f"corpus rows: {len(TEXTS):,} | total tokens: {TOTAL_TOKENS:,} | mean: {corpus_pdf['n_tokens'].mean():.1f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Smoke test

# COMMAND ----------

v = model.encode(["acute myocardial infarction", "shortness of breath"], normalize_embeddings=True)
print("dims:", v.shape[1], "| cosine:", float(np.dot(v[0], v[1])))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Benchmark: sweep batch size, measure encode tokens/s
# MAGIC Pure on-GPU encode over the whole corpus. `tokens/s` uses the ground-truth token counts from the
# MAGIC corpus table. We warm up once (CUDA graphs / kernel autotune) before timing.

# COMMAND ----------

# Warmup.
_ = model.encode(TEXTS[:256], batch_size=64, normalize_embeddings=True, show_progress_bar=False)
torch.cuda.synchronize()
print("warmup ok")

def bench(batch_size):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    _ = model.encode(TEXTS, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=False)
    torch.cuda.synchronize()
    wall = time.perf_counter() - t0
    return {
        "batch_size": batch_size,
        "rows": len(TEXTS),
        "total_tokens": TOTAL_TOKENS,
        "wall_s": round(wall, 3),
        "tokens_per_s": round(TOTAL_TOKENS / wall, 1),
        "rows_per_s": round(len(TEXTS) / wall, 1),
    }

results = []
for bs in BATCH_SIZES:
    r = bench(bs)
    print(r)
    results.append(r)

bench_df = pd.DataFrame(results).sort_values("tokens_per_s", ascending=False)
display(bench_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Persist results to the shared table + MLflow

# COMMAND ----------

import mlflow
from datetime import datetime, timezone

rows = []
for r in results:
    rows.append({
        "run_ts": datetime.now(timezone.utc),
        "path": "air_baseline", "gpu": GPU, "detail": f"sentence-transformers/{ATTN_IMPL}",
        "batch_size": int(r["batch_size"]), "concurrency": 1,
        "rows": int(r["rows"]), "total_tokens": int(r["total_tokens"]), "wall_s": float(r["wall_s"]),
        "tokens_per_s": float(r["tokens_per_s"]), "rows_per_s": float(r["rows_per_s"]),
        "p50_ms": None, "p95_ms": None, "p99_ms": None,
    })
spark.createDataFrame(pd.DataFrame(rows)).write.mode("append").saveAsTable(RESULTS_TABLE)

best = bench_df.iloc[0]
mlflow.set_experiment(f"/Users/scott.mckean@databricks.com/embedding_serving_bench")
with mlflow.start_run(run_name=f"air_baseline_{GPU}"):
    mlflow.log_params({"path": "air_baseline", "gpu": GPU, "attn_impl": ATTN_IMPL, "max_seq_len": MAX_SEQ_LEN})
    mlflow.log_metric("peak_tokens_per_s", float(best.tokens_per_s))
    mlflow.log_metric("peak_rows_per_s", float(best.rows_per_s))
    mlflow.log_table(bench_df, "sweep.json")

headline = (f"[AIR baseline / {GPU} / {ATTN_IMPL}] peak {best.tokens_per_s:,.0f} tokens/s "
            f"({best.rows_per_s:,.0f} rows/s) at batch_size={int(best.batch_size)}")
print(headline)
dbutils.notebook.exit(json.dumps({"gpu": GPU, "attn_impl": ATTN_IMPL,
                                  "peak_tokens_per_s": float(best.tokens_per_s), "headline": headline}))
