# Databricks notebook source
# MAGIC %md
# MAGIC # 00 · Setup + data
# MAGIC
# MAGIC One-time setup for the bioclinical ModernBERT embedding profiling suite:
# MAGIC 1. Create the `genai` schema + `huggingface` volume (they don't exist yet on shm-skunkworks).
# MAGIC 2. Snapshot `NeuML/bioclinical-modernbert-base-embeddings` to the volume (no HF pull at serve time).
# MAGIC 3. Download `qiaojin/PubMedQA`, build a clean bioclinical benchmark corpus, land it as a Delta table.
# MAGIC 4. Compute token statistics with the model's own tokenizer (so tokens/s is meaningful downstream).
# MAGIC 5. Create the shared results table the three profiling notebooks append to.
# MAGIC
# MAGIC > Runs on **serverless CPU** — no GPU needed. Later notebooks read from the volume + corpus table.

# COMMAND ----------

# MAGIC %pip install --quiet --upgrade "huggingface_hub>=0.25" "datasets>=2.20" "transformers>=4.48" hf_transfer
# MAGIC %restart_python

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration

# COMMAND ----------

# Redirect all HF caches to a writable dir. On serverless the default (/root/.cache) is not writable
# and Databricks' datasets patch fails with PermissionError. Must be set before importing datasets.
import os, tempfile
HF_CACHE = os.path.join(tempfile.gettempdir(), "hf_cache")
os.makedirs(HF_CACHE, exist_ok=True)
os.environ["HF_HOME"] = HF_CACHE
os.environ["HF_DATASETS_CACHE"] = os.path.join(HF_CACHE, "datasets")
os.environ["HF_HUB_CACHE"] = os.path.join(HF_CACHE, "hub")
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

CATALOG = "shm_skunkworks_catalog"
SCHEMA = "genai"
VOLUME = "huggingface"

MODEL_REPO_ID = "NeuML/bioclinical-modernbert-base-embeddings"
SNAPSHOT_DIR = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME}/bioclinical_modernbert"

CORPUS_TABLE = f"{CATALOG}.{SCHEMA}.pubmedqa_corpus"
RESULTS_TABLE = f"{CATALOG}.{SCHEMA}.embedding_bench_results"

# Cap corpus size for a snappy-but-representative benchmark. PubMedQA has ~272k unlabeled +
# 1k labeled contexts; 20k rows is plenty to saturate a GPU and get stable tokens/s.
CORPUS_ROWS = 20_000

print(f"model:   {MODEL_REPO_ID}")
print(f"snapshot:{SNAPSHOT_DIR}")
print(f"corpus:  {CORPUS_TABLE}  (target {CORPUS_ROWS:,} rows)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Create schema + volume

# COMMAND ----------

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")
spark.sql(f"CREATE VOLUME IF NOT EXISTS {CATALOG}.{SCHEMA}.{VOLUME}")
print("schema + volume ready")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Snapshot the model to the volume
# MAGIC `snapshot_download` preserves the canonical HF layout so sentence-transformers, transformers,
# MAGIC vLLM, and TEI can all reload it from the same directory.

# COMMAND ----------

import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

from huggingface_hub import snapshot_download

os.makedirs(SNAPSHOT_DIR, exist_ok=True)
snapshot_download(repo_id=MODEL_REPO_ID, local_dir=SNAPSHOT_DIR)
print("snapshot files:", sorted(os.listdir(SNAPSHOT_DIR)))

# COMMAND ----------

# Report the embedding dimension + max sequence length straight from the config so downstream
# notebooks don't hard-code them.
import json

with open(os.path.join(SNAPSHOT_DIR, "config.json")) as f:
    cfg = json.load(f)
EMBED_DIMS = cfg.get("hidden_size")
MAX_POS = cfg.get("max_position_embeddings")
print(f"architecture: {cfg.get('architectures')}")
print(f"hidden_size (embed dims): {EMBED_DIMS}")
print(f"max_position_embeddings:  {MAX_POS}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Download PubMedQA and build the benchmark corpus
# MAGIC `qiaojin/PubMedQA` (MIT). We use the `pqa_artificial` config (~211k rows) — each row has a
# MAGIC question plus a `context.contexts` list of abstract sentences. We concatenate question +
# MAGIC context into one realistic bioclinical passage per row.

# COMMAND ----------

from datasets import load_dataset

# pqa_artificial is the large unlabeled split — ideal for a throughput corpus.
ds = load_dataset("qiaojin/PubMedQA", "pqa_artificial", split="train", cache_dir=os.environ["HF_DATASETS_CACHE"])
print("raw rows:", len(ds))
print("columns:", ds.column_names)

# COMMAND ----------

def build_passage(row):
    q = (row.get("question") or "").strip()
    ctx = row.get("context") or {}
    sentences = ctx.get("contexts") if isinstance(ctx, dict) else None
    body = " ".join(s.strip() for s in sentences) if sentences else ""
    text = (q + " " + body).strip()
    return {"text": text}

corpus = ds.select(range(min(CORPUS_ROWS, len(ds)))).map(
    build_passage, remove_columns=ds.column_names
)
# Drop empties / dedup.
texts = [r["text"] for r in corpus if r["text"]]
texts = list(dict.fromkeys(texts))
print(f"clean corpus rows: {len(texts):,}")
print("example:", texts[0][:300], "...")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Token statistics with the model's tokenizer

# COMMAND ----------

import numpy as np
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(SNAPSHOT_DIR)
tok_counts = [len(tokenizer.encode(t, add_special_tokens=True, truncation=True, max_length=MAX_POS)) for t in texts]
tok_counts = np.array(tok_counts)

stats = {
    "rows": int(len(texts)),
    "total_tokens": int(tok_counts.sum()),
    "mean_tokens": float(tok_counts.mean()),
    "p50_tokens": float(np.percentile(tok_counts, 50)),
    "p95_tokens": float(np.percentile(tok_counts, 95)),
    "max_tokens": int(tok_counts.max()),
}
for k, v in stats.items():
    print(f"{k:14s}: {v:,.1f}" if isinstance(v, float) else f"{k:14s}: {v:,}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Land the corpus + token counts as a Delta table
# MAGIC Storing per-row token counts means every downstream benchmark computes tokens/s from the same
# MAGIC ground truth (no re-tokenizing, no drift).

# COMMAND ----------

import pandas as pd

pdf = pd.DataFrame({"id": range(len(texts)), "text": texts, "n_tokens": tok_counts.astype(int)})
(
    spark.createDataFrame(pdf)
    .write.mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(CORPUS_TABLE)
)
print(f"wrote {CORPUS_TABLE}: {spark.table(CORPUS_TABLE).count():,} rows")
display(spark.table(CORPUS_TABLE).limit(5))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Create the shared results table
# MAGIC Every profiling notebook appends one row per (path, gpu, batch_size, concurrency) run here.

# COMMAND ----------

spark.sql(f"""
CREATE TABLE IF NOT EXISTS {RESULTS_TABLE} (
  run_ts        TIMESTAMP,
  path          STRING,   -- 'air_baseline' | 'vllm_serving' | 'tei_air'
  gpu           STRING,   -- 'A10' | 'T4'
  detail        STRING,   -- attn impl / endpoint / image, free-form
  batch_size    INT,
  concurrency   INT,
  rows          BIGINT,
  total_tokens  BIGINT,
  wall_s        DOUBLE,
  tokens_per_s  DOUBLE,
  rows_per_s    DOUBLE,
  p50_ms        DOUBLE,
  p95_ms        DOUBLE,
  p99_ms        DOUBLE
) USING DELTA
""")
print(f"results table ready: {RESULTS_TABLE}")

# COMMAND ----------

import json
dbutils.notebook.exit(json.dumps({
    "snapshot_dir": SNAPSHOT_DIR,
    "embed_dims": EMBED_DIMS,
    "max_pos": MAX_POS,
    "corpus_table": CORPUS_TABLE,
    "results_table": RESULTS_TABLE,
    "token_stats": stats,
}))
