# Databricks notebook source
# MAGIC %md
# MAGIC # 02 · Serve vLLM as a pyfunc (in-process engine) + benchmark tokens/s
# MAGIC
# MAGIC The **only** remaining vLLM route for embeddings on this workspace, after ruling out:
# MAGIC - **Entrypoint route** (`02c`): Custom LLM Serving only accepts task `llm/v1/chat` — it rejects
# MAGIC   `llm/v1/embeddings` (*"must have a task type that is one of the supported types: llm/v1/chat"*).
# MAGIC
# MAGIC So we wrap an **in-process `vllm.LLM(runner="pooling").embed()`** in a pyfunc on standard GPU serving.
# MAGIC
# MAGIC ### Lessons baked in
# MAGIC - **Don't gate on a notebook smoke test.** The AIR serverless *notebook* is FIPS-hostile to vLLM's
# MAGIC   model-inspection subprocess (`FATAL FIPS SELFTEST FAILURE`; unsetting FORCE_FIPS flips it to an
# MAGIC   SSLContext error). The **serving container is a different image** — it's the real test.
# MAGIC - **`env_pack` packaging**: use `extra_pip_requirements` (append vLLM, let it pull its own torch) —
# MAGIC   NOT `pip_requirements=[...,"torch"]`, which forced a conflicting fresh install and broke env_pack.
# MAGIC - **FIPS in the serving container**: `load_context` pops `OPENSSL_FORCE_FIPS_MODE` before importing
# MAGIC   vLLM (the `document_intelligence/alternates/docling_parser.py` precedent).
# MAGIC
# MAGIC > Build on **serverless GPU (A10)**. `gpu` widget picks the endpoint workload type.

# COMMAND ----------

# MAGIC %pip install vllm==0.11.2 transformers==4.57.6 mlflow==3.12.0 hf_transfer==0.1.9 "databricks-sdk>=0.102.0"
# MAGIC %restart_python

# COMMAND ----------

dbutils.widgets.dropdown("gpu", "A10", ["A10", "T4"], "endpoint GPU")

# COMMAND ----------

import os, tempfile
os.chdir(tempfile.mkdtemp())

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration

# COMMAND ----------

from databricks.sdk.service.serving import ServingModelWorkloadType

CATALOG, SCHEMA = "shm_skunkworks_catalog", "genai"
SNAPSHOT_DIR = f"/Volumes/{CATALOG}/{SCHEMA}/huggingface/bioclinical_modernbert"
CORPUS_TABLE = f"{CATALOG}.{SCHEMA}.pubmedqa_corpus"
RESULTS_TABLE = f"{CATALOG}.{SCHEMA}.embedding_bench_results"

GPU = dbutils.widgets.get("gpu")
WORKLOAD_TYPE = ServingModelWorkloadType.GPU_MEDIUM if GPU == "A10" else ServingModelWorkloadType.GPU_SMALL

ARTIFACTS_PATH = "bioclinical_modernbert"
UC_MODEL_NAME = f"{CATALOG}.{SCHEMA}.bioclinical_modernbert_vllm_pyfunc"
ENDPOINT_NAME = f"shm-bioclin-mbert-vllmpf-{GPU.lower()}"
DTYPE = "bfloat16" if GPU == "A10" else "float16"

print(f"gpu={GPU} | workload={WORKLOAD_TYPE} | endpoint={ENDPOINT_NAME}")

# COMMAND ----------

import shutil
if not os.path.exists(ARTIFACTS_PATH):
    shutil.copytree(SNAPSHOT_DIR, ARTIFACTS_PATH)
print("local weights:", sorted(os.listdir(ARTIFACTS_PATH))[:6], "...")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Log the pyfunc-wrapped vLLM engine
# MAGIC No notebook smoke test (it can't build vLLM under AIR FIPS). `load_context` runs only in the
# MAGIC serving container.

# COMMAND ----------

import mlflow, pandas as pd
from mlflow.models import infer_signature

class VLLMEmbedModel(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        import os, torch
        os.environ.pop("OPENSSL_FORCE_FIPS_MODE", None)  # FIPS fix in the serving container
        # Pick dtype by GPU: bf16 on Ampere+ (A10), fp16 on Turing (T4 has no bf16).
        cap = torch.cuda.get_device_capability(0)[0] if torch.cuda.is_available() else 0
        dtype = "bfloat16" if cap >= 8 else "float16"
        from vllm import LLM
        self.llm = LLM(model=context.artifacts["model_dir"], runner="pooling", dtype=dtype,
                       max_model_len=512, gpu_memory_utilization=0.60, enforce_eager=True)

    def predict(self, context, model_input, params=None):
        if hasattr(model_input, "columns"):
            texts = model_input["text"].astype(str).tolist()
        else:
            texts = [str(t) for t in model_input]
        outs = self.llm.embed(texts)
        return [list(o.outputs.embedding) for o in outs]

input_example = pd.DataFrame({"text": ["chest pain radiating to the left arm"]})
signature = infer_signature(input_example, [[0.0] * 768])

with mlflow.start_run(run_name=f"vllm_pyfunc_embed_{GPU}"):
    model_info = mlflow.pyfunc.log_model(
        artifact_path="model",
        python_model=VLLMEmbedModel(),
        artifacts={"model_dir": ARTIFACTS_PATH},
        # Append vLLM (imported lazily in load_context, so infer won't catch it); let vLLM pull its
        # own torch. Do NOT use pip_requirements=[...,"torch"] — that broke env_pack.
        extra_pip_requirements=["vllm==0.11.2", "transformers==4.57.6", "mlflow==3.12.0"],
        input_example=input_example, signature=signature,
    )
print("logged:", model_info.model_uri)

# COMMAND ----------

# Register WITHOUT env_pack. env_pack runs `pip install -r requirements.txt` IN THIS NOTEBOOK, whose
# AIR FIPS/OpenSSL environment crashes pip at startup (ssl.SSLError [CRYPTO] unknown error). Plain
# registration defers the container env build to Model Serving's SERVER-SIDE builder (the same pipeline
# that ships FMAPI's vLLM containers) — a different, non-FIPS-hostile environment. Tradeoff: no express
# deploy (slower cold start), which is irrelevant for a throughput benchmark.
mlflow.set_registry_uri("databricks-uc")
model_version = mlflow.register_model(model_info.model_uri, UC_MODEL_NAME)
print("registered (no env_pack):", UC_MODEL_NAME, "v", model_version.version)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Deploy (standard GPU serving) — the serving container is the real FIPS test

# COMMAND ----------

from databricks.sdk import WorkspaceClient
from datetime import timedelta
from databricks.sdk.service.serving import EndpointCoreConfigInput, ServedEntityInput

config = EndpointCoreConfigInput(
    name=ENDPOINT_NAME,
    served_entities=[ServedEntityInput(
        entity_name=UC_MODEL_NAME, entity_version=str(model_version.version),
        workload_type=WORKLOAD_TYPE, workload_size="Small", scale_to_zero_enabled=False,
    )],
)
w = WorkspaceClient()
existing = [e.name for e in w.serving_endpoints.list()]
if ENDPOINT_NAME in existing:
    w.serving_endpoints.update_config_and_wait(name=ENDPOINT_NAME, served_entities=config.served_entities,
                                               timeout=timedelta(minutes=50))
else:
    w.serving_endpoints.create_and_wait(name=ENDPOINT_NAME, config=config, timeout=timedelta(minutes=50))
print("endpoint ready:", ENDPOINT_NAME)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Benchmark tokens/s

# COMMAND ----------

from mlflow.deployments import get_deploy_client
from concurrent.futures import ThreadPoolExecutor
import numpy as np, time

client = get_deploy_client("databricks")
corpus_pdf = spark.table(CORPUS_TABLE).select("text", "n_tokens").toPandas()
TEXTS = corpus_pdf["text"].tolist()[:4000]
TOK_BY_TEXT = dict(zip(corpus_pdf["text"], corpus_pdf["n_tokens"]))

# vLLM REJECTS inputs longer than max_model_len (512) — truncate client-side to 512 tokens (matches
# how the sentence-transformers endpoint was measured). tokens/s counts tokens actually processed.
from transformers import AutoTokenizer
_tok = AutoTokenizer.from_pretrained(ARTIFACTS_PATH)
_orig = dict(TOK_BY_TEXT)
def _trunc(t):
    ids = _tok.encode(t, add_special_tokens=True, truncation=True, max_length=512)
    return _tok.decode(ids, skip_special_tokens=True)
_pairs = [(o, _trunc(o)) for o in TEXTS]
TEXTS = [tt for _, tt in _pairs]
TOK_BY_TEXT = {tt: min(int(_orig[o]), 512) for o, tt in _pairs}

def embed_batch(texts):
    t0 = time.perf_counter()
    client.predict(endpoint=ENDPOINT_NAME, inputs={"dataframe_records": [{"text": t} for t in texts]})
    return time.perf_counter() - t0

print("warmup:", round(embed_batch(TEXTS[:8]), 3), "s")

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
        r = run_benchmark(TEXTS, bs, conc); print(r); results.append(r)
bench_df = pd.DataFrame(results).sort_values("tokens_per_s", ascending=False)
display(bench_df)

# COMMAND ----------

from datetime import datetime, timezone
import json
rows = [{
    "run_ts": datetime.now(timezone.utc), "path": "vllm_serving", "gpu": GPU,
    "detail": f"pyfunc/vllm.embed/{DTYPE}", "batch_size": int(r["batch_size"]),
    "concurrency": int(r["concurrency"]), "rows": int(r["rows"]), "total_tokens": int(r["total_tokens"]),
    "wall_s": float(r["wall_s"]), "tokens_per_s": float(r["tokens_per_s"]), "rows_per_s": float(r["rows_per_s"]),
    "p50_ms": float(r["p50_ms"]), "p95_ms": float(r["p95_ms"]), "p99_ms": float(r["p99_ms"]),
} for r in results]
spark.createDataFrame(pd.DataFrame(rows)).write.mode("append").saveAsTable(RESULTS_TABLE)

best = bench_df.iloc[0]
headline = (f"[vLLM pyfunc / {GPU}] peak {best.tokens_per_s:,.0f} tokens/s ({best.rows_per_s:,.0f} rows/s) "
            f"at batch_size={int(best.batch_size)}, concurrency={int(best.concurrency)}; p50 {best.p50_ms}ms")
print(headline)
dbutils.notebook.exit(json.dumps({"gpu": GPU, "peak_tokens_per_s": float(best.tokens_per_s), "headline": headline}))
