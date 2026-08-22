# Databricks notebook source
# MAGIC %md
# MAGIC # 02c · Serve vLLM via the Custom LLM Serving *entrypoint* route + benchmark tokens/s
# MAGIC
# MAGIC The supported way to run vLLM on Databricks Model Serving — mirrors the proven
# MAGIC `custom_models/hf_chat_serving_air.py` chat template, swapped to **embeddings**
# MAGIC (`--runner pooling`, task `llm/v1/embeddings`). This route:
# MAGIC - runs vLLM's **own OpenAI server as the serving container's main process** (like FMAPI), so the
# MAGIC   serving infra handles vLLM startup — no pyfunc spawning an in-process engine, no FIPS crash;
# MAGIC - uses `env_pack` to **snapshot the working notebook env** (vLLM already installed) instead of a
# MAGIC   fresh `pip install vllm+torch` — which is what broke the pyfunc route's `env_pack`.
# MAGIC
# MAGIC ### PREREQUISITE (workspace gate)
# MAGIC Endpoint create fails with *"Served entity ... with entrypoint is not supported for your workspace"*
# MAGIC unless **Custom LLM Serving** is enabled: **Admin Settings → Previews → "Custom LLM Serving" → On**.
# MAGIC
# MAGIC > Run on **serverless GPU (A10)**. Set the `gpu` widget for the endpoint workload type.

# COMMAND ----------

# MAGIC %pip install vllm==0.11.2 transformers==4.57.6 openai==2.17.0 opencv-python-headless==4.12.* mlflow==3.12.0 hf_transfer==0.1.9 "databricks-sdk>=0.102.0"
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
SERVED_MODEL_NAME = "bioclinical-modernbert"
UC_MODEL_NAME = f"{CATALOG}.{SCHEMA}.bioclinical_modernbert_vllm_entry"
ENDPOINT_NAME = f"shm-bioclin-mbert-vllm-{GPU.lower()}"

DTYPE = "bfloat16" if GPU == "A10" else "float16"
MAX_MODEL_LEN = 512
GPU_MEMORY_UTILIZATION = 0.85
LOCAL_PORT = 3080
SERVING_PORT = 8080

print(f"gpu={GPU} | workload={WORKLOAD_TYPE} | endpoint={ENDPOINT_NAME}")

# COMMAND ----------

import shutil
if not os.path.exists(ARTIFACTS_PATH):
    shutil.copytree(SNAPSHOT_DIR, ARTIFACTS_PATH)
print("local weights:", sorted(os.listdir(ARTIFACTS_PATH))[:6], "...")

# COMMAND ----------

# MAGIC %md
# MAGIC ## vLLM embeddings entrypoint

# COMMAND ----------

def entrypoint(port: int) -> str:
    args = [
        "python", "-u", "-m", "vllm.entrypoints.openai.api_server",
        "--model", ARTIFACTS_PATH,
        "--served-model-name", SERVED_MODEL_NAME,
        "--runner", "pooling",           # embedding/pooling mode
        "--host", "0.0.0.0",
        "--port", str(port),
        "--dtype", DTYPE,
        "--max-model-len", str(MAX_MODEL_LEN),
        "--gpu-memory-utilization", str(GPU_MEMORY_UTILIZATION),
    ]
    return " ".join(args)

print(entrypoint(SERVING_PORT))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Local smoke test — vLLM's OpenAI server (standalone subprocess) serving embeddings
# MAGIC The standalone api_server subprocess works on serverless GPU (only the *in-process* `LLM()` hits
# MAGIC the FIPS inspection crash). Confirms the model + `--runner pooling` before deploying.

# COMMAND ----------

import subprocess, requests, time

log = open("process.log", "w")
subprocess.Popen(["bash", "-lc", entrypoint(LOCAL_PORT)], stdout=log, stderr=subprocess.STDOUT,
                 text=True, start_new_session=True)

ready = False
deadline = time.time() + 420
while time.time() < deadline:
    try:
        if requests.get(f"http://localhost:{LOCAL_PORT}/health", timeout=2).ok:
            ready = True; break
    except Exception:
        pass
    time.sleep(4)
print("vLLM server ready:", ready)
if not ready:
    print("".join(open("process.log").readlines()[-40:]))

# COMMAND ----------

VLLM_OK = False
if ready:
    r = requests.post(f"http://localhost:{LOCAL_PORT}/v1/embeddings",
                      json={"model": SERVED_MODEL_NAME, "input": ["acute myocardial infarction", "shortness of breath"]},
                      timeout=30)
    data = r.json()
    dims = len(data["data"][0]["embedding"])
    print("embeddings ok | dims:", dims, "| n:", len(data["data"]))
    VLLM_OK = True

# COMMAND ----------

# MAGIC %sh pkill -f vllm.entrypoints.openai.api_server || true

# COMMAND ----------

# MAGIC %md
# MAGIC ## Log + register (entrypoint, task=llm/v1/embeddings)
# MAGIC `env_pack` snapshots THIS notebook's env (vLLM already installed) — do not pin vllm in
# MAGIC pip_requirements (that forces a fresh install and breaks packaging).

# COMMAND ----------

import mlflow
from mlflow.models import infer_signature

class EmbedPlaceholder(mlflow.pyfunc.PythonModel):
    def predict(self, context, model_input, params=None):
        return [[0.0]]  # serving runs the entrypoint, not predict

signature = infer_signature({"input": ["chest pain radiating to the left arm"]}, [[0.0] * 768])

with mlflow.start_run(run_name=f"vllm_entry_embed_{GPU}"):
    model_info = mlflow.pyfunc.log_model(
        name=SERVED_MODEL_NAME,
        python_model=EmbedPlaceholder(),
        artifacts={"model_dir": ARTIFACTS_PATH},
        metadata={"task": "llm/v1/embeddings", "entrypoint": entrypoint(SERVING_PORT)},
        signature=signature,
        extra_pip_requirements=["mlflow==3.12.0"],
    )
print("logged:", model_info.model_uri)

# COMMAND ----------

mlflow.set_registry_uri("databricks-uc")
model_version = mlflow.register_model(model_info.model_uri, UC_MODEL_NAME, env_pack="databricks_model_serving")
print("registered:", UC_MODEL_NAME, "v", model_version.version)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Deploy (fixed concurrency — entrypoint endpoints don't scale to zero)

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
# MAGIC ## Benchmark tokens/s (OpenAI embeddings API)

# COMMAND ----------

from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor
import numpy as np, pandas as pd

HOST = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().get()
TOKEN = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
client = OpenAI(api_key=TOKEN, base_url=f"{HOST}/serving-endpoints")

corpus_pdf = spark.table(CORPUS_TABLE).select("text", "n_tokens").toPandas()
TEXTS = corpus_pdf["text"].tolist()[:4000]
TOK_BY_TEXT = dict(zip(corpus_pdf["text"], corpus_pdf["n_tokens"]))

def embed_batch(texts):
    t0 = time.perf_counter()
    client.embeddings.create(model=ENDPOINT_NAME, input=texts)
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
    "detail": f"entrypoint/llm-v1-embeddings/{DTYPE}", "batch_size": int(r["batch_size"]),
    "concurrency": int(r["concurrency"]), "rows": int(r["rows"]), "total_tokens": int(r["total_tokens"]),
    "wall_s": float(r["wall_s"]), "tokens_per_s": float(r["tokens_per_s"]), "rows_per_s": float(r["rows_per_s"]),
    "p50_ms": float(r["p50_ms"]), "p95_ms": float(r["p95_ms"]), "p99_ms": float(r["p99_ms"]),
} for r in results]
spark.createDataFrame(pd.DataFrame(rows)).write.mode("append").saveAsTable(RESULTS_TABLE)

best = bench_df.iloc[0]
headline = (f"[vLLM entrypoint / {GPU}] peak {best.tokens_per_s:,.0f} tokens/s ({best.rows_per_s:,.0f} rows/s) "
            f"at batch_size={int(best.batch_size)}, concurrency={int(best.concurrency)}; p50 {best.p50_ms}ms")
print(headline)
dbutils.notebook.exit(json.dumps({"gpu": GPU, "peak_tokens_per_s": float(best.tokens_per_s), "headline": headline}))
