# Databricks notebook source
# MAGIC %md
# MAGIC # 02b · Serve via sentence-transformers pyfunc on Model Serving + benchmark tokens/s
# MAGIC
# MAGIC The **working** custom-model-serving path on this workspace. vLLM is blocked here (see `02`: the
# MAGIC entrypoint route is disabled workspace-wide, and vLLM's model-inspection subprocess dies with a
# MAGIC `FATAL FIPS SELFTEST FAILURE` on the AI Runtime image). So we serve `NeuML/bioclinical-modernbert-
# MAGIC base-embeddings` as a **sentence-transformers `pyfunc`** on standard GPU Model Serving and benchmark
# MAGIC endpoint **tokens/s** on **A10** (`GPU_MEDIUM`) and **T4** (`GPU_SMALL`).
# MAGIC
# MAGIC This is the production-appropriate answer and the fair apples-to-apples comparison vs TEI.
# MAGIC
# MAGIC > Build/register on **serverless GPU (A10)** so the packed env carries CUDA torch. Set the `gpu`
# MAGIC > widget to choose the endpoint workload type.

# COMMAND ----------

# MAGIC %pip install --upgrade "mlflow>=3.1" "sentence-transformers>=3.0" "transformers>=4.48" "databricks-sdk>=0.102.0"
# MAGIC %restart_python

# COMMAND ----------

dbutils.widgets.dropdown("gpu", "A10", ["A10", "T4"], "endpoint GPU")

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

EMBED_DIMS = 768
MAX_SEQ_LEN = 512
UC_MODEL_NAME = f"{CATALOG}.{SCHEMA}.bioclinical_modernbert_st"
ENDPOINT_NAME = f"shm-bioclin-mbert-st-{GPU.lower()}"

print(f"gpu={GPU} | workload={WORKLOAD_TYPE} | endpoint={ENDPOINT_NAME}")

# COMMAND ----------

import time, json
import numpy as np, pandas as pd
import mlflow
from mlflow.models import infer_signature

mlflow.set_tracking_uri("databricks")
mlflow.set_registry_uri("databricks-uc")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Wrap as a sentence-transformers pyfunc
# MAGIC Detects the endpoint GPU at load time and picks a safe attention impl (SDPA — works on both A10
# MAGIC and T4; flash-attn isn't in the serving image) and dtype (bf16 on Ampere, fp16 on Turing).

# COMMAND ----------

class EmbeddingModel(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        import torch
        from sentence_transformers import SentenceTransformer
        if torch.cuda.is_available():
            cap = torch.cuda.get_device_capability(0)[0]
            dtype = torch.bfloat16 if cap >= 8 else torch.float16
            device = "cuda"
        else:
            dtype, device = torch.float32, "cpu"
        self.model = SentenceTransformer(
            context.artifacts["repository"], device=device,
            model_kwargs={"attn_implementation": "sdpa", "torch_dtype": dtype},
        )
        self.model.max_seq_length = 512

    def predict(self, context, model_input, params=None):
        if isinstance(model_input, pd.DataFrame):
            texts = model_input["text"].astype(str).tolist()
        else:
            texts = [str(t) for t in model_input]
        bs = (params or {}).get("batch_size", 64)
        emb = self.model.encode(texts, batch_size=bs, normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(emb).tolist()

# COMMAND ----------

input_example = pd.DataFrame({"text": ["chest pain radiating to the left arm, elevated troponin"]})
signature = infer_signature(input_example, [[0.0] * EMBED_DIMS], params={"batch_size": 64})

with mlflow.start_run(run_name=f"st_embed_{GPU}"):
    model_info = mlflow.pyfunc.log_model(
        artifact_path="model",
        python_model=EmbeddingModel(),
        artifacts={"repository": SNAPSHOT_DIR},
        pip_requirements=["sentence-transformers", "torch", "transformers", "accelerate"],
        input_example=input_example, signature=signature,
    )
print("logged:", model_info.model_uri)

# COMMAND ----------

# Local smoke test (loads on the driver GPU/CPU).
loaded = mlflow.pyfunc.load_model(model_info.model_uri)
vecs = loaded.predict(pd.DataFrame({"text": ["shortness of breath", "acute myocardial infarction"]}))
print("dims:", len(vecs[0]), "| cosine:", float(np.dot(vecs[0], vecs[1])))

# COMMAND ----------

model_version = mlflow.register_model(model_info.model_uri, UC_MODEL_NAME, env_pack="databricks_model_serving")
print("registered:", UC_MODEL_NAME, "v", model_version.version)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Deploy the endpoint

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
    w.serving_endpoints.update_config_and_wait(
        name=ENDPOINT_NAME, served_entities=config.served_entities, timeout=timedelta(minutes=45))
else:
    w.serving_endpoints.create_and_wait(name=ENDPOINT_NAME, config=config, timeout=timedelta(minutes=45))
print("endpoint ready:", ENDPOINT_NAME)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Benchmark endpoint tokens/s

# COMMAND ----------

from mlflow.deployments import get_deploy_client
from concurrent.futures import ThreadPoolExecutor

client = get_deploy_client("databricks")

corpus_pdf = spark.table(CORPUS_TABLE).select("text", "n_tokens").toPandas()
TEXTS = corpus_pdf["text"].tolist()[:4000]
TOK_BY_TEXT = dict(zip(corpus_pdf["text"], corpus_pdf["n_tokens"]))
print("bench rows:", len(TEXTS))

def embed_batch(texts):
    t0 = time.perf_counter()
    client.predict(endpoint=ENDPOINT_NAME,
                   inputs={"dataframe_records": [{"text": t} for t in texts], "params": {"batch_size": 64}})
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
        r = run_benchmark(TEXTS, bs, conc)
        print(r)
        results.append(r)

bench_df = pd.DataFrame(results).sort_values("tokens_per_s", ascending=False)
display(bench_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Persist results

# COMMAND ----------

from datetime import datetime, timezone
rows = [{
    "run_ts": datetime.now(timezone.utc), "path": "st_pyfunc_serving", "gpu": GPU,
    "detail": "sentence-transformers/sdpa", "batch_size": int(r["batch_size"]),
    "concurrency": int(r["concurrency"]), "rows": int(r["rows"]), "total_tokens": int(r["total_tokens"]),
    "wall_s": float(r["wall_s"]), "tokens_per_s": float(r["tokens_per_s"]), "rows_per_s": float(r["rows_per_s"]),
    "p50_ms": float(r["p50_ms"]), "p95_ms": float(r["p95_ms"]), "p99_ms": float(r["p99_ms"]),
} for r in results]
spark.createDataFrame(pd.DataFrame(rows)).write.mode("append").saveAsTable(RESULTS_TABLE)

best = bench_df.iloc[0]
headline = (f"[ST pyfunc serving / {GPU}] peak {best.tokens_per_s:,.0f} tokens/s ({best.rows_per_s:,.0f} rows/s) "
            f"at batch_size={int(best.batch_size)}, concurrency={int(best.concurrency)}; "
            f"p50 {best.p50_ms}ms p95 {best.p95_ms}ms")
print(headline)
dbutils.notebook.exit(json.dumps({"gpu": GPU, "peak_tokens_per_s": float(best.tokens_per_s), "headline": headline}))
