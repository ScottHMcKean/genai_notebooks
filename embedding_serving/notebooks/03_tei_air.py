# Databricks notebook source
# MAGIC %md
# MAGIC # 03 · Text Embeddings Inference (TEI) on AI Runtime GPU + benchmark tokens/s
# MAGIC
# MAGIC TEI isn't available as a Model Serving container here (only vLLM custom serving is enabled), so we
# MAGIC run the **TEI server directly on AI Runtime GPU compute** and benchmark the same PubMedQA corpus —
# MAGIC the apples-to-apples counterpart to the vLLM endpoint and the AIR baseline (TEI-vs-Databricks throughput).
# MAGIC
# MAGIC **This is the experimental step.** TEI ships as a Docker image / Rust binary, neither of which is a
# MAGIC first-class citizen inside a Databricks notebook container. We attempt, in order:
# MAGIC 1. **Docker** (`docker run` the GPU image) if a docker daemon is reachable.
# MAGIC 2. **Prebuilt/cargo-built `text-embeddings-router`** binary run as a subprocess.
# MAGIC
# MAGIC …and record exactly how far we get. Image/feature set is GPU-specific:
# MAGIC - **A10 (Ampere SM 8.6):** `ghcr.io/huggingface/text-embeddings-inference:1.9` / cargo `-F flash-attn`.
# MAGIC - **T4 (Turing SM 7.5):** `ghcr.io/huggingface/text-embeddings-inference:turing-1.9` (Flash-Attention off) / cargo `-F turing`.
# MAGIC
# MAGIC > Attach to a **classic single-node GPU cluster** — `g5.xlarge` (A10) or `g4dn.xlarge` (T4). Set the `gpu` widget.

# COMMAND ----------

dbutils.widgets.dropdown("gpu", "A10", ["A10", "T4"], "GPU label")

# COMMAND ----------

# MAGIC %sh nvidia-smi; echo "---"; nvcc --version 2>/dev/null || echo "no nvcc"; echo "---"; which docker || echo "no docker"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration

# COMMAND ----------

CATALOG, SCHEMA = "shm_skunkworks_catalog", "genai"
SNAPSHOT_DIR = f"/Volumes/{CATALOG}/{SCHEMA}/huggingface/bioclinical_modernbert"
CORPUS_TABLE = f"{CATALOG}.{SCHEMA}.pubmedqa_corpus"
RESULTS_TABLE = f"{CATALOG}.{SCHEMA}.embedding_bench_results"

GPU = dbutils.widgets.get("gpu")
TEI_VERSION = "1.9"
DOCKER_IMAGE = f"ghcr.io/huggingface/text-embeddings-inference:{'turing-' if GPU == 'T4' else ''}{TEI_VERSION}"
CARGO_FEATURE = "turing" if GPU == "T4" else "flash-attn"   # cargo build feature flag
TEI_PORT = 8080
MAX_SEQ_LEN = 512

print(f"gpu={GPU} | docker image={DOCKER_IMAGE} | cargo feature={CARGO_FEATURE}")

# COMMAND ----------

import os, shutil, subprocess, time, requests, tempfile
# Writable scratch base. Serverless has no /local_disk0, so use the temp dir; HOME may also be
# read-only on serverless, so point it here for rustup/cargo.
SCRATCH = tempfile.mkdtemp(prefix="tei_")
os.environ["HOME"] = SCRATCH
LOG_PATH = os.path.join(SCRATCH, "tei.log")

# TEI reads a local model dir; copy the snapshot to local scratch.
LOCAL_MODEL = os.path.join(SCRATCH, "bioclinical_modernbert")
if not os.path.exists(LOCAL_MODEL):
    shutil.copytree(SNAPSHOT_DIR, LOCAL_MODEL)
print("scratch:", SCRATCH)
print("local model:", sorted(os.listdir(LOCAL_MODEL))[:6], "...")

TEI_UP = False

# COMMAND ----------

# MAGIC %md
# MAGIC ## Attempt 1 — Docker
# MAGIC Cleanest if a docker daemon is present. Mounts the local model dir, runs the GPU image.

# COMMAND ----------

has_docker = subprocess.run(["bash", "-lc", "which docker && docker info >/dev/null 2>&1 && echo OK"],
                            capture_output=True, text=True).stdout.strip().endswith("OK")
print("docker usable:", has_docker)

if has_docker:
    subprocess.run(["bash", "-lc", "docker rm -f tei 2>/dev/null || true"])
    cmd = (f"docker run -d --name tei --gpus all -p {TEI_PORT}:80 "
           f"-v {LOCAL_MODEL}:/data/model {DOCKER_IMAGE} "
           f"--model-id /data/model --max-batch-tokens 16384 --max-client-batch-size 128")
    print(cmd)
    print(subprocess.run(["bash", "-lc", cmd], capture_output=True, text=True).stdout)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Attempt 2 — build/run the `text-embeddings-router` binary
# MAGIC Only runs if docker wasn't usable. Installs Rust + builds TEI with the GPU feature, then launches
# MAGIC the router as a subprocess. This can take 10–20 min to compile and may hit CUDA-toolkit / OpenSSL
# MAGIC (FIPS) issues on AI Runtime — if so, that blocker is the finding.

# COMMAND ----------

# Preflight: a CUDA build needs nvcc. If it's absent (AIR ships the CUDA *runtime*, often not the
# full toolkit), the GPU cargo build cannot succeed — record that as the blocker instead of a 20-min
# doomed compile.
HAS_NVCC = subprocess.run(["bash", "-lc", "which nvcc"], capture_output=True, text=True).returncode == 0
print("nvcc present:", HAS_NVCC)

if not has_docker and HAS_NVCC:
    build = f"""
set -e
export HOME={SCRATCH}
export CARGO_HOME={SCRATCH}/.cargo RUSTUP_HOME={SCRATCH}/.rustup
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source $CARGO_HOME/env
export CUDA_COMPUTE_CAP={ '75' if GPU=='T4' else '86' }
cargo install --git https://github.com/huggingface/text-embeddings-inference \
  --rev v{TEI_VERSION}.0 text-embeddings-router -F {CARGO_FEATURE} --locked 2>&1 | tail -30 || echo "CARGO_BUILD_FAILED"
"""
    print("building TEI (this is slow)...")
    out = subprocess.run(["bash", "-lc", build], capture_output=True, text=True)
    print(out.stdout[-3000:])
    print("STDERR tail:", out.stderr[-1500:])

# COMMAND ----------

if not has_docker and HAS_NVCC:
    launch = f"""
set -e
export HOME={SCRATCH}
export CARGO_HOME={SCRATCH}/.cargo
source $CARGO_HOME/env
nohup text-embeddings-router --model-id {LOCAL_MODEL} --port {TEI_PORT} \
  --max-batch-tokens 16384 --max-client-batch-size 128 > {LOG_PATH} 2>&1 &
echo launched
"""
    print(subprocess.run(["bash", "-lc", launch], capture_output=True, text=True).stdout)
elif not has_docker:
    print("Neither docker nor nvcc available on this serverless GPU compute — TEI server cannot be "
          "built or run here. This is the finding: TEI needs a classic GPU cluster (docker/prebuilt "
          "binary) or Model Serving BYOC, neither enabled on this serverless-only workspace.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Wait for readiness

# COMMAND ----------

# Only poll if we actually launched something (docker container or built binary).
LAUNCHED = has_docker or HAS_NVCC
if LAUNCHED:
    deadline = time.time() + 240
    while time.time() < deadline:
        try:
            if requests.get(f"http://localhost:{TEI_PORT}/health", timeout=2).ok:
                TEI_UP = True
                break
        except Exception:
            pass
        time.sleep(4)
else:
    print("Nothing launched (no docker, no nvcc) — skipping readiness poll.")
print("TEI up:", TEI_UP)

if not TEI_UP:
    print("---- diagnostics ----")
    if has_docker:
        print(subprocess.run(["bash", "-lc", "docker logs tei 2>&1 | tail -40"], capture_output=True, text=True).stdout)
    else:
        print(subprocess.run(["bash", "-lc", f"tail -40 {LOG_PATH} 2>/dev/null"], capture_output=True, text=True).stdout)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Smoke test the `/embed` endpoint

# COMMAND ----------

if TEI_UP:
    import numpy as np
    r = requests.post(f"http://localhost:{TEI_PORT}/embed",
                      json={"inputs": ["acute myocardial infarction", "shortness of breath"]}, timeout=30)
    emb = r.json()
    print("n:", len(emb), "| dims:", len(emb[0]), "| cosine:",
          float(np.dot(emb[0], emb[1]) / (np.linalg.norm(emb[0]) * np.linalg.norm(emb[1]))))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Benchmark tokens/s (only if TEI is up)
# MAGIC Same protocol as the vLLM notebook: batch `batch_size` texts per request over `concurrency`
# MAGIC parallel clients; tokens/s from the corpus's ground-truth token counts. Localhost, so no network hop.

# COMMAND ----------

if TEI_UP:
    import pandas as pd, numpy as np
    from concurrent.futures import ThreadPoolExecutor

    corpus_pdf = spark.table(CORPUS_TABLE).select("text", "n_tokens").toPandas()
    TEXTS = corpus_pdf["text"].tolist()[:4000]
    TOK_BY_TEXT = dict(zip(corpus_pdf["text"], corpus_pdf["n_tokens"]))

    def embed_batch(texts):
        t0 = time.perf_counter()
        requests.post(f"http://localhost:{TEI_PORT}/embed", json={"inputs": texts,
                      "truncate": True}, timeout=60)
        return time.perf_counter() - t0

    _ = embed_batch(TEXTS[:8])  # warmup

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
else:
    results = []
    print("TEI not up — skipping benchmark. See diagnostics above; this is the finding for this GPU.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Persist results

# COMMAND ----------

import json
from datetime import datetime, timezone

if TEI_UP and results:
    import pandas as pd
    rows = [{
        "run_ts": datetime.now(timezone.utc), "path": "tei_air", "gpu": GPU,
        "detail": f"{DOCKER_IMAGE if has_docker else 'cargo-'+CARGO_FEATURE}", "batch_size": int(r["batch_size"]),
        "concurrency": int(r["concurrency"]), "rows": int(r["rows"]), "total_tokens": int(r["total_tokens"]),
        "wall_s": float(r["wall_s"]), "tokens_per_s": float(r["tokens_per_s"]), "rows_per_s": float(r["rows_per_s"]),
        "p50_ms": float(r["p50_ms"]), "p95_ms": float(r["p95_ms"]), "p99_ms": float(r["p99_ms"]),
    } for r in results]
    spark.createDataFrame(pd.DataFrame(rows)).write.mode("append").saveAsTable(RESULTS_TABLE)
    best = bench_df.iloc[0]
    headline = (f"[TEI / {GPU}] peak {best.tokens_per_s:,.0f} tokens/s ({best.rows_per_s:,.0f} rows/s) "
                f"at batch_size={int(best.batch_size)}, concurrency={int(best.concurrency)}")
else:
    headline = f"[TEI / {GPU}] did not start — see notebook diagnostics (docker={has_docker})."

print(headline)
dbutils.notebook.exit(json.dumps({"gpu": GPU, "tei_up": TEI_UP, "headline": headline}))
