# Serving a fine-tuned bioclinical ModernBERT embedding model on Databricks — findings & throughput

**Model:** [`NeuML/bioclinical-modernbert-base-embeddings`](https://huggingface.co/NeuML/bioclinical-modernbert-base-embeddings)
(ModernBERT-base, 768-dim, `max_position_embeddings` 8192)
**Goal:** deploy the model on custom Model Serving and profile embedding **throughput (tokens/s)** across
serving paths (AI Runtime baseline, vLLM, Text Embeddings Inference) on **A10** vs **T4**, to pick the
best way to serve a fine-tuned ModernBERT embedding model on Databricks.

The workspace used is **serverless-only** (no classic clusters), which shaped several choices below.

---

## TL;DR

- **Use vLLM on A10.** A pyfunc-wrapped in-process vLLM engine served the model at **~89.5k tokens/s** —
  ~**1.8×** the sentence-transformers endpoint and the fastest path measured.
- **vLLM's win is A10-specific.** On T4 it's a hair *slower* than sentence-transformers (no Flash-Attention
  on Turing; batching overhead doesn't pay off for a small encoder). **A10 ≫ T4** everywhere → **A10 is the
  right GPU** for ModernBERT.
- **TEI wasn't runnable on this serverless-only workspace** (no docker, no CUDA build toolchain, no custom
  container support) — a workspace-capability limitation, not a TEI shortcoming. On Databricks here,
  **vLLM-on-A10 is the high-throughput path**.
- **The `llm/v1/embeddings` custom-serving *entrypoint* is the cleaner packaging** and is documented as
  supported, but this workspace's current Model Serving release accepts only `llm/v1/chat` for the
  entrypoint route. The pyfunc route delivers the same vLLM engine today.

## Results

PubMedQA corpus (19,999 rows, mean 320 tok/row), inputs capped at **512 tokens**. tokens/s = input tokens
embedded per second, **peak** over a batch-size × client-concurrency sweep.

| Path | GPU | tokens/s | rows/s | best (bs / conc) | p50 latency |
|---|---:|---:|---:|---|---:|
| AI Runtime baseline (sentence-transformers, SDPA, raw) | A10 | **68,999** | 215 | 64 / 1 | — |
| **vLLM pyfunc endpoint** | **A10** | **89,530** ⭐ | 277 | 64 / 4 | 914 ms |
| vLLM pyfunc endpoint | T4 | 21,593 | 67 | 64 / 8 | 7,661 ms |
| sentence-transformers pyfunc endpoint | A10 | 49,498 | 152 | 64 / 4 | 1,677 ms |
| sentence-transformers pyfunc endpoint | T4 | 23,276 | 72 | 16 / 4 | 895 ms |
| Text Embeddings Inference (TEI) | A10/T4 | — | — | — | not runnable here (see below) |

Notes:
- **AI Runtime baseline** is the raw on-GPU encode ceiling (no serving/network overhead). `flash-attn` is
  **not** on the AI Runtime image, so ModernBERT ran with **SDPA** (not FA2) — the honest A10 ceiling
  without a flash-attn build.
- **vLLM A10 (89.5k) even exceeds the raw baseline (69k)** because vLLM's continuous batching + optimized
  kernels beat a naive single-process `SentenceTransformer.encode`, and the endpoint fans work across
  concurrent requests.
- Full sweep and per-run rows are written to a Delta results table (`<catalog>.<schema>.embedding_bench_results`).

## Recommendation

1. **Serve on A10 (`GPU_MEDIUM`) with vLLM** (pyfunc route below) for the high-throughput path.
2. Use **sentence-transformers pyfunc** if you want the simplest, most portable deployment (works A10 & T4,
   truncates long inputs automatically) at ~half the A10 throughput.
3. **Skip T4** for ModernBERT — no Flash-Attention on Turing, ~2–4× slower, and vLLM gives no benefit there.
4. Prefer the **`llm/v1/embeddings` entrypoint** once it's available on the workspace's Model Serving
   release — it's the cleaner, supported packaging for the same engine.

## Expected TEI throughput (estimated — not measured)

We could **not** run TEI on this workspace (see constraints below), so these are **estimates from a quick
review of a few public TEI benchmarks**, anchored to the throughput we *did* measure. Treat as
**medium-confidence** ballpark figures, not a TEI run on this exact model.

Method: the closest public signal is a BERT-base, 768-dim encoder (`bge-base-en-v1.5`) on A10-class GPUs at
seq-len 512 — TEI ≈ **~450 docs/s** ([HF Inference Endpoints TEI blog][1]) vs sentence-transformers ≈
**~206 rows/s** ([HF `uv-scripts` benchmark][2]) → **TEI ≈ ~2.2× sentence-transformers**. Applying that to
our measured sentence-transformers endpoint and cross-checking against our vLLM result:

| GPU | Estimated rows/s | Estimated tokens/s | Basis |
|---|---:|---:|---|
| A10 | ~300–350 | ~95k–115k | ~2.2× our ST endpoint (152 rows/s); ≈ our vLLM (277) or a bit above |
| T4 | ~80–110 | ~25k–35k | only ~1.1–1.5× our ST T4 (72) — no Flash-Attention on Turing |

- **A10 is where TEI shines** — Flash-Attention 2 + unpadding on Ampere (ModernBERT's fast path). That's
  the ~2× gap over sentence-transformers, and why TEI-on-A10 would likely land **around or slightly above
  our working vLLM (89.5k tok/s)**. Variable-length inputs (our corpus averages ~320 tok, not a flat 512)
  could push it higher via unpadding.
- **T4 gets little of that** — TEI's Turing image runs with Flash-Attention **off**, so on T4 it collapses
  to roughly the sentence-transformers/vLLM T4 ballpark. The GPU, not the server, is the bottleneck.
- **req/s** depends on batching: with dynamic batching and single-doc requests, ~300+ req/s on A10; a
  32-doc-per-request batch is ~10 req/s.

**Bottom line:** on A10, TEI would likely deliver ~95k–115k tok/s — the **same ballpark as the vLLM pyfunc
already deployed here (89.5k)**. The dominant lever is **A10 + Flash-Attention**, not TEI vs vLLM. Firming
this up requires actually running TEI, which needs compute that allows a container or the Rust binary
(a classic GPU cluster or custom-container serving) — not available on this serverless-only workspace.

Sources (quick review):
- [1] HF — Deploy Embedding Models with Inference Endpoints (TEI, bge-base A10G ~450 req/s): https://huggingface.co/blog/inference-endpoints-embeddings
- [2] HF `uv-scripts/embeddings` benchmark (bge-base A10G ~206 rows/s, 20k rows, seq-512): https://huggingface.co/datasets/uv-scripts/embeddings
- TEI docs & benchmark charts: https://huggingface.co/docs/text-embeddings-inference/index
- TEI repo: https://github.com/huggingface/text-embeddings-inference
- ModernBERT paper (tokens/s efficiency): https://aclanthology.org/2025.acl-long.127.pdf
- gigagpu TEI throughput across GPUs (docs/sec): https://gigagpu.com/embedding-throughput-benchmark-gigagpu/

---

## Environment constraints discovered

| Constraint | Impact |
|---|---|
| **Serverless-only (no classic clusters)** | Serverless GPU offers **A10 / H100, not T4**. T4 is reachable only via Model Serving `GPU_SMALL`. So the AI Runtime baseline and any "run a server on the box" approach (TEI) can only use A10 here. |
| **AI Runtime build env is FIPS/OpenSSL-strict** | vLLM spawns a subprocess to inspect the model architecture; in the notebook that subprocess dies on an OpenSSL error (`FATAL FIPS SELFTEST FAILURE`; unsetting `OPENSSL_FORCE_FIPS_MODE` flips it to `ssl.SSLError [CRYPTO] unknown error`; `OPENSSL_CONF=/dev/null` doesn't help). So **vLLM cannot be built in the AIR notebook** — but the Model Serving container builds the env separately and runs it fine. |
| **`env_pack` runs pip in the notebook** | `mlflow.register_model(..., env_pack="databricks_model_serving")` runs `pip install` **in the notebook**, where pip itself crashes on the same OpenSSL error while building an SSL context. Registering **without** `env_pack` defers the container build server-side and sidesteps it. |
| **`flash-attn` not preinstalled** | ModernBERT falls back to SDPA (FA2 is where ModernBERT's speed comes from; the A10 baseline understates a flash-attn-enabled deployment). |
| **No docker / no CUDA toolchain on serverless GPU** | TEI (a Rust binary / Docker image) can't be run or built here → the TEI benchmark is blocked. |

## The vLLM story (three blockers, and how we got past them)

Getting vLLM to serve embeddings took routing around three independent issues:

1. **Entrypoint route accepted only `llm/v1/chat` on this workspace.** The Custom LLM Serving *entrypoint*
   path (`metadata={"task": "...", "entrypoint": ...}`) failed at endpoint create for
   `task="llm/v1/embeddings"`: *"must have a task type that is one of the supported types: llm/v1/chat."*
   The public docs **do** list `llm/v1/embeddings` as supported, so this is a Model Serving **release
   difference**, not a permanent limitation (see below).
2. **Can't build vLLM in the AIR notebook** (OpenSSL/FIPS, above). Fix: **don't smoke-test vLLM in the
   notebook** — the Model Serving container is a separate build/runtime that loads it fine.
3. **`env_pack` pip install crashes** (same OpenSSL error, above). Fix: **register without `env_pack`** so
   the container env builds server-side. Also: pick dtype by GPU (**bf16 on A10, fp16 on T4** — Turing has
   no bf16), and **truncate inputs to `max_model_len`** client-side (vLLM *rejects* over-length inputs
   rather than truncating like sentence-transformers/TEI).

**Working vLLM recipe (pyfunc):** wrap `vllm.LLM(runner="pooling").embed()` in a pyfunc; register with
`mlflow.register_model(uri, name)` **without** `env_pack`; deploy on `GPU_MEDIUM`. See
`notebooks/02_vllm_serving.py`.

## `llm/v1/embeddings` entrypoint availability

The [Serve custom LLMs](https://docs.databricks.com/en/machine-learning/model-serving/serve-custom-llms.html)
docs list both `llm/v1/chat` and `llm/v1/embeddings` as supported entrypoint tasks (with a `--runner pooling`
vLLM example), requiring `mlflow>=3.12` and `databricks-sdk>=0.102.0`. Our registration matched the docs and
passed logging + registration; only the endpoint-create task check rejected embeddings on this workspace,
which currently allows `llm/v1/chat` only. This is a Model Serving **release/rollout difference** — if you
need the embeddings entrypoint, check availability with your Databricks account team, or use a workspace
where it's enabled. `notebooks/02c_vllm_entrypoint.py` is ready to deploy once it's available. Until then the
pyfunc route (above) serves the same vLLM engine.

---

## Notebooks (`embedding_serving/notebooks/`)

| Notebook | Purpose |
|---|---|
| `00_setup_and_data.py` | Create the schema + volume, snapshot the model, build the PubMedQA corpus + token stats, create the results table. |
| `01_airuntime_baseline.py` | Raw on-GPU encode tokens/s with sentence-transformers (runtime ceiling), A10. |
| `02_vllm_serving.py` | **Working vLLM path**: pyfunc-wrapped in-process engine, register **without** `env_pack`, deploy, benchmark. A10 & T4. |
| `02b_st_pyfunc_serving.py` | sentence-transformers pyfunc endpoint (the simple, portable path). A10 & T4. |
| `02c_vllm_entrypoint.py` | vLLM via the Custom LLM Serving **entrypoint** — ready to deploy where `llm/v1/embeddings` is available. |
| `03_tei_air.py` | TEI attempt on serverless GPU — documents the docker/toolchain blocker. |
| `04_bench_endpoint.py` | Standalone endpoint benchmark (decoupled from deploy; supports client-side truncation for vLLM). |

**Data / UC objects:** created under a `<catalog>.<schema>` you set at the top of each notebook — model
snapshot volume `huggingface/bioclinical_modernbert`, corpus `pubmedqa_corpus`, results
`embedding_bench_results`.

## Reproduce

Run `00` (serverless CPU) → `01`/`02`/`02b` on serverless GPU (`GPU_1xA10`), passing the `gpu` widget
(`A10`/`T4`). Endpoints deploy to `GPU_MEDIUM` (A10) / `GPU_SMALL` (T4). Use `04_bench_endpoint.py` to
re-benchmark any endpoint (pass `max_tokens=512` for vLLM). All runs append to the `embedding_bench_results`
table.
