# Vector Search benchmark config.
# Plain Python so notebooks, scripts, and local tools all import the same thing.
# No YAML, no notebook-path magic.

CATALOG = "shm"
SCHEMA = "genai"
VOLUME = "hf_cache"

# Source dataset (only used by 01_setup_data.py)
HF_DATASET = "KShivendu/dbpedia-entities-openai-1M"
TEXT_MAX_CHARS = 2048

# Embedding models -- both 1024-dim
EMBED_DIM = 1024
PPT_MODEL = "databricks-gte-large-en"
PT_HF_MODEL = "Qwen/Qwen3-Embedding-0.6B"
PT_UC_MODEL = "shm.genai.qwen3_emb_06b"
PT_ENDPOINT = "shm_qwen3_emb_06b"

# Scales. Flip DEV_MODE to run the full benchmark.
DEV_MODE = False
DEV_SCALES = ["10k"]
SCALES = {"10k": 10_000, "100k": 100_000, "1m": 1_000_000}
MODELS = ["ppt", "pt"]

# Databricks Vector Search
VS_ENDPOINT = "one-env-shared-endpoint-1"

# Lakebase
LAKEBASE_PROJECT = "vs-bench"
LAKEBASE_BRANCH = "production"
LAKEBASE_ENDPOINT = "primary"
LAKEBASE_DB = "vsbench"       # default `postgres` db can't host CREATE EXTENSION
# Profile is only used when running the Databricks CLI locally. Inside a
# Databricks notebook there's no ~/.databrickscfg -- the CLI auto-authenticates
# via DATABRICKS_HOST / OAuth env vars, so we omit -p PROFILE entirely.
LAKEBASE_PROFILE = "DEFAULT"

# Benchmark
BENCH_QUERIES = 200
BENCH_WARMUP = 20
BENCH_TOPK = 10
HNSW_EF_SEARCH = 80


def active_scales():
    """Return the scales dict to iterate over based on DEV_MODE."""
    if DEV_MODE:
        return {k: SCALES[k] for k in DEV_SCALES if k in SCALES}
    return dict(SCALES)


def combos():
    """All (scale, model) pairs active for this run."""
    return [(s, m) for s in active_scales() for m in MODELS]


def embed_table(scale, model):
    return f"{CATALOG}.{SCHEMA}.dbpedia_{scale}_{model}"


def vs_index_name(scale, model):
    return f"{CATALOG}.{SCHEMA}.dbpedia_{scale}_{model}_vsidx"


def pg_table_name(scale, model):
    return f"dbpedia_{scale}_{model}"
