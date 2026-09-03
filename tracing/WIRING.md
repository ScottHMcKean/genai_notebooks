# How this is wired — custom-agent tracing, governance & artifacts

Reference for the Monte Carlo forecasting demo: the components, the exact queries that
make each piece work (verified on a Databricks workspace), the measured latencies, and
the docs behind each decision. Companion to [`README.md`](README.md); the runnable
walkthrough is [`01_custom_agent_tracing.py`](01_custom_agent_tracing.py).

---

## Key components

| Component | File | What it does |
|---|---|---|
| **Agent** | [`agent.py`](agent.py) | `ForecastChatAgent`, an MLflow **ResponsesAgent**. LLM tool-loop over one tool, `run_monte_carlo` (seeded NumPy). `mlflow.openai.autolog()` traces every LLM call; `@mlflow.trace` gives the sim its own TOOL span. |
| **Walkthrough** | [`01_custom_agent_tracing.py`](01_custom_agent_tracing.py) | Notebook, sections a→e: autolog, UC trace storage, history retrieval, governance, latency. |
| **App** | [`app/`](app/README.md) | FastAPI chat over the same agent. Non-streaming JSON turns, per-user history grouped by session, distribution-chart artifacts. |
| **Trace store** | UC Delta tables | `…forecast_traces_otel_{spans,logs,metrics,annotations}` — OpenTelemetry spans, governed by Unity Catalog. |
| **Governed view** | UC view | `…forecast_traces_my_spans` — trace-scoped row security (`current_user()`). |
| **Artifacts** | UC Volume | `…/forecast_demo/artifacts/<trace-id>.png` — one distribution chart per trace. |

**The mental model:** on Databricks, MLflow traces *are* OpenTelemetry and, once an
experiment is bound to a UC trace location, *are* governed Delta tables. So chat history
is a `search_traces` query, and per-user isolation is Unity Catalog — no separate history
DB, no separate access layer.

---

## Data flow (one turn)

```
user ──▶ App /api/chat ──▶ start_span("forecast_turn")  ← stamps mlflow.trace.user/session
                              │
                              ├─▶ AGENT.predict ──▶ LLM (autolog span)
                              │                 └─▶ run_monte_carlo (TOOL span)
                              │                 └─▶ LLM explains (autolog span)
                              ├─▶ chart PNG ──▶ UC Volume  (+ trace tag artifact.chart)
                              └─▶ trace ──(async, OTLP)──▶ UC Delta tables
history / governance / latency  ◀── mlflow.search_traces + SQL over those tables
```

---

## Key queries

### a · Instrument (autolog + one manual span)

```python
import mlflow
mlflow.openai.autolog()                                   # every LLM call, no code in the loop

@mlflow.trace(span_type=mlflow.entities.SpanType.TOOL, name="monte_carlo_simulation")
def run_monte_carlo(...): ...
```

Tag **user + session** on a synchronous root span opened at the call site (stable trace
boundary; avoids the rapid-fire export race we hit when relying on the auto-traced
`predict` alone):

```python
with mlflow.start_span(name="forecast_turn", span_type=mlflow.entities.SpanType.AGENT) as span:
    span.set_inputs({"question": q, "user": user, "session_id": sid})
    mlflow.update_current_trace(metadata={
        "mlflow.trace.user": user,          # exported to the OTel attribute user.id
        "mlflow.trace.session": sid,        # exported to session.id
    })
    resp = AGENT.predict(...)
    span.set_outputs({"answer": answer})
```

### b · Store OpenTelemetry traces in Unity Catalog

```python
import os, mlflow
from mlflow.entities.trace_location import UnityCatalog

os.environ["MLFLOW_TRACING_SQL_WAREHOUSE_ID"] = SQL_WAREHOUSE_ID   # required
mlflow.set_tracking_uri("databricks")
mlflow.set_experiment(
    experiment_name=EXPERIMENT_PATH,
    trace_location=UnityCatalog(catalog_name=CATALOG, schema_name=SCHEMA,
                                table_prefix="forecast_traces"),
)
```

Creates four tables: `forecast_traces_otel_{spans,logs,metrics,annotations}`.
**Binding is one-shot** — a UC trace location attaches only to an experiment with no
traces yet and can't be reassigned; use a fresh experiment to rebind.

### c · Retrieve traces (this *is* the history store)

`user`/`session` are trace **metadata**; filter with backtick-quoted keys:

```python
# one user's conversation, in order
mlflow.search_traces(
    experiment_ids=[EID],
    filter_string="metadata.`mlflow.trace.user` = 'alice@example.com' "
                  "AND metadata.`mlflow.trace.session` = 'sess-b5e8fabf'",
    order_by=["timestamp_ms ASC"],
    return_type="list",          # stable Trace objects (MLflow 3 renamed the DataFrame columns)
)
```

The question/answer live on the **root span's** inputs/outputs (not `trace.data.request`,
which reflects the inner auto-traced `predict`):

```python
root = next(s for s in trace.data.spans if s.parent_id is None)
question, answer = root.inputs.get("question"), root.outputs.get("answer")
```

**History grouped by session** (one box per conversation): group the caller's traces by
`mlflow.trace.session`, newest first, title = the session's opening question.

### d · Governance — a user only sees their own traces

**App layer** — the only history query the app runs is pinned to the authenticated caller
(identity from the Apps `X-Forwarded-Email` header, never the request body):

```python
mlflow.search_traces(experiment_ids=[EID],
    filter_string=f"metadata.`mlflow.trace.user` = '{caller}'", return_type="list")
```

**Data layer** — the OTel table is *per span*, and the user lands only on each trace's
**root** span (as the attribute `user.id`; session is `session.id`), inside the
`attributes` VARIANT. A dotted key needs bracket syntax. Because a naive row filter on
`user.id` would hide every child span, use a **trace-scoped secure view**:

```sql
-- inspect the schema first: attributes is a VARIANT; user.id is on the root span only
SELECT attributes:['user.id']::string  AS user_id,
       attributes:['session.id']::string AS session_id
FROM   forecast_traces_otel_spans
WHERE  parent_span_id IS NULL OR parent_span_id = '';

-- view: resolve each trace's owner from its root span, return ALL spans of owned traces
CREATE OR REPLACE VIEW forecast_traces_my_spans AS
WITH trace_owner AS (
  SELECT trace_id, max(attributes:['user.id']::string) AS trace_user
  FROM forecast_traces_otel_spans GROUP BY trace_id)
SELECT s.*, o.trace_user
FROM forecast_traces_otel_spans s
JOIN trace_owner o ON s.trace_id = o.trace_id
WHERE is_account_group_member('trace-admins')   -- auditors see everything
   OR o.trace_user = current_user();             -- everyone else: only their own
```

Grant the **view**, not the base table: `GRANT SELECT ON VIEW …_my_spans TO <group>`.
Verify by querying the view as a non-owner → **0 rows**.

### e · Latency

**Generation** — from span durations already in UC:

```sql
SELECT coalesce(attributes:['mlflow.spanType']::string, name) AS span_type,
       count(*) AS n,
       round(percentile((end_time_unix_nano - start_time_unix_nano)/1e6, 0.5), 0) AS p50_ms,
       round(percentile((end_time_unix_nano - start_time_unix_nano)/1e6, 0.95), 0) AS p95_ms
FROM   forecast_traces_otel_spans
GROUP BY 1 ORDER BY p50_ms DESC;
```

**Retrieval** — time the tracking-server reads:

```python
import time
t = time.time(); mlflow.search_traces(experiment_ids=[EID], return_type="list"); search_ms = (time.time()-t)*1000
t = time.time(); mlflow.get_trace(trace_id);                                     get_ms    = (time.time()-t)*1000
```

### Artifacts (chart per trace, in a UC Volume)

```python
# write (app SP needs WRITE VOLUME) — slug the UC trace id 'trace:/…/<hex>' into a filename
path = f"{VOLUME}/{re.sub(r'[^A-Za-z0-9_-]', '_', trace_id)}.png"
ws.files.upload(path, io.BytesIO(png), overwrite=True)
mlflow.set_trace_tag(trace_id, "artifact.chart", path)     # lineage: trace → artifact
# read (serving) — a single Files-API GET, base64-inlined
png = ws.files.download(path).contents.read()
```

```sql
CREATE VOLUME IF NOT EXISTS <catalog>.<schema>.artifacts;
GRANT READ VOLUME, WRITE VOLUME ON VOLUME <catalog>.<schema>.artifacts TO `<app-sp>`;
```

**Volume vs. MLflow run artifacts:** a Volume read is one Files-API GET (sub-second);
run artifacts add hops (resolve run → artifact store) and bind the file to a run rather
than the trace the user is viewing. Same UC governance either way — the Volume is the
faster serving path.

---

## Measured latency (this workspace, small dataset)

| Generation (per turn) | p50 | p95 |
|---|---|---|
| Full turn (root span) | ~5.1 s | ~7.7 s |
| LLM call | ~3.5 s | ~5.5 s |
| Monte Carlo tool | **~10 ms** | ~13 ms |

→ A turn is essentially two LLM calls; the simulation is negligible.

| Retrieval | cold | warm p50 |
|---|---|---|
| `search_traces` (all / by user / by user+session) | ~5–6 s | **~3.2 s** |
| `get_trace` by id | ~1.7 s | **~0.6 s** |

→ `search_traces` goes through the SQL warehouse (~3.2 s warm, ~flat across filters at
this size). `get_trace` by id is a direct fetch. Load history on open; neither is on the
chat hot path.

---

## Permissions checklist (deployed app's service principal)

- `CAN_QUERY` on the LLM serving endpoint (Foundation Model APIs are open to all
  principals by default).
- `CAN_EDIT` on the MLflow experiment (to write traces).
- `CAN_USE` on the SQL warehouse (to read traces via `search_traces`).
- UC: `USE CATALOG` + `USE SCHEMA`; `SELECT, MODIFY` on each `…_otel_*` table (write +
  read traces); `SELECT` on the view; `READ VOLUME, WRITE VOLUME` on the artifacts volume.
- Note: `ALL_PRIVILEGES` is **not** sufficient for UC trace tables — grant the explicit
  privileges above.

---

## Doc references

- **Store OpenTelemetry traces in Unity Catalog** — <https://docs.databricks.com/aws/en/mlflow3/genai/tracing/trace-unity-catalog>
- **Track users & sessions** — <https://docs.databricks.com/aws/en/mlflow3/genai/tracing/track-users-sessions/>
- **MLflow — search traces (filter syntax)** — <https://mlflow.org/docs/latest/genai/tracing/search-traces.md>
- **MLflow — automatic tracing / autolog** — <https://mlflow.org/docs/latest/genai/tracing/app-instrumentation/automatic.md>
- **MLflow — ResponsesAgent** — <https://mlflow.org/docs/latest/genai/flavors/responses-agent-intro.md>
- **Row filters & column masks (UC)** — <https://docs.databricks.com/aws/en/tables/row-and-column-filters>
- **UC row-filter / column-mask patterns** — <https://docs.databricks.com/aws/en/data-governance/unity-catalog/abac/common-patterns>
- **Unity Catalog Volumes** — <https://docs.databricks.com/aws/en/volumes/>
- **Databricks Apps (deploy, identity headers)** — <https://docs.databricks.com/aws/en/dev-tools/databricks-apps/>
