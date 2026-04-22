**Subject:** MLflow 3 on Databricks — the opinionated end-to-end evaluation flow

Team,

If you've opened the MLflow 3 UI on Databricks lately, you'll see the GenAI panel is
now organized as a single linear workflow:

> Trace → Sessions → Judges → Evaluation Datasets → Evaluation Runs → Labeling Schemas → Labeling Sessions → Prompts / Agent Versioning

The one thing to internalize: **that ordering is the recommended build order.** Do
them in sequence, don't skip ahead. Here's what each stage is for and when to reach
for it.

**1. Trace.** Start every GenAI project by instrumenting. Use framework autologging
(`mlflow.langchain.autolog()`, `mlflow.openai.autolog()`) plus `@mlflow.trace` on any
tool function so tool invocations get their own named `SpanType.TOOL` spans. Do this
on day one — every stage below reads from traces.
Docs: https://mlflow.org/docs/latest/genai/tracing/

**2. Sessions.** For multi-turn apps, tag each trace with
`mlflow.update_current_trace(metadata={"mlflow.trace.session": session_id})`. The UI
groups by this automatically. One gotcha: session metadata is immutable, so set it
before the first tool call on each turn.
Docs: https://docs.databricks.com/aws/en/mlflow3/genai/tracing/track-users-sessions/

**3. Judges.** Start with the built-ins: `RelevanceToQuery`, `RetrievalGroundedness`,
`Safety`, `Correctness`. Add a custom `@scorer` only when you need to inspect
something the built-ins can't see — typically a specific tool span's inputs or
outputs. Custom code scorers are deterministic and cheap; reach for them before you
reach for another LLM judge.
Docs: https://mlflow.org/docs/latest/genai/concepts/scorers/

**4. Evaluation Datasets.** Curate from real traces plus reviewer expectations, not
from synthetic prompts. Create a Unity-Catalog-backed dataset with
`mlflow.genai.create_dataset()` and grow it one production failure at a time. Every
row you add is a regression test you'll never have to write again.
Docs: https://mlflow.org/docs/latest/genai/concepts/evaluation-datasets/

**5. Evaluation Runs.** `mlflow.genai.evaluate()` with a `predict_fn`, the dataset,
and your scorer list. One run per app version. The UI's side-by-side compare view is
where prompt and code changes earn their keep. If a change doesn't move a metric you
care about, it wasn't an improvement.
Docs: https://mlflow.org/docs/latest/genai/eval-monitor/

**6. Labeling Schemas.** Define before you open the session. Pick `type="feedback"`
when you want reviewer opinions, `type="expectation"` when you want ground truth that
syncs back to the dataset. A missing schema late in a review cycle means throwing
labels away.
Docs: https://docs.databricks.com/aws/en/mlflow3/genai/human-feedback/concepts/labeling-schemas

**7. Labeling Sessions.** Route *failing* traces from an eval run into a labeling
session. Don't bulk-label; you'll burn reviewer time on examples that were already
right. Sync expectations from the session back into the evaluation dataset so the
next evaluation run uses the new ground truth.
Docs: https://docs.databricks.com/aws/en/mlflow3/genai/human-feedback/concepts/labeling-sessions

**8. Prompts / Agent Versioning.** Every prompt lives in the UC Prompt Registry. Use
aliases (`@production`, `@candidate`) and load by alias in your agent — promoting is
a one-line alias move, no deploy. Register the agent itself as a `LoggedModel` and
bump it every time code changes. Never edit in place.
Docs: https://docs.databricks.com/aws/en/mlflow3/genai/prompt-version-mgmt/prompt-registry/

**Three rules we've found worth keeping:**

- Don't ship a custom LLM judge you haven't aligned on at least ten labeled examples.
- Every prompt change gets a prompt version; every code change gets a LoggedModel
  version. Never edit in place.
- Trace in dev; monitor the same scorers in prod. There should not be a second eval
  system.

**Companion notebook.** `mlflow_walkthrough/arxiv_eval_walkthrough.ipynb` runs the
whole flow end-to-end using a LangGraph ReAct agent against the public arXiv API. It
ships a weak v1 on purpose so you can watch a custom scorer catch a hallucinated
citation on a specific tool span, collect labels on the failures, then fix the
behaviour by versioning the prompt — no code change.

Pick your app. Start with traces. Grow outward.
