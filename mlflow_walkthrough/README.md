# MLflow 3 GenAI Evaluation Walkthrough

End-to-end, runnable tour of the Databricks MLflow 3 evaluation lifecycle using a
small LangGraph ReAct agent against the public arXiv API.

## The flow

Matches the Databricks MLflow 3 UI ordering:

1. **Trace** — autolog + named `SpanType.TOOL` spans on the arXiv tools
2. **Sessions** — multi-turn traces grouped by `mlflow.trace.session` metadata
3. **Judges** — built-in (`RelevanceToQuery`, `Safety`) + a custom `@scorer` that
   reads tool spans to catch hallucinated citations
4. **Evaluation Datasets** — Unity Catalog-backed with ground-truth arXiv IDs
5. **Evaluation Runs** — `mlflow.genai.evaluate()` on v1 and v2 of the agent
6. **Labeling Schemas** — three schemas (answer quality, cited correctly, notes)
7. **Labeling Sessions** — Review App session over the v1 failures
8. **Prompts / Agent Versioning** — UC Prompt Registry with `@production` / `@candidate`
   aliases

## Files

- `arxiv_eval_walkthrough.ipynb` — the notebook, run top to bottom
- `arxiv_tools.py` — `search_arxiv` and `fetch_arxiv_paper`; both wrapped as
  LangChain tools and decorated with `@mlflow.trace(span_type=SpanType.TOOL, ...)`
- `arxiv_agent.py` — LangGraph ReAct agent builder + `run_turn` session helper
- `eval_questions.py` — eight seed research questions with expected arXiv IDs
- `EMAIL.md` — shareable ~500-word walkthrough of the same flow

## Prerequisites

- Databricks workspace with Foundation Model API access
- A Unity Catalog `catalog.schema` you can write to (for the Prompt Registry and the
  evaluation dataset)
- Internet egress to `export.arxiv.org` (the only external dependency; no auth)
- Databricks Serverless environment v4 (or equivalent ML Runtime)

## Running it

1. Open `arxiv_eval_walkthrough.ipynb` in a Databricks notebook.
2. Edit the `CATALOG` and `SCHEMA` cell (Step 0) to a UC location you can write to.
3. Run all cells. Expected runtime: 3–5 minutes.
4. When you hit the labeling session cell (Step 7), open the printed Review App URL,
   label a few traces, then continue.

## What to look for in the UI

- **Traces tab:** each agent turn has an `AGENT` span with `LLM` and named TOOL
  children (`search_arxiv`, `fetch_arxiv_paper`).
- **Sessions tab:** the two-turn demo in Step 2 shows both turns grouped.
- **Evaluation tab:** `eval-v1` and `eval-v2` runs side by side; `citation_correctness`
  improves on v2.
- **Prompts tab:** `{catalog}.{schema}.arxiv_agent_system_prompt` with two versions
  and two aliases.
