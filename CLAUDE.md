# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project status

This is an early-stage scaffold. The four agents in `src/agents/` are written, but several modules they import do not exist yet and will need to be created when extending the system:

- `src.data.openf1_client.OpenF1Client` (used by `data_agent.py`) — async client for `https://api.openf1.org/v1`; must implement `get_race_results`, `get_lap_times`, `get_pit_stops`, `get_sector_times`, all keyed by an OpenF1 session key.
- `src.rag.retriever.get_retriever(top_k)` (used by `rag_agent.py`) — returns a LangChain retriever backed by the Chroma store at `CHROMA_PERSIST_DIR`.
- `src.utils.config.settings` — pydantic/env-driven settings exposing at least `groq_api_key`; envs listed in `.env.example`.
- `src.utils.s3_uploader.upload_report(report, filename)` — async upload to `S3_BUCKET_NAME`, returning a URL.

`chainlit_app/`, `docker/`, `scripts/`, `knowledge_base/raw_docs/`, and the `tests/*` subpackages are empty placeholders.

## Architecture

The system is a LangGraph pipeline that generates a post-race F1 intelligence report. All nodes share `RaceReportState` (`src/state.py`), a TypedDict seeded with `race_id` (OpenF1 session key), `season`, `round_number`, `circuit_name`.

Flow (`src/graph.py`, compiled once as `race_graph`):

```
data_agent → rag_agent → analysis_agent → report_agent → END
```

- **data_agent** populates `state["raw_data"]` from OpenF1. On exception it sets `state["error"]` and downstream nodes short-circuit by returning `{}`.
- **rag_agent** runs three circuit/driver/season queries against Chroma, dedupes chunks, writes `state["historical_context"]`.
- **analysis_agent** and **report_agent** both call Groq (`llama-3.1-70b-versatile`) via `ChatGroq`. Analysis produces structured storylines; report turns them into a fixed-section Markdown document and attempts S3 upload (failure is non-fatal — `s3_report_url` stays `None`).

The error-propagation contract (each downstream node checks `state.get("error")` and no-ops) is what keeps a failed OpenF1 fetch from crashing the whole graph — preserve it when adding nodes.

## Commands

No Makefile, tox, or CI config is present. Use raw tooling:

- Install: `pip install -r requirements.txt`
- Tests: `pytest` (run a single test: `pytest tests/path/to/test_x.py::test_name`)
- Chainlit UI (once `chainlit_app/` has an entry module): `chainlit run chainlit_app/<app>.py`

Environment variables required to actually run the graph are in `.env.example` (Groq key, AWS creds + bucket, Chroma persist dir, OpenF1 base URL).
