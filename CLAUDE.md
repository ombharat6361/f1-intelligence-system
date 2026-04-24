# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project status

Core pipeline and UI are fully implemented. Knowledge base is populated. All tests are written. Only Docker remains as a gap.

### Implemented

- `src/agents/` — all four agent nodes (`data_agent`, `rag_agent`, `analysis_agent`, `report_agent`).
- `src/state.py` — `RaceReportState` TypedDict shared across all nodes.
- `src/graph.py` — LangGraph wiring; compiles via `build_graph()`, used as a singleton in the Chainlit app.
- `src/data/openf1_client.py` — async `OpenF1Client` with `get_race_results`, `get_lap_times`, `get_pit_stops`, `get_sector_times`. Uses `httpx` + `tenacity` retries; gracefully degrades on missing/empty API responses. Config import is lazy (no env var needed at import time).
- `src/utils/config.py` — frozen dataclass `Settings` loaded from env via `python-dotenv`. `GROQ_API_KEY` defaults to `""` at load time and only fails when `ChatGroq` is instantiated — other parts of the system (ingestion, data fetching) can import without the key set.
- `src/utils/s3_uploader.py` — async `upload_report(report, filename)` uploading to **Cloudflare R2** via boto3 (S3-compatible, custom endpoint `https://<account_id>.r2.cloudflarestorage.com`). Returns `R2_PUBLIC_URL/<filename>` if set, otherwise the internal endpoint path. Upload failure in `report_agent` is non-fatal.
- `src/rag/retriever.py` — exports `get_retriever(top_k)` returning a LangChain retriever over a local ChromaDB store. Embedding model: `sentence-transformers/all-MiniLM-L6-v2` (local, no API key). Module-level cached. Returns `[]` gracefully if the store is empty.
- `chainlit_app/app.py` — Chainlit UI. Collects session key, season, round number, circuit name via conversational prompts, streams pipeline progress using nested `cl.Step`, renders the final markdown report, and posts the R2 URL if available.
- `scripts/fetch_wiki_docs.py` — fetches 55 Wikipedia articles (24 circuits, 26 drivers, 5 seasons 2020–2024) into `knowledge_base/raw_docs/`. Skips already-downloaded files, rate-limits at 0.5 s/request. Supports `--only {circuits,drivers,seasons}`, `--seasons`, `--dry-run`.
- `scripts/ingest_docs.py` — chunks `.txt`/`.md`/`.pdf` files from `knowledge_base/raw_docs/` and embeds them into ChromaDB. Supports `--reset` to wipe and re-ingest.
- `knowledge_base/raw_docs/` — populated with 55 Wikipedia articles (3 078 chunks in ChromaDB).
- `tests/data/test_openf1_client.py` — 12 integration tests against the live OpenF1 API (session `9158`, 2024 Australian GP).
- `tests/agents/` — tests for all four agent nodes.
- `tests/rag/` — tests for the retriever.
- `pytest.ini` — sets `asyncio_mode = auto`.

### Still empty

`docker/`.

## Architecture

The system is a LangGraph pipeline that generates a post-race F1 intelligence report. All nodes share `RaceReportState` (`src/state.py`), a TypedDict seeded with `race_id` (OpenF1 session key), `season`, `round_number`, `circuit_name`.

Flow:

```
data_agent → rag_agent → analysis_agent → report_agent → END
```

- **data_agent** populates `state["raw_data"]` from OpenF1. On exception it sets `state["error"]`; downstream nodes check this and return `{}` to short-circuit.
- **rag_agent** runs three queries (circuit history, drivers, season context) against ChromaDB, dedupes chunks, writes `state["historical_context"]`.
- **analysis_agent** calls Groq (`llama-3.1-70b-versatile`) to identify 4–6 race storylines from raw data + historical context.
- **report_agent** calls Groq to generate a fixed-section Markdown report, then attempts R2 upload (non-fatal on failure, can be disabled via `DISABLE_R2=true`).

The error-propagation contract (each node checks `state.get("error")` and no-ops) keeps a failed OpenF1 fetch from crashing the whole graph — preserve it when adding nodes.

## Commands

- Install: `pip install -r requirements.txt`
- Fetch knowledge base docs: `python scripts/fetch_wiki_docs.py`
- Ingest into ChromaDB: `python scripts/ingest_docs.py [--reset]`
- Tests: `pytest` (single test: `pytest tests/path/to/test_file.py::test_name`)
- Chainlit UI: `chainlit run chainlit_app/app.py`

## Environment variables

All vars are documented in `.env.example`. Required to run the full pipeline:

| Variable | Purpose |
|---|---|
| `GROQ_API_KEY` | LLaMA 3.1 70B via Groq (analysis + report agents) |
| `DISABLE_R2` | Set to `true` to skip R2 upload entirely (default: `false`) |
| `R2_ACCOUNT_ID` | Cloudflare account ID (from R2 dashboard sidebar) |
| `R2_ACCESS_KEY_ID` | R2 API token key ID |
| `R2_SECRET_ACCESS_KEY` | R2 API token secret |
| `R2_BUCKET_NAME` | R2 bucket name (default: `f1-intelligence-reports`) |
| `R2_PUBLIC_URL` | Optional public URL base for uploaded reports |
| `CHROMA_PERSIST_DIR` | Local path to ChromaDB store (default: `./knowledge_base/chroma_db`) |
| `OPENF1_BASE_URL` | OpenF1 API base (default: `https://api.openf1.org/v1`) |

R2 vars are optional — set `DISABLE_R2=true` to skip upload, and the report still renders in the UI. Upload failure is always non-fatal.
