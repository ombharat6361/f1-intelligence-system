# F1 Intelligence System

A LangGraph pipeline that generates post-race intelligence reports for any Formula 1 race. Ask for a race in plain English — "2024 Australian GP" or "last year's Monaco race" — and get a structured analyst-style report with key storylines, strategy breakdown, driver of the day, and championship implications.

## How it works

```
User input → LLM extraction → OpenF1 session lookup → LangGraph pipeline → Markdown report
```

The pipeline has four nodes:

- **data_agent** — fetches race results, lap times, pit stops, and sector times from the [OpenF1 API](https://openf1.org)
- **rag_agent** — retrieves circuit history, driver profiles, and season context from a local ChromaDB knowledge base
- **analysis_agent** — identifies 4–6 key race storylines using LLaMA 3.1 70B via Groq
- **report_agent** — generates a structured Markdown report and optionally uploads it to Cloudflare R2

The Chainlit UI accepts natural-language race requests, resolves them to an OpenF1 session via an LLM call (handling typos and relative references like "last year"), and streams pipeline progress in real time.

## Setup

**1. Install dependencies**

```bash
pip install -r requirements.txt
```

**2. Configure environment**

```bash
cp .env.example .env
```

Fill in at minimum `GROQ_API_KEY`. Set `DISABLE_R2=true` to skip cloud upload — the report still renders in the UI.

**3. Build the knowledge base**

```bash
python scripts/fetch_wiki_docs.py      # downloads 55 Wikipedia articles (~2 min)
python scripts/ingest_docs.py          # chunks and embeds into ChromaDB (~5 min)
```

**4. Run the app**

```bash
chainlit run chainlit_app/app.py
```

## Environment variables

| Variable | Required | Description |
|---|---|---|
| `GROQ_API_KEY` | Yes | LLaMA 3.1 70B via Groq |
| `DISABLE_R2` | No | Set `true` to skip R2 upload (default: `false`) |
| `R2_ACCOUNT_ID` | If R2 enabled | Cloudflare account ID |
| `R2_ACCESS_KEY_ID` | If R2 enabled | R2 API token key ID |
| `R2_SECRET_ACCESS_KEY` | If R2 enabled | R2 API token secret |
| `R2_BUCKET_NAME` | If R2 enabled | Bucket name (default: `f1-intelligence-reports`) |
| `R2_PUBLIC_URL` | No | Public URL base for uploaded reports |
| `CHROMA_PERSIST_DIR` | No | ChromaDB path (default: `./knowledge_base/chroma_db`) |
| `OPENF1_BASE_URL` | No | OpenF1 base URL (default: `https://api.openf1.org/v1`) |

## Knowledge base

The knowledge base covers 24 circuits, 26 drivers, and seasons 2020–2024.

To re-ingest from scratch:

```bash
python scripts/ingest_docs.py --reset
```

To fetch only specific categories:

```bash
python scripts/fetch_wiki_docs.py --only circuits
python scripts/fetch_wiki_docs.py --only drivers
python scripts/fetch_wiki_docs.py --seasons 2023 2024
```

## Tests

```bash
pytest                          # all tests
pytest tests/agents/            # agent unit tests (mocked, no credentials needed)
pytest tests/rag/               # RAG retriever tests
pytest tests/test_graph.py      # graph integration test
pytest tests/data/              # live OpenF1 API tests (requires network)
```

## Project structure

```
src/
  agents/         — four LangGraph node functions
  data/           — OpenF1 API client
  rag/            — ChromaDB retriever
  utils/          — config, R2 uploader
  graph.py        — LangGraph wiring
  state.py        — shared RaceReportState TypedDict
chainlit_app/
  app.py          — Chainlit UI
scripts/
  fetch_wiki_docs.py  — Wikipedia scraper
  ingest_docs.py      — ChromaDB ingestion
knowledge_base/
  raw_docs/       — downloaded Wikipedia articles
  chroma_db/      — vector store (generated)
tests/
```
