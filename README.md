# F1 Intelligence System

A conversational Formula 1 analysis platform that combines live race data, a historical knowledge base, and LLM-powered analysis to answer any F1 question and generate comprehensive post-race reports.

Ask natural questions like *"How did Gasly perform in the 2024 Monaco GP?"*, get follow-up answers with context, compare drivers head-to-head with interactive charts, or generate full race reports — all from a single chat interface.

## Architecture

```
User Question (Chainlit UI)
       |
       +---> Conversational Mode ---> FastF1 data + RAG context ---> Groq LLM ---> Answer + Charts
       |
       +---> Report Mode ---> LangGraph Pipeline:
                                 data_agent --> rag_agent --> analysis_agent --> report_agent --> R2 Upload
```

The system operates in two modes:

**Conversational mode** — For natural questions. Detects race references from the message, fetches timing data from FastF1, retrieves historical context from ChromaDB, and answers via Groq/LLaMA. The LLM decides when to include interactive Plotly charts.

**Report mode** — Triggered by phrases like "generate a report for...". Runs the full LangGraph pipeline: fetch data, retrieve historical context, analyze storylines, generate a structured markdown report, and optionally upload to Cloudflare R2.

### Pipeline Nodes

| Node | Role | Input | Output |
|------|------|-------|--------|
| **data_agent** | Fetch race data from FastF1 | `season`, `round_number` | `raw_data` (results, laps, pits, sectors, weather) |
| **rag_agent** | Retrieve historical context from ChromaDB | `circuit_name`, `season` | `historical_context` |
| **analysis_agent** | Identify key storylines via LLM | `raw_data`, `historical_context` | `analysis` |
| **report_agent** | Generate structured markdown report | `analysis`, `raw_data` | `report`, `s3_report_url` |

All nodes share a typed `RaceReportState` dict. Each node checks for upstream errors and no-ops if one occurred, so a failed data fetch degrades gracefully rather than crashing the pipeline.

## Tech Stack

| Component | Technology | Why |
|-----------|-----------|-----|
| Orchestration | LangGraph | Typed shared state, streaming, error propagation, conditional routing |
| Data source | FastF1 | Official F1 timing data, local caching, rich data (tyre compounds, telemetry, weather) |
| Knowledge base | ChromaDB + sentence-transformers | Local vector store, no API keys needed, 3,078 chunks from 55 Wikipedia articles |
| LLM | Groq (LLaMA 3.3 70B) | Fast inference (~500 tok/s on LPU hardware), free tier, strong reasoning |
| Frontend | Chainlit | Chat UI with streaming steps, Plotly chart support, session management |
| Report storage | Cloudflare R2 | S3-compatible, free egress, shareable report URLs |
| Charts | Plotly | Interactive, dark-themed, LLM-controlled via tag system |

## Project Structure

```
f1-intelligence-system/
  chainlit_app/
    app.py                 # Chainlit UI — conversational handler, chart builders, report pipeline
  src/
    state.py               # RaceReportState TypedDict shared across all nodes
    graph.py               # LangGraph wiring (data -> rag -> analysis -> report -> END)
    agents/
      data_agent.py        # Fetches race data via FastF1Client
      rag_agent.py         # 3-query retrieval from ChromaDB, deduplication
      analysis_agent.py    # LLM analysis — storylines, DOTD, strategy
      report_agent.py      # LLM report generation + R2 upload
    data/
      fastf1_client.py     # FastF1 wrapper — async, cached sessions, DataFrame-to-dict conversion
    rag/
      retriever.py         # ChromaDB retriever with cached embedding model
    utils/
      config.py            # Settings dataclass loaded from .env
      s3_uploader.py       # Async R2 upload via boto3
  scripts/
    fetch_wiki_docs.py     # Fetch 55 Wikipedia articles (circuits, drivers, seasons)
    ingest_docs.py         # Chunk and embed docs into ChromaDB
  tests/
    agents/                # Mocked tests for all 4 agents
    data/                  # Mocked tests for FastF1Client
    rag/                   # Mocked tests for retriever
  knowledge_base/
    raw_docs/              # Downloaded Wikipedia articles
    chroma_db/             # ChromaDB vector store (gitignored)
  cache/
    fastf1/                # FastF1 timing data cache (gitignored)
```

## Setup

### Prerequisites

- Python 3.13+
- [uv](https://docs.astral.sh/uv/) package manager
- A [Groq API key](https://console.groq.com/keys) (free tier)
- Cloudflare R2 credentials (optional — reports still render in the UI without it)

### Installation

```bash
git clone https://github.com/ombharat6361/f1-intelligence-system.git
cd f1-intelligence-system
uv sync
```

### Environment Variables

Copy `.env.example` to `.env` and fill in your keys:

```bash
cp .env.example .env
```

| Variable | Required | Purpose |
|----------|----------|---------|
| `GROQ_API_KEY` | Yes | LLaMA 3.3 70B via Groq |
| `DISABLE_R2` | No | Set `true` to skip R2 upload (default: `false`) |
| `R2_ACCOUNT_ID` | No | Cloudflare account ID |
| `R2_ACCESS_KEY_ID` | No | R2 API token key ID |
| `R2_SECRET_ACCESS_KEY` | No | R2 API token secret |
| `R2_BUCKET_NAME` | No | R2 bucket name (default: `f1-intelligence-reports`) |
| `R2_PUBLIC_URL` | No | Public URL base for uploaded reports |
| `CHROMA_PERSIST_DIR` | No | ChromaDB path (default: `./knowledge_base/chroma_db`) |
| `FASTF1_CACHE_DIR` | No | FastF1 cache path (default: `./cache/fastf1`) |

### Build the Knowledge Base

```bash
# Fetch Wikipedia articles (circuits, drivers, seasons)
uv run python scripts/fetch_wiki_docs.py

# Chunk and embed into ChromaDB
uv run python scripts/ingest_docs.py
```

### Run

```bash
uv run chainlit run chainlit_app/app.py
```

Open [http://localhost:8000](http://localhost:8000) and start asking questions.

## Usage

### Natural Questions

```
How did Gasly perform in the 2024 Monaco GP?
```
```
Compare Verstappen and Norris at the 2024 Australian GP
```
```
What happened at the 2023 Bahrain race?
```

The app detects race references (year + GP name, country, or round number), fetches data, and answers with real numbers. Follow-up questions remember the current race context.

### Supported Input Formats

| Format | Example |
|--------|---------|
| Year + GP name | `2024 Australian GP` |
| Year + country | `2024 Monaco` |
| Year + city | `2024 Melbourne` |
| Year + round | `2024 round 3` or `2024 r3` |

### Full Reports

```
Generate a report for the 2024 Australian GP
```

Triggers the full pipeline and produces a structured markdown report with sections: Race Summary, Key Storylines, Driver of the Day, Strategy Breakdown, Championship Standings Update, and Looking Ahead.

### Charts

The LLM automatically includes interactive Plotly charts when a visual would help — lap time comparisons, sector breakdowns, pit stop analysis, and head-to-head driver comparisons.

## Available Race Data

Data fetched per session via FastF1:

| Category | Fields |
|----------|--------|
| Race results | Position, grid position, status (Finished/Retired/Lapped), points, driver, team |
| Lap times | Fastest lap per driver (duration, lap number) |
| Sector times | Best S1/S2/S3 per driver |
| Pit stops | Count, total duration, individual stop times and laps |
| Weather | Air temp, track temp, humidity, pressure, wind speed/direction, rainfall |

## Testing

```bash
uv run pytest               # all 50 tests
uv run pytest -v             # verbose output
uv run pytest tests/data/    # just the FastF1 client tests
```

All tests are mocked — no network calls or API keys needed. Runs in ~2 seconds.

## License

MIT License
