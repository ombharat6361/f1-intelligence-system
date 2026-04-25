# Plan: Replace OpenF1 with FastF1 + Jolpica

## Context

OpenF1 can't distinguish DNFs from classified finishes, lacks tyre compounds, weather, and telemetry. FastF1 wraps official F1 timing data and solves all of these. Its built-in Jolpica interface (Ergast successor) adds championship standings, which the report currently has to hallucinate or guess from RAG context.

## Data flow — before vs. after

```
BEFORE:
  chainlit_app/app.py                      src/agents/data_agent.py
  ┌─────────────────────┐                  ┌──────────────────────┐
  │ OpenF1Client         │                  │ OpenF1Client          │
  │ .get_race_session()  │──► race_id ──►  │ .get_race_results()   │
  │ (year, circuit_name) │   stored in     │ .get_lap_times()      │──► raw_data
  │                      │   session +     │ .get_pit_stops()      │
  └─────────────────────┘   state          │ .get_sector_times()   │
                                           └──────────────────────┘

AFTER:
  chainlit_app/app.py                      src/agents/data_agent.py
  ┌─────────────────────┐                  ┌──────────────────────────────┐
  │ No lookup step       │                  │ FastF1Client                  │
  │ (year, circuit_name) │──► directly ──► │ .load_session() — one call    │
  │ passed to pipeline   │   in state      │ then extracts:                │
  └─────────────────────┘                  │  results, laps, pit_stops,    │──► raw_data
                                           │  sectors, tyre_strategies,    │    (enriched)
                                           │  weather, standings           │
                                           └──────────────────────────────┘
```

## Changes by file

### 1. `src/state.py` — remove `race_id`

Remove the `race_id: str` field. FastF1 uses `(season, circuit_name)` directly — both already in state. Every file that references `race_id` in a `RaceReportState` dict will need updating (data_agent, app.py, all tests).

### 2. `src/utils/config.py` — swap `openf1_base_url` for `fastf1_cache_dir`

- Remove `openf1_base_url` field from `Settings`
- Add `fastf1_cache_dir: str` with default `".fastf1_cache"`
- Wire it: `os.getenv("FASTF1_CACHE_DIR", ".fastf1_cache")`

### 3. `src/data/fastf1_client.py` — new file (replaces `openf1_client.py`)

Sync class wrapping FastF1. All methods return plain `list[dict]` (not DataFrames). Session loaded once via `load_session()`, cached on the instance.

```python
import fastf1
from src.utils.config import settings

class FastF1Client:
    def __init__(self):
        fastf1.Cache.enable_cache(settings.fastf1_cache_dir)
        self._session = None

    def load_session(self, season: int, circuit_name: str):
        """Load session once; all other methods read from it."""
        self._session = fastf1.get_session(season, circuit_name, 'R')
        self._session.load()

    def get_race_results(self) -> list[dict]: ...
    def get_lap_times(self) -> list[dict]: ...
    def get_pit_stops(self) -> list[dict]: ...
    def get_sector_times(self) -> list[dict]: ...
    def get_tyre_strategies(self) -> list[dict]: ...
    def get_weather(self) -> list[dict]: ...
    def get_standings(self, season: int) -> dict: ...
```

Key details:
- `get_race_results`: iterate `session.results` rows → `{driver_number, position, full_name, team_name, abbreviation, status, grid_position, points}`
- `get_lap_times`: per driver, find min `LapTime` → `{driver_number, lap_duration, lap_number}`
- `get_pit_stops`: group laps by driver, detect pit stops via `PitInTime` not NaT → `{driver_number, count, total_duration, stops: [{lap_number, pit_duration, compound}]}`
- `get_sector_times`: per driver, min of `Sector1Time`, `Sector2Time`, `Sector3Time` → `{driver_number, s1, s2, s3}`
- `get_tyre_strategies`: group laps by `(Driver, Stint)` → `{driver, stints: [{stint_number, compound, start_lap, end_lap, laps}]}`
- `get_weather`: `session.weather_data` → `{air_temp, track_temp, humidity, rainfall, wind_speed}` (sampled — take mean or a few representative rows)
- `get_standings`: use `fastf1.ergast.Interface()` → `get_driver_standings(season)` and `get_constructor_standings(season)` → `{drivers: [...], constructors: [...]}`

### 4. `src/agents/data_agent.py` — rewrite to use FastF1Client

- Import `FastF1Client` instead of `OpenF1Client`
- FastF1 is sync (session.load() blocks) → wrap in `asyncio.to_thread`
- Single `to_thread` call: load session, then extract everything from cached session
- Remove all `race_id` references; use `state["season"]` and `state["circuit_name"]`
- Add `tyre_strategies`, `weather`, `championship_standings` to `raw_data` dict

```python
async def data_node(state: RaceReportState) -> dict:
    try:
        raw_data = await asyncio.to_thread(_fetch_all, state["season"], state["circuit_name"])
        return {"raw_data": raw_data}
    except Exception as exc:
        return {"error": str(exc), "raw_data": {}}

def _fetch_all(season: int, circuit_name: str) -> dict:
    client = FastF1Client()
    client.load_session(season, circuit_name)
    return {
        "race_results": client.get_race_results(),
        "lap_times": client.get_lap_times(),
        "pit_stops": client.get_pit_stops(),
        "sector_times": client.get_sector_times(),
        "tyre_strategies": client.get_tyre_strategies(),
        "weather": client.get_weather(),
        "championship_standings": client.get_standings(season),
    }
```

### 5. `chainlit_app/app.py` — remove session lookup step

- Remove `_handle_race_request`'s `OpenF1Client().get_race_session()` call and the "Looking up session..." step
- Remove `race_id` from session storage and `initial_state` in `run_pipeline()`
- Set `circuit_name` directly from the LLM-extracted value (no longer overwritten by OpenF1's `circuit_short_name`)
- Update `_STEPS` dict: change "Fetching live race data from OpenF1" → "Fetching race data via FastF1"
- Bad GP names will raise inside `fastf1.get_session()` → caught by data_agent's try/except → propagated via `state["error"]`

### 6. `.env.example` — update env vars

- Remove `OPENF1_BASE_URL` line
- Add `FASTF1_CACHE_DIR=.fastf1_cache`

### 7. `requirements.txt` — update dependencies

- Add `fastf1>=3.0.0`
- Keep `httpx` and `tenacity` (used elsewhere or by fastf1 internally)

### 8. Delete old code

- `src/data/openf1_client.py`
- `tests/data/test_openf1_client.py`

### 9. Tests

**`tests/data/test_fastf1_client.py`** (new) — mock `fastf1.get_session()` to return a mock session object with `.results`, `.laps`, `.weather_data` as DataFrames. Verify each method returns `list[dict]` with expected keys.

**`tests/agents/test_data_agent.py`** — rewrite mocks:
- Patch `src.agents.data_agent.FastF1Client` (or `_fetch_all`)
- Remove `race_id` from `BASE_STATE`
- Verify `raw_data` now includes `tyre_strategies`, `weather`, `championship_standings`

**`tests/agents/test_rag_agent.py`** — remove `race_id` from `BASE_STATE` and `ERROR_STATE`

**`tests/agents/test_analysis_agent.py`** — remove `race_id` from `BASE_STATE` and `ERROR_STATE`

**`tests/agents/test_report_agent.py`** — remove `race_id` from `BASE_STATE` and `ERROR_STATE`

### 10. `CLAUDE.md` — update documentation

- Update architecture section to reference FastF1 instead of OpenF1
- Update environment variables table
- Update file listings

## Implementation order

1. `requirements.txt` — add `fastf1>=3.0.0`
2. `src/utils/config.py` — swap `openf1_base_url` → `fastf1_cache_dir`
3. `src/state.py` — remove `race_id`
4. `src/data/fastf1_client.py` — create new client
5. `src/agents/data_agent.py` — rewrite to use FastF1Client
6. `chainlit_app/app.py` — remove lookup step, remove `race_id`
7. `.env.example` — update vars
8. `tests/data/test_fastf1_client.py` — new tests
9. `tests/agents/test_data_agent.py` — update mocks and state
10. `tests/agents/test_rag_agent.py`, `test_analysis_agent.py`, `test_report_agent.py` — remove `race_id` from BASE_STATE
11. Delete `src/data/openf1_client.py` and `tests/data/test_openf1_client.py`
12. `CLAUDE.md` — update docs

## Verification

1. `pytest tests/` — all unit tests pass (OpenF1 tests deleted, new FastF1 tests in place)
2. `chainlit run chainlit_app/app.py` — request "2024 Australian GP", verify report generates
3. Check `raw_data` in logs includes `tyre_strategies`, `weather`, `championship_standings`
4. Follow-up question "who retired?" → should get accurate DNF info from `status` field
5. Follow-up question "what tyres did Verstappen use?" → should answer from `tyre_strategies`

## Design decision: no separate Jolpica client

The draft plan proposed a standalone `jolpica_client.py`. FastF1 already ships a Jolpica interface via `fastf1.ergast.Interface()` with `get_driver_standings()` and `get_constructor_standings()`. Using it avoids a redundant HTTP client and keeps all F1 data access in one place.