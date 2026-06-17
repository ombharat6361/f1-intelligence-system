"""
Data Agent — fetches race data via the FastF1 library.

Populates state['raw_data'] with:
  - race_results    : final positions, drivers, constructors
  - lap_times       : fastest lap per driver
  - pit_stops       : pit stop counts and durations
  - sector_times    : fastest sector per driver
  - weather         : sampled weather readings throughout the session
"""

import logging
from src.state import RaceReportState
from src.data.fastf1_client import FastF1Client

logger = logging.getLogger(__name__)


async def data_node(state: RaceReportState) -> dict:
    logger.info("data_agent: fetching data for season %d round %d",
                state["season"], state["round_number"])
    client = FastF1Client()
    year = state["season"]
    rnd = state["round_number"]

    try:
        race_results = await client.get_race_results(year, rnd)
        lap_times = await client.get_lap_times(year, rnd)
        pit_stops = await client.get_pit_stops(year, rnd)
        sector_times = await client.get_sector_times(year, rnd)
        weather = await client.get_weather(year, rnd)

        result: dict = {
            "raw_data": {
                "race_results": race_results,
                "lap_times": lap_times,
                "pit_stops": pit_stops,
                "sector_times": sector_times,
                "weather": weather,
            }
        }

        if not state.get("circuit_name"):
            result["circuit_name"] = await client.get_circuit_name(year, rnd)

        return result
    except Exception as exc:
        logger.error("data_agent failed: %s", exc)
        return {"error": str(exc), "raw_data": {}}
