"""
Data Agent — fetches live race data from the OpenF1 API.

Populates state['raw_data'] with:
  - race_results    : final positions, drivers, constructors
  - lap_times       : fastest lap per driver
  - pit_stops       : pit stop counts and durations
  - sector_times    : fastest sector per driver
"""

import logging
from src.state import RaceReportState
from src.data.openf1_client import OpenF1Client

logger = logging.getLogger(__name__)


async def data_node(state: RaceReportState) -> dict:
    logger.info("data_agent: fetching data for session %s", state["race_id"])
    client = OpenF1Client()

    try:
        race_results = await client.get_race_results(state["race_id"])
        lap_times = await client.get_lap_times(state["race_id"])
        pit_stops = await client.get_pit_stops(state["race_id"])
        sector_times = await client.get_sector_times(state["race_id"])

        return {
            "raw_data": {
                "race_results": race_results,
                "lap_times": lap_times,
                "pit_stops": pit_stops,
                "sector_times": sector_times,
            }
        }
    except Exception as exc:
        logger.error("data_agent failed: %s", exc)
        return {"error": str(exc), "raw_data": {}}
