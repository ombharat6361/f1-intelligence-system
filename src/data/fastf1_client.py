"""
Client for race data via the FastF1 library.

FastF1 identifies sessions by (year, round_number, session_type). The heavy
``session.load()`` call is synchronous and runs in a thread executor so it
never blocks the event loop. Downloaded data is cached on disk; first load
for a session takes ~10-30 s, subsequent loads are instant.

Every public method returns list[dict] and degrades to [] on failure, matching
the contract downstream agents expect.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any

import fastf1
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class FastF1Client:
    def __init__(self, cache_dir: str | None = None):
        if cache_dir is None:
            from src.utils.config import settings
            cache_dir = settings.fastf1_cache_dir
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        fastf1.Cache.enable_cache(cache_dir)
        self._session: fastf1.Session | None = None
        self._year: int | None = None
        self._round: int | None = None

    async def _ensure_session(self, year: int, round_number: int) -> fastf1.Session:
        if self._session is not None and self._year == year and self._round == round_number:
            return self._session
        session = await asyncio.to_thread(self._load_session, year, round_number)
        self._session = session
        self._year = year
        self._round = round_number
        return session

    def _load_session(self, year: int, round_number: int) -> fastf1.Session:
        session = fastf1.get_session(year, round_number, "R")
        session.load()
        return session

    def _clean(self, val: Any) -> Any:
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return None
        if isinstance(val, (np.integer,)):
            return int(val)
        if isinstance(val, (np.floating,)):
            return float(val)
        if isinstance(val, pd.Timedelta):
            return val.total_seconds()
        return val

    # ---- public methods -----------------------------------------------------

    async def get_race_results(self, year: int, round_number: int) -> list[dict]:
        try:
            session = await self._ensure_session(year, round_number)
            results = session.results
            if results is None or results.empty:
                return []

            out = []
            for _, row in results.iterrows():
                out.append({
                    "driver_number": self._clean(row.get("DriverNumber")),
                    "position": self._clean(row.get("Position")),
                    "full_name": row.get("FullName") or row.get("BroadcastName"),
                    "team_name": row.get("TeamName"),
                    "country_code": row.get("CountryCode"),
                    "status": row.get("Status"),
                    "grid_position": self._clean(row.get("GridPosition")),
                    "points": self._clean(row.get("Points")),
                })
            out.sort(key=lambda r: (r["position"] is None, r["position"] or 999))
            return out
        except Exception as exc:
            logger.error("get_race_results failed: %s", exc)
            return []

    async def get_lap_times(self, year: int, round_number: int) -> list[dict]:
        try:
            session = await self._ensure_session(year, round_number)
            laps = session.laps
            if laps is None or laps.empty:
                return []

            fastest: dict[int, dict] = {}
            for _, lap in laps.iterrows():
                num = self._clean(lap.get("DriverNumber"))
                lt = lap.get("LapTime")
                if num is None or pd.isna(lt):
                    continue
                duration = lt.total_seconds()
                current = fastest.get(num)
                if current is None or duration < current["lap_duration"]:
                    fastest[num] = {
                        "driver_number": num,
                        "lap_duration": round(duration, 3),
                        "lap_number": self._clean(lap.get("LapNumber")),
                    }
            return sorted(fastest.values(), key=lambda r: r["lap_duration"])
        except Exception as exc:
            logger.error("get_lap_times failed: %s", exc)
            return []

    async def get_pit_stops(self, year: int, round_number: int) -> list[dict]:
        try:
            session = await self._ensure_session(year, round_number)
            laps = session.laps
            if laps is None or laps.empty:
                return []

            from collections import defaultdict

            pit_in_times: dict[int, pd.Timedelta] = {}
            agg: dict[int, dict] = defaultdict(
                lambda: {"count": 0, "total_duration": 0.0, "stops": []}
            )

            for _, lap in laps.iterrows():
                num = self._clean(lap.get("DriverNumber"))
                if num is None:
                    continue

                pit_in = lap.get("PitInTime")
                pit_out = lap.get("PitOutTime")

                if not pd.isna(pit_in):
                    pit_in_times[num] = pit_in

                if not pd.isna(pit_out) and num in pit_in_times:
                    pit_duration = round((pit_out - pit_in_times.pop(num)).total_seconds(), 3)
                    entry = agg[num]
                    entry["count"] += 1
                    entry["total_duration"] += pit_duration
                    entry["stops"].append({
                        "lap_number": self._clean(lap.get("LapNumber")),
                        "pit_duration": pit_duration,
                    })

            return [
                {"driver_number": num, **data}
                for num, data in sorted(agg.items())
            ]
        except Exception as exc:
            logger.error("get_pit_stops failed: %s", exc)
            return []

    async def get_sector_times(self, year: int, round_number: int) -> list[dict]:
        try:
            session = await self._ensure_session(year, round_number)
            laps = session.laps
            if laps is None or laps.empty:
                return []

            from collections import defaultdict
            best: dict[int, dict] = defaultdict(lambda: {"s1": None, "s2": None, "s3": None})

            for _, lap in laps.iterrows():
                num = self._clean(lap.get("DriverNumber"))
                if num is None:
                    continue
                for key, field in (("s1", "Sector1Time"), ("s2", "Sector2Time"), ("s3", "Sector3Time")):
                    v = lap.get(field)
                    if pd.isna(v):
                        continue
                    secs = round(v.total_seconds(), 3)
                    current = best[num][key]
                    if current is None or secs < current:
                        best[num][key] = secs

            return [{"driver_number": num, **data} for num, data in sorted(best.items())]
        except Exception as exc:
            logger.error("get_sector_times failed: %s", exc)
            return []

    async def get_weather(self, year: int, round_number: int) -> list[dict]:
        try:
            session = await self._ensure_session(year, round_number)
            weather = session.weather_data
            if weather is None or weather.empty:
                return []

            sampled = weather.iloc[::10]
            out = []
            for _, row in sampled.iterrows():
                out.append({
                    "air_temp": self._clean(row.get("AirTemp")),
                    "track_temp": self._clean(row.get("TrackTemp")),
                    "humidity": self._clean(row.get("Humidity")),
                    "pressure": self._clean(row.get("Pressure")),
                    "wind_speed": self._clean(row.get("WindSpeed")),
                    "wind_direction": self._clean(row.get("WindDirection")),
                    "rainfall": self._clean(row.get("Rainfall")),
                })
            return out
        except Exception as exc:
            logger.error("get_weather failed: %s", exc)
            return []

    async def get_circuit_name(self, year: int, round_number: int) -> str:
        try:
            session = await self._ensure_session(year, round_number)
            return session.event.get("EventName", "") or session.event.get("Location", "")
        except Exception:
            return ""
