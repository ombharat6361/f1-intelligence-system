"""
Async client for the OpenF1 API (https://openf1.org).

OpenF1 returns inconsistent and sometimes empty payloads, especially for older
sessions and sprints. Every public method degrades gracefully: missing fields
become None, missing endpoints return []. Callers should assume any field can
be absent.

All session-scoped queries use `session_key` — what the rest of the codebase
calls `race_id`.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.utils.config import settings

logger = logging.getLogger(__name__)

_RETRYABLE = (httpx.TransportError, httpx.HTTPStatusError)


class OpenF1Client:
    def __init__(self, base_url: str | None = None, timeout: float = 20.0):
        self._base_url = (base_url or settings.openf1_base_url).rstrip("/")
        self._timeout = timeout

    # ---- HTTP ----------------------------------------------------------------

    @retry(
        reraise=True,
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type(_RETRYABLE),
    )
    async def _get(self, path: str, **params: Any) -> list[dict]:
        url = f"{self._base_url}/{path.lstrip('/')}"
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.get(url, params=params)
            # Retry on 5xx only; 4xx is the caller's problem.
            if 500 <= resp.status_code < 600:
                resp.raise_for_status()
            if resp.status_code >= 400:
                logger.warning("OpenF1 %s returned %s: %s", path, resp.status_code, resp.text[:200])
                return []
            data = resp.json()
            return data if isinstance(data, list) else []

    # ---- Public methods ------------------------------------------------------

    async def get_race_results(self, session_key: str) -> list[dict]:
        """
        Final classification: one row per driver with position, team, driver info.

        Built by joining /drivers with the last known /position record per driver.
        """
        drivers, positions = await _gather(
            self._get("drivers", session_key=session_key),
            self._get("position", session_key=session_key),
        )

        # Latest position per driver_number.
        latest: dict[int, dict] = {}
        for row in positions:
            num = row.get("driver_number")
            date = row.get("date")
            if num is None or date is None:
                continue
            prev = latest.get(num)
            if prev is None or date > prev.get("date", ""):
                latest[num] = row

        results = []
        for d in drivers:
            num = d.get("driver_number")
            pos = latest.get(num, {}).get("position") if num is not None else None
            results.append({
                "driver_number": num,
                "position": pos,
                "full_name": d.get("full_name") or d.get("broadcast_name"),
                "team_name": d.get("team_name"),
                "country_code": d.get("country_code"),
            })

        results.sort(key=lambda r: (r["position"] is None, r["position"] or 999))
        return results

    async def get_lap_times(self, session_key: str) -> list[dict]:
        """Fastest lap per driver."""
        laps = await self._get("laps", session_key=session_key)

        fastest: dict[int, dict] = {}
        for lap in laps:
            num = lap.get("driver_number")
            duration = lap.get("lap_duration")
            if num is None or duration is None:
                continue
            current = fastest.get(num)
            if current is None or duration < current["lap_duration"]:
                fastest[num] = {
                    "driver_number": num,
                    "lap_duration": duration,
                    "lap_number": lap.get("lap_number"),
                }

        return sorted(fastest.values(), key=lambda r: r["lap_duration"])

    async def get_pit_stops(self, session_key: str) -> list[dict]:
        """Pit stop count and total duration per driver."""
        stops = await self._get("pit", session_key=session_key)

        agg: dict[int, dict] = defaultdict(lambda: {"count": 0, "total_duration": 0.0, "stops": []})
        for s in stops:
            num = s.get("driver_number")
            if num is None:
                continue
            entry = agg[num]
            entry["count"] += 1
            duration = s.get("pit_duration")
            if duration is not None:
                entry["total_duration"] += duration
            entry["stops"].append({
                "lap_number": s.get("lap_number"),
                "pit_duration": duration,
            })

        return [
            {"driver_number": num, **data}
            for num, data in sorted(agg.items())
        ]

    async def get_sector_times(self, session_key: str) -> list[dict]:
        """Fastest sector 1/2/3 per driver."""
        laps = await self._get("laps", session_key=session_key)

        best: dict[int, dict] = defaultdict(lambda: {"s1": None, "s2": None, "s3": None})
        for lap in laps:
            num = lap.get("driver_number")
            if num is None:
                continue
            for key, field in (("s1", "duration_sector_1"), ("s2", "duration_sector_2"), ("s3", "duration_sector_3")):
                v = lap.get(field)
                if v is None:
                    continue
                current = best[num][key]
                if current is None or v < current:
                    best[num][key] = v

        return [{"driver_number": num, **data} for num, data in sorted(best.items())]


# ---- helpers ----------------------------------------------------------------

async def _gather(*coros):
    import asyncio
    return await asyncio.gather(*coros)
