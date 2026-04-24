"""
Integration tests for OpenF1Client against the live API.

Uses session 9158 (2024 Australian GP) — a completed race with full data.
Run with: pytest tests/data/test_openf1_client.py -v
"""

import pytest
from src.data.openf1_client import OpenF1Client

SESSION_KEY = "9158"  # 2024 Australian GP
BASE_URL = "https://api.openf1.org/v1"


@pytest.fixture
def client():
    return OpenF1Client(base_url=BASE_URL)


@pytest.mark.asyncio
async def test_get_race_results_returns_list(client):
    results = await client.get_race_results(SESSION_KEY)
    assert isinstance(results, list)
    assert len(results) > 0


@pytest.mark.asyncio
async def test_get_race_results_fields(client):
    results = await client.get_race_results(SESSION_KEY)
    first = results[0]
    assert "driver_number" in first
    assert "position" in first
    assert "full_name" in first
    assert "team_name" in first


@pytest.mark.asyncio
async def test_get_race_results_sorted_by_position(client):
    results = await client.get_race_results(SESSION_KEY)
    positions = [r["position"] for r in results if r["position"] is not None]
    assert positions == sorted(positions)


@pytest.mark.asyncio
async def test_get_lap_times_returns_list(client):
    lap_times = await client.get_lap_times(SESSION_KEY)
    assert isinstance(lap_times, list)
    assert len(lap_times) > 0


@pytest.mark.asyncio
async def test_get_lap_times_fields(client):
    lap_times = await client.get_lap_times(SESSION_KEY)
    first = lap_times[0]
    assert "driver_number" in first
    assert "lap_duration" in first
    assert "lap_number" in first


@pytest.mark.asyncio
async def test_get_lap_times_sorted_fastest_first(client):
    lap_times = await client.get_lap_times(SESSION_KEY)
    durations = [r["lap_duration"] for r in lap_times]
    assert durations == sorted(durations)


@pytest.mark.asyncio
async def test_get_pit_stops_returns_list(client):
    pit_stops = await client.get_pit_stops(SESSION_KEY)
    assert isinstance(pit_stops, list)
    assert len(pit_stops) > 0


@pytest.mark.asyncio
async def test_get_pit_stops_fields(client):
    pit_stops = await client.get_pit_stops(SESSION_KEY)
    first = pit_stops[0]
    assert "driver_number" in first
    assert "count" in first
    assert "total_duration" in first
    assert "stops" in first
    assert isinstance(first["stops"], list)


@pytest.mark.asyncio
async def test_get_pit_stops_count_matches_stops_list(client):
    pit_stops = await client.get_pit_stops(SESSION_KEY)
    for driver in pit_stops:
        assert driver["count"] == len(driver["stops"])


@pytest.mark.asyncio
async def test_get_sector_times_returns_list(client):
    sector_times = await client.get_sector_times(SESSION_KEY)
    assert isinstance(sector_times, list)
    assert len(sector_times) > 0


@pytest.mark.asyncio
async def test_get_sector_times_fields(client):
    sector_times = await client.get_sector_times(SESSION_KEY)
    first = sector_times[0]
    assert "driver_number" in first
    assert "s1" in first
    assert "s2" in first
    assert "s3" in first


@pytest.mark.asyncio
async def test_invalid_session_key_returns_empty(client):
    results = await client.get_race_results("0000000")
    assert isinstance(results, list)
    assert results == []
