"""
Unit tests for FastF1Client.

fastf1.get_session is mocked so these run offline without downloading data.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.data.fastf1_client import FastF1Client


def _make_results_df():
    return pd.DataFrame({
        "DriverNumber": [1, 4, 16],
        "Position": [1.0, 2.0, 3.0],
        "FullName": ["Max Verstappen", "Lando Norris", "Charles Leclerc"],
        "BroadcastName": ["M VERSTAPPEN", "L NORRIS", "C LECLERC"],
        "TeamName": ["Red Bull Racing", "McLaren", "Ferrari"],
        "CountryCode": ["NED", "GBR", "MON"],
    })


def _make_laps_df():
    return pd.DataFrame({
        "DriverNumber": [1, 1, 1, 4, 4, 4],
        "LapTime": [
            pd.Timedelta(seconds=79.5),
            pd.Timedelta(seconds=80.1),
            pd.Timedelta(seconds=82.0),
            pd.Timedelta(seconds=79.8),
            pd.Timedelta(seconds=79.2),
            pd.Timedelta(seconds=81.5),
        ],
        "LapNumber": [10, 20, 21, 10, 20, 21],
        "Sector1Time": [
            pd.Timedelta(seconds=25.1),
            pd.Timedelta(seconds=25.3),
            pd.Timedelta(seconds=26.0),
            pd.Timedelta(seconds=25.5),
            pd.Timedelta(seconds=24.9),
            pd.Timedelta(seconds=25.8),
        ],
        "Sector2Time": [
            pd.Timedelta(seconds=30.2),
            pd.Timedelta(seconds=30.0),
            pd.Timedelta(seconds=30.5),
            pd.Timedelta(seconds=30.1),
            pd.Timedelta(seconds=30.3),
            pd.Timedelta(seconds=30.4),
        ],
        "Sector3Time": [
            pd.Timedelta(seconds=24.2),
            pd.Timedelta(seconds=24.8),
            pd.Timedelta(seconds=25.0),
            pd.Timedelta(seconds=24.2),
            pd.Timedelta(seconds=24.0),
            pd.Timedelta(seconds=24.5),
        ],
        "PitInTime": [
            pd.NaT,
            pd.Timedelta(seconds=3600),
            pd.NaT,
            pd.NaT,
            pd.Timedelta(seconds=3700),
            pd.NaT,
        ],
        "PitOutTime": [
            pd.NaT,
            pd.NaT,
            pd.Timedelta(seconds=3620),
            pd.NaT,
            pd.NaT,
            pd.Timedelta(seconds=3718.5),
        ],
    })


def _make_weather_df():
    return pd.DataFrame({
        "AirTemp": [25.0, 25.5, 26.0],
        "TrackTemp": [40.0, 41.0, 42.0],
        "Humidity": [50.0, 48.0, 47.0],
        "Pressure": [1013.0, 1013.0, 1012.0],
        "WindSpeed": [3.0, 3.5, 4.0],
        "WindDirection": [180, 185, 190],
        "Rainfall": [False, False, False],
    })


def _make_mock_session():
    session = MagicMock()
    session.results = _make_results_df()
    session.laps = _make_laps_df()
    session.weather_data = _make_weather_df()
    session.event = {"EventName": "Australian Grand Prix", "Location": "Melbourne"}
    return session


@pytest.fixture
def client():
    with patch("src.data.fastf1_client.fastf1") as mock_ff1:
        mock_ff1.get_session.return_value = _make_mock_session()
        c = FastF1Client.__new__(FastF1Client)
        c._session = None
        c._year = None
        c._round = None
        mock_ff1.Cache = MagicMock()
        yield c


# --- get_race_results ---

async def test_get_race_results_returns_list(client):
    results = await client.get_race_results(2024, 3)
    assert isinstance(results, list)
    assert len(results) == 3


async def test_get_race_results_fields(client):
    results = await client.get_race_results(2024, 3)
    r = results[0]
    assert "driver_number" in r
    assert "position" in r
    assert "full_name" in r
    assert "team_name" in r


async def test_get_race_results_sorted_by_position(client):
    results = await client.get_race_results(2024, 3)
    positions = [r["position"] for r in results]
    assert positions == sorted(p for p in positions if p is not None)


# --- get_lap_times ---

async def test_get_lap_times_returns_list(client):
    laps = await client.get_lap_times(2024, 3)
    assert isinstance(laps, list)
    assert len(laps) == 2


async def test_get_lap_times_fields(client):
    laps = await client.get_lap_times(2024, 3)
    lap = laps[0]
    assert "driver_number" in lap
    assert "lap_duration" in lap
    assert "lap_number" in lap
    assert isinstance(lap["lap_duration"], float)


async def test_get_lap_times_sorted_fastest_first(client):
    laps = await client.get_lap_times(2024, 3)
    durations = [l["lap_duration"] for l in laps]
    assert durations == sorted(durations)


# --- get_pit_stops ---

async def test_get_pit_stops_returns_list(client):
    pits = await client.get_pit_stops(2024, 3)
    assert isinstance(pits, list)
    assert len(pits) == 2


async def test_get_pit_stops_fields(client):
    pits = await client.get_pit_stops(2024, 3)
    p = pits[0]
    assert "driver_number" in p
    assert "count" in p
    assert "total_duration" in p
    assert "stops" in p
    assert isinstance(p["stops"], list)


async def test_get_pit_stops_count_matches_stops_list(client):
    pits = await client.get_pit_stops(2024, 3)
    for p in pits:
        assert p["count"] == len(p["stops"])


# --- get_sector_times ---

async def test_get_sector_times_returns_list(client):
    sectors = await client.get_sector_times(2024, 3)
    assert isinstance(sectors, list)
    assert len(sectors) == 2


async def test_get_sector_times_fields(client):
    sectors = await client.get_sector_times(2024, 3)
    s = sectors[0]
    assert "driver_number" in s
    assert "s1" in s
    assert "s2" in s
    assert "s3" in s


# --- get_weather ---

async def test_get_weather_returns_list(client):
    weather = await client.get_weather(2024, 3)
    assert isinstance(weather, list)
    assert len(weather) > 0


async def test_get_weather_fields(client):
    weather = await client.get_weather(2024, 3)
    w = weather[0]
    assert "air_temp" in w
    assert "track_temp" in w
    assert "humidity" in w


# --- get_circuit_name ---

async def test_get_circuit_name(client):
    name = await client.get_circuit_name(2024, 3)
    assert name == "Australian Grand Prix"


# --- graceful degradation ---

async def test_empty_results_returns_empty_list(client):
    client._session = MagicMock()
    client._session.results = pd.DataFrame()
    client._year = 2024
    client._round = 3
    results = await client.get_race_results(2024, 3)
    assert results == []
