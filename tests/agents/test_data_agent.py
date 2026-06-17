"""
Unit tests for data_agent.data_node.

FastF1Client is mocked so these run offline without network access.
"""

from unittest.mock import AsyncMock, patch

import pytest

from src.agents.data_agent import data_node
from src.state import RaceReportState

BASE_STATE: RaceReportState = {
    "season": 2024,
    "round_number": 3,
    "circuit_name": "Australia",
    "raw_data": {},
    "historical_context": "",
    "analysis": "",
    "report": "",
    "error": None,
    "s3_report_url": None,
}

MOCK_RESULTS = [{"driver_number": 1, "position": 1, "full_name": "Max Verstappen", "team_name": "Red Bull"}]
MOCK_LAPS = [{"driver_number": 1, "lap_duration": 79.5, "lap_number": 15}]
MOCK_PITS = [{"driver_number": 1, "count": 1, "total_duration": 2.4, "stops": [{"lap": 20}]}]
MOCK_SECTORS = [{"driver_number": 1, "s1": 25.1, "s2": 30.2, "s3": 24.2}]
MOCK_WEATHER = [{"air_temp": 25.0, "track_temp": 40.0}]


@pytest.fixture
def mock_client():
    with patch("src.agents.data_agent.FastF1Client") as MockClass:
        instance = MockClass.return_value
        instance.get_race_results = AsyncMock(return_value=MOCK_RESULTS)
        instance.get_lap_times = AsyncMock(return_value=MOCK_LAPS)
        instance.get_pit_stops = AsyncMock(return_value=MOCK_PITS)
        instance.get_sector_times = AsyncMock(return_value=MOCK_SECTORS)
        instance.get_weather = AsyncMock(return_value=MOCK_WEATHER)
        instance.get_circuit_name = AsyncMock(return_value="Australian Grand Prix")
        yield instance


async def test_data_node_returns_raw_data_dict(mock_client):
    result = await data_node(BASE_STATE)
    assert "raw_data" in result
    assert isinstance(result["raw_data"], dict)


async def test_data_node_raw_data_has_all_keys(mock_client):
    result = await data_node(BASE_STATE)
    raw = result["raw_data"]
    assert "race_results" in raw
    assert "lap_times" in raw
    assert "pit_stops" in raw
    assert "sector_times" in raw
    assert "weather" in raw


async def test_data_node_raw_data_values(mock_client):
    result = await data_node(BASE_STATE)
    raw = result["raw_data"]
    assert raw["race_results"] == MOCK_RESULTS
    assert raw["lap_times"] == MOCK_LAPS
    assert raw["pit_stops"] == MOCK_PITS
    assert raw["sector_times"] == MOCK_SECTORS
    assert raw["weather"] == MOCK_WEATHER


async def test_data_node_no_error_on_success(mock_client):
    result = await data_node(BASE_STATE)
    assert "error" not in result


async def test_data_node_sets_error_on_client_exception():
    with patch("src.agents.data_agent.FastF1Client") as MockClass:
        instance = MockClass.return_value
        instance.get_race_results = AsyncMock(side_effect=RuntimeError("network failure"))
        result = await data_node(BASE_STATE)

    assert "error" in result
    assert "network failure" in result["error"]
    assert result["raw_data"] == {}


async def test_data_node_calls_all_five_client_methods(mock_client):
    await data_node(BASE_STATE)
    year, rnd = BASE_STATE["season"], BASE_STATE["round_number"]
    mock_client.get_race_results.assert_called_once_with(year, rnd)
    mock_client.get_lap_times.assert_called_once_with(year, rnd)
    mock_client.get_pit_stops.assert_called_once_with(year, rnd)
    mock_client.get_sector_times.assert_called_once_with(year, rnd)
    mock_client.get_weather.assert_called_once_with(year, rnd)


async def test_data_node_auto_populates_circuit_name_when_empty(mock_client):
    state = {**BASE_STATE, "circuit_name": ""}
    result = await data_node(state)
    assert result.get("circuit_name") == "Australian Grand Prix"


async def test_data_node_does_not_override_provided_circuit_name(mock_client):
    result = await data_node(BASE_STATE)
    assert "circuit_name" not in result
