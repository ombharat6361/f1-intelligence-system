"""
Unit tests for analysis_agent.analysis_node.

ChatGroq is mocked so these run without a GROQ_API_KEY.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.agents.analysis_agent import analysis_node
from src.state import RaceReportState

BASE_STATE: RaceReportState = {
    "race_id": "9158",
    "season": 2024,
    "round_number": 3,
    "circuit_name": "Australia",
    "raw_data": {"race_results": [{"driver_number": 1, "position": 1}]},
    "historical_context": "Albert Park is a street circuit in Melbourne.",
    "analysis": "",
    "report": "",
    "error": None,
    "s3_report_url": None,
}

ERROR_STATE: RaceReportState = {**BASE_STATE, "error": "data fetch failed"}

MOCK_ANALYSIS = "Storyline 1: Verstappen dominates. Storyline 2: Safety car drama."


def _make_chain_mock(content: str) -> MagicMock:
    response = MagicMock()
    response.content = content
    chain = MagicMock()
    chain.ainvoke = AsyncMock(return_value=response)
    return chain


@pytest.fixture
def mock_groq():
    """Patch ChatGroq and the prompt | llm chain."""
    with patch("src.agents.analysis_agent.ChatGroq") as MockGroq:
        with patch("src.agents.analysis_agent.ANALYSIS_PROMPT") as mock_prompt:
            chain = _make_chain_mock(MOCK_ANALYSIS)
            mock_prompt.__or__ = MagicMock(return_value=chain)
            yield chain


async def test_analysis_node_skips_on_error(mock_groq):
    result = await analysis_node(ERROR_STATE)
    assert result == {}
    mock_groq.ainvoke.assert_not_called()


async def test_analysis_node_returns_analysis(mock_groq):
    result = await analysis_node(BASE_STATE)
    assert "analysis" in result


async def test_analysis_node_analysis_is_llm_content(mock_groq):
    result = await analysis_node(BASE_STATE)
    assert result["analysis"] == MOCK_ANALYSIS


async def test_analysis_node_passes_raw_data_to_chain(mock_groq):
    await analysis_node(BASE_STATE)
    call_kwargs = mock_groq.ainvoke.call_args[0][0]
    assert "raw_data" in call_kwargs
    assert str(BASE_STATE["raw_data"]) in call_kwargs["raw_data"]


async def test_analysis_node_passes_historical_context_to_chain(mock_groq):
    await analysis_node(BASE_STATE)
    call_kwargs = mock_groq.ainvoke.call_args[0][0]
    assert "historical_context" in call_kwargs
    assert BASE_STATE["historical_context"] in call_kwargs["historical_context"]


async def test_analysis_node_uses_fallback_when_no_context():
    state = {**BASE_STATE, "historical_context": ""}
    with patch("src.agents.analysis_agent.ChatGroq"):
        with patch("src.agents.analysis_agent.ANALYSIS_PROMPT") as mock_prompt:
            chain = _make_chain_mock(MOCK_ANALYSIS)
            mock_prompt.__or__ = MagicMock(return_value=chain)
            await analysis_node(state)
            call_kwargs = chain.ainvoke.call_args[0][0]
            # Empty string falls through to the .get() default
            assert "historical_context" in call_kwargs
