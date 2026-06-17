"""
Unit tests for report_agent.report_node.

ChatGroq and upload_report are mocked so these run without credentials.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.agents.report_agent import report_node
from src.state import RaceReportState

BASE_STATE: RaceReportState = {
    "season": 2024,
    "round_number": 3,
    "circuit_name": "Australia",
    "raw_data": {"race_results": [{"driver_number": 1, "position": 1}]},
    "historical_context": "Albert Park is a street circuit.",
    "analysis": "Storyline 1: Verstappen won. Storyline 2: Safety car.",
    "report": "",
    "error": None,
    "s3_report_url": None,
}

ERROR_STATE: RaceReportState = {**BASE_STATE, "error": "analysis failed"}

MOCK_REPORT = "# Australia Grand Prix — Race Report\n## Race Summary\nVerstappen won."
MOCK_S3_URL = "https://pub.r2.dev/2024_R03_Australia.md"


def _make_chain_mock(content: str) -> MagicMock:
    response = MagicMock()
    response.content = content
    chain = MagicMock()
    chain.ainvoke = AsyncMock(return_value=response)
    return chain


@pytest.fixture
def mock_groq_and_upload():
    with patch("src.agents.report_agent.ChatGroq"):
        with patch("src.agents.report_agent.REPORT_PROMPT") as mock_prompt:
            with patch("src.agents.report_agent.upload_report", new_callable=AsyncMock) as mock_upload:
                chain = _make_chain_mock(MOCK_REPORT)
                mock_prompt.__or__ = MagicMock(return_value=chain)
                mock_upload.return_value = MOCK_S3_URL
                yield chain, mock_upload


async def test_report_node_skips_on_error(mock_groq_and_upload):
    chain, mock_upload = mock_groq_and_upload
    result = await report_node(ERROR_STATE)
    assert result == {}
    chain.ainvoke.assert_not_called()
    mock_upload.assert_not_called()


async def test_report_node_returns_report(mock_groq_and_upload):
    result = await report_node(BASE_STATE)
    assert "report" in result


async def test_report_node_report_is_llm_content(mock_groq_and_upload):
    result = await report_node(BASE_STATE)
    assert result["report"] == MOCK_REPORT


async def test_report_node_returns_s3_url(mock_groq_and_upload):
    result = await report_node(BASE_STATE)
    assert result["s3_report_url"] == MOCK_S3_URL


async def test_report_node_passes_circuit_season_round_to_chain(mock_groq_and_upload):
    chain, _ = mock_groq_and_upload
    await report_node(BASE_STATE)
    call_kwargs = chain.ainvoke.call_args[0][0]
    assert call_kwargs["circuit_name"] == "Australia"
    assert call_kwargs["season"] == 2024
    assert call_kwargs["round_number"] == 3


async def test_report_node_s3_upload_failure_is_nonfatal():
    with patch("src.agents.report_agent.ChatGroq"):
        with patch("src.agents.report_agent.REPORT_PROMPT") as mock_prompt:
            with patch("src.agents.report_agent.upload_report", new_callable=AsyncMock) as mock_upload:
                chain = _make_chain_mock(MOCK_REPORT)
                mock_prompt.__or__ = MagicMock(return_value=chain)
                mock_upload.side_effect = Exception("R2 connection refused")

                result = await report_node(BASE_STATE)

    assert result["report"] == MOCK_REPORT
    assert result["s3_report_url"] is None


async def test_report_node_filename_encodes_season_round_circuit(mock_groq_and_upload):
    _, mock_upload = mock_groq_and_upload
    await report_node(BASE_STATE)
    filename = mock_upload.call_args[0][1]
    assert "2024" in filename
    assert "03" in filename
    assert "Australia" in filename
