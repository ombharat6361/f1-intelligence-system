"""
Integration tests for the LangGraph pipeline (src/graph.py).

Node functions are patched directly on src.graph so no external dependencies
(Groq, OpenF1, Chroma, R2) are needed.
"""

from unittest.mock import AsyncMock, patch

from src.graph import build_graph
from src.state import RaceReportState

SEED: RaceReportState = {
    "race_id": "9158",
    "season": 2024,
    "circuit_name": "Australia",
    "raw_data": {},
    "historical_context": "",
    "analysis": "",
    "report": "",
    "error": None,
    "s3_report_url": None,
}


def test_build_graph_compiles():
    assert build_graph() is not None


async def test_happy_path_populates_report():
    with patch("src.graph.data_node", new=AsyncMock(return_value={"raw_data": {"race_results": []}})), \
         patch("src.graph.rag_node", new=AsyncMock(return_value={"historical_context": "context"})), \
         patch("src.graph.analysis_node", new=AsyncMock(return_value={"analysis": "analysis"})), \
         patch("src.graph.report_node", new=AsyncMock(return_value={"report": "# Report", "s3_report_url": None})):
        result = await build_graph().ainvoke(SEED)

    assert result["report"] == "# Report"
    assert not result.get("error")


async def test_data_node_error_leaves_report_empty():
    with patch("src.graph.data_node", new=AsyncMock(return_value={"error": "API down", "raw_data": {}})), \
         patch("src.graph.rag_node", new=AsyncMock(return_value={})), \
         patch("src.graph.analysis_node", new=AsyncMock(return_value={})), \
         patch("src.graph.report_node", new=AsyncMock(return_value={})):
        result = await build_graph().ainvoke(SEED)

    assert result["error"] == "API down"
    assert result.get("report", "") == ""
