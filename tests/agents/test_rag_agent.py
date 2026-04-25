"""
Unit tests for rag_agent.rag_node.

The ChromaDB retriever is mocked so these run offline without an embedding model.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.documents import Document

from src.agents.rag_agent import rag_node
from src.state import RaceReportState

BASE_STATE: RaceReportState = {
    "race_id": "9158",
    "season": 2024,
    "circuit_name": "Australia",
    "raw_data": {"race_results": []},
    "historical_context": "",
    "analysis": "",
    "report": "",
    "error": None,
    "s3_report_url": None,
}

ERROR_STATE: RaceReportState = {**BASE_STATE, "error": "upstream failure"}

CHUNK_A = "Albert Park is a street circuit in Melbourne."
CHUNK_B = "Max Verstappen dominated the 2024 season."
CHUNK_C = "The 2024 Australian GP had drama at Turn 1."


def _make_retriever(*per_query_chunks: list[str]) -> MagicMock:
    """Return a mock retriever whose ainvoke returns successive chunk lists."""
    retriever = MagicMock()
    side_effects = [
        [Document(page_content=c) for c in chunks]
        for chunks in per_query_chunks
    ]
    retriever.ainvoke = AsyncMock(side_effect=side_effects)
    return retriever


@pytest.fixture
def mock_retriever_factory():
    """Patch get_retriever and return a factory so each test controls returned docs."""
    with patch("src.agents.rag_agent.get_retriever") as mock_get:
        yield mock_get


async def test_rag_node_skips_on_upstream_error(mock_retriever_factory):
    result = await rag_node(ERROR_STATE)
    assert result == {}
    mock_retriever_factory.assert_not_called()


async def test_rag_node_returns_historical_context(mock_retriever_factory):
    mock_retriever_factory.return_value = _make_retriever(
        [CHUNK_A], [CHUNK_B], [CHUNK_C]
    )
    result = await rag_node(BASE_STATE)
    assert "historical_context" in result
    assert isinstance(result["historical_context"], str)


async def test_rag_node_joins_chunks_with_separator(mock_retriever_factory):
    mock_retriever_factory.return_value = _make_retriever(
        [CHUNK_A], [CHUNK_B], [CHUNK_C]
    )
    result = await rag_node(BASE_STATE)
    ctx = result["historical_context"]
    assert CHUNK_A in ctx
    assert CHUNK_B in ctx
    assert CHUNK_C in ctx
    assert "---" in ctx


async def test_rag_node_deduplicates_chunks(mock_retriever_factory):
    # CHUNK_A appears in two different query results — should appear only once.
    mock_retriever_factory.return_value = _make_retriever(
        [CHUNK_A, CHUNK_B], [CHUNK_A, CHUNK_C], []
    )
    result = await rag_node(BASE_STATE)
    ctx = result["historical_context"]
    assert ctx.count(CHUNK_A) == 1


async def test_rag_node_runs_three_queries(mock_retriever_factory):
    retriever = _make_retriever([CHUNK_A], [CHUNK_B], [CHUNK_C])
    mock_retriever_factory.return_value = retriever
    await rag_node(BASE_STATE)
    assert retriever.ainvoke.call_count == 3


async def test_rag_node_queries_include_circuit_and_season(mock_retriever_factory):
    retriever = _make_retriever([CHUNK_A], [CHUNK_B], [CHUNK_C])
    mock_retriever_factory.return_value = retriever
    await rag_node(BASE_STATE)

    called_queries = [call.args[0] for call in retriever.ainvoke.call_args_list]
    assert any("Australia" in q for q in called_queries)
    assert any("2024" in q for q in called_queries)


async def test_rag_node_handles_empty_retriever(mock_retriever_factory):
    mock_retriever_factory.return_value = _make_retriever([], [], [])
    result = await rag_node(BASE_STATE)
    assert result["historical_context"] == ""
