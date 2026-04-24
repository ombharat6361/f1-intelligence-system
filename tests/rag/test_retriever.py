"""
Unit tests for rag/retriever.py.

Chroma and HuggingFaceEmbeddings are mocked so no embedding model is loaded.
"""

from unittest.mock import MagicMock, patch, AsyncMock

import pytest
from langchain_core.retrievers import BaseRetriever

import src.rag.retriever as retriever_module
from src.rag.retriever import get_retriever


@pytest.fixture(autouse=True)
def reset_store_cache():
    """Ensure the module-level _store cache is cleared between tests."""
    retriever_module._store = None
    yield
    retriever_module._store = None


def _make_mock_store() -> MagicMock:
    store = MagicMock()
    store.as_retriever = MagicMock(return_value=MagicMock(spec=BaseRetriever))
    return store


@pytest.fixture
def mock_chroma():
    with patch("src.rag.retriever.Chroma") as MockChroma:
        with patch("src.rag.retriever.HuggingFaceEmbeddings") as MockEmb:
            store = _make_mock_store()
            MockChroma.return_value = store
            MockEmb.return_value = MagicMock()
            yield MockChroma, MockEmb, store


def test_get_retriever_returns_base_retriever(mock_chroma):
    retriever = get_retriever()
    assert retriever is not None


def test_get_retriever_calls_as_retriever_with_top_k(mock_chroma):
    _, _, store = mock_chroma
    get_retriever(top_k=7)
    store.as_retriever.assert_called_once()
    call_kwargs = store.as_retriever.call_args[1]
    assert call_kwargs["search_kwargs"]["k"] == 7


def test_get_retriever_default_top_k_is_5(mock_chroma):
    _, _, store = mock_chroma
    get_retriever()
    call_kwargs = store.as_retriever.call_args[1]
    assert call_kwargs["search_kwargs"]["k"] == 5


def test_get_retriever_uses_similarity_search(mock_chroma):
    _, _, store = mock_chroma
    get_retriever()
    call_kwargs = store.as_retriever.call_args[1]
    assert call_kwargs["search_type"] == "similarity"


def test_store_is_cached_after_first_call(mock_chroma):
    MockChroma, _, _ = mock_chroma
    get_retriever()
    get_retriever()
    # Chroma constructor called only once despite two get_retriever calls
    assert MockChroma.call_count == 1


def test_store_uses_correct_collection_name(mock_chroma):
    MockChroma, _, _ = mock_chroma
    get_retriever()
    call_kwargs = MockChroma.call_args[1]
    assert call_kwargs["collection_name"] == "f1_knowledge"


def test_store_uses_minilm_embedding_model(mock_chroma):
    _, MockEmb, _ = mock_chroma
    get_retriever()
    call_kwargs = MockEmb.call_args[1]
    assert "all-MiniLM-L6-v2" in call_kwargs["model_name"]
