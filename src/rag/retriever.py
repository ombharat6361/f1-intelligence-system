"""
RAG retriever backed by a local ChromaDB vector store.

get_retriever(top_k) returns a LangChain retriever that can be called with
retriever.ainvoke(query) and returns a list of Document objects.

The Chroma store must be populated first — run scripts/ingest_docs.py.
If the store is empty or missing, retrieval returns [] rather than crashing,
so the pipeline degrades gracefully (analysis runs with no historical context).
"""

from __future__ import annotations

import logging

from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.retrievers import BaseRetriever

from src.utils.config import settings

logger = logging.getLogger(__name__)

_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
_COLLECTION_NAME = "f1_knowledge"

# Module-level cache so the embedding model is loaded once per process.
_store: Chroma | None = None


def _get_store() -> Chroma:
    global _store
    if _store is None:
        embeddings = HuggingFaceEmbeddings(model_name=_EMBEDDING_MODEL)
        _store = Chroma(
            collection_name=_COLLECTION_NAME,
            embedding_function=embeddings,
            persist_directory=settings.chroma_persist_dir,
        )
    return _store


def get_retriever(top_k: int = 5) -> BaseRetriever:
    """Return a retriever over the local Chroma knowledge base."""
    store = _get_store()
    return store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": top_k},
    )
