"""
Ingest documents from knowledge_base/raw_docs/ into the ChromaDB vector store.

Supported formats: .txt, .md, .pdf

Usage:
    python scripts/ingest_docs.py
    python scripts/ingest_docs.py --docs-dir path/to/docs --reset
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Allow running from repo root without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langchain_community.document_loaders import (
    DirectoryLoader,
    TextLoader,
    PyPDFLoader,
)
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.utils.config import settings

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
_COLLECTION_NAME = "f1_knowledge"
_CHUNK_SIZE = 800
_CHUNK_OVERLAP = 100


def load_documents(docs_dir: Path):
    docs = []

    for pattern, loader_cls, kwargs in [
        ("**/*.txt", TextLoader, {"encoding": "utf-8"}),
        ("**/*.md",  TextLoader, {"encoding": "utf-8"}),
        ("**/*.pdf", PyPDFLoader, {}),
    ]:
        loader = DirectoryLoader(
            str(docs_dir),
            glob=pattern,
            loader_cls=loader_cls,
            loader_kwargs=kwargs,
            silent_errors=True,
        )
        loaded = loader.load()
        docs.extend(loaded)
        logger.info("Loaded %d docs matching %s", len(loaded), pattern)

    return docs


def ingest(docs_dir: Path, chroma_dir: str, reset: bool) -> None:
    if not docs_dir.exists():
        logger.error("Docs directory not found: %s", docs_dir)
        sys.exit(1)

    raw_docs = load_documents(docs_dir)
    if not raw_docs:
        logger.warning("No documents found in %s — nothing to ingest.", docs_dir)
        return

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=_CHUNK_SIZE,
        chunk_overlap=_CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(raw_docs)
    logger.info("Split into %d chunks", len(chunks))

    embeddings = HuggingFaceEmbeddings(model_name=_EMBEDDING_MODEL)

    if reset:
        logger.info("--reset: clearing existing collection")
        store = Chroma(
            collection_name=_COLLECTION_NAME,
            embedding_function=embeddings,
            persist_directory=chroma_dir,
        )
        store.delete_collection()

    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=_COLLECTION_NAME,
        persist_directory=chroma_dir,
    )
    logger.info("Ingested %d chunks into %s", len(chunks), chroma_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest F1 docs into ChromaDB")
    parser.add_argument(
        "--docs-dir",
        type=Path,
        default=Path("knowledge_base/raw_docs"),
        help="Directory of source documents (default: knowledge_base/raw_docs)",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Clear the existing collection before ingesting",
    )
    args = parser.parse_args()

    ingest(
        docs_dir=args.docs_dir,
        chroma_dir=settings.chroma_persist_dir,
        reset=args.reset,
    )
