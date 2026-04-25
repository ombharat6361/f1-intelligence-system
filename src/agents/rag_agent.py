"""
RAG Agent — retrieves historical circuit and driver context from ChromaDB.

Populates state['historical_context'] with a string of relevant passages
that feed into the Analysis Agent's prompt.
"""

import logging
from src.state import RaceReportState
from src.rag.retriever import get_retriever

logger = logging.getLogger(__name__)

# How many chunks to retrieve per query
TOP_K = 5


async def rag_node(state: RaceReportState) -> dict:
    if state.get("error"):
        logger.warning("rag_agent: skipping due to upstream error")
        return {}

    logger.info("rag_agent: querying knowledge base for %s", state["circuit_name"])
    retriever = get_retriever(top_k=TOP_K)

    queries = [
        f"{state['circuit_name']} Grand Prix circuit history and characteristics",
        f"Key drivers competing at {state['circuit_name']} recent seasons",
        f"Championship standings context season {state['season']}",
    ]

    chunks: list[str] = []
    for query in queries:
        logger.debug("rag_agent: query=%s", query)
        docs = await retriever.ainvoke(query)
        for doc in docs:
            logger.debug("rag_agent: chunk (source=%s): %s",
                         doc.metadata.get("source", "unknown"), doc.page_content[:200])
        chunks.extend(doc.page_content for doc in docs)

    # Deduplicate while preserving order
    seen: set[str] = set()
    unique_chunks = [c for c in chunks if not (c in seen or seen.add(c))]

    historical_context = "\n\n---\n\n".join(unique_chunks)
    logger.info("rag_agent: retrieved %d unique chunks", len(unique_chunks))

    return {"historical_context": historical_context}
