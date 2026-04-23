"""
Analysis Agent — uses Groq/LLaMA to identify key storylines from the race.

Populates state['analysis'] with structured findings that the Report Agent
will turn into the final narrative.
"""

import logging
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from src.state import RaceReportState
from src.utils.config import settings

logger = logging.getLogger(__name__)

ANALYSIS_PROMPT = ChatPromptTemplate.from_messages([
    ("system", (
        "You are an expert Formula 1 analyst with deep knowledge of race strategy, "
        "driver performance, and championship dynamics. Be precise and insightful."
    )),
    ("human", (
        "Analyze this Formula 1 race and identify the 4-6 most important storylines.\n\n"
        "## Race Data\n{raw_data}\n\n"
        "## Historical Context\n{historical_context}\n\n"
        "For each storyline cover:\n"
        "- What happened\n"
        "- Why it matters (strategy, championship, driver career)\n"
        "- Any surprising or counter-intuitive elements\n\n"
        "Also identify:\n"
        "1. Driver of the Day (with justification)\n"
        "2. Key strategic decision that defined the race\n"
        "3. Championship implications\n"
    )),
])


async def analysis_node(state: RaceReportState) -> dict:
    if state.get("error"):
        logger.warning("analysis_agent: skipping due to upstream error")
        return {}

    logger.info("analysis_agent: generating analysis")
    llm = ChatGroq(
        model="llama-3.1-70b-versatile",
        api_key=settings.groq_api_key,
        temperature=0.3,
    )
    chain = ANALYSIS_PROMPT | llm

    response = await chain.ainvoke({
        "raw_data": str(state["raw_data"]),
        "historical_context": state.get("historical_context", "No historical context available."),
    })

    return {"analysis": response.content}
