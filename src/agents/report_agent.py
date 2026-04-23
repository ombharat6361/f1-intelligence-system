"""
Report Agent — generates the final structured race report.

Populates state['report'] with a markdown-formatted document and
optionally uploads it to S3.
"""

import logging
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from src.state import RaceReportState
from src.utils.config import settings
from src.utils.s3_uploader import upload_report

logger = logging.getLogger(__name__)

REPORT_PROMPT = ChatPromptTemplate.from_messages([
    ("system", (
        "You are writing the official post-race intelligence report for a professional "
        "F1 analytics platform. Use clear, engaging prose. Format in Markdown."
    )),
    ("human", (
        "Generate a comprehensive post-race report for the {circuit_name} Grand Prix "
        "(Season {season}, Round {round_number}).\n\n"
        "## Analysis Findings\n{analysis}\n\n"
        "## Raw Race Data\n{raw_data}\n\n"
        "Structure the report with EXACTLY these sections:\n"
        "# {circuit_name} Grand Prix — Race Report\n"
        "## Race Summary\n"
        "## Key Storylines\n"
        "## Driver of the Day\n"
        "## Strategy Breakdown\n"
        "## Championship Standings Update\n"
        "## Looking Ahead\n"
    )),
])


async def report_node(state: RaceReportState) -> dict:
    if state.get("error"):
        logger.warning("report_agent: skipping due to upstream error")
        return {}

    logger.info("report_agent: generating final report")
    llm = ChatGroq(
        model="llama-3.1-70b-versatile",
        api_key=settings.groq_api_key,
        temperature=0.5,
    )
    chain = REPORT_PROMPT | llm

    response = await chain.ainvoke({
        "circuit_name": state["circuit_name"],
        "season": state["season"],
        "round_number": state["round_number"],
        "analysis": state.get("analysis", ""),
        "raw_data": str(state.get("raw_data", {})),
    })

    report = response.content
    s3_url = None

    try:
        filename = f"{state['season']}_R{state['round_number']:02d}_{state['circuit_name'].replace(' ', '_')}.md"
        s3_url = await upload_report(report, filename)
        logger.info("report_agent: uploaded to S3 at %s", s3_url)
    except Exception as exc:
        logger.warning("report_agent: S3 upload failed (non-fatal): %s", exc)

    return {"report": report, "s3_report_url": s3_url}
