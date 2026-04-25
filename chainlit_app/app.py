"""
Chainlit UI for the F1 Intelligence System.

Conversational interface: users ask for a race in natural language, get a
structured report, then ask follow-up questions about the data.

Start with:
    chainlit run chainlit_app/app.py
"""

import json
import logging
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import chainlit as cl
from langchain_groq import ChatGroq
from src.graph import build_graph
from src.utils.config import settings

logging.basicConfig(
    level=settings.log_level.upper(),
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler("f1_intelligence.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)

_graph = build_graph()

_STEPS = {
    "data_agent":     "Fetching live race data from OpenF1",
    "rag_agent":      "Retrieving historical context from knowledge base",
    "analysis_agent": "Analysing key storylines with LLaMA 3.1 70B",
    "report_agent":   "Generating final race report",
}


async def _llm_json(prompt: str) -> dict | None:
    llm = ChatGroq(model=settings.model_name, api_key=settings.groq_api_key, temperature=0)
    response = await llm.ainvoke(prompt)
    try:
        return json.loads(response.content)
    except Exception:
        return None


async def _extract_race(text: str) -> dict | None:
    current_year = datetime.now().year
    return await _llm_json(
        f"Extract the F1 race year and circuit name from this message.\n"
        f"Current year: {current_year}. The ongoing F1 season is {current_year}.\n"
        f'Respond with ONLY valid JSON: {{"year": <int>, "circuit": "<string>"}}\n'
        f'If you cannot determine both, respond with: {{"error": "unclear"}}\n\n'
        f"Message: {text}"
    )


async def _classify_intent(text: str) -> dict | None:
    current_year = datetime.now().year
    return await _llm_json(
        f"You are an intent classifier for an F1 analysis chatbot. "
        f"The user has just received a race report and is now sending a follow-up message.\n\n"
        f"Determine the user's intent:\n"
        f"1. Asking a question about the current race (follow-up)\n"
        f"2. Requesting a report for a different race (new race)\n\n"
        f"Current year: {current_year}. The ongoing F1 season is {current_year}.\n\n"
        f'If follow-up, respond: {{"intent": "follow_up"}}\n'
        f'If new race, respond: {{"intent": "new_race", "year": <int>, "circuit": "<string>"}}\n\n'
        f"Message: {text}"
    )


async def _answer_follow_up(question: str) -> str:
    raw_data = cl.user_session.get("raw_data")
    report = cl.user_session.get("report")
    circuit_name = cl.user_session.get("circuit_name")
    season = cl.user_session.get("season")

    llm = ChatGroq(model=settings.model_name, api_key=settings.groq_api_key, temperature=0.3)
    prompt = (
        f"You are an expert F1 analyst. The user is asking about the "
        f"{circuit_name} Grand Prix {season}.\n\n"
        f"## Race Data\n{json.dumps(raw_data, indent=2)}\n\n"
        f"## Generated Report\n{report}\n\n"
        f"Answer the user's question concisely using the data above. "
        f"If the data doesn't contain the answer, say so.\n\n"
        f"Question: {question}"
    )
    response = await llm.ainvoke(prompt)
    return response.content


async def _handle_race_request(year: int, circuit: str):
    """Look up session and run the pipeline for a race."""
    async with cl.Step(name="Looking up session…", show_input=False):
        from src.data.openf1_client import OpenF1Client
        session = await OpenF1Client().get_race_session(year, circuit)

    if session is None:
        await cl.Message(
            content=f"Couldn't find a race matching '{circuit}' in {year}. Try a different name."
        ).send()
        return

    cl.user_session.set("race_id", session["session_key"])
    cl.user_session.set("season", year)
    cl.user_session.set("circuit_name", session["circuit_name"])
    await cl.Message(
        content=f"Found: **{session['circuit_name']} Grand Prix {year}**. Generating report…"
    ).send()
    await run_pipeline()


@cl.on_chat_start
async def on_chat_start():
    await cl.Message(
        content=(
            "## F1 Intelligence System\n\n"
            "I generate post-race intelligence reports and answer questions about any Formula 1 race.\n\n"
            "Which race would you like? (e.g. `2024 Australian GP`, `last year's Monaco race`)"
        )
    ).send()
    cl.user_session.set("awaiting", "race_query")


@cl.on_message
async def on_message(message: cl.Message):
    awaiting = cl.user_session.get("awaiting")
    text = message.content.strip()

    if awaiting == "race_query":
        async with cl.Step(name="Understanding your request…", show_input=False):
            extracted = await _extract_race(text)

        if not extracted or "error" in extracted:
            await cl.Message(
                content="I couldn't work out which race you meant. Try something like `2024 Australian GP` or `Monaco 2023`."
            ).send()
            return

        await _handle_race_request(extracted["year"], extracted["circuit"])
        return

    if awaiting == "follow_up":
        async with cl.Step(name="Understanding your request…", show_input=False):
            intent = await _classify_intent(text)

        if intent and intent.get("intent") == "new_race" and "year" in intent and "circuit" in intent:
            await _handle_race_request(intent["year"], intent["circuit"])
            return

        async with cl.Step(name="Answering…", show_input=False):
            answer = await _answer_follow_up(text)
        await cl.Message(content=answer).send()
        return

    cl.user_session.set("awaiting", "race_query")
    await cl.Message(
        content="Which race would you like? (e.g. `2024 Australian GP`, `last year's Monaco race`)"
    ).send()


async def run_pipeline():
    race_id      = cl.user_session.get("race_id")
    season       = cl.user_session.get("season")
    circuit_name = cl.user_session.get("circuit_name")

    initial_state = {
        "race_id":      race_id,
        "season":       season,
        "circuit_name": circuit_name,
        "raw_data":     {},
        "historical_context": "",
        "analysis":     "",
        "report":       "",
        "error":        None,
        "s3_report_url": None,
    }

    result = {**initial_state}

    async with cl.Step(name="Pipeline", show_input=False) as pipeline_step:
        async for chunk in _graph.astream(initial_state, stream_mode="updates"):
            for node_name, node_output in chunk.items():
                label = _STEPS.get(node_name, node_name)
                async with cl.Step(name=label, parent_id=pipeline_step.id) as step:
                    if node_output.get("error"):
                        step.output = f"Error: {node_output['error']}"
                    else:
                        step.output = "Done"
                await step.update()
                result.update(node_output)

    if result.get("error"):
        await cl.Message(
            content=f"Pipeline failed: `{result['error']}`\n\nTry a different race."
        ).send()
        cl.user_session.set("awaiting", "race_query")
        return

    report = result.get("report", "")
    s3_url = result.get("s3_report_url")

    cl.user_session.set("raw_data", result.get("raw_data", {}))
    cl.user_session.set("report", report)

    if report:
        await cl.Message(content=report).send()

    if s3_url:
        await cl.Message(content=f"Report uploaded: [{s3_url}]({s3_url})").send()

    await cl.Message(content="Ask me anything about this race, or request a different one.").send()
    cl.user_session.set("awaiting", "follow_up")
