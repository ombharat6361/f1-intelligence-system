"""
Chainlit UI for the F1 Intelligence System.

Users provide a session key (OpenF1 race_id) plus optional metadata.
The LangGraph pipeline runs and streams step-by-step progress, then
renders the final markdown report in the chat.

Start with:
    chainlit run chainlit_app/app.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import chainlit as cl
from src.graph import build_graph

# Build once at startup — embedding model load is expensive.
_graph = build_graph()

_STEPS = {
    "data_agent":     "Fetching live race data from OpenF1",
    "rag_agent":      "Retrieving historical context from knowledge base",
    "analysis_agent": "Analysing key storylines with LLaMA 3.1 70B",
    "report_agent":   "Generating final race report",
}


@cl.on_chat_start
async def on_chat_start():
    await cl.Message(
        content=(
            "## F1 Intelligence System\n\n"
            "I generate a post-race intelligence report for any Formula 1 session "
            "in the [OpenF1](https://openf1.org) database.\n\n"
            "**To get started, provide the OpenF1 session key.**\n"
            "You can find it at `https://api.openf1.org/v1/sessions` — "
            "look for the `session_key` field of the race you want.\n\n"
            "Example: `9158` → 2024 Australian Grand Prix"
        )
    ).send()

    await cl.Message(
        content="Please enter the **OpenF1 session key** (e.g. `9158`):"
    ).send()
    cl.user_session.set("awaiting", "session_key")


@cl.on_message
async def on_message(message: cl.Message):
    awaiting = cl.user_session.get("awaiting")
    text = message.content.strip()

    # --- collect inputs one at a time ---

    if awaiting == "session_key":
        if not text.isdigit():
            await cl.Message(content="That doesn't look like a valid session key. Please enter a number.").send()
            return
        cl.user_session.set("race_id", text)
        cl.user_session.set("awaiting", "season")
        await cl.Message(content="Got it. What **season** (year) is this race from? (e.g. `2024`)").send()
        return

    if awaiting == "season":
        if not text.isdigit() or not (2018 <= int(text) <= 2030):
            await cl.Message(content="Please enter a valid season year between 2018 and 2030.").send()
            return
        cl.user_session.set("season", int(text))
        cl.user_session.set("awaiting", "round_number")
        await cl.Message(content="What is the **round number** in that season? (e.g. `3`)").send()
        return

    if awaiting == "round_number":
        if not text.isdigit() or not (1 <= int(text) <= 24):
            await cl.Message(content="Please enter a round number between 1 and 24.").send()
            return
        cl.user_session.set("round_number", int(text))
        cl.user_session.set("awaiting", "circuit_name")
        await cl.Message(content="Finally, what is the **circuit name**? (e.g. `Melbourne`, `Monaco`)").send()
        return

    if awaiting == "circuit_name":
        cl.user_session.set("circuit_name", text)
        cl.user_session.set("awaiting", None)
        await run_pipeline()
        return

    # If no active flow, offer to run another report.
    await cl.Message(
        content="Enter a new **OpenF1 session key** to generate another report:"
    ).send()
    cl.user_session.set("awaiting", "session_key")


async def run_pipeline():
    race_id      = cl.user_session.get("race_id")
    season       = cl.user_session.get("season")
    round_number = cl.user_session.get("round_number")
    circuit_name = cl.user_session.get("circuit_name")

    await cl.Message(
        content=(
            f"Running pipeline for **{circuit_name} Grand Prix** "
            f"(Season {season}, Round {round_number}, Session `{race_id}`)…"
        )
    ).send()

    initial_state = {
        "race_id":      race_id,
        "season":       season,
        "round_number": round_number,
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
            content=f"Pipeline failed: `{result['error']}`\n\nCheck the session key and try again."
        ).send()
        cl.user_session.set("awaiting", "session_key")
        await cl.Message(content="Enter a new **OpenF1 session key** to try again:").send()
        return

    report = result.get("report", "")
    s3_url = result.get("s3_report_url")

    if report:
        await cl.Message(content=report).send()

    if s3_url:
        await cl.Message(content=f"Report uploaded: [{s3_url}]({s3_url})").send()

    await cl.Message(
        content="Report complete. Enter another **OpenF1 session key** to generate a new report:"
    ).send()
    cl.user_session.set("awaiting", "session_key")
