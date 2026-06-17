"""
Chainlit UI for the F1 Intelligence System.

Conversational F1 assistant. Users can ask any F1 question naturally:
  - "How did Gasly perform in the 2024 Monaco GP?"
  - "Tell me about the 2023 Bahrain GP"
  - "Generate a full report for 2024 Australian GP"

The app detects race references, fetches data from FastF1, retrieves
historical context from the knowledge base, and uses Groq to answer.
The LLM can request charts by including [CHART:type] tags in its response.
Asking for a "report" triggers the full LangGraph pipeline.

Start with:
    chainlit run chainlit_app/app.py
"""

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import fastf1
import chainlit as cl
import plotly.graph_objects as go
from langchain_groq import ChatGroq
from src.graph import build_graph
from src.data.fastf1_client import FastF1Client
from src.rag.retriever import get_retriever
from src.utils.config import settings

fastf1.Cache.enable_cache(settings.fastf1_cache_dir)
Path(settings.fastf1_cache_dir).mkdir(parents=True, exist_ok=True)

_graph = build_graph()

_STEPS = {
    "data_agent":     "Fetching race data via FastF1",
    "rag_agent":      "Retrieving historical context from knowledge base",
    "analysis_agent": "Analysing key storylines with LLaMA 3.1 70B",
    "report_agent":   "Generating final race report",
}

_CHART_TYPES = [
    "lap_times",
    "sector_times",
    "race_positions",
    "pit_stops",
    "driver_comparison",
]

_SYSTEM_PROMPT = (
    "You are an expert Formula 1 analyst with deep knowledge of race strategy, "
    "driver performance, and championship dynamics. You answer questions about F1 "
    "based on the race data and historical context provided. Be precise, insightful, "
    "and conversational. If race data is provided, use it to give specific answers "
    "with real numbers and positions. If no data is available for a question, say so "
    "honestly rather than guessing.\n\n"
    "You can include charts in your response when a visual would help the user "
    "understand the data. To include a chart, add one of these tags on its own line:\n"
    "  [CHART:lap_times] — bar chart of fastest lap per driver\n"
    "  [CHART:sector_times] — grouped bar of best S1/S2/S3 per driver\n"
    "  [CHART:race_positions] — horizontal bar of finishing positions\n"
    "  [CHART:pit_stops] — bar chart of pit stop durations per driver\n"
    "  [CHART:driver_comparison:Driver1,Driver2] — head-to-head comparison of two drivers\n"
    "Only include charts when they add value to your answer. Do not include charts "
    "for simple factual questions. Place the tag where the chart should appear in your response."
)


def _detect_race(text: str) -> dict | None:
    year_match = re.search(r"\b(20[12]\d)\b", text)
    if not year_match:
        return None
    year = int(year_match.group(1))
    rest = text.lower()

    round_match = re.search(r"(?:round\s*|r)(\d{1,2})\b", rest)
    if round_match:
        rnd = int(round_match.group(1))
        if 1 <= rnd <= 24:
            return {"season": year, "round_number": rnd, "circuit_name": ""}
        return None

    try:
        schedule = fastf1.get_event_schedule(year)
    except Exception:
        return None

    gp_names = re.findall(
        r"([\w\s\-]+?)\s*(?:grand\s*prix|gp|race)\b", rest, re.IGNORECASE
    )
    search_terms = []
    if gp_names:
        search_terms.extend(t.strip() for t in gp_names)

    known_locations = set()
    for _, event in schedule.iterrows():
        if int(event["RoundNumber"]) == 0:
            continue
        for field in ("EventName", "Country", "Location"):
            val = str(event.get(field, "")).lower()
            if val:
                known_locations.add(val)
                for word in val.split():
                    if len(word) > 3:
                        known_locations.add(word)

    for loc in known_locations:
        if loc in rest:
            search_terms.append(loc)

    best_match = None
    best_score = 0
    for query in search_terms:
        query = query.strip().replace("grand prix", "").replace("gp", "").strip()
        if not query:
            continue
        for _, event in schedule.iterrows():
            rnd = int(event["RoundNumber"])
            if rnd == 0:
                continue
            candidates = [
                str(event.get("EventName", "")).lower(),
                str(event.get("Country", "")).lower(),
                str(event.get("Location", "")).lower(),
            ]
            for candidate in candidates:
                if query in candidate or candidate.replace(" grand prix", "") in query:
                    score = len(query)
                    if query == candidate or query == candidate.replace(" grand prix", ""):
                        score += 100
                    if score > best_score:
                        best_score = score
                        best_match = {
                            "season": year,
                            "round_number": rnd,
                            "circuit_name": str(event.get("EventName", "")),
                        }

    return best_match


def _wants_full_report(text: str) -> bool:
    patterns = [
        r"\b(?:generate|create|make|write|build|produce)\b.*\breport\b",
        r"\bfull\s+report\b",
        r"\brace\s+report\b",
        r"\breport\b.*\b(?:for|on|about)\b",
    ]
    return any(re.search(p, text, re.IGNORECASE) for p in patterns)


async def _fetch_race_data(season: int, round_number: int) -> dict:
    client = FastF1Client()
    try:
        return {
            "race_results": await client.get_race_results(season, round_number),
            "lap_times": await client.get_lap_times(season, round_number),
            "pit_stops": await client.get_pit_stops(season, round_number),
            "sector_times": await client.get_sector_times(season, round_number),
            "weather": await client.get_weather(season, round_number),
        }
    except Exception:
        return {}


async def _fetch_rag_context(query: str) -> str:
    try:
        retriever = get_retriever(top_k=5)
        docs = await retriever.ainvoke(query)
        return "\n\n".join(doc.page_content for doc in docs)
    except Exception:
        return ""


# ---- Chart generation -------------------------------------------------------

def _driver_name_map(data: dict) -> dict[int, str]:
    mapping = {}
    for r in data.get("race_results", []):
        num = r.get("driver_number")
        name = r.get("full_name", f"#{num}")
        if num is not None:
            mapping[num] = name
    return mapping


def _chart_lap_times(data: dict) -> go.Figure | None:
    laps = data.get("lap_times")
    if not laps:
        return None
    names = _driver_name_map(data)
    drivers = [names.get(lt["driver_number"], f"#{lt['driver_number']}") for lt in laps]
    times = [lt["lap_duration"] for lt in laps]
    fig = go.Figure(go.Bar(x=drivers, y=times, marker_color="#e10600"))
    fig.update_layout(
        title="Fastest Lap Per Driver",
        xaxis_title="Driver", yaxis_title="Time (s)",
        template="plotly_dark", height=400,
    )
    return fig


def _chart_sector_times(data: dict) -> go.Figure | None:
    sectors = data.get("sector_times")
    if not sectors:
        return None
    names = _driver_name_map(data)
    drivers = [names.get(s["driver_number"], f"#{s['driver_number']}") for s in sectors]
    fig = go.Figure()
    for key, label, color in [("s1", "Sector 1", "#e10600"), ("s2", "Sector 2", "#1e41ff"), ("s3", "Sector 3", "#00d2be")]:
        vals = [s.get(key) or 0 for s in sectors]
        fig.add_trace(go.Bar(name=label, x=drivers, y=vals, marker_color=color))
    fig.update_layout(
        title="Best Sector Times Per Driver",
        barmode="group", xaxis_title="Driver", yaxis_title="Time (s)",
        template="plotly_dark", height=400,
    )
    return fig


def _chart_race_positions(data: dict) -> go.Figure | None:
    results = data.get("race_results")
    if not results:
        return None
    top = [r for r in results if r.get("position") is not None][:20]
    top.reverse()
    names = [r.get("full_name", "?") for r in top]
    positions = [r["position"] for r in top]
    fig = go.Figure(go.Bar(x=positions, y=names, orientation="h", marker_color="#e10600"))
    fig.update_layout(
        title="Race Finishing Positions",
        xaxis_title="Position", yaxis=dict(autorange="reversed"),
        template="plotly_dark", height=max(300, len(top) * 25),
    )
    return fig


def _chart_pit_stops(data: dict) -> go.Figure | None:
    pits = data.get("pit_stops")
    if not pits:
        return None
    names = _driver_name_map(data)
    drivers = []
    durations = []
    for p in pits:
        for stop in p.get("stops", []):
            d = stop.get("pit_duration")
            if d is not None:
                label = names.get(p["driver_number"], f"#{p['driver_number']}")
                lap = stop.get("lap_number", "?")
                drivers.append(f"{label} (L{lap})")
                durations.append(d)
    if not drivers:
        return None
    fig = go.Figure(go.Bar(x=drivers, y=durations, marker_color="#ff8700"))
    fig.update_layout(
        title="Pit Stop Durations",
        xaxis_title="Driver (Lap)", yaxis_title="Duration (s)",
        template="plotly_dark", height=400,
    )
    return fig


def _chart_driver_comparison(data: dict, driver_names: list[str]) -> go.Figure | None:
    names_map = _driver_name_map(data)
    rev_map = {v.lower(): k for k, v in names_map.items()}
    for r in data.get("race_results", []):
        abbr = r.get("full_name", "")
        last = abbr.split()[-1].lower() if abbr else ""
        if last:
            rev_map[last] = r.get("driver_number")

    nums = []
    for dn in driver_names:
        dn_lower = dn.strip().lower()
        num = rev_map.get(dn_lower)
        if num is None:
            for key, val in rev_map.items():
                if dn_lower in key or key in dn_lower:
                    num = val
                    break
        if num is not None:
            nums.append(num)

    if len(nums) < 2:
        return None

    laps = {lt["driver_number"]: lt for lt in data.get("lap_times", [])}
    sects = {s["driver_number"]: s for s in data.get("sector_times", [])}

    categories = ["Fastest Lap", "Sector 1", "Sector 2", "Sector 3"]
    fig = go.Figure()
    colors = ["#e10600", "#1e41ff", "#00d2be", "#ff8700"]

    for i, num in enumerate(nums[:2]):
        name = names_map.get(num, f"#{num}")
        lap_data = laps.get(num, {})
        sect_data = sects.get(num, {})
        values = [
            lap_data.get("lap_duration", 0),
            sect_data.get("s1", 0) or 0,
            sect_data.get("s2", 0) or 0,
            sect_data.get("s3", 0) or 0,
        ]
        fig.add_trace(go.Bar(name=name, x=categories, y=values, marker_color=colors[i]))

    fig.update_layout(
        title=f"Driver Comparison: {names_map.get(nums[0], '?')} vs {names_map.get(nums[1], '?')}",
        barmode="group", yaxis_title="Time (s)",
        template="plotly_dark", height=400,
    )
    return fig


_CHART_BUILDERS = {
    "lap_times": _chart_lap_times,
    "sector_times": _chart_sector_times,
    "race_positions": _chart_race_positions,
    "pit_stops": _chart_pit_stops,
}


def _extract_and_build_charts(text: str, race_data: dict) -> tuple[str, list[go.Figure]]:
    charts = []
    tag_pattern = re.compile(r"\[CHART:([\w_]+)(?::([^\]]+))?\]")

    def replacer(match):
        chart_type = match.group(1)
        args = match.group(2)

        if chart_type == "driver_comparison" and args:
            driver_names = [d.strip() for d in args.split(",")]
            fig = _chart_driver_comparison(race_data, driver_names)
        else:
            builder = _CHART_BUILDERS.get(chart_type)
            fig = builder(race_data) if builder else None

        if fig:
            charts.append(fig)
        return ""

    cleaned = tag_pattern.sub(replacer, text).strip()
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned, charts


# ---- Chainlit handlers -------------------------------------------------------

@cl.on_chat_start
async def on_chat_start():
    cl.user_session.set("history", [])
    cl.user_session.set("current_race", None)
    cl.user_session.set("current_race_data", None)
    await cl.Message(
        content=(
            "## F1 Intelligence System\n\n"
            "Ask me anything about Formula 1! For example:\n"
            "- *How did Gasly perform in the 2024 Monaco GP?*\n"
            "- *What happened at the 2023 Bahrain race?*\n"
            "- *Compare Verstappen and Norris in the 2024 Australian GP*\n"
            "- *Generate a full report for the 2024 Australian GP*\n"
        )
    ).send()


@cl.on_message
async def on_message(message: cl.Message):
    text = message.content.strip()
    history = cl.user_session.get("history") or []
    current_race = cl.user_session.get("current_race")
    race_data = cl.user_session.get("current_race_data")

    detected = _detect_race(text)
    if detected:
        current_race = detected
        cl.user_session.set("current_race", current_race)
        race_data = None
        cl.user_session.set("current_race_data", None)

    if _wants_full_report(text):
        race = detected or current_race
        if not race:
            await cl.Message(
                content="Which race should I generate a report for? (e.g. *2024 Australian GP*)"
            ).send()
            return
        await _run_report_pipeline(race)
        return

    context_parts = []

    if current_race:
        if race_data is None:
            race_label = current_race.get("circuit_name") or f"Round {current_race['round_number']}"
            async with cl.Step(name=f"Fetching data for {current_race['season']} {race_label}") as step:
                race_data = await _fetch_race_data(current_race["season"], current_race["round_number"])
                if race_data:
                    step.output = "Done"
                    cl.user_session.set("current_race_data", race_data)
                else:
                    step.output = "No data available"
            await step.update()

        if race_data:
            race_label = current_race.get("circuit_name") or f"Round {current_race['round_number']}"
            context_parts.append(
                f"## Race Data ({current_race['season']} {race_label})\n{_format_race_data(race_data)}"
            )

    rag_context = await _fetch_rag_context(text)
    if rag_context:
        context_parts.append(f"## Historical Context\n{rag_context}")

    context_block = "\n\n".join(context_parts) if context_parts else ""

    messages = [{"role": "system", "content": _SYSTEM_PROMPT}]
    for h in history[-10:]:
        messages.append(h)

    user_content = text
    if context_block:
        user_content = f"{text}\n\n---\n\n{context_block}"
    messages.append({"role": "user", "content": user_content})

    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=settings.groq_api_key,
        temperature=0.3,
    )

    response = await llm.ainvoke(messages)
    answer = response.content

    history.append({"role": "user", "content": text})
    history.append({"role": "assistant", "content": answer})
    cl.user_session.set("history", history)

    if race_data:
        cleaned_answer, charts = _extract_and_build_charts(answer, race_data)
    else:
        cleaned_answer, charts = answer, []

    await cl.Message(content=cleaned_answer).send()
    for fig in charts:
        await cl.Message(content="", elements=[cl.Plotly(figure=fig, display="inline")]).send()


def _format_race_data(data: dict) -> str:
    parts = []
    if data.get("race_results"):
        lines = []
        for r in data["race_results"]:
            pos = r.get("position", "?")
            name = r.get("full_name", "?")
            team = r.get("team_name", "?")
            status = r.get("status", "")
            grid = r.get("grid_position", "?")
            pts = r.get("points", 0)
            line = f"P{pos}: {name} ({team}) | Grid: P{grid} | Status: {status}"
            if pts:
                line += f" | Points: {pts}"
            lines.append(line)
        parts.append("**Race Results:**\n" + "\n".join(lines))

    if data.get("lap_times"):
        lines = []
        for lt in data["lap_times"]:
            num = lt.get("driver_number")
            dur = lt.get("lap_duration")
            lap = lt.get("lap_number")
            lines.append(f"#{num}: {dur}s (Lap {lap})")
        parts.append("**Fastest Lap Per Driver:**\n" + "\n".join(lines))

    if data.get("sector_times"):
        lines = []
        for s in data["sector_times"]:
            num = s.get("driver_number")
            s1 = s.get("s1")
            s2 = s.get("s2")
            s3 = s.get("s3")
            lines.append(f"#{num}: S1={s1}s S2={s2}s S3={s3}s")
        parts.append("**Best Sector Times Per Driver:**\n" + "\n".join(lines))

    if data.get("pit_stops"):
        lines = []
        for p in data["pit_stops"]:
            num = p.get("driver_number")
            count = p.get("count")
            total = round(p.get("total_duration", 0), 3)
            lines.append(f"#{num}: {count} stop(s), total {total}s")
        parts.append("**Pit Stops:**\n" + "\n".join(lines))

    if data.get("weather"):
        w = data["weather"][0]
        air = w.get("air_temp")
        track = w.get("track_temp")
        hum = w.get("humidity")
        parts.append(f"**Weather:** Air {air}°C, Track {track}°C, Humidity {hum}%")

    return "\n\n".join(parts) if parts else "No data available."


async def _run_report_pipeline(race: dict):
    season = race["season"]
    round_number = race["round_number"]
    circuit_name = race.get("circuit_name", "")

    label = f"Season {season}, Round {round_number}"
    if circuit_name:
        label = f"**{circuit_name}** ({label})"

    await cl.Message(content=f"Generating full report for {label}…").send()

    initial_state = {
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
                step_label = _STEPS.get(node_name, node_name)
                async with cl.Step(name=step_label, parent_id=pipeline_step.id) as step:
                    if node_output.get("error"):
                        step.output = f"Error: {node_output['error']}"
                    else:
                        step.output = "Done"
                await step.update()
                result.update(node_output)

    if result.get("error"):
        await cl.Message(
            content=f"Pipeline failed: `{result['error']}`\n\nCheck the race and try again."
        ).send()
        return

    report = result.get("report", "")
    s3_url = result.get("s3_report_url")

    if report:
        await cl.Message(content=report).send()
    if s3_url:
        await cl.Message(content=f"Report uploaded: [{s3_url}]({s3_url})").send()
