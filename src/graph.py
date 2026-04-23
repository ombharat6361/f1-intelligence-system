"""
LangGraph state machine wiring all 4 agents together.

Flow: data_node → rag_node → analysis_node → report_node
"""

from langgraph.graph import StateGraph, END

from src.state import RaceReportState
from src.agents.data_agent import data_node
from src.agents.rag_agent import rag_node
from src.agents.analysis_agent import analysis_node
from src.agents.report_agent import report_node


def build_graph() -> StateGraph:
    graph = StateGraph(RaceReportState)

    graph.add_node("data_agent", data_node)
    graph.add_node("rag_agent", rag_node)
    graph.add_node("analysis_agent", analysis_node)
    graph.add_node("report_agent", report_node)

    graph.set_entry_point("data_agent")
    graph.add_edge("data_agent", "rag_agent")
    graph.add_edge("rag_agent", "analysis_agent")
    graph.add_edge("analysis_agent", "report_agent")
    graph.add_edge("report_agent", END)

    return graph.compile()


# Singleton — import this in Chainlit app
race_graph = build_graph()
