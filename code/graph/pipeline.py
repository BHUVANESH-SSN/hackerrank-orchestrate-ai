"""
LangGraph pipeline — state machine orchestrating all agents.
"""
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from graph.state import GraphState
from models.schemas import Action
from agents.intake import run_intake
from agents.classifier import run_classifier
from agents.risk import run_risk
from agents.retrieval import run_retrieval
from agents.synthesis import run_synthesis, run_faithfulness_check
from agents.escalation import run_escalation
from agents.output_formatter import run_output_formatter


def route_after_risk(state: GraphState) -> str:
    ### use of this function: route after risk
    """Conditional edge: after risk assessment, decide path."""
    action = state["risk_result"].action
    if action == Action.ESCALATE:
        return "escalation"
    else:
        return "retrieval"


def route_after_faithfulness(state: GraphState) -> str:
    ### use of this function: route after faithfulness
    """Conditional edge: after faithfulness check, decide path."""
    faith = state.get("faithfulness_result")
    if faith and faith.is_faithful:
        return "output"
    else:
        return "escalation"


def build_graph() -> StateGraph:
    ### use of this function: build graph
    """Build and compile the LangGraph state machine."""
    graph = StateGraph(GraphState)

    graph.add_node("intake", run_intake)
    graph.add_node("classifier", run_classifier)
    graph.add_node("risk", run_risk)
    graph.add_node("retrieval", run_retrieval)
    graph.add_node("synthesis", run_synthesis)
    graph.add_node("faithfulness", run_faithfulness_check)
    graph.add_node("escalation", run_escalation)
    graph.add_node("output", run_output_formatter)

    graph.set_entry_point("intake")
    graph.add_edge("intake", "classifier")
    graph.add_edge("classifier", "risk")
    graph.add_edge("retrieval", "synthesis")
    graph.add_edge("synthesis", "faithfulness")
    graph.add_edge("escalation", "output")
    graph.add_edge("output", END)

    graph.add_conditional_edges(
        "risk",
        route_after_risk,
        {"escalation": "escalation", "retrieval": "retrieval"},
    )
    graph.add_conditional_edges(
        "faithfulness",
        route_after_faithfulness,
        {"output": "output", "escalation": "escalation"},
    )

    return graph


_app = None


def get_pipeline():
    ### use of this function: get pipeline
    """Get or create the singleton compiled pipeline."""
    global _app
    if _app is None:
        memory = MemorySaver()
        
        _app = build_graph().compile(checkpointer=memory)
    return _app
