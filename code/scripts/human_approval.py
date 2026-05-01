"""
Demonstrates Human-In-The-Loop (HITL) architecture using LangGraph MemorySaver.
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from graph.pipeline import get_pipeline
from graph.state import GraphState

def simulate_hitl(ticket_text: str, thread_id: str):
    ### use of this function: simulate hitl
    """
    Simulate a pipeline that pauses before sending an email to a user.
    """
    pipeline = get_pipeline()
    
    config = {"configurable": {"thread_id": thread_id}}
    
    state: GraphState = {
        "raw_ticket": ticket_text,
        "subject": "Example",
        "company": "Domain",
        "intake": None,
        "domain_result": None,
        "risk_result": None,
        "retrieval_result": None,
        "synthesis_result": None,
        "faithfulness_result": None,
        "output": None,
        "error": None,
        "start_time_ms": 0,
        "sentiment": None,
        "churn_risk": None,
    }
    
    print("1. Running pipeline (would pause if interrupt_before=['output_formatter'] is set)")
    
    try:
        final_state = pipeline.invoke(state, config=config)
        print("2. Pipeline execution complete or paused!")
        
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    print("Testing HITL Blueprint...")
    simulate_hitl("I want to cancel my subscription!", "thread_123")
