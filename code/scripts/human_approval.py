"""
Demonstrates Human-In-The-Loop (HITL) architecture using LangGraph MemorySaver.
"""
import sys
from pathlib import Path

# Add the 'code' directory to sys.path so we can import 'graph'
sys.path.append(str(Path(__file__).resolve().parent.parent))

from graph.pipeline import get_pipeline
from graph.state import GraphState

def simulate_hitl(ticket_text: str, thread_id: str):
    """
    Simulate a pipeline that pauses before sending an email to a user.
    """
    pipeline = get_pipeline()
    
    # Create thread configuration
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
    # Note: In a true HITL setup, the pipeline is compiled with:
    # workflow.compile(checkpointer=MemorySaver(), interrupt_before=["output_formatter"])
    
    try:
        final_state = pipeline.invoke(state, config=config)
        print("2. Pipeline execution complete or paused!")
        
        # To resume, an admin would review final_state["synthesis_result"].response
        # and then run: pipeline.invoke(None, config=config)
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    print("Testing HITL Blueprint...")
    simulate_hitl("I want to cancel my subscription!", "thread_123")
