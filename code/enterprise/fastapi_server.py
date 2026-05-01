"""
Enterprise FastAPI Server Blueprint.
Run with: uvicorn enterprise.fastapi_server:app --host 0.0.0.0 --port 8000 --workers 4

Demonstrates handling high-concurrency traffic using Asyncio and a BackgroundTasks queue
to prevent the server from crashing under load.
"""
from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel
from typing import Dict, Any

from graph.pipeline import get_pipeline
from graph.state import GraphState

app = FastAPI(title="HackerRank AI Triage API", version="2.0.0")

class TicketRequest(BaseModel):
    ticket_id: str
    subject: str
    body: str

class TicketResponse(BaseModel):
    status: str
    message: str

PROCESSED_TICKETS = {}

async def process_ticket_async(ticket_id: str, subject: str, body: str):
    ### use of this function: process ticket async
    """Background task to run the LangGraph pipeline asynchronously."""
    pipeline = get_pipeline()
    
    state: GraphState = {
        "raw_ticket": body,
        "subject": subject,
        "company": "FastAPI-Ingest",
        "ticket_id": ticket_id,
        "intake": None,
        "domain_result": None,
        "risk_result": None,
        "retrieval_result": None,
        "synthesis_result": None,
        "faithfulness_result": None,
        "final_output": None,
        "error": None,
        "start_time_ms": 0,
        "sentiment": None,
        "churn_risk": None,
    }
    
    try:
        final_state = pipeline.invoke(state, config={"configurable": {"thread_id": ticket_id}})
        PROCESSED_TICKETS[ticket_id] = final_state.get("final_output")
    except Exception as e:
        PROCESSED_TICKETS[ticket_id] = {"error": str(e)}

@app.post("/ingest", response_model=TicketResponse)
async def ingest_ticket(request: TicketRequest, background_tasks: BackgroundTasks):
    ### use of this function: ingest ticket
    """
    Accepts a ticket from Zendesk/Salesforce, immediately returns 200 OK, 
    and processes it in the background queue.
    """
    if not request.body:
        raise HTTPException(status_code=400, detail="Ticket body is required")
        
    background_tasks.add_task(process_ticket_async, request.ticket_id, request.subject, request.body)
    
    return TicketResponse(
        status="Queued", 
        message=f"Ticket {request.ticket_id} added to the processing queue."
    )

@app.get("/status/{ticket_id}")
async def get_ticket_status(ticket_id: str) -> Dict[str, Any]:
    ### use of this function: get ticket status
    """Check the processing status of a ticket."""
    if ticket_id in PROCESSED_TICKETS:
        return {"status": "Complete", "result": PROCESSED_TICKETS[ticket_id]}
    return {"status": "Processing or Not Found"}
