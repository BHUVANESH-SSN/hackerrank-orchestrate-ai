"""
LangGraph state definition for the multi-agent pipeline.
"""
from typing import TypedDict, Optional

from models.schemas import (
    IntakeResult,
    DomainResult,
    RiskResult,
    RetrievalResult,
    SynthesisResult,
    FaithfulnessResult,
    TicketOutput,
)


class GraphState(TypedDict):
    """State passed through the LangGraph pipeline."""

    ticket_id: str
    raw_ticket: str
    subject: str
    company: str

    intake: Optional[IntakeResult]
    domain_result: Optional[DomainResult]
    risk_result: Optional[RiskResult]
    retrieval_result: Optional[RetrievalResult]
    synthesis_result: Optional[SynthesisResult]
    faithfulness_result: Optional[FaithfulnessResult]
    final_output: Optional[TicketOutput]

    error: Optional[str]
    start_time_ms: int
    
    sentiment: Optional[str]
    churn_risk: Optional[bool]
