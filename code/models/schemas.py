"""
Pydantic v2 models for the multi-domain support triage system.
Aligned to the actual output CSV schema: status, product_area, response, justification, request_type.
"""
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


# ──────────────────────────────────────────────
# Enums
# ──────────────────────────────────────────────

class Domain(str, Enum):
    HACKERRANK = "hackerrank"
    CLAUDE = "claude"
    VISA = "visa"
    UNKNOWN = "unknown"


class ProductArea(str, Enum):
    # HackerRank
    ASSESSMENTS = "assessments"
    CODING_CHALLENGES = "coding_challenges"
    PROCTORING = "proctoring"
    HR_BILLING = "hr_billing"
    HR_ACCOUNT = "hr_account"
    HR_BUGS = "hr_bugs"
    HR_GENERAL = "hr_general"
    # -- Community
    COMMUNITY = "community"
    SCREEN = "screen"
    # Claude
    CLAUDE_API = "claude_api"
    CLAUDE_UI = "claude_ui"
    CLAUDE_BILLING = "claude_billing"
    CLAUDE_SAFETY = "claude_safety"
    CLAUDE_ACCOUNT = "claude_account"
    CLAUDE_GENERAL = "claude_general"
    PRIVACY = "privacy"
    CONVERSATION_MANAGEMENT = "conversation_management"
    # Visa
    VISA_FRAUD = "visa_fraud"
    VISA_DISPUTES = "visa_disputes"
    VISA_CARD_SERVICES = "visa_card_services"
    VISA_MERCHANT = "visa_merchant"
    VISA_ELIGIBILITY = "visa_eligibility"
    VISA_GENERAL = "visa_general"
    TRAVEL_SUPPORT = "travel_support"
    GENERAL_SUPPORT = "general_support"
    # Fallback
    UNKNOWN = "unknown"


class Action(str, Enum):
    REPLY = "reply"
    ESCALATE = "escalate"
    PARTIAL_REPLY = "partial_reply"


class Status(str, Enum):
    """Output CSV status values — capitalized to match sample data."""
    REPLIED = "Replied"
    ESCALATED = "Escalated"


class RequestType(str, Enum):
    """Output CSV request_type values."""
    PRODUCT_ISSUE = "product_issue"
    FEATURE_REQUEST = "feature_request"
    BUG = "bug"
    INVALID = "invalid"


class UrgencyLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


# ──────────────────────────────────────────────
# Agent Result Models
# ──────────────────────────────────────────────

class IntakeResult(BaseModel):
    raw_text: str
    ticket_summary: str = Field(description="1-sentence distillation of the issue")
    urgency_signals: list[str] = Field(default_factory=list)
    pii_detected: bool = False
    sentiment: str = "neutral"
    keywords: list[str] = Field(default_factory=list)


class DomainResult(BaseModel):
    domain: Domain
    product_area: ProductArea
    classification_confidence: float = Field(ge=0.0, le=1.0)
    classification_reasoning: str
    request_type: RequestType = RequestType.PRODUCT_ISSUE


class RiskResult(BaseModel):
    action: Action
    risk_score: int = Field(ge=0, le=10)
    urgency: UrgencyLevel
    escalation_reason: Optional[str] = None
    escalation_team: Optional[str] = None


class RetrievedChunk(BaseModel):
    chunk_text: str
    source_url: str
    source_title: str
    relevance_score: float
    domain: Domain


class RetrievalResult(BaseModel):
    chunks: list[RetrievedChunk]
    query_used: str
    total_candidates: int


class SynthesisResult(BaseModel):
    response_text: str
    sources: list[str]
    is_grounded: bool
    confidence: float = Field(ge=0.0, le=1.0)


class FaithfulnessResult(BaseModel):
    is_faithful: bool
    faithfulness_score: float = Field(ge=0.0, le=1.0)
    reasoning: str


class TicketOutput(BaseModel):
    """Final output for one ticket — maps to output CSV row."""
    ticket_id: str
    original_text: str
    subject: str = ""
    company: str = ""
    domain: Domain
    product_area: ProductArea
    action: Action
    request_type: RequestType = RequestType.PRODUCT_ISSUE
    risk_score: int
    response: str
    justification: str = ""
    sources: list[str] = Field(default_factory=list)
    escalation_reason: Optional[str] = None
    escalation_team: Optional[str] = None
    faithfulness_score: Optional[float] = None
    processing_time_ms: int = 0
