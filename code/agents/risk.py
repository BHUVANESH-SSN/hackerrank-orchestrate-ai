"""
Risk agent — deterministic risk scoring and reply/escalate decision.
No LLM calls. Uses utils/safety.py logic.
"""
import json
import json
import re
from typing import Any

from groq import Groq

from config.settings import settings
from models.schemas import (
    Action,
    ProductArea,
    RiskResult,
    UrgencyLevel,
)
from utils.safety import get_risk_score


def run_risk(state: dict[str, Any]) -> dict[str, Any]:
    ### use of this function: run risk
    """Assess risk and decide REPLY vs ESCALATE."""
    try:
        text = state["raw_ticket"]
        intake = state["intake"]
        domain_result = state["domain_result"]
        product_area = domain_result.product_area

        risk_score, team, reason = get_risk_score(text)

        if product_area == ProductArea.VISA_FRAUD:
            risk_score = max(risk_score, 10)
            team = team or "Visa Fraud & Security Team"
            reason = reason or "Fraud-classified ticket"

        if product_area == ProductArea.PROCTORING:
            risk_score = max(risk_score, 5)

        if product_area in (ProductArea.CLAUDE_ACCOUNT, ProductArea.HR_ACCOUNT):
            risk_score = max(risk_score, 5)

        if product_area == ProductArea.VISA_DISPUTES:
            risk_score = max(risk_score, 5)

        if intake.pii_detected:
            risk_score = max(risk_score, 6)
            team = team or "Data Protection Team"
            reason = reason or "PII detected in ticket — handle with care"

        request_type = domain_result.request_type
        if request_type.value == "invalid":
            risk_score = min(risk_score, 3)
            team = None
            reason = None

        threshold = settings.risk_threshold 

        sentiment = "neutral"
        churn_risk = False

        risk_score = max(0, min(risk_score, 10))

        if risk_score >= threshold:
            action = Action.ESCALATE
            urgency = UrgencyLevel.CRITICAL if risk_score >= 9 else UrgencyLevel.HIGH
        else:
            action = Action.REPLY
            urgency = UrgencyLevel.LOW

        return {
            "risk_result": RiskResult(
                action=action,
                risk_score=risk_score,
                urgency=urgency,
                escalation_reason=reason,
                escalation_team=team,
            ),
            "sentiment": sentiment,
            "churn_risk": churn_risk,
        }
    except Exception as e:
        return {
            "risk_result": RiskResult(
                action=Action.ESCALATE,
                risk_score=10,
                urgency=UrgencyLevel.CRITICAL,
                escalation_reason=f"Risk assessment error: {str(e)[:100]}",
                escalation_team="Tier 2 Support",
            ),
            "error": f"Risk error: {str(e)}",
        }
