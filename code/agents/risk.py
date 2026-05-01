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
    """Assess risk and decide REPLY vs ESCALATE."""
    try:
        text = state["raw_ticket"]
        intake = state["intake"]
        domain_result = state["domain_result"]
        product_area = domain_result.product_area

        # 1. Check hard escalation triggers from safety.py
        risk_score, team, reason = get_risk_score(text)

        # 2. Domain-specific overrides
        if product_area == ProductArea.VISA_FRAUD:
            risk_score = max(risk_score, 10)
            team = team or "Visa Fraud & Security Team"
            reason = reason or "Fraud-classified ticket"

        if product_area == ProductArea.PROCTORING:
            risk_score = max(risk_score, 8)
            team = team or "HackerRank Integrity & Appeals Team"
            reason = reason or "Proctoring dispute — affects candidate career"

        if product_area in (ProductArea.CLAUDE_ACCOUNT, ProductArea.HR_ACCOUNT):
            risk_score = max(risk_score, 5)

        if product_area == ProductArea.VISA_DISPUTES:
            risk_score = max(risk_score, 7)
            team = team or "Visa Disputes Resolution Team"
            reason = reason or "Dispute requires human review"

        # 3. PII detected → flag
        if intake.pii_detected:
            risk_score = max(risk_score, 6)
            team = team or "Data Protection Team"
            reason = reason or "PII detected in ticket — handle with care"

        # 4. Request type check — invalid requests should be replied to (not escalated)
        request_type = domain_result.request_type
        if request_type.value == "invalid":
            # Invalid requests get low risk — we just reply saying it's out of scope
            risk_score = min(risk_score, 3)
            team = None
            reason = None

        # 5. Decision
        threshold = settings.risk_threshold  # default 6

        if risk_score >= threshold:
            action = Action.ESCALATE
            urgency = UrgencyLevel.CRITICAL if risk_score >= 9 else UrgencyLevel.HIGH
        else:
            action = Action.REPLY
            urgency = UrgencyLevel.LOW

        # Try to use LLM for Sentiment and Churn Detection
        sentiment = "neutral"
        churn_risk = False
        try:
            client = Groq(api_key=settings.groq_api_key)
            prompt = f"Analyze this support ticket: '{text[:500]}'.\nRespond mathematically valid JSON only with keys: 'sentiment' (positive/neutral/negative/angry) and 'churn_risk' (boolean, is the user threatening to leave or cancel?)."
            resp = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=100,
                temperature=0.0,
            )
            data = json.loads(re.search(r'\{[^{}]*\}', resp.choices[0].message.content.strip()).group())
            sentiment = data.get("sentiment", "neutral").lower()
            churn_risk = data.get("churn_risk", False)
            
            if sentiment == "angry" or churn_risk:
                risk_score += 5
                urgency = UrgencyLevel.HIGH
                team = "Customer Retention"
                reason = "High churn risk or extreme frustration detected"
        except Exception:
            pass # Fall back to heuristic if LLM fails

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
