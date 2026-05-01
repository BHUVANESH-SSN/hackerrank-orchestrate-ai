"""
Safety utilities — PII detection, risk scoring, escalation triggers, PII redaction.
All deterministic (no LLM calls).
"""
import re
from typing import Optional


# ──────────────────────────────────────────────
# Escalation Triggers
# Maps regex pattern → (risk_score, escalation_team, reason)
# ──────────────────────────────────────────────

ESCALATION_TRIGGERS: dict[str, tuple[int, str, str]] = {
    # Visa — fraud / unauthorized
    r"(fraud|unauthorized.*(charge|transaction)|stolen card|card compromised|someone used my card)":
        (10, "Visa Fraud & Security Team", "Potential fraud or unauthorized transaction detected"),
    r"(dispute|chargeback|transaction dispute)":
        (8, "Visa Disputes Resolution Team", "Transaction dispute requires human review"),

    # All domains — account security
    r"(account.*hacked|account.*compromised|someone.*logged in|unauthorized.*access|can.?t log.*in|locked.*out|account.*suspended|account.*banned)":
        (9, "Account Security Team", "Account security incident"),

    # HackerRank — proctoring
    r"(proctoring.*wrong|false.*positive|incorrectly.*flagged|cheating.*flag|assessment.*disqualif|unfairly.*penaliz)":
        (9, "HackerRank Integrity & Appeals Team", "Proctoring false positive — affects candidate career"),

    # Legal/compliance
    r"(legal|lawsuit|attorney|GDPR|data breach|subpoena|court order|regulatory)":
        (10, "Legal & Compliance Team", "Legal or regulatory matter"),

    # Identity theft
    r"(identity.*stolen|identity.*theft|someone.*my identity)":
        (10, "Visa Fraud & Security Team", "Identity theft reported — immediate action required"),

    # PII / data exposure
    r"(my data.*exposed|data.*leak|personal.*information.*shared)":
        (10, "Data Protection Officer", "Potential data exposure incident"),

    # Security vulnerability
    r"(security.*vulnerability|security.*bug|bug.*bounty|exploit|vulnerability)":
        (8, "Security Response Team", "Security vulnerability report"),

    # Billing disputes (specific amounts)
    r"(charged.*\$[\d]+|incorrect.*charge|overcharged|double.?charged|refund.*\$[\d]+)":
        (7, "Billing Support Team", "Specific billing dispute requires human verification"),
}

# ──────────────────────────────────────────────
# PII Detection Patterns
# ──────────────────────────────────────────────

_PII_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("credit_card", re.compile(r"\b(?:\d[ -]*?){13,19}\b")),
    ("email", re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b")),
    ("phone", re.compile(r"\b(?:\+?1[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)?\d{3}[-.\s]?\d{4}\b")),
    ("ssn", re.compile(r"\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b")),
    ("order_id", re.compile(r"\b(cs_live_[a-zA-Z0-9]+|order[_\s]?[Ii][Dd][:\s]*\w+)\b")),
]


def detect_pii(text: str) -> bool:
    """Return True if any PII-like patterns are found in the text."""
    for _name, pattern in _PII_PATTERNS:
        if pattern.search(text):
            return True
    return False


def redact_pii(text: str) -> str:
    """Replace PII patterns with [REDACTED]."""
    result = text
    for _name, pattern in _PII_PATTERNS:
        result = pattern.sub("[REDACTED]", result)
    return result


def get_risk_score(text: str) -> tuple[int, Optional[str], Optional[str]]:
    """
    Scan text against ESCALATION_TRIGGERS.
    Returns (highest_risk_score, escalation_team, reason).
    Defaults to (2, None, None) if no trigger matches.
    """
    best_score = 2
    best_team: Optional[str] = None
    best_reason: Optional[str] = None

    for pattern, (score, team, reason) in ESCALATION_TRIGGERS.items():
        if re.search(pattern, text, re.IGNORECASE):
            if score > best_score:
                best_score = score
                best_team = team
                best_reason = reason

    return best_score, best_team, best_reason
