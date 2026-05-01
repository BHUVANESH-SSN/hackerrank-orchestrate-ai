"""
Intake agent — deterministic ticket parsing (no LLM calls).
Extracts urgency signals, PII, sentiment, keywords, and summary.
"""
import re
from typing import Any

from models.schemas import IntakeResult
from utils.safety import detect_pii


def run_intake(state: dict[str, Any]) -> dict[str, Any]:
    ### use of this function: run intake
    """Parse the raw ticket text deterministically."""
    try:
        text = state["raw_ticket"]
        subject = state.get("subject", "")

        full_text = f"{subject} {text}" if subject else text

        urgency_keywords = [
            "urgent", "immediately", "asap", "emergency", "critical",
            "fraud", "stolen", "hacked", "compromised", "locked out",
            "can't access", "lost access", "unauthorized", "dispute",
            "refund", "overcharged", "false positive", "disqualified",
            "legal", "lawsuit", "identity theft", "security vulnerability",
            "blocker", "not working", "down", "failing",
        ]
        found_signals = [kw for kw in urgency_keywords if kw.lower() in full_text.lower()]

        pii = detect_pii(full_text)

        negative_words = [
            "angry", "frustrated", "terrible", "awful", "worst",
            "unacceptable", "disgusting", "annoyed", "furious", "upset",
            "ridiculous", "horrible", "pathetic",
        ]
        positive_words = [
            "thank", "great", "love", "excellent", "helpful",
            "appreciate", "happy", "good", "amazing", "wonderful",
        ]

        neg_count = sum(1 for w in negative_words if w in full_text.lower())
        pos_count = sum(1 for w in positive_words if w in full_text.lower())

        if neg_count > pos_count:
            sentiment = "negative"
        elif pos_count > neg_count:
            sentiment = "positive"
        else:
            sentiment = "neutral"

        keywords = list(set(re.findall(r'\b[A-Za-z]{4,}\b', full_text)))[:20]

        summary_text = text.strip().replace("\n", " ").replace("\r", "")
        summary_text = re.sub(r'\s+', ' ', summary_text)
        summary = summary_text[:200] if summary_text else "No content"

        return {
            "intake": IntakeResult(
                raw_text=text,
                ticket_summary=summary,
                urgency_signals=found_signals,
                pii_detected=pii,
                sentiment=sentiment,
                keywords=keywords,
            )
        }
    except Exception as e:
        return {
            "intake": IntakeResult(
                raw_text=state.get("raw_ticket", ""),
                ticket_summary="Error parsing ticket",
                urgency_signals=[],
                pii_detected=False,
                sentiment="neutral",
                keywords=[],
            ),
            "error": f"Intake error: {str(e)}",
        }
