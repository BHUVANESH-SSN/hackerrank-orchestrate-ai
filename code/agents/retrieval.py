"""
Retrieval agent — runs hybrid search on the domain's corpus.
"""
from typing import Any

from config.settings import settings
from models.schemas import Action, RiskResult, UrgencyLevel
from rag.retriever import get_retriever


def run_retrieval(state: dict[str, Any]) -> dict[str, Any]:
    """Run hybrid retrieval against the domain corpus."""
    try:
        domain = state["domain_result"].domain
        intake = state["intake"]

        # Build a rich retrieval query from intake
        keywords_str = " ".join(intake.keywords[:10])
        query = f"{intake.ticket_summary} {keywords_str}"

        # Run hybrid retrieval
        retriever = get_retriever()
        result = retriever.retrieve(
            domain=domain,
            query=query,
            top_k=settings.rerank_top_k,
        )

        # If no results found, update risk to escalate
        update: dict[str, Any] = {"retrieval_result": result}
        if not result.chunks:
            update["risk_result"] = RiskResult(
                action=Action.ESCALATE,
                risk_score=7,
                urgency=UrgencyLevel.HIGH,
                escalation_reason="No relevant documentation found in corpus for this query",
                escalation_team="Tier 2 Support",
            )

        return update

    except Exception as e:
        from models.schemas import RetrievalResult
        return {
            "retrieval_result": RetrievalResult(
                chunks=[],
                query_used="",
                total_candidates=0,
            ),
            "risk_result": RiskResult(
                action=Action.ESCALATE,
                risk_score=7,
                urgency=UrgencyLevel.HIGH,
                escalation_reason=f"Retrieval error: {str(e)[:100]}",
                escalation_team="Tier 2 Support",
            ),
            "error": f"Retrieval error: {str(e)}",
        }
