"""
Output formatter — assembles the final TicketOutput from pipeline state.
Also provides CSV row formatting.
"""
import time
from typing import Any

from models.schemas import (
    Action,
    Domain,
    ProductArea,
    RequestType,
    Status,
    TicketOutput,
)


def run_output_formatter(state: dict[str, Any]) -> dict[str, Any]:
    """Assemble the final TicketOutput from all state fields."""
    try:
        risk = state["risk_result"]
        domain_result = state["domain_result"]
        intake = state.get("intake")
        synthesis = state.get("synthesis_result")
        faithfulness = state.get("faithfulness_result")

        # Determine response text
        if risk.action == Action.ESCALATE:
            response_text = state.get("escalation_response", "Your case has been escalated for review.")
            sources: list[str] = []
        elif synthesis:
            response_text = synthesis.response_text
            sources = synthesis.sources
        else:
            response_text = "Unable to generate response. Please contact support directly."
            sources = []

        # Build justification
        justification_parts: list[str] = []
        justification_parts.append(f"Domain: {domain_result.domain.value}")
        justification_parts.append(f"Product area: {domain_result.product_area.value}")
        justification_parts.append(domain_result.classification_reasoning)

        if risk.action == Action.ESCALATE and risk.escalation_reason:
            justification_parts.append(f"Escalation reason: {risk.escalation_reason}")

        if faithfulness and faithfulness.faithfulness_score < 0.7:
            justification_parts.append(f"Faithfulness concern: {faithfulness.reasoning}")

        justification = ". ".join(justification_parts)

        # Calculate processing time
        start = state.get("start_time_ms", 0)
        processing_time = int(time.time() * 1000) - start if start else 0

        output = TicketOutput(
            ticket_id=state.get("ticket_id", "unknown"),
            original_text=state.get("raw_ticket", ""),
            subject=state.get("subject", ""),
            company=state.get("company", ""),
            domain=domain_result.domain,
            product_area=domain_result.product_area,
            action=risk.action,
            request_type=domain_result.request_type,
            risk_score=risk.risk_score,
            response=response_text,
            justification=justification,
            sources=sources,
            escalation_reason=risk.escalation_reason,
            escalation_team=risk.escalation_team,
            faithfulness_score=faithfulness.faithfulness_score if faithfulness else None,
            processing_time_ms=processing_time,
        )

        return {"final_output": output}

    except Exception as e:
        # Fallback output
        return {
            "final_output": TicketOutput(
                ticket_id=state.get("ticket_id", "unknown"),
                original_text=state.get("raw_ticket", ""),
                domain=Domain.UNKNOWN,
                product_area=ProductArea.UNKNOWN,
                action=Action.ESCALATE,
                request_type=RequestType.PRODUCT_ISSUE,
                risk_score=10,
                response="System error — escalating for manual review.",
                justification=f"Pipeline error: {str(e)[:100]}",
                sources=[],
                escalation_reason="Pipeline error",
                escalation_team="Tier 2 Support",
            )
        }


def format_csv_row(output: TicketOutput) -> dict[str, str]:
    """Convert TicketOutput to a flat dict for the output CSV.

    Required output columns per problem_statement.md:
    status, product_area, response, justification, request_type
    """
    # Map action → status
    status = Status.ESCALATED if output.action == Action.ESCALATE else Status.REPLIED

    return {
        "status": status.value,
        "product_area": output.product_area.value,
        "response": output.response,
        "justification": output.justification,
        "request_type": output.request_type.value,
    }
