"""
Escalation agent — generates professional escalation responses using templates.
No LLM calls.
"""
from typing import Any

from models.schemas import Action, ProductArea


# ──────────────────────────────────────────────
# Escalation Response Templates
# ──────────────────────────────────────────────

ESCALATION_TEMPLATES: dict[str, str] = {
    "Visa Fraud & Security Team": """Thank you for contacting Visa Support. We take security concerns very seriously.

Your case has been flagged as a high-priority security matter and is being escalated to our Fraud & Security Team for immediate review.

Next steps:
- A security specialist will contact you within 24 hours
- Please do not attempt further transactions until resolved
- If you believe your card is compromised, call the number on the back of your card immediately to freeze it

We apologize for any inconvenience and appreciate your patience while we investigate.""",

    "Visa Disputes Resolution Team": """Thank you for reaching out regarding your transaction dispute.

Your case has been escalated to our Disputes Resolution Team for review. Transaction disputes require careful investigation to ensure a fair resolution.

Next steps:
- A disputes specialist will review your case within 2-3 business days
- Please gather any relevant receipts, statements, or correspondence with the merchant
- You will receive an update via your registered contact method

We appreciate your patience while we work to resolve this matter.""",

    "HackerRank Integrity & Appeals Team": """Thank you for bringing this to our attention. We understand how important assessment integrity is for your career.

Your case has been escalated to our Integrity & Appeals Team for review. All proctoring-related concerns are carefully investigated.

Next steps:
- Our review team will examine the assessment session details
- You will receive a response within 3-5 business days
- No further action is needed on your end at this time

We are committed to ensuring fair and accurate assessment experiences.""",

    "Account Security Team": """Thank you for reporting this account security concern. We take account protection very seriously.

Your case has been escalated to our Account Security Team for immediate review.

Next steps:
- A security specialist will investigate your account
- You may be asked to verify your identity through our secure process
- Please change your password on any other accounts where you used the same credentials

We will follow up as soon as possible to help secure your account.""",

    "Legal & Compliance Team": """Thank you for your communication. This matter has been noted and will be handled by our Legal & Compliance Team.

Your case has been forwarded to the appropriate team for review. Due to the nature of this inquiry, please note:

- A team representative will respond within 5-7 business days
- All communications will be handled through official channels
- Please retain any relevant documentation for your records

We appreciate your patience.""",

    "Data Protection Officer": """Thank you for bringing this data privacy concern to our attention. We take data protection very seriously and are committed to safeguarding your personal information.

Your case has been escalated to our Data Protection team for immediate review.

Next steps:
- Our privacy team will investigate your concern
- You will receive a detailed response within 5 business days
- If applicable, we will take immediate steps to address any data exposure

We appreciate you reporting this matter.""",

    "Security Response Team": """Thank you for reporting this security concern. We appreciate responsible disclosure and take all security reports seriously.

Your report has been escalated to our Security Response Team for investigation.

Next steps:
- A security engineer will review your report
- You will receive an acknowledgment within 48 hours
- If validated, we will work on a fix and keep you updated

We value your contribution to making our platform safer.""",

    "Billing Support Team": """Thank you for contacting us about this billing concern.

Your case has been escalated to our Billing Support Team for review. Billing matters require careful verification to ensure accuracy.

Next steps:
- A billing specialist will review your account and the specific charges
- You will receive a response within 2-3 business days
- Please have your account details and relevant transaction information ready

We apologize for any billing discrepancy and will work to resolve this promptly.""",

    "Tier 2 Support": """Thank you for reaching out to our support team.

Your request requires further investigation by our specialized support team and has been escalated accordingly.

Next steps:
- A qualified support agent will review your case
- You will receive a follow-up within 1-2 business days
- No further action is needed from you at this time

We appreciate your patience and are committed to resolving your issue.""",
}


def run_escalation(state: dict[str, Any]) -> dict[str, Any]:
    """Generate an escalation response using templates."""
    try:
        risk = state["risk_result"]
        team = risk.escalation_team or "Tier 2 Support"
        reason = risk.escalation_reason or "Requires specialized review"

        # Get template
        template = ESCALATION_TEMPLATES.get(team, ESCALATION_TEMPLATES["Tier 2 Support"])

        # For default template, fill in team and reason
        if team not in ESCALATION_TEMPLATES:
            template = (
                f"Thank you for reaching out to our support team.\n\n"
                f"Your request has been reviewed and requires attention from a specialist. "
                f"Your case has been escalated to **{team}** for further assistance.\n\n"
                f"**Reason for escalation:** {reason}\n\n"
                f"A team member will follow up with you as soon as possible. "
                f"We appreciate your patience and apologize for any inconvenience."
            )

        return {"escalation_response": template}

    except Exception:
        return {
            "escalation_response": (
                "Thank you for contacting support. Your case has been escalated for further review. "
                "A team member will follow up with you shortly."
            )
        }
