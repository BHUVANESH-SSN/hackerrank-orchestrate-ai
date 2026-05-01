"""
Classifier agent — uses Groq API (Llama) for domain + product area classification with keyword fallback.
"""
import json
import re
import time
from typing import Any

from groq import Groq

from config.settings import settings
from models.schemas import Domain, ProductArea, RequestType, DomainResult


# ──────────────────────────────────────────────
# Keyword fallback maps
# ──────────────────────────────────────────────

DOMAIN_KEYWORDS: dict[Domain, list[str]] = {
    Domain.HACKERRANK: [
        "hackerrank", "assessment", "proctoring", "coding test", "recruiter",
        "candidate", "challenge", "submission", "test", "interview", "hiring",
        "certificate", "mock interview", "resume builder", "prep kit",
    ],
    Domain.CLAUDE: [
        "claude", "anthropic", "claude.ai", "api key", "model", "prompt",
        "conversation", "bedrock", "ai assistant", "lti",
    ],
    Domain.VISA: [
        "visa", "card", "transaction", "payment", "cvv", "pin", "atm",
        "merchant", "dispute", "chargeback", "traveller", "cheque",
    ],
}


def _keyword_classify_domain(text: str, company: str) -> Domain:
    """Fallback classification using keyword matching and company field."""
    text_lower = text.lower()

    # Company field is the strongest signal
    company_lower = company.lower().strip() if company else ""
    if company_lower in ("hackerrank", "hacker rank"):
        return Domain.HACKERRANK
    if company_lower in ("claude", "anthropic"):
        return Domain.CLAUDE
    if company_lower in ("visa",):
        return Domain.VISA

    # Keyword scoring
    scores: dict[Domain, int] = {d: 0 for d in Domain}
    for domain, keywords in DOMAIN_KEYWORDS.items():
        for kw in keywords:
            if kw.lower() in text_lower:
                scores[domain] += 1

    best = max(scores, key=lambda d: scores[d])
    if scores[best] > 0:
        return best
    return Domain.UNKNOWN


def _keyword_classify_request_type(text: str) -> RequestType:
    """Classify request type from text using keywords."""
    text_lower = text.lower()

    bug_keywords = ["bug", "broken", "not working", "error", "crash", "down", "failing", "glitch", "issue"]
    if any(kw in text_lower for kw in bug_keywords):
        return RequestType.BUG

    feature_keywords = ["feature", "would be nice", "can you add", "wish", "suggest", "request", "want to"]
    if any(kw in text_lower for kw in feature_keywords):
        return RequestType.FEATURE_REQUEST

    invalid_keywords = ["iron man", "batman", "movie", "what is the name of", "delete all files"]
    if any(kw in text_lower for kw in invalid_keywords):
        return RequestType.INVALID

    return RequestType.PRODUCT_ISSUE


def run_classifier(state: dict[str, Any]) -> dict[str, Any]:
    """Classify the ticket into domain + product area using Groq API with keyword fallback."""
    try:
        intake = state["intake"]
        text = state["raw_ticket"]
        company = state.get("company", "")
        subject = state.get("subject", "")

        combined = f"{subject}\n{text}" if subject else text

        # Try LLM classification via Groq
        client = Groq(api_key=settings.groq_api_key)

        system_prompt = """You are a support ticket classifier. Given a support ticket, you must classify it into exactly one domain and product area, and determine the request type.

Domains and their product areas:
- hackerrank: screen, assessments, coding_challenges, proctoring, hr_billing, hr_account, hr_bugs, hr_general, community
- claude: claude_api, claude_ui, claude_billing, claude_safety, claude_account, claude_general, privacy, conversation_management
- visa: visa_fraud, visa_disputes, visa_card_services, visa_merchant, visa_eligibility, visa_general, travel_support, general_support

Request types: product_issue, feature_request, bug, invalid

Respond ONLY with valid JSON matching this schema:
{
  "domain": "<domain>",
  "product_area": "<product_area>",
  "request_type": "<request_type>",
  "classification_confidence": <0.0-1.0>,
  "classification_reasoning": "<one sentence>"
}

Rules:
- If the ticket mentions card, transaction, payment, bank, CVV, PIN, merchant, dispute, traveller cheque → domain is "visa"
- If the ticket mentions assessment, test, coding challenge, proctoring, candidate, recruiter, HackerRank, certificate, mock interview, resume builder → domain is "hackerrank"
- If the ticket mentions Claude, Anthropic, AI assistant, API key, model, claude.ai, bedrock → domain is "claude"
- If the company field explicitly says a company name, use that as the primary domain signal
- For out-of-scope requests (asking about movies, unrelated topics, malicious intent), set request_type to "invalid"
- Product area must be from the exact list above
- For "not working" or "down" issues, use request_type "bug"
- For asking how to do something that the product supports, use request_type "product_issue"
"""

        user_msg = f"Company: {company}\nSubject: {subject}\n\nFull text: {combined[:800]}"

        # Robust retry for Groq 429 rate limits
        for attempt in range(8):
            try:
                response = client.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_msg},
                    ],
                    max_tokens=300,
                    temperature=0.1,
                )
                break
            except Exception as e:
                err_str = str(e)
                if "429" in err_str or "Rate limit" in err_str:
                    wait_s = (2 ** attempt) + 10
                    time.sleep(wait_s)
                elif attempt == 7:
                    raise e
                else:
                    time.sleep(2)


        response_text = response.choices[0].message.content.strip()

        # Try to extract JSON from response
        json_match = re.search(r'\{[^{}]*\}', response_text, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group())
        else:
            data = json.loads(response_text)

        domain = Domain(data.get("domain", "unknown"))
        product_area_str = data.get("product_area", "unknown")

        try:
            product_area = ProductArea(product_area_str)
        except ValueError:
            product_area = ProductArea.UNKNOWN

        try:
            request_type = RequestType(data.get("request_type", "product_issue"))
        except ValueError:
            request_type = _keyword_classify_request_type(combined)

        confidence = float(data.get("classification_confidence", 0.8))
        reasoning = data.get("classification_reasoning", "Classified by LLM")

        # Keyword fallback if confidence is low
        if confidence < 0.5:
            domain = _keyword_classify_domain(combined, company)

        return {
            "domain_result": DomainResult(
                domain=domain,
                product_area=product_area,
                classification_confidence=confidence,
                classification_reasoning=reasoning,
                request_type=request_type,
            )
        }

    except Exception as e:
        # Full keyword fallback
        text = state.get("raw_ticket", "")
        company = state.get("company", "")
        combined = f"{state.get('subject', '')} {text}"

        domain = _keyword_classify_domain(combined, company)
        request_type = _keyword_classify_request_type(combined)

        return {
            "domain_result": DomainResult(
                domain=domain,
                product_area=ProductArea.UNKNOWN,
                classification_confidence=0.3,
                classification_reasoning=f"Keyword fallback (API error: {str(e)[:80]})",
                request_type=request_type,
            )
        }
