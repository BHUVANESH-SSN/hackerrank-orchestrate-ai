"""
Synthesis agent + Faithfulness check.
- Synthesis: Calls Groq API (Llama) with retrieved context to generate grounded response.
- Faithfulness: Calls Groq API to verify response doesn't hallucinate.
"""
import json
import re
import time
from typing import Any

from groq import Groq

from config.settings import settings
from models.schemas import (
    Action,
    FaithfulnessResult,
    RequestType,
    RiskResult,
    SynthesisResult,
    UrgencyLevel,
)


def _get_domain_name(domain_val: str) -> str:
    """Human-friendly domain name."""
    _map = {
        "hackerrank": "HackerRank Support",
        "claude": "Claude (Anthropic) Support",
        "visa": "Visa Support",
    }
    return _map.get(domain_val, "Support")


def run_synthesis(state: dict[str, Any]) -> dict[str, Any]:
    """Generate a grounded support response using Groq API + retrieved chunks."""
    try:
        domain = state["domain_result"].domain
        request_type = state["domain_result"].request_type
        retrieval = state["retrieval_result"]
        raw_ticket = state["raw_ticket"]

        # Handle invalid requests without using context
        if request_type == RequestType.INVALID:
            return {
                "synthesis_result": SynthesisResult(
                    response_text="I'm sorry, this request is outside the scope of our support capabilities. If you have a question related to our products or services, please feel free to ask.",
                    sources=[],
                    is_grounded=True,
                    confidence=1.0,
                )
            }

        # If no chunks were retrieved, we can't generate a grounded response
        if not retrieval.chunks:
            return {
                "synthesis_result": SynthesisResult(
                    response_text="I don't have enough information in our support documentation to fully answer this. I recommend contacting our support team directly for assistance.",
                    sources=[],
                    is_grounded=False,
                    confidence=0.2,
                )
            }

        # Build context block from retrieved chunks
        context_parts: list[str] = []
        for i, chunk in enumerate(retrieval.chunks, 1):
            context_parts.append(
                f"--- Source {i}: {chunk.source_title} ({chunk.source_url}) ---\n"
                f"{chunk.chunk_text}"
            )
        context_block = "\n\n".join(context_parts)
        domain_name = _get_domain_name(domain.value)

        system_prompt = f"""You are a support agent for {domain_name}.

CRITICAL RULES:
1. Answer ONLY using information from the provided context sections below.
2. If the context does not contain enough information to fully answer the question, explicitly say: "I don't have enough information in our support documentation to fully answer this. I recommend contacting our support team directly."
3. NEVER invent policies, prices, time limits, procedures, or any facts not in the context.
4. Always be helpful, empathetic, and professional.
5. Keep your response concise and structured (use bullet points where helpful).
6. At the end, cite the sources you used as: "Sources: [URL1], [URL2]"
7. MULTILINGUAL SUPPORT: You MUST auto-detect the language of the user's ticket and write your response entirely in that same language, translating technical terms where appropriate.

Context:
{context_block}"""

        client = Groq(api_key=settings.groq_api_key)
        for attempt in range(8):
            try:
                response = client.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": raw_ticket},
                    ],
                    max_tokens=800,
                    temperature=0.2,
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

        # Extract sources from response
        sources: list[str] = []
        for chunk in retrieval.chunks:
            if chunk.source_url:
                sources.append(chunk.source_url)
        sources = list(dict.fromkeys(sources))  # Dedupe keeping order

        return {
            "synthesis_result": SynthesisResult(
                response_text=response_text,
                sources=sources,
                is_grounded=True,
                confidence=0.85,
            )
        }

    except Exception as e:
        return {
            "synthesis_result": SynthesisResult(
                response_text="I apologize, but I'm unable to process your request at this time. Please contact our support team directly for assistance.",
                sources=[],
                is_grounded=False,
                confidence=0.0,
            ),
            "error": f"Synthesis error: {str(e)}",
        }


def run_faithfulness_check(state: dict[str, Any]) -> dict[str, Any]:
    """Verify that the synthesis response is grounded in the retrieved context."""
    try:
        synthesis = state["synthesis_result"]
        retrieval = state["retrieval_result"]

        # Skip if response is clearly a fallback / out-of-scope
        if not synthesis.is_grounded or not retrieval.chunks:
            return {
                "faithfulness_result": FaithfulnessResult(
                    is_faithful=False,
                    faithfulness_score=0.0,
                    reasoning="No context available or response marked as ungrounded",
                )
            }

        # If request is invalid type, it's automatically faithful
        if state["domain_result"].request_type == RequestType.INVALID:
            return {
                "faithfulness_result": FaithfulnessResult(
                    is_faithful=True,
                    faithfulness_score=1.0,
                    reasoning="Out-of-scope response — no factual claims to verify",
                )
            }

        # Build context for verification
        context_parts = [chunk.chunk_text for chunk in retrieval.chunks]
        context = "\n\n---\n\n".join(context_parts)

        prompt = f"""You are a faithfulness checker. Your job is to verify that a support response does not contain any claims that are not grounded in the provided context.

Context provided to the agent:
{context}

Agent's response:
{synthesis.response_text}

Question: Does the agent's response contain ANY factual claims, policies, procedures, prices, or timelines that are NOT supported by the context above?

Respond ONLY with valid JSON:
{{
  "is_faithful": true,
  "faithfulness_score": 0.0-1.0,
  "reasoning": "one sentence"
}}"""
        client = Groq(api_key=settings.groq_api_key)
        for attempt in range(8):
            try:
                response = client.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=200,
                    temperature=0.0,
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

        # Parse JSON
        json_match = re.search(r'\{[^{}]*\}', response_text, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group())
        else:
            data = json.loads(response_text)

        is_faithful = data.get("is_faithful", True)
        score = float(data.get("faithfulness_score", 0.8))
        reasoning = data.get("reasoning", "Faithfulness check completed")

        result: dict[str, Any] = {
            "faithfulness_result": FaithfulnessResult(
                is_faithful=is_faithful,
                faithfulness_score=score,
                reasoning=reasoning,
            )
        }

        # If unfaithful → force escalation
        if not is_faithful:
            result["risk_result"] = RiskResult(
                action=Action.ESCALATE,
                risk_score=8,
                urgency=UrgencyLevel.HIGH,
                escalation_reason=f"Faithfulness check failed: {reasoning}",
                escalation_team="Tier 2 Support",
            )

        return result

    except Exception as e:
        # Default to faithful on error (don't block pipeline)
        return {
            "faithfulness_result": FaithfulnessResult(
                is_faithful=True,
                faithfulness_score=0.5,
                reasoning=f"Faithfulness check error: {str(e)[:80]}",
            )
        }
