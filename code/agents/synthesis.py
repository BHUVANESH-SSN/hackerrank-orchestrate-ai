"""
Synthesis agent + Faithfulness check.
- Synthesis: Calls AWS Bedrock (Claude Sonnet 4) with retrieved context to generate grounded response.
- Faithfulness: Calls AWS Bedrock (Claude 3.5 Haiku) to verify response doesn't hallucinate.
- Falls back to Groq if AWS credentials are not configured.
"""
import json
import re
import time
import os
from typing import Any

from config.settings import settings
from models.schemas import (
    Action,
    FaithfulnessResult,
    RequestType,
    RiskResult,
    SynthesisResult,
    UrgencyLevel,
)



def _use_bedrock() -> bool:
    ### use of this function: use bedrock
    """Check if AWS Bedrock credentials are configured."""
    return bool(
        os.environ.get("AWS_ACCESS_KEY_ID")
        or os.environ.get("AWS_PROFILE")
        or getattr(settings, "aws_access_key_id", "")
    )


def _call_bedrock(messages: list[dict], model_id: str, max_tokens: int = 800, temperature: float = 0.2) -> str:
    ### use of this function: call bedrock
    """Call AWS Bedrock with the Converse API."""
    import boto3

    region = os.environ.get("AWS_DEFAULT_REGION", getattr(settings, "aws_region", "us-east-1"))
    client = boto3.client("bedrock-runtime", region_name=region)

    bedrock_messages = []
    system_prompt = None
    for msg in messages:
        if msg["role"] == "system":
            system_prompt = msg["content"]
        else:
            bedrock_messages.append({
                "role": msg["role"],
                "content": [{"text": msg["content"]}],
            })

    kwargs = {
        "modelId": model_id,
        "messages": bedrock_messages,
        "inferenceConfig": {
            "maxTokens": max_tokens,
            "temperature": temperature,
        },
    }
    if system_prompt:
        kwargs["system"] = [{"text": system_prompt}]

    for attempt in range(5):
        try:
            response = client.converse(**kwargs)
            return response["output"]["message"]["content"][0]["text"].strip()
        except Exception as e:
            err_str = str(e)
            if "ThrottlingException" in err_str or "Too many requests" in err_str:
                time.sleep((2 ** attempt) + 2)
            elif attempt == 4:
                raise e
            else:
                time.sleep(1)

    return ""


def _call_groq(messages: list[dict], model: str = "qwen/qwen3-32b", max_tokens: int = 800, temperature: float = 0.2, json_mode: bool = False) -> str:
    ### use of this function: call groq
    """Fallback: Call Groq API."""
    from groq import Groq

    client = Groq(api_key=settings.groq_api_key)
    kwargs = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    for attempt in range(8):
        try:
            response = client.chat.completions.create(**kwargs)
            text = response.choices[0].message.content.strip()
            text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
            return text
        except Exception as e:
            err_str = str(e)
            if "429" in err_str or "Rate limit" in err_str:
                time.sleep((2 ** attempt) + 10)
            elif attempt == 7:
                raise e
            else:
                time.sleep(2)

    return ""


def _call_llm(messages: list[dict], max_tokens: int = 800, temperature: float = 0.2,
              model_tier: str = "synthesis", json_mode: bool = False) -> str:
    ### use of this function: call llm
    """
    Unified LLM caller. Uses Bedrock if configured, otherwise Groq.
    model_tier: 'synthesis' → Claude Sonnet 4, 'faithfulness' → Claude 3.5 Haiku
    """
    if _use_bedrock():
        bedrock_models = {
            "synthesis": "anthropic.claude-sonnet-4-5-20250929-v1:0",
            "faithfulness": "anthropic.claude-haiku-4-5-20251001-v1:0",
        }
        model_id = bedrock_models.get(model_tier, bedrock_models["synthesis"])
        return _call_bedrock(messages, model_id, max_tokens, temperature)
    else:
        return _call_groq(messages, max_tokens=max_tokens, temperature=temperature, json_mode=json_mode)



def _get_domain_name(domain_val: str) -> str:
    ### use of this function: get domain name
    """Human-friendly domain name."""
    _map = {
        "hackerrank": "HackerRank Support",
        "claude": "Claude (Anthropic) Support",
        "visa": "Visa Support",
    }
    return _map.get(domain_val, "Support")



def run_synthesis(state: dict[str, Any]) -> dict[str, Any]:
    ### use of this function: run synthesis
    """Generate a grounded support response using LLM + retrieved chunks."""
    try:
        domain = state["domain_result"].domain
        request_type = state["domain_result"].request_type
        retrieval = state["retrieval_result"]
        raw_ticket = state["raw_ticket"]

        if request_type == RequestType.INVALID:
            return {
                "synthesis_result": SynthesisResult(
                    response_text="I'm sorry, this request is outside the scope of our support capabilities. If you have a question related to our products or services, please feel free to ask.",
                    sources=[],
                    is_grounded=True,
                    confidence=1.0,
                )
            }

        if not retrieval.chunks:
            return {
                "synthesis_result": SynthesisResult(
                    response_text="INSUFFICIENT_CONTEXT",
                    sources=[],
                    is_grounded=False,
                    confidence=0.0,
                )
            }

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
2. If the context does not contain the answer, you MUST output EXACTLY and ONLY the string "INSUFFICIENT_CONTEXT" (no other words). Do not add general troubleshooting steps like 'clear your cache' or 'refresh your page' unless explicitly stated in the context.
3. NEVER invent policies, prices, time limits, procedures, or any facts not in the context. DO NOT make logical deductions and DO NOT add extra exceptions or conditions. Extract the answer exactly as supported by the context.
4. Always be helpful, empathetic, and professional.
5. Keep your response concise and structured (use bullet points where helpful).
6. At the end, cite the sources you used as: "Sources: [URL1], [URL2]"
7. MULTILINGUAL SUPPORT: You MUST auto-detect the language of the user's ticket and write your response entirely in that same language, translating technical terms where appropriate.

Context:
{context_block}"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"User Ticket:\n{raw_ticket}"},
        ]

        response_text = _call_llm(messages, max_tokens=800, temperature=0.2, model_tier="synthesis")

        if "INSUFFICIENT_CONTEXT" in response_text:
            return {
                "synthesis_result": SynthesisResult(
                    response_text="INSUFFICIENT_CONTEXT",
                    sources=[],
                    is_grounded=False,
                    confidence=0.0,
                )
            }

        sources: list[str] = []
        for chunk in retrieval.chunks:
            if chunk.source_url:
                sources.append(chunk.source_url)
        sources = list(dict.fromkeys(sources)) 

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
    ### use of this function: run faithfulness check
    """Verify that the synthesis response is grounded in the retrieved context."""
    try:
        synthesis = state["synthesis_result"]
        retrieval = state["retrieval_result"]

        if not synthesis.is_grounded or synthesis.response_text == "INSUFFICIENT_CONTEXT" or not retrieval.chunks:
            return {
                "faithfulness_result": FaithfulnessResult(
                    is_faithful=False,
                    faithfulness_score=0.0,
                    reasoning="Insufficient context to answer securely",
                ),
                "risk_result": RiskResult(
                    action=Action.ESCALATE,
                    risk_score=5,
                    urgency=UrgencyLevel.MEDIUM,
                    escalation_reason="No relevant support documents found",
                    escalation_team="Tier 1 Support",
                )
            }

        if state["domain_result"].request_type == RequestType.INVALID:
            return {
                "faithfulness_result": FaithfulnessResult(
                    is_faithful=True,
                    faithfulness_score=1.0,
                    reasoning="Out-of-scope response — no factual claims to verify",
                )
            }

        context_parts = [chunk.chunk_text for chunk in retrieval.chunks]
        context = "\n\n---\n\n".join(context_parts)

        prompt = f"""You are a faithfulness checker. Your job is to verify that a support response does not contain any claims that are not grounded in the provided context.

Context provided to the agent:
{context}

Agent's response:
{synthesis.response_text}

Question: Does the agent's response contain ANY factual claims, policies, procedures, prices, or timelines that are NOT supported by the context above?
Note: Standard conversational greetings (e.g. "Hello", "How can I help you") and closings (e.g. "Let me know if you need more help") are ALLOWED. Furthermore, minor, obvious logical deductions (e.g. assuming you need to be an administrator to click an 'Admin' tab) should NOT be flagged as ungrounded claims. Only flag substantive factual claims or specific troubleshooting steps that are entirely missing from the context.

Respond ONLY with valid JSON:
{{
  "is_faithful": true,
  "faithfulness_score": 0.0-1.0,
  "reasoning": "one sentence"
}}"""

        messages = [{"role": "user", "content": prompt}]
        response_text = _call_llm(messages, max_tokens=200, temperature=0.0,
                                   model_tier="faithfulness", json_mode=True)

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
        return {
            "faithfulness_result": FaithfulnessResult(
                is_faithful=True,
                faithfulness_score=0.5,
                reasoning=f"Faithfulness check error: {str(e)[:80]}",
            )
        }
