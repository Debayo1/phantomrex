"""Grounded optional LLM communication. It has no execution or configuration tools."""

import json
from dataclasses import asdict, is_dataclass


ACTION_PHRASES = (
    "place an order", "send a buy", "send a sell", "buy now", "sell now",
    "change the lot", "increase the lot", "decrease the lot", "disable guard",
    "enable guard", "switch account", "go live", "promote the model", "rollback the model",
)


def build_communication_payload(context, question=None):
    """Build a fact-only prompt; secrets and tools are intentionally absent."""
    context_data = asdict(context) if is_dataclass(context) else dict(context)
    system = (
        "You are a fact-grounded trading communication assistant. "
        "Use only supplied facts. Never invent numbers or market forecasts. "
        "Never execute or recommend an order, lot, stop, target, account switch, guard change, "
        "or model promotion. Return JSON with exactly these keys: headline, summary, facts_used, warnings."
    )
    user = json.dumps({
        "question": question or "Explain the current decision.",
        "context": context_data,
    }, sort_keys=True)
    return {"system": system, "user": user}


def validate_communication_response(context, response):
    """Validate and return a grounded response or None."""
    if not isinstance(response, str):
        return None
    text = response.strip()
    if not text or len(text) > 1200:
        return None
    lower = text.lower()
    if any(phrase in lower for phrase in ACTION_PHRASES):
        return None

    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    if set(payload) != {"headline", "summary", "facts_used", "warnings"}:
        return None
    if not all(isinstance(payload[key], str) for key in ("headline", "summary")):
        return None
    if not isinstance(payload["facts_used"], list) or not isinstance(payload["warnings"], list):
        return None

    context_data = asdict(context) if is_dataclass(context) else dict(context)
    decision = str(context_data.get("decision", ""))
    reason = str(context_data.get("reason", ""))
    combined = f"{payload['headline']} {payload['summary']}".lower()
    if decision == "TRADE_APPROVED" and "not entered" in combined:
        return None
    if decision.startswith("NO_") and "approved" in combined:
        return None
    if reason == "INSUFFICIENT_DATA" and "edge is sufficient" in combined:
        return None
    return payload


def polish_with_template_fallback(context, template_text, llm_response=None):
    """Keep template text as the fail-safe output."""
    validated = validate_communication_response(context, llm_response)
    if validated is None:
        return template_text
    return f"{validated['headline']}\n{validated['summary']}"
