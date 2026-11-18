import pytest

from decision_explanation import (
    DecisionContext,
    build_suggestion,
    render_decision,
    render_suggestion,
)
from llm_communication import (
    build_communication_payload,
    polish_with_template_fallback,
    validate_communication_response,
)


def test_insufficient_data_explanation_is_fail_closed():
    context = DecisionContext(
        decision="NO_TRADE",
        reason="INSUFFICIENT_DATA",
        strategy="M1 breakout",
        symbol="XAUUSDm",
        regime="strong_uptrend",
        closed_trades=12,
    )
    text = render_decision(context)

    assert "only 12 closed demo trades" in text
    assert "The minimum is 30" in text
    assert "will not place a trade" in text


def test_trade_approved_explanation_reports_facts_only():
    context = DecisionContext(
        decision="TRADE_APPROVED",
        reason="RISK_GUARDS_CLEAR",
        strategy="Trend pullback",
        symbol="XAUUSDm",
        h1_trend=1,
        h1_adx=31.2,
        spread_ratio=0.8,
        confidence=0.72,
    )

    text = render_decision(context)

    assert "H1 was bullish" in text
    assert "ADX was 31.20" in text
    assert "72.0%" in text


def test_suggestion_collects_data_when_below_minimum():
    suggestion = build_suggestion([{
        "closed_trades": 18,
        "net_usd": 2.4,
    }])

    text = render_suggestion(suggestion)

    assert suggestion["suggestion"] == "COLLECT_MORE_DATA"
    assert "18 closed trades" in text
    assert "cannot execute" in text


def test_empty_suggestion_is_advisory_and_safe():
    suggestion = build_suggestion([])

    assert suggestion["suggestion"] == "COLLECT_MORE_DATA"
    assert suggestion["advisory"] is True


def test_llm_payload_has_no_tools_or_secrets():
    context = DecisionContext(
        decision="NO_TRADE",
        reason="INSUFFICIENT_DATA",
        closed_trades=12,
    )

    payload = build_communication_payload(context, "Why did you stop?")

    assert "tool" not in payload["system"].lower()
    assert "api_key" not in payload["system"].lower()
    assert "Never execute" in payload["system"]


def test_llm_action_response_is_rejected_and_template_is_used():
    context = DecisionContext(
        decision="NO_TRADE",
        reason="INSUFFICIENT_DATA",
        strategy="M1 breakout",
        symbol="XAUUSDm",
        closed_trades=12,
    )
    template = render_decision(context)
    response = '{"headline":"Place a buy now","summary":"Action","facts_used":[],"warnings":[]}'

    assert validate_communication_response(context, response) is None
    assert polish_with_template_fallback(context, template, response) == template


def test_llm_json_response_must_match_fixed_shape():
    context = DecisionContext(
        decision="NO_TRADE",
        reason="INSUFFICIENT_DATA",
    )

    assert validate_communication_response(context, '{"headline":"Wait","summary":"No evidence","facts_used":[],"warnings":[],"extra":true}') is None


def test_valid_llm_json_is_polished_but_cannot_execute():
    context = DecisionContext(
        decision="NO_TRADE",
        reason="INSUFFICIENT_DATA",
    )
    template = render_decision(context)
    response = '{"headline":"Insufficient evidence","summary":"I am staying inactive until more closed trades exist.","facts_used":["closed_trades"],"warnings":["No model edge claim"]}'

    polished = polish_with_template_fallback(context, template, response)

    assert "Insufficient evidence" in polished
    assert "staying inactive" in polished
