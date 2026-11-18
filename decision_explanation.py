"""Deterministic, ledger-backed human-readable trading decisions."""

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Optional


MIN_CLOSED_TRADES = 30


@dataclass(frozen=True)
class DecisionContext:
    decision: str
    reason: str
    strategy: str = "Unknown"
    symbol: str = "Unknown"
    regime: str = "UNKNOWN/UNSAFE"
    session: str = "unknown"
    volatility_bucket: str = "unknown"
    h1_trend: Optional[int] = None
    h1_adx: Optional[float] = None
    atr: Optional[float] = None
    spread_ratio: Optional[float] = None
    closed_trades: int = 0
    minimum_closed_trades: int = MIN_CLOSED_TRADES
    confidence: Optional[float] = None
    facts: Dict[str, Any] = field(default_factory=dict)


def _trend_text(value):
    if value == 1:
        return "bullish"
    if value == -1:
        return "bearish"
    if value == 0:
        return "neutral"
    return "unavailable"


def _number(value, suffix=""):
    return "unavailable" if value is None else f"{value:.2f}{suffix}"


def render_decision(context: DecisionContext) -> str:
    """Render a factual sentence without changing trading state."""
    strategy = context.strategy
    symbol = context.symbol

    if context.decision == "TRADE_APPROVED":
        adx = _number(context.h1_adx)
        spread = _number(context.spread_ratio, " ATR")
        confidence = (
            f"{context.confidence * 100:.1f}%" if context.confidence is not None else "unavailable"
        )
        return (
            f"{symbol} {strategy} was approved. H1 was {_trend_text(context.h1_trend)}, "
            f"ADX was {adx}, spread was {spread}, and measured confidence was {confidence}. "
            "Existing risk guards remained active and no guard was overridden."
        )

    if context.reason == "INSUFFICIENT_DATA":
        return (
            f"I cannot evaluate {strategy} because only {context.closed_trades} closed demo "
            f"trades are available. The minimum is {context.minimum_closed_trades}, and the "
            f"current {symbol} regime is {context.regime}. I will not place a trade."
        )

    if context.reason in ("UNKNOWN/UNSAFE", "DATA_UNAVAILABLE", "STALE_DATA"):
        return (
            f"I cannot safely evaluate {strategy} because market or data quality is "
            f"{context.regime}. H1/M5 or required context is unavailable or stale. "
            "I will not place a trade."
        )

    if context.reason in ("NEWS_FILTER", "DAILY_LOSS_GUARD", "POSITION_LIMIT"):
        detail = {
            "NEWS_FILTER": "the economic-news shield",
            "DAILY_LOSS_GUARD": "the daily-loss guard",
            "POSITION_LIMIT": "the position limit",
        }[context.reason]
        return (
            f"{symbol} {strategy} was not entered because {detail} blocked the opportunity. "
            "Existing risk controls remain active."
        )

    if context.reason == "SPREAD_LIMIT":
        return (
            f"{symbol} {strategy} was not entered because spread is "
            f"{_number(context.spread_ratio, ' times ATR')} and exceeds the configured limit. "
            "I will not place a trade."
        )

    if context.reason == "VOLATILITY_LIMIT":
        return (
            f"{symbol} {strategy} was not entered because volatility is "
            f"{context.volatility_bucket}. The configured volatility limit blocked the opportunity."
        )

    if context.reason == "NO_VALID_SETUP":
        return (
            f"There is no valid {strategy} setup for {symbol} in the current market. "
            "I will not place a trade."
        )

    if context.reason == "MODEL_SHADOW":
        return (
            f"The {strategy} model is in shadow mode. It is observing {symbol} but does not "
            "currently control a broker order."
        )

    return (
        f"{symbol} {strategy} is not active. The recorded reason is {context.reason}. "
        "I will not place a trade."
    )


def build_suggestion(summary_rows: Iterable[Dict[str, Any]], topic: str = "research") -> Dict[str, Any]:
    """Build an advisory-only suggestion from ledger summary facts."""
    rows = list(summary_rows)
    if not rows:
        return {
            "suggestion": "COLLECT_MORE_DATA",
            "confidence": "insufficient",
            "headline": "Collect more demo trades",
            "summary": (
                "There are no small-profit ledger rows yet. Keep the system in shadow or "
                "demo collection mode until closed-trade evidence exists."
            ),
            "advisory": True,
        }

    closed = sum(int(row.get("closed_trades") or 0) for row in rows)
    net = sum(float(row.get("net_usd") or 0.0) for row in rows)
    if closed < MIN_CLOSED_TRADES:
        return {
            "suggestion": "COLLECT_MORE_DATA",
            "confidence": "insufficient" if closed == 0 else "weak",
            "headline": "Collect more closed demo evidence",
            "summary": (
                f"The ledger currently contains {closed} closed trades, below the "
                f"{MIN_CLOSED_TRADES}-trade minimum. Net USD is {net:+.2f}. Keep strategies "
                "in demo/shadow collection and do not infer live readiness."
            ),
            "advisory": True,
        }

    positive = net > 0
    return {
        "suggestion": "CONTINUE_DEMO" if positive else "REVIEW_STRATEGY",
        "confidence": "moderate",
        "headline": "Continue demo validation" if positive else "Review weak strategy evidence",
        "summary": (
            f"The ledger contains {closed} closed trades with net USD {net:+.2f}. "
            "Continue chronological demo or shadow validation and require a fresh out-of-sample "
            "period before requesting owner approval."
        ),
        "advisory": True,
        "topic": topic,
    }


def render_suggestion(suggestion: Dict[str, Any]) -> str:
    """Render an advisory suggestion that cannot initiate an action."""
    return (
        f"Suggestion: {suggestion['headline']}.\n"
        f"{suggestion['summary']}\n"
        f"Evidence confidence: {suggestion['confidence']}. "
        "This is advisory and cannot execute, change risk, switch accounts, or promote a model."
    )
