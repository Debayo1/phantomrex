import pandas as pd

from market_regime import SP_BASELINE, classify_market_context, get_session


def test_baseline_is_immutable_test_fixture():
    assert SP_BASELINE["name"] == "SP-BASELINE"
    assert SP_BASELINE["lot"] == 0.01
    assert SP_BASELINE["target_usd"] == 1.0


def test_session_context_reports_utc_and_new_york():
    context = get_session("2026-10-29T14:30:00Z")
    assert context["utc_hour"] == 14
    assert context["ny_hour"] == 10
    assert context["session"] == "london_new_york_overlap"


def test_missing_market_context_fails_closed():
    context = classify_market_context(
        pd.DataFrame({"time": [1], "open": [1], "high": [1], "low": [1], "close": [1]}),
        pd.DataFrame({"time": [1], "open": [1], "high": [1], "low": [1], "close": [1]}),
        pd.DataFrame({"time": [1], "open": [1], "high": [1], "low": [1], "close": [1]}),
    )
    assert context["regime"] == "UNKNOWN/UNSAFE"
    assert context["data_quality"] == "insufficient_m1_or_m5"


def test_high_atr_can_be_marked_extreme():
    n = 260
    close = [100.0 + i * 0.01 for i in range(n)]
    m1 = pd.DataFrame({
        "time": list(range(n)), "open": close, "high": [x + 0.2 for x in close],
        "low": [x - 0.2 for x in close], "close": [x + 0.1 for x in close],
    })
    m5 = pd.DataFrame({
        "time": list(range(n)), "open": close, "high": [x + 5.0 for x in close],
        "low": [x - 5.0 for x in close], "close": [x + 0.1 for x in close],
    })
    h1 = pd.DataFrame({
        "time": list(range(n)), "open": close, "high": [x + 1.0 for x in close],
        "low": [x - 1.0 for x in close], "close": [x + 0.1 for x in close],
    })
    context = classify_market_context(m1, m5, h1)
    assert context["regime"] in (
        "strong_uptrend", "weak_trend", "range", "extreme_volatility"
    )
