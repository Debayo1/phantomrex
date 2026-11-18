"""Explainable baseline market-context and regime classification."""

from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import ta


NY_TZ = ZoneInfo("America/New_York")
UTC_TZ = timezone.utc

SP_BASELINE = {
    "name": "SP-BASELINE",
    "strategies": ("Trend pullback", "Range fade", "M1 breakout"),
    "target_usd": 1.0,
    "max_loss_usd": 5.0,
    "max_hold_minutes": 30,
    "lot": 0.01,
    "magic_offsets": {"Trend pullback": 7, "Range fade": 8, "M1 breakout": 9},
}


def get_session(timestamp):
    timestamp = pd.Timestamp(timestamp)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")
    ny = timestamp.tz_convert(NY_TZ)
    hour = int(timestamp.hour)
    ny_hour = int(ny.hour)
    if 7 <= hour < 12:
        session = "london"
    elif 12 <= hour < 16:
        session = "london_new_york_overlap"
    elif 16 <= hour < 21:
        session = "new_york"
    else:
        session = "off_hours_asia_or_rollover"
    return {
        "utc_hour": hour,
        "ny_hour": ny_hour,
        "session": session,
    }


def _finite(value):
    value = float(value)
    return value if np.isfinite(value) else None


def classify_market_context(m1, m5, h1, tick=None):
    """Build an explainable context snapshot; never returns an entry decision."""
    m1 = pd.DataFrame(m1).copy().sort_values("time")
    m5 = pd.DataFrame(m5).copy().sort_values("time")
    h1 = pd.DataFrame(h1).copy().sort_values("time")

    if len(m1) < 2 or len(m5) < 55:
        return {
            "utc_hour": None,
            "ny_hour": None,
            "session": "unknown",
            "h1_trend": 0,
            "h1_adx": None,
            "m5_trend": 0,
            "m1_m5_agreement": 0,
            "atr": None,
            "atr_percentile": None,
            "volatility_bucket": "unknown",
            "spread_ratio": None,
            "regime": "UNKNOWN/UNSAFE",
            "data_quality": "insufficient_m1_or_m5",
        }

    m1_times = pd.to_datetime(m1["time"], unit="s", utc=True)
    signal_time = m1_times.iloc[-2]

    m5 = m5.copy()
    m5["ema20"] = ta.trend.EMAIndicator(m5["close"], window=20).ema_indicator()
    m5["ema50"] = ta.trend.EMAIndicator(m5["close"], window=50).ema_indicator()
    m5["atr"] = ta.volatility.AverageTrueRange(
        m5["high"], m5["low"], m5["close"], window=14
    ).average_true_range()
    m5_close = float(m5.iloc[-2]["close"])
    m5_ema20 = float(m5.iloc[-2]["ema20"])
    m5_ema50 = float(m5.iloc[-2]["ema50"])
    m5_trend = 1 if m5_ema20 > m5_ema50 and m5_close > m5_ema50 else (
        -1 if m5_ema20 < m5_ema50 and m5_close < m5_ema50 else 0
    )

    atr_values = m5["atr"].dropna().astype(float)
    atr = _finite(m5.iloc[-2]["atr"])
    atr_percentile = None
    if atr is not None and len(atr_values):
        atr_percentile = float((atr_values <= atr).mean())
    if atr_percentile is None:
        volatility_bucket = "unknown"
    elif atr_percentile < 0.25:
        volatility_bucket = "low"
    elif atr_percentile < 0.75:
        volatility_bucket = "normal"
    elif atr_percentile < 0.95:
        volatility_bucket = "high"
    else:
        volatility_bucket = "extreme"

    h1_trend = 0
    h1_adx = None
    if len(h1) >= 50:
        h1_frame = h1.copy()
        h1_frame["ema50"] = ta.trend.EMAIndicator(h1_frame["close"], window=50).ema_indicator()
        h1_frame["ema200"] = ta.trend.EMAIndicator(h1_frame["close"], window=200).ema_indicator()
        if len(h1_frame) >= 200:
            h1_row = h1_frame.iloc[-2]
            h1_trend = 1 if h1_row["ema50"] > h1_row["ema200"] else (
                -1 if h1_row["ema50"] < h1_row["ema200"] else 0
            )
        if len(h1_frame) >= 28:
            adx = ta.trend.ADXIndicator(
                h1_frame["high"], h1_frame["low"], h1_frame["close"], window=14
            ).adx()
            h1_adx = _finite(adx.iloc[-2])

    m1_close = float(m1.iloc[-2]["close"])
    m1_open = float(m1.iloc[-2]["open"])
    m1_trend = 1 if m1_close > m1_open else (-1 if m1_close < m1_open else 0)
    m1_m5_agreement = int(m1_trend == m5_trend and m1_trend != 0)

    spread_ratio = None
    if tick is not None and atr:
        spread = float(getattr(tick, "ask", 0.0) - getattr(tick, "bid", 0.0))
        spread_ratio = spread / atr

    if h1_adx is not None and h1_adx >= 25 and h1_trend != 0:
        trend_label = "strong_uptrend" if h1_trend == 1 else "strong_downtrend"
        regime = trend_label
    elif h1_trend == 0:
        regime = "range"
    else:
        regime = "weak_trend"
    if volatility_bucket == "extreme":
        regime = "extreme_volatility"

    session_context = get_session(signal_time)
    return {
        **session_context,
        "h1_trend": h1_trend,
        "h1_adx": h1_adx,
        "m5_trend": m5_trend,
        "m1_trend": m1_trend,
        "m1_m5_agreement": m1_m5_agreement,
        "m5_close": m5_close,
        "m5_ema20": m5_ema20,
        "m5_ema50": m5_ema50,
        "atr": atr,
        "atr_percentile": atr_percentile,
        "volatility_bucket": volatility_bucket,
        "spread_ratio": _finite(spread_ratio) if spread_ratio is not None else None,
        "regime": regime,
        "data_quality": "ok",
    }
