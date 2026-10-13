import pytest
import pandas as pd
import numpy as np

from ai_features import prepare_indicators, H1_BAR_DURATION


def test_h1_shift_prevents_lookahead_leakage():
    """
    Phase 5.2 Feature Leakage Test:
    Ensures that an M5 candle only ever accesses H1 bars that have completed and closed.
    MT5 stamps bars with their OPEN time. An H1 bar opening at 10:00:00 closes at 11:00:00.
    Therefore, M5 bars between 10:00 and 10:55 must NOT see the 10:00 H1 bar, but rather
    the 09:00 H1 bar (which closed at 10:00).
    """
    base_time = pd.Timestamp("2026-01-01 00:00:00")
    # 250 H1 bars: flat downwards for 239 bars, then strong upward reversal at index 240
    h1_times = [base_time + pd.Timedelta(hours=i) for i in range(250)]
    h1_closes = [200.0 - i * 0.2 for i in range(240)] + [250.0 + i * 2.0 for i in range(10)]
    h1_highs = [c + 0.5 for c in h1_closes]
    h1_lows = [c - 0.5 for c in h1_closes]
    h1_opens = [c + 0.1 for c in h1_closes]

    df_high = pd.DataFrame({
        "time": h1_times,
        "open": h1_opens,
        "high": h1_highs,
        "low": h1_lows,
        "close": h1_closes,
    })

    target_h1_idx = 240
    target_h1_open_time = h1_times[target_h1_idx]      # e.g., Day 10 00:00:00
    target_h1_close_time = target_h1_open_time + pd.Timedelta(hours=1) # Day 10 01:00:00

    # Create M5 candles spanning around target_h1_open_time and target_h1_close_time
    m5_start = h1_times[target_h1_idx - 5]
    m5_end = h1_times[target_h1_idx + 2]

    m5_times = []
    curr = m5_start
    while curr <= m5_end:
        m5_times.append(curr)
        curr += pd.Timedelta(minutes=5)

    m5_closes = [150.0 + i * 0.1 for i in range(len(m5_times))]
    m5_highs = [c + 0.1 for c in m5_closes]
    m5_lows = [c - 0.1 for c in m5_closes]
    m5_opens = [c - 0.05 for c in m5_closes]

    df_low = pd.DataFrame({
        "time": m5_times,
        "open": m5_opens,
        "high": m5_highs,
        "low": m5_lows,
        "close": m5_closes,
        "tick_volume": [100] * len(m5_times)
    })

    # Run prepare_indicators in "fast" mode so the reversal flips trend immediately
    merged = prepare_indicators(df_low, df_high, trend_mode="fast")

    m5_bar_1005 = merged[merged["time"] == target_h1_open_time + pd.Timedelta(minutes=5)]
    assert not m5_bar_1005.empty

    m5_bar_1055 = merged[merged["time"] == target_h1_open_time + pd.Timedelta(minutes=55)]
    assert not m5_bar_1055.empty

    m5_bar_1100 = merged[merged["time"] == target_h1_close_time]
    assert not m5_bar_1100.empty

    trend_1005 = m5_bar_1005["h1_trend"].iloc[0]
    trend_1055 = m5_bar_1055["h1_trend"].iloc[0]
    trend_1100 = m5_bar_1100["h1_trend"].iloc[0]

    # While candle 240 is forming (10:05 to 10:55), M5 bars only see the prior closed candle (-1)
    assert trend_1005 == -1
    assert trend_1055 == -1
    # Only at 11:00, when candle 240 has completed and closed, does the trend flip to +1
    assert trend_1100 == 1


def test_no_future_candles_accessed():
    """
    Verifies that dropping the last 10 H1 candles alters indicators only for M5 bars
    after the cutoff, with zero ripple effect backward into previous M5 bars.
    """
    base_time = pd.Timestamp("2026-01-01 00:00:00")
    h1_times = [base_time + pd.Timedelta(hours=i) for i in range(250)]
    h1_closes = [100.0 + i * 1.0 for i in range(250)]

    df_high = pd.DataFrame({
        "time": h1_times,
        "open": h1_closes,
        "high": [c + 1 for c in h1_closes],
        "low": [c - 1 for c in h1_closes],
        "close": h1_closes
    })

    m5_times = [base_time + pd.Timedelta(hours=200) + pd.Timedelta(minutes=5*i) for i in range(200)]
    m5_closes = [300.0 + i * 0.1 for i in range(len(m5_times))]
    df_low = pd.DataFrame({
        "time": m5_times,
        "open": m5_closes,
        "high": [c + 0.5 for c in m5_closes],
        "low": [c - 0.5 for c in m5_closes],
        "close": m5_closes
    })

    merged_full = prepare_indicators(df_low, df_high)

    df_high_truncated = df_high.iloc[:240].copy()
    merged_trunc = prepare_indicators(df_low, df_high_truncated)

    cutoff_time = df_high_truncated["time"].max()
    sub_full = merged_full[merged_full["time"] < cutoff_time]
    sub_trunc = merged_trunc[merged_trunc["time"] < cutoff_time]

    assert len(sub_full) == len(sub_trunc)
    np.testing.assert_allclose(sub_full["h1_adx"].values, sub_trunc["h1_adx"].values)
    np.testing.assert_allclose(sub_full["h1_trend"].values, sub_trunc["h1_trend"].values)
