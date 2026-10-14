"""
tests/test_v7_cr_model.py
Unit tests for Variant V7 (Addendum 8: 9am CR model).

1. Synthetic day where a short setup forms: check entry, stop, target, and R.
2. A day touching both sides in one bar is skipped.
3. No lookahead: 08:00 range is not used before 09:00, and FVG/OB logic uses only closed bars.
4. Time-zone test across DST transitions.
"""

import pytest
import pandas as pd
import numpy as np
import zoneinfo
from datetime import datetime, timedelta

from research.run_research import (
    simulate_v7_cr_model, convert_utc_to_ny, TARGET_SYMBOLS
)


def test_v7_dst_transition_timezone_conversion():
    """Test 4: Time-zone test across DST changes (spring forward and fall back)."""
    utc_tz = zoneinfo.ZoneInfo("UTC")
    ny_tz = zoneinfo.ZoneInfo("America/New_York")

    # 1. Spring Forward (US DST starts 2nd Sunday in March):
    # In 2026, DST starts March 8, 2026.
    # Before DST (EST = UTC-5): 13:00 UTC == 08:00 NY
    dt_winter_utc = datetime(2026, 3, 5, 13, 0, tzinfo=utc_tz)
    dt_winter_ny = convert_utc_to_ny(dt_winter_utc)
    assert dt_winter_ny.hour == 8, f"Expected 08:00 NY for winter UTC 13:00, got {dt_winter_ny}"
    assert dt_winter_ny.utcoffset() == timedelta(hours=-5)

    # After DST (EDT = UTC-4): 12:00 UTC == 08:00 NY
    dt_summer_utc = datetime(2026, 3, 15, 12, 0, tzinfo=utc_tz)
    dt_summer_ny = convert_utc_to_ny(dt_summer_utc)
    assert dt_summer_ny.hour == 8, f"Expected 08:00 NY for summer UTC 12:00, got {dt_summer_ny}"
    assert dt_summer_ny.utcoffset() == timedelta(hours=-4)

    # 2. Fall Back (US DST ends 1st Sunday in November):
    # In 2026, DST ends November 1, 2026.
    # Before fallback (EDT = UTC-4): 12:00 UTC == 08:00 NY
    dt_oct_utc = datetime(2026, 10, 25, 12, 0, tzinfo=utc_tz)
    dt_oct_ny = convert_utc_to_ny(dt_oct_utc)
    assert dt_oct_ny.hour == 8
    assert dt_oct_ny.utcoffset() == timedelta(hours=-4)

    # After fallback (EST = UTC-5): 13:00 UTC == 08:00 NY
    dt_nov_utc = datetime(2026, 11, 5, 13, 0, tzinfo=utc_tz)
    dt_nov_ny = convert_utc_to_ny(dt_nov_utc)
    assert dt_nov_ny.hour == 8
    assert dt_nov_ny.utcoffset() == timedelta(hours=-5)


def test_v7_dual_touch_single_bar_skipped():
    """Test 2: A day touching both sides of the 08:00 range in one bar is skipped."""
    utc_tz = zoneinfo.ZoneInfo("UTC")
    # Date in summer: EDT (UTC-4), 08:00 NY is 12:00 UTC
    base_utc = datetime(2026, 7, 15, 12, 0, tzinfo=utc_tz)

    times = [base_utc + timedelta(minutes=i) for i in range(120)]
    # Hour 08:00 NY is minutes 0 to 59 (UTC 12:00 to 12:59)
    # Range: Low = 2000.0, High = 2020.0
    opens = [2010.0] * 120
    highs = [2015.0] * 120
    lows = [2005.0] * 120
    closes = [2010.0] * 120

    # Set 08:00 hour range
    highs[10] = 2020.0
    lows[20] = 2000.0

    # Hour 09:00 NY is minutes 60 to 119 (UTC 13:00 to 13:59)
    # Minute 65 (09:05 NY): bar touches BOTH high (>=2020) and low (<=2000)
    highs[65] = 2025.0
    lows[65] = 1995.0

    df_m1 = pd.DataFrame({
        "time": [t.replace(tzinfo=None) for t in times],
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "spread": [25] * 120
    })

    sym_cfg = {"name": "XAUUSDm", "point": 0.001, "spread_est": 0.25, "contract_size": 100.0}
    stats = simulate_v7_cr_model(df_m1, sym_cfg)

    assert stats["valid_range_days"] == 1
    assert stats["dual_touch_days"] == 1
    assert len(stats["trades"]) == 0, "Dual touch day must produce zero trades!"


def test_v7_zero_lookahead_range_and_confirmations():
    """Test 3: No lookahead - 08:00 range is not used before 09:00, and FVG/OB logic uses only closed bars."""
    utc_tz = zoneinfo.ZoneInfo("UTC")
    base_utc = datetime(2026, 7, 15, 12, 0, tzinfo=utc_tz)

    # 1. Feed only data during the 08:00 hour (minutes 0 to 59)
    times_08 = [base_utc + timedelta(minutes=i) for i in range(50)]
    df_early = pd.DataFrame({
        "time": [t.replace(tzinfo=None) for t in times_08],
        "open": [2010.0] * 50,
        "high": [2015.0] * 50,
        "low": [2005.0] * 50,
        "close": [2010.0] * 50,
        "spread": [25] * 50
    })
    sym_cfg = {"name": "XAUUSDm", "point": 0.001, "spread_est": 0.25, "contract_size": 100.0}
    stats_early = simulate_v7_cr_model(df_early, sym_cfg)
    # Range is not complete and 09:00 has not been reached -> 0 trades
    assert len(stats_early["trades"]) == 0

    # 2. Add hour 09 data, but with NO closed confirming bar
    times_full = [base_utc + timedelta(minutes=i) for i in range(120)]
    opens = [2010.0] * 120
    highs = [2015.0] * 120
    lows = [2005.0] * 120
    closes = [2010.0] * 120
    highs[10] = 2020.0  # H_ref
    lows[20] = 2000.0   # L_ref
    # Touch at 09:05 (minute 65)
    highs[65] = 2022.0
    # But prices just oscillate with no FVG or OB confirmation
    df_no_confirm = pd.DataFrame({
        "time": [t.replace(tzinfo=None) for t in times_full],
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "spread": [25] * 120
    })
    stats_no_confirm = simulate_v7_cr_model(df_no_confirm, sym_cfg)
    assert stats_no_confirm["valid_range_days"] == 1
    assert stats_no_confirm["touch_days"] == 1
    assert stats_no_confirm["confirmed_days"] == 0
    assert len(stats_no_confirm["trades"]) == 0


def test_v7_synthetic_day_short_setup():
    """Test 1: Synthetic day where a short setup forms: verify entry, stop, target, and R."""
    utc_tz = zoneinfo.ZoneInfo("UTC")
    # In EDT, 08:00 NY is 12:00 UTC
    base_utc = datetime(2026, 7, 15, 12, 0, tzinfo=utc_tz)

    # 240 minutes = 4 hours (08:00 NY to 12:00 NY)
    times = [base_utc + timedelta(minutes=i) for i in range(240)]
    opens = [2010.0] * 240
    highs = [2012.0] * 240
    lows = [2008.0] * 240
    closes = [2010.0] * 240

    # 1. 08:00 NY Reference Range (minutes 0 to 59):
    # High = 2020.0, Low = 2000.0
    highs[15] = 2020.0
    lows[30] = 2000.0

    # 2. 09:00 NY Setup Window (minutes 60 to 119):
    # Minute 70 (09:10 NY): Touch High!
    highs[70] = 2022.0
    closes[70] = 2018.0

    # Minute 71 (09:11 NY): Small base
    opens[71] = 2018.0
    highs[71] = 2019.0
    lows[71] = 2017.0
    closes[71] = 2018.0

    # Bullish FVG Formation (candles at minute 72, 73, 74):
    # Candle 1 (minute 72): High = 2019.0
    opens[72] = 2018.0
    highs[72] = 2019.0
    lows[72] = 2017.5
    closes[72] = 2018.5

    # Candle 2 (minute 73): strong up move
    opens[73] = 2018.5
    highs[73] = 2025.0
    lows[73] = 2018.5
    closes[73] = 2024.0

    # Candle 3 (minute 74): Low = 2021.0 > Candle 1 High (2019.0) -> Bullish FVG [2019.0, 2021.0]
    # Also candle 3 is bullish: close 2025.0 > open 2024.0, Low = 2021.0 (Order Block candidate)
    opens[74] = 2024.0
    highs[74] = 2026.0
    lows[74] = 2021.0
    closes[74] = 2025.0

    # Minute 75 (09:15 NY): Strong Bearish confirming candle:
    # Closes at 2018.0:
    # 1. Below Candle 1 High (2019.0) -> IFVG CONFIRMED!
    # 2. Below Order Block Low (2021.0) -> ORDER BLOCK CONFIRMED!
    opens[75] = 2025.0
    highs[75] = 2025.0
    lows[75] = 2017.0
    closes[75] = 2018.0

    # Minute 76 (09:16 NY): ENTRY BAR!
    # Opens at 2018.0.
    opens[76] = 2018.0
    highs[76] = 2019.0
    lows[76] = 2014.0
    closes[76] = 2015.0

    # From minute 77 to 90, price trends down to target L_ref = 2000.0
    for m in range(77, 90):
        opens[m] = 2015.0 - (m - 77) * 1.5
        highs[m] = opens[m] + 0.5
        lows[m] = opens[m] - 1.5
        closes[m] = lows[m]

    # At minute 87, price reaches 1999.0 (target hit!)
    lows[87] = 1999.0
    closes[87] = 2000.0

    df_m1 = pd.DataFrame({
        "time": [t.replace(tzinfo=None) for t in times],
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "spread": [25] * 240  # 25 points = 0.025 on Gold with point=0.001
    })

    sym_cfg = {"name": "XAUUSDm", "point": 0.001, "spread_est": 0.25, "contract_size": 100.0}
    stats = simulate_v7_cr_model(df_m1, sym_cfg)

    assert stats["valid_range_days"] == 1
    assert stats["touch_days"] == 1
    assert stats["confirmed_days"] == 1
    assert len(stats["trades"]) == 1

    trade = stats["trades"][0]
    # Check entry price = open of bar 76 (2018.0)
    assert trade["entry_price"] == 2018.0
    assert trade["dir"] == -1
    assert trade["tp"] == 2000.0

    # Stop: highest high from touch (minute 70) to entry (minute 76) + spread
    # Highest high in [70..76] is 2026.0 (at minute 74)
    # Spread price = 0.25 -> SL = 2026.0 + 0.25 = 2026.25
    expected_sl = 2026.25
    expected_stop_dist = expected_sl - 2018.0  # 8.25
    assert abs(trade["sl"] - expected_sl) < 1e-4
    assert abs(trade["stop_dist"] - expected_stop_dist) < 1e-4

    # Target hit at 2000.0 -> Exit price = 2000.0
    assert trade["exit_price"] == 2000.0
    assert trade["exit_reason"] == "TP"

    # Gross R = (2018.0 - 2000.0) / 8.25 = 18.0 / 8.25 = 2.1818R
    expected_gross_r = 18.0 / expected_stop_dist
    assert abs(trade["gross_r"] - expected_gross_r) < 1e-4
    assert trade["net_r"] > 0


def test_v7_paper_logger_safety_and_logging(tmp_path, monkeypatch):
    """Test paper_logger_v7 safety (no orders) and CSV record formatting."""
    import ast
    paper_logger_path = "research/paper_logger_v7.py"

    with open(paper_logger_path, "r", encoding="utf-8") as f:
        tree = ast.parse(f.read())

    # AST check: no order_send
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute) and node.func.attr == "order_send":
                pytest.fail("Security violation: paper_logger_v7.py must never call order_send!")

    from research.paper_logger_v7 import record_paper_trade, ensure_paper_log_initialized
    import research.paper_logger_v7 as pl

    test_csv = tmp_path / "test_paper_log.csv"
    monkeypatch.setattr(pl, "PAPER_LOG_FILE", str(test_csv))

    record = {
        "ny_date": "2026-09-15",
        "symbol": "XAUUSDm",
        "h_ref": 2020.0,
        "l_ref": 2000.0,
        "touch_side": "HIGH",
        "touch_time": "2026-09-15 09:10:00",
        "confirmed": True,
        "entry_time": "2026-09-15 09:16:00",
        "entry_price": 2018.0,
        "sl": 2026.25,
        "tp": 2000.0,
        "stop_dist": 8.25,
        "exit_time": "2026-09-15 09:27:00",
        "exit_price": 2000.0,
        "exit_reason": "TP",
        "gross_r": 2.182,
        "net_r": 2.145,
        "status": "COMPLETED"
    }

    record_paper_trade(record)
    assert test_csv.exists()

    df = pd.read_csv(test_csv)
    assert len(df) == 1
    assert df.loc[0, "symbol"] == "XAUUSDm"
    assert df.loc[0, "exit_reason"] == "TP"
    assert df.loc[0, "status"] == "COMPLETED"
