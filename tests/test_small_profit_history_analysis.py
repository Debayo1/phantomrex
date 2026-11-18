from datetime import datetime, timezone
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from research.analyze_small_profit_history import build_trade_records, render_report


def _deal(**overrides):
    values = {
        "symbol": "XAUUSDm",
        "magic": 101007,
        "position_id": 123,
        "time": int(datetime(2026, 10, 28, 23, 10, tzinfo=timezone.utc).timestamp()),
        "entry": 0,
        "type": 0,
        "order": 500,
        "volume": 0.01,
        "price": 2700.0,
        "profit": 0.0,
        "commission": -0.02,
        "swap": 0.0,
        "fee": 0.0,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_history_joins_position_with_signal_and_entry_minute_candles():
    entry_time = int(
        datetime(2026, 10, 28, 23, 10, tzinfo=timezone.utc).timestamp()
    )
    exit_time = int(
        datetime(2026, 10, 28, 23, 12, tzinfo=timezone.utc).timestamp()
    )
    deals = [
        _deal(time=entry_time, price=2700.1),
        _deal(
            time=exit_time,
            entry=1,
            type=1,
            price=2701.0,
            profit=0.9,
            commission=-0.02,
        ),
    ]
    rates = pd.DataFrame([
        {
            "time": pd.Timestamp("2026-10-28T23:09:00Z"),
            "open": 2699.0,
            "high": 2701.0,
            "low": 2698.0,
            "close": 2700.0,
        },
        {
            "time": pd.Timestamp("2026-10-28T23:10:00Z"),
            "open": 2700.0,
            "high": 2701.0,
            "low": 2699.8,
            "close": 2700.8,
        },
        {
            "time": pd.Timestamp("2026-10-28T23:11:00Z"),
            "open": 2700.8,
            "high": 2701.2,
            "low": 2700.5,
            "close": 2701.0,
        },
    ])
    ticks = np.array(
        [(entry_time * 1000, 2699.9, 2700.0), (entry_time * 1000 + 1000, 2701.0, 2701.1)],
        dtype=[
            ("time_msc", "i8"),
            ("bid", "f8"),
            ("ask", "f8"),
        ],
    )

    [trade] = build_trade_records(
        deals,
        {"XAUUSDm": rates},
        {"XAUUSDm": 0.01},
        {
            500: SimpleNamespace(
                sl=2698.1,
                tp=2701.1,
                time_setup_msc=entry_time * 1000 - 20,
                time_done_msc=entry_time * 1000,
            )
        },
        {("XAUUSDm", 123): ticks},
    )

    assert trade["strategy"] == "Trend pullback"
    assert trade["direction"] == "BUY"
    assert trade["utc_hour"] == 23
    assert trade["signal_body_fraction"] == pytest.approx(1 / 3)
    assert trade["entry_minute_body_fraction"] == pytest.approx(2 / 3)
    assert trade["signal_to_fill_gap_points"] == pytest.approx(10)
    assert trade["hold_seconds"] == 120
    assert trade["net_usd"] == pytest.approx(0.86)
    assert trade["entry_candle_is_full_minute"] is True
    assert trade["execution_latency_ms"] == 20
    assert trade["mae_price"] == pytest.approx(0.2)
    assert trade["mfe_price"] == pytest.approx(0.9)
    assert trade["mae_r"] == pytest.approx(0.1)
    assert trade["mfe_r"] == pytest.approx(0.45)
    report = render_report([trade])
    assert "Candle shape and excursions by outcome" in report
    assert "median order setup-to-done:** 20 ms" in report


def test_history_report_discloses_entry_candle_and_fill_gap_limitations():
    report = render_report([], ["M1 bars unavailable."])

    assert "complete entry-minute candle is only known after that minute ends" in report
    assert "not pure broker slippage" in report
    assert "M1 bars unavailable." in report
