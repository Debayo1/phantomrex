import json

import pytest

from small_profit_ledger import (
    get_direction_summary,
    get_recent_closed,
    get_summary,
    record_bar,
    update_record,
)


def test_history_persists_signal_decision_and_closed_result(tmp_path):
    database = str(tmp_path / "small_profit.db")
    signal_id = record_bar(
        symbol="XAUUSDm",
        strategy="Trend pullback",
        signal_time="2026-10-29 10:00:00",
        status="SIGNAL",
        direction=1,
        signal_price=2700.0,
        signal_open=2699.0,
        signal_high=2701.0,
        signal_low=2698.0,
        signal_close=2700.0,
        spread_points=25,
        features={"rsi14": 54.0, "m5_ema20": 2695.0},
        db_path=database,
    )
    update_record(
        signal_id,
        "OPEN",
        db_path=database,
        position_id=1234,
        volume=0.01,
        entry_price=2700.1,
        stop_loss=2698.0,
        take_profit=2701.1,
        planned_risk_usd=2.5,
    )

    retry_id = record_bar(
        symbol="XAUUSDm",
        strategy="Trend pullback",
        signal_time="2026-10-29 10:00:00",
        status="SIGNAL",
        direction=1,
        signal_price=2700.0,
        features={"rsi14": 54.0},
        db_path=database,
    )
    assert retry_id == signal_id
    update_record(
        signal_id,
        "CLOSED",
        db_path=database,
        close_time="2026-10-29 10:02:00+00:00",
        close_price=2701.1,
        realized_net_usd=0.82,
        exit_reason="TP",
    )

    summary = get_summary(database)[0]
    direction_summary = get_direction_summary(database)[0]
    recent = get_recent_closed(db_path=database)

    assert summary["bars_seen"] == 1
    assert summary["signal_bars"] == 1
    assert summary["closed_trades"] == 1
    assert summary["wins"] == 1
    assert summary["net_usd"] == pytest.approx(0.82)
    assert summary["expectancy_usd"] == pytest.approx(0.82)
    assert direction_summary["direction"] == 1
    assert direction_summary["closed_trades"] == 1
    assert direction_summary["expectancy_usd"] == pytest.approx(0.82)
    assert recent[0]["exit_reason"] == "TP"
    assert recent[0]["realized_net_usd"] == pytest.approx(0.82)


def test_no_signal_bar_and_skip_reasons_are_counted(tmp_path):
    database = str(tmp_path / "small_profit.db")
    record_bar(
        symbol="BTCUSDm",
        strategy="Range fade",
        signal_time="2026-10-29 10:00:00",
        status="NO_SIGNAL",
        decision_reason="No candidate signal on closed M1 bar",
        db_path=database,
    )
    signal_id = record_bar(
        symbol="BTCUSDm",
        strategy="Range fade",
        signal_time="2026-10-29 10:01:00",
        status="SIGNAL",
        direction=-1,
        features={"rsi14": 77.0},
        db_path=database,
    )
    update_record(signal_id, "SKIPPED", db_path=database, decision_reason="News filter")

    summary = get_summary(database)[0]

    assert summary["bars_seen"] == 2
    assert summary["no_signal_bars"] == 1
    assert summary["signal_bars"] == 1
    assert summary["skipped"] == 1


def test_features_are_saved_as_json(tmp_path):
    import sqlite3

    database = str(tmp_path / "small_profit.db")
    record_bar(
        symbol="BTCUSDm",
        strategy="M1 breakout",
        signal_time="2026-10-29 10:00:00",
        status="SIGNAL",
        features={"rsi14": 63.5, "m5_ema50": 68200},
        db_path=database,
    )
    with sqlite3.connect(database) as connection:
        encoded = connection.execute(
            "SELECT features_json FROM small_profit_history"
        ).fetchone()[0]

    assert json.loads(encoded) == {"m5_ema50": 68200, "rsi14": 63.5}


def test_update_rejects_unapproved_columns(tmp_path):
    database = str(tmp_path / "small_profit.db")
    signal_id = record_bar(
        symbol="BTCUSDm",
        strategy="M1 breakout",
        signal_time="2026-10-29 10:00:00",
        status="SIGNAL",
        db_path=database,
    )

    with pytest.raises(ValueError, match="Unsupported"):
        update_record(signal_id, "CLOSED", db_path=database, magic=1234)
