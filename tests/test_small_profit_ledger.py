import json
import os
import sqlite3
from concurrent.futures import ThreadPoolExecutor

import pytest

import db_ledger
import small_profit_ledger
from small_profit_ledger import (
    get_candle_outcome_summary,
    get_direction_summary,
    get_recent_closed,
    get_regime_summary,
    get_summary,
    record_bar,
    update_record,
)


def test_ledgers_use_the_same_absolute_database_path():
    expected = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "signals.db")
    expected = os.path.abspath(expected)

    assert os.path.isabs(small_profit_ledger.DB_PATH)
    assert os.path.isabs(db_ledger.DB_PATH)
    assert small_profit_ledger.DB_PATH == expected
    assert db_ledger.DB_PATH == expected


def test_legacy_history_table_migrates_candle_and_excursion_columns(tmp_path):
    database = str(tmp_path / "small_profit.db")
    with sqlite3.connect(database) as connection:
        connection.execute("""
            CREATE TABLE small_profit_history (
                id INTEGER PRIMARY KEY,
                symbol TEXT NOT NULL,
                strategy TEXT NOT NULL,
                status TEXT NOT NULL,
                position_id INTEGER
            );
        """)

    small_profit_ledger.init_db(database)

    with sqlite3.connect(database) as connection:
        columns = {
            row[1]
            for row in connection.execute("PRAGMA table_info(small_profit_history)")
        }
    assert {
        "entry_candle_json",
        "mae_price",
        "mfe_price",
        "mae_r",
        "mfe_r",
        "hold_seconds",
    }.issubset(columns)


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
        features={
            "rsi14": 54.0,
            "m5_ema20": 2695.0,
            "signal_body_fraction": 0.6,
            "signal_upper_wick_fraction": 0.25,
            "signal_lower_wick_fraction": 0.15,
        },
        db_path=database,
    )
    update_record(
        signal_id,
        "OPEN",
        db_path=database,
        position_id=1234,
        volume=0.01,
        entry_price=2700.1,
        entry_candle_json=json.dumps({
            "open": 2700.0,
            "high": 2700.3,
            "low": 2699.9,
            "close": 2700.2,
            "body_fraction": 0.5,
        }),
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
        mae_price=0.5,
        mfe_price=1.5,
        mae_r=0.25,
        mfe_r=0.75,
        hold_seconds=120.0,
    )

    summary = get_summary(database)[0]
    direction_summary = get_direction_summary(database)[0]
    candle_summary = get_candle_outcome_summary(database)[0]
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
    assert candle_summary["outcome"] == "WIN"
    assert candle_summary["avg_signal_body_fraction"] == pytest.approx(0.6)
    assert candle_summary["avg_signal_upper_wick_fraction"] == pytest.approx(0.25)
    assert candle_summary["avg_mae_r"] == pytest.approx(0.25)
    assert candle_summary["avg_mfe_r"] == pytest.approx(0.75)
    assert candle_summary["avg_hold_seconds"] == pytest.approx(120.0)
    assert recent[0]["exit_reason"] == "TP"
    assert recent[0]["mae_r"] == pytest.approx(0.25)
    assert recent[0]["realized_net_usd"] == pytest.approx(0.82)


def test_regime_summary_groups_closed_net_results(tmp_path):
    database = str(tmp_path / "small_profit.db")
    signal_id = record_bar(
        symbol="XAUUSDm",
        strategy="Trend pullback",
        signal_time="2026-10-29 10:00:00",
        status="SIGNAL",
        direction=1,
        features={
            "baseline": "SP-BASELINE",
            "regime": "strong_uptrend",
            "session": "london_new_york_overlap",
            "volatility_bucket": "normal",
        },
        db_path=database,
    )
    update_record(
        signal_id,
        "CLOSED",
        db_path=database,
        close_time="2026-10-29 10:02:00+00:00",
        realized_net_usd=0.75,
    )

    [summary] = get_regime_summary(database)

    assert summary["symbol"] == "XAUUSDm"
    assert summary["strategy"] == "Trend pullback"
    assert summary["regime"] == "strong_uptrend"
    assert summary["session"] == "london_new_york_overlap"
    assert summary["volatility_bucket"] == "normal"
    assert summary["closed_trades"] == 1
    assert summary["wins"] == 1
    assert summary["net_usd"] == pytest.approx(0.75)
    assert summary["expectancy_usd"] == pytest.approx(0.75)


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


def test_concurrent_ledger_reads_and_writes_are_serialized(tmp_path):
    database = str(tmp_path / "small_profit.db")
    small_profit_ledger.init_db(database)

    def write_record(index):
        return record_bar(
            symbol="BTCUSDm",
            strategy="M1 breakout",
            signal_time=f"2026-10-29 10:{index:02d}:00",
            status="NO_SIGNAL",
            db_path=database,
        )

    def read_summary(_):
        return get_summary(database)

    with ThreadPoolExecutor(max_workers=8) as executor:
        writes = list(executor.map(write_record, range(20)))
        summaries = list(executor.map(read_summary, range(20)))

    assert len(set(writes)) == 20
    assert summaries[-1][0]["bars_seen"] == 20


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
