import os
from types import SimpleNamespace
import pytest

from db_ledger import (
    init_db,
    log_signal,
    get_earliest_signal_timestamp,
    get_signal_by_position,
    compute_performance_stats,
    is_bot_magic,
)


def make_mock_deal(
    ticket=1001,
    order=2001,
    position_id=5001,
    magic=101001,
    entry=1,  # DEAL_ENTRY_OUT
    profit=30.0,
    commission=-1.5,
    swap=-0.5,
    fee=-0.2,
    symbol="XAUUSDm"
):
    return SimpleNamespace(
        ticket=ticket,
        order=order,
        position_id=position_id,
        magic=magic,
        entry=entry,
        profit=profit,
        commission=commission,
        swap=swap,
        fee=fee,
        symbol=symbol,
    )


def test_magic_range_filtering():
    """Verifies that magic range matching strictly includes 1..10 and excludes outliers."""
    # Valid bot magics: magic_base < magic <= magic_base + 10
    assert is_bot_magic(101001) is True   # Gold TP1/Core
    assert is_bot_magic(101002) is True   # Gold TP2
    assert is_bot_magic(101010) is True   # Gold boundary
    assert is_bot_magic(102001) is True   # BTC TP1/Core
    assert is_bot_magic(102010) is True   # BTC boundary

    # Invalid / manual magics
    assert is_bot_magic(0) is False        # Manual trade
    assert is_bot_magic(101000) is False   # Base itself
    assert is_bot_magic(101011) is False   # Outside range
    assert is_bot_magic(109999) is False   # Old un-ranged magic
    assert is_bot_magic(999999) is False


def test_deal_to_signal_join_on_position_id(tmp_path):
    """Verifies that closed deals join to signals.db on deal.position_id != deal.ticket."""
    test_db = str(tmp_path / "signals_test.db")
    init_db(test_db)

    # Signal logged at entry with position_ticket = 77777
    log_signal(
        symbol="XAUUSDm",
        direction=1,
        status="EXECUTED",
        position_ticket=77777,
        entry_price=2650.0,
        initial_sl=2635.0,
        initial_risk_price=15.0,
        initial_risk_currency=15.0,
        confidence=0.68,
        threshold=0.65,
        db_path=test_db,
    )

    # Deal has deal.ticket = 88888, but deal.position_id = 77777
    deals = [
        make_mock_deal(
            ticket=88888,
            order=99999,
            position_id=77777,
            magic=101001,
            entry=1,
            profit=27.0,
            commission=-1.5,
            swap=0.0,
            fee=0.0,
        )
    ]

    stats = compute_performance_stats(deals, signals_db_path=test_db)
    assert stats["total_closed"] == 1
    assert stats["matched_count"] == 1
    assert stats["unmatched_count"] == 0
    # Net result = 27.0 - 1.5 = 25.5
    assert pytest.approx(stats["expectancy_usd"], rel=1e-4) == 25.5
    # R-multiple = 25.5 / 15.0 = 1.70 R
    assert pytest.approx(stats["expectancy_r"], rel=1e-4) == 1.70
    assert stats["win_rate_pct"] == 100.0


def test_fee_commission_swap_summing_and_r_multiple(tmp_path):
    """Verifies exact summing of profit, commission, swap, and fee across entry and exit deals."""
    test_db = str(tmp_path / "signals_fees.db")
    init_db(test_db)

    log_signal(
        symbol="BTCUSDm",
        direction=-1,
        status="EXECUTED",
        position_ticket=90001,
        entry_price=68000.0,
        initial_sl=68500.0,
        initial_risk_price=500.0,
        initial_risk_currency=20.0,
        confidence=0.72,
        db_path=test_db,
    )

    # Entry deal: profit=0, comm=-2.0, fee=-0.5
    deal_in = make_mock_deal(
        ticket=101,
        position_id=90001,
        magic=102001,
        entry=0,  # DEAL_ENTRY_IN
        profit=0.0,
        commission=-2.0,
        swap=0.0,
        fee=-0.5,
        symbol="BTCUSDm",
    )
    # Exit deal: profit=36.0, comm=-2.0, swap=-1.5
    deal_out = make_mock_deal(
        ticket=102,
        position_id=90001,
        magic=102001,
        entry=1,  # DEAL_ENTRY_OUT
        profit=36.0,
        commission=-2.0,
        swap=-1.5,
        fee=0.0,
        symbol="BTCUSDm",
    )

    stats = compute_performance_stats([deal_in, deal_out], signals_db_path=test_db)
    assert stats["matched_count"] == 1
    # Net result = 0.0 - 2.0 - 0.5 + 36.0 - 2.0 - 1.5 = 30.0
    assert pytest.approx(stats["expectancy_usd"], rel=1e-4) == 30.0
    # R-multiple = 30.0 / 20.0 = 1.50 R
    assert pytest.approx(stats["expectancy_r"], rel=1e-4) == 1.50


def test_breakeven_trade_handling(tmp_path):
    """Verifies that a trade closing with tiny positive buffer is classified as BE."""
    test_db = str(tmp_path / "signals_be.db")
    init_db(test_db)

    log_signal(
        symbol="XAUUSDm",
        direction=1,
        status="EXECUTED",
        position_ticket=55555,
        initial_risk_currency=15.0,
        confidence=0.64,
        db_path=test_db,
    )

    deal = make_mock_deal(
        ticket=1001,
        position_id=55555,
        magic=101001,
        entry=1,
        profit=1.20,
        commission=-0.60,
        swap=0.0,
        fee=0.0,
    )

    stats = compute_performance_stats([deal], signals_db_path=test_db)
    assert stats["matched_count"] == 1
    # Net result = 1.20 - 0.60 = 0.60
    # R-multiple = 0.60 / 15.0 = 0.04 R
    assert pytest.approx(stats["expectancy_r"], rel=1e-4) == 0.04
    assert stats["exit_reasons"]["pct_be"] == 100.0


def test_unmatched_deal_tagging(tmp_path):
    """Verifies that deals without a matching signal row are labeled unmatched and excluded from R."""
    test_db = str(tmp_path / "signals_unmatched.db")
    init_db(test_db)

    # No signal logged for position 99999
    deal = make_mock_deal(
        ticket=1001,
        position_id=99999,
        magic=101001,
        entry=1,
        profit=50.0,
        commission=-1.0,
    )

    stats = compute_performance_stats([deal], signals_db_path=test_db)
    assert stats["total_closed"] == 1
    assert stats["matched_count"] == 0
    assert stats["unmatched_count"] == 1
    assert stats["exit_reasons"]["pct_unmatched"] == 100.0
    # Excluded from matched win rate and R calculations (no guessing)
    assert stats["win_rate_pct"] == 0.0
    assert stats["expectancy_r"] == 0.0


def test_confidence_calibration_table_thresholds(tmp_path):
    """Verifies that a calibration bucket requires >= 10 matched trades before displaying win rate."""
    test_db = str(tmp_path / "signals_calib.db")
    init_db(test_db)

    deals = []

    # Bucket 1: 60%-65% -> Create 9 trades (below 10 threshold)
    for i in range(1, 10):
        pos_id = 1000 + i
        log_signal(
            symbol="XAUUSDm",
            direction=1,
            status="EXECUTED",
            position_ticket=pos_id,
            initial_risk_currency=10.0,
            confidence=0.62,
            db_path=test_db,
        )
        deals.append(make_mock_deal(ticket=i, position_id=pos_id, magic=101001, entry=1, profit=10.0))

    # Bucket 2: 70%+ -> Create 10 trades (7 wins, 3 losses -> 70% win rate)
    for i in range(1, 11):
        pos_id = 2000 + i
        log_signal(
            symbol="XAUUSDm",
            direction=1,
            status="EXECUTED",
            position_ticket=pos_id,
            initial_risk_currency=10.0,
            confidence=0.75,
            db_path=test_db,
        )
        pnl = 15.0 if i <= 7 else -10.0
        deals.append(make_mock_deal(ticket=100 + i, position_id=pos_id, magic=101001, entry=1, profit=pnl))

    stats = compute_performance_stats(deals, signals_db_path=test_db)
    calib = {b["bucket"]: b for b in stats["calibration"]}

    # Bucket 60%–65% has 9 trades: must be Insufficient data
    b60 = calib["60%–65%"]
    assert b60["count"] == 9
    assert b60["status"] == "PENDING"
    assert b60["realized_win_rate"] is None
    assert "Insufficient data" in b60["display"]

    # Bucket 70%+ has 10 trades: must display realized win rate (70.0%)
    b70 = calib["70%+"]
    assert b70["count"] == 10
    assert b70["status"] == "CALIBRATED"
    assert b70["realized_win_rate"] == 70.0
    assert "70.0%" in b70["display"]
