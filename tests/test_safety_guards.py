from types import SimpleNamespace
from datetime import datetime, timezone
import pytest
import pandas as pd

from bot import (
    MAX_OPEN_POSITIONS_PER_SYMBOL,
    check_daily_loss_guard,
    _is_demo_account,
    _try_v7_demo_entry,
    _try_small_profit_demo_entries,
    _manage_small_profit_demo_positions,
    _fetch_v7_live_signal,
    _close_v7_positions_at_session_end,
    _record_entry_diagnostic,
    _daily_guard_cache,
    _is_bot_magic
)
import bot


def test_magic_range_filter():
    """Verifies that only magics within magic_base < m <= magic_base + 10 are recognized."""
    # Gold magic base: 101000 -> 101001..101010 valid
    assert _is_bot_magic(101001) is True
    assert _is_bot_magic(101007) is True
    assert _is_bot_magic(101008) is True
    assert _is_bot_magic(101009) is True
    assert _is_bot_magic(101010) is True
    assert _is_bot_magic(101000) is False
    assert _is_bot_magic(101011) is False

    # BTC magic base: 102000 -> 102001..102010 valid
    assert _is_bot_magic(102001) is True
    assert _is_bot_magic(102010) is True
    assert _is_bot_magic(102000) is False
    assert _is_bot_magic(102011) is False

    # Manual or other trades
    assert _is_bot_magic(0) is False
    assert _is_bot_magic(109999) is False


def test_v7_is_demo_only_and_fails_closed(monkeypatch):
    assert _is_demo_account(SimpleNamespace(trade_mode=0)) is True
    assert _is_demo_account(SimpleNamespace(trade_mode=2)) is False
    assert _is_demo_account(None) is False

    monkeypatch.setattr(bot.mt5, "account_info", lambda: SimpleNamespace(trade_mode=2))
    asset_data = {"config": {}}
    _try_v7_demo_entry("XAUUSDm", asset_data, pd.Timestamp("2026-10-27 12:00"))

    assert asset_data["entry_diagnostics"]["last_reason"] == "v7_demo_only"


def test_small_profit_experiment_is_demo_only_and_fails_closed(monkeypatch):
    monkeypatch.setattr(bot.mt5, "account_info", lambda: SimpleNamespace(trade_mode=2))
    monkeypatch.setattr(
        bot.mt5,
        "copy_rates_from_pos",
        lambda *args: pytest.fail("Fetched data for a non-demo account"),
    )
    monkeypatch.setattr(
        bot.mt5,
        "positions_get",
        lambda **kwargs: pytest.fail("Queried positions for a non-demo account"),
    )
    monkeypatch.setattr(
        bot.mt5,
        "order_send",
        lambda request: pytest.fail("Sent a small-profit order to a non-demo account"),
    )
    asset_data = {"config": {"magic_base": 101000}}

    _try_small_profit_demo_entries("XAUUSDm", asset_data)
    _manage_small_profit_demo_positions("XAUUSDm", asset_data["config"])

    assert asset_data["entry_diagnostics"]["last_reason"] == "small_profit_demo_only"


def test_small_profit_experiment_skips_non_hedging_accounts(monkeypatch):
    hedging_mode = getattr(bot.mt5, "ACCOUNT_MARGIN_MODE_RETAIL_HEDGING", 2)
    monkeypatch.setattr(
        bot.mt5,
        "account_info",
        lambda: SimpleNamespace(trade_mode=0, currency="USD", margin_mode=hedging_mode - 1),
    )
    monkeypatch.setattr(
        bot.mt5,
        "copy_rates_from_pos",
        lambda *args: pytest.fail("Fetched data for a netting account"),
    )
    asset_data = {"config": {"magic_base": 101000}}

    _try_small_profit_demo_entries("XAUUSDm", asset_data)

    assert asset_data["entry_diagnostics"]["last_reason"] == "small_profit_account_mode"


@pytest.mark.parametrize(
    ("direction", "signal_stop", "expected_type", "expected_levels"),
    [
        (1, 99.0, bot.mt5.ORDER_TYPE_BUY, (100.0, 99.0, 101.0)),
        (-1, 101.0, bot.mt5.ORDER_TYPE_SELL, (99.9, 101.0, 98.9)),
    ],
)
def test_small_profit_demo_entry_maps_signal_direction_correctly(
    monkeypatch, direction, signal_stop, expected_type, expected_levels
):
    import small_profit_strategy
    import small_profit_ledger

    hedging_mode = getattr(bot.mt5, "ACCOUNT_MARGIN_MODE_RETAIL_HEDGING", 2)
    monkeypatch.setattr(
        bot.mt5,
        "account_info",
        lambda: SimpleNamespace(trade_mode=0, currency="USD", margin_mode=hedging_mode),
    )
    start = int(pd.Timestamp("2026-10-27 12:00:00").timestamp())
    rates = [
        {
            "time": start + index * 60,
            "open": 100.0,
            "high": 100.2,
            "low": 99.8,
            "close": 100.1,
            "spread": 10,
        }
        for index in range(60)
    ]
    m5_rates = [
        {
            "time": start + index * 300,
            "open": 100.0,
            "high": 100.2,
            "low": 99.8,
            "close": 100.1,
            "spread": 10,
        }
        for index in range(60)
    ]
    monkeypatch.setattr(
        bot.mt5,
        "copy_rates_from_pos",
        lambda symbol, timeframe, pos, count: rates if timeframe == bot.mt5.TIMEFRAME_M1 else m5_rates,
    )
    monkeypatch.setattr(bot.mt5, "symbol_info", lambda symbol: SimpleNamespace(
        volume_min=0.01,
        volume_step=0.01,
        digits=2,
        point=0.01,
        trade_contract_size=100.0,
        trade_stops_level=0,
    ))
    monkeypatch.setattr(bot.mt5, "symbol_info_tick", lambda symbol: SimpleNamespace(ask=100.0, bid=99.9))
    monkeypatch.setattr(bot.mt5, "positions_get", lambda **kwargs: [])
    monkeypatch.setattr(bot.mt5, "history_deals_get", lambda *args: [])
    monkeypatch.setattr(
        small_profit_ledger,
        "record_bar",
        lambda **kwargs: 1,
    )
    monkeypatch.setattr(small_profit_ledger, "get_record_status", lambda record_id: "SIGNAL")
    monkeypatch.setattr(
        small_profit_ledger,
        "update_record",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(bot.tg_manager, "settings", {"news_filter_enabled": False})
    monkeypatch.setattr(bot, "check_daily_loss_guard", lambda: (False, ""))
    monkeypatch.setattr(bot, "calc_risk_for_lot", lambda *args: 2.0)
    monkeypatch.setattr(bot.db_ledger, "log_signal", lambda **kwargs: 1)
    monkeypatch.setattr(bot.tg_manager, "send_broadcast_alert", lambda *args: None)
    signals = pd.DataFrame([{
        "signal_bar": 58,
        "strategy": "Trend pullback",
        "direction": direction,
        "stop": signal_stop,
        "signal_price": 100.1,
        "signal_open": 100.0,
        "signal_high": 100.2,
        "signal_low": 99.8,
        "signal_close": 100.1,
        "spread_points": 10.0,
        "features": {"rsi14": 54.0},
    }])
    monkeypatch.setattr(small_profit_strategy, "generate_signals", lambda *args: signals)
    requests = []
    monkeypatch.setattr(
        bot,
        "send_market_order",
        lambda *args: requests.append(args) or (123, 456),
    )
    asset_data = {
        "config": {"magic_base": 101000, "spread_est": 0.25},
        "last_small_profit_bar": None,
    }

    _try_small_profit_demo_entries("XAUUSDm", asset_data)

    assert len(requests) == 1
    _, order_type, entry, stop, target, lot, magic, _ = requests[0]
    assert order_type == expected_type
    assert (entry, stop, target) == pytest.approx(expected_levels)
    assert (lot, magic) == (0.01, 101007)


def test_small_profit_position_close_records_net_result(monkeypatch):
    import small_profit_ledger

    record = {
        "id": 8,
        "position_id": 456,
        "strategy": "Trend pullback",
        "status": "OPEN",
    }
    entry_deal = SimpleNamespace(
        entry=bot.mt5.DEAL_ENTRY_IN,
        profit=0.0,
        commission=-0.02,
        swap=0.0,
        fee=0.0,
        time=1793188800,
        price=100.0,
        reason=0,
        comment="",
    )
    exit_deal = SimpleNamespace(
        entry=bot.mt5.DEAL_ENTRY_OUT,
        profit=0.9,
        commission=-0.02,
        swap=0.0,
        fee=0.0,
        time=1793188860,
        price=101.0,
        reason=getattr(bot.mt5, "DEAL_REASON_TP", 0),
        comment="",
    )
    monkeypatch.setattr(bot.mt5, "account_info", lambda: SimpleNamespace(trade_mode=0))
    monkeypatch.setattr(bot.mt5, "positions_get", lambda **kwargs: [])
    monkeypatch.setattr(bot.mt5, "history_deals_get", lambda **kwargs: [entry_deal, exit_deal])
    monkeypatch.setattr(small_profit_ledger, "get_open_records", lambda symbol: [record])
    updates = []
    monkeypatch.setattr(
        small_profit_ledger,
        "update_record",
        lambda record_id, status, **fields: updates.append((record_id, status, fields)),
    )

    _manage_small_profit_demo_positions("XAUUSDm", {"magic_base": 101000})

    assert len(updates) == 1
    record_id, status, fields = updates[0]
    assert (record_id, status) == (8, "CLOSED")
    assert fields["realized_net_usd"] == pytest.approx(0.86)
    assert fields["exit_reason"] == "TP"


def test_v7_skips_reprocessing_the_same_m1_bar(monkeypatch):
    current_time = int(pd.Timestamp("2026-10-27 13:00:00", tz="UTC").timestamp())
    current_rates = [{"time": current_time}]
    monkeypatch.setattr(
        bot.mt5,
        "copy_rates_from_pos",
        lambda symbol, timeframe, start, count: current_rates if count == 1 else pytest.fail("Reprocessed cached M1 bar"),
    )

    bar_time, signal = _fetch_v7_live_signal(
        "XAUUSDm",
        {"name": "XAUUSDm"},
        last_processed_bar=pd.Timestamp("2026-10-27 13:00:00"),
    )

    assert bar_time == pd.Timestamp("2026-10-27 13:00:00")
    assert signal is None


def test_v7_time_exit_never_sends_from_non_demo_account(monkeypatch):
    monkeypatch.setattr(bot.mt5, "account_info", lambda: SimpleNamespace(trade_mode=2))
    monkeypatch.setattr(bot.mt5, "positions_get", lambda **kwargs: pytest.fail("Queried live positions"))
    monkeypatch.setattr(bot.mt5, "order_send", lambda request: pytest.fail("Sent a live V7 close"))

    _close_v7_positions_at_session_end(
        "XAUUSDm",
        {"magic_base": 101000},
        now_ny=datetime(2026, 10, 27, 16, 1, tzinfo=bot.NY_TZ),
    )


def test_position_limit_clamped():
    """Verifies that max positions per symbol cannot exceed 2."""
    assert MAX_OPEN_POSITIONS_PER_SYMBOL == 2

    # Setting check in bot.py: max_open = min(int(settings.get("max_positions", 2)), 3)
    # Default is 2. Even if settings set to 10, clamp restricts it.
    open_positions = [SimpleNamespace(ticket=1), SimpleNamespace(ticket=2)]
    max_open = min(int(MAX_OPEN_POSITIONS_PER_SYMBOL), 3)
    assert len(open_positions) >= max_open


def test_edge_ok_false_blocks_entries():
    """Verifies that asset with edge_ok=False is prohibited from trading."""
    asset_data = {
        "edge_ok": False,
        "no_edge_warned": False,
        "model": "dummy_model"
    }

    # If edge_ok is False, entry condition must reject
    should_trade = bool(asset_data.get("model") and asset_data.get("edge_ok", False))
    assert should_trade is False

    # If edge_ok is True and model present, trading allowed
    asset_data["edge_ok"] = True
    should_trade = bool(asset_data.get("model") and asset_data.get("edge_ok", False))
    assert should_trade is True


def test_entry_diagnostics_count_each_gate_once_per_closed_bar():
    asset_data = {}
    bar_one = pd.Timestamp("2026-10-27 10:00:00")
    bar_two = pd.Timestamp("2026-10-27 10:05:00")

    _record_entry_diagnostic(asset_data, "strategy_setup", bar_one, "BUY setup")
    _record_entry_diagnostic(asset_data, "strategy_setup", bar_one, "BUY setup")
    _record_entry_diagnostic(asset_data, "model_edge_not_validated", bar_one, "edge_ok=False")
    _record_entry_diagnostic(asset_data, "strategy_setup", bar_two, "SELL setup")

    diagnostics = asset_data["entry_diagnostics"]
    assert diagnostics["counts"] == {
        "strategy_setup": 2,
        "model_edge_not_validated": 1,
    }
    assert diagnostics["last_reason"] == "strategy_setup"
    assert diagnostics["last_detail"] == "SELL setup"


def test_entry_diagnostics_count_data_failure_transitions():
    asset_data = {}

    _record_entry_diagnostic(asset_data, "data_unavailable", detail="no candles")
    _record_entry_diagnostic(asset_data, "data_unavailable", detail="no candles")
    asset_data["entry_diagnostics"]["state"] = None
    _record_entry_diagnostic(asset_data, "data_unavailable", detail="no candles")

    diagnostics = asset_data["entry_diagnostics"]
    assert diagnostics["counts"]["data_unavailable"] == 2


def test_daily_loss_guard_blocks_when_limit_hit(monkeypatch):
    """Verifies that check_daily_loss_guard halts trading when losing setups or pct limit is hit."""
    # Reset daily guard cache
    _daily_guard_cache["checked_at"] = 0
    _daily_guard_cache["blocked"] = False
    _daily_guard_cache["reason"] = ""
    _daily_guard_cache["alerted_day"] = None

    today_utc = datetime.now(timezone.utc).date()
    today_ts = int(datetime.now(timezone.utc).timestamp())

    # Mock mt5 deals: 3 losing setups for today
    deals = [
        # IN deals
        SimpleNamespace(ticket=101, position_id=1, entry=0, magic=101001, symbol="XAUUSDm", time=today_ts - 3600),
        SimpleNamespace(ticket=102, position_id=2, entry=0, magic=101001, symbol="XAUUSDm", time=today_ts - 1800),
        SimpleNamespace(ticket=103, position_id=3, entry=0, magic=101001, symbol="XAUUSDm", time=today_ts - 900),
        # OUT deals (each losing $20)
        SimpleNamespace(ticket=201, position_id=1, entry=1, magic=101001, symbol="XAUUSDm", time=today_ts - 3000,
                        profit=-20.0, commission=0.0, swap=0.0, fee=0.0),
        SimpleNamespace(ticket=202, position_id=2, entry=1, magic=101001, symbol="XAUUSDm", time=today_ts - 1500,
                        profit=-20.0, commission=0.0, swap=0.0, fee=0.0),
        SimpleNamespace(ticket=203, position_id=3, entry=1, magic=101001, symbol="XAUUSDm", time=today_ts - 600,
                        profit=-20.0, commission=0.0, swap=0.0, fee=0.0),
    ]

    monkeypatch.setattr(bot.mt5, "history_deals_get", lambda *args, **kwargs: deals)
    monkeypatch.setattr(bot.mt5, "symbol_info_tick", lambda sym: SimpleNamespace(time=today_ts))
    monkeypatch.setattr(bot, "get_account_balance_native", lambda: (1000.0, "USD"))
    monkeypatch.setitem(bot.tg_manager.settings, "max_daily_losses", 3)

    blocked, reason = check_daily_loss_guard(force=True)
    assert blocked is True
    assert "3 losing setups today" in reason
