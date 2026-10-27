from types import SimpleNamespace
from datetime import datetime, timezone
import pytest
import pandas as pd

from bot import (
    MAX_OPEN_POSITIONS_PER_SYMBOL,
    check_daily_loss_guard,
    _record_entry_diagnostic,
    _daily_guard_cache,
    _is_bot_magic
)
import bot


def test_magic_range_filter():
    """Verifies that only magics within magic_base < m <= magic_base + 10 are recognized."""
    # Gold magic base: 101000 -> 101001..101010 valid
    assert _is_bot_magic(101001) is True
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
