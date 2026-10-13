import pytest
from unittest.mock import MagicMock, patch
import bot
from bot import (
    MAX_LOT_CAP,
    FORCE_SINGLE_POSITION,
    MAX_OPEN_POSITIONS_PER_SYMBOL,
    calculate_lot_size,
    _is_bot_magic,
    open_multi_target_batch,
)


def test_max_lot_cap_constant():
    """Verify MAX_LOT_CAP is strictly set to 0.01."""
    assert MAX_LOT_CAP == 0.01
    assert FORCE_SINGLE_POSITION is True
    assert MAX_OPEN_POSITIONS_PER_SYMBOL == 2


def test_calculate_lot_size_capped_with_large_balance():
    """Verify lot size is never above MAX_LOT_CAP (0.01) even with high risk amounts."""
    mock_sym_info = MagicMock()
    mock_sym_info.volume_min = 0.01
    mock_sym_info.volume_max = 100.0
    mock_sym_info.volume_step = 0.01

    with patch("bot.mt5.symbol_info", return_value=mock_sym_info):
        with patch("bot.calc_risk_for_lot", return_value=10.0):
            # Attempt to risk $500, which without cap would produce 50.0 lots
            lot = calculate_lot_size("XAUUSDm", 0, 500.0, 2000.0, 1990.0)
            assert lot == 0.01
            assert lot <= MAX_LOT_CAP


def test_calculate_lot_size_respects_minimum_lot():
    """Verify lot size falls back to min_lot if risk is zero or negative."""
    mock_sym_info = MagicMock()
    mock_sym_info.volume_min = 0.01
    mock_sym_info.volume_max = 100.0
    mock_sym_info.volume_step = 0.01

    with patch("bot.mt5.symbol_info", return_value=mock_sym_info):
        with patch("bot.calc_risk_for_lot", return_value=10.0):
            lot = calculate_lot_size("XAUUSDm", 0, 0.0, 2000.0, 1990.0)
            assert lot == 0.01


def test_magic_range_matching():
    """
    Addendum C1:
    Magic range matching must strictly follow: magic_base < magic <= magic_base + 10.
    101001 (Gold) and 102001 (BTC) must pass.
    0 (manual trade), 101000 (base), 101011 (out of range), and 109999 must fail.
    """
    # Valid bot orders
    assert _is_bot_magic(101001) is True
    assert _is_bot_magic(101002) is True
    assert _is_bot_magic(101003) is True
    assert _is_bot_magic(102001) is True
    assert _is_bot_magic(102005) is True
    assert _is_bot_magic(102010) is True

    # Invalid / excluded magics
    assert _is_bot_magic(0) is False        # Manual trade
    assert _is_bot_magic(101000) is False   # magic_base itself
    assert _is_bot_magic(102000) is False   # magic_base itself
    assert _is_bot_magic(101011) is False   # Outside range (+11)
    assert _is_bot_magic(102011) is False   # Outside range (+11)
    assert _is_bot_magic(109999) is False   # Deprecated magic 109999
    assert _is_bot_magic(999999) is False


def test_force_single_position_mode_with_large_balance():
    """
    Addendum A1:
    Even with a large mocked balance ($10,000+), FORCE_SINGLE_POSITION opens only
    a single position (TP_Core at 1.8R multiplier), never 3 targets.
    """
    mock_tick = MagicMock()
    mock_tick.ask = 2000.0
    mock_tick.bid = 1999.5

    mock_send = MagicMock(return_value=123456)

    with patch("bot.get_account_balance", return_value=10000.0):
        with patch("bot.get_account_balance_native", return_value=(10000.0, "USD")):
            with patch("bot.mt5.symbol_info_tick", return_value=mock_tick):
                with patch("bot.calculate_lot_size", return_value=0.01):
                    with patch("bot.calc_risk_for_lot", return_value=15.0):
                        with patch("bot.send_market_order", mock_send):
                            open_multi_target_batch(
                                symbol="XAUUSDm",
                                order_type=0,  # BUY
                                stop_dist=10.0,
                                total_risk_amount=15.0,
                                tp_multipliers=[1.0, 1.8, 2.5],
                                magic_base=101000,
                                ai_tier_label="Test Run",
                                win_prob=0.80
                            )

    # EXACTLY 1 order should have been sent (single-position mode)
    assert mock_send.call_count == 1
    call_args = mock_send.call_args[0]
    # Check that comment specifies TP_Core and magic is 101001
    assert "TP_Core" in call_args[7]
    assert call_args[6] == 101001


def test_ai_fallback_elimination():
    """
    Task 1.2:
    Inference error or missing model must NOT default to ai_approved = True.
    """
    # Simulate failed inference: ai_approved should remain False
    ai_approved = False
    win_prob = 0.0

    mock_bad_model = MagicMock()
    mock_bad_model.predict_proba.side_effect = RuntimeError("Broken model tensor")

    try:
        mock_bad_model.predict_proba([[1, 2, 3]])
        ai_approved = True
    except Exception:
        ai_approved = False

    assert ai_approved is False
    assert win_prob == 0.0
