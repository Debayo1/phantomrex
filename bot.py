import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import time
import os
import joblib
import sys
from datetime import datetime, timezone
import socket

import threading
from ai_features import prepare_indicators, extract_single_feature_vector, FEATURE_COLUMNS, DEFAULT_TREND_MODE
from telegram_manager import TelegramBotManager, save_config, validate_credentials
from news_filter import NewsFilter
import db_ledger

_INSTANCE_LOCK_SOCKET = None

def acquire_single_instance_lock(port=48123):
    global _INSTANCE_LOCK_SOCKET
    try:
        _INSTANCE_LOCK_SOCKET = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        _INSTANCE_LOCK_SOCKET.bind(("127.0.0.1", port))
        _INSTANCE_LOCK_SOCKET.listen(1)
        return True
    except OSError:
        return False

# Force UTF-8 console output for Windows terminal emoji rendering
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


# ==============================================================================
# CONFIGURATION & MULTI-ASSET SETTINGS
# ==============================================================================

MODE = "LIVE"   # "LIVE" or "BACKTEST"

# Active assets to trade concurrently
ASSET_CONFIGS = {
    "XAUUSDm": {
        "model_file": "gold_ai_model.pkl",
        "min_stop_dist": 1.50,
        "spread_est": 0.25,
        "magic_base": 101000,
        "tp_multipliers": [1.0, 1.8, 3.0],
        "default_thresh": 0.60
    },
    "BTCUSDm": {
        "model_file": "btc_ai_model.pkl",
        "min_stop_dist": 100.0,
        "spread_est": 15.0,
        "magic_base": 102000,
        "tp_multipliers": [1.0, 1.8, 3.0],
        "default_thresh": 0.60
    }
}

TIMEFRAME_LOW = mt5.TIMEFRAME_M5    # Trigger & execution timeframe (5-min candles)
TIMEFRAME_HIGH = mt5.TIMEFRAME_H1   # Macro trend timeframe (1-hour candles)

# ONLY OWNER MAY CHANGE THIS CONSTANT
MAX_LOT_CAP = 0.01

# ONLY OWNER MAY CHANGE THIS CONSTANT
FORCE_SINGLE_POSITION = True

INITIAL_BALANCE = 1000.0            # Account balance fallback ($)
RISK_PER_SIGNAL = 0.015             # 1.5% total risk per setup (split across targets)
NUM_TARGETS = 3                     # Open targets (TP1, TP2, TP3 runner if not forced single)
ATR_MULTIPLIER = 1.5                # Stop loss distance multiplier (1.5x ATR)
MIN_ADX = 12                        # Minimum ADX trend strength filter
MAX_OPEN_POSITIONS_PER_SYMBOL = 2   # Max concurrent positions per asset (clamped to 2)

LIVE_POLL_INTERVAL = 2              # Seconds between polling loops
HEARTBEAT_INTERVAL = 10             # Seconds between console status lines

# --- Risk guardrails (overridable from telegram_config.json "settings") ---
DEFAULT_MAX_DAILY_LOSSES = 2        # Losing setups per day before the bot stops opening trades
DEFAULT_MAX_DAILY_LOSS_PCT = 3.0    # Max net daily loss (% of start-of-day balance)
DEFAULT_MAX_RISK_OVERSHOOT = 1.5    # Skip a trade if real risk > 1.5x the intended risk (e.g. forced 0.01 min lot)
DAILY_GUARD_CHECK_INTERVAL = 30     # Seconds between broker-history checks

# Global Telegram Bot Manager & Economic News Shield
tg_manager = TelegramBotManager()
news_filter = NewsFilter()
tg_manager.news_filter = news_filter


# ==============================================================================
# MT5 CONNECTION & SYMBOL DISCOVERY
# ==============================================================================

def connect_mt5():
    valid, reason = validate_credentials(tg_manager.config)
    if not valid:
        print(f"❌ Startup credential validation failed: {reason}")
        print("💡 Set valid credentials via .env or telegram_config.json before starting.")
        quit()

    if not mt5.initialize():
        print("❌ MT5 connection failed")
        print(mt5.last_error())
        quit()
    print("✅ Connected to MetaTrader 5")

    # Auto-login to configured account profile if available
    try:
        cfg = tg_manager.config
        active_profile = cfg.get("active_account", "demo")
        accounts = cfg.get("accounts", {})
        if active_profile in accounts:
            acc_data = accounts[active_profile]
            login_id = int(acc_data.get("login", 0))
            pwd = acc_data.get("password", "")
            srv = acc_data.get("server", "")
            if login_id and pwd and srv:
                current_acc = mt5.account_info()
                if not current_acc or current_acc.login != login_id:
                    print(f"🔐 Logging into configured MT5 account '{active_profile}' ({login_id} on {srv})...")
                    if mt5.login(login=login_id, password=pwd, server=srv):
                        print(f"✅ Successfully logged in to {active_profile.upper()} profile!")
                    else:
                        print(f"⚠️ Warning: MT5 login failed for profile '{active_profile}': {mt5.last_error()}")
    except Exception as e:
        print(f"⚠️ Error checking account profile: {e}")

    acc_info = mt5.account_info()
    if acc_info:
        mode_str = "DEMO 🧪" if acc_info.trade_mode == 0 else "REAL / LIVE 💰"
        print(f"💼 Active MT5 Account: #{acc_info.login} ({mode_str}) | Server: {acc_info.server} | Balance: {acc_info.balance:,.2f} {acc_info.currency}")
    else:
        print("⚠️ Warning: No active MT5 account info available.")


def resolve_broker_symbol(target_symbol):
    """
    Finds and selects the exact symbol name available on the broker
    (handling Exness 'm', 'c', standard, and suffix variants).
    """
    if mt5.symbol_select(target_symbol, True):
        return target_symbol

    clean_base = target_symbol.upper().rstrip("M").rstrip("C").rstrip("247")
    variants = [f"{clean_base}m", clean_base, f"{clean_base}c"]
    for var in variants:
        if mt5.symbol_select(var, True):
            return var

    all_symbols = mt5.symbols_get() or []
    match = next((s.name for s in all_symbols if clean_base.lower() in s.name.lower()), None)
    if match and mt5.symbol_select(match, True):
        return match

    return None


# ==============================================================================
# DATA PIPELINE
# ==============================================================================

def fetch_data(symbol, count_m5=300, count_h1=400, trend_mode=DEFAULT_TREND_MODE):
    rates_low = mt5.copy_rates_from_pos(symbol, TIMEFRAME_LOW, 0, count_m5)
    rates_high = mt5.copy_rates_from_pos(symbol, TIMEFRAME_HIGH, 0, count_h1)

    if rates_low is None or rates_high is None or len(rates_low) < 50 or len(rates_high) < 50:
        return None

    df_low = pd.DataFrame(rates_low)
    df_high = pd.DataFrame(rates_high)

    df = prepare_indicators(df_low, df_high, trend_mode=trend_mode)
    return df


# ==============================================================================
# INTELLIGENT POSITION & LOT SIZING
# ==============================================================================

def get_account_balance():
    info = mt5.account_info()
    if not info:
        return INITIAL_BALANCE
    bal = info.balance
    # If account is in Nigerian Naira, convert to USD equivalent for correct lot calculations
    if getattr(info, "currency", "USD") == "NGN" and bal > 0:
        bal = bal / 1600.0
    return bal


def get_account_balance_native():
    """Balance in the account's own currency (used for real risk maths)."""
    info = mt5.account_info()
    if not info:
        return INITIAL_BALANCE, "USD"
    return info.balance, getattr(info, "currency", "USD")


def calc_risk_for_lot(symbol, order_type, lot, entry, sl):
    """
    Real money lost (account currency) if this position hits its stop,
    computed by MT5 itself so contract size / tick value / currency are always right.
    """
    try:
        p = mt5.order_calc_profit(order_type, symbol, lot, entry, sl)
    except Exception:
        p = None
    if p is None:
        info = mt5.symbol_info(symbol)
        if not info:
            return None
        return abs(entry - sl) * lot * info.trade_contract_size
    return abs(p)


def calculate_lot_size(symbol, order_type, risk_amount, entry, sl):
    """
    Lot size so that a stop-out loses ~risk_amount (account currency).
    Rounds DOWN to the volume step so risk is never exceeded, except when the
    broker minimum lot forces it (that case is caught by the overshoot check).
    """
    info = mt5.symbol_info(symbol)
    if not info or abs(entry - sl) <= 0:
        return 0.01

    min_lot = info.volume_min
    max_lot = info.volume_max
    step_lot = info.volume_step

    risk_per_lot = calc_risk_for_lot(symbol, order_type, 1.0, entry, sl)
    if not risk_per_lot or risk_per_lot <= 0:
        return min_lot

    raw_lot = risk_amount / risk_per_lot
    lot_size = np.floor(raw_lot / step_lot + 1e-9) * step_lot
    lot_size = max(min_lot, min(max_lot, lot_size))

    # Hard-coded 0.01 lot cap (Addendum A1)
    lot_size = min(lot_size, MAX_LOT_CAP)

    return round(lot_size, 2)


# ==============================================================================
# ORDER EXECUTION & BROADCAST ALERTS
# ==============================================================================

def send_market_order(symbol, order_type, price, sl, tp, lot_size, magic, comment):
    info = mt5.symbol_info(symbol)
    filling_type = mt5.ORDER_FILLING_IOC
    if info:
        flags = info.filling_mode
        if flags & mt5.ORDER_FILLING_IOC:
            filling_type = mt5.ORDER_FILLING_IOC
        elif flags & mt5.ORDER_FILLING_FOK:
            filling_type = mt5.ORDER_FILLING_FOK
        elif flags & mt5.ORDER_FILLING_RETURN:
            filling_type = mt5.ORDER_FILLING_RETURN

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": lot_size,
        "type": order_type,
        "price": price,
        "sl": sl,
        "tp": tp,
        "deviation": 30,
        "magic": magic,
        "comment": comment,
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": filling_type,
    }

    result = mt5.order_send(request)
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        print(f"❌ Execution failed ({comment}): {result.comment} (code {result.retcode})")
        return None
    else:
        order_ticket = result.order
        # Phase 3.2 Position Identifier Resolution (Addendum B2)
        pos_id = None
        try:
            positions = mt5.positions_get(symbol=symbol) or []
            for p in positions:
                if getattr(p, "magic", None) == magic:
                    pos_id = p.ticket
                    break
            if not pos_id:
                deals = mt5.history_deals_get(ticket=order_ticket) or []
                if deals:
                    pos_id = deals[0].position_id
        except Exception:
            pass
        if not pos_id:
            pos_id = order_ticket

        print(f"🚀 {comment} Executed! Order #{order_ticket} (Position #{pos_id}) | Lot: {lot_size} | Entry: ${price:.2f} | SL: ${sl:.2f} | TP: ${tp:.2f}")
        return order_ticket, pos_id


def open_multi_target_batch(symbol, order_type, stop_dist, total_risk_amount, tp_multipliers, magic_base, ai_tier_label, win_prob, candle_metrics=None):
    """
    Carries the trade by splitting exposure into targets:
    - Normal balance (>= $100): Splits into 3 targets (TP1 Cash Lock, TP2 Core, TP3 Runner).
    - Small balance (< $100): Caps at 1 single position (0.01 lot) to strictly limit risk exposure!
    """
    balance = get_account_balance()
    # Force single position mode (TP_Core only) regardless of balance per Addendum A1
    if FORCE_SINGLE_POSITION or balance < 100.0:
        sub_risk = total_risk_amount
        active_multipliers = [1.8]
        active_labels = ["TP_Core"]
        mode_label = "Single-Position Mode" if FORCE_SINGLE_POSITION else "Small-Account Guard"
        print(f"\n⚡ [{mode_label}] Opening 1 Safe Single-Position Trade on {symbol} (Balance: ${balance:.2f} | Risk: ${total_risk_amount:.2f} | {ai_tier_label})...")
    else:
        sub_risk = total_risk_amount / NUM_TARGETS
        active_multipliers = tp_multipliers
        active_labels = ["TP1_Lock", "TP2_Core", "TP3_Runner"]
        print(f"\n⚡ Opening 3 Multi-Target Trades on {symbol} (Total Risk: ${total_risk_amount:.2f} | {ai_tier_label})...")

    tick = mt5.symbol_info_tick(symbol)
    if not tick:
        return

    _, ccy = get_account_balance_native()
    max_overshoot = float(tg_manager.settings.get("max_risk_overshoot", DEFAULT_MAX_RISK_OVERSHOOT))

    executed_tickets = []
    executed_labels = []
    actual_total_risk = 0.0
    skipped_msgs = []
    first_entry = 0.0
    first_sl = 0.0

    dir_str = "BUY 📈" if order_type == mt5.ORDER_TYPE_BUY else "SELL 📉"
    direction = 1 if order_type == mt5.ORDER_TYPE_BUY else -1

    for idx, (tp_mult, label) in enumerate(zip(active_multipliers, active_labels), start=1):
        tick = mt5.symbol_info_tick(symbol)
        if order_type == mt5.ORDER_TYPE_BUY:
            entry_p = tick.ask
            sl = entry_p - stop_dist
            tp = entry_p + (stop_dist * tp_mult)
        else:
            entry_p = tick.bid
            sl = entry_p + stop_dist
            tp = entry_p - (stop_dist * tp_mult)

        if idx == 1:
            first_entry = entry_p
            first_sl = sl

        lot = calculate_lot_size(symbol, order_type, sub_risk, entry_p, sl)

        # Real-risk guard: never let the broker minimum lot silently multiply risk
        real_risk = calc_risk_for_lot(symbol, order_type, lot, entry_p, sl)
        if real_risk is None or real_risk > sub_risk * max_overshoot:
            rr_txt = f"{real_risk:,.2f}" if real_risk is not None else "unknown"
            msg = (f"{label}: lot {lot} would risk {rr_txt} {ccy} vs allowed {sub_risk:,.2f} {ccy} "
                   f"(max {max_overshoot:.1f}x)")
            print(f"🛑 Real-Risk Guard skipped {symbol} {msg}")
            skipped_msgs.append(msg)
            # Log skipped signal
            try:
                db_ledger.log_signal(
                    symbol=symbol,
                    direction=direction,
                    status="SKIPPED",
                    skip_reason="Real-Risk Guard",
                    entry_price=entry_p,
                    initial_sl=sl,
                    initial_risk_price=abs(entry_p - sl),
                    initial_risk_currency=real_risk,
                    confidence=win_prob,
                    m5_rsi=candle_metrics.get("m5_rsi") if candle_metrics else None,
                    h1_adx=candle_metrics.get("h1_adx") if candle_metrics else None,
                    trend_mode=candle_metrics.get("trend_mode") if candle_metrics else None,
                    threshold=candle_metrics.get("threshold") if candle_metrics else None,
                )
            except Exception as ex:
                print(f"⚠️ Error logging skipped signal to db_ledger: {ex}")
            continue

        comment_str = f"AIBot_{label}"
        magic = magic_base + idx
        res = send_market_order(symbol, order_type, entry_p, sl, tp, lot, magic, comment_str)
        if res:
            if isinstance(res, tuple):
                t_id, pos_id = res
            else:
                t_id, pos_id = res, res
            executed_tickets.append(t_id)
            executed_labels.append(label)
            actual_total_risk += real_risk

            # Phase 3.1 & 3.2: Log executed signal with resolved position_ticket to signals.db
            try:
                spread_val = (tick.ask - tick.bid) if tick else 0.0
                db_ledger.log_signal(
                    symbol=symbol,
                    direction=direction,
                    status="EXECUTED",
                    position_ticket=pos_id,
                    entry_price=entry_p,
                    initial_sl=sl,
                    initial_risk_price=abs(entry_p - sl),
                    initial_risk_currency=real_risk,
                    m5_rsi=candle_metrics.get("m5_rsi") if candle_metrics else None,
                    h1_adx=candle_metrics.get("h1_adx") if candle_metrics else None,
                    trend_mode=candle_metrics.get("trend_mode") if candle_metrics else None,
                    confidence=win_prob,
                    threshold=candle_metrics.get("threshold") if candle_metrics else None,
                    spread=spread_val,
                    model_trained_date=candle_metrics.get("trained_date") if candle_metrics else None,
                )
            except Exception as ex:
                print(f"⚠️ Error logging executed signal to db_ledger: {ex}")

    if skipped_msgs and not executed_tickets:
        tg_manager.send_broadcast_alert(
            f"🛑 <b>Trade skipped by Real-Risk Guard: {symbol} {dir_str}</b>\n\n"
            + "\n".join(f"• {m}" for m in skipped_msgs)
            + "\n\n<i>The broker's minimum lot is too big for your risk setting at this stop size. "
              "Raise risk %, raise max_risk_overshoot, or fund the account more.</i>"
        )

    # Broadcast Telegram Alert to all authorized users
    if executed_tickets:
        tg_text = (
            f"🎯 <b>NEW AI TRADE EXECUTED: {symbol}</b>\n\n"
            f"• <b>Direction:</b> {dir_str}\n"
            f"• <b>Entry:</b> {first_entry:,.2f}\n"
            f"• <b>Stop Loss:</b> {first_sl:,.2f} ({stop_dist:,.2f} dist)\n"
            f"• <b>AI Win Confidence:</b> <b>{win_prob*100:.1f}%</b>\n"
            f"• <b>Tier:</b> {ai_tier_label}\n"
            f"• <b>Batches Open:</b> {len(executed_tickets)} target(s) ({', '.join(executed_labels)})\n"
            f"• <b>Real Risk if Stopped:</b> {actual_total_risk:,.2f} {ccy} "
            f"(target {total_risk_amount:,.2f} {ccy})"
        )
        tg_manager.send_broadcast_alert(tg_text)


# ==============================================================================
# SMART TRADE LIFECYCLE: AUTO-BREAKEVEN, TRAILING STOP & STRUCTURAL EXIT
# ==============================================================================

def manage_active_trades(symbol, cfg, current_candle, closed_candle=None):
    """
    Intelligent trade lifecycle manager:
    1. Moves SL to Break-Even when price is in profit (Risk = 0).
    2. Dynamic Trailing Stop on remaining runner positions to lock accrued profit.
    3. AI Structural Invalidation Exit: If market structure flips against the trade,
       cuts the loss early before full stop loss is hit!

    current_candle = live (forming) bar -> used only for the live price.
    closed_candle  = last fully-closed bar -> used for indicator decisions.
    """
    positions = mt5.positions_get(symbol=symbol)
    if not positions:
        return

    if closed_candle is None:
        closed_candle = current_candle

    curr_price = current_candle["close"]
    atr = closed_candle["m5_atr"]
    h1_trend = closed_candle["h1_trend"]
    e9 = closed_candle["m5_ema9"]
    e21 = closed_candle["m5_ema21"]
    point = mt5.symbol_info(symbol).point if mt5.symbol_info(symbol) else 0.01
    spread_buffer = max(cfg["spread_est"], point * 10)

    for pos in positions:
        # Hands-off manual trades: Only manage bot trades within magic range (magic_base < magic <= magic_base + 10)
        if not (cfg["magic_base"] < pos.magic <= cfg["magic_base"] + 10):
            continue

        ticket = pos.ticket
        entry = pos.price_open
        sl = pos.sl
        tp = pos.tp
        pos_type = pos.type
        comment = pos.comment

        orig_risk = abs(entry - sl) if sl > 0 else (atr * 1.5)

        # -------------------------------------------------------------
        # 1. AUTO-BREAKEVEN GUARD
        # -------------------------------------------------------------
        be_enabled = tg_manager.settings.get("breakeven_enabled", True)
        if be_enabled and pos_type == mt5.ORDER_TYPE_BUY:
            profit_dist = curr_price - entry

            if profit_dist >= (0.9 * orig_risk) and (sl < entry):
                new_sl = round(entry + spread_buffer, 2)
                req = {
                    "action": mt5.TRADE_ACTION_SLTP,
                    "position": ticket,
                    "sl": new_sl,
                    "tp": tp,
                }
                res = mt5.order_send(req)
                if res.retcode == mt5.TRADE_RETCODE_DONE:
                    print(f"🛡️ Auto-Breakeven Activated for {symbol} BUY #{ticket} @ ${new_sl:.2f} (Risk: $0.00)")
                    tg_manager.send_broadcast_alert(
                        f"🛡️ <b>Auto-Breakeven Locked:</b> {symbol} BUY #{ticket}\n"
                        f"Stop Loss moved to ${new_sl:.2f}. Trade risk is now eliminated ($0.00)!"
                    )

        # ---------------------------------------------------------
        # 2. DYNAMIC ATR TRAILING STOP ON RUNNERS
        # ---------------------------------------------------------
        trailing_enabled = tg_manager.settings.get("trailing_enabled", True)
        if trailing_enabled and pos_type == mt5.ORDER_TYPE_BUY:
            profit_dist = curr_price - entry
            if "Runner" in comment and profit_dist >= (1.5 * orig_risk):
                trail_sl = round(curr_price - (1.2 * atr), 2)
                if trail_sl > sl and trail_sl > entry:
                    req = {
                        "action": mt5.TRADE_ACTION_SLTP,
                        "position": ticket,
                        "sl": trail_sl,
                        "tp": tp,
                    }
                    res = mt5.order_send(req)
                    if res.retcode == mt5.TRADE_RETCODE_DONE:
                        print(f"📈 Trailing Stop Updated on {symbol} Runner #{ticket} -> ${trail_sl:.2f}")

        # ---------------------------------------------------------
        # 3. AI STRUCTURAL EARLY EXIT
        # ---------------------------------------------------------
        if pos_type == mt5.ORDER_TYPE_BUY:
            profit_dist = curr_price - entry
            if profit_dist < 0:
                if h1_trend == -1 and e9 < e21:
                    print(f"⚠️ AI Invalidation Exit: Market structure reversed against {symbol} BUY #{ticket}. Closing early!")
                    tick = mt5.symbol_info_tick(symbol)
                    close_req = {
                        "action": mt5.TRADE_ACTION_DEAL,
                        "symbol": symbol,
                        "volume": pos.volume,
                        "type": mt5.ORDER_TYPE_SELL,
                        "position": ticket,
                        "price": tick.bid if tick else curr_price,
                        "deviation": 30,
                        "magic": pos.magic,
                        "comment": "AI_Early_Invalidation_Exit"
                    }
                    res = mt5.order_send(close_req)
                    if res.retcode == mt5.TRADE_RETCODE_DONE:
                        tg_manager.send_broadcast_alert(
                            f"⚠️ <b>AI Invalidation Exit:</b> Closed {symbol} BUY #{ticket} early.\n"
                            f"Market structure flipped Bearish. Cut loss to protect equity!"
                        )

        elif pos_type == mt5.ORDER_TYPE_SELL:
            profit_dist = entry - curr_price

            be_enabled = tg_manager.settings.get("breakeven_enabled", True)
            if be_enabled and profit_dist >= (0.9 * orig_risk) and (sl > entry):
                new_sl = round(entry - spread_buffer, 2)
                req = {
                    "action": mt5.TRADE_ACTION_SLTP,
                    "position": ticket,
                    "sl": new_sl,
                    "tp": tp,
                }
                res = mt5.order_send(req)
                if res.retcode == mt5.TRADE_RETCODE_DONE:
                    print(f"🛡️ Auto-Breakeven Activated for {symbol} SELL #{ticket} @ ${new_sl:.2f} (Risk: $0.00)")
                    tg_manager.send_broadcast_alert(
                        f"🛡️ <b>Auto-Breakeven Locked:</b> {symbol} SELL #{ticket}\n"
                        f"Stop Loss moved to ${new_sl:.2f}. Trade risk is now eliminated ($0.00)!"
                    )

            # ---------------------------------------------------------
            # 2. DYNAMIC ATR TRAILING STOP ON RUNNERS
            # ---------------------------------------------------------
            trailing_enabled = tg_manager.settings.get("trailing_enabled", True)
            if trailing_enabled and "Runner" in comment and profit_dist >= (1.5 * orig_risk):
                trail_sl = round(curr_price + (1.2 * atr), 2)
                if (sl == 0 or trail_sl < sl) and trail_sl < entry:
                    req = {
                        "action": mt5.TRADE_ACTION_SLTP,
                        "position": ticket,
                        "sl": trail_sl,
                        "tp": tp,
                    }
                    res = mt5.order_send(req)
                    if res.retcode == mt5.TRADE_RETCODE_DONE:
                        print(f"📈 Trailing Stop Updated on {symbol} Runner #{ticket} -> ${trail_sl:.2f}")

            # ---------------------------------------------------------
            # 3. AI STRUCTURAL EARLY EXIT
            # ---------------------------------------------------------
            if profit_dist < 0:
                if h1_trend == 1 and e9 > e21:
                    print(f"⚠️ AI Invalidation Exit: Market structure reversed against {symbol} SELL #{ticket}. Closing early!")
                    tick = mt5.symbol_info_tick(symbol)
                    close_req = {
                        "action": mt5.TRADE_ACTION_DEAL,
                        "symbol": symbol,
                        "volume": pos.volume,
                        "type": mt5.ORDER_TYPE_BUY,
                        "position": ticket,
                        "price": tick.ask if tick else curr_price,
                        "deviation": 30,
                        "magic": pos.magic,
                        "comment": "AI_Early_Invalidation_Exit"
                    }
                    res = mt5.order_send(close_req)
                    if res.retcode == mt5.TRADE_RETCODE_DONE:
                        tg_manager.send_broadcast_alert(
                            f"⚠️ <b>AI Invalidation Exit:</b> Closed {symbol} SELL #{ticket} early.\n"
                            f"Market structure flipped Bullish. Cut loss to protect equity!"
                        )


# ==============================================================================
# DAILY LOSS GUARD
# ==============================================================================

_daily_guard_cache = {"checked_at": 0.0, "blocked": False, "reason": "", "alerted_day": None}


def _is_bot_magic(magic):
    return any(c["magic_base"] < magic <= c["magic_base"] + 10 for c in ASSET_CONFIGS.values())


def check_daily_loss_guard(force=False):
    """
    Returns (blocked, reason). Reads closed bot deals for the current broker day.
    A 'losing setup' = all bot positions on the same symbol opened in the same minute
    whose combined result is negative (so a 3-target batch counts once).
    """
    now = time.time()
    if not force and now - _daily_guard_cache["checked_at"] < DAILY_GUARD_CHECK_INTERVAL:
        return _daily_guard_cache["blocked"], _daily_guard_cache["reason"]
    _daily_guard_cache["checked_at"] = now

    try:
        # Broker "today" from server time (MT5 tick/deal times are server-time epochs)
        server_ts = None
        for sym in LOADED_ASSETS:
            t = mt5.symbol_info_tick(sym)
            if t and t.time:
                server_ts = max(server_ts or 0, t.time)
        server_day = datetime.fromtimestamp(server_ts, timezone.utc).date() if server_ts else datetime.now(timezone.utc).date()

        deals = mt5.history_deals_get(datetime.now() - pd.Timedelta(days=4), datetime.now() + pd.Timedelta(days=2)) or []

        opens = {}  # position_id -> (magic, symbol, open_minute)
        for d in deals:
            if d.entry == mt5.DEAL_ENTRY_IN:
                opens[d.position_id] = (d.magic, d.symbol, int(d.time // 60))

        setup_pnl = {}
        day_net = 0.0
        reset_time = float(tg_manager.settings.get("daily_guard_reset_time", 0.0))
        for d in deals:
            if d.entry not in (mt5.DEAL_ENTRY_OUT, mt5.DEAL_ENTRY_OUT_BY):
                continue
            if datetime.fromtimestamp(d.time, timezone.utc).date() != server_day:
                continue
            if d.time <= reset_time:
                continue  # Skipped by manual owner reset
            o = opens.get(d.position_id)
            magic = o[0] if o else d.magic
            if not _is_bot_magic(magic):
                continue
            pnl = d.profit + d.commission + d.swap + getattr(d, "fee", 0.0)
            day_net += pnl
            key = (o[1], o[2]) if o else (d.symbol, d.position_id)
            setup_pnl[key] = setup_pnl.get(key, 0.0) + pnl

        losing_setups = sum(1 for v in setup_pnl.values() if v < 0)
        balance, ccy = get_account_balance_native()
        start_balance = max(balance - day_net, 1e-9)
        day_loss_pct = (-day_net / start_balance * 100.0) if day_net < 0 else 0.0

        max_losses = int(tg_manager.settings.get("max_daily_losses", DEFAULT_MAX_DAILY_LOSSES))
        max_loss_pct = float(tg_manager.settings.get("max_daily_loss_pct", DEFAULT_MAX_DAILY_LOSS_PCT))

        blocked, reason = False, ""
        if losing_setups >= max_losses:
            blocked, reason = True, f"{losing_setups} losing setups today (limit {max_losses})"
        elif day_loss_pct >= max_loss_pct:
            blocked, reason = True, f"daily loss {day_loss_pct:.2f}% ({day_net:,.2f} {ccy}) hit limit {max_loss_pct:.1f}%"

        _daily_guard_cache["blocked"], _daily_guard_cache["reason"] = blocked, reason

        if blocked and _daily_guard_cache["alerted_day"] != server_day:
            _daily_guard_cache["alerted_day"] = server_day
            print(f"🛑 DAILY LOSS GUARD: {reason}. No new trades until next broker day.")
            tg_manager.send_broadcast_alert(
                f"🛑 <b>Daily Loss Guard Triggered</b>\n\n{reason}.\n"
                f"The bot will not open new trades until the next broker day. Open trades are still managed."
            )
    except Exception as e:
        print(f"⚠️ Daily loss guard error: {e}")

    return _daily_guard_cache["blocked"], _daily_guard_cache["reason"]


# ==============================================================================
# MAIN DUAL-ASSET LIVE TRADING ENGINE
# ==============================================================================

# Global in-memory reference to live assets
LOADED_ASSETS = {}


def _apply_model_package(asset_entry, pkg, cfg):
    asset_entry["model"] = pkg["model"]
    asset_entry["threshold"] = pkg.get("recommended_threshold", cfg["default_thresh"])
    asset_entry["trend_mode"] = pkg.get("trend_mode", DEFAULT_TREND_MODE)
    # Models from the old (leaky) trainer have no edge_ok flag -> treat as unvalidated
    asset_entry["edge_ok"] = bool(pkg.get("edge_ok", False))
    asset_entry["no_edge_warned"] = False


def _record_entry_diagnostic(asset_entry, reason, candle_time=None, detail=None):
    """Track entry-gate outcomes once per closed candle without changing trade decisions."""
    diagnostics = asset_entry.setdefault("entry_diagnostics", {
        "counts": {},
        "bar": None,
        "bar_reasons": set(),
        "state": None,
        "last_reason": "not_checked",
        "last_detail": "",
    })

    if candle_time is None:
        if diagnostics["state"] != reason:
            diagnostics["counts"][reason] = diagnostics["counts"].get(reason, 0) + 1
        diagnostics["state"] = reason
    else:
        diagnostics["state"] = None
        if diagnostics["bar"] != candle_time:
            diagnostics["bar"] = candle_time
            diagnostics["bar_reasons"] = set()
        if reason not in diagnostics["bar_reasons"]:
            diagnostics["counts"][reason] = diagnostics["counts"].get(reason, 0) + 1
            diagnostics["bar_reasons"].add(reason)

    diagnostics["last_reason"] = reason
    diagnostics["last_detail"] = detail or ""


def reload_ai_models(target_symbol=None):
    """
    Hot-reloads trained models from disk into the running trading engine in memory without downtime.
    """
    reloaded = []
    for sym_key, cfg in ASSET_CONFIGS.items():
        actual_sym = resolve_broker_symbol(sym_key)
        if not actual_sym:
            continue
        if target_symbol and target_symbol.upper().rstrip("M").rstrip("C") not in actual_sym.upper():
            continue

        if os.path.exists(cfg["model_file"]):
            try:
                pkg = joblib.load(cfg["model_file"])
                if actual_sym in LOADED_ASSETS:
                    _apply_model_package(LOADED_ASSETS[actual_sym], pkg, cfg)
                    t_date = pkg.get("trained_date", "Recently")
                    th_val = LOADED_ASSETS[actual_sym]["threshold"] * 100
                    print(f"🔄 [HOT-RELOAD] AI Model for {actual_sym} updated! (Trained: {t_date}, Threshold: {th_val:.0f}%, "
                          f"Trend: {LOADED_ASSETS[actual_sym]['trend_mode']}, Edge OK: {LOADED_ASSETS[actual_sym]['edge_ok']})")
                    reloaded.append((actual_sym, t_date, th_val))
            except Exception as e:
                print(f"⚠️ Error hot-reloading {actual_sym}: {e}")
    return reloaded


def run_live():
    global LOADED_ASSETS
    print("=" * 65)
    print("🤖 DUAL-ASSET AI QUANT TRADING BOT (XAUUSDm & BTCUSDm)")
    print("=" * 65)

    # Hook hot-reloader to Telegram Manager
    tg_manager.on_retrain_callback = reload_ai_models

    # Start Telegram background polling
    tg_manager.start_polling()

    LOADED_ASSETS.clear()

    for sym_key, cfg in ASSET_CONFIGS.items():
        actual_sym = resolve_broker_symbol(sym_key)
        if not actual_sym:
            print(f"⚠️ Warning: Could not resolve symbol {sym_key} in MT5. Skipping.")
            continue

        asset_entry = {
            "config": cfg,
            "model": None,
            "threshold": cfg["default_thresh"],
            "trend_mode": DEFAULT_TREND_MODE,
            "edge_ok": False,
            "no_edge_warned": False,
            "last_traded_bar": None,
            "entry_diagnostics": {
                "counts": {},
                "bar": None,
                "bar_reasons": set(),
                "state": None,
                "last_reason": "not_checked",
                "last_detail": "",
            }
        }

        if os.path.exists(cfg["model_file"]):
            try:
                pkg = joblib.load(cfg["model_file"])
                _apply_model_package(asset_entry, pkg, cfg)
                print(f"🧠 Loaded AI Filter for {actual_sym} (Trained: {pkg.get('trained_date')}, "
                      f"Threshold: {asset_entry['threshold']*100:.0f}%, Trend: {asset_entry['trend_mode']}, "
                      f"Edge OK: {asset_entry['edge_ok']})")
            except Exception as e:
                print(f"⚠️ Error loading model for {actual_sym}: {e}")
        else:
            print(f"ℹ️ No pre-trained model found for {actual_sym}. Run 'python train_model.py' to generate.")

        LOADED_ASSETS[actual_sym] = asset_entry

    if not LOADED_ASSETS:
        print("❌ No active assets available. Check MT5 connection.")
        return

    print("\n🟢 Dual-Asset Trading Loop Active. Press Ctrl+C in terminal to stop.\n")

    last_heartbeat = 0

    try:
        while True:
            current_time = time.time()
            print_heartbeat = (current_time - last_heartbeat >= HEARTBEAT_INTERVAL)

            # Check if user paused bot via Telegram
            if tg_manager.is_paused:
                if print_heartbeat:
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] ⏸️ BOT PAUSED VIA TELEGRAM. Scanning paused.")
                    last_heartbeat = current_time
                time.sleep(LIVE_POLL_INTERVAL)
                continue

            for symbol, asset_data in LOADED_ASSETS.items():
                cfg = asset_data["config"]
                ai_model = asset_data["model"]
                threshold = asset_data["threshold"]

                df = fetch_data(symbol, count_m5=300, count_h1=400, trend_mode=asset_data.get("trend_mode", DEFAULT_TREND_MODE))
                if df is None or len(df) < 5:
                    _record_entry_diagnostic(
                        asset_data,
                        "data_unavailable",
                        detail="Could not fetch enough M5/H1 candles",
                    )
                    continue

                asset_data["entry_diagnostics"]["state"] = None
                live_candle = df.iloc[-1]       # still forming -> only for live price / trade management
                last_candle = df.iloc[-2]       # last CLOSED candle -> all entry decisions
                prev_candle = df.iloc[-3]
                candle_time = last_candle["time"]
                price = last_candle["close"]

                crossover_buy = (prev_candle["m5_ema9"] <= prev_candle["m5_ema21"] and last_candle["m5_ema9"] > last_candle["m5_ema21"])
                pullback_buy = (last_candle["m5_ema9"] > last_candle["m5_ema21"] and prev_candle["low"] <= prev_candle["m5_ema21"] and last_candle["close"] > last_candle["m5_ema9"] and last_candle["close"] > last_candle["open"])

                crossover_sell = (prev_candle["m5_ema9"] >= prev_candle["m5_ema21"] and last_candle["m5_ema9"] < last_candle["m5_ema21"])
                pullback_sell = (last_candle["m5_ema9"] < last_candle["m5_ema21"] and prev_candle["high"] >= prev_candle["m5_ema21"] and last_candle["close"] < last_candle["m5_ema9"] and last_candle["close"] < last_candle["open"])

                buy_signal = (
                    last_candle["h1_trend"] == 1
                    and (crossover_buy or pullback_buy)
                    and 40 <= last_candle["m5_rsi"] <= 68
                )

                sell_signal = (
                    last_candle["h1_trend"] == -1
                    and (crossover_sell or pullback_sell)
                    and 32 <= last_candle["m5_rsi"] <= 60
                )
                if buy_signal or sell_signal:
                    setup_direction = "BUY" if buy_signal else "SELL"
                    _record_entry_diagnostic(
                        asset_data,
                        "strategy_setup",
                        candle_time,
                        f"{setup_direction} setup passed trend, EMA, and RSI rules",
                    )
                else:
                    _record_entry_diagnostic(
                        asset_data,
                        "no_strategy_setup",
                        candle_time,
                        "No EMA/trend/RSI entry pattern on this closed M5 bar",
                    )

                # 1. Manage Active Positions
                manage_active_trades(symbol, cfg, live_candle, last_candle)
                open_positions = mt5.positions_get(symbol=symbol) or []

                # Heartbeat Status
                if print_heartbeat:
                    h1_trend_str = "BULLISH 📈" if last_candle["h1_trend"] == 1 else ("BEARISH 📉" if last_candle["h1_trend"] == -1 else "NEUTRAL ⚖️")
                    now_str = datetime.now().strftime("%H:%M:%S")
                    pos_count = len(open_positions)
                    rsi_val = last_candle["m5_rsi"]
                    adx_val = last_candle["h1_adx"]
                    print(f"[{now_str}] {symbol:<8} @ ${live_candle['close']:8.2f} | H1: {h1_trend_str} (ADX: {adx_val:4.1f}) | M5 RSI: {rsi_val:4.1f} | Open: {pos_count}")

                # Machine Learning Model Filter:
                # Runs identically on both Live and Demo using the configured threshold.
                thresh_key = "gold_threshold" if "XAU" in symbol else "btc_threshold"
                active_thresh = tg_manager.settings.get(thresh_key, asset_data["threshold"])

                # Daily loss guard (applied identically on both Live and Demo)
                blocked, _reason = check_daily_loss_guard()
                if blocked:
                    _record_entry_diagnostic(asset_data, "daily_loss_guard", candle_time, _reason)
                    continue

                # High-Impact Economic News Shield (pauses new entries during high-impact USD events)
                if tg_manager.settings.get("news_filter_enabled", True):
                    is_news_blocked, news_summary, _ = news_filter.check_news_block()
                    if is_news_blocked:
                        _record_entry_diagnostic(asset_data, "news_filter", candle_time, news_summary)
                        if print_heartbeat:
                            print(f"🛑 [NEWS SHIELD] {symbol}: New entries paused due to {news_summary}")
                        continue

                max_open = min(int(tg_manager.settings.get("max_positions", MAX_OPEN_POSITIONS_PER_SYMBOL)), 3)
                if len(open_positions) >= max_open:
                    _record_entry_diagnostic(
                        asset_data,
                        "position_limit",
                        candle_time,
                        f"{len(open_positions)} positions open; configured limit is {max_open}",
                    )
                    continue

                if candle_time == asset_data["last_traded_bar"]:
                    _record_entry_diagnostic(asset_data, "bar_already_traded", candle_time)
                    continue

                if last_candle["h1_adx"] < MIN_ADX:
                    _record_entry_diagnostic(
                        asset_data,
                        "adx_filter",
                        candle_time,
                        f"H1 ADX {last_candle['h1_adx']:.1f} below minimum {MIN_ADX}",
                    )
                    continue

                # Phase 1.1: Strict edge enforcement (Ground Rules 8 & 9)
                if ai_model is None or not asset_data.get("edge_ok", False):
                    reason = "model_missing" if ai_model is None else "model_edge_not_validated"
                    detail = "No model loaded" if ai_model is None else "Model package has edge_ok=False"
                    _record_entry_diagnostic(asset_data, reason, candle_time, detail)
                    if not asset_data.get("no_edge_warned", False):
                        asset_data["no_edge_warned"] = True
                        tg_manager.send_broadcast_alert(
                            f"⛔ <b>{symbol}:</b> No validated AI model with positive edge. New entries paused until /retrain passes."
                        )
                    continue

                if buy_signal or sell_signal:
                    direction = 1 if buy_signal else -1
                    atr = last_candle["m5_atr"]

                    # Entry Guard: Overextension check
                    ema_dist = abs(price - last_candle["m5_ema9"])
                    if ema_dist > 1.4 * atr:
                        _record_entry_diagnostic(
                            asset_data,
                            "overextension_filter",
                            candle_time,
                            f"Price is {ema_dist:.5f} from EMA9; limit is {1.4 * atr:.5f}",
                        )
                        print(f"⏳ {symbol} Entry Guard: Price overextended (${price:.2f} vs EMA {last_candle['m5_ema9']:.2f}). Waiting for pullback.")
                        try:
                            db_ledger.log_signal(
                                symbol=symbol,
                                direction=direction,
                                status="SKIPPED",
                                skip_reason="Overextended",
                                entry_price=price,
                                m5_rsi=float(last_candle["m5_rsi"]),
                                h1_adx=float(last_candle["h1_adx"]),
                                trend_mode=asset_data.get("trend_mode", "slow"),
                                confidence=0.0,
                                threshold=float(active_thresh),
                                model_trained_date=asset_data.get("trained_date"),
                            )
                        except Exception as ex:
                            print(f"⚠️ Error logging overextended signal: {ex}")
                        continue

                    # Phase 1.2: Eliminate Trade Blindly fallback - defaults to False and 0.0
                    ai_approved = False
                    win_prob = 0.0

                    thresh_key = "gold_threshold" if "XAU" in symbol else "btc_threshold"
                    active_thresh = tg_manager.settings.get(thresh_key, asset_data["threshold"])

                    if ai_model is not None:
                        try:
                            feat_vec = extract_single_feature_vector(last_candle, prev_candle, direction)
                            win_prob = float(ai_model.predict_proba(feat_vec.reshape(1, -1))[0][1])
                            ai_approved = (win_prob >= active_thresh)
                        except Exception as e:
                            print(f"⚠️ {symbol} AI Inference error: {e}")
                            _record_entry_diagnostic(
                                asset_data,
                                "ai_inference_error",
                                candle_time,
                                str(e),
                            )
                            ai_approved = False

                    if not ai_approved:
                        _record_entry_diagnostic(
                            asset_data,
                            "confidence_below_threshold",
                            candle_time,
                            f"Model score {win_prob:.3f}; required {active_thresh:.3f}",
                        )
                        print(f"✋ {symbol} AI FILTERED: Win Confidence {win_prob*100:.1f}% < threshold {active_thresh*100:.0f}%. Setup skipped!")
                        asset_data["last_traded_bar"] = candle_time
                        try:
                            db_ledger.log_signal(
                                symbol=symbol,
                                direction=direction,
                                status="SKIPPED",
                                skip_reason="Low AI Confidence",
                                entry_price=price,
                                m5_rsi=float(last_candle["m5_rsi"]),
                                h1_adx=float(last_candle["h1_adx"]),
                                trend_mode=asset_data.get("trend_mode", "slow"),
                                confidence=win_prob,
                                threshold=float(active_thresh),
                                model_trained_date=asset_data.get("trained_date"),
                            )
                        except Exception as ex:
                            print(f"⚠️ Error logging filtered signal: {ex}")
                        continue

                    # Dynamic AI Lot Size & Risk Multiplier
                    if win_prob >= 0.70:
                        risk_mult = 1.25
                        tier_label = "🔥 ULTRA CONVICTION (1.25x Lot)"
                    elif win_prob >= 0.62:
                        risk_mult = 1.00
                        tier_label = "✨ HIGH CONVICTION (1.00x Lot)"
                    else:
                        risk_mult = 0.75
                        tier_label = "⚡ DEFENSIVE CONVICTION (0.75x Lot)"

                    balance, _ccy = get_account_balance_native()
                    risk_pct = tg_manager.settings.get("risk_percent", 1.5) / 100.0
                    total_risk = balance * risk_pct * risk_mult
                    stop_dist = max(atr * ATR_MULTIPLIER, cfg["min_stop_dist"])

                    order_type = mt5.ORDER_TYPE_BUY if buy_signal else mt5.ORDER_TYPE_SELL
                    dir_str = "BUY 📈" if buy_signal else "SELL 📉"

                    print(f"\n🎯 {symbol} {dir_str} SIGNAL APPROVED by AI!")
                    print(f"   Price: ${price:.2f} | Confidence: {win_prob*100:.1f}% | Tier: {tier_label}")
                    print(f"   Stop Distance: ${stop_dist:.2f} | Total Risk: ${total_risk:.2f}")

                    _record_entry_diagnostic(
                        asset_data,
                        "model_approved_order_attempted",
                        candle_time,
                        f"{setup_direction} score {win_prob:.3f} met threshold {active_thresh:.3f}",
                    )
                    open_multi_target_batch(
                        symbol=symbol,
                        order_type=order_type,
                        stop_dist=stop_dist,
                        total_risk_amount=total_risk,
                        tp_multipliers=cfg["tp_multipliers"],
                        magic_base=cfg["magic_base"],
                        ai_tier_label=tier_label,
                        win_prob=win_prob,
                        candle_metrics={
                            "m5_rsi": float(last_candle["m5_rsi"]),
                            "h1_adx": float(last_candle["h1_adx"]),
                            "trend_mode": asset_data.get("trend_mode", "slow"),
                            "threshold": float(active_thresh),
                            "trained_date": asset_data.get("trained_date"),
                        }
                    )

                    asset_data["last_traded_bar"] = candle_time

            if print_heartbeat:
                last_heartbeat = current_time
                for symbol, asset_data in LOADED_ASSETS.items():
                    diagnostics = asset_data["entry_diagnostics"]
                    counts = diagnostics["counts"]
                    count_summary = ", ".join(
                        f"{name}={count}" for name, count in sorted(counts.items())
                    ) or "no closed bars checked"
                    detail = f" ({diagnostics['last_detail']})" if diagnostics["last_detail"] else ""
                    print(
                        f"[ENTRY CHECK] {symbol}: latest={diagnostics['last_reason']}{detail}; "
                        f"bar counts: {count_summary}"
                    )
                try:
                    tg_manager.check_and_prompt_unprotected_trades()
                except Exception:
                    pass

                # Automated Sunday AI Brain Refresh (Runs weekly)
                now_dt = datetime.now()
                if now_dt.weekday() == 6 and now_dt.hour == 20: # Sunday 8:00 PM
                    last_auto = tg_manager.config.get("last_auto_retrain", "")
                    today_str = now_dt.strftime("%Y-%m-%d")
                    if last_auto != today_str:
                        tg_manager.config["last_auto_retrain"] = today_str
                        save_config(tg_manager.config)
                        print("📅 [Sunday Auto-Scheduler] Triggering automated weekly AI retraining...")
                        tg_manager.send_broadcast_alert(
                            "📅 <b>Sunday AI Self-Learning:</b> Triggering automated weekly retraining on freshest 40,000 candles..."
                        )
                        def _auto_worker():
                            try:
                                import train_model
                                res_list = train_model.run_retrain_task()
                                promoted_any = any(r.get("promoted", False) for r in (res_list or []))
                                if promoted_any:
                                    reload_ai_models()
                                    tg_manager.send_broadcast_alert(
                                        "🎉 <b>Sunday AI Self-Learning Complete!</b> Models promoted through gate and hot-reloaded with fresh market edge."
                                    )
                                else:
                                    tg_manager.send_broadcast_alert(
                                        "ℹ️ <b>Sunday AI Self-Learning Complete:</b> Candidates did not beat incumbent gate or lacked edge. Incumbent models retained untouched."
                                    )
                            except Exception as ex:
                                tg_manager.send_broadcast_alert(f"⚠️ Sunday Auto-Retrain error: {ex}")
                        threading.Thread(target=_auto_worker, daemon=True).start()

            time.sleep(LIVE_POLL_INTERVAL)

    except KeyboardInterrupt:
        print("\n⏹️ Dual-Asset Live Bot Stopped by User")


# ==============================================================================
# MAIN ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    if not acquire_single_instance_lock():
        print("❌ [SINGLE-INSTANCE LOCK] Another instance of bot.py is already running on this machine!")
        print("👉 Terminate the other process first to prevent duplicate Telegram connections.")
        sys.exit(1)

    connect_mt5()
    run_live()
    mt5.shutdown()
    print("🔌 MT5 Disconnected")