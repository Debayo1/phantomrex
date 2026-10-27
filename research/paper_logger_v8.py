"""Forward paper monitor for the fixed, research-only V8 SMC proxy."""

import csv
import json
import os
import sys
import time
from datetime import datetime, timedelta, timezone

import MetaTrader5 as mt5
import pandas as pd

RESEARCH_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(RESEARCH_DIR)
RUNTIME_DIR = os.path.join(PROJECT_ROOT, "runtime_data")
LOG_FILE = os.path.join(RUNTIME_DIR, "v8_forward_paper.csv")
STATE_FILE = os.path.join(RUNTIME_DIR, "v8_forward_state.json")

POLL_SECONDS = 60
M15_LOOKBACK = 3000
HTF_LOOKBACK = {"D1": 1500, "H4": 3000, "H2": 5000}
MAX_HOLD_BARS = 96
M15_DURATION = pd.Timedelta(minutes=15)
UTC = timezone.utc

TARGET_SYMBOLS = (
    {"name": "XAUUSDm", "point": 0.001, "spread_est": 0.25, "contract_size": 100.0},
    {"name": "BTCUSDm", "point": 0.01, "spread_est": 15.0, "contract_size": 1.0},
)

CSV_COLUMNS = (
    "event_utc", "symbol", "event", "entry_utc", "direction", "entry_price",
    "stop", "target", "exit_utc", "exit_price", "exit_reason", "gross_r",
    "net_r", "trade_id",
)


def assert_safe_environment():
    """Keep the forward paper monitor isolated from the live trading engine."""
    if "bot" in sys.modules or "telegram_manager" in sys.modules:
        raise RuntimeError("Unsafe live trading module loaded; refusing V8 paper monitoring.")


def _ensure_storage():
    os.makedirs(RUNTIME_DIR, exist_ok=True)
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, "w", newline="", encoding="utf-8") as file:
            csv.DictWriter(file, fieldnames=CSV_COLUMNS).writeheader()
    if not os.path.exists(STATE_FILE):
        _write_state({"version": 1, "symbols": {}})


def _write_state(state):
    os.makedirs(RUNTIME_DIR, exist_ok=True)
    temporary_path = STATE_FILE + ".tmp"
    with open(temporary_path, "w", encoding="utf-8") as file:
        json.dump(state, file, indent=2)
    os.replace(temporary_path, STATE_FILE)


def _append_event(event):
    with open(LOG_FILE, "a", newline="", encoding="utf-8") as file:
        csv.DictWriter(file, fieldnames=CSV_COLUMNS).writerow(event)


def _rates_frame(symbol, timeframe, count):
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, count)
    if rates is None or len(rates) == 0:
        raise RuntimeError(
            f"No MT5 rates for {symbol}, timeframe {timeframe}: {mt5.last_error()}"
        )
    frame = pd.DataFrame(rates).sort_values("time").drop_duplicates("time").reset_index(drop=True)
    frame["time"] = pd.to_datetime(frame["time"], unit="s").astype("datetime64[ns]")
    return frame


def _zero_event(symbol, event_name):
    return {
        "event_utc": datetime.now(UTC).isoformat(),
        "symbol": symbol,
        "event": event_name,
        "entry_utc": "",
        "direction": "",
        "entry_price": "",
        "stop": "",
        "target": "",
        "exit_utc": "",
        "exit_price": "",
        "exit_reason": "",
        "gross_r": "",
        "net_r": "",
        "trade_id": "",
    }


def _manage_open_trade(symbol, symbol_state, closed_bars, sym_cfg):
    trade = symbol_state.get("open_trade")
    if not trade:
        return

    entry_time = pd.Timestamp(trade["entry_utc"])
    direction = int(trade["direction"])
    stop = float(trade["stop"])
    target = float(trade["target"])
    entry_price = float(trade["entry_price"])
    stop_dist = float(trade["stop_dist"])
    cost_r = float(trade["cost_r"])

    for _, bar in closed_bars.iterrows():
        bar_time = pd.Timestamp(bar["time"])
        if bar_time < entry_time:
            continue

        high, low = float(bar["high"]), float(bar["low"])
        sl_hit = low <= stop if direction == 1 else high >= stop
        tp_hit = high >= target if direction == 1 else low <= target
        exit_price = None
        exit_reason = ""
        if sl_hit:
            exit_price = stop
            exit_reason = "SL_TIE" if tp_hit else "SL"
        elif tp_hit:
            exit_price = target
            exit_reason = "TP"
        elif (bar_time - entry_time) >= M15_DURATION * MAX_HOLD_BARS:
            exit_price = float(bar["close"])
            exit_reason = "TIMEOUT"

        if exit_price is None:
            continue

        gross_r = (exit_price - entry_price) * direction / stop_dist
        event = _zero_event(symbol, "CLOSED")
        event.update({
            "entry_utc": trade["entry_utc"],
            "direction": "BUY" if direction == 1 else "SELL",
            "entry_price": entry_price,
            "stop": stop,
            "target": target,
            "exit_utc": bar_time.isoformat(),
            "exit_price": exit_price,
            "exit_reason": exit_reason,
            "gross_r": round(gross_r, 6),
            "net_r": round(gross_r - cost_r, 6),
            "trade_id": trade["trade_id"],
        })
        _append_event(event)
        print(
            f"[V8 PAPER] {symbol} {event['direction']} closed {exit_reason}: "
            f"net {event['net_r']:+.3f}R (entry {trade['entry_utc']})"
        )
        symbol_state["open_trade"] = None
        return


def _find_current_entry(symbol, sym_cfg, rates):
    from research.run_research import simulate_smc_structure_ob_fvg

    m15 = rates["M15"]
    active_bar = m15.iloc[-1]
    tick = mt5.symbol_info_tick(symbol)
    if not tick:
        raise RuntimeError(f"No current MT5 tick for {symbol}: {mt5.last_error()}")
    active_epoch = int(pd.Timestamp(active_bar["time"]).timestamp())
    if not (active_epoch <= int(tick.time) < active_epoch + 15 * 60):
        return None

    trades = simulate_smc_structure_ob_fvg(
        m15,
        rates["D1"],
        rates["H4"],
        rates["H2"],
        sym_cfg,
        swing_bars=2,
        max_ob_lookback=10,
        max_swing_age=40,
        setup_expiry_bars=32,
        max_hold_bars=MAX_HOLD_BARS,
        tp_r=2.0,
    )
    active_time = pd.Timestamp(active_bar["time"])
    return next(
        (trade for trade in trades if pd.Timestamp(trade["entry_time"]) == active_time),
        None,
    )


def _record_entry(symbol, symbol_state, trade, sym_cfg, spread_points):
    direction = int(trade["dir"])
    spread_price = max(spread_points * sym_cfg["point"], sym_cfg["spread_est"])
    stop_dist = float(trade["stop_dist"])
    cost_r = spread_price * 1.25 / stop_dist
    entry_time = pd.Timestamp(trade["entry_time"])
    trade_id = f"{symbol}-{entry_time.strftime('%Y%m%dT%H%M')}"
    paper_trade = {
        "trade_id": trade_id,
        "entry_utc": entry_time.isoformat(),
        "direction": direction,
        "entry_price": float(trade["entry_price"]),
        "stop": float(trade["sl"]),
        "target": float(trade["tp"]),
        "stop_dist": stop_dist,
        "cost_r": cost_r,
    }

    event = _zero_event(symbol, "OPENED")
    event.update({
        "entry_utc": paper_trade["entry_utc"],
        "direction": "BUY" if direction == 1 else "SELL",
        "entry_price": paper_trade["entry_price"],
        "stop": paper_trade["stop"],
        "target": paper_trade["target"],
        "trade_id": trade_id,
    })
    _append_event(event)
    symbol_state["open_trade"] = paper_trade
    print(
        f"[V8 PAPER] {symbol} {event['direction']} opened at "
        f"{paper_trade['entry_price']:.5f}; SL {paper_trade['stop']:.5f}; "
        f"TP {paper_trade['target']:.5f}. No order sent."
    )


def run_once():
    assert_safe_environment()
    _ensure_storage()
    with open(STATE_FILE, "r", encoding="utf-8") as file:
        state = json.load(file)
    if state.get("version") != 1:
        raise RuntimeError("Unsupported V8 paper state version; preserve the file and inspect before continuing.")

    from research.run_research import setup_symbol

    for sym_cfg in TARGET_SYMBOLS:
        requested = sym_cfg["name"]
        symbol = setup_symbol(requested)
        if not symbol:
            raise RuntimeError(f"Could not select MT5 symbol {requested}: {mt5.last_error()}")
        rates = {
            "M15": _rates_frame(symbol, mt5.TIMEFRAME_M15, M15_LOOKBACK),
            "D1": _rates_frame(symbol, mt5.TIMEFRAME_D1, HTF_LOOKBACK["D1"]),
            "H4": _rates_frame(symbol, mt5.TIMEFRAME_H4, HTF_LOOKBACK["H4"]),
            "H2": _rates_frame(symbol, mt5.TIMEFRAME_H2, HTF_LOOKBACK["H2"]),
        }
        if len(rates["M15"]) < 2:
            raise RuntimeError(f"Insufficient M15 rates for {symbol}")

        closed_bars = rates["M15"].iloc[:-1].copy()
        newest_closed = pd.Timestamp(closed_bars.iloc[-1]["time"])
        symbol_state = state["symbols"].get(symbol)
        if symbol_state is None:
            symbol_state = {
                "last_closed_utc": newest_closed.isoformat(),
                "last_active_utc": pd.Timestamp(rates["M15"].iloc[-1]["time"]).isoformat(),
                "open_trade": None,
            }
            state["symbols"][symbol] = symbol_state
            print(f"[V8 PAPER] {symbol} baseline set at {newest_closed}; no historical signals backfilled.")
            _write_state(state)
            continue

        last_closed = pd.Timestamp(symbol_state["last_closed_utc"])
        new_closed = closed_bars[closed_bars["time"] > last_closed]
        if not new_closed.empty:
            _manage_open_trade(symbol, symbol_state, new_closed, sym_cfg)
            symbol_state["last_closed_utc"] = newest_closed.isoformat()

        active_time = pd.Timestamp(rates["M15"].iloc[-1]["time"])
        if (
            symbol_state.get("open_trade") is None
            and active_time > pd.Timestamp(symbol_state["last_active_utc"])
        ):
            trade = _find_current_entry(symbol, sym_cfg, rates)
            if trade:
                spread_points = float(rates["M15"].iloc[-1].get("spread", 0.0))
                _record_entry(symbol, symbol_state, trade, sym_cfg, spread_points)

        symbol_state["last_active_utc"] = active_time.isoformat()
        _write_state(state)


def run_paper_logger_v8(poll_seconds=POLL_SECONDS):
    """Continuously monitor closed M15 bars and write paper-only V8 events."""
    assert_safe_environment()
    if poll_seconds < 1:
        raise ValueError("poll_seconds must be at least one second")
    if not mt5.initialize():
        raise RuntimeError(f"MT5 initialization failed: {mt5.last_error()}")

    print("V8 SMC forward paper monitor running. No orders are sent. Press Ctrl+C to stop.")
    try:
        while True:
            try:
                run_once()
            except Exception as exc:
                print(f"V8 paper monitor iteration failed: {exc}")
            time.sleep(poll_seconds)
    except KeyboardInterrupt:
        print("\nV8 paper monitor stopped.")
    finally:
        mt5.shutdown()


if __name__ == "__main__":
    interval = int(sys.argv[1]) if len(sys.argv) > 1 else POLL_SECONDS
    run_paper_logger_v8(interval)
