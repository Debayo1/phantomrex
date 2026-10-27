"""
research/paper_logger_v7.py
Offline paper-logger for Variant V7 ("9am CR model") under Addendum 8.

Continuously monitors or back-evaluates recent trading days:
  - 08:00 NY H1 reference range [L_ref, H_ref]
  - 09:00-10:00 NY setup window (first touch, Bullish/Bearish FVG, IFVG, OB confirmation)
  - Records simulated entry, SL, TP, outcome, gross R, net R to research/v7_paper_log.csv
  - ZERO orders sent. Strictly isolated from bot.py and live execution.
"""

import os
import sys
import csv
from datetime import datetime, timedelta
import zoneinfo
import pandas as pd
import numpy as np

# Ensure paths
RESEARCH_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(RESEARCH_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
PAPER_LOG_FILE = os.path.join(RESEARCH_DIR, "v7_paper_log.csv")

NY_TZ = zoneinfo.ZoneInfo("America/New_York")
UTC_TZ = zoneinfo.ZoneInfo("UTC")

TARGET_SYMBOLS = [
    {"name": "XAUUSDm", "point": 0.001, "spread_est": 0.25, "contract_size": 100.0},
    {"name": "BTCUSDm", "point": 0.01, "spread_est": 15.0, "contract_size": 1.0}
]


def assert_safe_environment():
    """Ensures order-sending or live promotion modules are NOT loaded."""
    forbidden = ["bot", "telegram_manager", "model_promotion"]
    for mod in forbidden:
        if mod in sys.modules:
            raise RuntimeError(f"FATAL: Unsafe module '{mod}' imported in paper logger! Aborting.")


def ensure_paper_log_initialized():
    """Initializes research/v7_paper_log.csv with headers if it does not exist."""
    if not os.path.exists(PAPER_LOG_FILE):
        os.makedirs(RESEARCH_DIR, exist_ok=True)
        with open(PAPER_LOG_FILE, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "log_time", "ny_date", "symbol", "h_ref", "l_ref", "touch_side",
                "touch_time", "confirmed", "entry_time", "entry_price", "sl", "tp",
                "stop_dist", "exit_time", "exit_price", "exit_reason", "gross_r", "net_r", "status"
            ])


def record_paper_trade(entry_record):
    """Appends an evaluated paper trade or daily status to v7_paper_log.csv."""
    ensure_paper_log_initialized()
    log_time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open(PAPER_LOG_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            log_time_str,
            entry_record.get("ny_date", ""),
            entry_record.get("symbol", ""),
            f"{entry_record.get('h_ref', 0.0):.3f}",
            f"{entry_record.get('l_ref', 0.0):.3f}",
            entry_record.get("touch_side", "NONE"),
            entry_record.get("touch_time", ""),
            entry_record.get("confirmed", False),
            entry_record.get("entry_time", ""),
            f"{entry_record.get('entry_price', 0.0):.3f}" if entry_record.get("entry_price") else "",
            f"{entry_record.get('sl', 0.0):.3f}" if entry_record.get("sl") else "",
            f"{entry_record.get('tp', 0.0):.3f}" if entry_record.get("tp") else "",
            f"{entry_record.get('stop_dist', 0.0):.3f}" if entry_record.get("stop_dist") else "",
            entry_record.get("exit_time", ""),
            f"{entry_record.get('exit_price', 0.0):.3f}" if entry_record.get("exit_price") else "",
            entry_record.get("exit_reason", ""),
            f"{entry_record.get('gross_r', 0.0):+.3f}" if "gross_r" in entry_record else "",
            f"{entry_record.get('net_r', 0.0):+.3f}" if "net_r" in entry_record else "",
            entry_record.get("status", "")
        ])


def _filter_recent_trades(trades, days_back, now=None):
    """Keep trades within the rolling window, using the strategy's New York calendar."""
    reference_time = now or datetime.now(NY_TZ)
    cutoff_date = reference_time.astimezone(NY_TZ).date() - timedelta(days=days_back)
    return [trade for trade in trades if trade["entry_time"].date() >= cutoff_date]


def run_paper_logger(days_back=10):
    """
    Scans the past `days_back` days from MT5 and logs all V7 setups to paper log.
    Can be run as a daily scheduled script or offline auditor.
    """
    assert_safe_environment()
    ensure_paper_log_initialized()

    try:
        import MetaTrader5 as mt5
    except ImportError:
        print("MetaTrader5 package not available.")
        return

    if not mt5.terminal_info():
        if not mt5.initialize():
            print("❌ MT5 initialization failed:", mt5.last_error())
            return

    from research.run_research import fetch_m1_chunked, simulate_v7_cr_model, setup_symbol

    print(f"📋 Running V7 Paper Logger across past {days_back} days...")

    total_logged = 0
    for sym_cfg in TARGET_SYMBOLS:
        sym = sym_cfg["name"]
        resolved = setup_symbol(sym)
        if not resolved:
            continue

        df_m1 = fetch_m1_chunked(resolved, max_chunks=2, chunk_size=20000)
        if df_m1 is None or len(df_m1) == 0:
            continue

        stats = simulate_v7_cr_model(df_m1, sym_cfg)
        trades = stats.get("trades", [])

        # Filter trades to recent days
        recent_trades = _filter_recent_trades(trades, days_back)

        for t in recent_trades:
            record_paper_trade({
                "ny_date": str(t["entry_time"].date()),
                "symbol": sym,
                "h_ref": t["h_ref"],
                "l_ref": t["l_ref"],
                "touch_side": t["touch_side"],
                "touch_time": str(t["touch_time"]),
                "confirmed": True,
                "entry_time": str(t["entry_time"]),
                "entry_price": t["entry_price"],
                "sl": t["sl"],
                "tp": t["tp"],
                "stop_dist": t["stop_dist"],
                "exit_time": str(t["exit_time"]),
                "exit_price": t["exit_price"],
                "exit_reason": t["exit_reason"],
                "gross_r": t["gross_r"],
                "net_r": t["net_r"],
                "status": "COMPLETED"
            })
            total_logged += 1

        print(f"   [{sym}] Logged {len(recent_trades)} recent trades to {PAPER_LOG_FILE}")

    mt5.shutdown()
    print(f"✅ V7 Paper Logger completed: {total_logged} trades processed.")


if __name__ == "__main__":
    days = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    run_paper_logger(days_back=days)
