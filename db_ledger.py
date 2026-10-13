"""
Thread-safe SQLite signal & trade ledger (signals.db)
Stores signals, entry metrics, and initial risk parameters for closed deal joining.
Enforces WAL mode, threading locks, and institutional audit telemetry.
"""

import sqlite3
import threading
import os
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any, Tuple

DB_PATH = "signals.db"
_db_lock = threading.Lock()


def get_db_connection(db_path: str = DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: str = DB_PATH):
    with _db_lock:
        conn = get_db_connection(db_path)
        try:
            conn.execute("""
            CREATE TABLE IF NOT EXISTS signals (
                signal_id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                symbol TEXT NOT NULL,
                direction INTEGER NOT NULL,
                position_ticket INTEGER,
                entry_price REAL,
                initial_sl REAL,
                initial_risk_price REAL,
                initial_risk_currency REAL,
                m5_rsi REAL,
                h1_adx REAL,
                trend_mode TEXT,
                confidence REAL,
                threshold REAL,
                spread REAL,
                model_trained_date TEXT,
                status TEXT NOT NULL,
                skip_reason TEXT
            );
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_signals_pos ON signals(position_ticket);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_signals_time ON signals(timestamp);")
            conn.commit()
        finally:
            conn.close()


def log_signal(
    symbol: str,
    direction: int,
    status: str,
    position_ticket: Optional[int] = None,
    entry_price: Optional[float] = None,
    initial_sl: Optional[float] = None,
    initial_risk_price: Optional[float] = None,
    initial_risk_currency: Optional[float] = None,
    m5_rsi: Optional[float] = None,
    h1_adx: Optional[float] = None,
    trend_mode: Optional[str] = None,
    confidence: Optional[float] = None,
    threshold: Optional[float] = None,
    spread: Optional[float] = None,
    model_trained_date: Optional[str] = None,
    skip_reason: Optional[str] = None,
    timestamp: Optional[str] = None,
    db_path: str = DB_PATH
) -> int:
    init_db(db_path)
    if timestamp is None:
        timestamp = datetime.now().isoformat()

    with _db_lock:
        conn = get_db_connection(db_path)
        try:
            cursor = conn.cursor()
            cursor.execute("""
            INSERT INTO signals (
                timestamp, symbol, direction, position_ticket, entry_price,
                initial_sl, initial_risk_price, initial_risk_currency,
                m5_rsi, h1_adx, trend_mode, confidence, threshold,
                spread, model_trained_date, status, skip_reason
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """, (
                timestamp, symbol, direction, position_ticket, entry_price,
                initial_sl, initial_risk_price, initial_risk_currency,
                m5_rsi, h1_adx, trend_mode, confidence, threshold,
                spread, model_trained_date, status, skip_reason
            ))
            conn.commit()
            return cursor.lastrowid
        finally:
            conn.close()


def update_signal_position(signal_id: int, position_ticket: int, db_path: str = DB_PATH):
    with _db_lock:
        conn = get_db_connection(db_path)
        try:
            conn.execute("UPDATE signals SET position_ticket = ? WHERE signal_id = ?;", (position_ticket, signal_id))
            conn.commit()
        finally:
            conn.close()


def get_earliest_signal_timestamp(db_path: str = DB_PATH) -> Optional[str]:
    init_db(db_path)
    with _db_lock:
        conn = get_db_connection(db_path)
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT MIN(timestamp) FROM signals WHERE status = 'EXECUTED';")
            row = cursor.fetchone()
            return row[0] if row and row[0] else None
        finally:
            conn.close()


def get_signal_by_position(position_ticket: int, db_path: str = DB_PATH) -> Optional[Dict[str, Any]]:
    init_db(db_path)
    with _db_lock:
        conn = get_db_connection(db_path)
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM signals WHERE position_ticket = ? LIMIT 1;", (position_ticket,))
            row = cursor.fetchone()
            return dict(row) if row else None
        finally:
            conn.close()


def get_all_executed_signals(db_path: str = DB_PATH) -> List[Dict[str, Any]]:
    init_db(db_path)
    with _db_lock:
        conn = get_db_connection(db_path)
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM signals WHERE status = 'EXECUTED' ORDER BY timestamp ASC;")
            return [dict(r) for r in cursor.fetchall()]
        finally:
            conn.close()


def is_bot_magic(magic: int) -> bool:
    """Verifies magic number belongs to active bot range: magic_base < magic <= magic_base + 10."""
    for base in [101000, 102000, 103000, 104000]:
        if base < magic <= base + 10:
            return True
    return False


def compute_performance_stats(
    deals_list: List[Any],
    signals_db_path: str = DB_PATH,
    signals_override: Optional[Dict[int, Dict[str, Any]]] = None
) -> Dict[str, Any]:
    """
    Joins MT5 closed deals to signals.db using deal.position_id == signals.position_ticket.
    Enforces magic range filtering and computes institutional metrics, R-multiples, and calibration buckets.
    """
    # 1. Load executed signals indexed by position_ticket
    if signals_override is not None:
        sig_map = signals_override
    else:
        executed_signals = get_all_executed_signals(signals_db_path)
        sig_map = {s["position_ticket"]: s for s in executed_signals if s.get("position_ticket") is not None}

    # 2. Filter deals by magic range
    bot_deals = [d for d in deals_list if is_bot_magic(getattr(d, "magic", 0))]

    # 3. Group deals by position_id
    pos_deals_map: Dict[int, List[Any]] = {}
    for d in bot_deals:
        pid = getattr(d, "position_id", None)
        if pid:
            pos_deals_map.setdefault(pid, []).append(d)

    matched_positions = []
    unmatched_positions = []

    for pid, pdeals in pos_deals_map.items():
        # Check if closed: has an exit deal (entry in DEAL_ENTRY_OUT(1), DEAL_ENTRY_INOUT(2), DEAL_ENTRY_OUT_BY(3))
        has_out = any(getattr(d, "entry", 0) in (1, 2, 3) for d in pdeals)
        if not has_out:
            continue

        # Net Result = sum(profit + commission + swap + fee)
        net_result = sum(
            getattr(d, "profit", 0.0)
            + getattr(d, "commission", 0.0)
            + getattr(d, "swap", 0.0)
            + getattr(d, "fee", 0.0)
            for d in pdeals
        )

        sig = sig_map.get(pid)
        if sig:
            init_risk = sig.get("initial_risk_currency") or 0.0
            r_mult = (net_result / init_risk) if init_risk > 0.0 else 0.0

            # Determine exit reason
            if r_mult >= 1.6:
                reason = "TP"
            elif -0.15 <= r_mult <= 0.40 and net_result >= 0.0:
                reason = "BE"
            elif r_mult <= -0.80:
                reason = "SL"
            else:
                reason = "OTHER"

            matched_positions.append({
                "position_id": pid,
                "symbol": sig.get("symbol", getattr(pdeals[0], "symbol", "UNKNOWN")),
                "net_result": net_result,
                "r_multiple": r_mult,
                "initial_risk": init_risk,
                "confidence": sig.get("confidence", 0.0),
                "is_win": net_result > 0.0,
                "exit_reason": reason,
                "timestamp": sig.get("timestamp"),
            })
        else:
            unmatched_positions.append({
                "position_id": pid,
                "symbol": getattr(pdeals[0], "symbol", "UNKNOWN"),
                "net_result": net_result,
                "exit_reason": "UNMATCHED",
            })

    total_closed = len(matched_positions) + len(unmatched_positions)
    matched_count = len(matched_positions)
    unmatched_count = len(unmatched_positions)

    if matched_count > 0:
        wins = [p for p in matched_positions if p["is_win"]]
        losses = [p for p in matched_positions if not p["is_win"]]

        win_rate = (len(wins) / matched_count) * 100.0

        avg_win_usd = sum(p["net_result"] for p in wins) / len(wins) if wins else 0.0
        avg_loss_usd = sum(p["net_result"] for p in losses) / len(losses) if losses else 0.0

        avg_win_r = sum(p["r_multiple"] for p in wins) / len(wins) if wins else 0.0
        avg_loss_r = sum(p["r_multiple"] for p in losses) / len(losses) if losses else 0.0

        exp_usd = sum(p["net_result"] for p in matched_positions) / matched_count
        exp_r = sum(p["r_multiple"] for p in matched_positions) / matched_count

        # Max drawdown calculation
        cum_usd = 0.0
        peak_usd = 0.0
        max_dd_usd = 0.0

        cum_r = 0.0
        peak_r = 0.0
        max_dd_r = 0.0

        for p in matched_positions:
            cum_usd += p["net_result"]
            if cum_usd > peak_usd:
                peak_usd = cum_usd
            dd_u = peak_usd - cum_usd
            if dd_u > max_dd_usd:
                max_dd_usd = dd_u

            cum_r += p["r_multiple"]
            if cum_r > peak_r:
                peak_r = cum_r
            dd_r = peak_r - cum_r
            if dd_r > max_dd_r:
                max_dd_r = dd_r
    else:
        win_rate = 0.0
        avg_win_usd = 0.0
        avg_loss_usd = 0.0
        avg_win_r = 0.0
        avg_loss_r = 0.0
        exp_usd = 0.0
        exp_r = 0.0
        max_dd_usd = 0.0
        max_dd_r = 0.0

    # Exit reason percentages
    if total_closed > 0:
        tp_cnt = sum(1 for p in matched_positions if p["exit_reason"] == "TP")
        sl_cnt = sum(1 for p in matched_positions if p["exit_reason"] == "SL")
        be_cnt = sum(1 for p in matched_positions if p["exit_reason"] == "BE")
        other_cnt = sum(1 for p in matched_positions if p["exit_reason"] == "OTHER")

        pct_tp = (tp_cnt / total_closed) * 100.0
        pct_sl = (sl_cnt / total_closed) * 100.0
        pct_be = (be_cnt / total_closed) * 100.0
        pct_other = (other_cnt / total_closed) * 100.0
        pct_unmatched = (unmatched_count / total_closed) * 100.0
    else:
        pct_tp = pct_sl = pct_be = pct_other = pct_unmatched = 0.0

    # Confidence Calibration Table:
    # Buckets: 60%–65%, 65%–70%, 70%+
    # Requires >= 10 matched closed trades before displaying realized win rate
    bucket_defs = [
        {"name": "60%–65%", "min": 0.60, "max": 0.65},
        {"name": "65%–70%", "min": 0.65, "max": 0.70},
        {"name": "70%+",    "min": 0.70, "max": 1.01},
    ]

    calibration_table = []
    for b in bucket_defs:
        b_trades = [p for p in matched_positions if b["min"] <= p["confidence"] < b["max"]]
        cnt = len(b_trades)
        if cnt >= 10:
            b_wins = sum(1 for p in b_trades if p["is_win"])
            wr = (b_wins / cnt) * 100.0
            display = f"{wr:.1f}% ({b_wins}/{cnt} trades)"
            status = "CALIBRATED"
        else:
            wr = None
            display = f"Insufficient data ({cnt}/10 trades)"
            status = "PENDING"

        calibration_table.append({
            "bucket": b["name"],
            "count": cnt,
            "realized_win_rate": wr,
            "display": display,
            "status": status,
        })

    return {
        "total_closed": total_closed,
        "matched_count": matched_count,
        "unmatched_count": unmatched_count,
        "win_rate_pct": win_rate,
        "avg_win_usd": avg_win_usd,
        "avg_loss_usd": avg_loss_usd,
        "avg_win_r": avg_win_r,
        "avg_loss_r": avg_loss_r,
        "expectancy_usd": exp_usd,
        "expectancy_r": exp_r,
        "max_drawdown_usd": max_dd_usd,
        "max_drawdown_r": max_dd_r,
        "exit_reasons": {
            "pct_tp": pct_tp,
            "pct_sl": pct_sl,
            "pct_be": pct_be,
            "pct_other": pct_other,
            "pct_unmatched": pct_unmatched,
        },
        "calibration": calibration_table,
    }
