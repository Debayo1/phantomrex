"""Persistent signal and outcome tracking for demo small-profit experiments."""

import json
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "signals.db")
STRATEGIES = ("Trend pullback", "Range fade", "M1 breakout")
_TRADE_COLUMNS = {
    "decision_reason",
    "order_ticket",
    "position_id",
    "volume",
    "entry_time",
    "entry_price",
    "entry_candle_json",
    "stop_loss",
    "take_profit",
    "planned_risk_usd",
    "close_time",
    "close_price",
    "realized_net_usd",
    "exit_reason",
    "mae_price",
    "mfe_price",
    "mae_r",
    "mfe_r",
    "hold_seconds",
}


def _now():
    return datetime.now(timezone.utc).isoformat()


def _connect(db_path=DB_PATH):
    connection = sqlite3.connect(db_path, timeout=10)
    connection.execute("PRAGMA busy_timeout = 10000;")
    connection.execute("PRAGMA journal_mode = WAL;")
    connection.row_factory = sqlite3.Row
    return connection


@contextmanager
def _connection(db_path=DB_PATH):
    connection = _connect(db_path)
    try:
        yield connection
    finally:
        connection.close()


def init_db(db_path=DB_PATH):
    with _connection(db_path) as connection:
        connection.execute("""
            CREATE TABLE IF NOT EXISTS small_profit_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                signal_time TEXT NOT NULL,
                symbol TEXT NOT NULL,
                strategy TEXT NOT NULL,
                direction INTEGER,
                signal_price REAL,
                signal_open REAL,
                signal_high REAL,
                signal_low REAL,
                signal_close REAL,
                spread_points REAL,
                features_json TEXT NOT NULL DEFAULT '{}',
                status TEXT NOT NULL,
                decision_reason TEXT,
                order_ticket INTEGER,
                position_id INTEGER,
                volume REAL,
                entry_time TEXT,
                entry_price REAL,
                entry_candle_json TEXT,
                stop_loss REAL,
                take_profit REAL,
                planned_risk_usd REAL,
                close_time TEXT,
                close_price REAL,
                realized_net_usd REAL,
                exit_reason TEXT,
                mae_price REAL,
                mfe_price REAL,
                mae_r REAL,
                mfe_r REAL,
                hold_seconds REAL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                UNIQUE(symbol, strategy, signal_time)
            );
        """)
        existing_columns = {
            row["name"]
            for row in connection.execute("PRAGMA table_info(small_profit_history)")
        }
        added_columns = {
            "entry_candle_json": "TEXT",
            "mae_price": "REAL",
            "mfe_price": "REAL",
            "mae_r": "REAL",
            "mfe_r": "REAL",
            "hold_seconds": "REAL",
        }
        for column, column_type in added_columns.items():
            if column not in existing_columns:
                connection.execute(
                    f"ALTER TABLE small_profit_history ADD COLUMN {column} {column_type};"
                )
        connection.execute("""
            CREATE INDEX IF NOT EXISTS idx_small_profit_status
            ON small_profit_history(status, symbol, strategy);
        """)
        connection.execute("""
            CREATE INDEX IF NOT EXISTS idx_small_profit_position
            ON small_profit_history(position_id);
        """)
        connection.commit()


def record_bar(
    symbol,
    strategy,
    signal_time,
    status,
    direction=None,
    signal_price=None,
    signal_open=None,
    signal_high=None,
    signal_low=None,
    signal_close=None,
    spread_points=None,
    features=None,
    decision_reason=None,
    db_path=DB_PATH,
):
    """Insert or refresh one scheme's result for a closed M1 bar."""
    init_db(db_path)
    timestamp = pd_timestamp(signal_time)
    encoded_features = json.dumps(features or {}, sort_keys=True, default=str)
    now = _now()
    with _connection(db_path) as connection:
        cursor = connection.execute("""
            INSERT INTO small_profit_history (
                signal_time, symbol, strategy, direction, signal_price,
                signal_open, signal_high, signal_low, signal_close,
                spread_points, features_json, status, decision_reason,
                created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(symbol, strategy, signal_time) DO UPDATE SET
                direction = excluded.direction,
                signal_price = excluded.signal_price,
                signal_open = excluded.signal_open,
                signal_high = excluded.signal_high,
                signal_low = excluded.signal_low,
                signal_close = excluded.signal_close,
                spread_points = excluded.spread_points,
                features_json = CASE
                    WHEN small_profit_history.status IN ('SKIPPED', 'ORDER_FAILED', 'OPEN', 'CLOSED')
                    THEN small_profit_history.features_json
                    ELSE excluded.features_json
                END,
                status = CASE
                    WHEN small_profit_history.status IN ('SKIPPED', 'ORDER_FAILED', 'OPEN', 'CLOSED')
                    THEN small_profit_history.status
                    ELSE excluded.status
                END,
                decision_reason = CASE
                    WHEN small_profit_history.status IN ('SKIPPED', 'ORDER_FAILED', 'OPEN', 'CLOSED')
                    THEN small_profit_history.decision_reason
                    ELSE excluded.decision_reason
                END,
                updated_at = excluded.updated_at;
        """, (
            timestamp, symbol, strategy, direction, signal_price,
            signal_open, signal_high, signal_low, signal_close,
            spread_points, encoded_features, status, decision_reason, now, now,
        ))
        connection.commit()
        row = connection.execute("""
            SELECT id FROM small_profit_history
            WHERE symbol = ? AND strategy = ? AND signal_time = ?;
        """, (symbol, strategy, timestamp)).fetchone()
        return int(row["id"])


def get_record_status(record_id, db_path=DB_PATH):
    init_db(db_path)
    with _connection(db_path) as connection:
        row = connection.execute(
            "SELECT status FROM small_profit_history WHERE id = ?;",
            (record_id,),
        ).fetchone()
    if row is None:
        raise KeyError(f"Small-profit history row {record_id} does not exist")
    return row["status"]


def update_record(record_id, status, db_path=DB_PATH, **fields):
    invalid = set(fields).difference(_TRADE_COLUMNS)
    if invalid:
        raise ValueError(f"Unsupported small-profit fields: {sorted(invalid)}")
    init_db(db_path)
    assignments = ["status = ?", "updated_at = ?"]
    values = [status, _now()]
    for key, value in fields.items():
        assignments.append(f"{key} = ?")
        values.append(value)
    values.append(record_id)
    with _connection(db_path) as connection:
        cursor = connection.execute(
            f"UPDATE small_profit_history SET {', '.join(assignments)} WHERE id = ?;",
            values,
        )
        connection.commit()
        if cursor.rowcount != 1:
            raise KeyError(f"Small-profit history row {record_id} does not exist")


def get_open_records(symbol=None, db_path=DB_PATH):
    init_db(db_path)
    query = "SELECT * FROM small_profit_history WHERE status = 'OPEN'"
    values = []
    if symbol is not None:
        query += " AND symbol = ?"
        values.append(symbol)
    query += " ORDER BY entry_time"
    with _connection(db_path) as connection:
        return [dict(row) for row in connection.execute(query, values).fetchall()]


def get_summary(db_path=DB_PATH):
    init_db(db_path)
    with _connection(db_path) as connection:
        rows = connection.execute("""
            SELECT strategy,
                   COUNT(*) AS bars_seen,
                   SUM(CASE WHEN status = 'NO_SIGNAL' THEN 1 ELSE 0 END) AS no_signal_bars,
                   SUM(CASE WHEN status != 'NO_SIGNAL' THEN 1 ELSE 0 END) AS signal_bars,
                   SUM(CASE WHEN status = 'SKIPPED' THEN 1 ELSE 0 END) AS skipped,
                   SUM(CASE WHEN status = 'ORDER_FAILED' THEN 1 ELSE 0 END) AS order_failures,
                   SUM(CASE WHEN status = 'OPEN' THEN 1 ELSE 0 END) AS open_trades,
                   SUM(CASE WHEN status = 'CLOSED' THEN 1 ELSE 0 END) AS closed_trades,
                   SUM(CASE WHEN status = 'CLOSED' AND realized_net_usd > 0 THEN 1 ELSE 0 END) AS wins,
                   SUM(CASE WHEN status = 'CLOSED' AND realized_net_usd <= 0 THEN 1 ELSE 0 END) AS losses,
                   COALESCE(SUM(CASE WHEN status = 'CLOSED' THEN realized_net_usd ELSE 0 END), 0) AS net_usd,
                   AVG(CASE WHEN status = 'CLOSED' THEN realized_net_usd END) AS expectancy_usd
            FROM small_profit_history
            GROUP BY strategy
            ORDER BY strategy;
        """).fetchall()
    return [dict(row) for row in rows]


def get_recent_closed(limit=5, db_path=DB_PATH):
    if limit <= 0:
        raise ValueError("limit must be positive")
    init_db(db_path)
    with _connection(db_path) as connection:
        rows = connection.execute("""
            SELECT signal_time, symbol, strategy, direction, entry_price,
                   close_price, realized_net_usd, exit_reason, features_json,
                   entry_candle_json, mae_price, mfe_price, mae_r, mfe_r,
                   hold_seconds
            FROM small_profit_history
            WHERE status = 'CLOSED'
            ORDER BY close_time DESC
            LIMIT ?;
        """, (int(limit),)).fetchall()
    return [dict(row) for row in rows]


def get_candle_outcome_summary(db_path=DB_PATH):
    """Summarize candle anatomy and excursions by strategy, side, and outcome."""
    init_db(db_path)
    with _connection(db_path) as connection:
        rows = connection.execute("""
            SELECT strategy, direction, realized_net_usd, features_json,
                   mae_r, mfe_r, hold_seconds
            FROM small_profit_history
            WHERE status = 'CLOSED'
            ORDER BY strategy, direction, close_time;
        """).fetchall()

    groups = {}
    for row in rows:
        outcome = "WIN" if row["realized_net_usd"] > 0 else "LOSS"
        key = (row["strategy"], row["direction"], outcome)
        group = groups.setdefault(key, {
            "strategy": row["strategy"],
            "direction": row["direction"],
            "outcome": outcome,
            "closed_trades": 0,
            "body_fractions": [],
            "upper_wick_fractions": [],
            "lower_wick_fractions": [],
            "mae_r_values": [],
            "mfe_r_values": [],
            "hold_seconds_values": [],
        })
        group["closed_trades"] += 1
        features = json.loads(row["features_json"] or "{}")
        for field, column in (
            ("signal_body_fraction", "body_fractions"),
            ("signal_upper_wick_fraction", "upper_wick_fractions"),
            ("signal_lower_wick_fraction", "lower_wick_fractions"),
        ):
            value = features.get(field)
            if value is not None:
                group[column].append(float(value))
        for field, column in (
            ("mae_r", "mae_r_values"),
            ("mfe_r", "mfe_r_values"),
            ("hold_seconds", "hold_seconds_values"),
        ):
            value = row[field]
            if value is not None:
                group[column].append(float(value))

    result = []
    for group in groups.values():
        result.append({
            "strategy": group["strategy"],
            "direction": group["direction"],
            "outcome": group["outcome"],
            "closed_trades": group["closed_trades"],
            "avg_signal_body_fraction": _mean_or_none(group["body_fractions"]),
            "avg_signal_upper_wick_fraction": _mean_or_none(group["upper_wick_fractions"]),
            "avg_signal_lower_wick_fraction": _mean_or_none(group["lower_wick_fractions"]),
            "avg_mae_r": _mean_or_none(group["mae_r_values"]),
            "avg_mfe_r": _mean_or_none(group["mfe_r_values"]),
            "avg_hold_seconds": _mean_or_none(group["hold_seconds_values"]),
        })
    return result


def get_shadow_filter_summary(db_path=DB_PATH):
    """Compare realized closed trades in one shadow cohort with its reference cohort."""
    init_db(db_path)
    with _connection(db_path) as connection:
        rows = connection.execute("""
            SELECT symbol, strategy, direction, status, realized_net_usd,
                   features_json, mae_r, mfe_r, hold_seconds
            FROM small_profit_history
            WHERE symbol = 'XAUUSDm' AND strategy = 'M1 breakout' AND direction = -1;
        """).fetchall()

    shadow = []
    reference = []
    shadow_candidates = 0
    for row in rows:
        features = json.loads(row["features_json"] or "{}")
        assessment = features.get("shadow_filter", {})
        if assessment.get("rule_id") != "xau_m1_breakout_sell_00_utc_v1":
            continue
        would_skip = bool(assessment.get("would_skip"))
        if would_skip:
            shadow_candidates += 1
        if row["status"] != "CLOSED" or row["realized_net_usd"] is None:
            continue
        (shadow if would_skip else reference).append(row)

    def summarize(items):
        pnl = [float(row["realized_net_usd"]) for row in items]
        return {
            "closed_trades": len(items),
            "wins": sum(value > 0 for value in pnl),
            "net_usd": sum(pnl),
            "expectancy_usd": _mean_or_none(pnl),
            "avg_mae_r": _mean_or_none(
                [float(row["mae_r"]) for row in items if row["mae_r"] is not None]
            ),
            "avg_mfe_r": _mean_or_none(
                [float(row["mfe_r"]) for row in items if row["mfe_r"] is not None]
            ),
            "avg_hold_seconds": _mean_or_none(
                [
                    float(row["hold_seconds"])
                    for row in items if row["hold_seconds"] is not None
                ]
            ),
        }

    return {
        "rule_id": "xau_m1_breakout_sell_00_utc_v1",
        "shadow_candidates": shadow_candidates,
        "shadow": summarize(shadow),
        "reference": summarize(reference),
    }


def _mean_or_none(values):
    return sum(values) / len(values) if values else None


def get_regime_summary(db_path=DB_PATH):
    """Summarize closed trades by strategy and explainable market context."""
    init_db(db_path)
    with _connection(db_path) as connection:
        rows = connection.execute("""
            SELECT symbol, strategy, direction, realized_net_usd, features_json
            FROM small_profit_history
            WHERE status = 'CLOSED'
            ORDER BY close_time;
        """).fetchall()

    groups = {}
    for row in rows:
        features = json.loads(row["features_json"] or "{}")
        key = (
            row["symbol"],
            row["strategy"],
            features.get("regime", "unknown"),
            features.get("session", "unknown"),
            features.get("volatility_bucket", "unknown"),
        )
        group = groups.setdefault(key, {
            "symbol": key[0],
            "strategy": key[1],
            "regime": key[2],
            "session": key[3],
            "volatility_bucket": key[4],
            "closed_trades": 0,
            "wins": 0,
            "net_usd": 0.0,
            "returns": [],
        })
        pnl = float(row["realized_net_usd"] or 0.0)
        group["closed_trades"] += 1
        group["wins"] += int(pnl > 0)
        group["net_usd"] += pnl
        group["returns"].append(pnl)

    result = []
    for group in groups.values():
        returns = group.pop("returns")
        group["expectancy_usd"] = _mean_or_none(returns)
        group["win_rate"] = (
            group["wins"] / group["closed_trades"]
            if group["closed_trades"] else None
        )
        result.append(group)
    return sorted(result, key=lambda item: (
        item["symbol"], item["strategy"], item["regime"],
        item["session"], item["volatility_bucket"]
    ))


def get_direction_summary(db_path=DB_PATH):
    init_db(db_path)
    with _connection(db_path) as connection:
        rows = connection.execute("""
            SELECT strategy, direction,
                   COUNT(*) AS closed_trades,
                   SUM(CASE WHEN realized_net_usd > 0 THEN 1 ELSE 0 END) AS wins,
                   COALESCE(SUM(realized_net_usd), 0) AS net_usd,
                   AVG(realized_net_usd) AS expectancy_usd
            FROM small_profit_history
            WHERE status = 'CLOSED'
            GROUP BY strategy, direction
            ORDER BY strategy, direction DESC;
        """).fetchall()
    return [dict(row) for row in rows]


def pd_timestamp(value):
    """Normalize pandas/datetime/string timestamps to UTC ISO format."""
    from pandas import Timestamp

    timestamp = Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")
    return timestamp.isoformat()
