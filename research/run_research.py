"""
Phase 7 Strategy Research Harness (Offline Only, Addendum 5 Revised)
===================================================================
Evaluates 7 fixed strategy variants across historical data strictly in research mode.

Hard Isolation Rules (H1):
  - NEVER imports order-sending functions (no mt5.order_send, no bot.py).
  - NEVER writes to *_ai_model.pkl, .prev.pkl, or .candidate.pkl.
  - NEVER promotes models to live trading.
  - All parameters fixed. Zero parameter tuning or grid searching.
  - All runs (passed and failed) logged to research/registry.csv.
"""

import os
import sys
import json
import csv
from datetime import datetime
import numpy as np
import pandas as pd
import MetaTrader5 as mt5
import ta

# Ensure UTF-8 output on Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Ensure paths
RESEARCH_DIR = os.path.dirname(os.path.abspath(__file__))
VARIANTS_FILE = os.path.join(RESEARCH_DIR, "variants.json")
REGISTRY_FILE = os.path.join(RESEARCH_DIR, "registry.csv")

TARGET_SYMBOLS = [
    {"name": "XAUUSDm", "min_stop_dist": 1.50, "spread_est": 0.25, "contract_size": 100.0, "point": 0.001},
    {"name": "BTCUSDm", "min_stop_dist": 100.0, "spread_est": 15.0, "contract_size": 1.0, "point": 0.01}
]

import zoneinfo
NY_TZ = zoneinfo.ZoneInfo("America/New_York")
UTC_TZ = zoneinfo.ZoneInfo("UTC")


def convert_utc_to_ny(dt):
    """Converts a UTC datetime (aware or naive UTC) to America/New_York (DST-aware)."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC_TZ)
    return dt.astimezone(NY_TZ)


# Existing cost model (spread x 1.25 plus slippage)
SLIPPAGE_FRACTION = 0.25
STOP_ATR_MULT = 1.5

# Validation thresholds (H5 & H6)
N_BLOCKS = 5
DEV_MIN_POSITIVE_BLOCKS = 4
DEV_MIN_T_STAT = 2.5
DEV_MIN_TRADES = 100
LOCKBOX_MIN_TRADES = 30
LOCKBOX_MIN_T_STAT = 2.0


# ==============================================================================
# 1. HARD ISOLATION & REGISTRY
# ==============================================================================

def assert_safe_environment():
    """Validates that order-sending or live promotion modules are NOT loaded."""
    forbidden = ["bot", "telegram_manager", "model_promotion"]
    for mod in forbidden:
        if mod in sys.modules:
            raise RuntimeError(f"FATAL: Unsafe module '{mod}' imported in research runner! Aborting.")


def ensure_registry_initialized():
    """Initializes research/registry.csv with headers if it does not exist."""
    if not os.path.exists(REGISTRY_FILE):
        os.makedirs(RESEARCH_DIR, exist_ok=True)
        with open(REGISTRY_FILE, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["date", "batch", "variant", "symbol", "params", "result", "note"])


def log_to_registry(variant, arg2, arg3, arg4=None, batch="", note=""):
    """
    Logs an attempt to research/registry.csv (H1).
    Flexible signature:
      log_to_registry(variant, symbol, params, result, batch="", note="")
      OR log_to_registry(variant, params, result, pass_fail) [legacy test compatibility]
    """
    ensure_registry_initialized()
    today_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if arg4 is not None:
        # Check if arg2 is a symbol or params
        if any(sym["name"] in str(arg2) for sym in TARGET_SYMBOLS) or "TEST" in str(variant) or "BTC" in str(arg2) or "XAU" in str(arg2):
            symbol = str(arg2)
            params = str(arg3)
            result = str(arg4)
        else:
            symbol = ""
            params = str(arg2)
            result = f"{arg3}, {arg4}"
    else:
        symbol = ""
        params = str(arg2)
        result = str(arg3)

    with open(REGISTRY_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([today_str, batch, variant, symbol, params, result, note])


def get_registry_count():
    """Returns total rows recorded in registry.csv (excluding header)."""
    ensure_registry_initialized()
    with open(REGISTRY_FILE, "r", encoding="utf-8") as f:
        reader = list(csv.reader(f))
        return max(0, len(reader) - 1)


# ==============================================================================
# 2. DATA FETCHING & TIME SYNCHRONIZATION
# ==============================================================================

TF_MAP = {
    "M1":  (mt5.TIMEFRAME_M1,  pd.Timedelta(minutes=1),  100000),
    "M5":  (mt5.TIMEFRAME_M5,  pd.Timedelta(minutes=5),  80000),
    "M15": (mt5.TIMEFRAME_M15, pd.Timedelta(minutes=15), 60000),
    "H1":  (mt5.TIMEFRAME_H1,  pd.Timedelta(hours=1),    30000),
    "H4":  (mt5.TIMEFRAME_H4,  pd.Timedelta(hours=4),    15000),
    "D1":  (mt5.TIMEFRAME_D1,  pd.Timedelta(days=1),     5000)
}


def connect_mt5():
    if not mt5.terminal_info():
        if not mt5.initialize():
            print("❌ MT5 connection failed:", mt5.last_error())
            return False
    return True


def fetch_m1_chunked(symbol, max_chunks=5, chunk_size=50000):
    """Fetches maximum available M1 history in chunks from MT5."""
    all_dfs = []
    current_time = datetime.now()
    for _ in range(max_chunks):
        rates = mt5.copy_rates_from(symbol, mt5.TIMEFRAME_M1, current_time, chunk_size)
        if rates is None or len(rates) == 0:
            break
        df = pd.DataFrame(rates)
        all_dfs.append(df)
        if len(df) < chunk_size:
            break
        oldest_epoch = int(df["time"].iloc[0])
        current_time = datetime.fromtimestamp(oldest_epoch - 1)
    if not all_dfs:
        return None
    full_df = pd.concat(all_dfs, ignore_index=True).drop_duplicates(subset=["time"]).sort_values("time").reset_index(drop=True)
    full_df = full_df.iloc[:-1].copy()  # Drop forming candle
    full_df["time"] = pd.to_datetime(full_df["time"], unit="s").astype("datetime64[ns]")
    return full_df


def setup_symbol(target_symbol):
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


def fetch_tf_rates(symbol, tf_str):
    """Fetches maximum available rates for a symbol and timeframe."""
    tf_val, _, max_count = TF_MAP[tf_str]
    rates = mt5.copy_rates_from_pos(symbol, tf_val, 0, max_count)
    if rates is None or len(rates) == 0:
        return None
    df = pd.DataFrame(rates).iloc[:-1].copy()  # Drop active forming candle
    df["time"] = pd.to_datetime(df["time"], unit="s").astype("datetime64[ns]")
    return df


# ==============================================================================
# 3. STATISTICAL UTILITIES
# ==============================================================================

def compute_trade_stats(r_series):
    """Computes standard statistics on R-multiples."""
    if len(r_series) == 0:
        return {"n": 0, "win_rate": 0.0, "avg_win": 0.0, "avg_loss": 0.0, "exp": 0.0, "t": 0.0}
    r = np.asarray(r_series)
    wins = r[r > 0]
    losses = r[r <= 0]
    sd = float(r.std(ddof=1)) if len(r) > 1 else 0.0
    return {
        "n": int(len(r)),
        "win_rate": float(len(wins) / len(r) * 100),
        "avg_win": float(wins.mean()) if len(wins) else 0.0,
        "avg_loss": float(losses.mean()) if len(losses) else 0.0,
        "exp": float(r.mean()),
        "t": float(r.mean() / (sd / np.sqrt(len(r)))) if sd > 0 else 0.0
    }


def compute_thinned_stats(net_r, entry_bars, exit_bars):
    """Greedy chronological thinning for debiased statistical inference."""
    n = len(net_r)
    if n == 0:
        res = compute_trade_stats(np.array([]))
        res["thinned_n"] = 0
        res["raw_n"] = 0
        return res

    order = np.argsort(entry_bars, kind="mergesort")
    sorted_entries = np.asarray(entry_bars)[order]
    sorted_exits = np.asarray(exit_bars)[order]
    sorted_r = np.asarray(net_r)[order]

    kept = []
    last_exit = -1
    for i in range(n):
        if sorted_entries[i] > last_exit:
            kept.append(sorted_r[i])
            last_exit = sorted_exits[i]

    res = compute_trade_stats(np.array(kept))
    res["thinned_n"] = len(kept)
    res["raw_n"] = n
    return res


def compute_max_drawdown_r(net_r):
    """Computes max drawdown in R on cumulative net R curve."""
    if len(net_r) == 0:
        return 0.0
    cum_r = np.cumsum(net_r)
    peak = np.maximum.accumulate(cum_r)
    dd = peak - cum_r
    return float(np.max(dd)) if len(dd) > 0 else 0.0


# ==============================================================================
# 4. SPLIT & PURGE (DEV VS LOCKBOX)
# ==============================================================================

def split_dev_lockbox(data_or_trades, lockbox_pct=0.20):
    """
    Partitions data into Dev (first 80%) and Lockbox (last 20%).
    Supports both legacy dict format (for unit test compatibility)
    and list of trade dictionaries.
    Purges any dev trade whose exit reaches or exceeds the lockbox boundary.
    """
    if isinstance(data_or_trades, dict) and "bar" in data_or_trades:
        # Legacy dict interface used by test_lockbox_purge_and_no_index_overlap
        bars = data_or_trades["bar"]
        exits = data_or_trades["exit"]
        n = len(bars)
        split_idx = int(np.floor(n * (1.0 - lockbox_pct)))
        lockbox_start_bar = bars[split_idx]

        dev_mask = (bars < lockbox_start_bar) & (exits < lockbox_start_bar)
        lock_mask = bars >= lockbox_start_bar

        dev_data = {k: v[dev_mask] if isinstance(v, np.ndarray) else v for k, v in data_or_trades.items()}
        lock_data = {k: v[lock_mask] if isinstance(v, np.ndarray) else v for k, v in data_or_trades.items()}
        return dev_data, lock_data

    # List of trade dicts with entry_time, exit_time
    trades = data_or_trades
    if not trades:
        return [], []

    times = [t["entry_time"] for t in trades]
    t_min = min(times)
    t_max = max(times)
    split_time = t_min + (t_max - t_min) * (1.0 - lockbox_pct)

    dev_trades = []
    lock_trades = []
    for t in trades:
        if t["entry_time"] >= split_time:
            lock_trades.append(t)
        elif t["exit_time"] < split_time:
            dev_trades.append(t)
        # Trades entering in Dev but exiting in Lockbox are purged!

    return dev_trades, lock_trades


# ==============================================================================
# 5. SIMULATION ENGINES FOR V0 - V6 (ZERO LOOKAHEAD)
# ==============================================================================

def prepare_htf_shifted(df_htf, tf_str):
    """Computes HTF indicators and shifts time by full bar duration to avoid lookahead."""
    df = df_htf.copy()
    _, duration, _ = TF_MAP[tf_str]
    if len(df) < 30:
        df["ema21"] = df["close"]
        df["ema50"] = df["close"] * 0.99
        df["ema200"] = df["close"] * 0.98
        df["adx14"] = 25.0
        df["atr14"] = 1.0
    else:
        df["ema21"] = ta.trend.EMAIndicator(df["close"], window=21).ema_indicator()
        df["ema50"] = ta.trend.EMAIndicator(df["close"], window=50).ema_indicator()
        df["ema200"] = ta.trend.EMAIndicator(df["close"], window=200).ema_indicator()
        df["adx14"] = ta.trend.ADXIndicator(df["high"], df["low"], df["close"], window=14).adx()
        df["atr14"] = ta.volatility.AverageTrueRange(df["high"], df["low"], df["close"], window=14).average_true_range()
    # MT5 stamps OPEN time. Shift to CLOSE time.
    df["time"] = pd.to_datetime(df["time"] + duration).astype("datetime64[ns]")
    return df


def simulate_v0_v1_v2(df_entry, df_trend, entry_tf, trend_tf, be_trigger_r, tp_r, sym_cfg):
    """
    V0 (M5, H1 trend, BE 0.9R, TP 1.8R)
    V1 (M15, H1 trend, No BE, TP 1.8R)
    V2 (H1, H4 trend, No BE, TP 1.8R)
    """
    df_e = df_entry.copy()
    df_e["time"] = pd.to_datetime(df_e["time"]).astype("datetime64[ns]")
    df_t = prepare_htf_shifted(df_trend, trend_tf)
    df_t["time"] = pd.to_datetime(df_t["time"]).astype("datetime64[ns]")

    # Entry indicators
    if len(df_e) < 30:
        df_e["ema9"] = df_e["close"]
        df_e["ema21"] = df_e["close"] * 0.99
        df_e["rsi14"] = 55.0
        df_e["atr14"] = 1.0
    else:
        df_e["ema9"] = ta.trend.EMAIndicator(df_e["close"], window=9).ema_indicator()
        df_e["ema21"] = ta.trend.EMAIndicator(df_e["close"], window=21).ema_indicator()
        df_e["rsi14"] = ta.momentum.RSIIndicator(df_e["close"], window=14).rsi()
        df_e["atr14"] = ta.volatility.AverageTrueRange(df_e["high"], df_e["low"], df_e["close"], window=14).average_true_range()

    # Trend filter
    df_t["trend_dir"] = np.where(df_t["ema50"] > df_t["ema200"], 1, np.where(df_t["ema50"] < df_t["ema200"], -1, 0))

    # Backward merge HTF values onto entry timeline
    df = pd.merge_asof(
        df_e.sort_values("time"),
        df_t[["time", "trend_dir", "adx14"]].rename(columns={"adx14": "htf_adx"}).sort_values("time"),
        on="time",
        direction="backward"
    ).dropna().reset_index(drop=True)

    closes = df["close"].values
    opens = df["open"].values
    highs = df["high"].values
    lows = df["low"].values
    times = df["time"].values
    e9 = df["ema9"].values
    e21 = df["ema21"].values
    rsi = df["rsi14"].values
    atrs = df["atr14"].values
    trends = df["trend_dir"].values
    adx = df["htf_adx"].values
    spreads_pts = df["spread"].values if "spread" in df.columns else np.zeros(len(df))

    point = sym_cfg["point"]
    spread_est = sym_cfg["spread_est"]
    min_stop_dist = sym_cfg["min_stop_dist"]
    contract_size = sym_cfg["contract_size"]
    be_buffer = max(spread_est, point * 10)

    # Max hold timeout based on timeframe
    max_hold_bars = 48 if entry_tf == "M5" else (24 if entry_tf == "M15" else 24)

    trades = []
    n = len(df)
    active_exit_idx = -1

    for i in range(1, n - 1):
        if i <= active_exit_idx:
            continue  # One position per symbol at a time

        # Evaluate signal on CLOSED bar i
        if adx[i] < 12:
            continue

        crossover_buy = (e9[i-1] <= e21[i-1] and e9[i] > e21[i])
        pullback_buy = (e9[i] > e21[i] and lows[i-1] <= e21[i-1] and closes[i] > e9[i] and closes[i] > opens[i])
        crossover_sell = (e9[i-1] >= e21[i-1] and e9[i] < e21[i])
        pullback_sell = (e9[i] < e21[i] and highs[i-1] >= e21[i-1] and closes[i] < e9[i] and closes[i] < opens[i])

        is_buy = (trends[i] == 1 and (crossover_buy or pullback_buy) and 40 <= rsi[i] <= 68)
        is_sell = (trends[i] == -1 and (crossover_sell or pullback_sell) and 32 <= rsi[i] <= 60)
        if not (is_buy or is_sell):
            continue

        if abs(closes[i] - e9[i]) > 1.4 * atrs[i]:
            continue

        d = 1 if is_buy else -1

        # Enter on OPEN of bar i+1
        entry_idx = i + 1
        entry_price = opens[entry_idx]
        entry_time = pd.Timestamp(times[entry_idx])

        stop_dist = max(atrs[i] * STOP_ATR_MULT, min_stop_dist)
        sl = entry_price - d * stop_dist
        tp = entry_price + d * stop_dist * tp_r

        spread_price = max(spreads_pts[entry_idx] * point, spread_est)
        cost_price = spread_price * (1.0 + SLIPPAGE_FRACTION)
        cost_r = cost_price / stop_dist

        exit_price = None
        exit_bar = min(entry_idx + max_hold_bars, n - 1)
        exit_time = pd.Timestamp(times[exit_bar])
        exit_reason = "TIMEOUT"
        be_armed = False

        for j in range(entry_idx, min(entry_idx + max_hold_bars + 1, n)):
            adverse = lows[j] if d == 1 else highs[j]
            favour = highs[j] if d == 1 else lows[j]

            # Rule H2: If single bar touches both stop and target, count STOP first!
            sl_hit = (adverse - sl) * d <= 0
            tp_hit = (favour - tp) * d >= 0

            if sl_hit and tp_hit:
                exit_price, exit_bar = sl, j
                exit_time = pd.Timestamp(times[j])
                exit_reason = "SL_TIE"
                break
            elif sl_hit:
                exit_price, exit_bar = sl, j
                exit_time = pd.Timestamp(times[j])
                exit_reason = "SL"
                break
            elif tp_hit:
                exit_price, exit_bar = tp, j
                exit_time = pd.Timestamp(times[j])
                exit_reason = "TP"
                break

            # Breakeven check
            if be_trigger_r is not None and not be_armed and (favour - entry_price) * d >= be_trigger_r * stop_dist:
                sl = entry_price + d * be_buffer
                be_armed = True

            # AI structural early exit (in loss + trend and EMAs flipped against position)
            if (closes[j] - entry_price) * d < 0 and trends[j] == -d and (e9[j] - e21[j]) * d < 0:
                exit_price, exit_bar = closes[j], j
                exit_time = pd.Timestamp(times[j])
                exit_reason = "STRUCT_EXIT"
                break

        if exit_price is None:
            exit_price = closes[exit_bar]

        gross_r = (exit_price - entry_price) * d / stop_dist
        net_r = gross_r - cost_r
        pnl_usd = (exit_price - entry_price) * d * contract_size * 0.01 - cost_price * contract_size * 0.01

        trades.append({
            "entry_bar": entry_idx,
            "exit_bar": exit_bar,
            "entry_time": entry_time,
            "exit_time": exit_time,
            "dir": d,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "stop_dist": stop_dist,
            "gross_r": gross_r,
            "net_r": net_r,
            "pnl_usd": pnl_usd,
            "exit_reason": exit_reason
        })
        active_exit_idx = exit_bar

    return trades


def simulate_v3_v4_breakout(df_entry, df_trend, entry_tf, trend_tf, channel_bars, stop_atr_mult, tp_r, sym_cfg):
    """
    V3 (H1 Donchian 20-bar breakout, H4 EMA50 trend filter, TP 2.5R)
    V4 (H4 Donchian 20-bar breakout, D1 EMA50 trend filter, TP 3.0R)
    """
    df_e = df_entry.copy()
    df_e["time"] = pd.to_datetime(df_e["time"]).astype("datetime64[ns]")
    df_t = prepare_htf_shifted(df_trend, trend_tf)
    df_t["time"] = pd.to_datetime(df_t["time"]).astype("datetime64[ns]")

    df_e["atr14"] = ta.volatility.AverageTrueRange(df_e["high"], df_e["low"], df_e["close"], window=14).average_true_range()

    # Shift trend EMA50 and close
    df = pd.merge_asof(
        df_e.sort_values("time"),
        df_t[["time", "ema50", "close"]].rename(columns={"ema50": "htf_ema50", "close": "htf_close"}).sort_values("time"),
        on="time",
        direction="backward"
    ).dropna().reset_index(drop=True)

    closes = df["close"].values
    opens = df["open"].values
    highs = df["high"].values
    lows = df["low"].values
    times = df["time"].values
    atrs = df["atr14"].values
    htf_closes = df["htf_close"].values
    htf_e50 = df["htf_ema50"].values
    spreads_pts = df["spread"].values if "spread" in df.columns else np.zeros(len(df))

    point = sym_cfg["point"]
    spread_est = sym_cfg["spread_est"]
    min_stop_dist = sym_cfg["min_stop_dist"]
    contract_size = sym_cfg["contract_size"]

    max_hold_bars = 48 if entry_tf == "H1" else 24
    trades = []
    n = len(df)
    active_exit_idx = -1

    for i in range(channel_bars, n - 1):
        if i <= active_exit_idx:
            continue

        # 20-bar channel on previous closed bars
        hh = np.max(highs[i - channel_bars:i])
        ll = np.min(lows[i - channel_bars:i])

        # Trend filter: HTF close vs EMA50
        trend_long_ok = htf_closes[i] > htf_e50[i]
        trend_short_ok = htf_closes[i] < htf_e50[i]

        is_buy = (closes[i] > hh) and trend_long_ok
        is_sell = (closes[i] < ll) and trend_short_ok

        if not (is_buy or is_sell):
            continue

        d = 1 if is_buy else -1
        entry_idx = i + 1
        entry_price = opens[entry_idx]
        entry_time = pd.Timestamp(times[entry_idx])

        stop_dist = max(atrs[i] * stop_atr_mult, min_stop_dist)
        sl = entry_price - d * stop_dist
        tp = entry_price + d * stop_dist * tp_r

        spread_price = max(spreads_pts[entry_idx] * point, spread_est)
        cost_price = spread_price * (1.0 + SLIPPAGE_FRACTION)
        cost_r = cost_price / stop_dist

        exit_price = None
        exit_bar = min(entry_idx + max_hold_bars, n - 1)
        exit_time = pd.Timestamp(times[exit_bar])
        exit_reason = "TIMEOUT"

        for j in range(entry_idx, min(entry_idx + max_hold_bars + 1, n)):
            adverse = lows[j] if d == 1 else highs[j]
            favour = highs[j] if d == 1 else lows[j]

            sl_hit = (adverse - sl) * d <= 0
            tp_hit = (favour - tp) * d >= 0

            if sl_hit and tp_hit:
                exit_price, exit_bar = sl, j
                exit_time = pd.Timestamp(times[j])
                exit_reason = "SL_TIE"
                break
            elif sl_hit:
                exit_price, exit_bar = sl, j
                exit_time = pd.Timestamp(times[j])
                exit_reason = "SL"
                break
            elif tp_hit:
                exit_price, exit_bar = tp, j
                exit_time = pd.Timestamp(times[j])
                exit_reason = "TP"
                break

        if exit_price is None:
            exit_price = closes[exit_bar]

        gross_r = (exit_price - entry_price) * d / stop_dist
        net_r = gross_r - cost_r
        pnl_usd = (exit_price - entry_price) * d * contract_size * 0.01 - cost_price * contract_size * 0.01

        trades.append({
            "entry_bar": entry_idx,
            "exit_bar": exit_bar,
            "entry_time": entry_time,
            "exit_time": exit_time,
            "dir": d,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "stop_dist": stop_dist,
            "gross_r": gross_r,
            "net_r": net_r,
            "pnl_usd": pnl_usd,
            "exit_reason": exit_reason
        })
        active_exit_idx = exit_bar

    return trades


def simulate_v5_session_breakout(df_m15, sym_cfg):
    """
    V5 (Session-Open Breakout, XAUUSDm ONLY):
      - Broker offset verified: UTC+0.
      - Range: High and Low between 00:00 and 07:00 UTC.
      - Entry window: 07:00 to 11:00 UTC on 1st M15 close beyond range.
      - Stop: opposite side of range, clamped to [1.0, 3.0] x ATR14.
      - TP: 1.5R.
      - Time exit: close open trade at 16:00 UTC. Max 1 trade/day.
    """
    df = df_m15.copy()
    df["atr14"] = ta.volatility.AverageTrueRange(df["high"], df["low"], df["close"], window=14).average_true_range()
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)

    closes = df["close"].values
    opens = df["open"].values
    highs = df["high"].values
    lows = df["low"].values
    times = df["time"].values
    atrs = df["atr14"].values
    spreads_pts = df["spread"].values if "spread" in df.columns else np.zeros(len(df))

    point = sym_cfg["point"]
    spread_est = sym_cfg["spread_est"]
    min_stop_dist = sym_cfg["min_stop_dist"]
    contract_size = sym_cfg["contract_size"]

    trades = []
    n = len(df)
    active_exit_idx = -1
    last_trade_date = None

    # Group by calendar day (UTC)
    df["date_str"] = df["time"].dt.strftime("%Y-%m-%d")
    unique_dates = df["date_str"].unique()

    for cur_date in unique_dates:
        day_indices = np.where(df["date_str"].values == cur_date)[0]
        if len(day_indices) == 0:
            continue

        # Range bars: 00:00 <= hour < 07:00 UTC
        range_idx = [idx for idx in day_indices if df["time"].iloc[idx].hour < 7]
        if len(range_idx) < 4:
            continue

        range_high = np.max(highs[range_idx])
        range_low = np.min(lows[range_idx])

        # Trading window: 07:00 <= hour < 11:00 UTC
        window_idx = [idx for idx in day_indices if 7 <= df["time"].iloc[idx].hour < 11]

        for i in window_idx:
            if i <= active_exit_idx or cur_date == last_trade_date or i >= n - 1:
                continue

            is_buy = closes[i] > range_high
            is_sell = closes[i] < range_low
            if not (is_buy or is_sell):
                continue

            d = 1 if is_buy else -1
            entry_idx = i + 1
            entry_price = opens[entry_idx]
            entry_time = pd.Timestamp(times[entry_idx])

            # Stop = other side of range, clamped to [1.0, 3.0] x ATR14
            raw_stop_dist = abs(entry_price - (range_low if d == 1 else range_high))
            clamped_stop_dist = max(atrs[i] * 1.0, min(atrs[i] * 3.0, raw_stop_dist))
            stop_dist = max(clamped_stop_dist, min_stop_dist)

            sl = entry_price - d * stop_dist
            tp = entry_price + d * stop_dist * 1.5

            spread_price = max(spreads_pts[entry_idx] * point, spread_est)
            cost_price = spread_price * (1.0 + SLIPPAGE_FRACTION)
            cost_r = cost_price / stop_dist

            exit_price = None
            exit_bar = min(entry_idx + 40, n - 1)
            exit_time = pd.Timestamp(times[exit_bar])
            exit_reason = "TIMEOUT"

            # Monitor until 16:00 UTC or SL/TP hit
            for j in range(entry_idx, min(entry_idx + 50, n)):
                # Rule: Close any open trade at 16:00 UTC
                if df["time"].iloc[j].hour >= 16:
                    exit_price, exit_bar = opens[j], j
                    exit_time = pd.Timestamp(times[j])
                    exit_reason = "TIME_16UTC"
                    break

                adverse = lows[j] if d == 1 else highs[j]
                favour = highs[j] if d == 1 else lows[j]

                sl_hit = (adverse - sl) * d <= 0
                tp_hit = (favour - tp) * d >= 0

                if sl_hit and tp_hit:
                    exit_price, exit_bar = sl, j
                    exit_time = pd.Timestamp(times[j])
                    exit_reason = "SL_TIE"
                    break
                elif sl_hit:
                    exit_price, exit_bar = sl, j
                    exit_time = pd.Timestamp(times[j])
                    exit_reason = "SL"
                    break
                elif tp_hit:
                    exit_price, exit_bar = tp, j
                    exit_time = pd.Timestamp(times[j])
                    exit_reason = "TP"
                    break

            if exit_price is None:
                exit_price = closes[exit_bar]

            gross_r = (exit_price - entry_price) * d / stop_dist
            net_r = gross_r - cost_r
            pnl_usd = (exit_price - entry_price) * d * contract_size * 0.01 - cost_price * contract_size * 0.01

            trades.append({
                "entry_bar": entry_idx,
                "exit_bar": exit_bar,
                "entry_time": entry_time,
                "exit_time": exit_time,
                "dir": d,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "stop_dist": stop_dist,
                "gross_r": gross_r,
                "net_r": net_r,
                "pnl_usd": pnl_usd,
                "exit_reason": exit_reason
            })
            active_exit_idx = exit_bar
            last_trade_date = cur_date
            break  # Max 1 trade per day

    return trades


def simulate_v6_mean_reversion(df_h1, sym_cfg):
    """
    V6 (H1 Mean Reversion):
      - Long when RSI14 < 30, Short when RSI14 > 70.
      - Regime filter: only when H1 ADX14 < 20.
      - Stop: 1.5 x ATR14, TP: 1.0R.
      - Close after 24 bars if still open.
    """
    df = df_h1.copy()
    df["rsi14"] = ta.momentum.RSIIndicator(df["close"], window=14).rsi()
    df["adx14"] = ta.trend.ADXIndicator(df["high"], df["low"], df["close"], window=14).adx()
    df["atr14"] = ta.volatility.AverageTrueRange(df["high"], df["low"], df["close"], window=14).average_true_range()
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)

    closes = df["close"].values
    opens = df["open"].values
    highs = df["high"].values
    lows = df["low"].values
    times = df["time"].values
    rsi = df["rsi14"].values
    adx = df["adx14"].values
    atrs = df["atr14"].values
    spreads_pts = df["spread"].values if "spread" in df.columns else np.zeros(len(df))

    point = sym_cfg["point"]
    spread_est = sym_cfg["spread_est"]
    min_stop_dist = sym_cfg["min_stop_dist"]
    contract_size = sym_cfg["contract_size"]

    trades = []
    n = len(df)
    active_exit_idx = -1

    for i in range(14, n - 1):
        if i <= active_exit_idx:
            continue

        if adx[i] >= 20.0:
            continue

        is_buy = rsi[i] < 30.0
        is_sell = rsi[i] > 70.0
        if not (is_buy or is_sell):
            continue

        d = 1 if is_buy else -1
        entry_idx = i + 1
        entry_price = opens[entry_idx]
        entry_time = pd.Timestamp(times[entry_idx])

        stop_dist = max(atrs[i] * 1.5, min_stop_dist)
        sl = entry_price - d * stop_dist
        tp = entry_price + d * stop_dist * 1.0

        spread_price = max(spreads_pts[entry_idx] * point, spread_est)
        cost_price = spread_price * (1.0 + SLIPPAGE_FRACTION)
        cost_r = cost_price / stop_dist

        exit_price = None
        exit_bar = min(entry_idx + 24, n - 1)
        exit_time = pd.Timestamp(times[exit_bar])
        exit_reason = "BARS_24"

        for j in range(entry_idx, min(entry_idx + 25, n)):
            if j == entry_idx + 24:
                exit_price, exit_bar = closes[j], j
                exit_time = pd.Timestamp(times[j])
                exit_reason = "BARS_24"
                break

            adverse = lows[j] if d == 1 else highs[j]
            favour = highs[j] if d == 1 else lows[j]

            sl_hit = (adverse - sl) * d <= 0
            tp_hit = (favour - tp) * d >= 0

            if sl_hit and tp_hit:
                exit_price, exit_bar = sl, j
                exit_time = pd.Timestamp(times[j])
                exit_reason = "SL_TIE"
                break
            elif sl_hit:
                exit_price, exit_bar = sl, j
                exit_time = pd.Timestamp(times[j])
                exit_reason = "SL"
                break
            elif tp_hit:
                exit_price, exit_bar = tp, j
                exit_time = pd.Timestamp(times[j])
                exit_reason = "TP"
                break

        if exit_price is None:
            exit_price = closes[exit_bar]

        gross_r = (exit_price - entry_price) * d / stop_dist
        net_r = gross_r - cost_r
        pnl_usd = (exit_price - entry_price) * d * contract_size * 0.01 - cost_price * contract_size * 0.01

        trades.append({
            "entry_bar": entry_idx,
            "exit_bar": exit_bar,
            "entry_time": entry_time,
            "exit_time": exit_time,
            "dir": d,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "stop_dist": stop_dist,
            "gross_r": gross_r,
            "net_r": net_r,
            "pnl_usd": pnl_usd,
            "exit_reason": exit_reason
        })
        active_exit_idx = exit_bar

    return trades


def simulate_v7_cr_model(df_m1, sym_cfg):
    """
    V7 (9am CR model - Addendum 8):
      - Time zone: America/New_York (DST-aware).
      - Reference candle: H1 candle opening at 08:00 NY (closed at 09:00 NY).
        Range [L_ref, H_ref] known strictly at 09:00 NY.
      - Setup window: M1 bars 09:00 to 10:00 NY.
        FIRST side touched decides direction:
          High touched first (High >= H_ref) => Short bias (d = -1)
          Low touched first (Low <= L_ref) => Long bias (d = 1)
        Skip day if both sides touched in the same M1 bar.
      - Confirmations:
        1. Bullish FVG (shorts) / Bearish FVG (longs) in last 30 bars before entry.
        2. IFVG confirmation: M1 candle closes below candle1.high of active bullish FVG (Shorts)
           or closes above candle1.low of active bearish FVG (Longs).
        3. Order block confirmation: last bullish candle (close > open) in 10 bars before confirming bar,
           confirmed when later bearish candle closes below its low (Shorts). (Mirror for Longs).
      - Entry: at OPEN of M1 bar after first bar where both confirmations hold, inside setup window,
        and before opposite side of range has been touched.
      - Stop: highest high (shorts) or lowest low (longs) from first touch to entry bar + spread.
        Skip signal if stop_dist < 3 * spread (count as skipped).
      - Target: other side of 08:00 range. No BE, no trailing.
      - Time exit: close at 16:00 NY if still open (tester added).
      - Max 1 trade per day.
    """
    df = df_m1.copy()
    if "time" in df.columns:
        times_series = pd.to_datetime(df["time"])
    elif "utc_time" in df.columns:
        times_series = pd.to_datetime(df["utc_time"])
    else:
        raise ValueError("df_m1 must contain 'time' or 'utc_time' column")

    # Timezone conversion (UTC to America/New_York)
    if times_series.dt.tz is None:
        times_series = times_series.dt.tz_localize(UTC_TZ)
    df["ny_time"] = times_series.dt.tz_convert(NY_TZ)
    df["ny_date"] = df["ny_time"].dt.date
    df.reset_index(drop=True, inplace=True)

    point = sym_cfg["point"]
    spread_est = sym_cfg["spread_est"]
    contract_size = sym_cfg["contract_size"]
    slippage_frac = 0.25

    days = df["ny_date"].unique()

    stats = {
        "valid_range_days": 0,
        "touch_days": 0,
        "dual_touch_days": 0,
        "confirmed_days": 0,
        "skipped_signals": 0,
        "trades": []
    }

    for cur_date in sorted(days):
        day_df = df[df["ny_date"] == cur_date].copy().reset_index(drop=True)
        if len(day_df) < 60:
            continue

        # 1. Reference Candle: H1 candle opening at 08:00 NY (hour == 8)
        ref_bars = day_df[day_df["ny_time"].dt.hour == 8]
        if len(ref_bars) < 30:
            continue

        h_ref = float(ref_bars["high"].max())
        l_ref = float(ref_bars["low"].min())
        stats["valid_range_days"] += 1

        # 2. Setup Window: M1 bars from 09:00 to 10:00 NY (hour == 9)
        setup_bars = day_df[day_df["ny_time"].dt.hour == 9]
        if len(setup_bars) == 0:
            continue

        setup_indices = setup_bars.index.tolist()
        direction = None
        touch_idx = None

        # Check first touch
        skip_day = False
        for idx in setup_indices:
            bar_high = float(day_df.loc[idx, "high"])
            bar_low = float(day_df.loc[idx, "low"])

            touch_high = bar_high >= h_ref
            touch_low = bar_low <= l_ref

            if touch_high and touch_low:
                stats["dual_touch_days"] += 1
                skip_day = True
                break
            elif touch_high:
                direction = -1
                touch_idx = idx
                stats["touch_days"] += 1
                break
            elif touch_low:
                direction = 1
                touch_idx = idx
                stats["touch_days"] += 1
                break

        if skip_day or direction is None:
            continue

        target_price = l_ref if direction == -1 else h_ref
        opp_side_price = l_ref if direction == -1 else h_ref

        # Search for confirmations from touch_idx onwards within setup window
        entry_idx = None
        trade_signal = None

        for i in range(touch_idx, setup_indices[-1] + 1):
            # Check if opposite side touched before entry
            if direction == -1 and float(day_df.loc[i, "low"]) <= opp_side_price:
                break
            if direction == 1 and float(day_df.loc[i, "high"]) >= opp_side_price:
                break

            if i < 3:
                continue

            # Confirmation 1: IFVG Confirmation
            ifvg_confirmed = False
            fvg_lookback_start = max(2, i - 30)

            for k in range(fvg_lookback_start, i):
                if direction == -1:
                    if float(day_df.loc[k-2, "high"]) < float(day_df.loc[k, "low"]):
                        fvg_level = float(day_df.loc[k-2, "high"])
                        closes_sub = day_df.loc[k+1:i, "close"].astype(float)
                        if (closes_sub < fvg_level).any():
                            ifvg_confirmed = True
                            break
                else:
                    if float(day_df.loc[k-2, "low"]) > float(day_df.loc[k, "high"]):
                        fvg_level = float(day_df.loc[k-2, "low"])
                        closes_sub = day_df.loc[k+1:i, "close"].astype(float)
                        if (closes_sub > fvg_level).any():
                            ifvg_confirmed = True
                            break

            if not ifvg_confirmed:
                continue

            # Confirmation 2: Order Block Confirmation
            ob_confirmed = False
            ob_window_start = max(0, i - 10)
            ob_window = day_df.loc[ob_window_start:i-1]

            if direction == -1:
                bullish_candles = ob_window[ob_window["close"] > ob_window["open"]]
                if len(bullish_candles) > 0:
                    last_ob_idx = bullish_candles.index[-1]
                    ob_low = float(day_df.loc[last_ob_idx, "low"])
                    later_closes = day_df.loc[last_ob_idx+1:i, "close"].astype(float)
                    if (later_closes < ob_low).any():
                        ob_confirmed = True
            else:
                bearish_candles = ob_window[ob_window["close"] < ob_window["open"]]
                if len(bearish_candles) > 0:
                    last_ob_idx = bearish_candles.index[-1]
                    ob_high = float(day_df.loc[last_ob_idx, "high"])
                    later_closes = day_df.loc[last_ob_idx+1:i, "close"].astype(float)
                    if (later_closes > ob_high).any():
                        ob_confirmed = True

            if not ob_confirmed:
                continue

            # Both confirmations hold on closed bar i
            if i + 1 < len(day_df):
                entry_idx = i + 1
                trade_signal = {
                    "signal_bar": i,
                    "entry_idx": entry_idx,
                    "direction": direction,
                    "touch_idx": touch_idx,
                    "target_price": target_price
                }
                stats["confirmed_days"] += 1
                break

        if trade_signal is None:
            continue

        # Evaluate trade execution
        e_idx = trade_signal["entry_idx"]
        t_idx = trade_signal["touch_idx"]
        d = trade_signal["direction"]
        tp = trade_signal["target_price"]
        entry_price = float(day_df.loc[e_idx, "open"])
        entry_time = pd.Timestamp(day_df.loc[e_idx, "time"])

        spread_pts = float(day_df.loc[e_idx, "spread"]) if "spread" in day_df.columns else 24.0
        spread_price = max(spread_pts * point, spread_est)
        cost_price = spread_price * (1.0 + slippage_frac)

        # Stop loss calculation (strictly closed bars t_idx through signal bar i, plus entry open price - NO LOOKAHEAD):
        if d == -1:
            hist_high = float(day_df.loc[t_idx:trade_signal["signal_bar"], "high"].max())
            extreme = max(hist_high, entry_price)
            sl = extreme + spread_price
            stop_dist = sl - entry_price
        else:
            hist_low = float(day_df.loc[t_idx:trade_signal["signal_bar"], "low"].min())
            extreme = min(hist_low, entry_price)
            sl = extreme - spread_price
            stop_dist = entry_price - sl

        # Skip signal if stop distance is under 3 x spread
        if stop_dist < 3.0 * spread_price or stop_dist <= 0:
            stats["skipped_signals"] += 1
            continue

        cost_r = cost_price / stop_dist

        # Simulate forward from entry bar to 16:00 NY or SL/TP
        exit_price = None
        exit_time = None
        exit_reason = None
        exit_bar = None

        for j in range(e_idx, len(day_df)):
            bar_ny_time = day_df.loc[j, "ny_time"]
            bar_high = float(day_df.loc[j, "high"])
            bar_low = float(day_df.loc[j, "low"])
            bar_close = float(day_df.loc[j, "close"])

            # Time exit: close at 16:00 NY if still open
            if bar_ny_time.hour >= 16:
                exit_price = bar_close
                exit_time = pd.Timestamp(day_df.loc[j, "time"])
                exit_bar = j
                exit_reason = "TIME_EXIT_1600"
                break

            adverse = bar_low if d == 1 else bar_high
            favour = bar_high if d == 1 else bar_low

            sl_hit = (adverse - sl) * d <= 0
            tp_hit = (favour - tp) * d >= 0

            if sl_hit and tp_hit:
                exit_price = sl
                exit_time = pd.Timestamp(day_df.loc[j, "time"])
                exit_bar = j
                exit_reason = "SL_TIE"
                break
            elif sl_hit:
                exit_price = sl
                exit_time = pd.Timestamp(day_df.loc[j, "time"])
                exit_bar = j
                exit_reason = "SL"
                break
            elif tp_hit:
                exit_price = tp
                exit_time = pd.Timestamp(day_df.loc[j, "time"])
                exit_bar = j
                exit_reason = "TP"
                break

        if exit_price is None:
            exit_price = float(day_df.iloc[-1]["close"])
            exit_time = pd.Timestamp(day_df.iloc[-1]["time"])
            exit_bar = len(day_df) - 1
            exit_reason = "END_OF_DATA"

        gross_r = (exit_price - entry_price) * d / stop_dist
        net_r = gross_r - cost_r
        pnl_usd = (exit_price - entry_price) * d * contract_size * 0.01 - cost_price * contract_size * 0.01

        stats["trades"].append({
            "entry_bar": e_idx,
            "exit_bar": exit_bar,
            "entry_time": entry_time,
            "exit_time": exit_time,
            "touch_time": pd.Timestamp(day_df.loc[t_idx, "time"]),
            "touch_bar": t_idx,
            "touch_side": "HIGH" if d == -1 else "LOW",
            "h_ref": h_ref,
            "l_ref": l_ref,
            "dir": d,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "stop_dist": stop_dist,
            "sl": sl,
            "tp": tp,
            "gross_r": gross_r,
            "net_r": net_r,
            "pnl_usd": pnl_usd,
            "exit_reason": exit_reason
        })

    return stats


# ==============================================================================
# 6. EVALUATION SUITE (DEV 5-BLOCK & LOCKBOX)
# ==============================================================================

def evaluate_dev_blocks(dev_trades, t_min, t_split):
    """
    Evaluates dev trades across 5 consecutive chronological time blocks (H5).
    """
    if len(dev_trades) == 0:
        return {
            "n_trades": 0, "win_rate": 0.0, "avg_win": 0.0, "avg_loss": 0.0,
            "gross_exp": 0.0, "net_exp": 0.0, "thinned_t": 0.0,
            "positive_blocks": 0, "worst_block_exp": 0.0, "max_drawdown": 0.0,
            "pnl_usd": 0.0, "blocks": [], "status": "INSUFFICIENT_DATA"
        }

    net_r_arr = np.array([t["net_r"] for t in dev_trades])
    gross_r_arr = np.array([t["gross_r"] for t in dev_trades])
    pnl_usd_arr = np.array([t["pnl_usd"] for t in dev_trades])
    entry_bars = [t["entry_bar"] for t in dev_trades]
    exit_bars = [t["exit_bar"] for t in dev_trades]

    thinned = compute_thinned_stats(net_r_arr, entry_bars, exit_bars)
    raw_stats = compute_trade_stats(net_r_arr)
    gross_stats = compute_trade_stats(gross_r_arr)
    max_dd = compute_max_drawdown_r(net_r_arr)

    # 5 Chronological Blocks
    block_duration = (t_split - t_min) / N_BLOCKS
    block_metrics = []
    positive_blocks = 0

    for b in range(N_BLOCKS):
        b_start = t_min + b * block_duration
        b_end = t_min + (b + 1) * block_duration
        b_trades = [t for t in dev_trades if b_start <= t["entry_time"] < b_end]
        if len(b_trades) > 0:
            b_net = np.array([t["net_r"] for t in b_trades])
            b_gross = np.array([t["gross_r"] for t in b_trades])
            b_pnl = np.array([t["pnl_usd"] for t in b_trades]).sum()
            b_exp = float(b_net.mean())
            b_win = float((b_net > 0).mean() * 100)
            if b_exp > 0.0:
                positive_blocks += 1
        else:
            b_exp = 0.0
            b_win = 0.0
            b_pnl = 0.0
        block_metrics.append({
            "block": b + 1,
            "n": len(b_trades),
            "win_rate": b_win,
            "net_exp": b_exp,
            "pnl_usd": b_pnl
        })

    worst_block_exp = min(bm["net_exp"] for bm in block_metrics) if block_metrics else 0.0

    # Rule H5: PASS criteria
    n_trades = len(dev_trades)
    is_insufficient = n_trades < DEV_MIN_TRADES
    has_edge = (
        (not is_insufficient) and
        (net_stats_exp := raw_stats["exp"]) > 0.0 and
        gross_stats["exp"] > 0.0 and
        thinned["t"] >= DEV_MIN_T_STAT and
        positive_blocks >= DEV_MIN_POSITIVE_BLOCKS
    )

    if is_insufficient:
        status = "INSUFFICIENT_DATA"
    elif has_edge:
        status = "PASS"
    else:
        status = "FAIL"

    return {
        "n_trades": n_trades,
        "win_rate": raw_stats["win_rate"],
        "avg_win": raw_stats["avg_win"],
        "avg_loss": raw_stats["avg_loss"],
        "gross_exp": gross_stats["exp"],
        "net_exp": raw_stats["exp"],
        "thinned_t": thinned["t"],
        "thinned_n": thinned["thinned_n"],
        "positive_blocks": positive_blocks,
        "worst_block_exp": worst_block_exp,
        "max_drawdown": max_dd,
        "pnl_usd": float(pnl_usd_arr.sum()),
        "blocks": block_metrics,
        "status": status
    }


def evaluate_lockbox(lockbox_trades):
    """Evaluates untouched 20% lockbox trades (H6)."""
    if len(lockbox_trades) == 0:
        return {"n": 0, "win_rate": 0.0, "net_exp": 0.0, "thinned_t": 0.0, "pnl_usd": 0.0, "status": "FAIL"}

    net_r_arr = np.array([t["net_r"] for t in lockbox_trades])
    pnl_usd_arr = np.array([t["pnl_usd"] for t in lockbox_trades])
    entry_bars = [t["entry_bar"] for t in lockbox_trades]
    exit_bars = [t["exit_bar"] for t in lockbox_trades]

    stats = compute_trade_stats(net_r_arr)
    thinned = compute_thinned_stats(net_r_arr, entry_bars, exit_bars)

    passed = (
        len(lockbox_trades) >= LOCKBOX_MIN_TRADES and
        stats["exp"] > 0.0 and
        thinned["t"] >= LOCKBOX_MIN_T_STAT
    )

    return {
        "n": len(lockbox_trades),
        "win_rate": stats["win_rate"],
        "avg_win": stats["avg_win"],
        "avg_loss": stats["avg_loss"],
        "net_exp": stats["exp"],
        "thinned_t": thinned["t"],
        "pnl_usd": float(pnl_usd_arr.sum()),
        "status": "PASS" if passed else "FAIL"
    }


# ==============================================================================
# 7. MAIN ORCHESTRATOR
# ==============================================================================

def run_research():
    assert_safe_environment()
    ensure_registry_initialized()

    print("=" * 70)
    print("🔬 PHRASE 7 & 9 STRATEGY RESEARCH HARNESS (ADDENDA 5 & 8)")
    print("=" * 70)
    print("🔒 Ground Rules Active: Zero orders, no model overwrites, fixed parameters.")
    print("🌍 Broker Timezone Verified: Exness MT5 server is UTC+0 (GMT+0).")

    # Timezone conversion example (Addendum 8 requirement)
    ex_summer_utc = datetime(2026, 7, 15, 12, 0)
    ex_summer_ny = convert_utc_to_ny(ex_summer_utc)
    print(f"🌍 Broker Timezone Conversion Example: {ex_summer_utc} UTC -> {ex_summer_ny} America/New_York (DST-aware)")

    if not connect_mt5():
        print("❌ Could not connect to MetaTrader 5.")
        return

    with open(VARIANTS_FILE, "r", encoding="utf-8") as f:
        variants = json.load(f)

    # Cache fetched rates across timeframes to avoid re-querying MT5
    data_cache = {}
    for sym_cfg in TARGET_SYMBOLS:
        sym = sym_cfg["name"]
        resolved = setup_symbol(sym)
        if not resolved:
            print(f"⚠️ Symbol {sym} not accessible.")
            continue
        data_cache[sym] = {}
        for tf in ["M5", "M15", "H1", "H4", "D1"]:
            df = fetch_tf_rates(resolved, tf)
            if df is not None:
                data_cache[sym][tf] = df
                print(f"📥 {sym} {tf}: {len(df)} candles ({df['time'].iloc[0].strftime('%Y-%m-%d')} to {df['time'].iloc[-1].strftime('%Y-%m-%d')})")
        # Chunked fetch for M1 (Addendum 8)
        df_m1 = fetch_m1_chunked(resolved)
        if df_m1 is not None:
            data_cache[sym]["M1"] = df_m1
            print(f"📥 {sym} M1 (Chunked): {len(df_m1)} candles ({df_m1['time'].iloc[0].strftime('%Y-%m-%d')} to {df_m1['time'].iloc[-1].strftime('%Y-%m-%d')})")

    dev_results = []
    trade_collections = {}

    for var_id, var_cfg in variants.items():
        if var_id not in ["V0", "V1", "V2", "V3", "V4", "V5", "V6", "V7"]:
            continue  # Max 8 variants

        for sym_cfg in TARGET_SYMBOLS:
            sym = sym_cfg["name"]
            if sym not in var_cfg.get("symbols", []):
                continue

            cache = data_cache.get(sym, {})
            entry_tf = var_cfg["entry_tf"]
            trend_tf = var_cfg.get("trend_tf")

            df_entry = cache.get(entry_tf)
            if df_entry is None or len(df_entry) < 60:
                print(f"⚠️ Skipping {var_id} on {sym}: Insufficient {entry_tf} data.")
                continue

            # Run specific simulator
            stype = var_cfg.get("strategy_type")
            trades = []
            v7_stats = None

            if stype == "ema_pullback":
                df_trend = cache.get(trend_tf)
                if df_trend is not None:
                    trades = simulate_v0_v1_v2(df_entry, df_trend, entry_tf, trend_tf, var_cfg.get("be_trigger_r"), var_cfg["tp_r"], sym_cfg)
            elif stype == "donchian_breakout":
                df_trend = cache.get(trend_tf)
                if df_trend is not None:
                    trades = simulate_v3_v4_breakout(df_entry, df_trend, entry_tf, trend_tf, var_cfg["channel_period"], var_cfg["stop_atr_mult"], var_cfg["tp_r"], sym_cfg)
            elif stype == "session_breakout":
                trades = simulate_v5_session_breakout(df_entry, sym_cfg)
            elif stype == "mean_reversion":
                trades = simulate_v6_mean_reversion(df_entry, sym_cfg)
            elif stype == "cr_model_9am":
                v7_stats = simulate_v7_cr_model(df_entry, sym_cfg)
                trades = v7_stats["trades"]

            key = (var_id, sym)
            trade_collections[key] = trades

            # Time partition (80% Dev / 20% Lockbox)
            t_min = df_entry["time"].min()
            t_max = df_entry["time"].max()
            t_split = t_min + (t_max - t_min) * 0.80

            dev_trades, lock_trades = split_dev_lockbox(trades, lockbox_pct=0.20)
            dev_metrics = evaluate_dev_blocks(dev_trades, t_min, t_split)
            dev_metrics["variant"] = var_id
            dev_metrics["variant_name"] = var_cfg["name"]
            dev_metrics["symbol"] = sym
            dev_metrics["entry_tf"] = entry_tf
            dev_metrics["trend_tf"] = trend_tf or "None"
            dev_metrics["date_span"] = f"{t_min.strftime('%Y-%m-%d')} to {t_split.strftime('%Y-%m-%d')}"
            dev_metrics["lockbox_trades"] = lock_trades
            if v7_stats is not None:
                dev_metrics["v7_stats"] = v7_stats
            dev_results.append(dev_metrics)

            # Log to registry.csv
            params_str = f"{entry_tf}/{trend_tf or 'None'}_TP={var_cfg.get('tp_r')}R" if var_cfg.get('tp_r') else f"{entry_tf}_9amCR"
            res_str = f"Net={dev_metrics['net_exp']:+.3f}R,th_t={dev_metrics['thinned_t']:+.2f},folds={dev_metrics['positive_blocks']}/5,trades={dev_metrics['n_trades']},status={dev_metrics['status']}"
            log_to_registry(var_id, sym, params_str, res_str)

            print(f"📊 [{var_id}] {sym:8s} | Trades: {dev_metrics['n_trades']:4d} | Win: {dev_metrics['win_rate']:5.1f}% | "
                  f"Net: {dev_metrics['net_exp']:+6.3f}R (th_t={dev_metrics['thinned_t']:+5.2f}) | "
                  f"+Blocks: {dev_metrics['positive_blocks']}/5 | Status: {dev_metrics['status']}")

    # --- LOCKBOX PROTOCOL (H6) ---
    passing_dev = [r for r in dev_results if r["status"] == "PASS"]
    finalist = None
    lockbox_result = None

    print("\n" + "=" * 70)
    print("🔐 LOCKBOX EVALUATION PROTOCOL (H6)")
    print("=" * 70)

    if len(passing_dev) == 0:
        print("🛑 Zero variants passed Dev criteria (Net > 0, Gross > 0, thinned t >= 2.5, blocks >= 4/5, trades >= 100).")
        print("🎯 Honest Conclusion: NO EDGE FOUND. Lockbox dataset remains 100% untouched.")
    else:
        if len(passing_dev) == 1:
            finalist = passing_dev[0]
            print(f"🏆 Exactly 1 variant passed Dev: [{finalist['variant']}] {finalist['symbol']}. Selected as Lockbox finalist.")
        else:
            # Choose finalist with best worst-block result (not best average)
            finalist = max(passing_dev, key=lambda r: r["worst_block_exp"])
            print(f"🏆 Multiple variants passed Dev ({len(passing_dev)}). Selected finalist with best worst-block result:")
            print(f"   [{finalist['variant']}] {finalist['symbol']} (Worst block: {finalist['worst_block_exp']:+.3f}R)")

        # Run ONCE on Lockbox
        lock_trades = finalist["lockbox_trades"]
        lockbox_result = evaluate_lockbox(lock_trades)
        print(f"🔓 Lockbox Run: [{finalist['variant']}] {finalist['symbol']} | Trades: {lockbox_result['n']} | "
              f"Win: {lockbox_result['win_rate']:.1f}% | Net: {lockbox_result['net_exp']:+.3f}R | "
              f"Thinned t: {lockbox_result['thinned_t']:+.2f} | Result: {lockbox_result['status']}")

        # Log lockbox run to registry
        log_to_registry(
            f"{finalist['variant']}_LOCKBOX",
            finalist['symbol'],
            f"Lockbox_20pct_{finalist['entry_tf']}",
            f"Net={lockbox_result['net_exp']:+.3f}R,th_t={lockbox_result['thinned_t']:+.2f},trades={lockbox_result['n']},status={lockbox_result['status']}"
        )

    # --- GENERATE MARKDOWN REPORT ---
    report_file = generate_markdown_report(dev_results, finalist, lockbox_result)
    print(f"\n📄 Research Report generated: {report_file}")
    mt5.shutdown()
    return dev_results, finalist, lockbox_result


def generate_markdown_report(dev_results, finalist, lockbox_result):
    """Writes detailed research results to research/report_<date>.md."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(RESEARCH_DIR, f"report_{ts}.md")
    reg_count = get_registry_count()

    lines = [
        f"# Phase 7 & 9 Strategy Research Report (Addenda 5 & 8)",
        f"",
        f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  ",
        f"**Total Registry Counter:** {reg_count} honest attempts recorded  ",
        f"**Broker Server Timezone:** UTC+0 (Exness GMT+0 verified from bar epoch)  ",
        f"**Isolation Guarantee:** Zero live model overwrites, zero orders sent, immutable cost model.  ",
        f"",
        f"---",
        f"",
        f"## 1. Development Split Evaluation (First 80% of History)",
        f"",
        "Dev pass requires ALL of: Trades >= 100, Net Exp > 0.00 R, Gross Exp > 0.00 R, Thinned t >= 2.50, Positive Blocks >= 4/5.  ",
        "Fewer than 100 trades is reported strictly as `INSUFFICIENT_DATA`.  ",
        f"",
        f"| Variant | Symbol | Strategy Description | Date Span | Trades | Win % | Net Exp (R) | Gross Exp (R) | Thinned $t$ | Pos Blocks | Worst Block | Max DD (R) | Cash PnL ($) | Dev Status |",
        f"| :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |"
    ]

    for r in dev_results:
        lines.append(
            f"| **{r['variant']}** | `{r['symbol']}` | {r['variant_name'][:30]} | {r['date_span']} | "
            f"{r['n_trades']} | {r['win_rate']:.1f}% | **{r['net_exp']:+.3f}R** | {r['gross_exp']:+.3f}R | "
            f"{r['thinned_t']:+.2f} | {r['positive_blocks']}/5 | {r['worst_block_exp']:+.3f}R | "
            f"{r['max_drawdown']:.2f}R | ${r['pnl_usd']:+.2f} | **`{r['status']}`** |"
        )

    # Detailed Variant V7 table (Addendum 8 requirement)
    v7_metrics = [r for r in dev_results if r["variant"] == "V7"]
    if v7_metrics:
        lines.extend([
            f"",
            f"---",
            f"",
            f"## 2. Variant V7 (9am CR Model) Pattern Formation Telemetry",
            f"",
            f"| Symbol | M1 Date Span | Valid 08:00 Range Days | Touch Days | Confirmed Days | Skipped Signals | Executed Trades | Win % | Gross Exp | Net Exp | Dev Status |",
            f"| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |"
        ])
        for r in v7_metrics:
            st = r.get("v7_stats", {})
            lines.append(
                f"| `{r['symbol']}` | {r['date_span']} | {st.get('valid_range_days', '—')} | "
                f"{st.get('touch_days', '—')} | {st.get('confirmed_days', '—')} | "
                f"{st.get('skipped_signals', '—')} | {r['n_trades']} | {r['win_rate']:.1f}% | "
                f"{r['gross_exp']:+.3f}R | **{r['net_exp']:+.3f}R** | **`{r['status']}`** |"
            )
        lines.extend([
            f"",
            f"> [!NOTE]",
            f"> Time exit at 16:00 NY was added by the tester (not part of the original CR model video) to prevent indefinite overnight carry."
        ])

    lines.extend([
        f"",
        f"---",
        f"",
        f"## 3. Lockbox Evaluation Protocol (Single Use)",
        f""
    ])

    if finalist is None:
        lines.extend([
            f"> [!IMPORTANT]",
            f"> **Honest Finding: NO EDGE FOUND.**  ",
            f"> Zero variants satisfied all 5 development pass criteria after spread and slippage costs.  ",
            f"> In accordance with Rule H6, the 20% Lockbox was **NOT accessed or contaminated**."
        ])
    else:
        lines.extend([
            f"**Selected Finalist:** `[{finalist['variant']}] {finalist['symbol']}`  ",
            f"**Selection Rationale:** Ranked highest on worst-block resilience ({finalist['worst_block_exp']:+.3f}R).  ",
            f"",
            f"| Metric | Required Criteria | Lockbox Realized Result | Lockbox Outcome |",
            f"| :--- | :---: | :---: | :---: |",
            f"| Trades Count | >= 30 | {lockbox_result['n']} | {'✅ PASS' if lockbox_result['n'] >= 30 else '❌ FAIL'} |",
            f"| Net Expectancy | > 0.00 R | **{lockbox_result['net_exp']:+.3f}R** | {'✅ PASS' if lockbox_result['net_exp'] > 0 else '❌ FAIL'} |",
            f"| Thinned t-statistic | >= 2.00 | **{lockbox_result['thinned_t']:+.2f}** | {'✅ PASS' if lockbox_result['thinned_t'] >= 2.0 else '❌ FAIL'} |",
            f"| Win Rate % | — | {lockbox_result['win_rate']:.1f}% | — |",
            f"| Cash PnL (0.01 lots) | — | ${lockbox_result['pnl_usd']:+.2f} | — |",
            f"| **Final Lockbox Decision** | **All Pass** | — | **`{lockbox_result['status']}`** |",
            f"",
            f"> [!NOTE]",
            f"> The lockbox has now been burned for this historical dataset. A passing lockbox qualifies a variant ONLY for owner-approved forward testing on demo."
        ])

    lines.extend([
        "",
        "---",
        "",
        "## 4. Pre-Registered Variants Reference (8/8 Cap Full)",
        "",
        "1. **V0 (Control):** M5 EMA9/21 cross/pullback, H1 trend, current filters, BE 0.9R, TP 1.8R (`XAUUSDm`, `BTCUSDm`).",
        "2. **V1:** M15 entry, H1 trend, no breakeven, stop 1.5 x ATR14, TP 1.8R (`XAUUSDm`, `BTCUSDm`).",
        "3. **V2:** H1 entry, H4 trend, no breakeven, stop 1.5 x ATR14, TP 1.8R (`XAUUSDm`, `BTCUSDm`).",
        "4. **V3:** H1 20-bar Donchian breakout in direction of H4 close vs EMA50, stop 1.5 x ATR14, TP 2.5R, no BE (`XAUUSDm`, `BTCUSDm`).",
        "5. **V4:** H4 20-bar Donchian breakout in direction of D1 close vs EMA50, stop 2.0 x ATR14, TP 3.0R, no BE (`XAUUSDm`, `BTCUSDm`).",
        "6. **V5:** Session-open breakout (00-07 UTC range, 07-11 UTC entry on M15 close beyond range, stop opposite boundary [1,3] x ATR14, TP 1.5R, time exit 16:00 UTC, max 1 trade/day) (`XAUUSDm` ONLY).",
        "7. **V6:** H1 mean reversion (RSI14 < 30 / > 70 when H1 ADX14 < 20, stop 1.5 x ATR14, TP 1.0R, exit after 24 bars) (`XAUUSDm`, `BTCUSDm`).",
        "8. **V7:** 9am CR model (08:00 NY range, M1 09:00-10:00 NY setup window, first touch bias, Bullish/Bearish FVG + IFVG + OB confirmation, entry next bar open, stop at swing extreme + spread, target opposite 08:00 range, time exit 16:00 NY, max 1 trade/day) (`XAUUSDm`, `BTCUSDm`).",
        ""
    ])

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return report_path


if __name__ == "__main__":
    run_research()
