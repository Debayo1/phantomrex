"""
Phase 7 Strategy Research Harness (Offline Only, Addendum 5)
============================================================
Evaluates strategy variants across historical data strictly in research mode.
Hard Isolation Rules (E1):
  - NEVER imports order-sending functions (no mt5.order_send, no bot.py).
  - NEVER writes to *_ai_model.pkl, .prev.pkl, or .candidate.pkl.
  - NEVER promotes models to live trading.
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
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score
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
    {"name": "XAUUSDm", "min_stop_dist": 1.50, "spread_est": 0.25},
    {"name": "BTCUSDm", "min_stop_dist": 100.0, "spread_est": 15.0}
]

# Existing cost model (spread x 1.25 + slippage)
SLIPPAGE_FRACTION = 0.25
STOP_ATR_MULT = 1.5

# Validation thresholds
N_FOLDS = 5
MIN_POSITIVE_FOLDS = 4
DEV_MIN_T_STAT = 2.5
DEV_MIN_TRADES = 100
LOCKBOX_MIN_TRADES = 30
LOCKBOX_MIN_T_STAT = 2.0
THRESHOLD_GRID = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75]

FEATURE_COLUMNS = [
    "feat_rsi_aligned",
    "feat_ema_spread_aligned",
    "feat_body_momentum_aligned",
    "feat_rejection_wick_aligned",
    "feat_opposite_wick_aligned",
    "feat_dist_ema9_aligned",
    "feat_macro_adx",
    "feat_atr_ratio",
    "feat_volatility_squeeze",
    "feat_hour",
    "feat_is_active_session",
    "feat_prev_body_aligned",
    "feat_vol_ratio"
]


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
            writer.writerow(["date", "variant", "params", "dev_result", "pass_fail"])


def log_to_registry(variant_id, params_str, dev_result_str, pass_fail_str):
    """Logs an attempt (pass or fail) to research/registry.csv."""
    ensure_registry_initialized()
    today_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(REGISTRY_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([today_str, variant_id, params_str, dev_result_str, pass_fail_str])


def get_registry_count():
    """Returns total rows recorded in registry.csv (excluding header)."""
    ensure_registry_initialized()
    with open(REGISTRY_FILE, "r", encoding="utf-8") as f:
        reader = list(csv.reader(f))
        return max(0, len(reader) - 1)


# ==============================================================================
# 2. DATA FETCHING & MULTI-TIMEFRAME ALIGNMENT
# ==============================================================================

TF_MAP = {
    "M5": (mt5.TIMEFRAME_M5, pd.Timedelta(minutes=5), 48, 90000),     # 48 bars = 4h
    "M15": (mt5.TIMEFRAME_M15, pd.Timedelta(minutes=15), 16, 35000),  # 16 bars = 4h
    "H1": (mt5.TIMEFRAME_H1, pd.Timedelta(hours=1), 24, 15000),       # 24 bars = 24h
    "H4": (mt5.TIMEFRAME_H4, pd.Timedelta(hours=4), 12, 5000)         # 12 bars = 48h
}


def connect_mt5():
    if not mt5.terminal_info():
        if not mt5.initialize():
            print("❌ MT5 connection failed:", mt5.last_error())
            return False
    return True


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


def fetch_tf_rates(symbol, tf_str, count):
    """Fetches maximum available rates for a symbol and timeframe."""
    tf_val, _, _, _ = TF_MAP[tf_str]
    rates = mt5.copy_rates_from_pos(symbol, tf_val, 0, count)
    if rates is None or len(rates) == 0:
        return None
    df = pd.DataFrame(rates).iloc[:-1]  # Drop active forming candle
    df["time"] = pd.to_datetime(df["time"], unit="s")
    return df


def prepare_multi_tf_indicators(df_entry, df_trend, entry_tf_str, trend_tf_str):
    """
    Computes entry and macro indicators and merges them backward on timestamps.
    Leak-free: Macro timeframe is shifted by its full bar duration (CLOSE time alignment).
    """
    df_e = df_entry.copy()
    df_t = df_trend.copy()

    _, _, _, _ = TF_MAP[entry_tf_str]
    _, trend_duration, _, _ = TF_MAP[trend_tf_str]

    # --- Macro Trend Indicators ---
    df_t["macro_ema21"] = ta.trend.EMAIndicator(df_t["close"], window=21).ema_indicator()
    df_t["macro_ema50"] = ta.trend.EMAIndicator(df_t["close"], window=50).ema_indicator()
    df_t["macro_ema200"] = ta.trend.EMAIndicator(df_t["close"], window=200).ema_indicator()
    df_t["macro_adx"] = ta.trend.ADXIndicator(df_t["high"], df_t["low"], df_t["close"], window=14).adx()

    df_t["macro_trend"] = np.where(
        df_t["macro_ema50"] > df_t["macro_ema200"], 1,
        np.where(df_t["macro_ema50"] < df_t["macro_ema200"], -1, 0)
    )

    # Shift macro candle time to close time
    df_t["time"] = df_t["time"] + trend_duration
    df_t["time"] = df_t["time"].astype("datetime64[ns]")
    df_e["time"] = df_e["time"].astype("datetime64[ns]")

    # --- Entry Indicators ---
    df_e["entry_ema9"] = ta.trend.EMAIndicator(df_e["close"], window=9).ema_indicator()
    df_e["entry_ema21"] = ta.trend.EMAIndicator(df_e["close"], window=21).ema_indicator()
    df_e["entry_rsi"] = ta.momentum.RSIIndicator(df_e["close"], window=14).rsi()
    df_e["entry_atr"] = ta.volatility.AverageTrueRange(df_e["high"], df_e["low"], df_e["close"], window=14).average_true_range()
    df_e["entry_atr5"] = ta.volatility.AverageTrueRange(df_e["high"], df_e["low"], df_e["close"], window=5).average_true_range()
    df_e["entry_atr20"] = ta.volatility.AverageTrueRange(df_e["high"], df_e["low"], df_e["close"], window=20).average_true_range()

    if "tick_volume" in df_e.columns:
        df_e["vol_sma20"] = df_e["tick_volume"].rolling(window=20).mean()
        df_e["entry_vol_ratio"] = df_e["tick_volume"] / (df_e["vol_sma20"] + 1e-5)
    else:
        df_e["entry_vol_ratio"] = 1.0

    df_merged = pd.merge_asof(
        df_e.sort_values("time"),
        df_t[["time", "macro_trend", "macro_adx"]].sort_values("time"),
        on="time",
        direction="backward"
    )
    df_merged.dropna(inplace=True)
    df_merged.reset_index(drop=True, inplace=True)
    return df_merged


def extract_features(row, prev_row, direction):
    """Extracts invariant normalized features aligned to setup direction."""
    candle_range = max(row["high"] - row["low"], 1e-5)
    body = row["close"] - row["open"]
    upper_wick = row["high"] - max(row["open"], row["close"])
    lower_wick = min(row["open"], row["close"]) - row["low"]
    prev_body = prev_row["close"] - prev_row["open"]

    f_rsi = (row["entry_rsi"] - 50.0) * direction
    f_ema_spread = ((row["entry_ema9"] - row["entry_ema21"]) / row["entry_atr"]) * direction
    f_body = (body / candle_range) * direction
    f_rej_wick = (lower_wick / candle_range) if direction == 1 else (upper_wick / candle_range)
    f_opp_wick = (upper_wick / candle_range) if direction == 1 else (lower_wick / candle_range)
    f_dist_ema9 = ((row["close"] - row["entry_ema9"]) / row["entry_atr"]) * direction
    f_macro_adx = float(row["macro_adx"])
    f_atr_ratio = row["entry_atr5"] / (row["entry_atr20"] + 1e-5)
    f_squeeze = row["entry_atr"] / (candle_range + 1e-5)
    hour = row["time"].hour
    f_hour = float(hour)
    f_active_session = 1.0 if (7 <= hour <= 11 or 12 <= hour <= 17) else 0.0
    f_prev_body = (prev_body / (abs(prev_body) + candle_range)) * direction
    f_vol_ratio = float(row["entry_vol_ratio"])

    return [
        f_rsi, f_ema_spread, f_body, f_rej_wick, f_opp_wick,
        f_dist_ema9, f_macro_adx, f_atr_ratio, f_squeeze,
        f_hour, f_active_session, f_prev_body, f_vol_ratio
    ]


# ==============================================================================
# 3. STRATEGY SETUP SIMULATION
# ==============================================================================

def generate_variant_setups(df, variant_cfg, min_stop_dist, spread_est, point):
    """
    Simulates trades under the specified variant parameters.
    Conservative intrabar rule: if SL and TP are both touched in one bar, SL wins.
    """
    closes = df["close"].values
    highs = df["high"].values
    lows = df["low"].values
    opens = df["open"].values
    atrs = df["entry_atr"].values
    e9 = df["entry_ema9"].values
    e21 = df["entry_ema21"].values
    rsi = df["entry_rsi"].values
    adx = df["macro_adx"].values
    macro_trends = df["macro_trend"].values
    spreads_pts = df["spread"].values if "spread" in df.columns else np.zeros(len(df))

    be_trigger_r = variant_cfg.get("be_trigger_r")
    tp_mult = variant_cfg.get("tp_r", 1.8)
    entry_tf = variant_cfg.get("entry_tf", "M5")
    _, _, lookahead_bars, _ = TF_MAP[entry_tf]

    be_buffer = max(spread_est, point * 10)

    feats, labels, gross_r, net_r, bar_idx, exit_idx, be_hit, dirs = [], [], [], [], [], [], [], []

    for i in range(1, len(df) - lookahead_bars):
        if adx[i] < 12:
            continue

        crossover_buy = (e9[i-1] <= e21[i-1] and e9[i] > e21[i])
        pullback_buy = (e9[i] > e21[i] and lows[i-1] <= e21[i-1] and closes[i] > e9[i] and closes[i] > opens[i])
        crossover_sell = (e9[i-1] >= e21[i-1] and e9[i] < e21[i])
        pullback_sell = (e9[i] < e21[i] and highs[i-1] >= e21[i-1] and closes[i] < e9[i] and closes[i] < opens[i])

        is_buy = (macro_trends[i] == 1 and (crossover_buy or pullback_buy) and 40 <= rsi[i] <= 68)
        is_sell = (macro_trends[i] == -1 and (crossover_sell or pullback_sell) and 32 <= rsi[i] <= 60)
        if not (is_buy or is_sell):
            continue

        # Skip overextended entries
        if abs(closes[i] - e9[i]) > 1.4 * atrs[i]:
            continue

        d = 1 if is_buy else -1
        entry = closes[i]
        stop = max(atrs[i] * STOP_ATR_MULT, min_stop_dist)
        tp = entry + d * stop * tp_mult
        sl = entry - d * stop
        spread_price = max(spreads_pts[i] * point, spread_est)
        cost_r = (spread_price * (1.0 + SLIPPAGE_FRACTION)) / stop

        exit_price = None
        exit_bar = i + lookahead_bars
        be = False

        for j in range(i + 1, i + lookahead_bars + 1):
            adverse = lows[j] if d == 1 else highs[j]
            favour = highs[j] if d == 1 else lows[j]

            # Stop hit
            if (adverse - sl) * d <= 0:
                exit_price, exit_bar = sl, j
                break
            # Take profit hit
            if (favour - tp) * d >= 0:
                exit_price, exit_bar = tp, j
                break
            # Breakeven check
            if be_trigger_r is not None and not be and (favour - entry) * d >= be_trigger_r * stop:
                sl = entry + d * be_buffer
                be = True
            # Structural early exit (in loss + trend and EMAs flipped)
            if (closes[j] - entry) * d < 0 and macro_trends[j] == -d and (e9[j] - e21[j]) * d < 0:
                exit_price, exit_bar = closes[j], j
                break

        if exit_price is None:
            exit_price = closes[exit_bar]

        g = (exit_price - entry) * d / stop
        n = g - cost_r

        feats.append(extract_features(df.iloc[i], df.iloc[i-1], d))
        labels.append(1 if n > 0 else 0)
        gross_r.append(g)
        net_r.append(n)
        bar_idx.append(i)
        exit_idx.append(exit_bar)
        be_hit.append(be)
        dirs.append(d)

    return {
        "X": np.array(feats),
        "y": np.array(labels),
        "gross_r": np.array(gross_r),
        "net_r": np.array(net_r),
        "bar": np.array(bar_idx),
        "exit": np.array(exit_idx),
        "be": np.array(be_hit),
        "dir": np.array(dirs),
        "lookahead_bars": lookahead_bars
    }


def compute_trade_stats(r):
    """Computes standard statistics on R-multiples."""
    if len(r) == 0:
        return {"n": 0, "win_rate": 0.0, "avg_win": 0.0, "avg_loss": 0.0, "exp": 0.0, "t": 0.0}
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


def compute_thinned_stats(net_r, bars, exits):
    """Greedy chronological thinning for debiased statistical inference."""
    n = len(net_r)
    if n == 0:
        res = compute_trade_stats(np.array([]))
        res["thinned_n"] = 0
        res["raw_n"] = 0
        return res

    order = np.argsort(bars, kind="mergesort")
    sorted_bars = bars[order]
    sorted_exits = exits[order]
    sorted_r = net_r[order]

    kept = []
    last_exit = -1
    for i in range(n):
        if sorted_bars[i] >= last_exit:
            kept.append(i)
            last_exit = sorted_exits[i]

    thinned_r = sorted_r[kept]
    stats = compute_trade_stats(thinned_r)
    stats["thinned_n"] = len(kept)
    stats["raw_n"] = n
    return stats


# ==============================================================================
# 4. 20% LOCKBOX SPLIT & PURGE (E2)
# ==============================================================================

def split_dev_lockbox(data, lockbox_pct=0.20):
    """
    Splits setups chronologically:
      - First 80% as Development dataset
      - Final 20% as Untouched Lockbox
    Purges any dev trade whose exit bar extends into or past the lockbox start bar.
    """
    n = len(data["y"])
    if n == 0:
        empty = {k: np.array([]) for k in data if k != "lookahead_bars"}
        empty["lookahead_bars"] = data.get("lookahead_bars", 48)
        return empty, empty

    split_idx = int(np.floor(n * (1.0 - lockbox_pct)))
    split_idx = max(1, min(split_idx, n - 1)) if n > 1 else 0

    lockbox_start_bar = data["bar"][split_idx]
    lockbox_indices = np.arange(split_idx, n)

    dev_candidates = np.arange(0, split_idx)
    clean_dev_mask = data["exit"][dev_candidates] < lockbox_start_bar
    dev_indices = dev_candidates[clean_dev_mask]

    dev_data = {k: (data[k][dev_indices] if isinstance(data[k], np.ndarray) else data[k]) for k in data}
    lockbox_data = {k: (data[k][lockbox_indices] if isinstance(data[k], np.ndarray) else data[k]) for k in data}
    return dev_data, lockbox_data


# ==============================================================================
# 5. WALK-FORWARD AI VALIDATION (E4)
# ==============================================================================

def make_model(scale_pos_weight):
    return XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        gamma=1.0,
        min_child_weight=3,
        random_state=42,
        eval_metric="logloss"
    )


def walk_forward_cv(data, n_folds=N_FOLDS):
    """Purged cross-validation on dev dataset."""
    X, y, bars, exits = data["X"], data["y"], data["bar"], data["exit"]
    n = len(y)
    fold_size = n // n_folds
    lookahead = data.get("lookahead_bars", 48)

    oos_prob = np.full(n, np.nan)
    fold_ranges = []

    for k in range(n_folds):
        val_start = k * fold_size
        val_end = n if k == n_folds - 1 else (k + 1) * fold_size
        val_idx = np.arange(val_start, val_end)
        fold_ranges.append((val_start, val_end))

        val_min_bar = bars[val_start]
        val_max_exit = exits[val_end - 1]

        # Purge training set of overlapping trades
        train_idx = [
            i for i in range(n)
            if i < val_start or i >= val_end
            if exits[i] + lookahead < val_min_bar or bars[i] > val_max_exit + lookahead
        ]

        if len(train_idx) < 30 or len(val_idx) < 5:
            continue

        y_train = y[train_idx]
        if len(np.unique(y_train)) < 2:
            continue

        spw = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
        model = make_model(spw)
        model.fit(X[train_idx], y_train)

        probs = model.predict_proba(X[val_idx])[:, 1]
        oos_prob[val_idx] = probs

    return oos_prob, fold_ranges


def evaluate_threshold_dev(data, oos_prob, fold_ranges, th):
    """Scores out-of-sample performance at a given probability threshold."""
    sel = np.where(oos_prob >= th)[0]
    net_stats = compute_trade_stats(data["net_r"][sel])
    gross_stats = compute_trade_stats(data["gross_r"][sel])
    thinned = compute_thinned_stats(data["net_r"][sel], data["bar"][sel], data["exit"][sel])

    fold_exps = []
    for (v_start, v_end) in fold_ranges:
        f_idx = sel[(sel >= v_start) & (sel < v_end)]
        f_stats = compute_trade_stats(data["net_r"][f_idx])
        fold_exps.append(f_stats["exp"])

    pos_folds = sum(1 for e in fold_exps if e > 0)
    worst_fold = min(fold_exps) if fold_exps else -999.0

    return {
        "th": th,
        "net": net_stats,
        "gross": gross_stats,
        "thinned": thinned,
        "positive_folds": pos_folds,
        "worst_fold_exp": worst_fold,
        "fold_exps": fold_exps
    }


# ==============================================================================
# 6. RUNNER EXECUTION LOGIC (E4 & E5)
# ==============================================================================

def run_evaluation_for_variant(symbol_cfg, variant_id, variant_cfg):
    """
    Evaluates a single variant for a single asset:
      1. Stage 1: No-AI Baseline (must have gross > 0 AND net > 0).
      2. Stage 2: AI walk-forward filter (only if baseline passes).
    """
    assert_safe_environment()
    sym_name = symbol_cfg["name"]
    res_sym = setup_symbol(sym_name)
    if not res_sym:
        return {"error": f"Symbol {sym_name} not found"}

    info = mt5.symbol_info(res_sym)
    point = info.point if info else 0.01

    entry_tf = variant_cfg["entry_tf"]
    trend_tf = variant_cfg["trend_tf"]

    _, _, _, count_e = TF_MAP[entry_tf]
    _, _, _, count_t = TF_MAP[trend_tf]

    df_entry = fetch_tf_rates(res_sym, entry_tf, count_e)
    df_trend = fetch_tf_rates(res_sym, trend_tf, count_t)

    if df_entry is None or df_trend is None or len(df_entry) < 500 or len(df_trend) < 200:
        return {"error": f"Insufficient rates for {res_sym}"}

    # Date span
    date_start = df_entry["time"].iloc[0].strftime("%Y-%m-%d")
    date_end = df_entry["time"].iloc[-1].strftime("%Y-%m-%d")
    span_str = f"{date_start} to {date_end}"

    # Indicator calculation & setup generation
    df_prep = prepare_multi_tf_indicators(df_entry, df_trend, entry_tf, trend_tf)
    all_data = generate_variant_setups(
        df_prep, variant_cfg, symbol_cfg["min_stop_dist"], symbol_cfg["spread_est"], point
    )
    n_setups_total = len(all_data["y"])

    # 80% Dev vs 20% Lockbox Split
    dev_data, lockbox_data = split_dev_lockbox(all_data, lockbox_pct=0.20)
    n_dev = len(dev_data["y"])
    n_lockbox = len(lockbox_data["y"])

    params_str = f"{entry_tf}/{trend_tf}_BE={variant_cfg.get('be_trigger_r')}_TP={variant_cfg.get('tp_r')}"

    if n_dev < 30:
        dev_res = "INSUFFICIENT_SETUPS"
        pass_fail = "FAIL (too few dev setups)"
        log_to_registry(f"{variant_id}_{sym_name}", params_str, dev_res, pass_fail)
        return {
            "variant_id": variant_id, "symbol": sym_name, "span": span_str,
            "total_setups": n_setups_total, "dev_setups": n_dev, "lockbox_setups": n_lockbox,
            "stage1_passed": False, "ai_tested": False, "pass_fail": pass_fail,
            "dev_data": dev_data, "lockbox_data": lockbox_data, "baseline": None
        }

    # -------------------------------------------------------------
    # STAGE 1: NO-AI BASELINE EVALUATION (E4.1 & E4.2)
    # -------------------------------------------------------------
    base_raw = compute_trade_stats(dev_data["net_r"])
    base_gross = compute_trade_stats(dev_data["gross_r"])
    base_thinned = compute_thinned_stats(dev_data["net_r"], dev_data["bar"], dev_data["exit"])

    baseline_info = {
        "raw_trades": base_raw["n"],
        "thinned_trades": base_thinned["thinned_n"],
        "win_rate": base_raw["win_rate"],
        "avg_win": base_raw["avg_win"],
        "avg_loss": base_raw["avg_loss"],
        "gross_exp": base_gross["exp"],
        "net_exp": base_raw["exp"],
        "thinned_t": base_thinned["t"]
    }

    # E4.2: Gate - gross expectancy > 0 AND net expectancy > 0
    stage1_passed = (base_gross["exp"] > 0.0 and base_raw["exp"] > 0.0)

    if not stage1_passed:
        dev_res = f"Gross={base_gross['exp']:+.3f}R,Net={base_raw['exp']:+.3f}R,t={base_thinned['t']:+.2f}"
        pass_fail = "FAIL (no edge without AI)"
        log_to_registry(f"{variant_id}_{sym_name}", params_str, dev_res, pass_fail)
        return {
            "variant_id": variant_id, "symbol": sym_name, "span": span_str,
            "total_setups": n_setups_total, "dev_setups": n_dev, "lockbox_setups": n_lockbox,
            "stage1_passed": False, "ai_tested": False, "pass_fail": pass_fail,
            "dev_data": dev_data, "lockbox_data": lockbox_data, "baseline": baseline_info,
            "ai_result": None
        }

    # -------------------------------------------------------------
    # STAGE 2: AI WALK-FORWARD ML FILTER (E4.2 & E4.4)
    # -------------------------------------------------------------
    oos_prob, fold_ranges = walk_forward_cv(dev_data, n_folds=N_FOLDS)
    valid = ~np.isnan(oos_prob)
    auc = roc_auc_score(dev_data["y"][valid], oos_prob[valid]) if valid.sum() > 0 and len(np.unique(dev_data["y"][valid])) > 1 else 0.5

    best_eval = None
    for th in THRESHOLD_GRID:
        ev = evaluate_threshold_dev(dev_data, oos_prob, fold_ranges, th)
        if best_eval is None or (ev["net"]["exp"] > best_eval["net"]["exp"] and ev["net"]["n"] >= 30):
            best_eval = ev

    if best_eval is None:
        best_eval = evaluate_threshold_dev(dev_data, oos_prob, fold_ranges, 0.50)

    # Dev Pass Criteria (E4.4): net exp > 0, thinned t >= 2.5, positive folds >= 4 of 5, trades >= 100
    ai_passes = (
        best_eval["net"]["exp"] > 0.0 and
        best_eval["thinned"]["t"] >= DEV_MIN_T_STAT and
        best_eval["positive_folds"] >= MIN_POSITIVE_FOLDS and
        best_eval["net"]["n"] >= DEV_MIN_TRADES
    )

    pass_fail = "PASS" if ai_passes else "FAIL"
    dev_res = (
        f"AI_Net={best_eval['net']['exp']:+.3f}R,th_t={best_eval['thinned']['t']:+.2f},"
        f"folds={best_eval['positive_folds']}/5,trades={best_eval['net']['n']}"
    )
    log_to_registry(f"{variant_id}_{sym_name}", params_str, dev_res, pass_fail)

    return {
        "variant_id": variant_id,
        "symbol": sym_name,
        "span": span_str,
        "total_setups": n_setups_total,
        "dev_setups": n_dev,
        "lockbox_setups": n_lockbox,
        "stage1_passed": True,
        "ai_tested": True,
        "pass_fail": pass_fail,
        "baseline": baseline_info,
        "ai_result": {
            "auc": auc,
            "threshold": best_eval["th"],
            "trades": best_eval["net"]["n"],
            "thinned_trades": best_eval["thinned"]["thinned_n"],
            "win_rate": best_eval["net"]["win_rate"],
            "net_exp": best_eval["net"]["exp"],
            "gross_exp": best_eval["gross"]["exp"],
            "thinned_t": best_eval["thinned"]["t"],
            "positive_folds": best_eval["positive_folds"],
            "worst_fold_exp": best_eval["worst_fold_exp"]
        },
        "dev_data": dev_data,
        "lockbox_data": lockbox_data
    }


def run_lockbox_evaluation(finalist, symbol_cfg, variant_cfg):
    """
    Evaluates the single chosen finalist strictly ONCE on the 20% Lockbox (E5).
    Lockbox requires: >= 30 trades, net expectancy > 0, thinned t >= 2.0.
    """
    lockbox_data = finalist["lockbox_data"]
    dev_data = finalist["dev_data"]
    n_lb = len(lockbox_data["y"])

    if n_lb == 0:
        return {"status": "FAIL (empty lockbox)", "n_trades": 0, "net_exp": 0.0, "thinned_t": 0.0}

    # Train validation model strictly on DEV data
    spw_dev = (dev_data["y"] == 0).sum() / max((dev_data["y"] == 1).sum(), 1)
    dev_model = make_model(spw_dev)
    dev_model.fit(dev_data["X"], dev_data["y"])

    # Score ONCE on lockbox
    th = finalist["ai_result"]["threshold"]
    lb_probs = dev_model.predict_proba(lockbox_data["X"])[:, 1]
    sel_lb = np.where(lb_probs >= th)[0]

    lb_net = compute_trade_stats(lockbox_data["net_r"][sel_lb])
    lb_gross = compute_trade_stats(lockbox_data["gross_r"][sel_lb])
    lb_thinned = compute_thinned_stats(
        lockbox_data["net_r"][sel_lb],
        lockbox_data["bar"][sel_lb],
        lockbox_data["exit"][sel_lb]
    )

    lb_passes = (
        lb_net["n"] >= LOCKBOX_MIN_TRADES and
        lb_net["exp"] > 0.0 and
        lb_thinned["t"] >= LOCKBOX_MIN_T_STAT
    )

    return {
        "status": "PASS" if lb_passes else "FAIL",
        "threshold": th,
        "n_trades": lb_net["n"],
        "thinned_trades": lb_thinned["thinned_n"],
        "win_rate": lb_net["win_rate"],
        "net_exp": lb_net["exp"],
        "gross_exp": lb_gross["exp"],
        "thinned_t": lb_thinned["t"]
    }


# ==============================================================================
# 7. MAIN RUNNER & REPORT WRITER
# ==============================================================================

def run_all_research():
    """Main execution entry point for Phase 7 Strategy Research."""
    assert_safe_environment()

    if not os.path.exists(VARIANTS_FILE):
        raise FileNotFoundError(f"Missing {VARIANTS_FILE}. Pre-register variants before running!")

    with open(VARIANTS_FILE, "r", encoding="utf-8") as f:
        variants = json.load(f)

    if len(variants) > 8:
        raise ValueError(f"Exceeded max 8 variants rule! Found {len(variants)} variants.")

    if not connect_mt5():
        print("❌ Could not connect to MetaTrader 5.")
        return False

    all_results = []
    print("\n" + "=" * 70)
    print("🔬 RUNNING PHASE 7 STRATEGY RESEARCH HARNESS (OFFLINE ONLY)")
    print(f"📋 Evaluating {len(variants)} Pre-Registered Variants across {len(TARGET_SYMBOLS)} Symbols")
    print("=" * 70)

    for v_id, v_cfg in variants.items():
        print(f"\n▶ Testing Variant [{v_id}]: {v_cfg['name']}")
        for sym_cfg in TARGET_SYMBOLS:
            print(f"  • Asset: {sym_cfg['name']}...", end=" ", flush=True)
            res = run_evaluation_for_variant(sym_cfg, v_id, v_cfg)
            if "error" in res:
                print(f"❌ Error: {res['error']}")
            else:
                st1 = "PASS" if res["stage1_passed"] else "FAIL (No Edge without AI)"
                print(f"Done. Stage 1 Baseline: {st1} | Dev Status: {res['pass_fail']}")
            all_results.append((v_id, v_cfg, sym_cfg, res))

    # Identify passing variants (E5)
    passing_variants = [r for r in all_results if "error" not in r[3] and r[3]["pass_fail"] == "PASS"]

    lockbox_report = None
    if not passing_variants:
        lockbox_decision = "No variant passed E4 Dev criteria. Lockbox remains safely unburned (UNOPENED)."
    else:
        # If > 1 passes: pick the one with best worst-fold result
        chosen_finalist_tuple = max(
            passing_variants,
            key=lambda item: item[3]["ai_result"]["worst_fold_exp"]
        )
        f_vid, f_vcfg, f_symcfg, f_res = chosen_finalist_tuple
        lockbox_decision = (
            f"Chosen finalist for single-use lockbox: Variant {f_vid} on {f_symcfg['name']} "
            f"(Selected by best worst-fold expectancy: {f_res['ai_result']['worst_fold_exp']:+.3f}R)."
        )
        print(f"\n🔒 Executing Single-Use Lockbox Test for {f_vid} on {f_symcfg['name']}...")
        lb_res = run_lockbox_evaluation(f_res, f_symcfg, f_vcfg)
        lockbox_report = {
            "finalist": f"{f_vid} ({f_symcfg['name']})",
            "decision": lockbox_decision,
            "metrics": lb_res
        }
        print(f"🎯 Lockbox Outcome: {lb_res['status']} | Net Exp: {lb_res['net_exp']:+.3f}R | t: {lb_res['thinned_t']:+.2f}")

    # Generate Markdown Report
    ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(RESEARCH_DIR, f"report_{ts_str}.md")

    generate_markdown_report(report_path, all_results, lockbox_decision, lockbox_report)
    print(f"\n📄 Research Report successfully generated at: {report_path}")
    return report_path


def generate_markdown_report(report_path, results, lockbox_decision, lockbox_report):
    """Generates the official Phase 7 Strategy Research Report (E4.5)."""
    reg_count = get_registry_count()
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    md = []
    md.append(f"# Phase 7 Strategy Research Report")
    md.append(f"**Date:** {now_str}  ")
    md.append(f"**Total Registry Attempts Logged:** {reg_count}  ")
    md.append(f"**Cost Model:** Spread × 1.25 + Slippage (Immutable)  ")
    md.append("\n---\n")

    md.append("## 1. Ground Rules & Sandbox Compliance (E1)")
    md.append("- **Order Execution:** ZERO orders sent (isolated research runner).")
    md.append("- **Model Protection:** ZERO writes to `*_ai_model.pkl`, `.candidate.pkl`, or `.prev.pkl`.")
    md.append("- **Registry Integrity:** All attempts logged to `research/registry.csv`.")
    md.append("\n---\n")

    md.append("## 2. Dev Evaluation Table (E4)")
    md.append("| Variant | Asset | Timeframes | Date Span | Setups (Total / Dev) | No-AI Gross Exp | No-AI Net Exp | No-AI t-stat | AI Net Exp | AI Thinned t | +Folds | Dev Result |")
    md.append("| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")

    for v_id, v_cfg, sym_cfg, r in results:
        sym = sym_cfg["name"]
        tfs = f"{v_cfg['entry_tf']} / {v_cfg['trend_tf']}"
        if "error" in r:
            md.append(f"| **{v_id}** | {sym} | {tfs} | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | **ERROR: {r['error']}** |")
            continue

        span = r["span"]
        setups = f"{r['total_setups']} / {r['dev_setups']}"
        base = r["baseline"]

        if base:
            gr_exp = f"{base['gross_exp']:+.3f}R"
            nt_exp = f"{base['net_exp']:+.3f}R"
            th_t = f"{base['thinned_t']:+.2f}"
        else:
            gr_exp = nt_exp = th_t = "N/A"

        if r["ai_tested"] and r["ai_result"]:
            ai = r["ai_result"]
            ai_net = f"{ai['net_exp']:+.3f}R"
            ai_t = f"{ai['thinned_t']:+.2f}"
            folds = f"{ai['positive_folds']}/5"
        else:
            ai_net = ai_t = folds = "N/A"

        res_str = f"**{r['pass_fail']}**"
        md.append(f"| **{v_id}** | {sym} | {tfs} | {span} | {setups} | {gr_exp} | {nt_exp} | {th_t} | {ai_net} | {ai_t} | {folds} | {res_str} |")

    md.append("\n---\n")

    md.append("## 3. Lockbox Evaluation (E5)")
    md.append(f"**Decision Protocol:** {lockbox_decision}\n")

    if lockbox_report:
        lb = lockbox_report["metrics"]
        md.append(f"### Lockbox Single-Use Outcome: **{lb['status']}**")
        md.append(f"- **Finalist Tested:** `{lockbox_report['finalist']}`")
        md.append(f"- **Lockbox Trades:** {lb['n_trades']} (Thinned: {lb['thinned_trades']}) [Req: $\\ge 30$]")
        md.append(f"- **Realized Net Expectancy:** {lb['net_exp']:+.3f}R [Req: $> 0.00\\text{{R}}$]")
        md.append(f"- **Realized Gross Expectancy:** {lb['gross_exp']:+.3f}R")
        md.append(f"- **Thinned t-statistic:** {lb['thinned_t']:+.2f} [Req: $\\ge 2.0$]")
        md.append(f"- **Lockbox Status:** **BURNED** (Single-use completed).")
    else:
        md.append("> [!NOTE]")
        md.append("> **No variants passed Dev Stage 1 & Stage 2 criteria.**")
        md.append("> Per Rule E5, the research terminates honestly with **'No edge found'**. The 20% Lockbox remains unopened and unburned.")

    md.append("\n---\n")
    md.append("## 4. Conclusion & Next Steps")
    md.append("- In accordance with **Phase 7 Mandate**, nothing here was promoted to live trading.")
    md.append("- Live models, risk parameters, and bot operations remain 100% untouched.")
    md.append("- All operational rules from [GRADUATION.md](file:///c:/Users/HP%201030/Desktop/forexbot/GRADUATION.md) remain in full effect.")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md))


if __name__ == "__main__":
    run_all_research()
