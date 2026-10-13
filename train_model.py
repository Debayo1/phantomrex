import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import joblib
import os
import shutil
import sys
import json
from datetime import datetime
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score

from ai_features import prepare_indicators, extract_single_feature_vector, FEATURE_COLUMNS, TREND_MODES

# Force UTF-8 console output for Windows terminal emoji rendering
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

TARGET_SYMBOLS = [
    {"name": "XAUUSDm", "model_file": "gold_ai_model.pkl",   "min_stop_dist": 1.50,  "spread_est": 0.25},
    {"name": "BTCUSDm", "model_file": "btc_ai_model.pkl",    "min_stop_dist": 100.0, "spread_est": 15.0},
    {"name": "US30m",   "model_file": "us30_ai_model.pkl",   "min_stop_dist": 30.0,  "spread_est": 3.0},
    {"name": "USTECm",  "model_file": "nas100_ai_model.pkl", "min_stop_dist": 15.0,  "spread_est": 2.0}
]

TIMEFRAME_LOW = mt5.TIMEFRAME_M5
TIMEFRAME_HIGH = mt5.TIMEFRAME_H1

DEFAULT_CANDLES_LOW = 40000   # ~6 months of M5 candles
DEFAULT_CANDLES_HIGH = 6000   # ~6 months of H1 candles

CANDLES_LOW = DEFAULT_CANDLES_LOW
CANDLES_HIGH = DEFAULT_CANDLES_HIGH


def get_history_candle_counts():
    """
    Returns (candles_low, candles_high) with environment variable override support.
    """
    try:
        low = int(os.environ.get("CANDLES_LOW", DEFAULT_CANDLES_LOW))
    except (ValueError, TypeError):
        low = DEFAULT_CANDLES_LOW
    try:
        high = int(os.environ.get("CANDLES_HIGH", DEFAULT_CANDLES_HIGH))
    except (ValueError, TypeError):
        high = DEFAULT_CANDLES_HIGH
    return low, high


# --- Trade simulation (mirrors live bot.py behaviour) ---
STOP_ATR_MULT = 1.5          # bot.py ATR_MULTIPLIER
TP_MULT = 1.8                # TP_Core target (the single position used on small accounts)
BE_TRIGGER_R = 0.9           # bot.py moves SL to breakeven at 0.9R
LOOKAHEAD_BARS = 48          # max holding time (4h on M5); also used as purge gap
SLIPPAGE_FRACTION = 0.25     # extra slippage allowance = 25% of spread

# --- Validation (Master Plan Phase 2) ---
N_FOLDS = 5
MIN_TRADES_FOR_THRESHOLD = 30
MIN_POSITIVE_FOLDS = 4       # PRD success metric: positive expectancy in >= 4 of 5 folds
MIN_HOLDOUT_TRADES = 30      # Holdout set requires at least 30 filtered trades
MIN_T_STAT = 2.5             # Expectancy t-stat must be >= 2.5 on debiased thinned sequence
HOLDOUT_SPLIT_RATIO = 0.15   # 15% untouched holdout
THRESHOLD_GRID = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75]


def make_model(scale_pos_weight):
    return XGBClassifier(
        n_estimators=250,
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


def connect_mt5():
    if not mt5.terminal_info():
        if not mt5.initialize():
            print("❌ MT5 connection failed")
            print(mt5.last_error())
            return False
    print("✅ Connected to MetaTrader 5")
    return True


def setup_symbol(target_symbol):
    if mt5.symbol_select(target_symbol, True):
        return target_symbol

    clean_base = target_symbol.upper().rstrip("M").rstrip("C").rstrip("247")
    variants = [f"{clean_base}m", clean_base, f"{clean_base}c"]
    for var in variants:
        if mt5.symbol_select(var, True):
            print(f"ℹ️ Selected broker symbol: {var}")
            return var

    all_symbols = mt5.symbols_get() or []
    match = next((s.name for s in all_symbols if clean_base.lower() in s.name.lower()), None)
    if match and mt5.symbol_select(match, True):
        print(f"ℹ️ Selected broker symbol: {match}")
        return match
    else:
        print(f"❌ Could not select symbol: {target_symbol}")
        return None


def fetch_raw(symbol):
    candles_low, candles_high = get_history_candle_counts()
    print(f"📥 Fetching historical market data for {symbol} ({candles_low} M5 candles, {candles_high} H1 candles)...")
    rates_low = mt5.copy_rates_from_pos(symbol, TIMEFRAME_LOW, 0, candles_low)
    rates_high = mt5.copy_rates_from_pos(symbol, TIMEFRAME_HIGH, 0, candles_high)

    if rates_low is None or rates_high is None:
        print(f"❌ Failed to fetch market data for {symbol}")
        return None, None

    df_low = pd.DataFrame(rates_low).iloc[:-1]    # drop the still-forming candle
    df_high = pd.DataFrame(rates_high).iloc[:-1]
    return df_low, df_high


# ==============================================================================
# 7.1  TRADE-OUTCOME LABELING (full trade simulation incl. breakeven + costs)
# ==============================================================================

def generate_candidate_labels(df, min_stop_dist, spread_est, point):
    """
    For every strategy setup, simulates the live trade:
      initial SL (1.5 ATR, min distance) -> TP at 1.8R -> SL to breakeven at 0.9R
      -> AI structural early exit -> time-out after LOOKAHEAD_BARS.
    Conservative intrabar rule: if SL and TP are both touched in one bar, SL wins.
    Returns features, labels (1 = net profit after costs) and per-trade R results.
    """
    closes = df["close"].values
    highs = df["high"].values
    lows = df["low"].values
    opens = df["open"].values
    atrs = df["m5_atr"].values
    e9 = df["m5_ema9"].values
    e21 = df["m5_ema21"].values
    rsi = df["m5_rsi"].values
    adx = df["h1_adx"].values
    h1_trends = df["h1_trend"].values
    spreads_pts = df["spread"].values if "spread" in df.columns else np.zeros(len(df))

    be_buffer = max(spread_est, point * 10)   # same as bot.py spread_buffer

    feats, labels, gross_r, net_r, bar_idx, exit_idx, be_hit, dirs = [], [], [], [], [], [], [], []

    for i in range(1, len(df) - LOOKAHEAD_BARS):
        if adx[i] < 12:
            continue

        crossover_buy = (e9[i-1] <= e21[i-1] and e9[i] > e21[i])
        pullback_buy = (e9[i] > e21[i] and lows[i-1] <= e21[i-1] and closes[i] > e9[i] and closes[i] > opens[i])
        crossover_sell = (e9[i-1] >= e21[i-1] and e9[i] < e21[i])
        pullback_sell = (e9[i] < e21[i] and highs[i-1] >= e21[i-1] and closes[i] < e9[i] and closes[i] < opens[i])

        is_buy = (h1_trends[i] == 1 and (crossover_buy or pullback_buy) and 40 <= rsi[i] <= 68)
        is_sell = (h1_trends[i] == -1 and (crossover_sell or pullback_sell) and 32 <= rsi[i] <= 60)
        if not (is_buy or is_sell):
            continue

        # Entry guard (same as live): skip overextended entries
        if abs(closes[i] - e9[i]) > 1.4 * atrs[i]:
            continue

        d = 1 if is_buy else -1
        entry = closes[i]
        stop = max(atrs[i] * STOP_ATR_MULT, min_stop_dist)
        tp = entry + d * stop * TP_MULT
        sl = entry - d * stop
        spread_price = max(spreads_pts[i] * point, spread_est)
        cost_r = (spread_price * (1.0 + SLIPPAGE_FRACTION)) / stop

        exit_price = None
        exit_bar = i + LOOKAHEAD_BARS
        be = False
        for j in range(i + 1, i + LOOKAHEAD_BARS + 1):
            adverse = lows[j] if d == 1 else highs[j]
            favour = highs[j] if d == 1 else lows[j]
            if (adverse - sl) * d <= 0:                  # stop touched
                exit_price, exit_bar = sl, j
                break
            if (favour - tp) * d >= 0:                   # take profit touched
                exit_price, exit_bar = tp, j
                break
            if not be and (favour - entry) * d >= BE_TRIGGER_R * stop:
                sl = entry + d * be_buffer               # breakeven from next bar
                be = True
            # AI structural early exit (in loss + H1 trend and M5 EMAs flipped)
            if (closes[j] - entry) * d < 0 and h1_trends[j] == -d and (e9[j] - e21[j]) * d < 0:
                exit_price, exit_bar = closes[j], j
                break
        if exit_price is None:
            exit_price = closes[exit_bar]

        g = (exit_price - entry) * d / stop
        n = g - cost_r

        feats.append(extract_single_feature_vector(df.iloc[i], df.iloc[i-1], d))
        labels.append(1 if n > 0 else 0)
        gross_r.append(g)
        net_r.append(n)
        bar_idx.append(i)
        exit_idx.append(exit_bar)
        be_hit.append(be)
        dirs.append(d)

    return {
        "X": np.array(feats), "y": np.array(labels),
        "gross_r": np.array(gross_r), "net_r": np.array(net_r),
        "bar": np.array(bar_idx), "exit": np.array(exit_idx),
        "be": np.array(be_hit), "dir": np.array(dirs)
    }


def trade_stats(r):
    """Win rate / avg win / avg loss / expectancy, all in R multiples."""
    if len(r) == 0:
        return {"n": 0, "win_rate": 0.0, "avg_win": 0.0, "avg_loss": 0.0, "exp": 0.0, "total": 0.0, "t": 0.0}
    wins, losses = r[r > 0], r[r <= 0]
    sd = float(r.std(ddof=1)) if len(r) > 1 else 0.0
    return {
        "n": int(len(r)),
        "win_rate": float(len(wins) / len(r) * 100),
        "avg_win": float(wins.mean()) if len(wins) else 0.0,
        "avg_loss": float(losses.mean()) if len(losses) else 0.0,
        "exp": float(r.mean()),
        "total": float(r.sum()),
        "t": float(r.mean() / (sd / np.sqrt(len(r)))) if sd > 0 else 0.0
    }


# ==============================================================================
# 7.2  HONEST HOLDOUT SPLIT, THINNING, & DEBIASED VALIDATION (Phase 2)
# ==============================================================================

def split_dev_holdout(data, holdout_pct=HOLDOUT_SPLIT_RATIO):
    """
    Chronologically splits candidate setups into:
      - Development dataset (first ~85%)
      - Untouched holdout dataset (final ~15%)
    Purges any trade in dev whose exit bar extends into or past the holdout start bar.
    """
    n = len(data["y"])
    if n == 0:
        empty = {k: np.array([]) for k in data}
        return empty, empty

    split_idx = int(np.floor(n * (1.0 - holdout_pct)))
    split_idx = max(1, min(split_idx, n - 1)) if n > 1 else 0

    if split_idx == 0:
        empty = {k: np.array([]) for k in data}
        return empty, data

    holdout_start_bar = data["bar"][split_idx]
    holdout_indices = np.arange(split_idx, n)

    dev_candidates = np.arange(0, split_idx)
    # Purge: keep dev trades that completely exited strictly before holdout_start_bar
    clean_dev_mask = data["exit"][dev_candidates] < holdout_start_bar
    dev_indices = dev_candidates[clean_dev_mask]

    dev_data = {k: data[k][dev_indices] for k in data}
    holdout_data = {k: data[k][holdout_indices] for k in data}
    return dev_data, holdout_data


def compute_thinned_stats(net_r, bars, exits):
    """
    Greedy chronological thinning:
    Retains the first setup, then skips any subsequent setup that entered
    before the previous retained trade's exit bar.
    Computes trade_stats on the thinned non-overlapping trades.
    """
    n = len(net_r)
    if n == 0:
        res = trade_stats(np.array([]))
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
    stats = trade_stats(thinned_r)
    stats["thinned_n"] = len(kept)
    stats["raw_n"] = n
    return stats


def evaluate_edge_criteria(n_holdout, holdout_net_exp, holdout_gross_exp, dev_thinned_t, dev_positive_folds):
    """
    Evaluates edge criteria per Master Plan Phase 2.4:
    1. Filtered holdout trades >= 30
    2. Holdout net expectancy > 0.00R (after spread & slippage costs)
    3. Holdout gross expectancy > 0.00R
    4. Debiased non-overlapping walk-forward t-statistic >= 2.5
    5. Positive walk-forward folds >= 4 of 5 (MIN_POSITIVE_FOLDS = 4)

    Small-Holdout Handling:
      If n_holdout < 30: status = 'INSUFFICIENT_DATA', edge_ok = False.
      If any other criteria fail: status = 'FAIL', edge_ok = False.
      If all pass: status = 'PASS', edge_ok = True.
    """
    checks = {
        "holdout_trades_ge_30": bool(n_holdout >= MIN_HOLDOUT_TRADES),
        "holdout_net_exp_gt_0": bool(holdout_net_exp > 0.0),
        "holdout_gross_exp_gt_0": bool(holdout_gross_exp > 0.0),
        "dev_thinned_t_ge_2_5": bool(dev_thinned_t >= MIN_T_STAT),
        "dev_positive_folds_ge_4": bool(dev_positive_folds >= MIN_POSITIVE_FOLDS),
    }

    if not checks["holdout_trades_ge_30"]:
        status = "INSUFFICIENT_DATA"
        edge_ok = False
    elif not all(checks.values()):
        status = "FAIL"
        edge_ok = False
    else:
        status = "PASS"
        edge_ok = True

    return edge_ok, status, checks


def to_json_serializable(obj):
    """Recursively converts NumPy scalar/array types to native Python types for clean JSON dumping."""
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif isinstance(obj, (np.integer, int)):
        return int(obj)
    elif isinstance(obj, (np.floating, float)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {str(k): to_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [to_json_serializable(x) for x in obj]
    return obj


def save_audit_report(report_data, output_path=None):
    """Saves structured training audit report JSON to reports/ directory."""
    os.makedirs("reports", exist_ok=True)
    if not output_path:
        symbol = report_data.get("symbol", "unknown")
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join("reports", f"{symbol}_training_report_{ts}.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(to_json_serializable(report_data), f, indent=2)
    return output_path


def walk_forward(data):
    X, y, bars, exits = data["X"], data["y"], data["bar"], data["exit"]
    n = len(y)
    blocks = np.array_split(np.arange(n), N_FOLDS + 1)
    oos_prob = np.full(n, np.nan)
    fold_ranges = []

    for k in range(1, N_FOLDS + 1):
        test_idx = blocks[k]
        test_start_bar = bars[test_idx[0]]
        train_idx = np.concatenate(blocks[:k])
        # Purge: drop training trades still open when the test period starts
        train_idx = train_idx[exits[train_idx] < test_start_bar]
        if len(train_idx) < 50 or len(np.unique(y[train_idx])) < 2:
            continue
        spw = (y[train_idx] == 0).sum() / max((y[train_idx] == 1).sum(), 1)
        m = make_model(spw)
        m.fit(X[train_idx], y[train_idx])
        oos_prob[test_idx] = m.predict_proba(X[test_idx])[:, 1]
        fold_ranges.append((k, test_idx))
    return oos_prob, fold_ranges


def evaluate_threshold(data, oos_prob, fold_ranges, th):
    net, gross, bars, exits = data["net_r"], data["gross_r"], data["bar"], data["exit"]
    per_fold = []
    for k, idx in fold_ranges:
        sel = idx[oos_prob[idx] >= th]
        per_fold.append({"fold": k, **trade_stats(net[sel])})
    all_idx = np.concatenate([idx for _, idx in fold_ranges]) if fold_ranges else np.array([], dtype=int)
    sel_all = all_idx[oos_prob[all_idx] >= th]
    pooled_net = trade_stats(net[sel_all])
    pooled_gross = trade_stats(gross[sel_all])
    thinned = compute_thinned_stats(net[sel_all], bars[sel_all], exits[sel_all])
    positive_folds = sum(1 for f in per_fold if f["n"] > 0 and f["exp"] > 0)
    worst = min((f["exp"] for f in per_fold if f["n"] > 0), default=0.0)
    return {
        "th": th,
        "net": pooled_net,
        "gross": pooled_gross,
        "thinned": thinned,
        "folds": per_fold,
        "positive_folds": positive_folds,
        "worst_fold_exp": worst
    }


def validate_mode(df, symbol_cfg, point, mode):
    data = generate_candidate_labels(df, symbol_cfg["min_stop_dist"], symbol_cfg["spread_est"], point)
    n = len(data["y"])
    print(f"\n  ── Trend mode '{mode}': {n} setups "
          f"({int((data['dir'] == 1).sum())} buys / {int((data['dir'] == -1).sum())} sells)")
    if n < 150:
        print("  ⚠️ Too few setups for walk-forward validation.")
        return None

    # Step 2.2: Split 85% dev vs 15% holdout with boundary purge
    dev_data, holdout_data = split_dev_holdout(data, holdout_pct=HOLDOUT_SPLIT_RATIO)
    n_dev = len(dev_data["y"])
    n_holdout_total = len(holdout_data["y"])
    print(f"  ── Split: {n_dev} dev setups (purged) | {n_holdout_total} untouched holdout setups")
    if n_dev < 100:
        print("  ⚠️ Too few dev setups after holdout split & purge.")
        return None

    base_all = trade_stats(dev_data["net_r"])
    print(f"  Dev baseline (no AI), after costs: win {base_all['win_rate']:.1f}% | "
          f"exp {base_all['exp']:+.3f}R | avgW {base_all['avg_win']:+.2f}R avgL {base_all['avg_loss']:+.2f}R")

    # Step 2.3: Walk-forward strictly on dev data
    oos_prob, fold_ranges = walk_forward(dev_data)
    valid = ~np.isnan(oos_prob)
    auc = roc_auc_score(dev_data["y"][valid], oos_prob[valid]) if valid.sum() > 0 and len(np.unique(dev_data["y"][valid])) > 1 else 0.5
    print(f"  Dev out-of-sample ROC-AUC: {auc:.3f}  (0.50 = coin flip)")

    all_th_evals = []
    best = None
    for th in THRESHOLD_GRID:
        ev = evaluate_threshold(dev_data, oos_prob, fold_ranges, th)
        all_th_evals.append(ev)
        nt, gr, thn = ev["net"], ev["gross"], ev["thinned"]
        print(f"   ≥{th*100:.0f}%: trades {nt['n']:4d} (thinned {thn['n']:3d}) | win {nt['win_rate']:5.1f}% | "
              f"exp gross {gr['exp']:+.3f}R → net {nt['exp']:+.3f}R (raw t={nt['t']:+.2f}, thinned t={thn['t']:+.2f}) | "
              f"+folds {ev['positive_folds']}/{len(fold_ranges)} | worst fold {ev['worst_fold_exp']:+.3f}R")

        if nt["n"] >= MIN_TRADES_FOR_THRESHOLD:
            dev_passes = (thn["t"] >= MIN_T_STAT and ev["positive_folds"] >= MIN_POSITIVE_FOLDS)
            if best is None:
                best = ev
            else:
                best_dev_passes = (best["thinned"]["t"] >= MIN_T_STAT and best["positive_folds"] >= MIN_POSITIVE_FOLDS)
                if dev_passes and not best_dev_passes:
                    best = ev
                elif dev_passes == best_dev_passes and nt["exp"] > best["net"]["exp"]:
                    best = ev

    if best is None:
        best = max(all_th_evals, key=lambda x: (x["net"]["n"] >= 10, x["net"]["exp"]))

    # Step 2.3 & 2.6: Train validation model strictly on dev split
    spw_dev = (dev_data["y"] == 0).sum() / max((dev_data["y"] == 1).sum(), 1)
    dev_model = make_model(spw_dev)
    dev_model.fit(dev_data["X"], dev_data["y"])

    # Step 2.3: Evaluate chosen threshold exactly once on untouched 15% holdout
    chosen_th = best["th"]
    if n_holdout_total > 0:
        holdout_probs = dev_model.predict_proba(holdout_data["X"])[:, 1]
        sel_holdout = np.where(holdout_probs >= chosen_th)[0]
        n_holdout = len(sel_holdout)
        holdout_net = trade_stats(holdout_data["net_r"][sel_holdout])
        holdout_gross = trade_stats(holdout_data["gross_r"][sel_holdout])
        holdout_thinned = compute_thinned_stats(
            holdout_data["net_r"][sel_holdout],
            holdout_data["bar"][sel_holdout],
            holdout_data["exit"][sel_holdout]
        )
    else:
        holdout_probs = np.array([])
        n_holdout = 0
        holdout_net = trade_stats(np.array([]))
        holdout_gross = trade_stats(np.array([]))
        holdout_thinned = compute_thinned_stats(np.array([]), np.array([]), np.array([]))

    # Step 2.4: Evaluate edge criteria
    edge_ok, status, criteria = evaluate_edge_criteria(
        n_holdout=n_holdout,
        holdout_net_exp=holdout_net["exp"],
        holdout_gross_exp=holdout_gross["exp"],
        dev_thinned_t=best["thinned"]["t"],
        dev_positive_folds=best["positive_folds"]
    )

    print(f"  🎯 Holdout Evaluation (Cutoff {chosen_th*100:.0f}%): {n_holdout}/{n_holdout_total} trades | "
          f"win {holdout_net['win_rate']:.1f}% | net exp {holdout_net['exp']:+.3f}R (thinned t={holdout_thinned['t']:+.2f}) | "
          f"status: {status} (edge_ok={edge_ok})")

    best["auc"] = float(auc)
    best["dev_data"] = dev_data
    best["holdout_data"] = holdout_data
    best["full_data"] = data
    best["mode"] = mode
    best["baseline"] = base_all
    best["n_folds"] = len(fold_ranges)
    best["all_thresholds"] = all_th_evals
    best["dev_model"] = dev_model
    best["holdout_probs"] = holdout_probs
    best["n_holdout"] = n_holdout
    best["holdout_net"] = holdout_net
    best["holdout_gross"] = holdout_gross
    best["holdout_thinned"] = holdout_thinned
    best["edge_ok"] = edge_ok
    best["status"] = status
    best["criteria"] = criteria

    return best


def train_single_asset(symbol_cfg):
    symbol_name = symbol_cfg["name"]
    model_save_path = symbol_cfg["model_file"]

    print("\n" + "=" * 60)
    print(f"🚀 LEAK-FREE WALK-FORWARD TRAINING: {symbol_name}")
    print("=" * 60)

    resolved_symbol = setup_symbol(symbol_name)
    if not resolved_symbol:
        return {"success": False, "symbol": symbol_name, "error": "Symbol not accessible in MT5"}

    info = mt5.symbol_info(resolved_symbol)
    point = info.point if info else 0.01

    df_low, df_high = fetch_raw(resolved_symbol)
    if df_low is None or len(df_low) < 500:
        return {"success": False, "symbol": symbol_name, "error": "Insufficient market data"}

    candles_low, candles_high = get_history_candle_counts()

    results = []
    for mode in TREND_MODES:
        df = prepare_indicators(df_low, df_high, trend_mode=mode)
        res = validate_mode(df, symbol_cfg, point, mode)
        if res:
            results.append(res)

    if not results:
        return {"success": False, "symbol": symbol_name, "error": "Not enough setups to validate"}

    # Prefer a mode that passes the edge check, then INSUFFICIENT_DATA over FAIL, then highest holdout net exp
    chosen = sorted(
        results,
        key=lambda r: (
            r["edge_ok"],
            r["status"] == "INSUFFICIENT_DATA",
            r["holdout_net"]["exp"] if r["n_holdout"] > 0 else -999.0,
            r["net"]["exp"]
        ),
        reverse=True
    )[0]

    full_data = chosen["full_data"]
    th = chosen["th"]

    print(f"\n✅ Chosen trend mode: '{chosen['mode']}' | cutoff {th*100:.0f}% | "
          f"Holdout net expectancy {chosen['holdout_net']['exp']:+.3f}R over {chosen['n_holdout']} trades "
          f"(Dev OOS: {chosen['net']['exp']:+.3f}R over {chosen['net']['n']} trades, thinned t={chosen['thinned']['t']:+.2f}) | "
          f"edge check: {chosen['status']}")
    if not chosen["edge_ok"]:
        if chosen["status"] == "INSUFFICIENT_DATA":
            print(f"⚠️ Insufficient holdout samples ({chosen['n_holdout']}/{MIN_HOLDOUT_TRADES} trades). "
                  f"Live bot will NOT trade this asset with this model.")
        else:
            print("⛔ Edge criteria failed. Live bot will NOT trade this asset with this model.")

    # Fit final production model on full historical dataset
    spw_full = (full_data["y"] == 0).sum() / max((full_data["y"] == 1).sum(), 1)
    final_model = make_model(spw_full)
    final_model.fit(full_data["X"], full_data["y"])

    # Phase 4.1 Candidate File Isolation (Addendum B1):
    # Save newly trained model ONLY to <model_file>.candidate.pkl.
    # Never directly overwrite <model_file> or .prev.pkl during training.
    candidate_save_path = model_save_path.replace(".pkl", ".candidate.pkl")

    # Format holdout timestamps for post-trained slice evaluation (Phase 4.2)
    holdout_ts = []
    if len(chosen["holdout_data"]["bar"]) > 0:
        for b in chosen["holdout_data"]["bar"]:
            t_raw = df_low.iloc[b]["time"]
            if isinstance(t_raw, (int, float, np.integer, np.floating)):
                holdout_ts.append(datetime.fromtimestamp(float(t_raw)).strftime("%Y-%m-%d %H:%M"))
            else:
                holdout_ts.append(str(t_raw))

    package = {
        "model": final_model,
        "validation_model": chosen["dev_model"],
        "holdout_predictions": chosen["holdout_probs"],
        "holdout_bars": chosen["holdout_data"]["bar"],
        "holdout_timestamps": holdout_ts,
        "holdout_exits": chosen["holdout_data"]["exit"],
        "holdout_net_r": chosen["holdout_data"]["net_r"],
        "features": FEATURE_COLUMNS,
        "symbol": resolved_symbol,
        "recommended_threshold": float(th),
        "trend_mode": chosen["mode"],
        "edge_ok": chosen["edge_ok"],
        "status": chosen["status"],
        "edge_criteria": chosen["criteria"],
        "holdout": {
            "n_total": len(chosen["holdout_data"]["y"]),
            "n_filtered": chosen["n_holdout"],
            "win_rate": chosen["holdout_net"]["win_rate"],
            "avg_win_r": chosen["holdout_net"]["avg_win"],
            "avg_loss_r": chosen["holdout_net"]["avg_loss"],
            "expectancy_r_net": chosen["holdout_net"]["exp"],
            "expectancy_r_gross": chosen["holdout_gross"]["exp"],
            "raw_t": chosen["holdout_net"]["t"],
            "thinned_t": chosen["holdout_thinned"]["t"],
            "thinned_trades": chosen["holdout_thinned"]["n"],
        },
        "oos": {
            "auc": chosen["auc"],
            "trades": chosen["net"]["n"],
            "thinned_trades": chosen["thinned"]["n"],
            "win_rate": chosen["net"]["win_rate"],
            "avg_win_r": chosen["net"]["avg_win"],
            "avg_loss_r": chosen["net"]["avg_loss"],
            "expectancy_r_net": chosen["net"]["exp"],
            "expectancy_r_gross": chosen["gross"]["exp"],
            "raw_t": chosen["net"]["t"],
            "thinned_t": chosen["thinned"]["t"],
            "positive_folds": chosen["positive_folds"],
            "n_folds": chosen["n_folds"],
            "worst_fold_exp_r": chosen["worst_fold_exp"],
        },
        "validation": "15% untouched holdout, boundary purged, leak-free H1, thinned t>=2.5, costs included",
        "trained_date": datetime.now().strftime("%Y-%m-%d %H:%M")
    }
    joblib.dump(package, candidate_save_path)
    print(f"💾 Candidate package saved to {candidate_save_path} (live and .prev.pkl untouched)")

    audit_report = {
        "symbol": symbol_name,
        "resolved_symbol": resolved_symbol,
        "timestamp": datetime.now().isoformat(),
        "history_candles": {
            "candles_low": candles_low,
            "candles_high": candles_high,
        },
        "trend_mode": chosen["mode"],
        "recommended_threshold": float(th),
        "edge_ok": chosen["edge_ok"],
        "status": chosen["status"],
        "edge_criteria": chosen["criteria"],
        "holdout_metrics": {
            "n_total": len(chosen["holdout_data"]["y"]),
            "n_filtered": chosen["n_holdout"],
            "win_rate": chosen["holdout_net"]["win_rate"],
            "expectancy_net_r": chosen["holdout_net"]["exp"],
            "expectancy_gross_r": chosen["holdout_gross"]["exp"],
            "avg_win_r": chosen["holdout_net"]["avg_win"],
            "avg_loss_r": chosen["holdout_net"]["avg_loss"],
            "raw_t": chosen["holdout_net"]["t"],
            "thinned_t": chosen["holdout_thinned"]["t"],
            "thinned_trades": chosen["holdout_thinned"]["n"],
        },
        "dev_walk_forward_metrics": {
            "n_setups_dev": len(chosen["dev_data"]["y"]),
            "auc": chosen["auc"],
            "trades": chosen["net"]["n"],
            "thinned_trades": chosen["thinned"]["n"],
            "win_rate": chosen["net"]["win_rate"],
            "avg_win_r": chosen["net"]["avg_win"],
            "avg_loss_r": chosen["net"]["avg_loss"],
            "expectancy_net_r": chosen["net"]["exp"],
            "expectancy_gross_r": chosen["gross"]["exp"],
            "raw_t": chosen["net"]["t"],
            "thinned_t": chosen["thinned"]["t"],
            "positive_folds": chosen["positive_folds"],
            "n_folds": chosen["n_folds"],
            "worst_fold_exp_r": chosen["worst_fold_exp"],
            "fold_details": chosen["folds"],
        },
        "threshold_grid_evaluations": [
            {
                "threshold": float(item["th"]),
                "trades": item["net"]["n"],
                "thinned_trades": item["thinned"]["n"],
                "win_rate": item["net"]["win_rate"],
                "expectancy_net_r": item["net"]["exp"],
                "expectancy_gross_r": item["gross"]["exp"],
                "raw_t": item["net"]["t"],
                "thinned_t": item["thinned"]["t"],
                "positive_folds": item["positive_folds"],
                "worst_fold_exp_r": item["worst_fold_exp"],
            }
            for item in chosen.get("all_thresholds", [])
        ],
        "model_file": model_save_path
    }
    report_file = save_audit_report(audit_report)
    print(f"📊 Saved audit report to {report_file}")

    return {
        "success": True,
        "symbol": symbol_name,
        "resolved_symbol": resolved_symbol,
        "total_samples": int(len(full_data["y"])),
        "win_samples": int(full_data["y"].sum()),
        "win_rate": chosen["holdout_net"]["win_rate"] if chosen["n_holdout"] > 0 else chosen["net"]["win_rate"],
        "recommended_threshold": float(th),
        "cv_roc_auc": chosen["auc"],
        "trend_mode": chosen["mode"],
        "edge_ok": chosen["edge_ok"],
        "status": chosen["status"],
        "holdout_trades": chosen["n_holdout"],
        "holdout_expectancy_r": chosen["holdout_net"]["exp"],
        "holdout_thinned_t": chosen["holdout_thinned"]["t"],
        "oos_trades": chosen["net"]["n"],
        "oos_expectancy_r": chosen["net"]["exp"],
        "oos_avg_win_r": chosen["net"]["avg_win"],
        "oos_avg_loss_r": chosen["net"]["avg_loss"],
        "positive_folds": chosen["positive_folds"],
        "n_folds": chosen["n_folds"],
        "trained_date": package["trained_date"],
        "model_file": model_save_path,
        "candidate_file": candidate_save_path,
        "report_file": report_file
    }


def run_retrain_task(target_symbol=None):
    """
    Executes training on target_symbol ('XAUUSDm', 'BTCUSDm', or None for both active assets).
    Follows Phase 4 candidate isolation & atomic promotion gate.
    """
    import model_promotion

    if not connect_mt5():
        return [{"success": False, "error": "Could not connect to MT5"}]

    active_symbols = [s for s in TARGET_SYMBOLS if s["name"] in ["XAUUSDm", "BTCUSDm"]]
    results = []
    for cfg in active_symbols:
        sym = cfg["name"]
        if target_symbol:
            clean_target = target_symbol.upper().rstrip("M").rstrip("C")
            if clean_target not in sym.upper():
                continue
        res = train_single_asset(cfg)
        if res and res.get("success"):
            # Phase 4.2 & 4.3 Fair Holdout Comparison Gate and Atomic Promotion
            promo = model_promotion.promote_candidate(cfg["model_file"])
            res["promoted"] = promo.get("promoted", False)
            res["promotion_reason"] = promo.get("reason", "")
            if promo.get("promoted"):
                print(f"🏆 Candidate for {sym} PROMOTED to live model! Reason: {promo['reason']}")
            else:
                print(f"🛑 Candidate for {sym} REJECTED by gate! Reason: {promo['reason']}")
        if res:
            results.append(res)
    return results


def train_all_models():
    if not connect_mt5():
        print("⛔ Open and log in to MetaTrader 5, then run this again.")
        return
    for cfg in TARGET_SYMBOLS:
        if cfg["name"] in ["XAUUSDm", "BTCUSDm"]:
            train_single_asset(cfg)
    mt5.shutdown()
    print("\n🎉 TRAINING FINISHED")


if __name__ == "__main__":
    train_all_models()
