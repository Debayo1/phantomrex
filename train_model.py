import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import joblib
import os
import shutil
import sys
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

CANDLES_LOW = 40000   # ~6 months of M5 candles
CANDLES_HIGH = 6000   # ~6 months of H1 candles

# --- Trade simulation (mirrors live bot.py behaviour) ---
STOP_ATR_MULT = 1.5          # bot.py ATR_MULTIPLIER
TP_MULT = 1.8                # TP_Core target (the single position used on small accounts)
BE_TRIGGER_R = 0.9           # bot.py moves SL to breakeven at 0.9R
LOOKAHEAD_BARS = 48          # max holding time (4h on M5); also used as purge gap
SLIPPAGE_FRACTION = 0.25     # extra slippage allowance = 25% of spread

# --- Validation ---
N_FOLDS = 5
MIN_TRADES_FOR_THRESHOLD = 30
MIN_POSITIVE_FOLDS = 4       # PRD success metric: positive expectancy in >= 4 of 5 folds
MIN_TRADES_FOR_EDGE = 100    # need a meaningful out-of-sample sample size before trusting an edge
MIN_T_STAT = 2.5             # expectancy must be clearly above zero (guards against lucky noise
                             # from trying several cutoffs x 2 trend modes)
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
    print(f"📥 Fetching historical market data for {symbol} ({CANDLES_LOW} M5 candles)...")
    rates_low = mt5.copy_rates_from_pos(symbol, TIMEFRAME_LOW, 0, CANDLES_LOW)
    rates_high = mt5.copy_rates_from_pos(symbol, TIMEFRAME_HIGH, 0, CANDLES_HIGH)

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
# 7.2  WALK-FORWARD VALIDATION WITH PURGE
# ==============================================================================

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
    net, gross = data["net_r"], data["gross_r"]
    per_fold = []
    for k, idx in fold_ranges:
        sel = idx[oos_prob[idx] >= th]
        per_fold.append({"fold": k, **trade_stats(net[sel])})
    all_idx = np.concatenate([idx for _, idx in fold_ranges]) if fold_ranges else np.array([], dtype=int)
    sel_all = all_idx[oos_prob[all_idx] >= th]
    pooled_net = trade_stats(net[sel_all])
    pooled_gross = trade_stats(gross[sel_all])
    positive_folds = sum(1 for f in per_fold if f["n"] > 0 and f["exp"] > 0)
    worst = min((f["exp"] for f in per_fold if f["n"] > 0), default=0.0)
    return {"th": th, "net": pooled_net, "gross": pooled_gross, "folds": per_fold,
            "positive_folds": positive_folds, "worst_fold_exp": worst}


def validate_mode(df, symbol_cfg, point, mode):
    data = generate_candidate_labels(df, symbol_cfg["min_stop_dist"], symbol_cfg["spread_est"], point)
    n = len(data["y"])
    print(f"\n  ── Trend mode '{mode}': {n} setups "
          f"({int((data['dir'] == 1).sum())} buys / {int((data['dir'] == -1).sum())} sells)")
    if n < 150:
        print("  ⚠️ Too few setups for walk-forward validation.")
        return None

    base_all = trade_stats(data["net_r"])
    print(f"  Take-every-setup (no AI), after costs: win {base_all['win_rate']:.1f}% | "
          f"exp {base_all['exp']:+.3f}R | avgW {base_all['avg_win']:+.2f}R avgL {base_all['avg_loss']:+.2f}R")

    oos_prob, fold_ranges = walk_forward(data)
    valid = ~np.isnan(oos_prob)
    auc = roc_auc_score(data["y"][valid], oos_prob[valid]) if valid.sum() > 0 and len(np.unique(data["y"][valid])) > 1 else 0.5
    print(f"  Out-of-sample ROC-AUC: {auc:.3f}  (0.50 = coin flip)")

    best = None
    for th in THRESHOLD_GRID:
        ev = evaluate_threshold(data, oos_prob, fold_ranges, th)
        nt, gr = ev["net"], ev["gross"]
        print(f"   ≥{th*100:.0f}%: trades {nt['n']:4d} | win {nt['win_rate']:5.1f}% | "
              f"exp gross {gr['exp']:+.3f}R → net {nt['exp']:+.3f}R (t={nt['t']:+.2f}) | "
              f"+folds {ev['positive_folds']}/{len(fold_ranges)} | worst fold {ev['worst_fold_exp']:+.3f}R")
        if nt["n"] >= MIN_TRADES_FOR_THRESHOLD and (best is None or nt["exp"] > best["net"]["exp"]):
            best = ev

    if best is None:
        return None
    best["auc"] = float(auc)
    best["data"] = data
    best["mode"] = mode
    best["baseline"] = base_all
    best["n_folds"] = len(fold_ranges)
    best["edge_ok"] = bool(
        best["net"]["exp"] > 0
        and best["positive_folds"] >= MIN_POSITIVE_FOLDS
        and best["net"]["n"] >= MIN_TRADES_FOR_EDGE
        and best["net"]["t"] >= MIN_T_STAT
    )
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

    results = []
    for mode in TREND_MODES:
        df = prepare_indicators(df_low, df_high, trend_mode=mode)
        res = validate_mode(df, symbol_cfg, point, mode)
        if res:
            results.append(res)

    if not results:
        return {"success": False, "symbol": symbol_name, "error": "Not enough setups to validate"}

    # Prefer a mode that passes the edge check, then the highest net expectancy
    chosen = sorted(results, key=lambda r: (r["edge_ok"], r["net"]["exp"]), reverse=True)[0]
    data = chosen["data"]
    th = chosen["th"]

    print(f"\n✅ Chosen trend mode: '{chosen['mode']}' | cutoff {th*100:.0f}% | "
          f"OOS net expectancy {chosen['net']['exp']:+.3f}R over {chosen['net']['n']} trades | "
          f"edge check: {'PASS' if chosen['edge_ok'] else 'FAIL'}")
    if not chosen["edge_ok"]:
        print("⛔ No reliable edge out-of-sample. Live bot will NOT trade this asset with this model.")

    spw = (data["y"] == 0).sum() / max((data["y"] == 1).sum(), 1)
    final_model = make_model(spw)
    final_model.fit(data["X"], data["y"])

    # 7.9 Keep previous model for rollback
    if os.path.exists(model_save_path):
        shutil.copyfile(model_save_path, model_save_path.replace(".pkl", ".prev.pkl"))

    package = {
        "model": final_model,
        "features": FEATURE_COLUMNS,
        "symbol": resolved_symbol,
        "recommended_threshold": float(th),
        "trend_mode": chosen["mode"],
        "edge_ok": chosen["edge_ok"],
        "oos": {
            "auc": chosen["auc"],
            "trades": chosen["net"]["n"],
            "win_rate": chosen["net"]["win_rate"],
            "avg_win_r": chosen["net"]["avg_win"],
            "avg_loss_r": chosen["net"]["avg_loss"],
            "expectancy_r_net": chosen["net"]["exp"],
            "expectancy_r_gross": chosen["gross"]["exp"],
            "positive_folds": chosen["positive_folds"],
            "n_folds": chosen["n_folds"],
            "worst_fold_exp_r": chosen["worst_fold_exp"],
        },
        "validation": "walk-forward, purged, leak-free H1, costs included",
        "trained_date": datetime.now().strftime("%Y-%m-%d %H:%M")
    }
    joblib.dump(package, model_save_path)
    print(f"💾 Saved to {model_save_path} (previous model kept as .prev.pkl)")

    return {
        "success": True,
        "symbol": symbol_name,
        "resolved_symbol": resolved_symbol,
        "total_samples": int(len(data["y"])),
        "win_samples": int(data["y"].sum()),
        "win_rate": chosen["net"]["win_rate"],          # out-of-sample, after costs
        "recommended_threshold": float(th),
        "cv_roc_auc": chosen["auc"],
        "trend_mode": chosen["mode"],
        "edge_ok": chosen["edge_ok"],
        "oos_trades": chosen["net"]["n"],
        "oos_expectancy_r": chosen["net"]["exp"],
        "oos_avg_win_r": chosen["net"]["avg_win"],
        "oos_avg_loss_r": chosen["net"]["avg_loss"],
        "positive_folds": chosen["positive_folds"],
        "n_folds": chosen["n_folds"],
        "trained_date": package["trained_date"],
        "model_file": model_save_path
    }


def run_retrain_task(target_symbol=None):
    """
    Executes training on target_symbol ('XAUUSDm', 'BTCUSDm', or None for both active assets).
    Safe for programmatic execution inside background threads.
    """
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
