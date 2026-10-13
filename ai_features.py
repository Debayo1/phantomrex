import pandas as pd
import numpy as np
import ta

FEATURE_COLUMNS = [
    "feat_rsi_aligned",
    "feat_ema_spread_aligned",
    "feat_body_momentum_aligned",
    "feat_rejection_wick_aligned",
    "feat_opposite_wick_aligned",
    "feat_dist_ema9_aligned",
    "feat_h1_adx",
    "feat_atr_ratio",
    "feat_volatility_squeeze",
    "feat_hour",
    "feat_is_active_session",
    "feat_prev_body_aligned",
    "feat_vol_ratio"
]

# H1 trend filter modes:
#   "slow" = H1 EMA50 vs EMA200 (original; very laggy, can stay one-sided for days)
#   "fast" = H1 EMA21 vs EMA50 AND H1 close on the same side of EMA50 (reacts sooner, allows buys earlier)
TREND_MODES = ("slow", "fast")
DEFAULT_TREND_MODE = "slow"

H1_BAR_DURATION = pd.Timedelta(hours=1)


def prepare_indicators(df_low: pd.DataFrame, df_high: pd.DataFrame, trend_mode: str = DEFAULT_TREND_MODE) -> pd.DataFrame:
    """
    Computes technical indicators for low (M5) and high (H1) timeframes
    and merges them safely onto the M5 timeline.
    Compatible with both commodities (XAUUSD) and cryptocurrencies (BTCUSD).

    Leak-free: H1 values are attached to M5 bars by the H1 bar's CLOSE time,
    so an M5 bar only ever sees H1 candles that had fully closed by then.
    """
    if trend_mode not in TREND_MODES:
        trend_mode = DEFAULT_TREND_MODE

    df_low = df_low.copy()
    df_high = df_high.copy()

    df_low["time"] = pd.to_datetime(df_low["time"], unit="s" if isinstance(df_low["time"].iloc[0], (int, float, np.number)) else None)
    df_high["time"] = pd.to_datetime(df_high["time"], unit="s" if isinstance(df_high["time"].iloc[0], (int, float, np.number)) else None)

    # --- H1 Macro Indicators ---
    if len(df_high) < 200:
        print(f"⚠️ Warning: prepare_indicators received only {len(df_high)} H1 candles. H1 EMA200 requires >= 200 bars!")
    df_high["h1_ema21"] = ta.trend.EMAIndicator(df_high["close"], window=21).ema_indicator()
    df_high["h1_ema50"] = ta.trend.EMAIndicator(df_high["close"], window=50).ema_indicator()
    df_high["h1_ema200"] = ta.trend.EMAIndicator(df_high["close"], window=200).ema_indicator()
    df_high["h1_adx"] = ta.trend.ADXIndicator(df_high["high"], df_high["low"], df_high["close"], window=14).adx()

    if trend_mode == "fast":
        up = (df_high["h1_ema21"] > df_high["h1_ema50"]) & (df_high["close"] > df_high["h1_ema50"])
        down = (df_high["h1_ema21"] < df_high["h1_ema50"]) & (df_high["close"] < df_high["h1_ema50"])
        df_high["h1_trend"] = np.where(up, 1, np.where(down, -1, 0))
    else:
        df_high["h1_trend"] = np.where(
            df_high["h1_ema50"] > df_high["h1_ema200"], 1,
            np.where(df_high["h1_ema50"] < df_high["h1_ema200"], -1, 0)
        )

    # MT5 stamps candles with their OPEN time. Shift to CLOSE time so the
    # still-forming (or, in history, the future-completed) H1 bar is never used.
    df_high["time"] = df_high["time"] + H1_BAR_DURATION
    # Newer pandas can change datetime resolution after arithmetic; keep both sides identical
    df_high["time"] = df_high["time"].astype("datetime64[ns]")
    df_low["time"] = df_low["time"].astype("datetime64[ns]")

    # Rule G1 Trend Alignment Guard:
    # Every M5 bar needs a closed H1 bar before it.
    # If the H1 history starts later than the oldest M5 bar, drop older M5 bars.
    min_h1_time = df_high["time"].min()
    if pd.notna(min_h1_time):
        df_low = df_low[df_low["time"] >= min_h1_time].copy()

    # --- M5 Micro & Volatility Indicators ---
    df_low["m5_ema9"] = ta.trend.EMAIndicator(df_low["close"], window=9).ema_indicator()
    df_low["m5_ema21"] = ta.trend.EMAIndicator(df_low["close"], window=21).ema_indicator()
    df_low["m5_rsi"] = ta.momentum.RSIIndicator(df_low["close"], window=14).rsi()
    df_low["m5_atr"] = ta.volatility.AverageTrueRange(df_low["high"], df_low["low"], df_low["close"], window=14).average_true_range()
    df_low["m5_atr5"] = ta.volatility.AverageTrueRange(df_low["high"], df_low["low"], df_low["close"], window=5).average_true_range()
    df_low["m5_atr20"] = ta.volatility.AverageTrueRange(df_low["high"], df_low["low"], df_low["close"], window=20).average_true_range()

    # Relative Volume (Tick Volume Surge)
    if "tick_volume" in df_low.columns:
        df_low["vol_sma20"] = df_low["tick_volume"].rolling(window=20).mean()
        df_low["m5_vol_ratio"] = df_low["tick_volume"] / (df_low["vol_sma20"] + 1e-5)
    else:
        df_low["m5_vol_ratio"] = 1.0

    # Align multi-timeframe timelines backwards to avoid lookahead bias
    df = pd.merge_asof(
        df_low.sort_values("time"),
        df_high[["time", "h1_trend", "h1_adx"]].sort_values("time"),
        on="time",
        direction="backward"
    )

    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def extract_single_feature_vector(last_candle: pd.Series, prev_candle: pd.Series, direction: int) -> np.ndarray:
    """
    Extracts a 1D feature array normalized for trade direction (+1 for BUY, -1 for SELL).
    Ensures tree models learn invariant patterns for winning setups across any asset.
    """
    c_range = max(last_candle["high"] - last_candle["low"], 0.01)
    prev_c_range = max(prev_candle["high"] - prev_candle["low"], 0.01)

    # Direction-aligned RSI: higher means more momentum in trade direction
    feat_rsi_aligned = float(last_candle["m5_rsi"] if direction == 1 else (100.0 - last_candle["m5_rsi"]))

    # Direction-aligned EMA spread
    feat_ema_spread_aligned = float(direction * (last_candle["m5_ema9"] - last_candle["m5_ema21"]) / max(last_candle["m5_atr"], 0.01))

    # Candle body momentum in trade direction
    feat_body_momentum_aligned = float(direction * (last_candle["close"] - last_candle["open"]) / c_range)

    # Rejection wick in trade direction (lower wick for BUY, upper wick for SELL)
    upper_wick = (last_candle["high"] - max(last_candle["open"], last_candle["close"])) / c_range
    lower_wick = (min(last_candle["open"], last_candle["close"]) - last_candle["low"]) / c_range

    feat_rejection_wick_aligned = float(lower_wick if direction == 1 else upper_wick)
    feat_opposite_wick_aligned = float(upper_wick if direction == 1 else lower_wick)

    # Distance from EMA9 in trade direction (preventing chasing overextended candles)
    feat_dist_ema9_aligned = float(direction * (last_candle["close"] - last_candle["m5_ema9"]) / max(last_candle["m5_atr"], 0.01))

    feat_h1_adx = float(last_candle["h1_adx"])
    feat_atr_ratio = float((last_candle["m5_atr"] / max(last_candle["close"], 0.01)) * 1000.0)
    feat_volatility_squeeze = float(last_candle["m5_atr5"] / max(last_candle["m5_atr20"], 0.001))

    hour = last_candle["time"].hour if hasattr(last_candle["time"], "hour") else 12
    feat_hour = float(hour)
    feat_is_active_session = 1.0 if 7 <= hour <= 20 else 0.0

    feat_prev_body_aligned = float(direction * (prev_candle["close"] - prev_candle["open"]) / prev_c_range)
    feat_vol_ratio = float(last_candle.get("m5_vol_ratio", 1.0))

    return np.array([
        feat_rsi_aligned,
        feat_ema_spread_aligned,
        feat_body_momentum_aligned,
        feat_rejection_wick_aligned,
        feat_opposite_wick_aligned,
        feat_dist_ema9_aligned,
        feat_h1_adx,
        feat_atr_ratio,
        feat_volatility_squeeze,
        feat_hour,
        feat_is_active_session,
        feat_prev_body_aligned,
        feat_vol_ratio
    ], dtype=np.float32)
