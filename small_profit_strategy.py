"""Signal generation for the demo-only small-profit strategy experiments."""

import numpy as np
import pandas as pd
import ta


STRATEGY_NAMES = ("Trend pullback", "Range fade", "M1 breakout")


def prepare_rates(frame, name):
    required = {"time", "open", "high", "low", "close", "spread"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{name} data missing columns: {sorted(missing)}")
    result = frame.copy().sort_values("time").drop_duplicates("time").reset_index(drop=True)
    result["time"] = pd.to_datetime(result["time"]).astype("datetime64[ns]")
    if len(result) == 0:
        raise ValueError(f"{name} data is empty")
    for column in ("open", "high", "low", "close", "spread"):
        values = result[column].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"{name} has non-finite {column} values")
        if column == "spread" and (values < 0).any():
            raise ValueError(f"{name} contains negative spreads")
    return result


def generate_signals(m1_rates, m5_rates):
    """Create the three fixed candidates using completed bars only."""
    m1 = prepare_rates(m1_rates, "M1")
    m5 = prepare_rates(m5_rates, "M5")
    if len(m1) < 30 or len(m5) < 55:
        return m1.iloc[:0].assign(direction=pd.Series(dtype=int), stop=pd.Series(dtype=float))

    m1["body"] = (m1["close"] - m1["open"]).abs()
    m1["bb_mid"] = m1["close"].rolling(20).mean()
    m1["bb_std"] = m1["close"].rolling(20).std(ddof=0)
    m1["bb_upper"] = m1["bb_mid"] + 2.0 * m1["bb_std"]
    m1["bb_lower"] = m1["bb_mid"] - 2.0 * m1["bb_std"]
    m1["rsi"] = ta.momentum.RSIIndicator(m1["close"], window=14).rsi()
    m1["prior_high_15"] = m1["high"].shift(1).rolling(15).max()
    m1["prior_low_15"] = m1["low"].shift(1).rolling(15).min()
    m1["global_bar"] = np.arange(len(m1), dtype=np.int64)
    m1["close_time"] = m1["time"] + pd.Timedelta(minutes=1)

    m5["ema20"] = ta.trend.EMAIndicator(m5["close"], window=20).ema_indicator()
    m5["ema50"] = ta.trend.EMAIndicator(m5["close"], window=50).ema_indicator()
    m5["close_time"] = m5["time"] + pd.Timedelta(minutes=5)
    aligned = pd.merge_asof(
        m1.sort_values("close_time"),
        m5[["close_time", "close", "ema20", "ema50"]].rename(
            columns={"close": "m5_close", "ema20": "m5_ema20", "ema50": "m5_ema50"}
        ).sort_values("close_time"),
        on="close_time",
        direction="backward",
    ).reset_index(drop=True)

    output = []
    for i in range(1, len(aligned) - 1):
        row = aligned.iloc[i]
        prev = aligned.iloc[i - 1]
        if not np.isfinite([
            row["bb_upper"], row["bb_lower"], row["rsi"],
            row["prior_high_15"], row["prior_low_15"],
            row["m5_close"], row["m5_ema20"], row["m5_ema50"],
        ]).all():
            continue

        candidates = []
        prior_body = float(prev["body"])
        current_body = float(row["body"])
        bull_trend = row["m5_close"] > row["m5_ema50"] and row["m5_ema20"] > row["m5_ema50"]
        bear_trend = row["m5_close"] < row["m5_ema50"] and row["m5_ema20"] < row["m5_ema50"]
        if (
            bull_trend and prev["close"] < prev["open"]
            and row["close"] > row["open"]
            and current_body > prior_body
            and row["close"] > prev["high"]
        ):
            candidates.append(("Trend pullback", 1, float(prev["low"])))
        elif (
            bear_trend and prev["close"] > prev["open"]
            and row["close"] < row["open"]
            and current_body > prior_body
            and row["close"] < prev["low"]
        ):
            candidates.append(("Trend pullback", -1, float(prev["high"])))

        if row["close"] < row["bb_lower"] and row["rsi"] < 30 and row["close"] > row["open"]:
            candidates.append(("Range fade", 1, float(row["low"])))
        elif row["close"] > row["bb_upper"] and row["rsi"] > 70 and row["close"] < row["open"]:
            candidates.append(("Range fade", -1, float(row["high"])))

        if bull_trend and row["close"] > row["prior_high_15"]:
            candidates.append(("M1 breakout", 1, float(row["low"])))
        elif bear_trend and row["close"] < row["prior_low_15"]:
            candidates.append(("M1 breakout", -1, float(row["high"])))

        for name, direction, stop in candidates:
            output.append({
                "signal_bar": int(row["global_bar"]),
                "signal_time": pd.Timestamp(row["time"]),
                "strategy": name,
                "direction": direction,
                "stop": stop,
                "signal_price": float(row["close"]),
                "signal_open": float(row["open"]),
                "signal_high": float(row["high"]),
                "signal_low": float(row["low"]),
                "signal_close": float(row["close"]),
                "spread_points": float(row["spread"]),
                "features": {
                    "rsi14": float(row["rsi"]),
                    "m5_close": float(row["m5_close"]),
                    "m5_ema20": float(row["m5_ema20"]),
                    "m5_ema50": float(row["m5_ema50"]),
                    "bollinger_upper": float(row["bb_upper"]),
                    "bollinger_lower": float(row["bb_lower"]),
                    "prior_high_15": float(row["prior_high_15"]),
                    "prior_low_15": float(row["prior_low_15"]),
                },
            })
    return pd.DataFrame(output)
