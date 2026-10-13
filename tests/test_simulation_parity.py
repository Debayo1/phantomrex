import pytest
import numpy as np
import pandas as pd

from train_model import generate_candidate_labels, STOP_ATR_MULT, TP_MULT, BE_TRIGGER_R, LOOKAHEAD_BARS


def test_risk_and_target_formula_parity():
    """
    Phase 5.4 Simulation Parity Test:
    Verifies that simulation parameters and live parameters share identical formulas
    for stop distance, breakeven trigger, breakeven buffer, and take profit.
    """
    entry = 2000.0
    atr = 2.0
    min_stop_dist = 1.5
    spread_est = 0.25
    point = 0.01

    # Live formulas
    live_stop = max(atr * 1.5, min_stop_dist)
    live_tp_buy = entry + (1.8 * live_stop)
    live_be_trigger_dist = 0.9 * live_stop
    live_be_buffer = max(spread_est, point * 10)
    live_be_sl_buy = entry + live_be_buffer

    # Simulation formulas
    sim_stop = max(atr * STOP_ATR_MULT, min_stop_dist)
    sim_tp_buy = entry + (1 * sim_stop * TP_MULT)
    sim_be_trigger_dist = BE_TRIGGER_R * sim_stop
    sim_be_buffer = max(spread_est, point * 10)
    sim_be_sl_buy = entry + sim_be_buffer

    # Tolerance checks
    assert abs(live_stop - sim_stop) < 1e-6
    assert abs(live_tp_buy - sim_tp_buy) < 1e-6
    assert abs(live_be_trigger_dist - sim_be_trigger_dist) < 1e-6
    assert abs(live_be_buffer - sim_be_buffer) < 1e-6
    assert abs(live_be_sl_buy - sim_be_sl_buy) < 1e-6


def test_simulation_trade_lifecycles():
    """
    Simulates synthetic price scenarios through generate_candidate_labels:
    1. Direct TP hit
    2. Direct SL hit
    3. Breakeven lock followed by pullback to breakeven
    4. AI Early Invalidation Exit
    """
    n_bars = 60
    # Construct synthetic dataframe with 60 bars
    df = pd.DataFrame({
        "time": [pd.Timestamp("2026-01-01 10:00:00") + pd.Timedelta(minutes=5*i) for i in range(n_bars)],
        "open": np.full(n_bars, 2000.0),
        "high": np.full(n_bars, 2001.0),
        "low": np.full(n_bars, 1999.0),
        "close": np.full(n_bars, 2000.0),
        "m5_ema9": np.full(n_bars, 2000.0),
        "m5_ema21": np.full(n_bars, 1998.0),
        "m5_rsi": np.full(n_bars, 50.0),
        "m5_atr": np.full(n_bars, 2.0),
        "m5_atr5": np.full(n_bars, 2.0),
        "m5_atr20": np.full(n_bars, 2.0),
        "m5_vol_ratio": np.full(n_bars, 1.0),
        "h1_adx": np.full(n_bars, 25.0),
        "h1_trend": np.full(n_bars, 1),
        "spread": np.full(n_bars, 25), # 25 points = 0.25 spread
    })

    # Trigger a BUY setup at bar 5:
    # crossover_buy: e9[4] <= e21[4] and e9[5] > e21[5]
    df.loc[4, "m5_ema9"] = 1997.0
    df.loc[4, "m5_ema21"] = 1998.0
    df.loc[5, "m5_ema9"] = 2000.0
    df.loc[5, "m5_ema21"] = 1998.0
    df.loc[5, "close"] = 2000.0
    df.loc[5, "open"] = 1999.0

    # Scenario A: Direct TP Hit
    # Entry = 2000, stop = 3.0, TP = 2005.4
    df_tp = df.copy()
    # At bar 8, high reaches 2006.0
    df_tp.loc[8, "high"] = 2006.0
    res_tp = generate_candidate_labels(df_tp, min_stop_dist=1.5, spread_est=0.25, point=0.01)
    assert len(res_tp["y"]) == 1
    assert res_tp["y"][0] == 1
    assert res_tp["exit"][0] == 8
    assert abs(res_tp["gross_r"][0] - 1.8) < 1e-4

    # Scenario B: Direct SL Hit
    # Entry = 2000, stop = 3.0, SL = 1997.0
    df_sl = df.copy()
    df_sl.loc[8, "low"] = 1996.0
    res_sl = generate_candidate_labels(df_sl, min_stop_dist=1.5, spread_est=0.25, point=0.01)
    assert len(res_sl["y"]) == 1
    assert res_sl["y"][0] == 0
    assert res_sl["exit"][0] == 8
    assert abs(res_sl["gross_r"][0] - (-1.0)) < 1e-4

    # Scenario C: Breakeven lock followed by pullback
    # At bar 7, price reaches 2003.0 (0.9R = 2.7 -> 2002.7 reached!)
    # At bar 8, price stays elevated (low = 2001.0)
    # At bar 9, price drops to 2000.0 (breakeven buffer = 0.25 -> SL moved to 2000.25)
    df_be = df.copy()
    df_be.loc[7, "high"] = 2003.0
    df_be.loc[8, "low"] = 2001.0
    df_be.loc[9, "low"] = 2000.0 # Touches 2000.25
    res_be = generate_candidate_labels(df_be, min_stop_dist=1.5, spread_est=0.25, point=0.01)
    assert len(res_be["y"]) == 1
    assert bool(res_be["be"][0]) is True
    assert res_be["exit"][0] == 9
    # Exit price should be ~2000.25, gross_r ~ +0.25/3.0 = +0.0833R
    assert res_be["gross_r"][0] > -0.1

    # Scenario D: AI Early Structural Invalidation
    # At bar 7, price enters loss: close = 1998.5
    # H1 trend flips to -1 and M5 EMAs flip: e9 < e21
    df_ai = df.copy()
    df_ai.loc[7, "close"] = 1998.5
    df_ai.loc[7, "low"] = 1998.0
    df_ai.loc[7, "h1_trend"] = -1
    df_ai.loc[7, "m5_ema9"] = 1997.0
    df_ai.loc[7, "m5_ema21"] = 1999.0
    res_ai = generate_candidate_labels(df_ai, min_stop_dist=1.5, spread_est=0.25, point=0.01)
    assert len(res_ai["y"]) >= 1
    assert res_ai["exit"][0] == 7
    # Loss is cut early at 1998.5 (-0.5R) rather than full -1.0R SL
    assert res_ai["gross_r"][0] == pytest.approx(-0.5, abs=0.05)
