"""
Test Suite for Phase 7 Strategy Research Harness (Addendum 5 Revised, H7 Requirements)
========================================================================================
Tests:
  1. Safety test: Runner cannot write live model files or send orders.
  2. No-lookahead test: Signals and HTF values never use data after entry time.
  3. Simulator test: Stop is counted first when both stop and TP are hit in the same bar.
  4. Lockbox test: Zero index overlap between Dev and Lockbox (with boundary purge).
  5. Registry test: Every run adds rows, including failed variants.
  6. Variants test: variants.json contains at most 9 pre-registered variants (V0-V8).
"""

import os
import ast
import csv
import pytest
import numpy as np
import pandas as pd

from research.run_research import (
    assert_safe_environment,
    split_dev_lockbox,
    log_to_registry,
    get_registry_count,
    simulate_v0_v1_v2,
    simulate_v3_v4_breakout,
    simulate_smc_structure_ob_fvg,
    _confirmed_smc_structure,
    TF_MAP,
    REGISTRY_FILE,
    VARIANTS_FILE
)


def test_runner_cannot_write_models_or_send_orders():
    """H7.5: Verify the runner cannot write model files or send live orders."""
    runner_path = os.path.join(os.path.dirname(__file__), "..", "research", "run_research.py")
    with open(runner_path, "r", encoding="utf-8") as f:
        code = f.read()

    tree = ast.parse(code)

    # 1. Verify no mt5.order_send calls exist anywhere in the AST
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute):
                assert node.func.attr != "order_send", "Found forbidden 'order_send' call in run_research.py!"

    # 2. Verify no file writes (open in write mode, joblib.dump, pickle.dump) target live model paths
    forbidden_files = ["gold_ai_model.pkl", "btc_ai_model.pkl", ".candidate.pkl", ".prev.pkl"]
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if (isinstance(node.func, ast.Name) and node.func.id == "open") or \
               (isinstance(node.func, ast.Attribute) and node.func.attr in ("dump", "save")):
                for arg in node.args:
                    if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                        for bad in forbidden_files:
                            assert bad not in arg.value, f"Forbidden file target in call: {arg.value}"

    # 3. Verify assert_safe_environment blocks unsafe modules safely
    import sys
    orig_bot = sys.modules.get("bot")
    try:
        sys.modules["bot"] = "mock_unsafe_module"
        with pytest.raises(RuntimeError, match="Unsafe module 'bot'"):
            assert_safe_environment()
    finally:
        if orig_bot is not None:
            sys.modules["bot"] = orig_bot
        else:
            sys.modules.pop("bot", None)


def test_simulator_tie_break_prefers_stop():
    """H7.2: Verify conservative rule: stop is counted first when both stop and TP hit in same bar."""
    sym_cfg = {
        "name": "XAUUSDm",
        "min_stop_dist": 1.50,
        "spread_est": 0.25,
        "contract_size": 100.0,
        "point": 0.01
    }

    h1_times = pd.date_range("2025-12-30 00:00", periods=60, freq="1h")
    # Rising trend so htf_close > htf_ema50
    df_trend = pd.DataFrame({
        "time": h1_times,
        "open": [1950.0 + i * 2.0 for i in range(60)],
        "high": [1955.0 + i * 2.0 for i in range(60)],
        "low": [1945.0 + i * 2.0 for i in range(60)],
        "close": [1952.0 + i * 2.0 for i in range(60)],
        "spread": [25] * 60
    })

    entry_times = pd.date_range("2026-01-01 00:00", periods=40, freq="1h")
    df_entry = pd.DataFrame({
        "time": entry_times,
        "open": 2000.0,
        "high": 2002.0,
        "low": 1998.0,
        "close": 2000.0,
        "spread": [25] * 40
    })
    # Channel breakout on bar 25
    df_entry.loc[25, "close"] = 2015.0
    df_entry.loc[25, "high"] = 2016.0
    # Entry on bar 26 touches BOTH TP (high 2050) and SL (low 1950)!
    df_entry.loc[26, "open"] = 2015.0
    df_entry.loc[26, "high"] = 2050.0
    df_entry.loc[26, "low"] = 1950.0

    trades = simulate_v3_v4_breakout(df_entry, df_trend, "H1", "H4", channel_bars=10, stop_atr_mult=1.5, tp_r=2.5, sym_cfg=sym_cfg)
    assert len(trades) > 0, "Expected at least 1 trade in tie-break test"
    for t in trades:
        # Rule H2: When both SL and TP are touched in the same bar, STOP is counted first
        assert t["exit_reason"] in ("SL", "SL_TIE"), f"Tie-break violation! Exit reason was {t['exit_reason']}"
        assert t["gross_r"] < 0, f"Tie-break violation! Expected negative R on dual touch, got {t['gross_r']}"


def test_no_lookahead_entry():
    """H7.1: Verify trade signals evaluate on closed bars and enter at the open of the next bar."""
    sym_cfg = {
        "name": "XAUUSDm",
        "min_stop_dist": 1.50,
        "spread_est": 0.25,
        "contract_size": 100.0,
        "point": 0.01
    }

    h1_times = pd.date_range("2025-12-30 00:00", periods=60, freq="1h")
    df_trend = pd.DataFrame({
        "time": h1_times,
        "open": [1950.0 + i * 2.0 for i in range(60)],
        "high": [1955.0 + i * 2.0 for i in range(60)],
        "low": [1945.0 + i * 2.0 for i in range(60)],
        "close": [1952.0 + i * 2.0 for i in range(60)],
        "spread": [25] * 60
    })

    entry_times = pd.date_range("2026-01-01 00:00", periods=40, freq="1h")
    df_entry = pd.DataFrame({
        "time": entry_times,
        "open": 2000.0,
        "high": 2002.0,
        "low": 1998.0,
        "close": 2000.0,
        "spread": [25] * 40
    })
    # Signal forms on bar 20
    df_entry.loc[20, "close"] = 2015.0
    df_entry.loc[20, "high"] = 2016.0
    # Next bar opens at 2017.0 (unique open)
    df_entry.loc[21, "open"] = 2017.0
    df_entry.loc[21, "close"] = 2025.0
    df_entry.loc[21, "high"] = 2030.0
    df_entry.loc[21, "low"] = 2016.0

    trades = simulate_v3_v4_breakout(df_entry, df_trend, "H1", "H4", channel_bars=10, stop_atr_mult=1.5, tp_r=2.5, sym_cfg=sym_cfg)
    assert len(trades) > 0, "Expected at least 1 trade in no-lookahead test"
    for t in trades:
        # 1. Entry price must strictly equal open price of entry bar at entry_time
        entry_row = df_entry[df_entry["time"] == t["entry_time"]].iloc[0]
        assert t["entry_price"] == entry_row["open"], (
            f"Lookahead violation! Entry price {t['entry_price']} != open price {entry_row['open']}"
        )
        assert t["entry_time"] == entry_row["time"]


def test_lockbox_purge_and_no_index_overlap():
    """H7.3: Verify lockbox rows are never used in dev statistics (no index overlap, with purge)."""
    n = 100
    bars = np.arange(0, n * 5, 5)
    exits = bars + np.random.randint(10, 50, size=n)

    synthetic_data = {
        "X": np.zeros((n, 5)),
        "y": np.random.randint(0, 2, size=n),
        "gross_r": np.random.uniform(-1, 2, size=n),
        "net_r": np.random.uniform(-1.2, 1.8, size=n),
        "bar": bars,
        "exit": exits,
        "be": np.zeros(n, dtype=bool),
        "dir": np.ones(n, dtype=int),
        "lookahead_bars": 48
    }

    dev_data, lockbox_data = split_dev_lockbox(synthetic_data, lockbox_pct=0.20)

    # 1. Dev + Lockbox counts must be <= total (due to purged boundary trades)
    assert len(dev_data["y"]) + len(lockbox_data["y"]) <= n

    # 2. Lockbox must start at exactly the 80% split bar
    split_idx = int(np.floor(n * 0.80))
    lockbox_start_bar = synthetic_data["bar"][split_idx]

    # 3. All dev trades must have completely exited STRICTLY BEFORE lockbox_start_bar
    assert np.all(dev_data["exit"] < lockbox_start_bar), (
        "Boundary purge failure: found dev trade exit extending into lockbox period!"
    )

    # 4. Zero overlap between dev bars and lockbox bars
    dev_bar_set = set(dev_data["bar"])
    lockbox_bar_set = set(lockbox_data["bar"])
    overlap = dev_bar_set.intersection(lockbox_bar_set)
    assert len(overlap) == 0, f"Index overlap detected between dev and lockbox: {overlap}"


def test_registry_count_increases_for_every_run(tmp_path, monkeypatch):
    """H7.4: Verify registry count increases for every variant run, including failed ones."""
    test_reg_file = str(tmp_path / "test_registry.csv")
    monkeypatch.setattr("research.run_research.REGISTRY_FILE", test_reg_file)

    # Initially empty / 0
    assert get_registry_count() == 0

    # Log run 1 (failed)
    log_to_registry("V0", "XAUUSDm", "M5/H1_TP=1.8", "Net=-0.05R,status=FAIL")
    assert get_registry_count() == 1

    # Log run 2 (failed due to insufficient data)
    log_to_registry("V5", "XAUUSDm", "M15_Session_TP=1.5", "Net=+0.01R,trades=15,status=INSUFFICIENT_DATA")
    assert get_registry_count() == 2

    # Log run 3 (passed)
    log_to_registry("V3", "BTCUSDm", "H1/H4_TP=2.5", "Net=+0.15R,trades=120,status=PASS")
    assert get_registry_count() == 3

    # Verify rows in CSV
    with open(test_reg_file, "r", encoding="utf-8") as f:
        reader = list(csv.reader(f))
        assert len(reader) == 4  # Header + 3 rows
        assert "FAIL" in reader[1][5]
        assert "INSUFFICIENT_DATA" in reader[2][5]
        assert "PASS" in reader[3][5]


def test_runtime_registry_is_outside_tracked_research_directory():
    """Runtime research writes must not target the tracked seed registry."""
    assert os.path.basename(os.path.dirname(REGISTRY_FILE)) == "runtime_data"
    assert os.path.basename(REGISTRY_FILE) == "strategy_registry.csv"


def test_variants_json_max_nine_pre_registered():
    """Verify variants.json contains original variants and one research-only SMC proxy."""
    assert os.path.exists(VARIANTS_FILE)
    import json
    with open(VARIANTS_FILE, "r", encoding="utf-8") as f:
        variants = json.load(f)

    assert len(variants) <= 9, f"Too many variants! Found {len(variants)}, max allowed is 9."
    assert len(variants) == 9, f"Expected 9 variants, found {len(variants)}."
    assert set(variants.keys()) == {"V0", "V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8"}
    assert variants["V8"]["strategy_type"] == "smc_structure_ob_fvg"
    assert variants["V8"]["entry_tf"] == "M15"
    assert variants["V8"]["trend_tfs"] == ["D1", "H4", "H2"]


def test_smc_confirmed_swings_do_not_use_unconfirmed_bars():
    df = pd.DataFrame({
        "time": pd.date_range("2026-01-01", periods=9, freq="2h"),
        "high": [10, 12, 14, 12, 11, 13, 15, 13, 12],
        "low": [5, 7, 9, 7, 6, 8, 10, 8, 7],
        "open": [8] * 9,
        "close": [9] * 9,
    })

    structure = _confirmed_smc_structure(df, "H2", swing_bars=2)

    assert TF_MAP["H2"][1] == pd.Timedelta(hours=2)
    assert structure.loc[4, "swing_high"] == 14
    assert structure.loc[4, "swing_high_index"] == 2
    assert structure.loc[2, "swing_high_index"] == -1


def test_smc_simulator_requires_all_timeframes_and_returns_no_trade_for_flat_data():
    times_m15 = pd.date_range("2026-01-01", periods=200, freq="15min")
    times_h2 = pd.date_range("2025-12-01", periods=120, freq="2h")
    times_h4 = pd.date_range("2025-11-01", periods=100, freq="4h")
    times_d1 = pd.date_range("2025-01-01", periods=100, freq="1D")

    def flat_frame(times):
        return pd.DataFrame({
            "time": times,
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.0,
            "spread": 10,
        })

    sym_cfg = {"point": 0.01, "spread_est": 0.1, "contract_size": 1.0}
    flat = flat_frame(times_m15)
    assert simulate_smc_structure_ob_fvg(flat, None, flat_frame(times_h4), flat_frame(times_h2), sym_cfg) == []
    assert simulate_smc_structure_ob_fvg(
        flat, flat_frame(times_d1), flat_frame(times_h4), flat_frame(times_h2), sym_cfg
    ) == []
