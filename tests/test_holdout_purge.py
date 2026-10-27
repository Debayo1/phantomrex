import os
import json
import csv
import numpy as np
import pytest

import train_model
from train_model import (
    get_history_candle_counts,
    DEFAULT_CANDLES_LOW,
    DEFAULT_CANDLES_HIGH,
    split_dev_holdout,
    compute_thinned_stats,
    evaluate_edge_criteria,
    to_json_serializable,
    save_audit_report,
    trade_stats,
)


def test_configurable_history_defaults_and_env_override(monkeypatch):
    """Verifies default candle history and environment variable overrides."""
    # Ensure env vars are clean and config file path is isolated to test defaults
    monkeypatch.setattr("train_model.CONFIG_FILE_PATH", "non_existent_config.json")
    monkeypatch.delenv("CANDLES_LOW", raising=False)
    monkeypatch.delenv("CANDLES_HIGH", raising=False)

    low, high = get_history_candle_counts()
    assert low == DEFAULT_CANDLES_LOW
    assert high == DEFAULT_CANDLES_HIGH

    # Override low
    monkeypatch.setenv("CANDLES_LOW", "50000")
    low, high = get_history_candle_counts()
    assert low == 50000
    assert high == DEFAULT_CANDLES_HIGH

    # Override high
    monkeypatch.setenv("CANDLES_HIGH", "8000")
    low, high = get_history_candle_counts()
    assert low == 50000
    assert high == 8000

    # Non-integer fallback
    monkeypatch.setenv("CANDLES_LOW", "invalid")
    low, high = get_history_candle_counts()
    assert low == DEFAULT_CANDLES_LOW


def test_retrain_registry_writes_to_runtime_data(tmp_path, monkeypatch):
    registry_path = tmp_path / "runtime_data" / "retrain_registry.csv"
    monkeypatch.setattr(train_model, "RETRAIN_REGISTRY_FILE", str(registry_path))
    chosen = {
        "n_holdout": 12,
        "holdout_net": {"exp": -0.1},
        "net": {"exp": 0.05},
        "thinned": {"t": 0.4},
        "positive_folds": 2,
        "n_folds": 5,
        "edge_ok": False,
        "status": "INSUFFICIENT_DATA",
    }

    train_model.log_retrain_to_registry("XAUUSDm", 40000, 6000, chosen)

    with registry_path.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    assert rows[0] == ["date", "variant", "params", "dev_result", "pass_fail"]
    assert len(rows) == 2
    assert rows[1][2] == "40000M5_6000H1_XAUUSDm"
    assert "INSUFFICIENT_DATA" in rows[1][4]


def test_split_dev_holdout_math():
    """Verifies 85/15 chronological split math without overlaps."""
    n = 100
    # Create non-overlapping trades
    bars = np.arange(0, n * 10, 10)
    exits = bars + 5

    data = {
        "X": np.zeros((n, 5)),
        "y": np.zeros(n, dtype=int),
        "gross_r": np.ones(n),
        "net_r": np.ones(n),
        "bar": bars,
        "exit": exits,
        "be": np.zeros(n, dtype=bool),
        "dir": np.ones(n, dtype=int),
    }

    dev, holdout = split_dev_holdout(data, holdout_pct=0.15)
    # 100 * 0.85 = 85
    assert len(holdout["y"]) == 15
    assert len(dev["y"]) == 85
    assert holdout["bar"][0] == bars[85]


def test_boundary_purge_strictly_excludes_overlapping_trades():
    """Verifies that any trade in dev extending into the holdout start bar is purged."""
    # 10 setups
    # split_idx for 15% of 10 is 8 (floor(10 * 0.85) = 8)
    # holdout_start_bar is bar[8] = 80
    bars = np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90])
    exits = np.array([
        15,  # 0: exits before 80 -> KEEP
        25,  # 1: exits before 80 -> KEEP
        35,  # 2: exits before 80 -> KEEP
        45,  # 3: exits before 80 -> KEEP
        55,  # 4: exits before 80 -> KEEP
        65,  # 5: exits before 80 -> KEEP
        75,  # 6: exits before 80 -> KEEP
        85,  # 7: exits at 85 >= 80 -> MUST BE PURGED!
        95,  # 8: holdout
        105, # 9: holdout
    ])

    data = {
        "X": np.zeros((10, 2)),
        "y": np.ones(10, dtype=int),
        "gross_r": np.ones(10),
        "net_r": np.ones(10),
        "bar": bars,
        "exit": exits,
        "be": np.zeros(10, dtype=bool),
        "dir": np.ones(10, dtype=int),
    }

    dev, holdout = split_dev_holdout(data, holdout_pct=0.15)
    holdout_start_bar = bars[8]
    assert holdout_start_bar == 80
    assert len(holdout["y"]) == 2

    # Dev candidates were 0..7. Setup 7 had exit 85 >= 80, so it must be purged!
    assert len(dev["y"]) == 7
    # Verify no trade in dev has exit >= holdout_start_bar
    assert np.all(dev["exit"] < holdout_start_bar)
    # Setup 7 bar index (70) must not be in dev["bar"]
    assert 70 not in dev["bar"]


def test_compute_thinned_stats_chronological_thinning():
    """Verifies greedy chronological thinning of overlapping trades."""
    # Trade 0: bar 10, exit 20, r=1.0  (Kept)
    # Trade 1: bar 15, exit 25, r=-1.0 (Overlaps Trade 0 -> Skipped)
    # Trade 2: bar 20, exit 30, r=1.5  (Starts at Trade 0 exit -> Kept)
    # Trade 3: bar 22, exit 35, r=-0.5 (Overlaps Trade 2 -> Skipped)
    # Trade 4: bar 30, exit 40, r=2.0  (Starts at Trade 2 exit -> Kept)
    bars = np.array([10, 15, 20, 22, 30])
    exits = np.array([20, 25, 30, 35, 40])
    net_r = np.array([1.0, -1.0, 1.5, -0.5, 2.0])

    stats = compute_thinned_stats(net_r, bars, exits)
    assert stats["raw_n"] == 5
    assert stats["thinned_n"] == 3
    # Expected kept: [1.0, 1.5, 2.0]
    expected_mean = (1.0 + 1.5 + 2.0) / 3.0
    assert pytest.approx(stats["exp"], rel=1e-4) == expected_mean


def test_evaluate_edge_criteria_insufficient_data():
    """Verifies that holdout trades < 30 returns INSUFFICIENT_DATA and edge_ok = False."""
    edge_ok, status, checks = evaluate_edge_criteria(
        n_holdout=29,
        holdout_net_exp=0.35,
        holdout_gross_exp=0.45,
        dev_thinned_t=3.2,
        dev_positive_folds=5,
    )
    assert edge_ok is False
    assert status == "INSUFFICIENT_DATA"
    assert checks["holdout_trades_ge_30"] is False
    assert checks["holdout_net_exp_gt_0"] is True


def test_evaluate_edge_criteria_fail_on_negative_exp():
    """Verifies that negative net expectancy marks FAIL and edge_ok = False."""
    edge_ok, status, checks = evaluate_edge_criteria(
        n_holdout=40,
        holdout_net_exp=-0.05,
        holdout_gross_exp=0.08,
        dev_thinned_t=3.0,
        dev_positive_folds=5,
    )
    assert edge_ok is False
    assert status == "FAIL"
    assert checks["holdout_trades_ge_30"] is True
    assert checks["holdout_net_exp_gt_0"] is False


def test_evaluate_edge_criteria_fail_on_low_t_stat():
    """Verifies that dev thinned t-stat < 2.5 marks FAIL and edge_ok = False."""
    edge_ok, status, checks = evaluate_edge_criteria(
        n_holdout=40,
        holdout_net_exp=0.20,
        holdout_gross_exp=0.30,
        dev_thinned_t=2.1,  # Below 2.5
        dev_positive_folds=5,
    )
    assert edge_ok is False
    assert status == "FAIL"
    assert checks["dev_thinned_t_ge_2_5"] is False


def test_evaluate_edge_criteria_fail_on_low_positive_folds():
    """Verifies that positive folds < 4 marks FAIL and edge_ok = False."""
    edge_ok, status, checks = evaluate_edge_criteria(
        n_holdout=40,
        holdout_net_exp=0.20,
        holdout_gross_exp=0.30,
        dev_thinned_t=2.8,
        dev_positive_folds=3,  # Below 4
    )
    assert edge_ok is False
    assert status == "FAIL"
    assert checks["dev_positive_folds_ge_4"] is False


def test_evaluate_edge_criteria_pass_all_conditions():
    """Verifies that all 5 criteria met marks PASS and edge_ok = True."""
    edge_ok, status, checks = evaluate_edge_criteria(
        n_holdout=35,
        holdout_net_exp=0.25,
        holdout_gross_exp=0.35,
        dev_thinned_t=2.7,
        dev_positive_folds=4,
    )
    assert edge_ok is True
    assert status == "PASS"
    assert all(checks.values())


def test_to_json_serializable_and_save_audit_report(tmp_path):
    """Verifies NumPy conversion and structured JSON report persistence."""
    data = {
        "int_val": np.int64(42),
        "float_val": np.float64(3.14159),
        "bool_val": np.bool_(True),
        "array_val": np.array([1, 2, 3]),
        "nested": {
            "sub_array": np.array([0.1, 0.2]),
            "sub_bool": np.bool_(False)
        }
    }
    serialized = to_json_serializable(data)
    # Check native types
    assert isinstance(serialized["int_val"], int)
    assert isinstance(serialized["float_val"], float)
    assert isinstance(serialized["bool_val"], bool)
    assert isinstance(serialized["array_val"], list)

    out_file = str(tmp_path / "test_report.json")
    save_audit_report(serialized, output_path=out_file)

    assert os.path.exists(out_file)
    with open(out_file, "r", encoding="utf-8") as f:
        loaded = json.load(f)
    assert loaded["int_val"] == 42
    assert loaded["bool_val"] is True
