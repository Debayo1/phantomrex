"""
Test Suite for Phase 7 Strategy Research Harness (E6 Requirements)
==================================================================
Tests:
  1. Runner cannot write live model files or send orders.
  2. Lockbox rows are never used in dev statistics (zero index overlap, with purge).
  3. Registry count increases for every variant run, including failed ones.
  4. variants.json contains at most 8 pre-registered variants.
"""

import os
import ast
import csv
import pytest
import numpy as np

from research.run_research import (
    assert_safe_environment,
    split_dev_lockbox,
    log_to_registry,
    get_registry_count,
    REGISTRY_FILE,
    VARIANTS_FILE
)


def test_runner_cannot_write_models_or_send_orders():
    """E6.1: Verify the runner cannot write model files or send live orders."""
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
            # Check open(filepath, 'w'/'wb') calls
            if (isinstance(node.func, ast.Name) and node.func.id == "open") or \
               (isinstance(node.func, ast.Attribute) and node.func.attr in ("dump", "save")):
                for arg in node.args:
                    if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                        for bad in forbidden_files:
                            assert bad not in arg.value, f"Forbidden file target in call: {arg.value}"

    # 3. Verify assert_safe_environment blocks unsafe modules safely without polluting sys.modules
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


def test_lockbox_purge_and_no_index_overlap():
    """E6.2: Verify lockbox rows are never used in dev statistics (no index overlap, with purge)."""
    # Create 100 synthetic setups
    n = 100
    bars = np.arange(0, n * 5, 5)
    # Give some setups longer exits that would bleed across boundaries
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
    """E6.3: Verify registry count increases for every variant run, including failed ones."""
    test_reg_file = str(tmp_path / "test_registry.csv")
    monkeypatch.setattr("research.run_research.REGISTRY_FILE", test_reg_file)

    # Initially empty / 0
    assert get_registry_count() == 0

    # Log run 1 (failed)
    log_to_registry("V0_TEST", "params=1", "net=-0.5R", "FAIL (no edge without AI)")
    assert get_registry_count() == 1

    # Log run 2 (failed due to error)
    log_to_registry("V1_TEST", "params=2", "net=-0.2R", "FAIL")
    assert get_registry_count() == 2

    # Log run 3 (passed)
    log_to_registry("V2_TEST", "params=3", "net=+0.3R", "PASS")
    assert get_registry_count() == 3

    # Verify rows in CSV
    with open(test_reg_file, "r", encoding="utf-8") as f:
        reader = list(csv.reader(f))
        assert len(reader) == 4  # Header + 3 rows
        assert reader[1][4] == "FAIL (no edge without AI)"
        assert reader[2][4] == "FAIL"
        assert reader[3][4] == "PASS"


def test_variants_json_max_eight_pre_registered():
    """Verify variants.json exists, has at most 8 variants, and defines V0-V7."""
    assert os.path.exists(VARIANTS_FILE)
    import json
    with open(VARIANTS_FILE, "r", encoding="utf-8") as f:
        variants = json.load(f)

    assert len(variants) <= 8, f"Too many variants! Found {len(variants)}, max allowed is 8."
    assert set(variants.keys()) == {"V0", "V1", "V2", "V3", "V4", "V5", "V6", "V7"}
