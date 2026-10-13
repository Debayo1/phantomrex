import os
import shutil
import tempfile
import pytest
import joblib
import numpy as np
from datetime import datetime, timedelta

import model_promotion


@pytest.fixture
def temp_model_dir():
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


def create_dummy_package(trained_date, edge_ok=True, status="PASS", expectancy=0.5,
                         holdout_ts=None, holdout_preds=None, holdout_net_r=None,
                         version="v1"):
    return {
        "model": f"model_obj_{version}",
        "validation_model": f"validation_model_{version}",
        "trained_date": trained_date,
        "edge_ok": edge_ok,
        "status": status,
        "recommended_threshold": 0.65,
        "trend_mode": "slow",
        "oos": {"expectancy_r_net": expectancy},
        "holdout_timestamps": holdout_ts or [],
        "holdout_predictions": holdout_preds or [],
        "holdout_net_r": holdout_net_r or [],
        "version": version
    }


def test_candidate_failure_preserves_live_and_prev(temp_model_dir):
    model_file = os.path.join(temp_model_dir, "test_model.pkl")
    prev_file = os.path.join(temp_model_dir, "test_model.prev.pkl")
    cand_file = os.path.join(temp_model_dir, "test_model.candidate.pkl")

    # Initial live and prev models
    live_pkg = create_dummy_package(trained_date="2026-01-01 10:00", version="live_v1")
    prev_pkg = create_dummy_package(trained_date="2025-12-01 10:00", version="prev_v0")
    joblib.dump(live_pkg, model_file)
    joblib.dump(prev_pkg, prev_file)

    live_bytes_before = open(model_file, "rb").read()
    prev_bytes_before = open(prev_file, "rb").read()

    # Candidate that fails edge_ok
    cand_pkg = create_dummy_package(trained_date="2026-02-01 10:00", edge_ok=False, status="FAIL", version="cand_fail")
    joblib.dump(cand_pkg, cand_file)

    res = model_promotion.promote_candidate(model_file)

    assert res["success"] is True
    assert res["promoted"] is False
    assert not os.path.exists(cand_file), "Candidate file must be cleaned up on rejection"

    # Live and prev files must be byte-identical to before
    assert open(model_file, "rb").read() == live_bytes_before
    assert open(prev_file, "rb").read() == prev_bytes_before


def test_two_consecutive_failed_retrains_never_touch_prev(temp_model_dir):
    model_file = os.path.join(temp_model_dir, "test_model.pkl")
    prev_file = os.path.join(temp_model_dir, "test_model.prev.pkl")
    cand_file = os.path.join(temp_model_dir, "test_model.candidate.pkl")

    live_pkg = create_dummy_package(trained_date="2026-01-01 10:00", version="live_orig")
    prev_pkg = create_dummy_package(trained_date="2025-12-01 10:00", version="prev_orig")
    joblib.dump(live_pkg, model_file)
    joblib.dump(prev_pkg, prev_file)

    prev_bytes_orig = open(prev_file, "rb").read()

    # Failed retrain 1
    cand1 = create_dummy_package(trained_date="2026-02-01 10:00", edge_ok=False, status="FAIL", version="cand1")
    joblib.dump(cand1, cand_file)
    res1 = model_promotion.promote_candidate(model_file)
    assert res1["promoted"] is False
    assert open(prev_file, "rb").read() == prev_bytes_orig

    # Failed retrain 2
    cand2 = create_dummy_package(trained_date="2026-02-02 10:00", edge_ok=False, status="INSUFFICIENT_DATA", version="cand2")
    joblib.dump(cand2, cand_file)
    res2 = model_promotion.promote_candidate(model_file)
    assert res2["promoted"] is False
    assert open(prev_file, "rb").read() == prev_bytes_orig


def test_successful_promotion_rotates_prev_and_replaces_live(temp_model_dir):
    model_file = os.path.join(temp_model_dir, "test_model.pkl")
    prev_file = os.path.join(temp_model_dir, "test_model.prev.pkl")
    cand_file = os.path.join(temp_model_dir, "test_model.candidate.pkl")

    # Initial live model
    live_pkg = create_dummy_package(trained_date="2026-01-01 10:00", version="live_v1")
    joblib.dump(live_pkg, model_file)

    # Candidate passing edge_ok (no post-train trades required if < 30)
    cand_pkg = create_dummy_package(trained_date="2026-02-01 10:00", edge_ok=True, status="PASS", version="cand_v2")
    joblib.dump(cand_pkg, cand_file)

    res = model_promotion.promote_candidate(model_file)

    assert res["success"] is True
    assert res["promoted"] is True
    assert not os.path.exists(cand_file), "Candidate file must be removed after promotion"

    # Verify live model is now cand_v2
    live_loaded = joblib.load(model_file)
    assert live_loaded["version"] == "cand_v2"

    # Verify prev model is now live_v1
    assert os.path.exists(prev_file)
    prev_loaded = joblib.load(prev_file)
    assert prev_loaded["version"] == "live_v1"


def test_rollback_model_restores_prev_atomically(temp_model_dir):
    model_file = os.path.join(temp_model_dir, "test_model.pkl")
    prev_file = os.path.join(temp_model_dir, "test_model.prev.pkl")

    # Setup: live is v2, prev is v1
    live_pkg = create_dummy_package(trained_date="2026-02-01 10:00", version="v2")
    prev_pkg = create_dummy_package(trained_date="2026-01-01 10:00", version="v1")
    joblib.dump(live_pkg, model_file)
    joblib.dump(prev_pkg, prev_file)

    res = model_promotion.rollback_model(model_file)

    assert res["success"] is True
    assert res["trained_date"] == "2026-01-01 10:00"

    # Verify live model was restored to v1
    live_loaded = joblib.load(model_file)
    assert live_loaded["version"] == "v1"

    # Rollback when no prev file exists
    non_existent = os.path.join(temp_model_dir, "no_prev.pkl")
    res_err = model_promotion.rollback_model(non_existent)
    assert res_err["success"] is False
    assert "No previous model backup found" in res_err["error"]


def test_fair_holdout_comparison_gate_logic(temp_model_dir):
    model_file = os.path.join(temp_model_dir, "test_model.pkl")

    # Scenario 1: No incumbent model exists -> immediately passes
    cand_pkg = create_dummy_package(trained_date="2026-01-01 10:00", edge_ok=True, status="PASS")
    passed, reason = model_promotion.evaluate_promotion_gate(cand_pkg, model_file)
    assert passed is True
    assert "First model initialization" in reason

    # Create incumbent model with trained_date 2026-01-15
    inc_pkg = create_dummy_package(trained_date="2026-01-15 12:00", expectancy=0.25, version="incumbent")
    joblib.dump(inc_pkg, model_file)

    # Scenario 2: Candidate edge_ok is False -> fails gate immediately
    failing_cand = create_dummy_package(trained_date="2026-02-01 10:00", edge_ok=False, status="FAIL")
    passed, reason = model_promotion.evaluate_promotion_gate(failing_cand, model_file)
    assert passed is False
    assert "failed edge validation" in reason

    # Scenario 3: Candidate has < 30 post-train holdout trades -> falls back to edge_ok (passes)
    # Generate 15 timestamps after 2026-01-15
    dt_base = datetime(2026, 1, 16)
    ts_list_15 = [(dt_base + timedelta(hours=i)).strftime("%Y-%m-%d %H:%M") for i in range(15)]
    preds_15 = [0.80] * 15
    net_r_15 = [0.5] * 15

    cand_small_slice = create_dummy_package(
        trained_date="2026-02-01 10:00",
        edge_ok=True,
        status="PASS",
        holdout_ts=ts_list_15,
        holdout_preds=preds_15,
        holdout_net_r=net_r_15
    )
    passed, reason = model_promotion.evaluate_promotion_gate(cand_small_slice, model_file)
    assert passed is True
    assert "insufficient post-train slice: 15/30 trades" in reason

    # Scenario 4: Candidate has >= 30 post-train holdout trades
    # 35 trades after 2026-01-15
    ts_list_35 = [(dt_base + timedelta(hours=i)).strftime("%Y-%m-%d %H:%M") for i in range(35)]
    preds_35 = [0.80] * 35

    # 4a: Candidate expectancy (0.35R) >= Incumbent expectancy (0.25R) -> PASS
    net_r_good = [0.35] * 35
    cand_good_slice = create_dummy_package(
        trained_date="2026-02-01 10:00",
        edge_ok=True,
        status="PASS",
        holdout_ts=ts_list_35,
        holdout_preds=preds_35,
        holdout_net_r=net_r_good
    )
    passed, reason = model_promotion.evaluate_promotion_gate(cand_good_slice, model_file)
    assert passed is True
    assert "Candidate exp: +0.350R >= Incumbent: +0.250R over 35 trades" in reason

    # 4b: Candidate expectancy (0.10R) < Incumbent expectancy (0.25R) -> REJECT
    net_r_bad = [0.10] * 35
    cand_bad_slice = create_dummy_package(
        trained_date="2026-02-01 10:00",
        edge_ok=True,
        status="PASS",
        holdout_ts=ts_list_35,
        holdout_preds=preds_35,
        holdout_net_r=net_r_bad
    )
    passed, reason = model_promotion.evaluate_promotion_gate(cand_bad_slice, model_file)
    assert passed is False
    assert "Candidate: +0.100R < Incumbent: +0.250R over 35 trades" in reason
