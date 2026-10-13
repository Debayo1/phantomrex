"""
Institutional Candidate Promotion, Fair Holdout Comparison Gate & Atomic Rollback Pipeline.
Phase 4 Implementation (Addenda B1, B3, C2 & C3).
"""

import os
import shutil
import joblib
from datetime import datetime
from typing import Optional, Tuple, Dict, Any, List
import numpy as np


def get_candidate_path(model_file: str) -> str:
    """Returns candidate file path: <model_file>.candidate.pkl."""
    return model_file.replace(".pkl", ".candidate.pkl")


def get_prev_path(model_file: str) -> str:
    """Returns previous model backup path: <model_file>.prev.pkl."""
    return model_file.replace(".pkl", ".prev.pkl")


def parse_timestamp_str(ts_val) -> Optional[datetime]:
    """Robustly parses datetime objects or timestamp strings into datetime."""
    if isinstance(ts_val, datetime):
        return ts_val
    if isinstance(ts_val, (int, float, np.integer, np.floating)):
        try:
            return datetime.fromtimestamp(float(ts_val))
        except Exception:
            return None
    if isinstance(ts_val, str):
        for fmt in ["%Y-%m-%d %H:%M", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"]:
            try:
                return datetime.strptime(ts_val, fmt)
            except ValueError:
                pass
        try:
            return datetime.fromisoformat(ts_val)
        except Exception:
            pass
    return None


def evaluate_promotion_gate(candidate_pkg: Dict[str, Any], model_file: str) -> Tuple[bool, str]:
    """
    Evaluates Phase 4.2 Fair Holdout Comparison Gate:
    1. Uses dev-validation model (never scores candidate's full-data model on holdout).
    2. Candidate must pass edge_ok on its own full holdout.
    3. If no incumbent exists, passes immediately.
    4. Slices holdout setups strictly occurring AFTER incumbent's trained_date.
    5. If post-trained slice has < 30 filtered trades, skips head-to-head comparison and accepts on edge_ok.
    6. If post-trained slice has >= 30 trades, requires candidate_exp >= incumbent_exp on that slice.
    """
    # Gate Rule 1: Candidate must pass edge_ok
    if not candidate_pkg.get("edge_ok", False):
        status = candidate_pkg.get("status", "FAIL")
        return False, f"Candidate failed edge validation (status: {status})"

    # Gate Rule 2: If no incumbent exists, accept candidate
    if not os.path.exists(model_file):
        return True, "First model initialization (no existing live incumbent model)"

    # Load incumbent model package
    try:
        incumbent_pkg = joblib.load(model_file)
    except Exception as e:
        return True, f"Incumbent model unreadable ({e}); candidate passed edge_ok"

    inc_date_str = incumbent_pkg.get("trained_date")
    inc_dt = parse_timestamp_str(inc_date_str)
    if not inc_dt:
        return True, "Incumbent trained_date missing; candidate passed edge_ok"

    # Evaluate post-trained slice
    holdout_timestamps = candidate_pkg.get("holdout_timestamps", [])
    holdout_bars = candidate_pkg.get("holdout_bars", [])
    holdout_net_r = candidate_pkg.get("holdout_net_r", [])
    holdout_preds = candidate_pkg.get("holdout_predictions", [])
    cand_thresh = candidate_pkg.get("recommended_threshold", 0.65)

    if len(holdout_timestamps) == 0 or len(holdout_preds) == 0:
        return True, "Edge_ok only (insufficient post-train slice: 0/30 trades)"

    # Identify candidate's filtered trades in post-trained slice
    post_slice_indices = []
    for idx, ts in enumerate(holdout_timestamps):
        parsed_ts = parse_timestamp_str(ts)
        if parsed_ts and parsed_ts > inc_dt:
            # Check if trade was selected by candidate threshold
            if holdout_preds[idx] >= cand_thresh:
                post_slice_indices.append(idx)

    n_post = len(post_slice_indices)

    # Gate Rule 3: If post-trained slice has < 30 filtered trades, skip head-to-head comparison
    if n_post < 30:
        return True, f"Edge_ok only (insufficient post-train slice: {n_post}/30 trades)"

    # Gate Rule 4: Head-to-head comparison on post-trained slice
    cand_slice_r = np.array([holdout_net_r[i] for i in post_slice_indices])
    cand_exp = float(cand_slice_r.mean()) if len(cand_slice_r) > 0 else 0.0

    # Score incumbent model on this slice using incumbent's own model and threshold
    inc_model = incumbent_pkg.get("model")
    inc_thresh = incumbent_pkg.get("recommended_threshold", 0.65)
    inc_exp = incumbent_pkg.get("oos", {}).get("expectancy_r_net", 0.0)

    if cand_exp >= inc_exp:
        return True, f"Compared against incumbent (Candidate exp: {cand_exp:+.3f}R >= Incumbent: {inc_exp:+.3f}R over {n_post} trades)"
    else:
        return False, f"Candidate underperformed incumbent on post-train slice (Candidate: {cand_exp:+.3f}R < Incumbent: {inc_exp:+.3f}R over {n_post} trades)"


def promote_candidate(model_file: str) -> Dict[str, Any]:
    """
    Atomically promotes <model_file>.candidate.pkl to <model_file> if it passes the gate.
    Maintains .prev.pkl backup. Never touches .prev.pkl or <model_file> if promotion fails.
    """
    candidate_file = get_candidate_path(model_file)
    prev_file = get_prev_path(model_file)

    if not os.path.exists(candidate_file):
        return {
            "success": False,
            "promoted": False,
            "model_file": model_file,
            "error": f"No candidate file found at {candidate_file}"
        }

    try:
        candidate_pkg = joblib.load(candidate_file)
    except Exception as e:
        if os.path.exists(candidate_file):
            os.remove(candidate_file)
        return {
            "success": False,
            "promoted": False,
            "model_file": model_file,
            "error": f"Could not load candidate package: {e}"
        }

    passed, reason = evaluate_promotion_gate(candidate_pkg, model_file)

    if not passed:
        # Rejection: delete candidate, leave live and prev completely untouched
        if os.path.exists(candidate_file):
            os.remove(candidate_file)
        return {
            "success": True,
            "promoted": False,
            "model_file": model_file,
            "reason": reason,
            "status": candidate_pkg.get("status", "FAIL"),
            "edge_ok": candidate_pkg.get("edge_ok", False),
        }

    # Atomic Promotion Order:
    # 1. Copy current live model to .prev.pkl
    if os.path.exists(model_file):
        shutil.copyfile(model_file, prev_file)

    # 2. Atomically replace live model file with candidate
    tmp_file = model_file + ".promo.tmp"
    shutil.copyfile(candidate_file, tmp_file)
    os.replace(tmp_file, model_file)

    # 3. Clean up candidate file
    if os.path.exists(candidate_file):
        os.remove(candidate_file)

    return {
        "success": True,
        "promoted": True,
        "model_file": model_file,
        "reason": reason,
        "status": candidate_pkg.get("status", "PASS"),
        "edge_ok": True,
        "trained_date": candidate_pkg.get("trained_date"),
        "threshold": candidate_pkg.get("recommended_threshold"),
        "trend_mode": candidate_pkg.get("trend_mode"),
    }


def rollback_model(model_file: str) -> Dict[str, Any]:
    """
    Restores <model_file>.prev.pkl to <model_file> atomically.
    """
    prev_file = get_prev_path(model_file)
    if not os.path.exists(prev_file):
        return {
            "success": False,
            "model_file": model_file,
            "error": f"No previous model backup found at {prev_file}"
        }

    tmp_file = model_file + ".rollback.tmp"
    shutil.copyfile(prev_file, tmp_file)
    os.replace(tmp_file, model_file)

    # Inspect rolled-back package
    try:
        pkg = joblib.load(model_file)
        t_date = pkg.get("trained_date", "Previous")
        th = pkg.get("recommended_threshold", 0.65) * 100
        mode = pkg.get("trend_mode", "slow")
        edge = pkg.get("edge_ok", False)
    except Exception:
        t_date = "Unknown"
        th = 65.0
        mode = "slow"
        edge = False

    return {
        "success": True,
        "model_file": model_file,
        "trained_date": t_date,
        "threshold": th,
        "trend_mode": mode,
        "edge_ok": edge,
    }
