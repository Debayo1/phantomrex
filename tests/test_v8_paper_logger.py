import ast
import csv

import pandas as pd

from research import paper_logger_v8 as paper


def test_v8_paper_logger_never_sends_orders():
    with open(paper.__file__, "r", encoding="utf-8") as file:
        tree = ast.parse(file.read())

    assert all(
        not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "order_send"
        )
        for node in ast.walk(tree)
    )


def test_v8_first_run_only_sets_baseline(tmp_path, monkeypatch):
    log_path = tmp_path / "v8_forward_paper.csv"
    state_path = tmp_path / "v8_forward_state.json"
    monkeypatch.setattr(paper, "RUNTIME_DIR", str(tmp_path))
    monkeypatch.setattr(paper, "LOG_FILE", str(log_path))
    monkeypatch.setattr(paper, "STATE_FILE", str(state_path))

    paper._ensure_storage()

    with open(log_path, "r", newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    assert rows == []


def test_v8_open_trade_closes_conservatively_on_stop_target_tie(tmp_path, monkeypatch):
    log_path = tmp_path / "v8_forward_paper.csv"
    monkeypatch.setattr(paper, "RUNTIME_DIR", str(tmp_path))
    monkeypatch.setattr(paper, "LOG_FILE", str(log_path))
    paper._ensure_storage()
    symbol_state = {
        "open_trade": {
            "trade_id": "XAUUSDm-test",
            "entry_utc": "2026-10-27T13:00:00",
            "direction": 1,
            "entry_price": 100.0,
            "stop": 95.0,
            "target": 110.0,
            "stop_dist": 5.0,
            "cost_r": 0.05,
        }
    }
    closed_bars = pd.DataFrame([{
        "time": pd.Timestamp("2026-10-27T13:00:00"),
        "high": 111.0,
        "low": 94.0,
        "close": 100.0,
    }])

    paper._manage_open_trade(
        "XAUUSDm",
        symbol_state,
        closed_bars,
        {"point": 0.001, "spread_est": 0.25, "contract_size": 100.0},
    )

    assert symbol_state["open_trade"] is None
    with open(log_path, "r", newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    assert len(rows) == 1
    assert rows[0]["event"] == "CLOSED"
    assert rows[0]["exit_reason"] == "SL_TIE"
    assert float(rows[0]["net_r"]) < 0
