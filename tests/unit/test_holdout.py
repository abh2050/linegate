import json

import numpy as np
import pytest

from linegate.reporting import holdout


POLICY = {"threshold": 0.03, "band_low": 0.01, "band_high": 0.25,
          "costs": {"scrap_cost_usd": 42.0, "field_failure_cost_usd": 1850.0, "inspection_cost_usd": 6.5,
                    "monthly_volume": 240000, "shifts_per_month": 90, "abstain_band_indifference_usd": 1500.0}}


def test_decision_metrics_count_escapes_and_shares():
    y = np.array([1, 0, 0, 1, 1, 0, 0, 0])
    p = np.array([0.005, 0.002, 0.1, 0.3, 0.02, 0.5, 0.009, 0.2])
    m = holdout.decision_metrics(y, p, POLICY, mcc_threshold=0.25)
    assert m["escaped_failures"] == 1                     # the positive shipped below band_low
    assert m["committed_share"] == pytest.approx(5 / 8)   # 3 ship + 2 inspect
    assert m["abstain_share"] == pytest.approx(3 / 8)
    assert m["parts"] == 8 and m["failures"] == 3


def test_second_run_returns_saved_results_without_evaluating(tmp_path):
    runs, results = tmp_path / "runs.json", tmp_path / "results.json"
    runs.write_text(json.dumps({"holdout_runs": 1}))
    results.write_text(json.dumps({"holdout": {"mcc": 0.2}}))
    called = []
    out = holdout.run_holdout(runs, results, evaluate=lambda split: called.append(split))
    assert out == {"holdout": {"mcc": 0.2}} and called == []


def test_counter_without_results_refuses(tmp_path):
    runs = tmp_path / "runs.json"
    runs.write_text(json.dumps({"holdout_runs": 1}))
    with pytest.raises(holdout.HoldoutError, match="refusing"):
        holdout.run_holdout(runs, tmp_path / "results.json", evaluate=lambda split: {})


def test_failed_rehearsal_never_opens_the_holdout(tmp_path):
    runs = tmp_path / "runs.json"
    runs.write_text(json.dumps({"holdout_runs": 0}))
    seen = []

    def evaluate(split):
        seen.append(split)
        return {"mcc_at_validation_threshold": 0.10}

    with pytest.raises(holdout.HoldoutError, match="rehearsal"):
        holdout.run_holdout(runs, tmp_path / "results.json", evaluate=evaluate, expected_validation_mcc=0.2267)
    assert seen == ["validation"] and json.loads(runs.read_text()) == {"holdout_runs": 0}


def test_successful_run_claims_counter_then_saves(tmp_path):
    runs, results = tmp_path / "runs.json", tmp_path / "results.json"
    runs.write_text(json.dumps({"holdout_runs": 0}))
    order = []

    def evaluate(split):
        order.append((split, json.loads(runs.read_text())["holdout_runs"]))
        return {"mcc_at_validation_threshold": 0.2267 if split == "validation" else 0.21}

    out = holdout.run_holdout(runs, results, evaluate=evaluate, expected_validation_mcc=0.2267)
    assert order == [("validation", 0), ("holdout", 1)]
    assert json.loads(runs.read_text()) == {"holdout_runs": 1} and out["holdout"]["mcc_at_validation_threshold"] == 0.21
