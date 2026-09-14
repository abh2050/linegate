"""The single holdout evaluation (Gate 7).

The exact evaluation path is rehearsed on validation first and must
reproduce the recorded validation MCC. Only then is the run counter claimed
(set to 1) and the holdout scored with the saved models, features built the
same way, and the committed policy. Results are saved; a second call returns
them and never scores the holdout again. A claimed counter without saved
results is refused rather than rerun.
"""

from __future__ import annotations

import json
from pathlib import Path

import duckdb
import lightgbm as lgb
import numpy as np

from linegate.cost.curve import CostParameters, dollars_per_shift
from linegate.cost.policy import POLICY_DIR
from linegate.dataio import DATA_DIR, PARQUET_DIR, duck
from linegate.dataio.resources import configure
from linegate.dataio.schema import KINDS
from linegate.features import compute
from linegate.model import evaluate as ev
from linegate.model.train import ARTIFACT_DIR

RUNS_PATH = DATA_DIR / "holdout_runs.json"
RESULTS_PATH = DATA_DIR / "artifacts" / "holdout" / "results.json"
LEAK_TRIPWIRE = 0.40


class HoldoutError(RuntimeError):
    """Raised when the holdout run cannot proceed safely."""


def decision_metrics(y: np.ndarray, p: np.ndarray, policy: dict, mcc_threshold: float) -> dict:
    params = CostParameters(**policy["costs"])
    at_mcc = ev.confusion_sweep(y, p, np.array([mcc_threshold]))
    at_policy = ev.confusion_sweep(y, p, np.array([policy["threshold"]]))
    ship, inspect = p < policy["band_low"], p >= policy["band_high"]
    positives = y == 1
    return {
        "parts": int(len(y)), "failures": int(positives.sum()),
        "mcc_at_validation_threshold": float(at_mcc.mcc[0]), "auc": ev.roc_auc(y, p) if positives.any() else None,
        "average_precision": ev.average_precision(y, p),
        "dollars_per_shift": float(dollars_per_shift(at_policy.tp, at_policy.fp, at_policy.tn, at_policy.fn, params)[0]),
        "ship_all_dollars_per_shift": float(dollars_per_shift(0, 0, int((~positives).sum()), int(positives.sum()), params)),
        "committed_share": float(np.mean(ship | inspect)), "abstain_share": float(np.mean(~(ship | inspect))),
        "escaped_failures": int((positives & ship).sum()), "failures_in_review": int((positives & ~(ship | inspect)).sum()),
        "failures_inspected": int((positives & inspect).sum()), "good_parts_inspected": int((~positives & inspect).sum()),
    }


def split_connection(split: str) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    configure(con, PARQUET_DIR / "_duck_tmp")
    for kind in KINDS:
        con.execute(duck.view_sql(PARQUET_DIR, split, kind))
    return con


def evaluate_split(split: str) -> dict:
    metrics = json.loads((ARTIFACT_DIR / "metrics.json").read_text())
    policy = json.loads((POLICY_DIR / "policy.json").read_text())
    categorical = json.loads((ARTIFACT_DIR / f"{compute.FEATURE_SET}_categorical_columns.json").read_text())
    with split_connection(split) as con:
        path = compute.materialize(con, split, DATA_DIR / "artifacts" / "holdout" / "features", categorical)
        compute.fit_route_codes(con, ARTIFACT_DIR / f"{compute.FEATURE_SET}_train.parquet")
        matrix = compute.load_matrix(con, path)
    boosters = [lgb.Booster(model_file=str(ARTIFACT_DIR / f"model_seed{s}.txt")) for s in metrics["model_config"]["seeds"]]
    if boosters[0].feature_name() != matrix.names:
        raise HoldoutError("feature columns differ from the trained models")
    p = np.mean([b.predict(matrix.X) for b in boosters], axis=0)
    return decision_metrics(matrix.y, p, policy, metrics["threshold"]) | {"split": split, "policy_version": policy["version"]}


def run_holdout(runs_path: Path = RUNS_PATH, results_path: Path = RESULTS_PATH, evaluate=evaluate_split,
                expected_validation_mcc: float | None = None, tolerance: float = 0.002) -> dict:
    if json.loads(runs_path.read_text())["holdout_runs"] >= 1:
        if results_path.exists():
            return json.loads(results_path.read_text())
        raise HoldoutError("the counter shows a holdout run but no results are saved; refusing to run the holdout again")
    rehearsal = evaluate("validation")
    if expected_validation_mcc is not None and abs(rehearsal["mcc_at_validation_threshold"] - expected_validation_mcc) > tolerance:
        raise HoldoutError(f"rehearsal on validation gave MCC {rehearsal['mcc_at_validation_threshold']:.4f}, "
                           f"expected {expected_validation_mcc:.4f}; the holdout was not opened")
    runs_path.write_text(json.dumps({"holdout_runs": 1}, indent=2) + "\n")
    result = evaluate("holdout")
    result["leak_tripwire"] = result["mcc_at_validation_threshold"] > LEAK_TRIPWIRE
    out = {"validation_rehearsal": rehearsal, "holdout": result}
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(out, indent=2) + "\n")
    return out
