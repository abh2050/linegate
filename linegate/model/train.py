"""Train the fixed-config LightGBM baseline on train, evaluate on validation (Gate 1).

Reads only the train and validation views of the catalog. Writes metrics,
the model, and validation scores to data/artifacts/baseline/. Metrics are
written before the MCC tripwire runs, so a failed gate still leaves results
to inspect. Never touches the holdout.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import duckdb
import lightgbm as lgb
import numpy as np
import yaml

from linegate.dataio import CONFIG_DIR, DATA_DIR, PARQUET_DIR
from linegate.dataio.duck import CATALOG_NAME
from linegate.dataio.resources import configure, worker_threads
from linegate.features import compute
from linegate.model import evaluate, guards

ARTIFACT_DIR = DATA_DIR / "artifacts" / "baseline"
GATE1_MIN_MCC = 0.12
EARLY_STOP_METRIC = "auc"
START = time.monotonic()


def log(message: str) -> None:
    print(f"[{time.monotonic() - START:7.1f}s] {message}", flush=True)


def load_model_config(path: Path = CONFIG_DIR / "model.yaml") -> dict:
    cfg = yaml.safe_load(path.read_text())
    if cfg.get("algorithm") != "lightgbm" or cfg.get("objective") != "binary":
        raise ValueError("the classifier is lightgbm with a binary objective")
    return cfg


def scale_pos_weight(setting, y: np.ndarray) -> float:
    if setting != "auto":
        return float(setting)
    positives = int(np.sum(y == 1))
    return (len(y) - positives) / positives


def lgb_params(cfg: dict, y: np.ndarray) -> dict:
    return {
        "objective": cfg["objective"], "metric": EARLY_STOP_METRIC, "num_leaves": cfg["num_leaves"],
        "learning_rate": cfg["learning_rate"], "min_child_samples": cfg["min_child_samples"],
        "scale_pos_weight": scale_pos_weight(cfg["scale_pos_weight"], y), "seed": cfg["random_state"],
        "num_threads": worker_threads(), "force_col_wise": True, "verbose": -1,
    }


def fit(cfg: dict, train: tuple, valid: tuple, names: list[str], categorical: list[str]) -> lgb.Booster:
    guards.check_feature_names(names)
    dtrain = lgb.Dataset(train[0], label=train[1], feature_name=names, categorical_feature=categorical or "auto")
    dvalid = lgb.Dataset(valid[0], label=valid[1], reference=dtrain)
    callbacks = [lgb.early_stopping(cfg["early_stopping_rounds"], verbose=False), lgb.log_evaluation(50)]
    return lgb.train(lgb_params(cfg, train[1]), dtrain, num_boost_round=cfg["n_estimators"],
                     valid_sets=[dvalid], valid_names=["validation"], callbacks=callbacks)


def catalog_connection(parquet_dir: Path = PARQUET_DIR) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    configure(con, parquet_dir / "_duck_tmp")
    con.execute(f"ATTACH '{parquet_dir / CATALOG_NAME}' AS catalog (READ_ONLY)")
    con.execute("USE catalog")
    return con


def build_matrices(con: duckdb.DuckDBPyConnection, out_dir: Path) -> tuple:
    paths = {}
    for split in ("train", "validation"):
        path = out_dir / f"baseline_{split}.parquet"
        paths[split] = path if path.exists() else compute.materialize(con, split, out_dir)
        log(f"features ready: {paths[split].name}")
    compute.fit_route_codes(con, paths["train"])
    return compute.load_matrix(con, paths["train"]), compute.load_matrix(con, paths["validation"])


def summarize(cfg, booster, names, y_val, p_val, seconds) -> dict:
    sweep = evaluate.confusion_sweep(y_val, p_val)
    threshold, mcc = evaluate.best_threshold(sweep)
    gains = booster.feature_importance("gain")
    top = sorted(zip(names, gains.tolist()), key=lambda kv: -kv[1])[:25]
    return {
        "validation_mcc": mcc, "threshold": threshold, "confusion": evaluate.confusion_at(sweep, threshold),
        "best_iteration": booster.best_iteration, "validation_auc": booster.best_score["validation"]["auc"],
        "validation_parts": int(len(y_val)), "validation_positive_rate": float(np.mean(y_val)),
        "feature_count": len(names), "train_seconds": round(seconds, 1), "top_gain_features": top,
        "model_config": cfg, "holdout_touched": False,
    }


def run(out_dir: Path = ARTIFACT_DIR) -> dict:
    cfg = load_model_config()
    with catalog_connection() as con:
        (_, y_tr, X_tr, names), (id_va, y_va, X_va, _) = build_matrices(con, out_dir)
    log(f"train {X_tr.shape}, validation {X_va.shape}, positives {int(y_tr.sum())}/{int(y_va.sum())}")
    started = time.monotonic()
    booster = fit(cfg, (X_tr, y_tr), (X_va, y_va), names, categorical=["route_code"])
    p_va = booster.predict(X_va, num_iteration=booster.best_iteration)
    metrics = summarize(cfg, booster, names, y_va, p_va, time.monotonic() - started)
    booster.save_model(str(out_dir / "model.txt"), num_iteration=booster.best_iteration)
    np.savez_compressed(out_dir / "validation_scores.npz", id=id_va, y=y_va, p=p_va)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    return metrics


def main() -> int:
    metrics = run()
    log(f"validation MCC {metrics['validation_mcc']:.4f} at threshold {metrics['threshold']}, "
        f"AUC {metrics['validation_auc']:.4f}, best iteration {metrics['best_iteration']}")
    try:
        guards.check_validation_mcc(metrics["validation_mcc"], GATE1_MIN_MCC, load_model_config()["leak_tripwire_mcc"])
    except (guards.LeakageSuspected, guards.PipelineBroken) as exc:
        log(f"FAIL: {type(exc).__name__}: {exc}")
        return 1
    log("gate-1 PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
