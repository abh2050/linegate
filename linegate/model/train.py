"""Train the LightGBM baseline on train, evaluate on validation (Gate 1).

Early stopping uses the latest share of the train split by part start time,
so validation labels never steer the number of trees. Each seed in the config
trains one booster and their scores are averaged. Reads only the train and
validation views of the catalog. Metrics are written before the MCC tripwire
runs, so a failed gate still leaves results to inspect. Never touches the
holdout.
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

from linegate.dataio import CONFIG_DIR, DATA_DIR, PARQUET_DIR, labels
from linegate.dataio.duck import CATALOG_NAME
from linegate.dataio.resources import configure, worker_threads
from linegate.features import compute
from linegate.features.baseline import select_categorical
from linegate.model import evaluate, guards

ARTIFACT_DIR = DATA_DIR / "artifacts" / "baseline"
GATE1_MIN_MCC = 0.12
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


def lgb_params(cfg: dict, y: np.ndarray, seed: int) -> dict:
    keys = ("objective", "num_leaves", "learning_rate", "min_child_samples", "feature_fraction",
            "bagging_fraction", "bagging_freq", "lambda_l2")
    return {k: cfg[k] for k in keys} | {
        "metric": cfg["early_stopping_metric"], "scale_pos_weight": scale_pos_weight(cfg["scale_pos_weight"], y),
        "seed": seed, "num_threads": worker_threads(), "force_col_wise": True, "verbose": -1,
    }


def early_stopping_mask(start: np.ndarray, share: float) -> np.ndarray:
    """The latest `share` of parts by start time. Parts without a timestamp are never in it."""
    count = int(round(share * len(start)))
    dated = np.flatnonzero(~np.isnan(start))
    latest = dated[np.argsort(start[dated], kind="stable")[::-1][:count]]
    mask = np.zeros(len(start), dtype=bool)
    mask[latest] = True
    return mask


def fit_ensemble(cfg: dict, X: np.ndarray, y: np.ndarray, start: np.ndarray,
                 names: list[str], categorical: list[str]) -> list[lgb.Booster]:
    guards.check_feature_names(names)
    stop = early_stopping_mask(start, cfg["early_stopping_share"])
    dfit = lgb.Dataset(X[~stop], label=y[~stop], feature_name=names,
                       categorical_feature=categorical or "auto", free_raw_data=False)
    dstop = lgb.Dataset(X[stop], label=y[stop], reference=dfit)
    boosters = []
    for seed in cfg["seeds"]:
        callbacks = [lgb.early_stopping(cfg["early_stopping_rounds"], verbose=False), lgb.log_evaluation(100)]
        boosters.append(lgb.train(lgb_params(cfg, y[~stop], seed), dfit, num_boost_round=cfg["n_estimators"],
                                  valid_sets=[dstop], valid_names=["train_tail"], callbacks=callbacks))
        log(f"seed {seed}: best iteration {boosters[-1].best_iteration}")
    return boosters


def predict_ensemble(boosters: list[lgb.Booster], X: np.ndarray) -> np.ndarray:
    return np.mean([b.predict(X, num_iteration=b.best_iteration) for b in boosters], axis=0)


def catalog_connection(parquet_dir: Path = PARQUET_DIR) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    configure(con, parquet_dir / "_duck_tmp")
    con.execute(f"ATTACH '{parquet_dir / CATALOG_NAME}' AS catalog (READ_ONLY)")
    con.execute("USE catalog")
    return con


def selected_categorical(con: duckdb.DuckDBPyConnection, cfg: dict, out_dir: Path) -> list[str]:
    path = out_dir / f"{compute.FEATURE_SET}_categorical_columns.json"
    if not path.exists():
        parts = con.execute("SELECT count(*) FROM train_categorical").fetchone()[0]
        columns = select_categorical(con, min_count=max(1, int(cfg["categorical_min_share"] * parts)))
        out_dir.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(columns, indent=2) + "\n")
    return json.loads(path.read_text())


def build_matrices(con: duckdb.DuckDBPyConnection, cfg: dict, out_dir: Path) -> tuple:
    categorical = selected_categorical(con, cfg, out_dir)
    paths = {}
    for split in ("train", "validation"):
        path = out_dir / f"{compute.FEATURE_SET}_{split}.parquet"
        paths[split] = path if path.exists() else compute.materialize(con, split, out_dir, categorical)
        log(f"features ready: {paths[split].name} ({len(categorical)} categorical columns coded)")
    compute.fit_route_codes(con, paths["train"])
    return compute.load_matrix(con, paths["train"]), compute.load_matrix(con, paths["validation"])


def threshold_report(boosters, train: compute.Matrix, cfg: dict, y_val: np.ndarray, p_val: np.ndarray) -> dict:
    sweep = evaluate.confusion_sweep(y_val, p_val)
    threshold, mcc = evaluate.best_threshold(sweep)
    stop = early_stopping_mask(train.start, cfg["early_stopping_share"])
    tail_threshold, _ = evaluate.best_threshold(
        evaluate.confusion_sweep(train.y[stop], predict_ensemble(boosters, train.X[stop])))
    lo, hi = evaluate.bootstrap_mcc(y_val, p_val, threshold)
    return {
        "validation_mcc": mcc, "threshold": threshold, "validation_mcc_ci90": [lo, hi],
        "confusion": evaluate.confusion_at(sweep, threshold), "train_tail_threshold": tail_threshold,
        "validation_mcc_at_train_tail_threshold": float(sweep.mcc[np.argmin(np.abs(sweep.thresholds - tail_threshold))]),
    }


def summarize(cfg, boosters, train: compute.Matrix, valid: compute.Matrix, p_val, seconds) -> dict:
    gains = np.sum([b.feature_importance("gain") for b in boosters], axis=0)
    top = sorted(zip(train.names, gains.tolist()), key=lambda kv: -kv[1])[:25]
    per_seed = [evaluate.best_threshold(evaluate.confusion_sweep(valid.y, b.predict(valid.X, num_iteration=b.best_iteration)))[1]
                for b in boosters]
    return threshold_report(boosters, train, cfg, valid.y, p_val) | {
        "validation_auc": evaluate.roc_auc(valid.y, p_val), "per_seed_validation_mcc": per_seed,
        "best_iterations": [b.best_iteration for b in boosters], "feature_set": compute.FEATURE_SET,
        "validation_parts": int(len(valid.y)), "validation_positive_rate": float(np.mean(valid.y)),
        "train_positive_rate": float(np.mean(train.y)), "feature_count": len(train.names),
        "train_seconds": round(seconds, 1), "top_gain_features": top, "model_config": cfg, "holdout_touched": False,
    }


def save(out_dir: Path, boosters, valid: compute.Matrix, p_val: np.ndarray, metrics: dict) -> None:
    for seed, booster in zip(metrics["model_config"]["seeds"], boosters):
        booster.save_model(str(out_dir / f"model_seed{seed}.txt"), num_iteration=booster.best_iteration)
    np.savez_compressed(out_dir / "validation_scores.npz", id=valid.ids, y=valid.y, p=p_val)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")


def run(out_dir: Path = ARTIFACT_DIR) -> dict:
    cfg = load_model_config()
    with catalog_connection() as con:
        train, valid = build_matrices(con, cfg, out_dir)
    log(f"train {train.X.shape}, validation {valid.X.shape}, positives {int(train.y.sum())}/{int(valid.y.sum())}")
    log(f"engineer-confirmed labels in the training-label store: {len(labels.load_confirmed())} (not yet merged)")
    started = time.monotonic()
    boosters = fit_ensemble(cfg, train.X, train.y, train.start, train.names, categorical=["route_code"])
    p_val = predict_ensemble(boosters, valid.X)
    metrics = summarize(cfg, boosters, train, valid, p_val, time.monotonic() - started)
    save(out_dir, boosters, valid, p_val, metrics)
    return metrics


def main() -> int:
    metrics = run()
    lo, hi = metrics["validation_mcc_ci90"]
    log(f"validation MCC {metrics['validation_mcc']:.4f} (90% CI {lo:.3f}-{hi:.3f}) at threshold {metrics['threshold']}; "
        f"at train-tail threshold {metrics['validation_mcc_at_train_tail_threshold']:.4f}; "
        f"AUC {metrics['validation_auc']:.4f}; per seed {[round(m, 4) for m in metrics['per_seed_validation_mcc']]}")
    try:
        guards.check_validation_mcc(metrics["validation_mcc"], GATE1_MIN_MCC, load_model_config()["leak_tripwire_mcc"])
    except (guards.LeakageSuspected, guards.PipelineBroken) as exc:
        log(f"FAIL: {type(exc).__name__}: {exc}")
        return 1
    log("gate-1 PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
