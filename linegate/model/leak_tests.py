"""The warden's three leak tests, run against sandboxed feature SQL.

Lift is validation AUC minus 0.5 from a small fixed LightGBM trained on the
feature alone. The Id shuffle test measures how much lift is lost when part
Ids are permuted. The strict time refit compares train-to-validation lift
with lift on a random split inside train. The row scope check recomputes the
feature on a random half of parts and counts values that changed.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import lightgbm as lgb
import numpy as np
from pydantic import BaseModel

from linegate.dataio import PARQUET_DIR
from linegate.features import sandbox
from linegate.model.evaluate import roc_auc

SEED = 17
ENCODE_MIN_COUNT = 50
LIFT_PARAMS = {"objective": "binary", "num_leaves": 15, "learning_rate": 0.1, "min_child_samples": 100,
               "seed": SEED, "deterministic": True, "num_threads": 4, "force_col_wise": True, "verbose": -1}
LIFT_ROUNDS = 150


class ShuffleResult(BaseModel):
    lift: float
    shuffled_lift: float
    collapse: float


class RefitResult(BaseModel):
    time_lift: float
    random_lift: float
    retention: float


class ScopeResult(BaseModel):
    rows_compared: int
    mismatched: int
    mismatch_share: float


def collapse(lift_value: float, shuffled: float, min_lift: float) -> float:
    if lift_value < min_lift:
        return 0.0
    return max(0.0, (lift_value - shuffled) / lift_value)


def retention(time_lift: float, random_lift: float, min_lift: float) -> float:
    if random_lift < min_lift:
        return 1.0
    return max(0.0, time_lift) / random_lift


def string_codes(train: np.ndarray, others: list[np.ndarray], min_count: int) -> tuple[np.ndarray, list[np.ndarray]]:
    counts = Counter(v for v in train if v is not None)
    ranked = [v for v, n in sorted(counts.items(), key=lambda kv: (-kv[1], str(kv[0]))) if n >= min_count]
    code = {v: i + 1 for i, v in enumerate(ranked)}
    convert = lambda arr: np.array([code.get(v, 0) for v in arr], dtype=np.float32)  # noqa: E731
    return convert(train), [convert(o) for o in others]


def encode(train: dict, others: list[dict], min_count: int = ENCODE_MIN_COUNT):
    names = sorted(train)
    categorical = [n for n in names if train[n].dtype == object]
    columns_a, columns_b = [], [[] for _ in others]
    for name in names:
        if name in categorical:
            a, bs = string_codes(train[name], [o[name] for o in others], min_count)
        else:
            a, bs = train[name].astype(np.float32), [o[name].astype(np.float32) for o in others]
        columns_a.append(a)
        for store, b in zip(columns_b, bs):
            store.append(b)
    return np.column_stack(columns_a), [np.column_stack(c) for c in columns_b], names, categorical


def lift(Xa, ya, Xb, yb, names: list[str], categorical: list[str]) -> float:
    data = lgb.Dataset(Xa, label=ya, feature_name=names, categorical_feature=categorical or "auto")
    booster = lgb.train(LIFT_PARAMS, data, num_boost_round=LIFT_ROUNDS)
    return max(0.0, roc_auc(yb, booster.predict(Xb)) - 0.5)


class LeakLab:
    """Computes frames once per (sql, split, mode) and runs the three tests."""

    def __init__(self, cfg: dict, parquet_dir: Path = PARQUET_DIR):
        self.cfg, self.parquet_dir = cfg, parquet_dir
        self._frames: dict = {}
        self._labels = {s: sandbox.load_labels(s, parquet_dir) for s in sandbox.SPLITS}

    def frame(self, sql: str, split: str, mode: str) -> sandbox.FeatureFrame:
        key = (sql, split, mode)
        if key not in self._frames:
            fraction = self.cfg["row_scope_sample_fraction"]
            with sandbox.open_sandbox(split, mode, SEED, fraction, self.parquet_dir) as sb:
                self._frames[key] = sandbox.compute_feature(sb, sql)
        return self._frames[key]

    def evict(self, sql: str, keep_plain: bool) -> None:
        """Drop cached frames for a reviewed feature; keep plain frames only if they will be evaluated."""
        for key in [k for k in self._frames if k[0] == sql and not (keep_plain and k[2] == "plain")]:
            del self._frames[key]

    def labels_for(self, frame: sandbox.FeatureFrame, split: str) -> np.ndarray:
        ids, y = self._labels[split]
        return y[np.searchsorted(ids, frame.ids)]

    def time_lift(self, sql: str, mode: str) -> float:
        tr, va = self.frame(sql, "train", mode), self.frame(sql, "validation", mode)
        Xa, (Xb,), names, categorical = encode(tr.columns, [va.columns])
        return lift(Xa, self.labels_for(tr, "train"), Xb, self.labels_for(va, "validation"), names, categorical)

    def id_shuffle_test(self, sql: str) -> ShuffleResult:
        plain, shuffled = self.time_lift(sql, "plain"), self.time_lift(sql, "shuffled")
        return ShuffleResult(lift=plain, shuffled_lift=shuffled, collapse=collapse(plain, shuffled, self.cfg["min_lift"]))

    def strict_time_refit(self, sql: str) -> RefitResult:
        tr = self.frame(sql, "train", "plain")
        y = self.labels_for(tr, "train")
        fit = np.random.default_rng(SEED).random(len(y)) < 0.8
        part_a = {k: v[fit] for k, v in tr.columns.items()}
        part_b = {k: v[~fit] for k, v in tr.columns.items()}
        Xa, (Xb,), names, categorical = encode(part_a, [part_b])
        random_lift = lift(Xa, y[fit], Xb, y[~fit], names, categorical)
        time_lift = self.time_lift(sql, "plain")
        return RefitResult(time_lift=time_lift, random_lift=random_lift,
                           retention=retention(time_lift, random_lift, self.cfg["min_lift"]))

    def row_scope_check(self, sql: str) -> ScopeResult:
        full, sub = self.frame(sql, "train", "plain"), self.frame(sql, "train", "subset")
        idx = np.searchsorted(full.ids, sub.ids)
        changed = np.zeros(len(sub.ids), dtype=bool)
        for name, values in sub.columns.items():
            changed |= ~values_equal(full.columns[name][idx], values)
        mismatched = int(changed.sum())
        return ScopeResult(rows_compared=len(sub.ids), mismatched=mismatched,
                           mismatch_share=mismatched / max(1, len(sub.ids)))


def values_equal(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if a.dtype == object or b.dtype == object:
        return np.array([x == y for x, y in zip(a, b)], dtype=bool)
    return (a == b) | (np.isnan(a) & np.isnan(b))
