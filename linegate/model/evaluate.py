"""Exact confusion matrix and MCC at every threshold of a sweep.

A part is predicted positive when its score is >= the threshold. Counts come
from sorted scores, so each threshold gets its own exact confusion matrix.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

SWEEP_THRESHOLDS = np.round(np.linspace(0.0, 1.0, 1001), 3)


@dataclass(frozen=True)
class Sweep:
    thresholds: np.ndarray
    tp: np.ndarray
    fp: np.ndarray
    tn: np.ndarray
    fn: np.ndarray
    mcc: np.ndarray


def mcc_from_counts(tp, fp, tn, fn) -> np.ndarray:
    tp, fp, tn, fn = (np.asarray(a, dtype=np.float64) for a in (tp, fp, tn, fn))
    denom = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(denom > 0, (tp * tn - fp * fn) / denom, 0.0)


def confusion_sweep(y: np.ndarray, p: np.ndarray, thresholds: np.ndarray = SWEEP_THRESHOLDS) -> Sweep:
    y = np.asarray(y).astype(bool)
    pos, neg = np.sort(p[y]), np.sort(p[~y])
    tp = len(pos) - np.searchsorted(pos, thresholds, side="left")
    fp = len(neg) - np.searchsorted(neg, thresholds, side="left")
    tn, fn = len(neg) - fp, len(pos) - tp
    return Sweep(thresholds, tp, fp, tn, fn, mcc_from_counts(tp, fp, tn, fn))


def best_threshold(sweep: Sweep) -> tuple[float, float]:
    i = int(np.argmax(sweep.mcc))
    return float(sweep.thresholds[i]), float(sweep.mcc[i])


def confusion_at(sweep: Sweep, threshold: float) -> dict[str, int]:
    i = int(np.argmin(np.abs(sweep.thresholds - threshold)))
    return {k: int(getattr(sweep, k)[i]) for k in ("tp", "fp", "tn", "fn")}


def roc_auc(y: np.ndarray, p: np.ndarray) -> float:
    ranks = np.empty(len(p))
    order = np.argsort(p, kind="mergesort")
    sorted_p = p[order]
    _, first, counts = np.unique(sorted_p, return_index=True, return_counts=True)
    ranks[order] = np.repeat(first + (counts + 1) / 2.0, counts)
    pos = np.asarray(y) == 1
    n_pos, n_neg = pos.sum(), (~pos).sum()
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def bootstrap_mcc(y: np.ndarray, p: np.ndarray, threshold: float, draws: int = 500, seed: int = 0) -> tuple[float, float]:
    """5th and 95th percentile of MCC at a fixed threshold over bootstrap resamples."""
    rng = np.random.default_rng(seed)
    y, predicted = np.asarray(y) == 1, np.asarray(p) >= threshold
    scores = []
    for _ in range(draws):
        i = rng.integers(0, len(y), len(y))
        yy, pp = y[i], predicted[i]
        scores.append(mcc_from_counts((pp & yy).sum(), (pp & ~yy).sum(), (~pp & ~yy).sum(), (~pp & yy).sum()))
    lo, hi = np.percentile(scores, [5, 95])
    return float(lo), float(hi)
