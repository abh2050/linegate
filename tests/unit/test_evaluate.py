import numpy as np

from linegate.model import evaluate


def brute_force_counts(y, p, t):
    pred = p >= t
    return (int((pred & (y == 1)).sum()), int((pred & (y == 0)).sum()),
            int((~pred & (y == 0)).sum()), int((~pred & (y == 1)).sum()))


def test_sweep_counts_match_brute_force_at_every_threshold():
    rng = np.random.default_rng(0)
    y = (rng.random(2000) < 0.05).astype(np.int8)
    p = np.round(rng.random(2000), 3)
    thresholds = np.linspace(0, 1, 101)
    sweep = evaluate.confusion_sweep(y, p, thresholds)
    for i, t in enumerate(thresholds):
        assert (sweep.tp[i], sweep.fp[i], sweep.tn[i], sweep.fn[i]) == brute_force_counts(y, p, t)


def test_mcc_perfect_and_degenerate():
    y = np.array([0, 0, 1, 1])
    sweep = evaluate.confusion_sweep(y, np.array([0.1, 0.2, 0.8, 0.9]), np.array([0.0, 0.5]))
    assert sweep.mcc[1] == 1.0
    assert sweep.mcc[0] == 0.0  # everything predicted positive: denominator is zero


def test_best_threshold_picks_max_mcc():
    y = np.array([0, 0, 0, 1, 1])
    p = np.array([0.1, 0.2, 0.6, 0.7, 0.9])
    t, mcc = evaluate.best_threshold(evaluate.confusion_sweep(y, p, np.array([0.05, 0.5, 0.65])))
    assert (t, mcc) == (0.65, 1.0)


def test_roc_auc_matches_pairwise_definition():
    rng = np.random.default_rng(3)
    y = (rng.random(300) < 0.2).astype(int)
    p = np.round(rng.random(300), 2)
    pos, neg = p[y == 1], p[y == 0]
    pairwise = ((pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()) / (len(pos) * len(neg))
    assert abs(evaluate.roc_auc(y, p) - pairwise) < 1e-12


def test_bootstrap_interval_brackets_point_estimate():
    rng = np.random.default_rng(4)
    y = (rng.random(5000) < 0.05).astype(int)
    p = np.clip(y * 0.4 + rng.random(5000) * 0.6, 0, 1)
    lo, hi = evaluate.bootstrap_mcc(y, p, threshold=0.5, draws=200)
    point = evaluate.confusion_sweep(y, p, np.array([0.5])).mcc[0]
    assert lo < point < hi
