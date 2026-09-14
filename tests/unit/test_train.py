import numpy as np
import pytest

from linegate.model import train


def test_repo_model_config_loads():
    cfg = train.load_model_config()
    assert cfg["algorithm"] == "lightgbm" and cfg["leak_tripwire_mcc"] == 0.40
    assert cfg["seeds"] and 0 < cfg["early_stopping_share"] < 0.5


def test_config_rejects_other_algorithms(tmp_path):
    path = tmp_path / "model.yaml"
    path.write_text("algorithm: gpt\nobjective: binary\n")
    with pytest.raises(ValueError, match="lightgbm"):
        train.load_model_config(path)


def test_auto_scale_pos_weight_is_negative_over_positive():
    assert train.scale_pos_weight("auto", np.array([0, 0, 0, 1])) == 3.0
    assert train.scale_pos_weight(1, np.array([0, 1])) == 1.0


def test_early_stopping_mask_takes_latest_parts_and_never_undated():
    start = np.array([5.0, np.nan, 1.0, 9.0, 3.0, 7.0, np.nan, 2.0, 8.0, 4.0])
    mask = train.early_stopping_mask(start, share=0.3)
    assert set(np.flatnonzero(mask)) == {3, 8, 5}  # times 9, 8, 7


def test_fit_learns_signal_and_ensemble_averages():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(6000, 3)).astype(np.float32)
    y = (X[:, 0] + 0.3 * rng.normal(size=6000) > 1.8).astype(np.int8)
    start = np.arange(6000, dtype=np.float64)
    cfg = train.load_model_config() | {"n_estimators": 80, "min_child_samples": 20,
                                       "early_stopping_rounds": 10, "feature_fraction": 1.0, "seeds": [1, 2]}
    boosters = train.fit_ensemble(cfg, X[:5000], y[:5000], start[:5000], ["a", "b", "c"], categorical=[])
    assert len(boosters) == 2 and all(1 <= b.best_iteration <= 80 for b in boosters)
    p = train.predict_ensemble(boosters, X[5000:])
    assert np.allclose(p, np.mean([b.predict(X[5000:], num_iteration=b.best_iteration) for b in boosters], axis=0))
    assert p[y[5000:] == 1].mean() > 5 * p[y[5000:] == 0].mean()
