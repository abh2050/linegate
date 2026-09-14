import numpy as np
import pytest

from linegate.model import train


def test_repo_model_config_loads():
    cfg = train.load_model_config()
    assert cfg["algorithm"] == "lightgbm" and cfg["leak_tripwire_mcc"] == 0.40


def test_config_rejects_other_algorithms(tmp_path):
    path = tmp_path / "model.yaml"
    path.write_text("algorithm: gpt\nobjective: binary\n")
    with pytest.raises(ValueError, match="lightgbm"):
        train.load_model_config(path)


def test_auto_scale_pos_weight_is_negative_over_positive():
    assert train.scale_pos_weight("auto", np.array([0, 0, 0, 1])) == 3.0
    assert train.scale_pos_weight(2.5, np.array([0, 1])) == 2.5


def test_fit_learns_separable_signal_and_early_stops():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(4000, 3)).astype(np.float32)
    y = (X[:, 0] > 1.2).astype(np.int8)
    cfg = train.load_model_config() | {"n_estimators": 60, "min_child_samples": 20, "early_stopping_rounds": 5}
    booster = train.fit(cfg, (X[:3000], y[:3000]), (X[3000:], y[3000:]), ["a", "b", "c"], categorical=[])
    p = booster.predict(X[3000:], num_iteration=booster.best_iteration)
    assert p[y[3000:] == 1].min() > np.median(p[y[3000:] == 0])
    assert 1 <= booster.best_iteration <= 60
