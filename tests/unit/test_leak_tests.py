import numpy as np
import pytest

from linegate.model import guards, leak_tests
from linegate.model.leak_tests import RefitResult, ScopeResult, ShuffleResult
from tests.unit.conftest_catalog import build_layout

CFG = {"id_shuffle_collapse_threshold": 0.60, "strict_split_min_retention": 0.50, "min_lift": 0.005,
       "row_scope_sample_fraction": 0.5}


@pytest.fixture(scope="module")
def lab(tmp_path_factory):
    return leak_tests.LeakLab(CFG, parquet_dir=build_layout(tmp_path_factory.mktemp("layout"), n=2000))


def test_collapse_and_retention_math():
    assert leak_tests.collapse(0.20, 0.02, min_lift=0.005) == pytest.approx(0.9)
    assert leak_tests.collapse(0.20, 0.25, min_lift=0.005) == 0.0
    assert leak_tests.collapse(0.001, 0.0, min_lift=0.005) == 0.0
    assert leak_tests.retention(0.05, 0.20, min_lift=0.005) == pytest.approx(0.25)
    assert leak_tests.retention(0.05, 0.001, min_lift=0.005) == 1.0


def test_lift_detects_signal_and_ignores_noise():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(6000, 1)).astype(np.float32)
    y = (x[:, 0] > 1.0).astype(np.int8)
    assert leak_tests.lift(x[:4000], y[:4000], x[4000:], y[4000:], ["x"], []) > 0.3
    noise = rng.normal(size=(6000, 1)).astype(np.float32)
    assert leak_tests.lift(noise[:4000], y[:4000], noise[4000:], y[4000:], ["x"], []) < 0.05


def test_encode_string_columns_by_train_frequency():
    train = {"route": np.array(["a", "b", "a", None, "a"], dtype=object), "x": np.array([1.0, 2, 3, 4, 5])}
    other = {"route": np.array(["b", "c", "a"], dtype=object), "x": np.array([1.0, 2, 3])}
    Xa, (Xb,), names, categorical = leak_tests.encode(train, [other], min_count=1)
    assert names == ["route", "x"] and categorical == ["route"]
    assert Xa[:, 0].tolist() == [1, 2, 1, 0, 1] and Xb[:, 0].tolist() == [2, 0, 1]


def test_row_local_feature_passes_all_three(lab):
    sql = "SELECT Id, L0_S0_F0 AS x FROM parts_numeric"
    shuffle, refit, scope = lab.id_shuffle_test(sql), lab.strict_time_refit(sql), lab.row_scope_check(sql)
    assert shuffle.lift > 0.2 and shuffle.collapse == 0.0
    assert scope.mismatched == 0 and scope.rows_compared > 0
    assert guards.warden_verdict(shuffle, refit, scope, CFG)[0] == "approve"


def test_neighbour_feature_fails_row_scope(lab):
    sql = ("SELECT Id, Id - lag(Id) OVER (ORDER BY start, Id) AS gap FROM "
           "(SELECT Id, least(*COLUMNS('^L[0-9]+_S[0-9]+_D[0-9]+$')) AS start FROM parts_date)")
    scope = lab.row_scope_check(sql)
    assert scope.mismatch_share > 0.3


def test_verdict_names_every_failed_test():
    shuffle = ShuffleResult(lift=0.2, shuffled_lift=0.01, collapse=0.95)
    refit = RefitResult(time_lift=0.02, random_lift=0.2, retention=0.1)
    scope = ScopeResult(rows_compared=100, mismatched=40, mismatch_share=0.4)
    decision, reasons = guards.warden_verdict(shuffle, refit, scope, CFG)
    assert decision == "quarantine" and len(reasons) == 3
    ok = guards.warden_verdict(ShuffleResult(lift=0.2, shuffled_lift=0.2, collapse=0.0),
                               RefitResult(time_lift=0.15, random_lift=0.2, retention=0.75),
                               ScopeResult(rows_compared=100, mismatched=0, mismatch_share=0.0), CFG)
    assert ok[0] == "approve" and ok[1]
