import duckdb
import numpy as np
import pytest

from linegate.dataio.splits import SPLITS_NAME
from linegate.features import sandbox
from linegate.features.sandbox import FeatureSQLError
from tests.unit.conftest_catalog import build_layout


@pytest.fixture
def layout(tmp_path):
    return build_layout(tmp_path)


def split_ids(layout, name):
    path = layout / SPLITS_NAME
    return sorted(r[0] for r in duckdb.sql(f"SELECT Id FROM read_parquet('{path}') WHERE split = '{name}'").fetchall())


@pytest.mark.parametrize("sql", [
    "SELECT Id, 1 AS x FROM read_parquet('data/parquet/train_numeric.parquet')",
    "SELECT Id, 1 AS x FROM validation_numeric",
    "SELECT Id, 1 AS x FROM parts_numeric; DROP TABLE id_map",
    "SELECT Id, Response AS x FROM holdout_numeric",
    "DELETE FROM id_map",
    "SELECT Id, 1 AS x FROM duckdb_views()",
])
def test_forbidden_sql_rejected(sql):
    with pytest.raises(FeatureSQLError):
        sandbox.check_feature_sql(sql)


def test_plain_sandbox_sees_one_split_without_labels(layout):
    with sandbox.open_sandbox("validation", "plain", parquet_dir=layout) as sb:
        cols = [r[0] for r in sb.con.execute("DESCRIBE parts_numeric").fetchall()]
        frame = sandbox.compute_feature(sb, "SELECT Id, L0_S0_F0 AS x FROM parts_numeric")
    assert "Response" not in cols
    assert frame.ids.tolist() == split_ids(layout, "validation")


def test_shuffled_sandbox_preserves_row_local_values(layout):
    sql = "SELECT Id, L0_S0_F0 * 2 AS x FROM parts_numeric"
    with sandbox.open_sandbox("train", "plain", parquet_dir=layout) as sb:
        plain = sandbox.compute_feature(sb, sql)
    with sandbox.open_sandbox("train", "shuffled", parquet_dir=layout) as sb:
        bound = dict(sb.con.execute("SELECT Id, bound_id FROM id_map").fetchall())
        shuffled = sandbox.compute_feature(sb, sql)
    assert sum(k != v for k, v in bound.items()) > 0.9 * len(bound)
    assert sorted(bound.values()) == sorted(bound)
    assert np.array_equal(plain.ids, shuffled.ids) and np.allclose(plain.columns["x"], shuffled.columns["x"])


def test_shuffled_sandbox_changes_id_order_features(layout):
    sql = "SELECT Id, Id - lag(Id) OVER (ORDER BY Id) AS gap FROM parts_numeric"
    with sandbox.open_sandbox("train", "plain", parquet_dir=layout) as sb:
        plain = sandbox.compute_feature(sb, sql)
    with sandbox.open_sandbox("train", "shuffled", parquet_dir=layout) as sb:
        shuffled = sandbox.compute_feature(sb, sql)
    assert not np.allclose(np.nan_to_num(plain.columns["gap"]), np.nan_to_num(shuffled.columns["gap"]))


def test_subset_sandbox_keeps_a_fraction(layout):
    with sandbox.open_sandbox("train", "subset", fraction=0.5, parquet_dir=layout) as sb:
        frame = sandbox.compute_feature(sb, "SELECT Id, L0_S0_F0 AS x FROM parts_numeric")
    total = len(split_ids(layout, "train"))
    assert 0.3 * total < len(frame.ids) < 0.7 * total


@pytest.mark.parametrize("sql,match", [
    ("SELECT L0_S0_F0 AS x FROM parts_numeric", "Id"),
    ("SELECT Id, L0_S0_F0 AS x FROM parts_numeric UNION ALL SELECT Id, 0 FROM parts_numeric", "one row per part"),
    ("SELECT Id, L0_S0_F0 AS x FROM parts_numeric WHERE L0_S0_F0 > 0", "one row per part"),
])
def test_feature_shape_enforced(layout, sql, match):
    with sandbox.open_sandbox("train", "plain", parquet_dir=layout) as sb, pytest.raises(FeatureSQLError, match=match):
        sandbox.compute_feature(sb, sql)


def test_labels_align_with_frames(layout):
    ids, y = sandbox.load_labels("train", parquet_dir=layout)
    assert ids.tolist() == split_ids(layout, "train") and set(np.unique(y)) <= {0, 1}
