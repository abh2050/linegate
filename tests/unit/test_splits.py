import duckdb
import pytest

from linegate.dataio import splits
from linegate.dataio.splits import SplitConfig, SplitError

CFG = SplitConfig(0.6, 0.2, 0.2, "data/holdout.lock")


def assignment(parquet_dir):
    path = parquet_dir / splits.SPLITS_NAME
    return dict(duckdb.sql(f"SELECT Id, split FROM read_parquet('{path}')").fetchall())


def test_least_ignores_nulls():
    assert duckdb.sql("SELECT least(NULL::DOUBLE, 2.0, NULL::DOUBLE)").fetchone()[0] == 2.0
    assert duckdb.sql("SELECT least(NULL::DOUBLE, NULL::DOUBLE)").fetchone()[0] is None


MIN_TS = {1: 5.0, 2: 1.0, 3: 3.0, 4: 2.0, 5: 8.0, 6: 4.0, 7: 7.0, 8: 6.0, 9: 9.0, 10: None}


def test_split_follows_min_timestamp_not_id(date_parquet):
    report = splits.write_splits(date_parquet.parent, CFG)
    result = assignment(date_parquet.parent)
    by_split = {name: [MIN_TS[i] for i, s in result.items() if s == name and MIN_TS[i] is not None]
                for name in splits.SPLIT_NAMES}
    assert max(by_split["train"]) < min(by_split["validation"])
    assert max(by_split["validation"]) < min(by_split["holdout"])
    assert max(by_split["train"]) == report["cut_train"]
    assert max(by_split["validation"]) == report["cut_validation"]
    assert result[10] == "train"
    assert report["parts_without_timestamp"] == 1
    assert sum(report["counts"].values()) == 10


def test_parts_sharing_a_timestamp_share_a_split(tmp_path):
    values = ", ".join(f"({i}, {1.0 if i <= 7 else 2.0})" for i in range(1, 11))
    path = tmp_path / "train_date.parquet"
    duckdb.sql(f'COPY (SELECT col0::BIGINT AS Id, col1::DOUBLE AS "L0_S0_D1" FROM (VALUES {values})) '
               f"TO '{path}' (FORMAT parquet)")
    with pytest.raises(SplitError, match="empty split"):
        splits.write_splits(tmp_path, CFG)


def test_digest_is_order_independent():
    assert splits.holdout_digest([3, 1, 2]) == splits.holdout_digest([1, 2, 3])
    assert splits.holdout_digest([1, 2]) != splits.holdout_digest([1, 2, 3])


def test_config_rejects_bad_fractions(tmp_path):
    path = tmp_path / "split.yaml"
    path.write_text("strategy: time_ordered\nkey: min_timestamp_per_part\ntrain_fraction: 0.7\n"
                    "validation_fraction: 0.2\nholdout_fraction: 0.2\nseal_path: x\n")
    with pytest.raises(SplitError, match="sum to 1"):
        splits.load_config(path)


def test_config_rejects_random_strategy(tmp_path):
    path = tmp_path / "split.yaml"
    path.write_text("strategy: random\nkey: min_timestamp_per_part\n")
    with pytest.raises(SplitError, match="time_ordered"):
        splits.load_config(path)


def test_repo_config_loads():
    assert splits.load_config() == CFG
