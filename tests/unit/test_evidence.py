import duckdb
import numpy as np
import pytest

from linegate.console import evidence
from tests.unit.conftest_catalog import build_layout


@pytest.fixture
def store(tmp_path):
    layout = build_layout(tmp_path, n=400)
    val = duckdb.sql(f"SELECT Id FROM read_parquet('{layout / 'splits.parquet'}') WHERE split = 'validation' ORDER BY Id").fetchnumpy()["Id"]
    scores = np.linspace(0.0, 0.5, len(val))
    np.savez(layout / "validation_scores.npz", id=val, y=np.zeros(len(val), dtype=np.int8), p=scores)
    for split in ("train", "validation"):
        duckdb.sql(
            f"COPY (SELECT n.Id, n.Response, 'L0_S0-L0_S1' AS route, n.L0_S0_F0 AS top_a, 1.0 AS route_code "
            f"FROM read_parquet('{layout / 'train_numeric.parquet'}') n SEMI JOIN (SELECT Id FROM read_parquet('{layout / 'splits.parquet'}') "
            f"WHERE split = '{split}') s USING (Id)) TO '{layout / f'features_{split}.parquet'}' (FORMAT parquet)")
    policy = {"band_low": 0.1, "band_high": 0.4, "threshold": 0.2}
    return evidence.PartStore(layout, layout / "validation_scores.npz", policy, layout / "features_train.parquet",
                              layout / "features_validation.parquet", ["top_a"], cache_dir=tmp_path / "cache")


def test_queue_is_the_abstain_band_of_validation_only(store):
    queue = store.queue()
    assert queue and all(0.1 <= item["score"] < 0.4 for item in queue)
    assert [i["score"] for i in queue] == sorted((i["score"] for i in queue), reverse=True)
    holdout = duckdb.sql(f"SELECT Id FROM read_parquet('{store.parquet_dir / 'splits.parquet'}') WHERE split = 'holdout'").fetchnumpy()["Id"]
    assert not set(i["part_id"] for i in queue) & set(holdout.tolist())


def test_parts_outside_the_queue_are_not_found(store):
    holdout_id = int(duckdb.sql(f"SELECT min(Id) FROM read_parquet('{store.parquet_dir / 'splits.parquet'}') WHERE split = 'holdout'").fetchone()[0])
    with pytest.raises(evidence.PartNotFound):
        store.get_part(holdout_id)


def test_part_record_lists_route_and_only_out_of_range_measurements(store):
    part_id = store.queue()[0]["part_id"]
    record = store.get_part(part_id)
    assert record["part_id"] == part_id and record["route"] == "L0_S0-L0_S1"
    for m in record["out_of_range"]:
        assert m["value"] < m["low"] or m["value"] > m["high"]
    assert set(record["columns"]) >= {"L0_S0_F0", "L0_S0_D1", "L0_S1_D3"}


def test_neighbors_come_from_train_with_labels(store):
    part_id = store.queue()[0]["part_id"]
    result = store.get_neighbors(part_id, k=5)
    train_ids = set(duckdb.sql(f"SELECT Id FROM read_parquet('{store.parquet_dir / 'splits.parquet'}') WHERE split = 'train'").fetchnumpy()["Id"].tolist())
    assert len(result["neighbors"]) == 5 and all(n["part_id"] in train_ids for n in result["neighbors"])
    assert result["failed"] == sum(n["response"] for n in result["neighbors"])
