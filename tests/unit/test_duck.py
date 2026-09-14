import duckdb
import pytest

from linegate.dataio import duck, splits
from tests.unit.test_splits import CFG


@pytest.fixture
def catalog_dir(date_parquet):
    parquet_dir = date_parquet.parent
    splits.write_splits(parquet_dir, CFG)
    for kind in ("numeric", "categorical"):
        duckdb.sql(f"COPY (SELECT Id FROM read_parquet('{date_parquet}')) "
                   f"TO '{parquet_dir / f'train_{kind}.parquet'}' (FORMAT parquet)")
    duck.build_catalog(parquet_dir)
    return parquet_dir


def test_catalog_exposes_only_train_and_validation(catalog_dir):
    with duck.connect(catalog_dir) as con:
        views = {r[0] for r in con.execute("SELECT view_name FROM duckdb_views() WHERE NOT internal").fetchall()}
    assert views == {f"{s}_{k}" for s in ("train", "validation") for k in ("numeric", "date", "categorical")}


def test_views_never_return_holdout_ids(catalog_dir):
    with duck.connect(catalog_dir) as con:
        seen = {r[0] for v in ("train_date", "validation_date", "train_numeric") for r in
                con.execute(f"SELECT Id FROM {v}").fetchall()}
    path = catalog_dir / splits.SPLITS_NAME
    holdout = {r[0] for r in duckdb.sql(f"SELECT Id FROM read_parquet('{path}') WHERE split = 'holdout'").fetchall()}
    assert holdout
    assert seen.isdisjoint(holdout)
    assert len(seen) == 10 - len(holdout)


def test_connection_is_read_only(catalog_dir):
    with duck.connect(catalog_dir) as con, pytest.raises(duckdb.Error, match="read-only"):
        con.execute("CREATE TABLE sneaky AS SELECT 1")


def test_connect_requires_built_catalog(tmp_path):
    with pytest.raises(FileNotFoundError):
        duck.connect(tmp_path)
