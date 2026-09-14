"""Build and open a read-only DuckDB catalog over the train and validation splits.

The catalog registers one view per split and column kind. It registers no
holdout view and no Kaggle test view, so no query through this catalog can
reach holdout rows or join train to test.
"""

from __future__ import annotations

import sys
from pathlib import Path

import duckdb

from linegate.dataio import PARQUET_DIR
from linegate.dataio.resources import configure
from linegate.dataio.schema import KINDS
from linegate.dataio.splits import SPLITS_NAME

CATALOG_NAME = "catalog.duckdb"
EXPOSED_SPLITS = ("train", "validation")


def view_sql(parquet_dir: Path, split: str, kind: str) -> str:
    source = parquet_dir / f"train_{kind}.parquet"
    splits = parquet_dir / SPLITS_NAME
    return (
        f"CREATE VIEW {split}_{kind} AS SELECT t.* FROM read_parquet('{source}') t "
        f"SEMI JOIN (SELECT Id FROM read_parquet('{splits}') WHERE split = '{split}') s USING (Id)"
    )


def build_catalog(parquet_dir: Path = PARQUET_DIR) -> Path:
    catalog = parquet_dir / CATALOG_NAME
    catalog.unlink(missing_ok=True)
    with duckdb.connect(str(catalog)) as con:
        for split in EXPOSED_SPLITS:
            for kind in KINDS:
                con.execute(view_sql(parquet_dir, split, kind))
    return catalog


def connect(parquet_dir: Path = PARQUET_DIR) -> duckdb.DuckDBPyConnection:
    catalog = parquet_dir / CATALOG_NAME
    if not catalog.exists():
        raise FileNotFoundError(f"{catalog} missing. Run `make gate-0`.")
    con = duckdb.connect(str(catalog), read_only=True)
    configure(con, parquet_dir / "_duck_tmp")
    return con


def main() -> int:
    catalog = build_catalog()
    with connect() as con:
        for (name,) in con.execute("SELECT view_name FROM duckdb_views() WHERE NOT internal ORDER BY 1").fetchall():
            print(f"view {name}")
    print(f"catalog {catalog}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
