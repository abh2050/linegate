"""Materialize the baseline feature set for a split and load it as a float32 matrix.

Numeric and date aggregates are computed in two single-table scans, written
to narrow Parquet files, then joined on Id within the same split. Route
codes are fitted on the train split only. The matrix never contains Id,
Response, or the raw route string.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import numpy as np

from linegate.features.baseline import date_sql, numeric_sql, route_expr, station_columns

ROUTE_MIN_COUNT = 50
KEY_COLUMNS = ("Id", "Response", "route")


def copy_to(con: duckdb.DuckDBPyConnection, sql: str, path: Path) -> Path:
    con.execute(f"COPY ({sql}) TO '{path}' (FORMAT parquet, COMPRESSION zstd)")
    return path


def materialize(con: duckdb.DuckDBPyConnection, split: str, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    groups = station_columns(con, split)
    num = copy_to(con, numeric_sql(groups, split), out_dir / f"_numeric_{split}.parquet")
    dat = copy_to(con, date_sql(groups, split), out_dir / f"_date_{split}.parquet")
    joined = (f"SELECT n.*, d.* EXCLUDE (Id), {route_expr(groups)} "
              f"FROM read_parquet('{num}') n JOIN read_parquet('{dat}') d USING (Id)")
    path = copy_to(con, joined, out_dir / f"baseline_{split}.parquet")
    num.unlink()
    dat.unlink()
    return path


def fit_route_codes(con: duckdb.DuckDBPyConnection, train_path: Path, min_count: int = ROUTE_MIN_COUNT) -> None:
    con.execute(
        "CREATE OR REPLACE TEMP TABLE route_codes AS "
        "SELECT route, row_number() OVER (ORDER BY n DESC, route) AS route_code FROM "
        f"(SELECT route, count(*) AS n FROM read_parquet('{train_path}') GROUP BY route) "
        f"WHERE n >= {int(min_count)}"
    )


def load_matrix(con: duckdb.DuckDBPyConnection, path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    cur = con.execute(
        f"SELECT b.*, coalesce(r.route_code, 0) AS route_code FROM read_parquet('{path}') b "
        "LEFT JOIN route_codes r USING (route) ORDER BY b.Id"
    )
    names = [d[0] for d in cur.description]
    data = cur.fetchnumpy()
    features = [n for n in names if n not in KEY_COLUMNS]
    X = np.empty((len(data["Id"]), len(features)), dtype=np.float32)
    for j, name in enumerate(features):
        X[:, j] = np.ma.filled(np.ma.asarray(data.pop(name)).astype(np.float32), np.nan)
    return np.asarray(data["Id"]), np.asarray(data["Response"]).astype(np.int8), X, features
