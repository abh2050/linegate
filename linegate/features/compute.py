"""Materialize the baseline feature set for a split and load it as a float32 matrix.

Numeric, date, and categorical features are computed in three single-table
scans, written to narrow Parquet files, then joined on Id within the same
split. Route
codes are fitted on the train split only. The matrix never contains Id,
Response, the raw route string, or absolute start time; start time is
returned separately for the early-stopping slice only.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import duckdb
import numpy as np

from linegate.features.baseline import (categorical_sql, categorical_stations, date_sql, numeric_sql, route_expr,
                                       station_columns)

FEATURE_SET = "baseline_v2"
ROUTE_MIN_COUNT = 50
KEY_COLUMNS = ("Id", "Response", "route", "start_ts")


@dataclass(frozen=True)
class Matrix:
    ids: np.ndarray
    y: np.ndarray
    X: np.ndarray
    names: list[str]
    start: np.ndarray


def copy_to(con: duckdb.DuckDBPyConnection, sql: str, path: Path) -> Path:
    con.execute(f"COPY ({sql}) TO '{path}' (FORMAT parquet, COMPRESSION zstd)")
    return path


def materialize(con: duckdb.DuckDBPyConnection, split: str, out_dir: Path, categorical: list[str]) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    groups = station_columns(con, split)
    num = copy_to(con, numeric_sql(groups, split), out_dir / f"_numeric_{split}.parquet")
    dat = copy_to(con, date_sql(groups, split), out_dir / f"_date_{split}.parquet")
    cat = copy_to(con, categorical_sql(categorical_stations(con, split), categorical, split),
                  out_dir / f"_categorical_{split}.parquet")
    joined = (f"SELECT n.*, d.* EXCLUDE (Id), c.* EXCLUDE (Id), {route_expr(groups)} "
              f"FROM read_parquet('{num}') n JOIN read_parquet('{dat}') d USING (Id) "
              f"JOIN read_parquet('{cat}') c USING (Id)")
    path = copy_to(con, joined, out_dir / f"{FEATURE_SET}_{split}.parquet")
    for part in (num, dat, cat):
        part.unlink()
    return path


def fit_route_codes(con: duckdb.DuckDBPyConnection, train_path: Path, min_count: int = ROUTE_MIN_COUNT) -> None:
    con.execute(
        "CREATE OR REPLACE TEMP TABLE route_codes AS "
        "SELECT route, row_number() OVER (ORDER BY n DESC, route) AS route_code FROM "
        f"(SELECT route, count(*) AS n FROM read_parquet('{train_path}') GROUP BY route) "
        f"WHERE n >= {int(min_count)}"
    )


def to_float32(column) -> np.ndarray:
    return np.ma.filled(np.ma.asarray(column).astype(np.float32), np.nan)


def load_matrix(con: duckdb.DuckDBPyConnection, path: Path) -> Matrix:
    cur = con.execute(
        f"SELECT b.*, coalesce(r.route_code, 0) AS route_code FROM read_parquet('{path}') b "
        "LEFT JOIN route_codes r USING (route) ORDER BY b.Id"
    )
    names = [d[0] for d in cur.description]
    data = cur.fetchnumpy()
    features = [n for n in names if n not in KEY_COLUMNS]
    X = np.empty((len(data["Id"]), len(features)), dtype=np.float32)
    for j, name in enumerate(features):
        X[:, j] = to_float32(data.pop(name))
    start = np.ma.filled(np.ma.asarray(data["start_ts"]).astype(np.float64), np.nan)
    return Matrix(np.asarray(data["Id"]), np.asarray(data["Response"]).astype(np.int8), X, features, start)
