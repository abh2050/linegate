"""Run feature SQL against exactly one split, through neutral view names.

The sandbox exposes parts_numeric, parts_date, and parts_categorical for one
split (train or validation), with Response removed. Three modes bind part
Ids: plain (real Ids), shuffled (a seeded permutation of the split's Ids),
and subset (a seeded fraction of the split's parts). Results are mapped back
to real Ids. SQL that names files, catalogs, other splits, or metadata, or
that writes anything, is rejected before it runs.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import duckdb
import numpy as np

from linegate.dataio import PARQUET_DIR
from linegate.dataio.resources import configure
from linegate.dataio.splits import SPLITS_NAME

SPLITS = ("train", "validation")
MODES = ("plain", "shuffled", "subset")
VIEWS = {"parts_numeric": "numeric", "parts_date": "date", "parts_categorical": "categorical"}
FORBIDDEN = re.compile(
    r"\b(read_\w+|parquet_\w+|glob|sniff_csv|query_table|query|attach|detach|copy|export|import|install|load|"
    r"pragma|set|reset|create|insert|update|delete|drop|alter|truncate|vacuum|checkpoint|call|use|"
    r"catalog|id_map|duckdb_\w+|information_schema|pg_\w+|sqlite_\w+|"
    r"(train|validation|holdout|test)_(numeric|date|categorical)|holdout\w*)\b",
    re.IGNORECASE,
)


class FeatureSQLError(ValueError):
    """Raised when feature SQL is unsafe or does not return one row per part."""


@dataclass
class FeatureFrame:
    ids: np.ndarray
    columns: dict[str, np.ndarray]


@dataclass
class Sandbox:
    con: duckdb.DuckDBPyConnection
    split: str
    mode: str

    def __enter__(self) -> "Sandbox":
        return self

    def __exit__(self, *exc) -> None:
        self.con.close()


def check_feature_sql(sql: str) -> None:
    body = sql.strip().rstrip(";").strip()
    if ";" in body:
        raise FeatureSQLError("one statement only")
    if not re.match(r"^(select|with)\b", body, re.IGNORECASE):
        raise FeatureSQLError("feature SQL must be a SELECT")
    match = FORBIDDEN.search(body)
    if match:
        raise FeatureSQLError(f"forbidden token in feature SQL: {match.group(0)!r}")


def id_map_sql(split: str, mode: str, seed: int, fraction: float, splits_path: Path) -> str:
    ids = f"SELECT Id FROM read_parquet('{splits_path}') WHERE split = '{split}'"
    if mode == "plain":
        return f"SELECT Id, Id AS bound_id FROM ({ids})"
    if mode == "subset":
        return f"SELECT Id, Id AS bound_id FROM ({ids}) WHERE hash(Id, {seed}) % 1000 < {int(fraction * 1000)}"
    return (f"WITH a AS (SELECT Id, row_number() OVER (ORDER BY Id) AS r FROM ({ids})), "
            f"b AS (SELECT Id AS bound_id, row_number() OVER (ORDER BY hash(Id, {seed}), Id) AS r FROM ({ids})) "
            "SELECT a.Id, b.bound_id FROM a JOIN b USING (r)")


def open_sandbox(split: str, mode: str, seed: int = 17, fraction: float = 0.5,
                 parquet_dir: Path = PARQUET_DIR) -> Sandbox:
    if split not in SPLITS or mode not in MODES:
        raise ValueError(f"split must be one of {SPLITS}, mode one of {MODES}")
    con = duckdb.connect()
    configure(con, parquet_dir / "_duck_tmp")
    con.execute(f"CREATE TABLE id_map AS {id_map_sql(split, mode, seed, fraction, parquet_dir / SPLITS_NAME)}")
    for view, kind in VIEWS.items():
        drop = "Id, Response" if kind == "numeric" else "Id"
        con.execute(f"CREATE VIEW {view} AS SELECT m.bound_id AS Id, t.* EXCLUDE ({drop}) "
                    f"FROM read_parquet('{parquet_dir / f'train_{kind}.parquet'}') t JOIN id_map m USING (Id)")
    return Sandbox(con, split, mode)


def column_array(values) -> np.ndarray:
    if isinstance(values, np.ma.MaskedArray) and values.dtype.kind in "fiub":
        return np.ma.filled(values.astype(np.float64), np.nan)
    if isinstance(values, np.ma.MaskedArray):
        return np.array([None if m else v for v, m in zip(values.data, np.ma.getmaskarray(values))], dtype=object)
    return values.astype(np.float64) if values.dtype.kind in "fiub" else values.astype(object)


def compute_feature(sb: Sandbox, sql: str) -> FeatureFrame:
    check_feature_sql(sql)
    try:
        cur = sb.con.execute(f"SELECT m.Id AS __part, f.* FROM ({sql}) f JOIN id_map m ON f.Id = m.bound_id ORDER BY __part")
    except duckdb.BinderException as exc:
        raise FeatureSQLError(f"feature SQL must return an Id column and valid columns: {exc}") from exc
    data = cur.fetchnumpy()
    ids = np.asarray(data.pop("__part"))
    data.pop("Id")
    expected = sb.con.execute("SELECT count(*) FROM id_map").fetchone()[0]
    if len(ids) != expected or len(np.unique(ids)) != len(ids):
        raise FeatureSQLError(f"feature must return one row per part: got {len(ids)} rows for {expected} parts")
    if not data:
        raise FeatureSQLError("feature SQL returned no feature columns")
    return FeatureFrame(ids, {name: column_array(values) for name, values in data.items()})


def load_labels(split: str, parquet_dir: Path = PARQUET_DIR) -> tuple[np.ndarray, np.ndarray]:
    """Trusted path: labels for one split, ordered by Id. Never exposed to feature SQL."""
    if split not in SPLITS:
        raise ValueError(f"labels are available for {SPLITS} only")
    rows = duckdb.sql(
        f"SELECT t.Id, t.Response FROM read_parquet('{parquet_dir / 'train_numeric.parquet'}') t "
        f"SEMI JOIN (SELECT Id FROM read_parquet('{parquet_dir / SPLITS_NAME}') WHERE split = '{split}') s USING (Id) "
        "ORDER BY t.Id").fetchnumpy()
    return np.asarray(rows["Id"]), np.asarray(rows["Response"]).astype(np.int8)
