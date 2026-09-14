"""Classify Bosch columns by kind and map them to line and station.

Column names follow L{line}_S{station}_F{n} for numeric and categorical
measurements and L{line}_S{station}_D{n} for dates. Anything else, apart from
Id and Response, is an error.
"""

from __future__ import annotations

import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import duckdb

from linegate.dataio import PARQUET_DIR

ID_COLUMN = "Id"
LABEL_COLUMN = "Response"
KINDS = ("numeric", "date", "categorical")
EXPECTED_FEATURES = {"numeric": 968, "date": 1156, "categorical": 2140}
EXPECTED_TRAIN_PARTS = 1_183_747
COLUMN_RE = re.compile(r"^L(\d+)_S(\d+)_([FD])(\d+)$")
KIND_LETTER = {"numeric": "F", "date": "D", "categorical": "F"}


class SchemaError(RuntimeError):
    """Raised when the data does not match the documented contract."""


@dataclass(frozen=True)
class ColumnRef:
    name: str
    line: str
    station: str


def parse_column(name: str, kind: str) -> ColumnRef:
    match = COLUMN_RE.match(name)
    if match is None or match.group(3) != KIND_LETTER[kind]:
        raise SchemaError(f"column {name!r} is not a valid {kind} column")
    return ColumnRef(name=name, line=f"L{match.group(1)}", station=f"S{match.group(2)}")


def feature_columns(header: list[str]) -> list[str]:
    return [c for c in header if c not in (ID_COLUMN, LABEL_COLUMN)]


def duck_type(column: str, kind: str) -> str:
    if column == ID_COLUMN:
        return "BIGINT"
    if column == LABEL_COLUMN:
        return "TINYINT"
    parse_column(column, kind)
    return "VARCHAR" if kind == "categorical" else "DOUBLE"


def station_map(columns_by_kind: dict[str, list[str]]) -> dict[str, dict[str, list[str]]]:
    stations: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    for kind, columns in columns_by_kind.items():
        for ref in (parse_column(c, kind) for c in columns):
            stations[ref.line][ref.station].append(ref.name)
    return {line: dict(by_station) for line, by_station in stations.items()}


def parquet_path(parquet_dir: Path, source: str, kind: str) -> Path:
    return parquet_dir / f"{source}_{kind}.parquet"


def parquet_columns(path: Path) -> list[str]:
    rows = duckdb.sql(f"DESCRIBE SELECT * FROM read_parquet('{path}')").fetchall()
    return [row[0] for row in rows]


def parquet_rows(path: Path) -> int:
    return duckdb.sql(f"SELECT count(*) FROM read_parquet('{path}')").fetchone()[0]


def check_kind(parquet_dir: Path, kind: str) -> int:
    train = parquet_columns(parquet_path(parquet_dir, "train", kind))
    test = parquet_columns(parquet_path(parquet_dir, "test", kind))
    features = feature_columns(train)
    for column in features:
        parse_column(column, kind)
    if len(features) != EXPECTED_FEATURES[kind]:
        raise SchemaError(f"{kind}: {len(features)} feature columns, expected {EXPECTED_FEATURES[kind]}")
    if feature_columns(test) != features:
        raise SchemaError(f"{kind}: test feature columns differ from train")
    if kind == "numeric" and LABEL_COLUMN not in train:
        raise SchemaError("train_numeric has no Response column")
    return len(features)


def check_contract(parquet_dir: Path = PARQUET_DIR) -> dict[str, int]:
    report = {kind: check_kind(parquet_dir, kind) for kind in KINDS}
    counts = {kind: parquet_rows(parquet_path(parquet_dir, "train", kind)) for kind in KINDS}
    if set(counts.values()) != {EXPECTED_TRAIN_PARTS}:
        raise SchemaError(f"train row counts {counts}, expected {EXPECTED_TRAIN_PARTS} in each")
    report["train_parts"] = EXPECTED_TRAIN_PARTS
    return report


def main() -> int:
    try:
        report = check_contract()
    except SchemaError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    for key, value in report.items():
        print(f"{key}: {value:,}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
