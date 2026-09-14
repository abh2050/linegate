"""Assign every labelled part to train, validation, or holdout by its first timestamp.

The key is the minimum date value across all date columns of a part. Cut
points are timestamps, so every part sharing a timestamp lands in the same
split. Id and row order play no part in the assignment. Parts with no date
values go to train, which keeps them out of validation and holdout.
"""

from __future__ import annotations

import hashlib
import sys
from dataclasses import dataclass
from pathlib import Path

import duckdb
import yaml

from linegate.dataio import CONFIG_DIR, PARQUET_DIR
from linegate.dataio.resources import configure
from linegate.dataio.schema import feature_columns, parquet_columns

SPLITS_NAME = "splits.parquet"
SPLIT_NAMES = ("train", "validation", "holdout")


class SplitError(RuntimeError):
    """Raised when the split config or the resulting assignment is invalid."""


@dataclass(frozen=True)
class SplitConfig:
    train_fraction: float
    validation_fraction: float
    holdout_fraction: float
    seal_path: str


def load_config(path: Path = CONFIG_DIR / "split.yaml") -> SplitConfig:
    raw = yaml.safe_load(path.read_text())
    if raw.get("strategy") != "time_ordered" or raw.get("key") != "min_timestamp_per_part":
        raise SplitError("only strategy=time_ordered with key=min_timestamp_per_part is supported")
    cfg = SplitConfig(raw["train_fraction"], raw["validation_fraction"],
                      raw["holdout_fraction"], raw["seal_path"])
    if abs(cfg.train_fraction + cfg.validation_fraction + cfg.holdout_fraction - 1.0) > 1e-9:
        raise SplitError("split fractions must sum to 1")
    return cfg


def min_timestamp_sql(date_columns: list[str], source: str) -> str:
    quoted = ", ".join(f'"{c}"' for c in date_columns)
    return f"SELECT Id, least({quoted}) AS min_ts FROM read_parquet('{source}')"


def assign(con: duckdb.DuckDBPyConnection, date_path: Path, cfg: SplitConfig) -> tuple[float, float]:
    date_columns = feature_columns(parquet_columns(date_path))
    con.execute(f"CREATE OR REPLACE TEMP TABLE part_time AS {min_timestamp_sql(date_columns, str(date_path))}")
    upper_train = cfg.train_fraction
    upper_val = cfg.train_fraction + cfg.validation_fraction
    cut_train, cut_val = con.execute(
        "SELECT quantile_disc(min_ts, ?), quantile_disc(min_ts, ?) FROM part_time WHERE min_ts IS NOT NULL",
        [upper_train, upper_val],
    ).fetchone()
    con.execute(
        "CREATE OR REPLACE TEMP TABLE assignment AS SELECT Id, CASE "
        "WHEN min_ts IS NULL OR min_ts <= $1 THEN 'train' "
        "WHEN min_ts <= $2 THEN 'validation' ELSE 'holdout' END AS split FROM part_time",
        [cut_train, cut_val],
    )
    return cut_train, cut_val


def holdout_digest(ids: list[int]) -> str:
    payload = "\n".join(str(i) for i in sorted(ids)).encode()
    return hashlib.sha256(payload).hexdigest()


def split_counts(con: duckdb.DuckDBPyConnection) -> dict[str, int]:
    rows = con.execute("SELECT split, count(*) FROM assignment GROUP BY split").fetchall()
    counts = {name: 0 for name in SPLIT_NAMES} | dict(rows)
    if min(counts.values()) == 0:
        raise SplitError(f"empty split in {counts}")
    return counts


def write_splits(parquet_dir: Path = PARQUET_DIR, cfg: SplitConfig | None = None) -> dict:
    cfg = cfg or load_config()
    out_path = parquet_dir / SPLITS_NAME
    with duckdb.connect() as con:
        configure(con, parquet_dir / "_duck_tmp")
        cut_train, cut_val = assign(con, parquet_dir / "train_date.parquet", cfg)
        counts = split_counts(con)
        no_time = con.execute("SELECT count(*) FROM part_time WHERE min_ts IS NULL").fetchone()[0]
        con.execute(f"COPY (SELECT Id, split FROM assignment ORDER BY split, Id) TO '{out_path}' (FORMAT parquet)")
    return {"counts": counts, "cut_train": cut_train, "cut_validation": cut_val, "parts_without_timestamp": no_time}


def main() -> int:
    try:
        report = write_splits()
    except SplitError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    total = sum(report["counts"].values())
    for name, count in report["counts"].items():
        print(f"{name}: {count:,} ({count / total:.4f})")
    print(f"cut points: train <= {report['cut_train']}, validation <= {report['cut_validation']}")
    print(f"parts without any timestamp (assigned to train): {report['parts_without_timestamp']:,}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
