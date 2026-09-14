"""Seal the holdout split by writing its Id hash to data/holdout.lock, once.

A second run with the same holdout is a no-op. A second run with a different
holdout fails. The holdout run counter starts at 0 and only the Gate 7
scorecard increments it.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import duckdb

from linegate.dataio import DATA_DIR, PARQUET_DIR, ROOT
from linegate.dataio.splits import SPLITS_NAME, holdout_digest, load_config

RUNS_PATH = DATA_DIR / "holdout_runs.json"


class SealError(RuntimeError):
    """Raised when an existing seal does not match the current holdout."""


def holdout_ids(splits_path: Path) -> list[int]:
    query = f"SELECT Id FROM read_parquet('{splits_path}') WHERE split = 'holdout'"
    return [row[0] for row in duckdb.sql(query).fetchall()]


def seal(splits_path: Path, lock_path: Path, runs_path: Path) -> tuple[str, bool]:
    ids = holdout_ids(splits_path)
    lock = {"sha256": holdout_digest(ids), "count": len(ids)}
    if lock_path.exists():
        existing = json.loads(lock_path.read_text())
        if existing != lock:
            raise SealError(f"{lock_path} holds {existing}, current holdout is {lock}")
        return lock["sha256"], False
    lock_path.write_text(json.dumps(lock, indent=2) + "\n")
    if not runs_path.exists():
        runs_path.write_text(json.dumps({"holdout_runs": 0}, indent=2) + "\n")
    return lock["sha256"], True


def main() -> int:
    lock_path = ROOT / load_config().seal_path
    try:
        digest, created = seal(PARQUET_DIR / SPLITS_NAME, lock_path, RUNS_PATH)
    except SealError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    print(f"{'sealed' if created else 'already sealed, hash matches'}: {digest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
