"""A tiny train/validation/holdout layout in the real file format, for sandbox and leak tests."""

from __future__ import annotations

from pathlib import Path

import duckdb
import numpy as np

from linegate.dataio import splits


def build_layout(tmp_path: Path, n: int = 400, seed: int = 0) -> Path:
    """n parts. Response depends on L0_S0_F0 only. Parts arrive in batches of 4 sharing a start time."""
    rng = np.random.default_rng(seed)
    ids = np.arange(1, n + 1)
    start = (ids - 1) // 4 * 1.0
    f0 = rng.normal(size=n)
    response = (f0 + 0.2 * rng.normal(size=n) > 1.2).astype(int)
    rows = ", ".join(f"({i}, {r}, {x:.6f}, {s + 0.5})" for i, r, x, s in zip(ids, response, f0, start))
    base = f"SELECT col0::BIGINT AS Id, col1::TINYINT AS Response, col2::DOUBLE AS L0_S0_F0, col3::DOUBLE AS d FROM (VALUES {rows})"
    duckdb.sql(f"COPY (SELECT Id, Response, L0_S0_F0 FROM ({base})) TO '{tmp_path / 'train_numeric.parquet'}' (FORMAT parquet)")
    duckdb.sql(f"COPY (SELECT Id, d - 0.5 AS L0_S0_D1, d AS L0_S1_D3 FROM ({base})) TO '{tmp_path / 'train_date.parquet'}' (FORMAT parquet)")
    duckdb.sql(f"COPY (SELECT Id, 'T1' AS L0_S0_F2 FROM ({base})) TO '{tmp_path / 'train_categorical.parquet'}' (FORMAT parquet)")
    splits.write_splits(tmp_path, splits.SplitConfig(0.6, 0.2, 0.2, "x"))
    return tmp_path
