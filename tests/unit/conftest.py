"""Tiny hand-built fixtures in the real Bosch file format.

These exist only to exercise the pipeline logic under pytest. Production code
never reads them: every production entry point verifies raw files against
data/manifest.json.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import duckdb
import pytest


def write_zip(directory: Path, name: str, header: list[str], rows: list[list], trailing_newline: bool = True) -> Path:
    lines = [",".join(header)] + [",".join("" if v is None else str(v) for v in row) for row in rows]
    body = "\n".join(lines) + ("\n" if trailing_newline else "")
    path = directory / name
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(name.removesuffix(".zip"), body)
    return path


def write_parquet(path: Path, select_sql: str) -> Path:
    duckdb.sql(f"COPY ({select_sql}) TO '{path}' (FORMAT parquet)")
    return path


@pytest.fixture
def date_parquet(tmp_path: Path) -> Path:
    """Ten parts. Ids deliberately do not follow time order; part 10 has no dates."""
    rows = [
        (1, 5.00, None), (2, None, 1.00), (3, 3.00, 9.00), (4, 2.00, None), (5, 8.00, 8.50),
        (6, 4.00, 4.00), (7, 7.00, None), (8, 6.00, 6.10), (9, 9.00, 9.50), (10, None, None),
    ]
    values = ", ".join(f"({i}, {'NULL' if a is None else a}, {'NULL' if b is None else b})" for i, a, b in rows)
    return write_parquet(
        tmp_path / "train_date.parquet",
        f'SELECT col0::BIGINT AS Id, col1::DOUBLE AS "L0_S0_D1", col2::DOUBLE AS "L0_S1_D3" FROM (VALUES {values})',
    )
