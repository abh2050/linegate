"""Size DuckDB to the machine so heavy queries leave the laptop usable.

Memory is capped at 45 percent of physical RAM, threads at the performance
core count, and spill-to-disk goes to a capped temp directory under
data/parquet. A query that needs more than that fails instead of swapping or
filling the disk.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import duckdb

from linegate.dataio import PARQUET_DIR

MEMORY_SHARE = 0.45
TEMP_CAP_GB = 3
GIB = 1 << 30


class ResourceError(RuntimeError):
    """Raised when the machine lacks the disk a step needs."""


def physical_memory_bytes() -> int:
    return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")


def worker_threads() -> int:
    try:
        out = subprocess.run(["sysctl", "-n", "hw.perflevel0.physicalcpu"], capture_output=True, text=True, check=True)
        return max(1, int(out.stdout.strip()))
    except (OSError, ValueError, subprocess.CalledProcessError):
        return max(1, (os.cpu_count() or 2) - 2)


def configure(con: duckdb.DuckDBPyConnection, temp_dir: Path = PARQUET_DIR / "_duck_tmp") -> None:
    temp_dir.mkdir(parents=True, exist_ok=True)
    memory_gb = max(1, int(physical_memory_bytes() * MEMORY_SHARE / GIB))
    con.execute(f"SET memory_limit = '{memory_gb}GB'")
    con.execute(f"SET threads = {worker_threads()}")
    con.execute(f"SET temp_directory = '{temp_dir}'")
    con.execute(f"SET max_temp_directory_size = '{TEMP_CAP_GB}GB'")
    con.execute("SET preserve_insertion_order = false")


def require_free_disk(path: Path, needed_bytes: int) -> None:
    free = shutil.disk_usage(path).free
    if free < needed_bytes:
        raise ResourceError(f"{path}: needs {needed_bytes / GIB:.1f} GiB free, has {free / GIB:.1f} GiB")
