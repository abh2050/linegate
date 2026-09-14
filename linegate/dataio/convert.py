"""Convert verified raw CSV zips to Parquet and prove the row counts round trip.

Files are processed one at a time: verify the zip hash, extract its CSV,
write Parquet through DuckDB with explicit column types, count the CSV rows
independently, compare, and delete the extracted CSV. No row is sampled or
skipped, and a count mismatch deletes the Parquet output and raises.
"""

from __future__ import annotations

import json
import shutil
import sys
import zipfile
from pathlib import Path

import duckdb

from linegate.dataio import MANIFEST_PATH, PARQUET_DIR, RAW_DIR
from linegate.dataio.fetch import RAW_FILES, ManifestError, load_manifest, verify_file
from linegate.dataio.resources import GIB, ResourceError, configure, require_free_disk
from linegate.dataio.schema import duck_type, parquet_rows

RECORD_NAME = "conversion.json"
PARQUET_HEADROOM_BYTES = 2 * GIB
# 2,140 VARCHAR columns exhaust an 8 GB cap at full parallelism; measured on an
# M3 Pro / 18 GB: 2 threads with 10k-row groups peaks at 4.2 GB RSS.
WIDE_KIND_LIMITS = {"categorical": {"threads": 2, "row_group_size": 10_000}}
DEFAULT_ROW_GROUP_SIZE = 50_000


class ConversionError(RuntimeError):
    """Raised when a Parquet file does not reproduce its source CSV."""


def csv_member(archive: zipfile.ZipFile) -> zipfile.ZipInfo:
    members = [m for m in archive.infolist() if m.filename.endswith(".csv")]
    if len(members) != 1:
        raise ConversionError(f"expected one CSV in {archive.filename}, found {len(members)}")
    return members[0]


def count_csv_rows(zip_path: Path, chunk_size: int = 1 << 24) -> int:
    newlines, last = 0, b"\n"
    with zipfile.ZipFile(zip_path) as archive, archive.open(csv_member(archive)) as handle:
        while block := handle.read(chunk_size):
            newlines += block.count(b"\n")
            last = block[-1:]
    lines = newlines + (0 if last == b"\n" else 1)
    return lines - 1


def read_header(zip_path: Path) -> list[str]:
    with zipfile.ZipFile(zip_path) as archive, archive.open(csv_member(archive)) as handle:
        return handle.readline().decode().strip().split(",")


def kind_of(zip_path: Path) -> tuple[str, str]:
    source, kind = zip_path.name.removesuffix(".csv.zip").split("_", 1)
    return source, kind


def columns_literal(header: list[str], kind: str) -> str:
    pairs = ", ".join(f"'{c}': '{duck_type(c, kind)}'" for c in header)
    return "{" + pairs + "}"


def write_parquet(csv_path: Path, header: list[str], kind: str, out_path: Path) -> None:
    limits = WIDE_KIND_LIMITS.get(kind, {})
    row_group_size = limits.get("row_group_size", DEFAULT_ROW_GROUP_SIZE)
    query = (
        f"COPY (SELECT * FROM read_csv('{csv_path}', header = true, auto_detect = false, "
        f"columns = {columns_literal(header, kind)})) "
        f"TO '{out_path}' (FORMAT parquet, COMPRESSION zstd, ROW_GROUP_SIZE {row_group_size})"
    )
    with duckdb.connect() as con:
        configure(con, out_path.parent / "_duck_tmp")
        if "threads" in limits:
            con.execute(f"SET threads = {limits['threads']}")
        con.execute(query)


def extract_csv(zip_path: Path, work_dir: Path) -> Path:
    work_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as archive:
        member = csv_member(archive)
        require_free_disk(work_dir, member.file_size + PARQUET_HEADROOM_BYTES)
        target = work_dir / Path(member.filename).name
        with archive.open(member) as src, target.open("wb") as dst:
            shutil.copyfileobj(src, dst, length=1 << 24)
    return target


def convert_one(zip_path: Path, parquet_dir: Path, work_dir: Path) -> dict:
    source, kind = kind_of(zip_path)
    out_path = parquet_dir / f"{source}_{kind}.parquet"
    header = read_header(zip_path)
    csv_path = extract_csv(zip_path, work_dir)
    try:
        write_parquet(csv_path, header, kind, out_path)
    finally:
        csv_path.unlink(missing_ok=True)
    csv_rows, pq_rows = count_csv_rows(zip_path), parquet_rows(out_path)
    if csv_rows != pq_rows:
        out_path.unlink(missing_ok=True)
        raise ConversionError(f"{zip_path.name}: csv rows {csv_rows} != parquet rows {pq_rows}")
    return {"parquet": out_path.name, "csv_rows": csv_rows, "parquet_rows": pq_rows, "columns": len(header)}


def is_current(entry: dict | None, digest: str, parquet_dir: Path) -> bool:
    if entry is None or entry.get("sha256") != digest:
        return False
    out_path = parquet_dir / entry["parquet"]
    return out_path.exists() and parquet_rows(out_path) == entry["csv_rows"]


def convert_all(raw_dir: Path = RAW_DIR, parquet_dir: Path = PARQUET_DIR,
                manifest_path: Path = MANIFEST_PATH) -> dict[str, dict]:
    manifest = load_manifest(manifest_path)
    parquet_dir.mkdir(parents=True, exist_ok=True)
    record_path = parquet_dir / RECORD_NAME
    record = json.loads(record_path.read_text()) if record_path.exists() else {}
    for name in RAW_FILES:
        zip_path = verify_file(raw_dir / name, manifest)
        if is_current(record.get(name), manifest[name], parquet_dir):
            continue
        entry = convert_one(zip_path, parquet_dir, raw_dir / "_work")
        record[name] = {"sha256": manifest[name], **entry}
        record_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    return record


def main() -> int:
    try:
        record = convert_all()
    except (ManifestError, ConversionError, ResourceError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    for name, entry in record.items():
        print(f"{name}: {entry['csv_rows']:,} csv rows == {entry['parquet_rows']:,} parquet rows")
    return 0


if __name__ == "__main__":
    sys.exit(main())
