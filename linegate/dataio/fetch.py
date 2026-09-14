"""Download the Bosch competition files and verify them against data/manifest.json.

Every raw file load in the codebase goes through `verify_file`. There is no
fallback: a missing manifest, a missing entry, or a hash mismatch raises.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import zipfile
from pathlib import Path

from linegate.dataio import DATA_DIR, MANIFEST_PATH, RAW_DIR

COMPETITION = "bosch-production-line-performance"
BUNDLE_PATH = DATA_DIR / f"{COMPETITION}.zip"
RAW_FILES = (
    "train_numeric.csv.zip",
    "train_date.csv.zip",
    "train_categorical.csv.zip",
    "test_numeric.csv.zip",
    "test_date.csv.zip",
    "test_categorical.csv.zip",
)


class ManifestError(RuntimeError):
    """Raised when a raw file cannot be proven to match the manifest."""


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def load_manifest(path: Path = MANIFEST_PATH) -> dict[str, str]:
    if not path.exists():
        raise ManifestError(f"{path} does not exist. Run `make manifest` after `make data`.")
    manifest = json.loads(path.read_text())
    missing = [name for name in RAW_FILES if name not in manifest]
    if missing:
        raise ManifestError(f"{path} has no entry for {missing}")
    return manifest


def verify_file(path: Path, manifest: dict[str, str]) -> Path:
    expected = manifest.get(path.name)
    if expected is None:
        raise ManifestError(f"{path.name} is not listed in the manifest")
    if not path.exists():
        raise ManifestError(f"{path} is missing. Run `make data`.")
    actual = sha256_file(path)
    if actual != expected:
        raise ManifestError(f"{path.name}: sha256 {actual} != manifest {expected}")
    return path


def verify_all(raw_dir: Path = RAW_DIR, manifest_path: Path = MANIFEST_PATH) -> list[Path]:
    manifest = load_manifest(manifest_path)
    return [verify_file(raw_dir / name, manifest) for name in RAW_FILES]


def download(raw_dir: Path = RAW_DIR) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    for name in RAW_FILES:
        if (raw_dir / name).exists():
            continue
        cmd = ["kaggle", "competitions", "download", "-c", COMPETITION, "-f", name, "-p", str(raw_dir)]
        subprocess.run(cmd, check=True)


def unpack(bundle: Path = BUNDLE_PATH, raw_dir: Path = RAW_DIR) -> list[Path]:
    """Copy the six data zips out of the full-competition bundle, byte for byte."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(bundle) as archive:
        names = set(archive.namelist())
        missing = [n for n in RAW_FILES if n not in names]
        if missing:
            raise ManifestError(f"{bundle.name} lacks {missing}")
        for name in RAW_FILES:
            if not (raw_dir / name).exists():
                archive.extract(name, raw_dir)
    return [raw_dir / name for name in RAW_FILES]


def write_manifest(raw_dir: Path = RAW_DIR, manifest_path: Path = MANIFEST_PATH) -> dict[str, str]:
    if manifest_path.exists():
        raise ManifestError(f"{manifest_path} already exists and is never overwritten")
    hashes = {name: sha256_file(raw_dir / name) for name in RAW_FILES}
    manifest_path.write_text(json.dumps(hashes, indent=2, sort_keys=True) + "\n")
    return hashes


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="linegate.dataio.fetch")
    parser.add_argument("command", choices=["download", "unpack", "verify", "write-manifest"])
    args = parser.parse_args(argv)
    try:
        if args.command == "download":
            download()
        elif args.command == "unpack":
            for path in unpack():
                print(f"unpacked {path.name}")
        elif args.command == "write-manifest":
            for name, digest in write_manifest().items():
                print(f"{digest}  {name}")
        else:
            for path in verify_all():
                print(f"verified {path.name}")
    except ManifestError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
