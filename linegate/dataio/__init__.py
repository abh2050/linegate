"""Data contract: fetch, verify, convert, classify, split, and expose the Bosch set."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PARQUET_DIR = DATA_DIR / "parquet"
MANIFEST_PATH = DATA_DIR / "manifest.json"
CONFIG_DIR = ROOT / "config"
