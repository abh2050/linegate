"""Append-only JSONL traces of agent runs. Every event carries a UTC timestamp."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from linegate.dataio import ROOT

TRACE_DIR = ROOT / "traces"


class TraceWriter:
    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, event: str, **fields) -> None:
        record = {"at": datetime.now(timezone.utc).isoformat(timespec="milliseconds"), "event": event} | fields
        with self.path.open("a") as handle:
            handle.write(json.dumps(record, default=str) + "\n")


def new_trace_path(agent: str) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return TRACE_DIR / f"{agent}-{stamp}.jsonl"
