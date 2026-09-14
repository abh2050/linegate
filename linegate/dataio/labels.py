"""Engineer-confirmed labels from the review console, appended to the training-label store."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from linegate.dataio import DATA_DIR

CONFIRMED_PATH = DATA_DIR / "training_set" / "confirmed_labels.jsonl"


def append_confirmed(path: Path, part_id: int, action: str, label: int | None) -> dict:
    record = {"part_id": part_id, "action": action, "label": label,
              "at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(record) + "\n")
    return record


def load_decisions(path: Path = CONFIRMED_PATH) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def load_confirmed(path: Path = CONFIRMED_PATH) -> list[dict]:
    """Decisions that carry a label (ship = 0, scrap = 1); senior reviews carry none."""
    return [d for d in load_decisions(path) if d["label"] is not None]
