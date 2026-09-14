"""Read cost parameters from the YAML block of the cost memo, strictly."""

from __future__ import annotations

import dataclasses
import re
from pathlib import Path

import yaml

from linegate.cost.curve import CostParameters

YAML_BLOCK = re.compile(r"```yaml\n(.*?)```", re.DOTALL)


class MemoError(ValueError):
    """Raised when the memo has no valid cost block."""


def read_cost_memo(path: Path) -> CostParameters:
    match = YAML_BLOCK.search(path.read_text())
    if match is None:
        raise MemoError(f"{path} has no ```yaml block")
    raw = yaml.safe_load(match.group(1)) or {}
    fields = {f.name for f in dataclasses.fields(CostParameters)}
    if set(raw) != fields:
        raise MemoError(f"memo fields must be exactly {sorted(fields)}; missing {sorted(fields - set(raw))}, "
                        f"unexpected {sorted(set(raw) - fields)}")
    if any(not isinstance(v, (int, float)) or v < 0 for v in raw.values()):
        raise MemoError("memo values must be non-negative numbers")
    return CostParameters(**raw)
