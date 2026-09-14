"""Feature store: every proposed feature with its hypothesis, SQL, provenance, and status.

A feature id is a hash of its name and SQL. Only the warden sets approved or
quarantined, every decision carries a written finding, and a quarantine is
final: nothing can approve a quarantined feature afterwards.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from linegate.dataio import DATA_DIR

REGISTRY_PATH = DATA_DIR / "feature_registry.json"
DECIDER = "warden"
NAME = re.compile(r"^[a-z][a-z0-9_]{0,62}$")


class RegistryError(ValueError):
    """Raised on an invalid proposal or an unauthorised decision."""


class QuarantineFinal(RegistryError):
    """Raised when anything tries to reverse a quarantine."""


@dataclass
class FeatureRecord:
    feature_id: str
    name: str
    hypothesis: str
    sql: str
    proposed_by: str
    status: str = "proposed"
    findings: list[dict] = field(default_factory=list)


def feature_id(name: str, sql: str) -> str:
    return "f_" + hashlib.sha256(f"{name}\n{sql}".encode()).hexdigest()[:10]


class FeatureRegistry:
    def __init__(self, path: Path = REGISTRY_PATH):
        self.path = path
        raw = json.loads(path.read_text()) if path.exists() else {}
        self.records = {fid: FeatureRecord(**rec) for fid, rec in raw.items()}

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {fid: dataclasses.asdict(rec) for fid, rec in self.records.items()}
        self.path.write_text(json.dumps(payload, indent=2) + "\n")

    def propose(self, name: str, hypothesis: str, sql: str, proposed_by: str) -> str:
        if not NAME.match(name):
            raise RegistryError(f"feature name {name!r} must be snake_case")
        if not hypothesis.strip():
            raise RegistryError("a plain-language hypothesis is required before a feature is proposed")
        fid = feature_id(name, sql)
        if fid not in self.records:
            self.records[fid] = FeatureRecord(fid, name, hypothesis.strip(), sql, proposed_by)
            self.save()
        return fid

    def get(self, fid: str) -> FeatureRecord:
        if fid not in self.records:
            raise RegistryError(f"unknown feature {fid}")
        return self.records[fid]

    def decide(self, fid: str, status: str, finding: str, actor: str) -> None:
        if actor != DECIDER:
            raise RegistryError(f"only the {DECIDER} decides feature status, not {actor}")
        if status not in ("approved", "quarantined"):
            raise RegistryError(f"invalid status {status}")
        record = self.get(fid)
        if record.status == "quarantined" and status != "quarantined":
            raise QuarantineFinal(f"{fid} is quarantined and cannot be approved")
        record.status = status
        record.findings.append({"actor": actor, "status": status, "finding": finding,
                                "at": datetime.now(timezone.utc).isoformat(timespec="seconds")})
        self.save()
