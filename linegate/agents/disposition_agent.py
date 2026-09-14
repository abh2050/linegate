"""Disposition agent: writes the evidence summary and recommendation for one queued part.

The agent may call get_part and get_neighbors for its own part only, check
its citations, and submit once. The submission is saved with the exact
columns and part Ids the tools returned, so the console re-validates it on
every render. The agent never scores the part; a missing or rejected
submission routes the part to senior review.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import yaml

from linegate.agents.runtime import Budget, OpenAIClient, Tool, ToolRefused, run_agent
from linegate.agents.tools import CitationsInput, DispositionInput, NeighborsInput, PartInput
from linegate.agents.traces import TraceWriter, new_trace_path
from linegate.console import citations
from linegate.dataio import CONFIG_DIR, ROOT

PROMPT = (ROOT / "linegate" / "agents" / "prompts" / "disposition.md").read_text()
DISPOSITION_DIR = ROOT / "data" / "artifacts" / "dispositions"


class DispositionSession:
    def __init__(self, part_id: int, store, cfg: dict):
        self.part_id, self.store, self.cfg = part_id, store, cfg
        self.seen_columns: set[str] = set()
        self.seen_parts: set[int] = set()
        self.record: dict | None = None
        self.submission: dict | None = None

    def own(self, part_id: int) -> None:
        if part_id != self.part_id:
            raise ToolRefused(f"this session covers part {self.part_id} only")

    def get_part(self, args: PartInput) -> dict:
        self.own(args.part_id)
        self.record = self.store.get_part(args.part_id)
        self.seen_columns |= set(self.record["columns"])
        self.seen_parts.add(args.part_id)
        return self.record

    def get_neighbors(self, args: NeighborsInput) -> dict:
        self.own(args.part_id)
        result = self.store.get_neighbors(args.part_id, min(args.k, self.cfg["neighbors_k"]))
        self.seen_parts |= {n["part_id"] for n in result["neighbors"]}
        return result

    def check(self, args: CitationsInput) -> dict:
        errors = citations.validate_citations([c.model_dump() for c in args.citations], self.store.schema_columns,
                                              self.seen_columns, self.seen_parts)
        return {"valid": not errors, "errors": errors}

    def submit(self, args: DispositionInput) -> dict:
        if self.record is None:
            raise ToolRefused("call get_part before submitting")
        if self.submission is not None:
            raise ToolRefused("a disposition was already submitted")
        self.submission = args.model_dump()
        return {"received": True}

    def tools(self) -> list[Tool]:
        return [
            Tool("get_part", "The part's route, out-of-range measurements with normal ranges, and present columns.", PartInput, self.get_part),
            Tool("get_neighbors", "Nearest similar train parts and whether each failed.", NeighborsInput, self.get_neighbors),
            Tool("validate_citations", "Check citations before submitting.", CitationsInput, self.check),
            Tool("submit_disposition", "Submit the disposition once.", DispositionInput, self.submit),
        ]


def save(session: DispositionSession, out_dir: Path) -> dict:
    if session.submission is None:
        return {"part_id": session.part_id, "status": "missing"}
    payload = {"part_id": session.part_id, "disposition": session.submission,
               "allowed_columns": sorted(session.seen_columns), "allowed_part_ids": sorted(session.seen_parts)}
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{session.part_id}.json").write_text(json.dumps(payload, indent=2) + "\n")
    result = citations.validate_disposition(session.submission, session.store.schema_columns,
                                            session.seen_columns, session.seen_parts, session.record)
    return {"part_id": session.part_id, "status": "valid" if result.valid else "rejected", "errors": result.errors}


def dispose(part_id: int, store, client, cfg: dict, trace_path: Path, out_dir: Path = DISPOSITION_DIR,
            pricing: tuple[float, float] = (0.0, 0.0)) -> dict:
    trace = TraceWriter(trace_path)
    session = DispositionSession(part_id, store, cfg)
    budget = Budget(max_usd=cfg["max_usd"], usd_per_mtok_in=pricing[0], usd_per_mtok_out=pricing[1],
                    max_calls={"submit_disposition": 1})
    run_agent(client, PROMPT, f"Part {part_id} is in the review queue. Assemble the evidence and submit a disposition.",
              session.tools(), budget, trace, cfg["max_turns"])
    outcome = save(session, out_dir)
    trace.write("disposition_outcome", **outcome)
    return outcome


def main(argv: list[str]) -> int:
    from linegate.console.api import build_store

    count = int(argv[0]) if argv else 5
    agents = yaml.safe_load((CONFIG_DIR / "agents.yaml").read_text())
    cfg = agents["disposition"]
    store, client, trace = build_store(), OpenAIClient(agents["model"]), new_trace_path("disposition")
    pricing = (agents["usd_per_million_input_tokens"], agents["usd_per_million_output_tokens"])
    todo = [q["part_id"] for q in store.queue()[:count] if not (DISPOSITION_DIR / f"{q['part_id']}.json").exists()]
    outcomes = [dispose(pid, store, client, cfg, trace, DISPOSITION_DIR, pricing) for pid in todo]
    for outcome in outcomes:
        print(json.dumps(outcome))
    print(f"trace: {trace}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
