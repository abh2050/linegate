"""Review console API: the abstain-band queue, part evidence, and engineer decisions.

Every disposition is re-validated against its saved tool results on every
request; a rejected one reaches the client only as errors, never as text.
Decisions append to the training-label store. Test-only endpoints exist
only when the app is created in test mode.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Literal

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from linegate.console import citations, evidence
from linegate.dataio import DATA_DIR, PARQUET_DIR, ROOT, labels

FRONTEND_DIST = ROOT / "frontend" / "dist"
LABEL_FOR = {"ship": 0, "scrap": 1, "senior_review": None}


class ActionInput(BaseModel):
    action: Literal["ship", "scrap", "senior_review"]


def render_disposition(store, disposition_dir: Path, record: dict) -> dict:
    path = disposition_dir / f"{record['part_id']}.json"
    if not path.exists():
        return {"status": "missing", "recommendation": "senior_review"}
    saved = json.loads(path.read_text())
    result = citations.validate_disposition(saved["disposition"], store.schema_columns, set(saved["allowed_columns"]),
                                            set(saved["allowed_part_ids"]), record)
    if not result.valid:
        return {"status": "rejected", "recommendation": "senior_review", "errors": result.errors}
    return {"status": "valid"} | saved["disposition"]


def create_app(store, disposition_dir: Path, confirmed_path: Path, test_mode: bool = False,
               static_dir: Path | None = None) -> FastAPI:
    app = FastAPI(title="linegate review console")

    def fetch(part_id: int) -> dict:
        try:
            return store.get_part(part_id)
        except evidence.PartNotFound:
            raise HTTPException(404, f"part {part_id} is not in the review queue") from None

    @app.get("/api/queue")
    def queue(limit: int = 100) -> dict:
        decided = {d["part_id"] for d in labels.load_decisions(confirmed_path)}
        items = [q for q in store.queue() if q["part_id"] not in decided]
        return {"total": len(items), "items": items[:limit]}

    @app.get("/api/parts/{part_id}")
    def part(part_id: int) -> dict:
        record = fetch(part_id)
        return {"part": record, "neighbors": store.get_neighbors(part_id),
                "disposition": render_disposition(store, disposition_dir, record)}

    @app.post("/api/parts/{part_id}/disposition")
    def decide(part_id: int, body: ActionInput) -> dict:
        fetch(part_id)
        return labels.append_confirmed(confirmed_path, part_id, body.action, LABEL_FOR[body.action])

    @app.get("/api/training-set/confirmed")
    def confirmed() -> list[dict]:
        return labels.load_confirmed(confirmed_path)

    if test_mode:
        @app.post("/api/test/dispositions")
        def inject(payload: dict) -> dict:
            disposition_dir.mkdir(parents=True, exist_ok=True)
            (disposition_dir / f"{int(payload['part_id'])}.json").write_text(json.dumps(payload))
            return {"stored": payload["part_id"]}

    if static_dir is not None and static_dir.exists():
        app.mount("/", StaticFiles(directory=static_dir, html=True), name="console")
    return app


def build_store() -> evidence.PartStore:
    from linegate.cost.policy import POLICY_DIR
    from linegate.features.compute import FEATURE_SET
    from linegate.model.train import ARTIFACT_DIR

    metrics = json.loads((ARTIFACT_DIR / "metrics.json").read_text())
    top = [n for n, _ in metrics["top_gain_features"] if n != "route_code"][:10]
    return evidence.PartStore(PARQUET_DIR, ARTIFACT_DIR / "validation_scores.npz",
                              json.loads((POLICY_DIR / "policy.json").read_text()),
                              ARTIFACT_DIR / f"{FEATURE_SET}_train.parquet", ARTIFACT_DIR / f"{FEATURE_SET}_validation.parquet",
                              top, cache_dir=DATA_DIR / "artifacts" / "console")


def main() -> None:
    parser = argparse.ArgumentParser(prog="linegate.console.api")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()
    confirmed = Path(os.environ.get("LINEGATE_CONFIRMED_PATH", labels.CONFIRMED_PATH))
    dispositions = Path(os.environ.get("LINEGATE_DISPOSITION_DIR", DATA_DIR / "artifacts" / "dispositions"))
    app = create_app(build_store(), dispositions, confirmed, test_mode=args.test, static_dir=FRONTEND_DIST)
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
