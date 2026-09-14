import json

import pytest
from fastapi.testclient import TestClient

from linegate.console import api


class FakeStore:
    schema_columns = {"L3_S32_F3850", "L1_S24_F1846"}

    def __init__(self):
        self.parts = {11: 0.3, 12: 0.2}

    def queue(self):
        return [{"part_id": k, "score": v} for k, v in sorted(self.parts.items(), key=lambda kv: -kv[1])]

    def get_part(self, part_id):
        if part_id not in self.parts:
            raise api.evidence.PartNotFound(part_id)
        return {"part_id": part_id, "score": self.parts[part_id], "route": "L3_S32", "columns": ["L3_S32_F3850"],
                "out_of_range": [{"column": "L3_S32_F3850", "value": 0.5, "low": -0.2, "high": 0.2}]}

    def get_neighbors(self, part_id, k=20):
        return {"neighbors": [{"part_id": 900, "response": 1, "distance": 0.1}], "failed": 1, "k": 1}


def stored(part_id, reason):
    return {"part_id": part_id, "allowed_columns": ["L3_S32_F3850"], "allowed_part_ids": [part_id, 900],
            "disposition": {"route": "L3_S32", "recommendation": "scrap", "reason": reason,
                            "out_of_range": [{"column": "L3_S32_F3850", "value": 0.5, "low": -0.2, "high": 0.2}],
                            "neighbor_summary": "1 of 1 similar parts failed.",
                            "citations": [{"kind": "column", "value": "L3_S32_F3850"}, {"kind": "part", "value": "900"}]}}


@pytest.fixture
def client(tmp_path):
    app = api.create_app(FakeStore(), tmp_path / "dispositions", tmp_path / "confirmed.jsonl", test_mode=True)
    return TestClient(app), tmp_path


def test_queue_and_keyboard_actions_write_confirmed_labels(client):
    c, tmp = client
    assert [i["part_id"] for i in c.get("/api/queue").json()["items"]] == [11, 12]
    assert c.post("/api/parts/11/disposition", json={"action": "scrap"}).json()["label"] == 1
    assert c.post("/api/parts/12/disposition", json={"action": "senior_review"}).json()["label"] is None
    assert c.get("/api/queue").json()["items"] == []
    confirmed = c.get("/api/training-set/confirmed").json()
    assert [(r["part_id"], r["label"]) for r in confirmed] == [(11, 1)]


def test_unknown_part_is_404_and_bad_action_is_422(client):
    c, _ = client
    assert c.get("/api/parts/999").status_code == 404
    assert c.post("/api/parts/999/disposition", json={"action": "ship"}).status_code == 404
    assert c.post("/api/parts/11/disposition", json={"action": "rework"}).status_code == 422


def test_grounded_disposition_renders(client):
    c, tmp = client
    (tmp / "dispositions").mkdir()
    (tmp / "dispositions" / "11.json").write_text(json.dumps(stored(11, "L3_S32_F3850 read 0.5, outside -0.2 to 0.2.")))
    body = c.get("/api/parts/11").json()
    assert body["disposition"]["status"] == "valid" and "L3_S32_F3850" in body["disposition"]["reason"]


def test_fabricated_column_is_rejected_before_it_reaches_the_client(client):
    c, _ = client
    fake = stored(11, "L3_S32_F3850 was high and L7_S77_F7777 confirms it.")
    assert c.post("/api/test/dispositions", json=fake).status_code == 200
    body = c.get("/api/parts/11").json()
    assert body["disposition"]["status"] == "rejected"
    assert "reason" not in body["disposition"] and "L7_S77_F7777" in " ".join(body["disposition"]["errors"])
    assert "confirms it" not in json.dumps(body)


def test_missing_disposition_routes_to_senior_review(client):
    c, _ = client
    assert c.get("/api/parts/12").json()["disposition"] == {"status": "missing", "recommendation": "senior_review"}


def test_test_endpoints_absent_outside_test_mode(tmp_path):
    app = api.create_app(FakeStore(), tmp_path / "d", tmp_path / "c.jsonl", test_mode=False)
    assert TestClient(app).post("/api/test/dispositions", json={}).status_code in (404, 405)
