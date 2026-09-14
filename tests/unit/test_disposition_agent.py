import json

from linegate.agents import disposition_agent
from linegate.agents.runtime import Completion, ToolCall
from tests.unit.test_console_api import FakeStore


class ScriptedClient:
    def __init__(self, turns):
        self.turns = list(turns)

    def complete(self, messages, tools):
        return self.turns.pop(0)


def tc(name, cid, **args):
    return ToolCall(id=cid, name=name, arguments=json.dumps(args))


SUBMIT = {"route": "L3_S32", "recommendation": "scrap", "reason": "L3_S32_F3850 read 0.5, outside -0.2 to 0.2.",
          "out_of_range": [{"column": "L3_S32_F3850", "value": 0.5, "low": -0.2, "high": 0.2}],
          "neighbor_summary": "1 of 1 similar parts failed.",
          "citations": [{"kind": "column", "value": "L3_S32_F3850"}, {"kind": "part", "value": "900"}]}
CFG = {"neighbors_k": 20, "max_usd": 1.0, "max_turns": 6}


def test_session_records_what_tools_returned_and_stores_submission(tmp_path):
    turns = [Completion("", [tc("get_part", "1", part_id=11), tc("get_neighbors", "2", part_id=11, k=20)], 1, 1),
             Completion("", [tc("submit_disposition", "3", **SUBMIT)], 1, 1), Completion("done", [], 1, 1)]
    out = disposition_agent.dispose(11, FakeStore(), ScriptedClient(turns), CFG, tmp_path / "t.jsonl", tmp_path / "d")
    saved = json.loads((tmp_path / "d" / "11.json").read_text())
    assert out["status"] == "valid" and saved["disposition"]["recommendation"] == "scrap"
    assert set(saved["allowed_columns"]) == {"L3_S32_F3850"} and set(saved["allowed_part_ids"]) == {11, 900}


def test_tools_refuse_other_parts(tmp_path):
    turns = [Completion("", [tc("get_part", "1", part_id=12)], 1, 1), Completion("stop", [], 1, 1)]
    out = disposition_agent.dispose(11, FakeStore(), ScriptedClient(turns), CFG, tmp_path / "t.jsonl", tmp_path / "d")
    events = [json.loads(l)["event"] for l in (tmp_path / "t.jsonl").read_text().splitlines()]
    assert "tool_refused" in events and out["status"] == "missing"


def test_ungrounded_submission_is_saved_but_marked_rejected(tmp_path):
    bad = SUBMIT | {"citations": [{"kind": "column", "value": "L3_S32_F3850"}, {"kind": "part", "value": "555"}]}
    turns = [Completion("", [tc("get_part", "1", part_id=11)], 1, 1),
             Completion("", [tc("submit_disposition", "2", **bad)], 1, 1), Completion("done", [], 1, 1)]
    out = disposition_agent.dispose(11, FakeStore(), ScriptedClient(turns), CFG, tmp_path / "t.jsonl", tmp_path / "d")
    assert out["status"] == "rejected" and any("555" in e for e in out["errors"])
