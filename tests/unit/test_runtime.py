import json

import pytest
from pydantic import BaseModel

from linegate.agents import runtime
from linegate.agents.runtime import Budget, Completion, Tool, ToolCall, ToolRefused
from linegate.agents.traces import TraceWriter


class EchoInput(BaseModel):
    text: str


class ScriptedClient:
    def __init__(self, turns):
        self.turns, self.seen = list(turns), []

    def complete(self, messages, tools):
        self.seen.append([dict(m) for m in messages])
        return self.turns.pop(0)


def call(name, args, cid="c1"):
    return ToolCall(id=cid, name=name, arguments=json.dumps(args) if isinstance(args, dict) else args)


def events(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def refuse(_):
    raise ToolRefused("not allowed yet")


TOOLS = [Tool("echo", "echo text", EchoInput, lambda a: {"echo": a.text}), Tool("guarded", "refuses", EchoInput, refuse)]


def test_schema_violation_is_rejected_traced_and_returned(tmp_path):
    client = ScriptedClient([Completion("", [call("echo", {"txt": 1})], 10, 5), Completion("done", [], 10, 5)])
    trace = TraceWriter(tmp_path / "t.jsonl")
    result = runtime.run_agent(client, "sys", "go", TOOLS, Budget(max_usd=1.0), trace, max_turns=5)
    assert result == "done"
    kinds = [e["event"] for e in events(tmp_path / "t.jsonl")]
    assert "tool_rejected" in kinds and "tool_result" not in kinds
    assert "schema validation failed" in client.seen[1][-1]["content"]


def test_unknown_tool_and_refusal_are_traced(tmp_path):
    client = ScriptedClient([Completion("", [call("merge", {}, "a"), call("guarded", {"text": "x"}, "b")], 1, 1),
                             Completion("ok", [], 1, 1)])
    trace = TraceWriter(tmp_path / "t.jsonl")
    runtime.run_agent(client, "sys", "go", TOOLS, Budget(max_usd=1.0), trace, max_turns=5)
    kinds = [e["event"] for e in events(tmp_path / "t.jsonl")]
    assert kinds.count("tool_rejected") == 1 and kinds.count("tool_refused") == 1


def test_budget_stops_the_loop(tmp_path):
    turns = [Completion("", [call("echo", {"text": "x"})], 1_000_000, 0) for _ in range(5)]
    budget = Budget(max_usd=1.0, usd_per_mtok_in=0.6, usd_per_mtok_out=0.0)
    trace = TraceWriter(tmp_path / "t.jsonl")
    runtime.run_agent(ScriptedClient(turns), "sys", "go", TOOLS, budget, trace, max_turns=5)
    kinds = [e["event"] for e in events(tmp_path / "t.jsonl")]
    assert "budget_stop" in kinds and kinds.count("model_turn") == 2


def test_per_tool_call_cap(tmp_path):
    turns = [Completion("", [call("echo", {"text": str(i)}, f"c{i}")], 1, 1) for i in range(3)] + [Completion("end", [], 1, 1)]
    budget = Budget(max_usd=1.0, max_calls={"echo": 2})
    trace = TraceWriter(tmp_path / "t.jsonl")
    runtime.run_agent(ScriptedClient(turns), "sys", "go", TOOLS, budget, trace, max_turns=10)
    kinds = [e["event"] for e in events(tmp_path / "t.jsonl")]
    assert kinds.count("tool_result") == 2 and kinds.count("tool_refused") == 1


def test_stop_hook_ends_the_loop(tmp_path):
    turns = [Completion("", [call("echo", {"text": str(i)}, f"c{i}")], 1, 1) for i in range(5)]
    state = {"n": 0}

    def echo(a):
        state["n"] += 1
        return {"echo": a.text}

    tools = [Tool("echo", "echo", EchoInput, echo)]
    trace = TraceWriter(tmp_path / "t.jsonl")
    runtime.run_agent(ScriptedClient(turns), "sys", "go", tools, Budget(max_usd=1.0), trace, max_turns=5,
                      should_stop=lambda: "enough" if state["n"] >= 2 else None)
    kinds = [e["event"] for e in events(tmp_path / "t.jsonl")]
    assert state["n"] == 2 and kinds[-1] == "agent_end"


def test_tool_exception_is_traced_and_the_loop_continues(tmp_path):
    def boom(_):
        raise KeyError("column missing")

    tools = [Tool("boom", "fails", EchoInput, boom)]
    client = ScriptedClient([Completion("", [call("boom", {"text": "x"})], 1, 1), Completion("recovered", [], 1, 1)])
    trace = TraceWriter(tmp_path / "t.jsonl")
    assert runtime.run_agent(client, "sys", "go", tools, Budget(max_usd=1.0), trace, max_turns=5) == "recovered"
    assert "tool_error" in [e["event"] for e in events(tmp_path / "t.jsonl")]
    assert "tool failed" in client.seen[1][-1]["content"]
