import json

import pytest

from linegate.agents import leakage_warden
from linegate.agents.runtime import Completion, ToolCall
from linegate.features.registry import FeatureRegistry
from linegate.model.leak_tests import RefitResult, ScopeResult, ShuffleResult
from tests.unit.test_leak_tests import CFG


class FakeLab:
    """Canned test results: 'leaky' features collapse and change with row scope, others hold."""

    def __init__(self):
        self.calls = []

    def id_shuffle_test(self, sql):
        self.calls.append("shuffle")
        return ShuffleResult(lift=0.2, shuffled_lift=0.01, collapse=0.95) if "lag" in sql else \
            ShuffleResult(lift=0.1, shuffled_lift=0.1, collapse=0.0)

    def strict_time_refit(self, sql):
        self.calls.append("refit")
        return RefitResult(time_lift=0.08, random_lift=0.1, retention=0.8)

    def row_scope_check(self, sql):
        self.calls.append("scope")
        return ScopeResult(rows_compared=50, mismatched=30 if "lag" in sql else 0, mismatch_share=0.6 if "lag" in sql else 0.0)


class ScriptedClient:
    def __init__(self, turns):
        self.turns = list(turns)

    def complete(self, messages, tools):
        return self.turns.pop(0)


def tc(name, fid, cid, **extra):
    return ToolCall(id=cid, name=name, arguments=json.dumps({"feature_id": fid} | extra))


def run_all_tests(fid):
    return Completion("", [tc("id_shuffle_test", fid, "1"), tc("strict_time_refit", fid, "2"), tc("row_scope_check", fid, "3")], 1, 1)


@pytest.fixture
def registry(tmp_path):
    reg = FeatureRegistry(tmp_path / "reg.json")
    leaky = reg.propose("gap", "Adjacent Ids fail together.", "SELECT Id, Id - lag(Id) OVER (ORDER BY Id) AS g FROM parts_date", "red_team")
    honest = reg.propose("f0", "High F0 fails.", "SELECT Id, L0_S0_F0 AS x FROM parts_numeric", "red_team")
    return reg, leaky, honest


def review(registry, fid, turns, tmp_path):
    reg = registry
    return leakage_warden.review_feature(fid, reg, ScriptedClient(turns), CFG | {"max_usd": 1.0, "max_turns": 6},
                                         tmp_path / "trace.jsonl", FakeLab())


def test_agent_approval_of_a_leak_is_overridden(registry, tmp_path):
    reg, leaky, _ = registry
    finding = "All tests look acceptable to me for this feature."
    decision = review(reg, leaky, [run_all_tests(leaky), Completion("", [tc("approve", leaky, "4", finding=finding)], 1, 1),
                                   Completion("done", [], 1, 1)], tmp_path)
    assert decision.final == "quarantined" and decision.agent == "approved" and decision.overridden
    assert reg.get(leaky).status == "quarantined"
    events = [json.loads(l)["event"] for l in (tmp_path / "trace.jsonl").read_text().splitlines()]
    assert "rule_override" in events


def test_decision_before_tests_is_refused(registry, tmp_path):
    reg, _, honest = registry
    finding = "Approving without looking at the evidence at all."
    decision = review(reg, honest, [Completion("", [tc("approve", honest, "1", finding=finding)], 1, 1),
                                    Completion("gave up", [], 1, 1)], tmp_path)
    events = [json.loads(l)["event"] for l in (tmp_path / "trace.jsonl").read_text().splitlines()]
    assert "tool_refused" in events
    assert decision.agent is None and decision.final == "approved"  # rule applied after running the tests itself


def test_agent_quarantine_of_a_leak_stands(registry, tmp_path):
    reg, leaky, _ = registry
    reason = "Lift collapsed by 95 percent under the Id shuffle test."
    decision = review(reg, leaky, [run_all_tests(leaky), Completion("", [tc("quarantine", leaky, "4", reason=reason)], 1, 1),
                                   Completion("done", [], 1, 1)], tmp_path)
    assert decision.final == decision.agent == "quarantined" and not decision.overridden
    assert reason in reg.get(leaky).findings[-1]["finding"]


def test_decisions_for_another_feature_are_refused(registry, tmp_path):
    reg, leaky, honest = registry
    reason = "Trying to quarantine a feature outside this review."
    review(reg, leaky, [run_all_tests(leaky), Completion("", [tc("quarantine", honest, "4", reason=reason)], 1, 1),
                        Completion("done", [], 1, 1)], tmp_path)
    assert reg.get(honest).status == "proposed"
