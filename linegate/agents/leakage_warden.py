"""Leakage warden: runs the three leak tests on one feature and records a decision.

The model runs the tests through tools and writes the finding. The runtime
refuses a decision until all three tests have run, refuses decisions about
any other feature, and applies the deterministic rule afterwards: a model
approval of a feature the rule quarantines becomes a quarantine. A model
quarantine always stands. If the model decides nothing, the rule decides.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from linegate.agents.runtime import Budget, LLMClient, Tool, ToolRefused, run_agent
from linegate.agents.tools import ApproveInput, FeatureRef, QuarantineInput
from linegate.agents.traces import TraceWriter
from linegate.dataio import ROOT
from linegate.features.registry import FeatureRegistry
from linegate.model import guards

PROMPT = (ROOT / "linegate" / "agents" / "prompts" / "warden.md").read_text()
TESTS = ("id_shuffle_test", "strict_time_refit", "row_scope_check")


@dataclass
class WardenDecision:
    feature_id: str
    name: str
    agent: str | None
    final: str
    overridden: bool
    evidence: list[str]


class WardenSession:
    def __init__(self, fid: str, registry: FeatureRegistry, lab, cfg: dict, trace: TraceWriter):
        self.fid, self.registry, self.lab, self.cfg, self.trace = fid, registry, lab, cfg, trace
        self.results: dict[str, object] = {}
        self.agent_decision: tuple[str, str] | None = None

    def own(self, args: FeatureRef) -> str:
        if args.feature_id != self.fid:
            raise ToolRefused(f"this review covers {self.fid} only")
        return self.registry.get(self.fid).sql

    def run_test(self, name: str) -> object:
        if name not in self.results:
            sql = self.registry.get(self.fid).sql
            self.results[name] = getattr(self.lab, name)(sql)
        return self.results[name]

    def test_tool(self, name: str, description: str) -> Tool:
        def handler(args: FeatureRef):
            self.own(args)
            return self.run_test(name)
        return Tool(name, description, FeatureRef, handler)

    def decide(self, decision: str, text: str) -> dict:
        missing = [t for t in TESTS if t not in self.results]
        if missing:
            raise ToolRefused(f"run {missing} before deciding")
        if self.agent_decision is not None:
            raise ToolRefused("a decision was already recorded for this feature")
        self.agent_decision = (decision, text)
        return {"recorded": decision}

    def tools(self) -> list[Tool]:
        return [
            self.test_tool("id_shuffle_test", "Lift with real Ids vs randomly permuted Ids; collapse is the share lost."),
            self.test_tool("strict_time_refit", "Train-to-validation lift vs random-split lift; retention is their ratio."),
            self.test_tool("row_scope_check", "Recompute on a random half of parts; count values that changed."),
            Tool("quarantine", "Quarantine the feature. Reason must name the test result.", QuarantineInput,
                 lambda a: (self.own(a), self.decide("quarantined", a.reason))[1]),
            Tool("approve", "Approve the feature. Finding must name the test results.", ApproveInput,
                 lambda a: (self.own(a), self.decide("approved", a.finding))[1]),
        ]

    def finalize(self) -> WardenDecision:
        shuffle, refit, scope = (self.run_test(t) for t in TESTS)
        rule, evidence = guards.warden_verdict(shuffle, refit, scope, self.cfg)
        rule_status = "quarantined" if rule == "quarantine" else "approved"
        agent = self.agent_decision[0] if self.agent_decision else None
        final = "quarantined" if "quarantined" in (agent, rule_status) else "approved"
        overridden = agent is not None and agent != final
        text = self.agent_decision[1] if self.agent_decision else "Model recorded no decision; rule applied."
        if overridden:
            self.trace.write("rule_override", feature_id=self.fid, agent=agent, final=final, evidence=evidence)
        elif agent is not None and agent != rule_status:
            self.trace.write("rule_disagreement", feature_id=self.fid, agent=agent, rule=rule_status, evidence=evidence)
        self.registry.decide(self.fid, final, f"{text} | Evidence: {'; '.join(evidence)}", actor="warden")
        name = self.registry.get(self.fid).name
        self.trace.write("warden_decision", feature_id=self.fid, name=name, agent=agent, final=final, evidence=evidence)
        return WardenDecision(self.fid, name, agent, final, overridden, evidence)


def task_message(record, cfg: dict) -> str:
    return (
        f"Feature {record.feature_id} ('{record.name}'), proposed by {record.proposed_by}.\n"
        f"Hypothesis: {record.hypothesis}\nSQL:\n{record.sql}\n\n"
        f"Quarantine rules: any row scope mismatch; Id shuffle collapse >= {cfg['id_shuffle_collapse_threshold']}; "
        f"strict time refit retention < {cfg['strict_split_min_retention']}. Run all three tests, then call "
        "approve or quarantine exactly once with a finding that names the numbers."
    )


def review_feature(fid: str, registry: FeatureRegistry, client: LLMClient, cfg: dict, trace_path: Path, lab,
                   pricing: tuple[float, float] = (0.0, 0.0)) -> WardenDecision:
    trace = TraceWriter(trace_path)
    session = WardenSession(fid, registry, lab, cfg, trace)
    budget = Budget(max_usd=cfg["max_usd"], usd_per_mtok_in=pricing[0], usd_per_mtok_out=pricing[1],
                    max_calls={"quarantine": 1, "approve": 1})
    try:
        run_agent(client, PROMPT, task_message(registry.get(fid), cfg), session.tools(), budget, trace, cfg["max_turns"])
    except Exception as exc:  # the rule still decides if the model call fails
        trace.write("agent_error", feature_id=fid, error=repr(exc))
    return session.finalize()
