import json

import pytest

from linegate.agents import hypothesis_agent
from linegate.agents.hypothesis_agent import SearchSession
from linegate.agents.leakage_warden import WardenDecision
from linegate.agents.runtime import ToolRefused
from linegate.agents.tools import EvaluateInput, ProposeInput
from linegate.agents.traces import TraceWriter
from linegate.features.registry import FeatureRegistry

CFG = {"max_evaluate_calls": 40, "max_usd": 8.0, "stop_after_failed_proposals": 3, "min_delta_mcc": 0.004,
       "noise_sd_multiplier": 2.0}
BASELINE = {"mcc": 0.20, "ap": 0.070, "sd_mcc": 0.005, "sd_ap": 0.002}  # bars: MCC >= 0.21, AP >= 0.074
AP = 0.075
GOOD = "SELECT Id, L0_S0_F0 AS x FROM parts_numeric"


class FakeScorer:
    def __init__(self, results):
        self.results, self.calls = list(results), []

    def score(self, records):
        self.calls.append([r.name for r in records])
        return self.results.pop(0)


def reviewer_for(registry, verdicts):
    def review(fid):
        rec = registry.get(fid)
        final = verdicts.get(rec.name, "approved")
        registry.decide(fid, final, "test finding", actor="warden")
        return WardenDecision(fid, rec.name, final, final, False, ["evidence"])
    return review


@pytest.fixture
def make_session(tmp_path):
    def build(scores, verdicts=None):
        reg = FeatureRegistry(tmp_path / "reg.json")
        session = SearchSession(reg, reviewer_for(reg, verdicts or {}), FakeScorer(scores), BASELINE, CFG,
                                TraceWriter(tmp_path / "t.jsonl"), holdout_ids=set())
        return session, reg
    return build


def propose(session, name, sql=GOOD):
    return session.propose(ProposeInput(name=name, hypothesis=f"{name} tracks a physical fault.", sql=sql))


def test_propose_runs_warden_and_blocks_window_sql(make_session):
    session, reg = make_session([])
    out = propose(session, "a")
    assert out["status"] == "approved" and reg.get(out["feature_id"]).status == "approved"
    with pytest.raises(ToolRefused, match="window"):
        propose(session, "b", "SELECT Id, lag(L0_S0_F0) OVER (ORDER BY L0_S0_F0) AS g FROM parts_numeric")


def test_evaluate_refuses_quarantined_and_unknown(make_session):
    session, reg = make_session([], verdicts={"leak": "quarantined"})
    leak = propose(session, "leak")["feature_id"]
    with pytest.raises(ToolRefused, match="quarantined"):
        session.evaluate(EvaluateInput(feature_ids=[leak]))
    with pytest.raises(ToolRefused, match="unknown"):
        session.evaluate(EvaluateInput(feature_ids=["f_0123456789"]))


def test_improvement_needs_mcc_and_ap_beyond_noise(make_session):
    scores = [
        {"mcc": 0.215, "ap": 0.071, "per_seed": [0.2] * 3, "holdout_ids_referenced": 0},  # MCC clears, AP does not
        {"mcc": 0.208, "ap": 0.080, "per_seed": [0.2] * 3, "holdout_ids_referenced": 0},  # AP clears, MCC within noise
        {"mcc": 0.211, "ap": 0.075, "per_seed": [0.2] * 3, "holdout_ids_referenced": 0},  # both clear
    ]
    session, _ = make_session(scores)
    ids = [propose(session, n)["feature_id"] for n in ("a", "b", "c")]
    results = [session.evaluate(EvaluateInput(feature_ids=[i])) for i in ids]
    assert [r["improved"] for r in results] == [False, False, True]
    assert session.accepted == [ids[2]] and session.best_mcc == 0.211 and session.best_ap == 0.075
    assert results[0]["mcc_bar"] == pytest.approx(0.21) and results[0]["ap_bar"] == pytest.approx(0.074)


def test_min_delta_is_a_floor_when_noise_is_tiny(tmp_path):
    reg = FeatureRegistry(tmp_path / "reg.json")
    quiet = {"mcc": 0.20, "ap": 0.07, "sd_mcc": 0.0001, "sd_ap": 0.0}
    session = SearchSession(reg, reviewer_for(reg, {}), FakeScorer([]), quiet, CFG, TraceWriter(tmp_path / "t.jsonl"), set())
    assert session.bars()[0] == pytest.approx(0.204)


def test_nudge_only_while_search_is_open(make_session):
    flat = {"mcc": 0.20, "ap": 0.07, "per_seed": [0.2] * 3, "holdout_ids_referenced": 0}
    session, _ = make_session([flat] * 3)
    assert "40 evaluate calls remain" in session.nudge()
    for name in ("a", "b", "c"):
        session.evaluate(EvaluateInput(feature_ids=[propose(session, name)["feature_id"]]))
    assert session.nudge() is None


def test_accepted_features_are_carried_into_later_evaluations(make_session):
    scores = [{"mcc": 0.22, "per_seed": [0.21, 0.21, 0.22], "ap": 0.08, "holdout_ids_referenced": 0},
              {"mcc": 0.221, "per_seed": [0.22, 0.22, 0.22], "ap": 0.081, "holdout_ids_referenced": 0}]
    session, _ = make_session(scores)
    a, b = (propose(session, n)["feature_id"] for n in ("a", "b"))
    session.evaluate(EvaluateInput(feature_ids=[a]))
    out = session.evaluate(EvaluateInput(feature_ids=[b]))
    assert session.scorer.calls == [["a"], ["a", "b"]] and not out["improved"]


def test_three_consecutive_failed_evaluations_stop_the_search(make_session):
    flat = {"mcc": 0.20, "per_seed": [0.19, 0.20, 0.21], "ap": AP, "holdout_ids_referenced": 0}
    session, _ = make_session([flat, flat, flat], verdicts={"q": "quarantined"})
    a, b, c = (propose(session, n)["feature_id"] for n in ("a", "b", "c"))
    propose(session, "q")
    assert session.consecutive_failures == 0 and session.stop_reason() is None  # quarantines do not count
    session.evaluate(EvaluateInput(feature_ids=[a]))
    session.evaluate(EvaluateInput(feature_ids=[b]))
    assert session.stop_reason() is None
    session.evaluate(EvaluateInput(feature_ids=[c]))
    assert session.consecutive_failures == 3 and "3 consecutive" in session.stop_reason()


def test_holdout_reference_is_traced_and_summary_counts(make_session, tmp_path):
    session, _ = make_session([{"mcc": 0.19, "per_seed": [0.1, 0.1, 0.1], "ap": AP, "holdout_ids_referenced": 2}],
                              verdicts={"q": "quarantined"})
    a = propose(session, "a")["feature_id"]
    propose(session, "q")
    session.evaluate(EvaluateInput(feature_ids=[a]))
    summary = session.summary()
    assert summary["proposals"] == 2 and summary["survival_rate"] == 0.5
    assert summary["holdout_ids_referenced"] == 2
    events = [json.loads(l) for l in (tmp_path / "t.jsonl").read_text().splitlines()]
    assert any(e["event"] == "evaluate" and e["holdout_ids_referenced"] == 2 for e in events)


def test_gate_check_reports_each_requirement():
    summary = {"improving_approved_features": [], "survival_rate": 0.5, "ap": AP, "holdout_ids_referenced": 0}
    problems = hypothesis_agent.gate_problems(summary, trace_complete=True, holdout_runs_unchanged=False)
    assert any("improved" in p for p in problems) and any("holdout_runs" in p for p in problems)
    ok = {"improving_approved_features": ["f_x"], "survival_rate": 0.5, "ap": AP, "holdout_ids_referenced": 0}
    assert hypothesis_agent.gate_problems(ok, trace_complete=True, holdout_runs_unchanged=True) == []


def test_sql_that_does_not_run_is_refused_before_registration(tmp_path):
    from linegate.features.sandbox import FeatureSQLError

    def dry_run(sql):
        raise FeatureSQLError("Referenced column not found")

    reg = FeatureRegistry(tmp_path / "reg.json")
    session = SearchSession(reg, reviewer_for(reg, {}), FakeScorer([]), BASELINE, CFG,
                            TraceWriter(tmp_path / "t.jsonl"), holdout_ids=set(), dry_run=dry_run)
    with pytest.raises(ToolRefused, match="not found"):
        propose(session, "bad")
    assert reg.records == {}


def test_explore_dry_run_catches_binder_errors(tmp_path):
    from linegate.features.sandbox import FeatureSQLError
    from tests.unit.conftest_catalog import build_layout

    layout = build_layout(tmp_path)
    explore = hypothesis_agent.ExploreTools.__new__(hypothesis_agent.ExploreTools)
    explore.parquet_dir = layout
    explore.dry_run("SELECT Id, L0_S0_F0 AS x FROM parts_numeric")
    with pytest.raises(FeatureSQLError, match="does not run"):
        explore.dry_run("SELECT Id, L9_S99_D1 AS x FROM parts_date")
    with pytest.raises(FeatureSQLError, match="Id"):
        explore.dry_run("SELECT L0_S0_F0 AS x FROM parts_numeric")
