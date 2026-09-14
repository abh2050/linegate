"""Run the holdout once and write docs/scorecard.md (Gate 7)."""

from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

from linegate.dataio import DATA_DIR, ROOT
from linegate.model.train import ARTIFACT_DIR
from linegate.reporting.holdout import RUNS_PATH, run_holdout

SCORECARD_PATH = ROOT / "docs" / "scorecard.md"
TRACE_DIR = ROOT / "traces"
BASELINE_V1_MCC = 0.1559  # ADR-0002, first Gate 1 run
LEADERBOARD_LEAKY_MCC = 0.49  # CLAUDE.md: 2016 leaderboard, Id ordering leak


def llm_spend(trace_dir: Path = TRACE_DIR) -> dict:
    by_agent: dict[str, float] = defaultdict(float)
    for path in sorted(trace_dir.glob("*.jsonl")):
        if path.name == "refusals.jsonl":
            continue
        agent, peak = re.sub(r"-[0-9T]+Z?$", "", path.stem), 0.0
        for line in path.read_text().splitlines():
            event = json.loads(line)
            if event["event"] == "agent_start":
                by_agent[agent] += peak
                peak = 0.0
            elif event["event"] == "model_turn":
                peak = max(peak, event.get("spent_usd", 0.0))
        by_agent[agent] += peak
    return {"total": sum(by_agent.values()), "by_agent": dict(by_agent)}


def load_json(path: Path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def registry_facts() -> dict:
    search = load_json(DATA_DIR / "feature_registry.json", {})
    red = load_json(DATA_DIR / "red_team_registry.json", {})
    decided = [r for r in search.values() if r["status"] != "proposed"]
    approved = sum(r["status"] == "approved" for r in decided)
    return {"survival": {"approved": approved, "proposed": len(decided), "rate": approved / len(decided) if decided else 0.0},
            "quarantined": [r["name"] for r in list(red.values()) + decided if r["status"] == "quarantined"],
            "red_team": {r["name"]: r["status"] for r in red.values()}}


def disposition_facts(trace_dir: Path = TRACE_DIR) -> dict:
    outcomes = [json.loads(l) for p in trace_dir.glob("disposition-*.jsonl") for l in p.read_text().splitlines()
                if '"disposition_outcome"' in l]
    return {"dispositions": len(outcomes), "fabricated_citations": sum(o["status"] == "rejected" for o in outcomes)}


def leak_lift(trace_dir: Path = TRACE_DIR) -> dict | None:
    for path in sorted(trace_dir.glob("warden-red-team-*.jsonl"), reverse=True):
        current = None
        for line in path.read_text().splitlines():
            event = json.loads(line)
            if event["event"] == "agent_start":
                current = "'mindate_id_diff_reverse'" in event["task"]
            elif current and event["event"] == "tool_result" and event["tool"] == "id_shuffle_test":
                return event["result"]
    return None


def gather(results: dict) -> dict:
    metrics = load_json(ARTIFACT_DIR / "metrics.json")
    summary = load_json(DATA_DIR / "artifacts" / "search" / "summary.json", {})
    policy = load_json(ROOT / "docs" / "policy" / "policy.json")
    refusals = ROOT / "traces" / "refusals.jsonl"
    return registry_facts() | disposition_facts() | {
        "baseline_mcc": metrics["validation_mcc"], "baseline_v1_mcc": BASELINE_V1_MCC,
        "search_reference_mcc": summary.get("reference", {}).get("mcc", metrics["validation_mcc"]),
        "post_search_mcc": summary.get("best_mcc", metrics["validation_mcc"]), "accepted_features": summary.get("accepted_names", []),
        "spend": llm_spend(), "holdout": results["holdout"], "validation_rehearsal": results["validation_rehearsal"],
        "gates": {str(g): g != 4 or bool(summary.get("improving_approved_features")) for g in range(8)},
        "leak_lift": leak_lift(), "policy_version": policy["version"],
        "merge_refusals": len(refusals.read_text().splitlines()) if refusals.exists() else 0,
        "holdout_runs": json.loads(RUNS_PATH.read_text())["holdout_runs"],
    }


def spend_per_point(facts: dict) -> str:
    gain = (facts["post_search_mcc"] - facts["search_reference_mcc"]) * 100
    if gain <= 0:
        return f"undefined: the search produced no MCC gain for ${facts['spend']['by_agent'].get('hypothesis', 0):.2f} of search spend"
    return f"${facts['spend']['by_agent'].get('hypothesis', 0) / gain:.2f} per MCC point"


def render(facts: dict) -> str:
    h, rehearsal = facts["holdout"], facts["validation_rehearsal"]
    gates = " · ".join(f"{g} {'pass' if ok else 'RED'}" for g, ok in facts["gates"].items())
    spend = ", ".join(f"{k} ${v:.2f}" for k, v in sorted(facts["spend"]["by_agent"].items()))
    lift = facts["leak_lift"] or {}
    return f"""# linegate scorecard

Gates: {gates}. Holdout runs: {facts['holdout_runs']}. Policy: `{facts['policy_version']}`.

## Model

| Measure | Value |
|---|---|
| Baseline MCC (validation, v1 baseline) | {facts['baseline_v1_mcc']:.4f} |
| Baseline MCC (validation, baseline_v2, 3 seeds) | {facts['baseline_mcc']:.4f} |
| Search reference MCC (baseline and placebo mean) | {facts['search_reference_mcc']:.4f} |
| Post-search MCC (validation) | {facts['post_search_mcc']:.4f}; accepted features: {', '.join(facts['accepted_features']) or 'none'} |
| Holdout MCC at the validation threshold | {h['mcc_at_validation_threshold']:.4f} (validation rehearsal {rehearsal['mcc_at_validation_threshold']:.4f}) |
| Holdout AUC / average precision | {h['auc']:.4f} / {h['average_precision']:.4f} |

## Search and agents

| Measure | Value |
|---|---|
| Proposal survival rate | {facts['survival']['rate']:.1%} ({facts['survival']['approved']} of {facts['survival']['proposed']} decided proposals) |
| Quarantined | {len(facts['quarantined'])}: {', '.join(facts['quarantined'])} |
| Token spend | ${facts['spend']['total']:.2f} total ({spend}); prices in config/agents.yaml are assumptions |
| Token spend per MCC point | {spend_per_point(facts)} |
| Fabricated citations (agent dispositions) | {facts['fabricated_citations']} of {facts['dispositions']} |
| Merge attempts refused | {facts['merge_refusals']} |

## Decisions on the holdout ({h['parts']:,} parts, {h['failures']:,} failures)

| Measure | Value |
|---|---|
| Committed share (shipped or inspected without review) | {h['committed_share']:.1%} |
| Abstain share (sent to human review) | {h['abstain_share']:.1%} |
| Escaped errors (failures shipped below the band) | {h['escaped_failures']:,} of {h['failures']:,} |
| Failures in review / inspected | {h['failures_in_review']:,} / {h['failures_inspected']:,} |
| Expected cost at the policy threshold | ${h['dollars_per_shift']:,.0f} per shift vs ${h['ship_all_dollars_per_shift']:,.0f} shipping everything |

## Honest vs leaky

| Model | MCC |
|---|---|
| Honest (holdout, this project) | {h['mcc_at_validation_threshold']:.4f} |
| Leaky (2016 leaderboard, Id ordering leak) | {LEADERBOARD_LEAKY_MCC:.2f} |

The leaky number cannot run on a line because it compares each part's Id with the next part that started at the same
moment, and on a live line that next part has not been built yet; the warden measured that feature's lift at
{lift.get('lift', float('nan')):.3f}, falling to {lift.get('shuffled_lift', float('nan')):.3f} once Id order was randomized.
"""


def main() -> int:
    metrics = load_json(ARTIFACT_DIR / "metrics.json")
    results = run_holdout(expected_validation_mcc=metrics["validation_mcc"])
    SCORECARD_PATH.write_text(render(gather(results)))
    print(SCORECARD_PATH.read_text())
    counter = json.loads(RUNS_PATH.read_text())["holdout_runs"]
    if counter != 1:
        print(f"FAIL: holdout_runs is {counter}, expected 1", file=sys.stderr)
        return 1
    if results["holdout"]["leak_tripwire"]:
        print("FAIL: LeakageSuspected: holdout MCC exceeds 0.40", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
