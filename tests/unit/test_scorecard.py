import json

from linegate.reporting import scorecard


def write_trace(path, events):
    path.write_text("\n".join(json.dumps(e) for e in events) + "\n")


def test_llm_spend_sums_the_peak_of_each_agent_run(tmp_path):
    write_trace(tmp_path / "hypothesis-1.jsonl", [
        {"event": "agent_start"}, {"event": "model_turn", "spent_usd": 0.1}, {"event": "model_turn", "spent_usd": 0.3},
        {"event": "agent_start"}, {"event": "model_turn", "spent_usd": 0.05}])
    write_trace(tmp_path / "warden-2.jsonl", [{"event": "agent_start"}, {"event": "model_turn", "spent_usd": 0.02}])
    spend = scorecard.llm_spend(tmp_path)
    assert round(spend["total"], 4) == 0.37 and round(spend["by_agent"]["hypothesis"], 4) == 0.35


def test_render_contains_every_required_item():
    facts = {
        "baseline_mcc": 0.2267, "baseline_v1_mcc": 0.1559, "search_reference_mcc": 0.221, "post_search_mcc": 0.221,
        "accepted_features": [], "survival": {"approved": 10, "proposed": 14, "rate": 10 / 14},
        "quarantined": ["mindate_id_diff", "l3_s29_dwell"], "red_team": {"mindate_id_diff": "quarantined"},
        "spend": {"total": 9.5, "by_agent": {"hypothesis": 8.0}}, "fabricated_citations": 0, "dispositions": 6,
        "holdout": {"mcc_at_validation_threshold": 0.2, "auc": 0.62, "average_precision": 0.07, "committed_share": 0.63,
                    "abstain_share": 0.37, "escaped_failures": 300, "failures": 700, "parts": 236616, "failures_in_review": 250, "failures_inspected": 150,
                    "dollars_per_shift": 12000.0, "ship_all_dollars_per_shift": 14000.0, "leak_tripwire": False},
        "validation_rehearsal": {"mcc_at_validation_threshold": 0.2267},
        "gates": {"0": True, "1": True, "2": True, "3": True, "4": False, "5": True, "6": True, "7": True},
        "leak_lift": {"lift": 0.2056, "shuffled_lift": 0.001}, "policy_version": "policy-x", "merge_refusals": 2,
        "holdout_runs": 1,
    }
    text = scorecard.render(facts)
    for needle in ("Baseline MCC", "Post-search MCC", "Proposal survival rate", "Quarantined", "mindate_id_diff",
                   "Token spend per MCC point", "Fabricated citations", "Committed share", "Abstain share",
                   "Escaped errors", "Leaky", "0.49", "cannot run on a line"):
        assert needle in text, needle
    assert "undefined" in text  # no MCC gain, so spend per point has no finite value
