import json
import subprocess

import numpy as np
import pytest

from linegate.agents import policy_watcher, pull_requests
from linegate.agents.runtime import Completion, ToolCall, ToolRefused
from linegate.cost import memo
from linegate.cost.curve import CostParameters

PARAMS = CostParameters(42.0, 1850.0, 6.5, 240000, 90, 1500.0)
MEMO = """# Cost memo

Finance signed off on these numbers.

```yaml
scrap_cost_usd: 42.00
field_failure_cost_usd: 2400.00
inspection_cost_usd: 6.50
monthly_volume: 240000
shifts_per_month: 90
abstain_band_indifference_usd: 1500
```
"""


def test_read_cost_memo_parses_the_yaml_block(tmp_path):
    path = tmp_path / "memo.md"
    path.write_text(MEMO)
    assert memo.read_cost_memo(path).field_failure_cost_usd == 2400.0


@pytest.mark.parametrize("bad", ["no yaml here", MEMO.replace("shifts_per_month: 90\n", ""),
                                 MEMO.replace("scrap_cost_usd: 42.00", "scrap_cost_usd: -1"),
                                 MEMO.replace("monthly_volume", "surprise: 1\nmonthly_volume")])
def test_read_cost_memo_rejects_bad_memos(tmp_path, bad):
    path = tmp_path / "memo.md"
    path.write_text(bad)
    with pytest.raises(memo.MemoError):
        memo.read_cost_memo(path)


@pytest.fixture
def repo(tmp_path):
    subprocess.run(["git", "init", "-q", "-b", "main"], cwd=tmp_path, check=True)
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "costs.yaml").write_text("field_failure_cost_usd: 1850.00   # keep this comment\n")
    subprocess.run(["git", "add", "-A"], cwd=tmp_path, check=True)
    subprocess.run(["git", "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-q", "-m", "init"], cwd=tmp_path, check=True)
    return tmp_path


def test_local_pull_request_creates_branch_without_touching_worktree(repo):
    store = pull_requests.LocalPullRequests(repo, repo / "prs", repo / "refusals.jsonl")
    url = store.open("policy/update", "Update policy", "body text", {"config/costs.yaml": "field_failure_cost_usd: 2400.00\n"})
    assert url.startswith("local://")
    shown = subprocess.run(["git", "show", "policy/update:config/costs.yaml"], cwd=repo, capture_output=True, text=True).stdout
    assert "2400" in shown and "1850" in (repo / "config" / "costs.yaml").read_text()
    assert subprocess.run(["git", "status", "--porcelain"], cwd=repo, capture_output=True, text=True).stdout.strip() in ("", "?? prs/")
    record = store.get(url)
    assert record["state"] == "open" and record["body"] == "body text"
    assert not hasattr(store, "merge")


def test_merge_attempt_is_refused_and_audited(repo):
    store = pull_requests.LocalPullRequests(repo, repo / "prs", repo / "refusals.jsonl")
    url = store.open("policy/x", "t", "b", {"a.txt": "x"})
    with pytest.raises(ToolRefused):
        store.refuse_merge(url, actor="policy_watcher")
    refusal = json.loads((repo / "refusals.jsonl").read_text().splitlines()[-1])
    assert refusal["action"] == "merge" and refusal["target"] == url and refusal["actor"] == "policy_watcher"
    assert store.get(url)["state"] == "open"


def test_update_costs_yaml_preserves_comments():
    text = "scrap_cost_usd: 42.00\nfield_failure_cost_usd: 1850.00   # money\nmemo_path: docs/cost_memo.md\n"
    new = policy_watcher.update_costs_yaml(text, {"field_failure_cost_usd": 2400.0, "scrap_cost_usd": 42.0})
    assert "field_failure_cost_usd: 2400.0   # money" in new and "memo_path: docs/cost_memo.md" in new


def test_pr_evidence_has_old_curve_new_curve_and_threshold_delta():
    rng = np.random.default_rng(0)
    y = (rng.random(20000) < 0.01).astype(np.int8)
    p = np.clip(0.3 * y + 0.2 * rng.random(20000) ** 4, 0, 1)
    old = policy_watcher.curve_snapshot(y, p, PARAMS)
    new = policy_watcher.curve_snapshot(y, p, CostParameters(42.0, 3700.0, 6.5, 240000, 90, 1500.0))
    body = policy_watcher.evidence_block(old, new)
    for section in ("## Old curve", "## New curve", "## Threshold delta"):
        assert section in body
    assert f"{new['threshold'] - old['threshold']:+.3f}" in body


class ScriptedClient:
    def __init__(self, turns):
        self.turns = list(turns)

    def complete(self, messages, tools):
        return self.turns.pop(0)


def tc(name, cid, **args):
    return ToolCall(id=cid, name=name, arguments=json.dumps(args))


def test_open_pr_refused_before_recompute(repo, tmp_path):
    memo_path = tmp_path / "memo.md"
    memo_path.write_text(MEMO)
    store = pull_requests.LocalPullRequests(repo, repo / "prs", repo / "refusals.jsonl")
    rng = np.random.default_rng(1)
    y = (rng.random(5000) < 0.02).astype(np.int8)
    p = np.clip(0.4 * y + 0.3 * rng.random(5000) ** 3, 0, 1)
    session = policy_watcher.WatcherSession(store, y, p, PARAMS, "field_failure_cost_usd: 1850.00\n", memo_path)
    with pytest.raises(ToolRefused, match="recompute"):
        session.open_pr(policy_watcher.OpenPRInput(branch="policy/memo-test", title="Update", body="Costs changed per memo."))
    params = session.read_memo(policy_watcher.MemoInput(path=str(memo_path)))
    session.recompute(policy_watcher.RecomputeInput(**params["memo"]))
    url = session.open_pr(policy_watcher.OpenPRInput(branch="policy/memo-test", title="Update", body="Costs changed per memo."))["url"]
    body = store.get(url)["body"]
    assert body.startswith("Costs changed per memo.") and "## Threshold delta" in body
