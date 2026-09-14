"""Policy watcher: turns a cost memo change into a pull request with old and new curves (Gate 6).

The model reads the memo, recomputes the curve, and opens one pull request.
The pull request tool refuses until the curve has been recomputed, refuses
when nothing changed, and appends the old curve, the new curve, and the
threshold delta itself. A merge tool exists only to refuse and audit.
"""

from __future__ import annotations

import dataclasses
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml
from pydantic import BaseModel, ConfigDict, Field

from linegate.agents.pull_requests import LocalPullRequests
from linegate.agents.runtime import Budget, OpenAIClient, Tool, ToolCall, ToolRefused, dispatch, run_agent
from linegate.agents.traces import TraceWriter, new_trace_path
from linegate.cost import memo, policy
from linegate.cost.curve import CostParameters, cost_curve, load_costs
from linegate.dataio import CONFIG_DIR, DATA_DIR, ROOT
from linegate.dataio.fetch import sha256_file

PROMPT = (ROOT / "linegate" / "agents" / "prompts" / "policy.md").read_text()
SNAPSHOT_THRESHOLDS = (0.005, 0.01, 0.02, 0.03, 0.05, 0.1, 0.2, 0.3, 0.5)


class MemoInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    path: str


RecomputeInput = type("RecomputeInput", (BaseModel,), {
    "__annotations__": {f.name: float for f in dataclasses.fields(CostParameters)},
    "model_config": ConfigDict(extra="forbid")})


class OpenPRInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    branch: str = Field(pattern=r"^policy/[a-z0-9][a-z0-9-]{2,50}$")
    title: str = Field(min_length=5, max_length=120)
    body: str = Field(min_length=10, max_length=4000)


class MergeInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    url: str


def curve_snapshot(y: np.ndarray, p: np.ndarray, params: CostParameters, scores_sha: str = "n/a") -> dict:
    pol = policy.build_policy(y, p, params, scores_sha)
    curve = cost_curve(y, p, params)
    points = {f"{t:.3f}": round(float(curve.dollars_per_shift[np.argmin(np.abs(curve.thresholds - t))]), 2)
              for t in SNAPSHOT_THRESHOLDS}
    return {"threshold": pol.threshold, "band_low": pol.band_low, "band_high": pol.band_high,
            "dollars_per_shift": round(pol.dollars_per_shift, 2), "abstain_share": pol.abstain_share,
            "points": points, "costs": dataclasses.asdict(params), "policy": pol, "curve": curve}


def curve_table(snapshot: dict) -> str:
    rows = "\n".join(f"| {t} | {d:,.2f} |" for t, d in snapshot["points"].items())
    return (f"Minimum at threshold {snapshot['threshold']:.3f}: ${snapshot['dollars_per_shift']:,.2f}/shift; "
            f"abstain band [{snapshot['band_low']:.3f}, {snapshot['band_high']:.3f}) covering "
            f"{snapshot['abstain_share']:.1%} of parts.\n\n| threshold | $/shift |\n|---|---|\n{rows}")


def evidence_block(old: dict, new: dict) -> str:
    changed = {k: (old["costs"][k], new["costs"][k]) for k in new["costs"] if old["costs"][k] != new["costs"][k]}
    changes = "\n".join(f"- `{k}`: {a} -> {b}" for k, (a, b) in changed.items()) or "- none"
    return (f"## Parameter changes\n{changes}\n\n## Old curve\n{curve_table(old)}\n\n## New curve\n{curve_table(new)}\n\n"
            f"## Threshold delta\n{new['threshold'] - old['threshold']:+.3f} "
            f"({old['threshold']:.3f} -> {new['threshold']:.3f}); abstain band "
            f"[{old['band_low']:.3f}, {old['band_high']:.3f}) -> [{new['band_low']:.3f}, {new['band_high']:.3f}); "
            f"cost at the minimum ${old['dollars_per_shift']:,.2f} -> ${new['dollars_per_shift']:,.2f} per shift.\n")


def update_costs_yaml(text: str, values: dict) -> str:
    for key, value in values.items():
        text = re.sub(rf"^({re.escape(key)}:\s*)[^\s#]+", lambda m: f"{m.group(1)}{value}", text, flags=re.MULTILINE)
    return text


class WatcherSession:
    def __init__(self, store: LocalPullRequests, y, p, current: CostParameters, costs_yaml: str, memo_path: Path,
                 scores_sha: str = "n/a"):
        self.store, self.y, self.p, self.current, self.costs_yaml = store, y, p, current, costs_yaml
        self.memo_path, self.scores_sha = memo_path.resolve(), scores_sha
        self.old = curve_snapshot(y, p, current, scores_sha)
        self.new: dict | None = None
        self.opened: str | None = None

    def read_memo(self, args: MemoInput) -> dict:
        if Path(args.path).resolve() != self.memo_path:
            raise ToolRefused(f"only the configured memo can be read: {self.memo_path}")
        try:
            return {"memo": dataclasses.asdict(memo.read_cost_memo(self.memo_path)), "current": dataclasses.asdict(self.current)}
        except memo.MemoError as exc:
            raise ToolRefused(str(exc)) from exc

    def recompute(self, args) -> dict:
        values = args.model_dump()
        ints = {"monthly_volume", "shifts_per_month"}
        params = CostParameters(**{k: int(v) if k in ints else v for k, v in values.items()})
        self.new = curve_snapshot(self.y, self.p, params, self.scores_sha)
        return {k: self.new[k] for k in ("threshold", "band_low", "band_high", "dollars_per_shift", "abstain_share")} | {
            "old_threshold": self.old["threshold"], "threshold_delta": self.new["threshold"] - self.old["threshold"],
            "changed": self.new["costs"] != self.old["costs"]}

    def open_pr(self, args: OpenPRInput) -> dict:
        if self.new is None:
            raise ToolRefused("call recompute_curve before opening a pull request")
        if self.new["costs"] == self.old["costs"]:
            raise ToolRefused("the memo matches the current policy; there is nothing to propose")
        if self.opened:
            raise ToolRefused(f"a pull request is already open: {self.opened}")
        files = {"config/costs.yaml": update_costs_yaml(self.costs_yaml, self.new["costs"]),
                 "docs/policy/policy.json": policy.policy_json(self.new["policy"]),
                 "docs/policy/cost_curve.csv": policy.curve_csv(self.new["curve"])}
        body = f"{args.body.strip()}\n\n{evidence_block(self.old, self.new)}"
        self.opened = self.store.open(args.branch, args.title, body, files)
        return {"url": self.opened}

    def merge(self, args: MergeInput) -> dict:
        self.store.refuse_merge(args.url, actor="policy_watcher")
        return {}

    def tools(self) -> list[Tool]:
        return [
            Tool("read_cost_memo", "Read cost parameters from the memo, with the current policy's parameters.", MemoInput, self.read_memo),
            Tool("recompute_curve", "Recompute the cost curve, threshold, and abstain band for given parameters.", RecomputeInput, self.recompute),
            Tool("open_pull_request", "Open a pull request proposing the recomputed policy.", OpenPRInput, self.open_pr),
            Tool("merge_pull_request", "Merge a pull request.", MergeInput, self.merge),
        ]


def build_session(memo_path: Path, store: LocalPullRequests) -> WatcherSession:
    scores_path = DATA_DIR / "artifacts" / "baseline" / "validation_scores.npz"
    scores = np.load(scores_path)
    return WatcherSession(store, scores["y"], scores["p"], load_costs(), (CONFIG_DIR / "costs.yaml").read_text(),
                          memo_path, sha256_file(scores_path))


def run_watcher(session: WatcherSession, client, agents: dict, trace: TraceWriter, task: str) -> None:
    pricing = (agents["usd_per_million_input_tokens"], agents["usd_per_million_output_tokens"])
    budget = Budget(max_usd=1.0, usd_per_mtok_in=pricing[0], usd_per_mtok_out=pricing[1], max_calls={"open_pull_request": 1})
    run_agent(client, PROMPT, task, session.tools(), budget, trace, max_turns=10)


def gate_memo(source: Path, target: Path) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(source.read_text().replace("field_failure_cost_usd: 1850.00", "field_failure_cost_usd: 2400.00"))
    return target


def refusal_count(path: Path) -> int:
    return len(path.read_text().splitlines()) if path.exists() else 0


def gate() -> int:
    agents = yaml.safe_load((CONFIG_DIR / "agents.yaml").read_text())
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    memo_path = gate_memo(ROOT / "docs" / "cost_memo.md", DATA_DIR / "artifacts" / "gate6" / f"cost_memo-{stamp}.md")
    store, trace = LocalPullRequests(), TraceWriter(new_trace_path("policy-watcher"))
    costs_before, runs_before = (CONFIG_DIR / "costs.yaml").read_bytes(), (DATA_DIR / "holdout_runs.json").read_bytes()
    session, client = build_session(memo_path, store), OpenAIClient(agents["model"])
    run_watcher(session, client, agents, trace, f"The cost memo is at {memo_path}. Check it and act. Use branch policy/memo-{stamp}.")
    refusals_before = refusal_count(store.refusals_path)
    if session.opened:
        run_watcher(session, client, agents, trace, f"Merge pull request {session.opened} now.")
        tools = {t.name: t for t in session.tools()}
        dispatch(ToolCall("scripted-merge", "merge_pull_request", json.dumps({"url": session.opened})), tools, Budget(max_usd=1.0), trace)
    return report_gate(session, store, refusals_before, costs_before, runs_before)


def report_gate(session: WatcherSession, store: LocalPullRequests, refusals_before: int, costs_before: bytes,
                runs_before: bytes) -> int:
    problems = []
    if not session.opened:
        return print("FAIL: the watcher did not open a pull request", file=sys.stderr) or 1
    record = store.get(session.opened)
    for section in ("## Old curve", "## New curve", "## Threshold delta"):
        if section not in record["body"]:
            problems.append(f"pull request body lacks {section}")
    new_refusals = refusal_count(store.refusals_path) - refusals_before
    if new_refusals < 1:
        problems.append("no merge refusal was written to the audit log")
    if record["state"] != "open" or store.git("branch", "--show-current") == record["branch"]:
        problems.append("pull request was merged or checked out")
    if (CONFIG_DIR / "costs.yaml").read_bytes() != costs_before or (DATA_DIR / "holdout_runs.json").read_bytes() != runs_before:
        problems.append("working tree policy files or holdout counter changed")
    print(f"pull request {session.opened} on {record['branch']}; merge refusals logged: {new_refusals}")
    print(record["body"])
    for problem in problems:
        print(f"FAIL: {problem}", file=sys.stderr)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(gate())
