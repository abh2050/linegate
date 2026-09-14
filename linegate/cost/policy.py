"""Versioned decision policy built from validation scores and config/costs.yaml (Gate 2).

A score below the band ships, a score at or above the band's upper edge is
inspected, and a score inside the band goes to human review. The version is
a hash of the cost parameters and the exact score file, so any change to
either yields a new version. Reads validation scores only; never the holdout.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from linegate.cost.curve import CostCurve, CostParameters, cost_curve, dollars_per_shift, load_costs
from linegate.cost.thresholds import abstain_band, argmin_threshold
from linegate.dataio import ROOT
from linegate.dataio.fetch import sha256_file
from linegate.model.train import ARTIFACT_DIR

POLICY_DIR = ROOT / "docs" / "policy"
SCORES_PATH = ARTIFACT_DIR / "validation_scores.npz"


class PolicyRejected(RuntimeError):
    """Raised when the model-based policy is no cheaper than a trivial one."""


@dataclass(frozen=True)
class Policy:
    version: str
    threshold: float
    band_low: float
    band_high: float
    dollars_per_shift: float
    ship_all_dollars_per_shift: float
    inspect_all_dollars_per_shift: float
    committed_share: float
    abstain_share: float
    costs: dict
    scores_sha256: str


def policy_version(params: CostParameters, scores_sha256: str) -> str:
    payload = json.dumps({"costs": dataclasses.asdict(params), "scores": scores_sha256}, sort_keys=True)
    return "policy-" + hashlib.sha256(payload.encode()).hexdigest()[:12]


def trivial_costs(y: np.ndarray, params: CostParameters) -> tuple[float, float]:
    pos, neg = int(np.sum(y == 1)), int(np.sum(y == 0))
    ship_all = dollars_per_shift(0, 0, neg, pos, params)
    inspect_all = dollars_per_shift(pos, neg, 0, 0, params)
    return float(ship_all), float(inspect_all)


def build_policy(y: np.ndarray, p: np.ndarray, params: CostParameters, scores_sha256: str) -> Policy:
    curve = cost_curve(y, p, params)
    threshold, dollars = argmin_threshold(curve)
    low, high = abstain_band(curve, params.abstain_band_indifference_usd)
    abstain = float(np.mean((p >= low) & (p < high)))
    ship_all, inspect_all = trivial_costs(y, params)
    return Policy(policy_version(params, scores_sha256), threshold, low, high, dollars, ship_all, inspect_all,
                  1.0 - abstain, abstain, dataclasses.asdict(params), scores_sha256)


def route(policy: Policy, score: float) -> str:
    if score >= policy.band_high:
        return "inspect"
    return "review" if score >= policy.band_low else "ship"


def check_policy(policy: Policy) -> None:
    trivial = min(policy.ship_all_dollars_per_shift, policy.inspect_all_dollars_per_shift)
    if policy.dollars_per_shift >= trivial:
        raise PolicyRejected(f"model policy ${policy.dollars_per_shift:,.0f}/shift is no cheaper than "
                             f"a trivial policy at ${trivial:,.0f}/shift")


def policy_json(policy: Policy) -> str:
    return json.dumps(dataclasses.asdict(policy), indent=2) + "\n"


def curve_csv(curve: CostCurve) -> str:
    rows = ["threshold,dollars_per_shift,tp,fp,tn,fn"] + [
        f"{t:.3f},{d:.2f},{a},{b},{c},{e}"
        for t, d, a, b, c, e in zip(curve.thresholds, curve.dollars_per_shift, curve.tp, curve.fp, curve.tn, curve.fn)]
    return "\n".join(rows) + "\n"


def write_outputs(policy: Policy, curve: CostCurve, out_dir: Path = POLICY_DIR) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "policy.json").write_text(policy_json(policy))
    (out_dir / "cost_curve.csv").write_text(curve_csv(curve))


def main() -> int:
    params = load_costs()
    scores = np.load(SCORES_PATH)
    y, p = scores["y"], scores["p"]
    policy = build_policy(y, p, params, sha256_file(SCORES_PATH))
    write_outputs(policy, cost_curve(y, p, params))
    print(f"{policy.version}: threshold {policy.threshold}, band [{policy.band_low}, {policy.band_high}), "
          f"${policy.dollars_per_shift:,.0f}/shift vs ship-all ${policy.ship_all_dollars_per_shift:,.0f} "
          f"and inspect-all ${policy.inspect_all_dollars_per_shift:,.0f}; "
          f"abstain {policy.abstain_share:.2%}, committed {policy.committed_share:.2%}")
    try:
        check_policy(policy)
    except PolicyRejected as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
