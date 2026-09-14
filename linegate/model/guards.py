"""Leak tripwires: identity-like feature names and implausible validation MCC."""

from __future__ import annotations

import re

IDENTITY_NAME = re.compile(r"(^|_)(id|row|index|order)(_|$)", re.IGNORECASE)


class LeakageSuspected(RuntimeError):
    """Raised when a result is too good to be honest or a feature smells of row identity."""


class PipelineBroken(RuntimeError):
    """Raised when a result is too weak for a working pipeline."""


def check_feature_names(names: list[str]) -> None:
    bad = [n for n in names if IDENTITY_NAME.search(n)]
    if bad:
        raise LeakageSuspected(f"identity-like feature names: {bad}")


def check_validation_mcc(mcc: float, low: float, high: float) -> None:
    if mcc > high:
        raise LeakageSuspected(f"validation MCC {mcc:.4f} exceeds tripwire {high}")
    if mcc < low:
        raise PipelineBroken(f"validation MCC {mcc:.4f} below floor {low}")


def warden_verdict(shuffle, refit, scope, cfg: dict) -> tuple[str, list[str]]:
    """Deterministic leak rule. Returns the decision and the test results that drove it."""
    failures = []
    if scope.mismatched > 0:
        failures.append(f"row scope: {scope.mismatch_share:.1%} of {scope.rows_compared:,} part values changed "
                        "when other parts were removed")
    if shuffle.collapse >= cfg["id_shuffle_collapse_threshold"]:
        failures.append(f"Id shuffle: lift fell from {shuffle.lift:.4f} to {shuffle.shuffled_lift:.4f} "
                        f"(collapse {shuffle.collapse:.0%} >= {cfg['id_shuffle_collapse_threshold']:.0%})")
    if refit.retention < cfg["strict_split_min_retention"]:
        failures.append(f"strict time refit: kept {refit.retention:.0%} of random-split lift "
                        f"(< {cfg['strict_split_min_retention']:.0%})")
    if failures:
        return "quarantine", failures
    return "approve", [
        f"row scope: 0 of {scope.rows_compared:,} values changed",
        f"Id shuffle: lift {shuffle.lift:.4f} -> {shuffle.shuffled_lift:.4f} (collapse {shuffle.collapse:.0%})",
        f"strict time refit: retention {refit.retention:.0%} (time lift {refit.time_lift:.4f}, "
        f"random lift {refit.random_lift:.4f})",
    ]
