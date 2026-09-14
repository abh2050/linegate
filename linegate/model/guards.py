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
