"""Cost-minimizing threshold and the abstain band around it.

The band spans every threshold whose expected cost per shift is within the
indifference amount of the minimum. It is the hull of that set, so it always
contains the minimum and never shrinks when the indifference amount grows.
"""

from __future__ import annotations

import numpy as np

from linegate.cost.curve import CostCurve


def argmin_threshold(curve: CostCurve) -> tuple[float, float]:
    i = int(np.argmin(curve.dollars_per_shift))
    return float(curve.thresholds[i]), float(curve.dollars_per_shift[i])


def abstain_band(curve: CostCurve, indifference_usd: float) -> tuple[float, float]:
    if indifference_usd < 0:
        raise ValueError("indifference must be non-negative")
    _, best = argmin_threshold(curve)
    near = np.flatnonzero(curve.dollars_per_shift <= best + indifference_usd)
    return float(curve.thresholds[near.min()]), float(curve.thresholds[near.max()])
