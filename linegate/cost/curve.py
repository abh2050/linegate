"""Expected dollars per shift at every threshold, from the exact confusion matrix.

A part scoring at or above the threshold is flagged: it is inspected, and a
true failure found at inspection is scrapped. A part below the threshold
ships: a true failure among them becomes a field failure. Validation rates
are scaled to parts per shift. No value is interpolated between thresholds.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from linegate.dataio import CONFIG_DIR
from linegate.model.evaluate import SWEEP_THRESHOLDS, confusion_sweep


@dataclass(frozen=True)
class CostParameters:
    scrap_cost_usd: float
    field_failure_cost_usd: float
    inspection_cost_usd: float
    monthly_volume: int
    shifts_per_month: int
    abstain_band_indifference_usd: float

    @property
    def parts_per_shift(self) -> float:
        return self.monthly_volume / self.shifts_per_month


@dataclass(frozen=True)
class CostCurve:
    thresholds: np.ndarray
    dollars_per_shift: np.ndarray
    tp: np.ndarray
    fp: np.ndarray
    tn: np.ndarray
    fn: np.ndarray


def load_costs(path: Path = CONFIG_DIR / "costs.yaml") -> CostParameters:
    raw = yaml.safe_load(path.read_text())
    fields = CostParameters.__dataclass_fields__
    return CostParameters(**{name: raw[name] for name in fields})


def dollars_per_shift(tp, fp, tn, fn, params: CostParameters) -> np.ndarray:
    tp, fp, tn, fn = (np.asarray(a, dtype=np.float64) for a in (tp, fp, tn, fn))
    total = tp * (params.inspection_cost_usd + params.scrap_cost_usd) + fp * params.inspection_cost_usd \
        + fn * params.field_failure_cost_usd
    return total / (tp + fp + tn + fn) * params.parts_per_shift


def cost_curve(y: np.ndarray, p: np.ndarray, params: CostParameters,
               thresholds: np.ndarray = SWEEP_THRESHOLDS) -> CostCurve:
    sweep = confusion_sweep(y, p, thresholds)
    dollars = dollars_per_shift(sweep.tp, sweep.fp, sweep.tn, sweep.fn, params)
    return CostCurve(sweep.thresholds, dollars, sweep.tp, sweep.fp, sweep.tn, sweep.fn)
