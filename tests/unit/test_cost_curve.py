import numpy as np

from linegate.cost import curve
from linegate.cost.curve import CostParameters

PARAMS = CostParameters(scrap_cost_usd=42.0, field_failure_cost_usd=1850.0, inspection_cost_usd=6.5,
                        monthly_volume=240000, shifts_per_month=90, abstain_band_indifference_usd=1500.0)


def test_repo_costs_load():
    assert curve.load_costs() == PARAMS


def brute_force_dollars(y, p, t, params):
    flagged = p >= t
    tp, fp, fn = (flagged & (y == 1)).sum(), (flagged & (y == 0)).sum(), (~flagged & (y == 1)).sum()
    per_part = (tp * (params.inspection_cost_usd + params.scrap_cost_usd) + fp * params.inspection_cost_usd
                + fn * params.field_failure_cost_usd) / len(y)
    return per_part * params.monthly_volume / params.shifts_per_month


def test_dollars_come_from_confusion_matrix_at_every_threshold():
    rng = np.random.default_rng(7)
    y = (rng.random(5000) < 0.02).astype(np.int8)
    p = np.clip(0.5 * y + 0.7 * rng.random(5000) ** 3, 0, 1)
    c = curve.cost_curve(y, p, PARAMS)
    for i, t in enumerate(c.thresholds):
        assert np.isclose(c.dollars_per_shift[i], brute_force_dollars(y, p, t, PARAMS))


def test_curve_is_not_interpolated_between_endpoints():
    rng = np.random.default_rng(7)
    y = (rng.random(5000) < 0.02).astype(np.int8)
    p = np.clip(0.5 * y + 0.7 * rng.random(5000) ** 3, 0, 1)
    c = curve.cost_curve(y, p, PARAMS)
    line = np.interp(c.thresholds, [0.0, 1.0], [c.dollars_per_shift[0], c.dollars_per_shift[-1]])
    assert np.max(np.abs(c.dollars_per_shift - line)) > 0.1 * c.dollars_per_shift.max()
    assert (c.tp + c.fp + c.tn + c.fn == len(y)).all()


def test_endpoints_are_inspect_everything_and_ship_everything():
    y = np.array([0, 0, 0, 1])
    p = np.array([0.1, 0.2, 0.3, 0.4])
    c = curve.cost_curve(y, p, PARAMS, thresholds=np.array([0.0, 1.0]))
    per_shift = PARAMS.monthly_volume / PARAMS.shifts_per_month / 4
    assert np.isclose(c.dollars_per_shift[0], (3 * 6.5 + (6.5 + 42.0)) * per_shift)
    assert np.isclose(c.dollars_per_shift[1], 1850.0 * per_shift)
