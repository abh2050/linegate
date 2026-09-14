import numpy as np

from linegate.cost import thresholds
from linegate.cost.curve import CostCurve


def make_curve(dollars):
    n = len(dollars)
    zeros = np.zeros(n, dtype=int)
    return CostCurve(np.linspace(0, 1, n), np.asarray(dollars, dtype=float), zeros, zeros, zeros, zeros)


def test_argmin_returns_cheapest_threshold():
    c = make_curve([900, 500, 300, 450, 800])
    assert thresholds.argmin_threshold(c) == (0.5, 300.0)


def test_band_contains_argmin_and_zero_indifference_collapses():
    c = make_curve([900, 500, 300, 450, 800])
    assert thresholds.abstain_band(c, 0.0) == (0.5, 0.5)
    lo, hi = thresholds.abstain_band(c, 200.0)
    assert lo <= 0.5 <= hi and (lo, hi) == (0.25, 0.75)


def test_band_widens_when_indifference_grows():
    rng = np.random.default_rng(0)
    dollars = 1000 + 50_000 * (np.linspace(0, 1, 1001) - 0.3) ** 2 + rng.normal(0, 5, 1001)
    c = make_curve(dollars)
    widths = []
    for indifference in (100, 500, 1500, 5000):
        lo, hi = thresholds.abstain_band(c, indifference)
        widths.append(hi - lo)
    assert all(b > a for a, b in zip(widths, widths[1:]))


def test_band_is_hull_of_near_optimal_thresholds():
    c = make_curve([300, 900, 310, 900, 305])
    assert thresholds.abstain_band(c, 20.0) == (0.0, 1.0)
