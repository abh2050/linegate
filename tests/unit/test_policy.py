import dataclasses

import numpy as np
import pytest

from linegate.cost import curve, policy
from tests.unit.test_cost_curve import PARAMS


@pytest.fixture
def scores():
    rng = np.random.default_rng(11)
    y = (rng.random(20000) < 0.01).astype(np.int8)
    p = np.clip(0.3 * y + 0.2 * rng.random(20000) ** 4, 0, 1)
    return y, p


def test_policy_version_is_stable_and_tracks_inputs(scores):
    y, p = scores
    a = policy.build_policy(y, p, PARAMS, scores_sha256="abc")
    b = policy.build_policy(y, p, PARAMS, scores_sha256="abc")
    c = policy.build_policy(y, p, dataclasses.replace(PARAMS, field_failure_cost_usd=900.0), scores_sha256="abc")
    d = policy.build_policy(y, p, PARAMS, scores_sha256="def")
    assert a.version == b.version
    assert len({a.version, c.version, d.version}) == 3


def test_route_ship_review_inspect(scores):
    y, p = scores
    pol = policy.build_policy(y, p, PARAMS, scores_sha256="abc")
    assert pol.band_low <= pol.threshold <= pol.band_high
    assert policy.route(pol, pol.band_low - 1e-6) == "ship"
    assert policy.route(pol, pol.band_high) == "inspect"
    if pol.band_high > pol.band_low:
        assert policy.route(pol, (pol.band_low + pol.band_high) / 2) == "review"


def test_policy_reports_trivial_baselines_and_shares(scores):
    y, p = scores
    pol = policy.build_policy(y, p, PARAMS, scores_sha256="abc")
    assert pol.dollars_per_shift <= min(pol.ship_all_dollars_per_shift, pol.inspect_all_dollars_per_shift)
    assert np.isclose(pol.abstain_share + pol.committed_share, 1.0)


def test_policy_that_cannot_beat_trivial_baselines_is_rejected():
    y = np.array([0, 1] * 500)
    p = np.full(1000, 0.5)
    with pytest.raises(policy.PolicyRejected):
        policy.check_policy(policy.build_policy(y, p, PARAMS, scores_sha256="abc"))
