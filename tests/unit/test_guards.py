import pytest

from linegate.model import guards


def test_mcc_above_tripwire_raises_leakage():
    with pytest.raises(guards.LeakageSuspected):
        guards.check_validation_mcc(0.41, low=0.12, high=0.40)


def test_mcc_below_floor_raises_broken_pipeline():
    with pytest.raises(guards.PipelineBroken):
        guards.check_validation_mcc(0.05, low=0.12, high=0.40)


def test_mcc_in_band_passes():
    guards.check_validation_mcc(0.22, low=0.12, high=0.40)


@pytest.mark.parametrize("name", ["Id", "row_index", "file_order", "prev_id_diff", "L0_S0_id"])
def test_identity_like_feature_names_rejected(name):
    with pytest.raises(guards.LeakageSuspected):
        guards.check_feature_names(["L0_S0_dwell", name])


def test_station_feature_names_accepted():
    guards.check_feature_names(["L0_S0_dwell", "L3_S29_num_range", "route_code", "elapsed"])
