import pytest

from linegate.features.registry import FeatureRegistry, QuarantineFinal, RegistryError

SQL = "SELECT Id, L0_S0_F0 AS x FROM parts_numeric"


def test_propose_requires_hypothesis_and_is_idempotent(tmp_path):
    reg = FeatureRegistry(tmp_path / "reg.json")
    with pytest.raises(RegistryError, match="hypothesis"):
        reg.propose("x", "  ", SQL, "hypothesis_agent")
    fid = reg.propose("x", "Parts with high F0 fail.", SQL, "hypothesis_agent")
    assert fid.startswith("f_") and reg.propose("x", "Parts with high F0 fail.", SQL, "hypothesis_agent") == fid
    assert FeatureRegistry(tmp_path / "reg.json").get(fid).status == "proposed"


def test_only_warden_decides_and_quarantine_is_final(tmp_path):
    reg = FeatureRegistry(tmp_path / "reg.json")
    fid = reg.propose("x", "Parts with high F0 fail.", SQL, "hypothesis_agent")
    with pytest.raises(RegistryError, match="warden"):
        reg.decide(fid, "approved", "looks fine to me", actor="hypothesis_agent")
    reg.decide(fid, "quarantined", "lift collapsed under Id shuffle", actor="warden")
    with pytest.raises(QuarantineFinal):
        reg.decide(fid, "approved", "second opinion", actor="warden")
    record = FeatureRegistry(tmp_path / "reg.json").get(fid)
    assert record.status == "quarantined" and record.findings[0]["actor"] == "warden"
