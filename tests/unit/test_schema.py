import pytest

from linegate.dataio import schema
from linegate.dataio.schema import SchemaError


def test_parse_column_extracts_line_and_station():
    ref = schema.parse_column("L3_S38_F3956", "numeric")
    assert (ref.line, ref.station) == ("L3", "S38")


@pytest.mark.parametrize("name,kind", [("L0_S0_D1", "numeric"), ("L0_S0_F0", "date"), ("row_index", "numeric")])
def test_parse_column_rejects_wrong_kind_or_shape(name, kind):
    with pytest.raises(SchemaError):
        schema.parse_column(name, kind)


def test_duck_types():
    assert schema.duck_type("Id", "numeric") == "BIGINT"
    assert schema.duck_type("Response", "numeric") == "TINYINT"
    assert schema.duck_type("L0_S0_F0", "numeric") == "DOUBLE"
    assert schema.duck_type("L0_S0_D1", "date") == "DOUBLE"
    assert schema.duck_type("L0_S1_F25", "categorical") == "VARCHAR"


def test_station_map_groups_all_kinds():
    result = schema.station_map({
        "numeric": ["L0_S0_F0", "L0_S0_F2", "L1_S24_F1"],
        "date": ["L0_S0_D1"],
        "categorical": ["L0_S1_F25"],
    })
    assert result == {
        "L0": {"S0": ["L0_S0_F0", "L0_S0_F2", "L0_S0_D1"], "S1": ["L0_S1_F25"]},
        "L1": {"S24": ["L1_S24_F1"]},
    }


def make_pair(tmp_path, kind, train_cols, test_cols):
    for source, cols in (("train", train_cols), ("test", test_cols)):
        select = ", ".join(f'1 AS "{c}"' for c in cols)
        path = schema.parquet_path(tmp_path, source, kind)
        __import__("duckdb").sql(f"COPY (SELECT {select}) TO '{path}' (FORMAT parquet)")


def test_check_kind_rejects_wrong_count(tmp_path):
    make_pair(tmp_path, "date", ["Id", "L0_S0_D1"], ["Id", "L0_S0_D1"])
    with pytest.raises(SchemaError, match="expected 1156"):
        schema.check_kind(tmp_path, "date")


def test_check_kind_rejects_train_test_mismatch(tmp_path, monkeypatch):
    monkeypatch.setitem(schema.EXPECTED_FEATURES, "date", 1)
    make_pair(tmp_path, "date", ["Id", "L0_S0_D1"], ["Id", "L0_S0_D3"])
    with pytest.raises(SchemaError, match="differ"):
        schema.check_kind(tmp_path, "date")


def test_check_kind_accepts_matching_contract(tmp_path, monkeypatch):
    monkeypatch.setitem(schema.EXPECTED_FEATURES, "numeric", 2)
    make_pair(tmp_path, "numeric", ["Id", "L0_S0_F0", "L0_S0_F2", "Response"], ["Id", "L0_S0_F0", "L0_S0_F2"])
    assert schema.check_kind(tmp_path, "numeric") == 2
