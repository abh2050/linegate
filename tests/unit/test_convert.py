import duckdb
import pytest

from linegate.dataio import convert
from linegate.dataio.convert import ConversionError
from tests.unit.conftest import write_zip

HEADER = ["Id", "L0_S1_F25", "L0_S1_F27"]
ROWS = [[4, None, "T1"], [7, "T48576", None], [11, None, None]]


def test_count_csv_rows_with_and_without_trailing_newline(tmp_path):
    with_nl = write_zip(tmp_path, "a.csv.zip", HEADER, ROWS)
    (tmp_path / "b").mkdir()
    without_nl = write_zip(tmp_path / "b", "a.csv.zip", HEADER, ROWS, trailing_newline=False)
    assert convert.count_csv_rows(with_nl) == 3
    assert convert.count_csv_rows(without_nl) == 3


def test_convert_one_round_trips_rows_and_types(tmp_path):
    zip_path = write_zip(tmp_path, "train_categorical.csv.zip", HEADER, ROWS)
    entry = convert.convert_one(zip_path, tmp_path, tmp_path / "work")
    assert entry["csv_rows"] == entry["parquet_rows"] == 3
    out = tmp_path / "train_categorical.parquet"
    types = dict(duckdb.sql(f"DESCRIBE SELECT * FROM read_parquet('{out}')").fetchall()[i][:2] for i in range(3))
    assert types == {"Id": "BIGINT", "L0_S1_F25": "VARCHAR", "L0_S1_F27": "VARCHAR"}
    rows = duckdb.sql(f"SELECT * FROM read_parquet('{out}') ORDER BY Id").fetchall()
    assert rows == [(4, None, "T1"), (7, "T48576", None), (11, None, None)]
    assert not list((tmp_path / "work").iterdir())


def test_convert_one_rejects_row_count_mismatch(tmp_path, monkeypatch):
    zip_path = write_zip(tmp_path, "train_categorical.csv.zip", HEADER, ROWS)
    monkeypatch.setattr(convert, "count_csv_rows", lambda _: 4)
    with pytest.raises(ConversionError, match="csv rows 4 != parquet rows 3"):
        convert.convert_one(zip_path, tmp_path, tmp_path / "work")
    assert not (tmp_path / "train_categorical.parquet").exists()


def test_convert_rejects_unparseable_column(tmp_path):
    zip_path = write_zip(tmp_path, "train_numeric.csv.zip", ["Id", "row_order"], [[1, 2]])
    with pytest.raises(Exception, match="row_order"):
        convert.convert_one(zip_path, tmp_path, tmp_path / "work")


def test_convert_all_requires_manifest(tmp_path):
    from linegate.dataio.fetch import ManifestError
    with pytest.raises(ManifestError):
        convert.convert_all(tmp_path, tmp_path, tmp_path / "manifest.json")
