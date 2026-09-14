import duckdb
import pytest

from linegate.features import baseline, compute

# Parts 1-4 in train. Stations: L0_S0 (numeric F0, F2; date D1, D3), L1_S24 (numeric F1; date D2).
NUMERIC = [(1, 0, 0.5, -0.5, 2.0), (2, 1, None, None, None), (3, 0, 0.1, None, 0.3), (4, 0, None, None, None)]
DATE = [(1, 10.0, 12.0, 20.0), (2, None, None, 30.0), (3, 11.0, 11.0, None), (4, None, None, None)]


def values(rows):
    return ", ".join("(" + ", ".join("NULL" if v is None else str(v) for v in row) + ")" for row in rows)


@pytest.fixture
def con(tmp_path):
    c = duckdb.connect()
    c.execute(f'CREATE VIEW train_numeric AS SELECT col0::BIGINT AS Id, col1::TINYINT AS Response, '
              f'col2::DOUBLE AS "L0_S0_F0", col3::DOUBLE AS "L0_S0_F2", col4::DOUBLE AS "L1_S24_F1" '
              f'FROM (VALUES {values(NUMERIC)})')
    c.execute(f'CREATE VIEW train_date AS SELECT col0::BIGINT AS Id, col1::DOUBLE AS "L0_S0_D1", '
              f'col2::DOUBLE AS "L0_S0_D3", col3::DOUBLE AS "L1_S24_D2" FROM (VALUES {values(DATE)})')
    return c


def rows_by_id(con, path):
    cur = con.execute(f"SELECT * FROM read_parquet('{path}') ORDER BY Id")
    names = [d[0] for d in cur.description]
    return {r[0]: dict(zip(names, r)) for r in cur.fetchall()}


def test_station_columns_group_by_line_and_station(con):
    groups = baseline.station_columns(con, "train")
    assert list(groups) == ["L0_S0", "L1_S24"]
    assert groups["L0_S0"] == {"numeric": ["L0_S0_F0", "L0_S0_F2"], "date": ["L0_S0_D1", "L0_S0_D3"]}


def test_baseline_features_values(con, tmp_path):
    path = compute.materialize(con, "train", tmp_path)
    rows = rows_by_id(con, path)
    assert rows[1]["L0_S0_num_min"] == -0.5 and rows[1]["L0_S0_num_max"] == 0.5 and rows[1]["L0_S0_num_range"] == 1.0
    assert rows[1]["L0_S0_num_missing"] == 0 and rows[3]["L0_S0_num_missing"] == 1
    assert rows[1]["L0_S0_dwell"] == 2.0 and rows[3]["L0_S0_dwell"] == 0.0
    assert rows[1]["elapsed"] == 10.0 and rows[4]["elapsed"] is None
    assert rows[1]["route"] == "L0_S0-L1_S24" and rows[2]["route"] == "L1_S24" and rows[4]["route"] == ""
    assert "Id" in rows[1] and "Response" in rows[1]


def test_route_codes_fit_on_train_and_rare_routes_get_zero(con, tmp_path):
    train_path = compute.materialize(con, "train", tmp_path)
    compute.fit_route_codes(con, train_path, min_count=2)
    ids, y, X, names = compute.load_matrix(con, train_path)
    assert "Id" not in names and "Response" not in names and "route" not in names
    codes = dict(zip(ids.tolist(), X[:, names.index("route_code")].tolist()))
    assert codes == {1: 1.0, 2: 0.0, 3: 1.0, 4: 0.0}  # parts 1 and 3 share a route
    compute.fit_route_codes(con, train_path, min_count=1)
    _, _, X, names = compute.load_matrix(con, train_path)
    codes = dict(zip(ids.tolist(), X[:, names.index("route_code")].tolist()))
    assert codes == {1: 1.0, 3: 1.0, 4: 2.0, 2: 3.0}  # most frequent first, ties broken by route string
    assert y.tolist() == [0, 1, 0, 0]
