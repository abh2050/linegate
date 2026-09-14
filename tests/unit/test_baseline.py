import duckdb
import pytest

from linegate.features import baseline, compute

# Parts 1-4 in train. Stations: L0_S0 (numeric F0, F2; date D1, D3; categorical F4), L1_S24 (numeric F1; date D2).
NUMERIC = [(1, 0, 0.5, -0.5, 2.0), (2, 1, None, None, None), (3, 0, 0.1, None, 0.3), (4, 0, None, None, None)]
DATE = [(1, 10.0, 12.0, 20.0), (2, None, None, 30.0), (3, 11.0, 11.0, None), (4, None, None, None)]
CATEGORICAL = [(1, "'T8'", "'T1'"), (2, "NULL", "'T1'"), (3, "'T16'", "NULL"), (4, "NULL", "NULL")]


def values(rows):
    return ", ".join("(" + ", ".join("NULL" if v is None else str(v) for v in row) + ")" for row in rows)


@pytest.fixture
def con():
    c = duckdb.connect()
    c.execute(f'CREATE VIEW train_numeric AS SELECT col0::BIGINT AS Id, col1::TINYINT AS Response, '
              f'col2::DOUBLE AS "L0_S0_F0", col3::DOUBLE AS "L0_S0_F2", col4::DOUBLE AS "L1_S24_F1" '
              f'FROM (VALUES {values(NUMERIC)})')
    c.execute(f'CREATE VIEW train_date AS SELECT col0::BIGINT AS Id, col1::DOUBLE AS "L0_S0_D1", '
              f'col2::DOUBLE AS "L0_S0_D3", col3::DOUBLE AS "L1_S24_D2" FROM (VALUES {values(DATE)})')
    c.execute(f'CREATE VIEW train_categorical AS SELECT col0::BIGINT AS Id, col1::VARCHAR AS "L0_S0_F4", '
              f'col2::VARCHAR AS "L1_S24_F5" FROM (VALUES {values(CATEGORICAL)})')
    return c


def rows_by_id(con, path):
    cur = con.execute(f"SELECT * FROM read_parquet('{path}') ORDER BY Id")
    names = [d[0] for d in cur.description]
    return {r[0]: dict(zip(names, r)) for r in cur.fetchall()}


def test_station_columns_group_by_line_and_station(con):
    groups = baseline.station_columns(con, "train")
    assert list(groups) == ["L0_S0", "L1_S24"]
    assert groups["L0_S0"] == {"numeric": ["L0_S0_F0", "L0_S0_F2"], "date": ["L0_S0_D1", "L0_S0_D3"]}


def test_categorical_selection_needs_coverage_and_variety(con):
    # L0_S0_F4 has 2 non-null parts with 2 distinct values; L1_S24_F5 is constant.
    assert baseline.select_categorical(con, min_count=2) == ["L0_S0_F4"]
    assert baseline.select_categorical(con, min_count=3) == []


def test_feature_values(con, tmp_path):
    rows = rows_by_id(con, compute.materialize(con, "train", tmp_path, ["L0_S0_F4"]))
    assert rows[1]["L0_S0_num_min"] == -0.5 and rows[1]["L0_S0_num_max"] == 0.5 and rows[1]["L0_S0_num_range"] == 1.0
    assert rows[1]["L0_S0_num_missing"] == 0 and rows[3]["L0_S0_num_missing"] == 1
    assert rows[1]["L0_S0_dwell"] == 2.0 and rows[3]["L0_S0_dwell"] == 0.0
    assert rows[1]["L1_S24_arrival"] == 10.0 and rows[1]["L0_S0_arrival"] == 0.0
    assert rows[1]["L1_span"] == 0.0 and rows[1]["L0_arrival"] == 0.0 and rows[1]["L1_arrival"] == 10.0
    assert rows[1]["elapsed"] == 10.0 and rows[4]["elapsed"] is None
    assert rows[1]["n_stations"] == 2 and rows[2]["n_stations"] == 1 and rows[4]["n_stations"] == 0
    assert rows[1]["start_ts"] == 10.0 and rows[2]["start_ts"] == 30.0
    assert rows[1]["cat_L0_S0_F4"] == 8 and rows[3]["cat_L0_S0_F4"] == 16 and rows[2]["cat_L0_S0_F4"] is None
    assert rows[1]["L0_S0_cat_count"] == 1 and rows[1]["L1_S24_cat_count"] == 1 and rows[4]["L1_S24_cat_count"] == 0
    assert rows[1]["route"] == "L0_S0-L1_S24" and rows[2]["route"] == "L1_S24" and rows[4]["route"] == ""


def test_matrix_excludes_keys_and_route_codes_fit_on_train(con, tmp_path):
    path = compute.materialize(con, "train", tmp_path, ["L0_S0_F4"])
    compute.fit_route_codes(con, path, min_count=2)
    m = compute.load_matrix(con, path)
    assert not {"Id", "Response", "route", "start_ts"} & set(m.names)
    codes = dict(zip(m.ids.tolist(), m.X[:, m.names.index("route_code")].tolist()))
    assert codes == {1: 1.0, 2: 0.0, 3: 1.0, 4: 0.0}  # parts 1 and 3 share a route
    assert m.y.tolist() == [0, 1, 0, 0]
    assert m.start[0] == 10.0 and m.start[3] != m.start[3]  # part 4 has no dates: NaN
    compute.fit_route_codes(con, path, min_count=1)
    m = compute.load_matrix(con, path)
    codes = dict(zip(m.ids.tolist(), m.X[:, m.names.index("route_code")].tolist()))
    assert codes == {1: 1.0, 3: 1.0, 4: 2.0, 2: 3.0}  # most frequent first, ties broken by route string
