import pytest

from linegate.agents import sql_guard
from linegate.agents.sql_guard import SQLRejected

HOLDOUT = {900001, 900002}
EXPLORE_VIEWS = {"parts_numeric", "parts_date", "parts_categorical", "parts_labels"}


def explore(sql):
    sql_guard.check_query(sql, allowed_tables=EXPLORE_VIEWS, holdout_ids=HOLDOUT, allow_windows=True)


def feature(sql):
    sql_guard.check_query(sql, allowed_tables=EXPLORE_VIEWS - {"parts_labels"}, holdout_ids=HOLDOUT, allow_windows=False)


def test_plain_exploration_query_passes():
    explore("WITH s AS (SELECT Id, L3_S32_F3850 AS x FROM parts_numeric) "
            "SELECT round(x, 2) AS b, avg(Response) FROM s JOIN parts_labels USING (Id) GROUP BY 1")


@pytest.mark.parametrize("sql", ["DELETE FROM parts_numeric", "INSERT INTO t VALUES (1)", "DROP VIEW parts_date",
                                 "COPY parts_date TO 'x.csv'", "SELECT 1; SELECT 2", "SELEC 1"])
def test_non_select_rejected(sql):
    with pytest.raises(SQLRejected):
        explore(sql)


@pytest.mark.parametrize("sql", ["SELECT * FROM read_parquet('data/parquet/train_numeric.parquet')",
                                 "SELECT * FROM train_numeric", "SELECT * FROM catalog.parts_numeric",
                                 "SELECT * FROM parts_date JOIN test_date USING (Id)"])
def test_tables_outside_the_sandbox_rejected(sql):
    with pytest.raises(SQLRejected):
        explore(sql)


@pytest.mark.parametrize("sql", [
    "SELECT Id, lag(Id) OVER (ORDER BY Id) FROM parts_numeric",
    "SELECT Id, row_number() OVER (ORDER BY L0_S0_D1, Id) FROM parts_date",
    "SELECT Id, lead(x) OVER (ORDER BY part_id) FROM (SELECT Id AS part_id, 1 AS x FROM parts_date)",
])
def test_windows_ordered_by_id_rejected(sql):
    with pytest.raises(SQLRejected, match="Id"):
        explore(sql)


def test_features_may_not_use_any_window():
    explore("SELECT Id, rank() OVER (ORDER BY L0_S0_D1) FROM parts_date")
    with pytest.raises(SQLRejected, match="window"):
        feature("SELECT Id, rank() OVER (ORDER BY L0_S0_D1) AS r FROM parts_date")


def test_features_may_not_read_labels():
    with pytest.raises(SQLRejected):
        feature("SELECT Id, Response AS y FROM parts_labels")


def test_holdout_id_literal_rejected():
    with pytest.raises(SQLRejected, match="holdout"):
        explore("SELECT * FROM parts_numeric WHERE Id IN (5, 900002)")
    explore("SELECT * FROM parts_numeric WHERE Id IN (5, 6)")
