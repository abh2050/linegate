"""Hand-written baseline feature set, expressed as DuckDB SQL over one split.

Per station: min, max, and range of its numeric measurements, the count of
missing numeric measurements, dwell time, arrival time relative to the
part's first timestamp, and the count of categorical values present. Per
line: span and arrival. Per part: total elapsed time, station count, route
string, and integer codes of the categorical columns selected on train.
Nothing here reads Id beyond carrying it as the join key, no feature uses
absolute time, and nothing compares a part to another part.
"""

from __future__ import annotations

from collections import defaultdict

import duckdb

from linegate.dataio.schema import feature_columns, parse_column


def view_columns(con: duckdb.DuckDBPyConnection, view: str) -> list[str]:
    return [row[0] for row in con.execute(f"DESCRIBE {view}").fetchall()]


def station_key(station: str) -> tuple[int, int]:
    line, st = station.split("_")
    return int(line[1:]), int(st[1:])


def station_columns(con: duckdb.DuckDBPyConnection, split: str) -> dict[str, dict[str, list[str]]]:
    groups: dict[str, dict[str, list[str]]] = defaultdict(lambda: {"numeric": [], "date": []})
    for kind in ("numeric", "date"):
        for column in feature_columns(view_columns(con, f"{split}_{kind}")):
            ref = parse_column(column, kind)
            groups[f"{ref.line}_{ref.station}"][kind].append(column)
    return {s: groups[s] for s in sorted(groups, key=station_key)}


def quoted(columns: list[str]) -> str:
    return ", ".join(f'"{c}"' for c in columns)


def present_count(columns: list[str]) -> str:
    return " + ".join(f'("{c}" IS NOT NULL)::SMALLINT' for c in columns)


def numeric_exprs(station: str, columns: list[str]) -> list[str]:
    lo, hi = f"least({quoted(columns)})", f"greatest({quoted(columns)})"
    missing = f"{len(columns)} - ({present_count(columns)})"
    return [f"{lo} AS {station}_num_min", f"{hi} AS {station}_num_max",
            f"{hi} - {lo} AS {station}_num_range", f"({missing}) AS {station}_num_missing"]


def numeric_sql(groups: dict, split: str) -> str:
    exprs = [e for s, g in groups.items() if g["numeric"] for e in numeric_exprs(s, g["numeric"])]
    return f"SELECT Id, Response, {', '.join(exprs)} FROM {split}_numeric"


def station_date_exprs(groups: dict, start: str) -> list[str]:
    exprs = []
    for s, g in groups.items():
        if g["date"]:
            cols = quoted(g["date"])
            exprs += [f"greatest({cols}) - least({cols}) AS {s}_dwell", f"least({cols}) - {start} AS {s}_arrival"]
    return exprs


def line_date_exprs(groups: dict, start: str) -> list[str]:
    lines: dict[str, list[str]] = defaultdict(list)
    for s, g in groups.items():
        lines[s.split("_")[0]] += g["date"]
    return [e for line, cols in lines.items() for e in (
        f"greatest({quoted(cols)}) - least({quoted(cols)}) AS {line}_span",
        f"least({quoted(cols)}) - {start} AS {line}_arrival")]


def date_sql(groups: dict, split: str) -> str:
    all_dates = quoted([c for g in groups.values() for c in g["date"]])
    start = f"least({all_dates})"
    seen = " + ".join(f"(greatest({quoted(g['date'])}) IS NOT NULL)::SMALLINT" for g in groups.values() if g["date"])
    exprs = station_date_exprs(groups, start) + line_date_exprs(groups, start) + [
        f"greatest({all_dates}) - {start} AS elapsed", f"({seen}) AS n_stations", f"{start} AS start_ts"]
    return f"SELECT Id, {', '.join(exprs)} FROM {split}_date"


def categorical_stations(con: duckdb.DuckDBPyConnection, split: str) -> dict[str, list[str]]:
    stations: dict[str, list[str]] = defaultdict(list)
    for column in feature_columns(view_columns(con, f"{split}_categorical")):
        ref = parse_column(column, "categorical")
        stations[f"{ref.line}_{ref.station}"].append(column)
    return {s: stations[s] for s in sorted(stations, key=station_key)}


def select_categorical(con: duckdb.DuckDBPyConnection, min_count: int, min_distinct: int = 2) -> list[str]:
    """Categorical columns worth coding, chosen from the train split only."""
    columns = feature_columns(view_columns(con, "train_categorical"))
    counts = con.execute("SELECT " + ", ".join(f'count("{c}")' for c in columns) + " FROM train_categorical").fetchone()
    covered = [c for c, n in zip(columns, counts) if n >= min_count]
    if not covered:
        return []
    distinct = con.execute("SELECT " + ", ".join(f'count(DISTINCT "{c}")' for c in covered)
                           + " FROM train_categorical").fetchone()
    return [c for c, n in zip(covered, distinct) if n >= min_distinct]


def categorical_sql(stations: dict[str, list[str]], selected: list[str], split: str) -> str:
    codes = [f'try_cast(substr("{c}", 2) AS BIGINT) AS cat_{c}' for c in selected]
    counts = [f"({present_count(cols)}) AS {s}_cat_count" for s, cols in stations.items()]
    return f"SELECT Id, {', '.join(codes + counts)} FROM {split}_categorical"


def route_expr(groups: dict) -> str:
    parts = []
    for s, g in groups.items():
        seen = [f"n.{s}_num_max IS NOT NULL"] if g["numeric"] else []
        seen += [f"d.{s}_dwell IS NOT NULL"] if g["date"] else []
        parts.append(f"CASE WHEN {' OR '.join(seen)} THEN '{s}' END")
    return f"concat_ws('-', {', '.join(parts)}) AS route"
