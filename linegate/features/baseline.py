"""Hand-written baseline feature set, expressed as DuckDB SQL over one split.

Per station: min, max, and range of its numeric measurements, the count of
missing numeric measurements, and dwell time (last minus first date). Per
part: total elapsed time and the station route as a string. Nothing here
reads Id beyond carrying it as the join key, and nothing compares a part to
another part.
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


def numeric_exprs(station: str, columns: list[str]) -> list[str]:
    lo, hi = f"least({quoted(columns)})", f"greatest({quoted(columns)})"
    missing = " + ".join(f'("{c}" IS NULL)::SMALLINT' for c in columns)
    return [f"{lo} AS {station}_num_min", f"{hi} AS {station}_num_max",
            f"{hi} - {lo} AS {station}_num_range", f"({missing}) AS {station}_num_missing"]


def numeric_sql(groups: dict, split: str) -> str:
    exprs = [e for s, g in groups.items() if g["numeric"] for e in numeric_exprs(s, g["numeric"])]
    return f"SELECT Id, Response, {', '.join(exprs)} FROM {split}_numeric"


def date_sql(groups: dict, split: str) -> str:
    all_dates = [c for g in groups.values() for c in g["date"]]
    exprs = [f"greatest({quoted(g['date'])}) - least({quoted(g['date'])}) AS {s}_dwell"
             for s, g in groups.items() if g["date"]]
    elapsed = f"greatest({quoted(all_dates)}) - least({quoted(all_dates)}) AS elapsed"
    return f"SELECT Id, {', '.join(exprs)}, {elapsed} FROM {split}_date"


def route_expr(groups: dict) -> str:
    parts = []
    for s, g in groups.items():
        seen = [f"n.{s}_num_max IS NOT NULL"] if g["numeric"] else []
        seen += [f"d.{s}_dwell IS NOT NULL"] if g["date"] else []
        parts.append(f"CASE WHEN {' OR '.join(seen)} THEN '{s}' END")
    return f"concat_ws('-', {', '.join(parts)}) AS route"
