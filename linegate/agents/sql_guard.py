"""Parse agent SQL with DuckDB's own parser and reject anything outside the sandbox contract.

Accepted: one SELECT that reads only the allowed sandbox views and its own
CTEs. Rejected: other statements, table functions (file readers), qualified
or unknown tables, windows ordered by an Id-like column, integer literals
that are holdout part Ids, and, for feature SQL, any window function at all.
"""

from __future__ import annotations

import json
import re

import duckdb

ID_LIKE = re.compile(r"(^|_)id($|_)", re.IGNORECASE)


class SQLRejected(ValueError):
    """Raised when agent SQL violates the sandbox contract."""


def parse(sql: str) -> dict:
    try:
        raw = duckdb.execute("SELECT json_serialize_sql(?)", [sql]).fetchone()[0]
    except duckdb.Error as exc:
        raise SQLRejected(f"could not parse SQL: {exc}") from exc
    tree = json.loads(raw)
    if tree.get("error"):
        raise SQLRejected(f"only a single SELECT is allowed: {tree.get('error_message')}")
    if len(tree["statements"]) != 1:
        raise SQLRejected("exactly one statement is allowed")
    return tree["statements"][0]


def nodes(tree):
    stack = [tree]
    while stack:
        item = stack.pop()
        if isinstance(item, dict):
            yield item
            stack.extend(item.values())
        elif isinstance(item, list):
            stack.extend(item)


def cte_names(tree) -> set[str]:
    names = set()
    for node in nodes(tree):
        for entry in node.get("cte_map", {}).get("map", []) if isinstance(node.get("cte_map"), dict) else []:
            names.add(entry["key"].lower())
    return names


def references_id(expression) -> bool:
    return any(n.get("class") == "COLUMN_REF" and ID_LIKE.search(n["column_names"][-1]) for n in nodes(expression))


def check_node(node: dict, tables: set[str], holdout_ids: set[int], allow_windows: bool) -> None:
    kind = node.get("type")
    if kind == "TABLE_FUNCTION":
        raise SQLRejected("table functions are not allowed; read the sandbox views")
    if kind == "BASE_TABLE":
        if node["schema_name"] or node["catalog_name"] or node["table_name"].lower() not in tables:
            raise SQLRejected(f"table {node['table_name']!r} is outside the sandbox; allowed: {sorted(tables)}")
    if node.get("class") == "WINDOW":
        if not allow_windows:
            raise SQLRejected("window functions are not allowed in feature SQL: a feature must depend on its own part only")
        if any(references_id(order) for order in node.get("orders", [])):
            raise SQLRejected("window ordered by an Id-like column: row identity is not a feature")
    if kind == "VALUE_CONSTANT":
        value = node["value"]
        if not value.get("is_null") and isinstance(value.get("value"), int) and value["value"] in holdout_ids:
            raise SQLRejected("query references a holdout part Id")


def check_query(sql: str, allowed_tables: set[str], holdout_ids: set[int], allow_windows: bool) -> None:
    tree = parse(sql)
    tables = {t.lower() for t in allowed_tables} | cte_names(tree)
    for node in nodes(tree):
        check_node(node, tables, holdout_ids, allow_windows)
