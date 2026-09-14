"""Reject any agent disposition that cites a column or part the tools did not return.

A column is citable only if it is in the dataset schema and appeared in a
get_part result during the same agent session. A part Id is citable only if
a tool returned it. Every column-shaped token in free text is checked too,
and each out-of-range entry must match the part record's actual reading.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

COLUMN_TOKEN = re.compile(r"\bL\d+_S\d+_[FD]\d+\b")
RECOMMENDATIONS = {"scrap", "ship", "senior_review"}
TEXT_FIELDS = ("route", "reason", "neighbor_summary")
REQUIRED = ("route", "recommendation", "reason", "out_of_range", "neighbor_summary", "citations")


@dataclass
class ValidationResult:
    valid: bool
    errors: list[str] = field(default_factory=list)


def column_errors(column: str, schema_columns: set, seen_columns: set, where: str) -> list[str]:
    if column not in schema_columns:
        return [f"{where}: column {column} is not in the schema"]
    if column not in seen_columns:
        return [f"{where}: column {column} was not returned by get_part"]
    return []


def validate_citations(citations: list[dict], schema_columns: set, seen_columns: set, seen_parts: set) -> list[str]:
    errors = []
    for citation in citations:
        kind, value = citation.get("kind"), str(citation.get("value", ""))
        if kind == "column":
            errors += column_errors(value, schema_columns, seen_columns, "citation")
        elif kind == "part":
            if not value.isdigit() or int(value) not in seen_parts:
                errors.append(f"citation: part {value} was not returned by a tool")
        else:
            errors.append(f"citation: unknown kind {kind!r}")
    return errors


def close(a, b) -> bool:
    try:
        return math.isclose(float(a), float(b), rel_tol=1e-4, abs_tol=1e-4)
    except (TypeError, ValueError):
        return False


def out_of_range_errors(entries: list[dict], record: dict) -> list[str]:
    actual = {m["column"]: m for m in record.get("out_of_range", [])}
    errors = []
    for entry in entries:
        truth = actual.get(entry.get("column"))
        if truth is None:
            errors.append(f"out_of_range: {entry.get('column')} is not out of range on this part")
        elif not all(close(entry.get(k), truth[k]) for k in ("value", "low", "high")):
            errors.append(f"out_of_range: {entry.get('column')} values do not match the part record")
    return errors


def validate_disposition(disposition: dict, schema_columns: set, seen_columns: set, seen_parts: set,
                         record: dict) -> ValidationResult:
    missing = [k for k in REQUIRED if k not in disposition]
    if missing:
        return ValidationResult(False, [f"missing fields: {missing}"])
    errors = []
    if disposition["recommendation"] not in RECOMMENDATIONS:
        errors.append(f"recommendation {disposition['recommendation']!r} is not one of {sorted(RECOMMENDATIONS)}")
    errors += validate_citations(disposition["citations"], schema_columns, seen_columns, seen_parts)
    errors += out_of_range_errors(disposition["out_of_range"], record)
    for name in TEXT_FIELDS:
        for token in sorted(set(COLUMN_TOKEN.findall(str(disposition[name])))):
            errors += column_errors(token, schema_columns, seen_columns, name)
    return ValidationResult(not errors, errors)
