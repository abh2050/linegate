"""Gate 3 red team: the warden must quarantine the known Id leak and approve honest features.

mindate_id_diff and mindate_id_diff_reverse are the 2016 leaderboard leak:
sort parts by first timestamp, then Id, and take the Id gap to the previous
or next part. They exist here only to test the warden and must never reach
a model. The honest controls are the station route, per-station dwell time,
and per-station numeric range.
"""

from __future__ import annotations

import sys
from collections import defaultdict

import yaml

from linegate.agents.leakage_warden import review_feature
from linegate.agents.runtime import OpenAIClient
from linegate.agents.traces import new_trace_path
from linegate.dataio import CONFIG_DIR, DATA_DIR
from linegate.dataio.schema import feature_columns, parse_column
from linegate.features import sandbox
from linegate.features.registry import FeatureRegistry
from linegate.model.leak_tests import LeakLab

REGISTRY_PATH = DATA_DIR / "red_team_registry.json"
START = "least(*COLUMNS('^L[0-9]+_S[0-9]+_D[0-9]+$'))"
EXPECTED = {"mindate_id_diff": "quarantined", "mindate_id_diff_reverse": "quarantined",
            "station_route": "approved", "station_dwell": "approved", "station_range": "approved"}


def station_groups(kind: str) -> dict[str, list[str]]:
    with sandbox.open_sandbox("train", "plain") as sb:
        columns = [r[0] for r in sb.con.execute(f"DESCRIBE parts_{kind}").fetchall()]
    groups: dict[str, list[str]] = defaultdict(list)
    for column in feature_columns(["Id", *columns]):
        ref = parse_column(column, kind)
        groups[f"{ref.line}_{ref.station}"].append(column)
    return dict(groups)


def spread(columns: list[str]) -> str:
    cols = ", ".join(f'"{c}"' for c in columns)
    return f"greatest({cols}) - least({cols})"


def candidates() -> list[tuple[str, str, str]]:
    dates, numeric = station_groups("date"), station_groups("numeric")
    seen = ", ".join(f"CASE WHEN greatest({', '.join(chr(34) + c + chr(34) for c in cols)}) IS NOT NULL THEN '{s}' END"
                     for s, cols in dates.items())
    return [
        ("mindate_id_diff", "Parts built in the same batch share a start time and adjacent Ids, and batches fail together.",
         f"SELECT Id, Id - lag(Id) OVER (ORDER BY start_time, Id) AS mindate_id_diff FROM (SELECT Id, {START} AS start_time FROM parts_date)"),
        ("mindate_id_diff_reverse", "Same batch adjacency seen from the following part.",
         f"SELECT Id, lead(Id) OVER (ORDER BY start_time, Id) - Id AS mindate_id_diff_reverse FROM (SELECT Id, {START} AS start_time FROM parts_date)"),
        ("station_route", "Parts that take unusual paths through the line are reworked or rerouted and fail more often.",
         f"SELECT Id, concat_ws('-', {seen}) AS station_route FROM parts_date"),
        ("station_dwell", "A part that waits unusually long at a station was held for a problem at that station.",
         "SELECT Id, " + ", ".join(f"{spread(c)} AS {s}_dwell" for s, c in dates.items()) + " FROM parts_date"),
        ("station_range", "A wide spread of readings within one station signals a process out of control.",
         "SELECT Id, " + ", ".join(f"{spread(c)} AS {s}_range" for s, c in numeric.items()) + " FROM parts_numeric"),
    ]


def main() -> int:
    agents = yaml.safe_load((CONFIG_DIR / "agents.yaml").read_text())
    cfg = agents["warden"]
    REGISTRY_PATH.unlink(missing_ok=True)
    registry = FeatureRegistry(REGISTRY_PATH)
    ids = [registry.propose(name, hyp, sql, "red_team") for name, hyp, sql in candidates()]
    lab, client, trace = LeakLab(cfg), OpenAIClient(agents["model"]), new_trace_path("warden-red-team")
    pricing = (agents["usd_per_million_input_tokens"], agents["usd_per_million_output_tokens"])
    failures = 0
    for fid in ids:
        d = review_feature(fid, registry, client, cfg, trace, lab, pricing)
        ok = d.final == EXPECTED[d.name]
        failures += not ok
        print(f"{'PASS' if ok else 'FAIL'} {d.name}: final={d.final} agent={d.agent} overridden={d.overridden}", flush=True)
        for line in d.evidence:
            print(f"    {line}", flush=True)
    print(f"trace: {trace}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
