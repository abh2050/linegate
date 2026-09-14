# linegate

Predict quality failures on the Bosch production line, price the decision in
dollars, and route the uncertain parts to a human.

## Stack

Python 3.12. uv for dependency management. DuckDB over Parquet for all
queries. Polars for frames. LightGBM for the classifier. Pydantic for tool
schemas. Anthropic SDK for agents. FastAPI plus React for the console.
pytest and Playwright for tests.

## Non negotiables

1. The classifier is LightGBM. No LLM scores a part. No agent sits in the
   inference path.
2. The holdout split is sealed. Do not read it, do not evaluate against it,
   do not compute statistics over it. `scripts/seal_holdout.py` runs once.
   `data/holdout_runs.json` must read 1 when the project ends.
3. Never construct a feature from row Id, row order, file order, or any
   comparison between one part and its neighbours in the raw ordering.
   `scripts/red_team_leak.py` contains the known leak and exists only to
   test the warden.
4. Never join train and test tables in any query.
5. Every raw file load verifies its sha256 against `data/manifest.json`.
   No code path substitutes generated or sampled data for the real set.
6. The leakage warden holds veto power over the feature search agent. The
   search agent cannot override a quarantine.
7. Agents write a hypothesis in plain language before they write a query.
   Both go into the trace.
8. If validation MCC exceeds 0.40, stop and raise `LeakageSuspected`.
   Honest models on this dataset land between 0.15 and 0.30. The 2016
   leaderboard reached 0.49 only through the Id ordering leak.
9. Adding a dependency or a framework requires an ADR in docs/decisions.
10. Run `make gate-N` before you claim milestone N is finished. A milestone
    is finished when its gate command exits zero.

## Dataset facts

Four raw files. 1,183,747 parts. 968 numeric columns, 1,156 date columns,
2,140 categorical columns. Positive rate is roughly 0.58 percent. Date
values are anonymized relative time in units of 0.01 weeks, not calendar
dates, and they still order correctly. Most parts touch a subset of
stations, so the null pattern encodes the route.

## Working style

Write the test before the implementation for anything in model/, cost/, and
agents/. Keep functions under 40 lines. Do not add a config option without
a caller. Commit after every passing gate.
