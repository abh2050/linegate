"""Hypothesis agent: searches for features, gated by the warden, scored by the seed ensemble (Gate 4).

Proposals run through the SQL guard and then the warden before anything else
sees them. Evaluate adds approved features to the current best set and
retrains the three-seed ensemble; the set grows only when both validation
MCC and average precision beat the current best by more than the noise
measured from placebo runs (random columns added to the baseline). The runtime
stops the search after the configured run of evaluated proposals that fail
to clear the bar, or when evaluate calls run out. Quarantines cost budget
but do not count toward the stop rule: they never reached the delta test.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import duckdb
import numpy as np
import yaml

from linegate.agents.leakage_warden import review_feature
from linegate.agents.runtime import Budget, OpenAIClient, Tool, ToolRefused, run_agent
from linegate.agents.sql_guard import SQLRejected, check_query
from linegate.agents.tools import EmptyInput, EvaluateInput, ProposeInput, RunSQLInput, SchemaInput
from linegate.agents.traces import TraceWriter, new_trace_path
from linegate.dataio import CONFIG_DIR, DATA_DIR, PARQUET_DIR, ROOT
from linegate.dataio.schema import feature_columns, parse_column, station_map
from linegate.dataio.splits import SPLITS_NAME
from linegate.features import sandbox
from linegate.features.registry import FeatureRegistry, RegistryError
from linegate.model import evaluate as ev
from linegate.model import guards, train
from linegate.model.leak_tests import LeakLab, encode

PROMPT = (ROOT / "linegate" / "agents" / "prompts" / "hypothesis.md").read_text()
FEATURE_VIEWS = {"parts_numeric", "parts_date", "parts_categorical"}
EXPLORE_VIEWS = FEATURE_VIEWS | {"parts_labels"}
RUNS_PATH = DATA_DIR / "holdout_runs.json"
SEARCH_DIR = DATA_DIR / "artifacts" / "search"
MAX_TURNS = 80
MAX_NEW_COLUMNS = 300
MAX_RESULT_CHARS = 30_000


class SearchSession:
    def __init__(self, registry, reviewer, scorer, reference: dict, cfg: dict, trace: TraceWriter, holdout_ids: set,
                 dry_run=lambda sql: None):
        self.registry, self.reviewer, self.scorer, self.cfg, self.trace = registry, reviewer, scorer, cfg, trace
        self.dry_run = dry_run
        self.holdout_ids = holdout_ids
        self.reference = reference
        self.baseline_mcc = self.best_mcc = reference["mcc"]
        self.best_ap = reference["ap"]
        self.accepted: list[str] = []
        self.improving: list[str] = []
        self.proposed: dict[str, str] = {}
        self.consecutive_failures = self.evaluations = self.holdout_refs = 0

    def propose(self, args: ProposeInput) -> dict:
        try:
            check_query(args.sql, FEATURE_VIEWS, self.holdout_ids, allow_windows=False)
            self.dry_run(args.sql)
            fid = self.registry.propose(args.name, args.hypothesis, args.sql, "hypothesis_agent")
        except (SQLRejected, RegistryError, sandbox.FeatureSQLError) as exc:
            raise ToolRefused(str(exc)) from exc
        record = self.registry.get(fid)
        if record.status == "proposed":
            decision = self.reviewer(fid)
            evidence = decision.evidence
        else:
            evidence = [record.findings[-1]["finding"]]
        status = self.registry.get(fid).status
        self.proposed[fid] = status
        self.trace.write("proposal", feature_id=fid, name=args.name, hypothesis=args.hypothesis, sql=args.sql,
                         status=status, evidence=evidence)
        return {"feature_id": fid, "status": status, "warden_evidence": evidence}

    def evaluable(self, fids: list[str]) -> list:
        records = []
        for fid in fids:
            try:
                record = self.registry.get(fid)
            except RegistryError as exc:
                raise ToolRefused(f"unknown feature {fid}") from exc
            if record.status != "approved":
                raise ToolRefused(f"{fid} is {record.status}; only warden-approved features can be evaluated")
            records.append(record)
        return records

    def evaluate(self, args: EvaluateInput) -> dict:
        if self.evaluations >= self.cfg["max_evaluate_calls"]:
            raise ToolRefused("evaluate budget exhausted")
        new = [r for r in self.evaluable(args.feature_ids) if r.feature_id not in self.accepted]
        if not new:
            raise ToolRefused("all of these features are already in the accepted set")
        result = self.scorer.score([self.registry.get(f) for f in self.accepted] + new)
        self.evaluations += 1
        self.holdout_refs += result["holdout_ids_referenced"]
        return self.record_outcome([r.feature_id for r in new], [r.name for r in new], result)

    def bars(self) -> tuple[float, float]:
        k = self.cfg["noise_sd_multiplier"]
        mcc_bar = self.best_mcc + max(self.cfg["min_delta_mcc"], k * self.reference["sd_mcc"])
        return mcc_bar, self.best_ap + k * self.reference["sd_ap"]

    def record_outcome(self, fids: list[str], names: list[str], result: dict) -> dict:
        mcc_bar, ap_bar = self.bars()
        improved = result["mcc"] >= mcc_bar and result["ap"] >= ap_bar
        before = (self.best_mcc, self.best_ap)
        if improved:
            self.accepted += fids
            self.improving += fids
            self.best_mcc, self.best_ap = result["mcc"], result["ap"]
            self.consecutive_failures = 0
        else:
            self.consecutive_failures += 1
        outcome = {"feature_ids": fids, "names": names, "validation_mcc": result["mcc"], "validation_ap": result["ap"],
                   "per_seed": result["per_seed"], "previous_best_mcc": before[0], "previous_best_ap": before[1],
                   "mcc_bar": mcc_bar, "ap_bar": ap_bar, "delta": result["mcc"] - before[0], "improved": improved,
                   "best_mcc": self.best_mcc, "holdout_ids_referenced": result["holdout_ids_referenced"]}
        self.trace.write("evaluate", **outcome)
        return outcome

    def nudge(self) -> str | None:
        if self.stop_reason() is not None:
            return None
        remaining = self.cfg["max_evaluate_calls"] - self.evaluations
        return (f"The search is not finished: {remaining} evaluate calls remain and the stop rule has not triggered "
                f"({self.consecutive_failures} of {self.cfg['stop_after_failed_proposals']} consecutive failed evaluations). "
                "Keep exploring with run_sql, then propose and evaluate new candidates.")

    def stop_reason(self) -> str | None:
        if self.consecutive_failures >= self.cfg["stop_after_failed_proposals"]:
            return f"stopped after {self.consecutive_failures} consecutive failed proposals"
        if self.evaluations >= self.cfg["max_evaluate_calls"]:
            return "evaluate call cap reached"
        return None

    def summary(self) -> dict:
        approved = sum(s == "approved" for s in self.proposed.values())
        return {
            "proposals": len(self.proposed), "approved": approved,
            "quarantined": sum(s == "quarantined" for s in self.proposed.values()),
            "survival_rate": approved / len(self.proposed) if self.proposed else 0.0,
            "evaluations": self.evaluations, "reference": self.reference, "baseline_mcc": self.baseline_mcc,
            "best_mcc": self.best_mcc, "best_ap": self.best_ap, "accepted": self.accepted,
            "accepted_names": [self.registry.get(f).name for f in self.accepted],
            "improving_approved_features": self.improving, "holdout_ids_referenced": self.holdout_refs,
            "quarantined_names": [self.registry.get(f).name for f, s in self.proposed.items() if s == "quarantined"],
        }

    def tools(self) -> list[Tool]:
        return [
            Tool("propose_feature", "Register a feature (Id plus feature columns, one row per part, from parts_* views). "
                 "The warden reviews it immediately and returns its verdict.", ProposeInput, self.propose),
            Tool("evaluate", "Add approved features to the current best set and retrain the 3-seed ensemble. "
                 "Returns validation MCC, average precision, the bars they had to clear, and whether they did.", EvaluateInput, self.evaluate),
        ]


class EnsembleScorer:
    def __init__(self, model_cfg: dict, lab: LeakLab, holdout_ids: np.ndarray):
        self.cfg, self.lab, self.holdout_ids = model_cfg, lab, holdout_ids
        with train.catalog_connection() as con:
            self.train, self.valid = train.build_matrices(con, model_cfg, train.ARTIFACT_DIR)

    def columns(self, records) -> tuple[dict, dict]:
        cols_tr, cols_va = {}, {}
        for record in records:
            tr, va = (self.lab.frame(record.sql, s, "plain") for s in ("train", "validation"))
            if not (np.array_equal(tr.ids, self.train.ids) and np.array_equal(va.ids, self.valid.ids)):
                raise ToolRefused(f"{record.name} does not cover every part of the split")
            self.referenced = getattr(self, "referenced", 0) + int(np.intersect1d(
                np.concatenate([tr.ids, va.ids]), self.holdout_ids).size)
            for column in tr.columns:
                cols_tr[f"fx_{record.name}__{column}"] = tr.columns[column]
                cols_va[f"fx_{record.name}__{column}"] = va.columns[column]
        if len(cols_tr) > MAX_NEW_COLUMNS:
            raise ToolRefused(f"{len(cols_tr)} new columns exceeds the cap of {MAX_NEW_COLUMNS}")
        return cols_tr, cols_va

    def score(self, records) -> dict:
        self.referenced = 0
        cols_tr, cols_va = self.columns(records)
        Xa, (Xb,), names, categorical = encode(cols_tr, [cols_va])
        try:
            guards.check_feature_names(self.train.names + names)
        except guards.LeakageSuspected as exc:
            raise ToolRefused(str(exc)) from exc
        return self.score_arrays(Xa, Xb, names, categorical) | {"holdout_ids_referenced": self.referenced}

    def score_arrays(self, Xa: np.ndarray, Xb: np.ndarray, names: list[str], categorical: list[str]) -> dict:
        X_train, X_valid = np.hstack([self.train.X, Xa]), np.hstack([self.valid.X, Xb])
        boosters = train.fit_ensemble(self.cfg, X_train, self.train.y, self.train.start, self.train.names + names,
                                      ["route_code"] + categorical)
        p = train.predict_ensemble(boosters, X_valid)
        per_seed = [ev.best_threshold(ev.confusion_sweep(self.valid.y, b.predict(X_valid, num_iteration=b.best_iteration)))[1]
                    for b in boosters]
        return {"mcc": ev.best_threshold(ev.confusion_sweep(self.valid.y, p))[1], "ap": ev.average_precision(self.valid.y, p),
                "per_seed": per_seed}

    def calibrate(self, placebo_runs: int, cache: Path) -> dict:
        """Reference MCC/AP and their spread over the baseline plus random-column placebo runs."""
        key = hashlib.sha256(json.dumps([self.cfg, self.train.names, placebo_runs], sort_keys=True).encode()).hexdigest()
        if cache.exists() and json.loads(cache.read_text()).get("key") == key:
            return json.loads(cache.read_text())
        empty = np.empty((len(self.train.y), 0), np.float32), np.empty((len(self.valid.y), 0), np.float32)
        runs = [self.score_arrays(*empty, [], [])]
        for d in range(placebo_runs):
            rng = np.random.default_rng(100 + d)
            cols = [f"placebo_{i}" for i in range(d + 1)]
            runs.append(self.score_arrays(rng.random((len(self.train.y), d + 1), dtype=np.float32),
                                          rng.random((len(self.valid.y), d + 1), dtype=np.float32), cols, []))
        mcc, ap = np.array([r["mcc"] for r in runs]), np.array([r["ap"] for r in runs])
        ref = {"key": key, "mcc": float(mcc.mean()), "ap": float(ap.mean()), "sd_mcc": float(mcc.std(ddof=1)),
               "sd_ap": float(ap.std(ddof=1)), "runs": runs}
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(ref, indent=2) + "\n")
        return ref


def cell(value) -> object:
    if isinstance(value, float):
        return round(value, 6)
    return value if isinstance(value, (int, str, type(None))) else str(value)


class ExploreTools:
    """Read-only exploration of the train split. Labels are visible here and never in feature SQL."""

    def __init__(self, holdout_ids: set, baseline: dict, parquet_dir: Path = PARQUET_DIR):
        self.holdout_ids, self.baseline, self.parquet_dir = holdout_ids, baseline, parquet_dir
        self.columns = {k: feature_columns([r[0] for r in duckdb.sql(
            f"DESCRIBE SELECT * FROM read_parquet('{parquet_dir / f'train_{k}.parquet'}')").fetchall()])
            for k in ("numeric", "date", "categorical")}

    def train_sandbox(self) -> sandbox.Sandbox:
        sb = sandbox.open_sandbox("train", "plain", parquet_dir=self.parquet_dir)
        sb.con.execute(f"CREATE VIEW parts_labels AS SELECT m.bound_id AS Id, t.Response FROM "
                       f"read_parquet('{self.parquet_dir / 'train_numeric.parquet'}') t JOIN id_map m USING (Id)")
        return sb

    def run_sql(self, args: RunSQLInput) -> dict:
        try:
            check_query(args.query, EXPLORE_VIEWS, self.holdout_ids, allow_windows=True)
        except SQLRejected as exc:
            raise ToolRefused(str(exc)) from exc
        with self.train_sandbox() as sb:
            try:
                cur = sb.con.execute(f"SELECT * FROM ({args.query.strip().rstrip(';')}) LIMIT {args.row_limit}")
            except duckdb.Error as exc:
                return {"error": str(exc)[:1000]}
            names = [d[0] for d in cur.description]
            rows = [[cell(v) for v in row] for row in cur.fetchall()]
        text = json.dumps(rows)
        return {"columns": names, "rows": rows if len(text) <= MAX_RESULT_CHARS else rows[: max(1, len(rows) * MAX_RESULT_CHARS // len(text))],
                "truncated": len(text) > MAX_RESULT_CHARS}

    def dry_run(self, sql: str) -> None:
        """Bind feature SQL against the train sandbox without computing it."""
        sandbox.check_feature_sql(sql)
        with sandbox.open_sandbox("train", "plain", parquet_dir=self.parquet_dir) as sb:
            try:
                names = [d[0] for d in sb.con.execute(f"SELECT * FROM ({sql.strip().rstrip(';')}) LIMIT 0").description]
            except duckdb.Error as exc:
                raise sandbox.FeatureSQLError(f"feature SQL does not run: {str(exc)[:800]}") from exc
        if "Id" not in names or len(names) < 2:
            raise sandbox.FeatureSQLError("feature SQL must return Id plus at least one feature column")

    def get_station_map(self, _: EmptyInput) -> dict:
        return station_map(self.columns)

    def get_schema(self, args: SchemaInput) -> dict:
        if args.station is None:
            return {kind: len(cols) for kind, cols in self.columns.items()} | {
                "views": {"parts_numeric": "Id + numeric", "parts_date": "Id + dates (0.01-week units)",
                          "parts_categorical": "Id + categorical strings like 'T1'", "parts_labels": "Id, Response (run_sql only)"}}
        cols = {k: [c for c in v if c.startswith(args.station + "_")] for k, v in self.columns.items()}
        with self.train_sandbox() as sb:
            coverage = {}
            for kind, names in cols.items():
                if names:
                    counts = sb.con.execute("SELECT count(*), " + ", ".join(f'count("{c}")' for c in names)
                                            + f" FROM parts_{kind}").fetchone()
                    coverage |= {c: round(n / counts[0], 5) for c, n in zip(names, counts[1:])}
        return {"columns": cols, "non_null_share_train": coverage}

    def get_baseline(self, _: EmptyInput) -> dict:
        keys = ("validation_mcc", "per_seed_validation_mcc", "validation_auc", "feature_count", "feature_set")
        return {k: self.baseline[k] for k in keys} | {"top_gain_features": self.baseline["top_gain_features"][:15]}

    def tools(self) -> list[Tool]:
        return [
            Tool("get_schema", "Column counts and view names, or one station's columns with non-null shares.", SchemaInput, self.get_schema),
            Tool("get_station_map", "Line -> station -> column names.", EmptyInput, self.get_station_map),
            Tool("run_sql", "Read-only SELECT over the train split views. State the hypothesis first.", RunSQLInput, self.run_sql),
            Tool("get_baseline", "Baseline validation MCC, per-seed MCC, and top features.", EmptyInput, self.get_baseline),
        ]


def gate_problems(summary: dict, trace_complete: bool, holdout_runs_unchanged: bool) -> list[str]:
    problems = []
    if not summary["improving_approved_features"]:
        problems.append("no warden-approved feature improved validation MCC")
    if not trace_complete:
        problems.append("trace is incomplete")
    if not holdout_runs_unchanged:
        problems.append("data/holdout_runs.json changed during the search")
    if summary["holdout_ids_referenced"]:
        problems.append(f"evaluate referenced {summary['holdout_ids_referenced']} holdout Ids")
    if "survival_rate" not in summary:
        problems.append("survival rate was not computed")
    return problems


def task_message(baseline: dict, reference: dict, cfg: dict) -> str:
    return (
        f"Baseline ({baseline['feature_set']}, {baseline['feature_count']} features): reference validation MCC "
        f"{reference['mcc']:.4f} and average precision {reference['ap']:.4f}, averaged over the baseline and "
        f"{cfg['placebo_runs']} placebo runs (MCC noise SD {reference['sd_mcc']:.4f}, AP noise SD {reference['sd_ap']:.4f}). "
        "It already has per-station numeric min/max/range and missing counts, per-station dwell and arrival offset, "
        "per-line span and arrival, total elapsed time, station count, route code, integer codes of 96 categorical "
        "columns, and per-station categorical counts.\n"
        "Feature SQL contract: a SELECT over parts_numeric, parts_date, parts_categorical that returns column Id plus "
        "one or more feature columns, exactly one row per part. No window functions, no labels, no Id arithmetic. "
        "Date columns are relative time; categorical values look like 'T1' and can be parsed with try_cast(substr(x, 2) AS BIGINT). "
        "Useful: least(*COLUMNS('^L3_S32_D')) style unpacking.\n"
        f"A feature batch is accepted when validation MCC beats the current best by more than "
        f"max({cfg['min_delta_mcc']}, {cfg['noise_sd_multiplier']} x MCC SD) and average precision beats it by more than "
        f"{cfg['noise_sd_multiplier']} x AP SD. Small effects are indistinguishable from noise; look for strong signals. You have {cfg['max_evaluate_calls']} evaluate calls. The search ends after "
        f"{cfg['stop_after_failed_proposals']} consecutive evaluate calls that do not clear the bar. Quarantines waste budget. "
        "Explore with run_sql first (failure rates by station, measurement, categorical value, timing), then propose and evaluate. "
        "Prefer features with a large, stable failure-rate contrast on train over tiny refinements of existing baseline aggregates."
    )


def holdout_id_array(parquet_dir: Path = PARQUET_DIR) -> np.ndarray:
    return np.asarray(duckdb.sql(f"SELECT Id FROM read_parquet('{parquet_dir / SPLITS_NAME}') "
                                 "WHERE split = 'holdout' ORDER BY Id").fetchnumpy()["Id"])


def trace_is_complete(path: Path) -> bool:
    events = [json.loads(line)["event"] for line in path.read_text().splitlines()] if path.exists() else []
    return "agent_start" in events and "search_summary" in events \
        and any(e in ("agent_end", "budget_stop") for e in events)


def build_search(agents: dict, holdout: np.ndarray, trace: TraceWriter, warden_trace: Path):
    hyp, wcfg = agents["hypothesis"], agents["warden"]
    baseline = json.loads((train.ARTIFACT_DIR / "metrics.json").read_text())
    pricing = (agents["usd_per_million_input_tokens"], agents["usd_per_million_output_tokens"])
    registry, lab, client = FeatureRegistry(), LeakLab(wcfg), OpenAIClient(agents["model"])
    explore = ExploreTools(set(holdout.tolist()), baseline)

    def reviewer(fid: str):
        decision = review_feature(fid, registry, client, wcfg, warden_trace, lab, pricing)
        lab.evict(registry.get(fid).sql, keep_plain=decision.final == "approved")
        return decision

    scorer = EnsembleScorer(train.load_model_config(), lab, holdout)
    reference = scorer.calibrate(hyp["placebo_runs"], SEARCH_DIR / "noise_reference.json")
    trace.write("noise_reference", **{k: v for k, v in reference.items() if k != "runs"})
    session = SearchSession(registry, reviewer, scorer, reference, hyp, trace, set(holdout.tolist()), dry_run=explore.dry_run)
    budget = Budget(max_usd=hyp["max_usd"], usd_per_mtok_in=pricing[0], usd_per_mtok_out=pricing[1],
                    max_calls={"evaluate": hyp["max_evaluate_calls"]})
    return client, session, explore.tools() + session.tools(), budget, task_message(baseline, reference, hyp)


def main() -> int:
    agents = yaml.safe_load((CONFIG_DIR / "agents.yaml").read_text())
    runs_before, holdout = RUNS_PATH.read_bytes(), holdout_id_array()
    trace_path, warden_trace = new_trace_path("hypothesis"), new_trace_path("warden-search")
    trace = TraceWriter(trace_path)
    client, session, tools, budget, task = build_search(agents, holdout, trace, warden_trace)
    run_agent(client, PROMPT, task, tools, budget, trace, MAX_TURNS, session.stop_reason,
              nudge=session.nudge, max_nudges=agents["hypothesis"]["max_nudges"])
    summary = session.summary() | {"agent_spend_usd": round(budget.spent_usd, 4), "trace": str(trace_path),
                                   "warden_trace": str(warden_trace)}
    trace.write("search_summary", **summary)
    SEARCH_DIR.mkdir(parents=True, exist_ok=True)
    (SEARCH_DIR / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    problems = gate_problems(summary, trace_is_complete(trace_path), RUNS_PATH.read_bytes() == runs_before)
    for problem in problems:
        print(f"FAIL: {problem}", file=sys.stderr)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
