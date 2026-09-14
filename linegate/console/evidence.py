"""Evidence for the review queue: route, out-of-range readings, and similar train parts.

The queue is the validation parts whose score falls in the policy's abstain
band. Any other part Id, holdout included, is not found. Normal ranges are
the 1st and 99th percentile of each numeric column over the train split.
Neighbours are the nearest train parts on the same route, by standardized
distance over a few top baseline features; Id and row order play no part.
"""

from __future__ import annotations

import json
from pathlib import Path

import duckdb
import numpy as np

from linegate.dataio.schema import feature_columns
from linegate.dataio.splits import SPLITS_NAME


class PartNotFound(KeyError):
    """Raised for any part outside the review queue."""


class PartStore:
    def __init__(self, parquet_dir: Path, scores_path: Path, policy: dict, train_features: Path,
                 valid_features: Path, neighbor_columns: list[str], cache_dir: Path):
        self.parquet_dir, self.policy, self.cache_dir = parquet_dir, policy, cache_dir
        self.valid_features, self.neighbor_columns = valid_features, neighbor_columns
        scores = np.load(scores_path)
        in_band = (scores["p"] >= policy["band_low"]) & (scores["p"] < policy["band_high"])
        order = np.argsort(-scores["p"][in_band], kind="stable")
        self._queue = [{"part_id": int(i), "score": float(s)}
                       for i, s in zip(scores["id"][in_band][order], scores["p"][in_band][order])]
        self._queue_ids = {item["part_id"]: item["score"] for item in self._queue}
        self.columns = {k: feature_columns(self.describe(f"train_{k}.parquet")) for k in ("numeric", "date", "categorical")}
        self.schema_columns = {c for cols in self.columns.values() for c in cols}
        self.ranges = self.normal_ranges()
        self._train = self.load_reference(train_features)

    def describe(self, name: str) -> list[str]:
        return [r[0] for r in duckdb.sql(f"DESCRIBE SELECT * FROM read_parquet('{self.parquet_dir / name}')").fetchall()]

    def normal_ranges(self) -> dict[str, list[float]]:
        path = self.cache_dir / "normal_ranges.json"
        if path.exists():
            return json.loads(path.read_text())
        cols = self.columns["numeric"]
        row = duckdb.sql(
            "SELECT " + ", ".join(f'approx_quantile("{c}", [0.01, 0.99])' for c in cols)
            + f" FROM read_parquet('{self.parquet_dir / 'train_numeric.parquet'}') t SEMI JOIN "
            f"(SELECT Id FROM read_parquet('{self.parquet_dir / SPLITS_NAME}') WHERE split = 'train') s USING (Id)").fetchone()
        ranges = {c: [float(q[0]), float(q[1])] for c, q in zip(cols, row) if q and q[0] is not None}
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(ranges) + "\n")
        return ranges

    def load_reference(self, path: Path) -> dict:
        cols = ", ".join(f'"{c}"' for c in self.neighbor_columns)
        data = duckdb.sql(f"SELECT Id, Response, route, {cols} FROM read_parquet('{path}')").fetchnumpy()
        X = np.column_stack([np.ma.filled(np.ma.asarray(data[c]).astype(np.float64), np.nan) for c in self.neighbor_columns])
        mean, std = np.nanmean(X, axis=0), np.nanstd(X, axis=0)
        std = np.where(std > 0, std, 1.0)
        Z = np.nan_to_num((X - mean) / std)
        return {"ids": np.asarray(data["Id"]), "y": np.asarray(data["Response"]), "route": np.asarray(data["route"], dtype=object),
                "Z": Z, "mean": mean, "std": std}

    def queue(self) -> list[dict]:
        return list(self._queue)

    def require(self, part_id: int) -> float:
        if part_id not in self._queue_ids:
            raise PartNotFound(part_id)
        return self._queue_ids[part_id]

    def row(self, kind: str, part_id: int) -> dict:
        cur = duckdb.execute(f"SELECT * FROM read_parquet('{self.parquet_dir / f'train_{kind}.parquet'}') WHERE Id = ?", [part_id])
        names = [d[0] for d in cur.description]
        values = cur.fetchone()
        return {n: v for n, v in zip(names, values) if n in self.columns[kind] and v is not None}

    def get_part(self, part_id: int) -> dict:
        score = self.require(part_id)
        numeric, dates = self.row("numeric", part_id), self.row("date", part_id)
        route = duckdb.execute(f"SELECT route FROM read_parquet('{self.valid_features}') WHERE Id = ?", [part_id]).fetchone()[0]
        out = [{"column": c, "value": round(float(v), 6), "low": round(self.ranges[c][0], 6), "high": round(self.ranges[c][1], 6)}
               for c, v in numeric.items() if c in self.ranges and not self.ranges[c][0] <= v <= self.ranges[c][1]]
        return {"part_id": part_id, "score": score, "route": route, "measurements_present": len(numeric),
                "out_of_range": out, "columns": sorted(numeric) + sorted(dates),
                "policy_band": [self.policy["band_low"], self.policy["band_high"]]}

    def get_neighbors(self, part_id: int, k: int = 20) -> dict:
        self.require(part_id)
        cols = ", ".join(f'"{c}"' for c in self.neighbor_columns)
        row = duckdb.execute(f"SELECT route, {cols} FROM read_parquet('{self.valid_features}') WHERE Id = ?", [part_id]).fetchone()
        ref = self._train
        x = np.array([np.nan if v is None else float(v) for v in row[1:]])
        z = np.nan_to_num((x - ref["mean"]) / ref["std"])
        candidates = np.flatnonzero(ref["route"] == row[0])
        if len(candidates) < k:
            candidates = np.arange(len(ref["ids"]))
        dist = np.linalg.norm(ref["Z"][candidates] - z, axis=1)
        nearest = candidates[np.argsort(dist, kind="stable")[:k]]
        neighbors = [{"part_id": int(ref["ids"][i]), "response": int(ref["y"][i]),
                      "distance": round(float(np.linalg.norm(ref["Z"][i] - z)), 4)} for i in nearest]
        return {"neighbors": neighbors, "failed": sum(n["response"] for n in neighbors), "k": len(neighbors),
                "same_route": bool(len(candidates) >= k and len(candidates) < len(ref["ids"]))}
