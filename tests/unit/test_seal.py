import json
import sys

import pytest

from linegate.dataio import ROOT, splits
from tests.unit.test_splits import CFG

sys.path.insert(0, str(ROOT / "scripts"))
import seal_holdout  # noqa: E402


@pytest.fixture
def splits_path(date_parquet):
    splits.write_splits(date_parquet.parent, CFG)
    return date_parquet.parent / splits.SPLITS_NAME


def test_seal_writes_lock_and_zero_counter(splits_path, tmp_path):
    lock, runs = tmp_path / "holdout.lock", tmp_path / "holdout_runs.json"
    digest, created = seal_holdout.seal(splits_path, lock, runs)
    assert created
    ids = seal_holdout.holdout_ids(splits_path)
    assert ids
    assert json.loads(lock.read_text()) == {"sha256": splits.holdout_digest(ids), "count": len(ids)}
    assert json.loads(runs.read_text()) == {"holdout_runs": 0}


def test_seal_is_idempotent_for_same_holdout(splits_path, tmp_path):
    lock, runs = tmp_path / "holdout.lock", tmp_path / "holdout_runs.json"
    seal_holdout.seal(splits_path, lock, runs)
    runs.write_text(json.dumps({"holdout_runs": 1}))
    digest, created = seal_holdout.seal(splits_path, lock, runs)
    assert not created
    assert json.loads(runs.read_text()) == {"holdout_runs": 1}


def test_seal_fails_when_holdout_changed(splits_path, tmp_path):
    lock = tmp_path / "holdout.lock"
    lock.write_text(json.dumps({"sha256": "0" * 64, "count": 2}))
    with pytest.raises(seal_holdout.SealError):
        seal_holdout.seal(splits_path, lock, tmp_path / "holdout_runs.json")
