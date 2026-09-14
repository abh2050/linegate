import hashlib
import json

import pytest

from linegate.dataio import fetch
from linegate.dataio.fetch import ManifestError


def full_manifest(tmp_path, overrides=None):
    raw = tmp_path / "raw"
    raw.mkdir()
    manifest = {}
    for name in fetch.RAW_FILES:
        (raw / name).write_bytes(name.encode())
        manifest[name] = hashlib.sha256(name.encode()).hexdigest()
    manifest.update(overrides or {})
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    return raw, path


def test_verify_all_accepts_matching_hashes(tmp_path):
    raw, manifest = full_manifest(tmp_path)
    assert [p.name for p in fetch.verify_all(raw, manifest)] == list(fetch.RAW_FILES)


def test_verify_rejects_hash_mismatch(tmp_path):
    raw, manifest = full_manifest(tmp_path, {"train_date.csv.zip": "0" * 64})
    with pytest.raises(ManifestError, match="train_date.csv.zip"):
        fetch.verify_all(raw, manifest)


def test_verify_rejects_missing_manifest(tmp_path):
    with pytest.raises(ManifestError, match="make manifest"):
        fetch.verify_all(tmp_path, tmp_path / "manifest.json")


def test_verify_rejects_manifest_missing_an_entry(tmp_path):
    raw, manifest = full_manifest(tmp_path)
    data = json.loads(manifest.read_text())
    del data["test_numeric.csv.zip"]
    manifest.write_text(json.dumps(data))
    with pytest.raises(ManifestError, match="test_numeric"):
        fetch.verify_all(raw, manifest)


def test_verify_rejects_unlisted_file(tmp_path):
    stray = tmp_path / "sampled_train.csv.zip"
    stray.write_bytes(b"x")
    with pytest.raises(ManifestError, match="not listed"):
        fetch.verify_file(stray, {})


def test_write_manifest_never_overwrites(tmp_path):
    raw, manifest = full_manifest(tmp_path)
    with pytest.raises(ManifestError, match="never overwritten"):
        fetch.write_manifest(raw, manifest)


def test_unpack_copies_data_zips_and_skips_sample_submission(tmp_path):
    import zipfile
    bundle = tmp_path / "bundle.zip"
    with zipfile.ZipFile(bundle, "w") as archive:
        for name in (*fetch.RAW_FILES, "sample_submission.csv.zip"):
            archive.writestr(name, name)
    paths = fetch.unpack(bundle, tmp_path / "raw")
    assert sorted(p.name for p in (tmp_path / "raw").iterdir()) == sorted(fetch.RAW_FILES)
    assert all(p.read_bytes() == p.name.encode() for p in paths)


def test_unpack_rejects_incomplete_bundle(tmp_path):
    import zipfile
    bundle = tmp_path / "bundle.zip"
    with zipfile.ZipFile(bundle, "w") as archive:
        archive.writestr("train_numeric.csv.zip", b"x")
    with pytest.raises(ManifestError, match="lacks"):
        fetch.unpack(bundle, tmp_path / "raw")
