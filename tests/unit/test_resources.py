import duckdb
import pytest

from linegate.dataio import resources


def test_configure_caps_memory_threads_and_temp(tmp_path):
    with duckdb.connect() as con:
        resources.configure(con, tmp_path / "tmp")
        settings = dict(con.execute(
            "SELECT name, value FROM duckdb_settings() WHERE name IN "
            "('threads', 'preserve_insertion_order', 'max_temp_directory_size')").fetchall())
    assert int(settings["threads"]) == resources.worker_threads()
    assert settings["preserve_insertion_order"] == "false"
    assert (tmp_path / "tmp").is_dir()


def test_require_free_disk_raises_when_short(tmp_path):
    with pytest.raises(resources.ResourceError, match="GiB free"):
        resources.require_free_disk(tmp_path, 1 << 60)
