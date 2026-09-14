UV := uv run

.PHONY: setup test data unpack manifest gate-0

setup:
	uv sync

test:
	$(UV) pytest tests/unit -q

# Needs Kaggle credentials and accepted competition rules.
data:
	$(UV) python -m linegate.dataio.fetch download

# Use when the full bundle was saved by hand to data/bosch-production-line-performance.zip.
unpack:
	$(UV) python -m linegate.dataio.fetch unpack

# Trust on first use: records hashes of the downloaded files. Refuses to overwrite.
manifest:
	$(UV) python -m linegate.dataio.fetch write-manifest

gate-0:
	$(UV) pytest tests/unit -q
	$(UV) python -m linegate.dataio.fetch verify
	$(UV) python -m linegate.dataio.convert
	$(UV) python -m linegate.dataio.schema
	$(UV) python -m linegate.dataio.splits
	$(UV) python scripts/seal_holdout.py
	$(UV) python -m linegate.dataio.duck
