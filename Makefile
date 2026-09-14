UV := uv run

.PHONY: setup test data unpack manifest gate-0 gate-1 gate-1-bg gate-1-status gate-2 gate-3

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

gate-1:
	$(UV) pytest tests/unit -q
	$(UV) python -m linegate.model.train

# Detached run that survives the terminal and Claude session closing.
# caffeinate keeps the Mac awake; the exit code lands in logs/gate-1.exit.
gate-1-bg:
	mkdir -p logs
	rm -f logs/gate-1.exit
	nohup caffeinate -i sh -c '$(MAKE) gate-1 > logs/gate-1.log 2>&1; echo $$? > logs/gate-1.exit' > /dev/null 2>&1 &
	@echo "started; follow with: make gate-1-status"

gate-1-status:
	@if [ -f logs/gate-1.exit ]; then echo "finished, exit $$(cat logs/gate-1.exit)"; else echo "running"; fi
	@grep -v VIRTUAL_ENV logs/gate-1.log | tail -15
	@if [ -f data/artifacts/baseline/metrics.json ]; then head -20 data/artifacts/baseline/metrics.json; fi

gate-2:
	$(UV) pytest tests/unit -q
	$(UV) python -m linegate.cost.policy

gate-3:
	$(UV) pytest tests/unit -q
	$(UV) python scripts/red_team_leak.py
