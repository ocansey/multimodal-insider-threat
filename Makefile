.PHONY: help setup test fixture smoke study clean

PYTHON ?= python3
RAW    ?= ~/Downloads/r4.2
ARTIFACTS ?= data/artifacts/cert

help:
	@echo "make setup      install dependencies"
	@echo "make test       run the test suite (no data needed)"
	@echo "make smoke      generate synthetic data and run the whole pipeline"
	@echo "make prepare    reduce the real CERT release  (RAW=path/to/r4.2)"
	@echo "make study      train and evaluate on real artefacts"
	@echo "make clean      remove generated reports and fixtures"

setup:
	$(PYTHON) -m pip install -r requirements.txt

test:
	$(PYTHON) -m pytest tests -q

smoke:
	$(PYTHON) scripts/run_experiment.py --fixture --arms full --epochs 4

prepare:
	$(PYTHON) scripts/prepare_local.py --raw $(RAW) --out $(ARTIFACTS)

study:
	$(PYTHON) scripts/run_experiment.py --artifacts $(ARTIFACTS)

clean:
	rm -rf reports/tables reports/fixture reports/figures data/fixtures/* data/artifacts/fixture
	find . -name '__pycache__' -type d -exec rm -rf {} + 2>/dev/null || true
	find . -name '.pytest_cache' -type d -exec rm -rf {} + 2>/dev/null || true
