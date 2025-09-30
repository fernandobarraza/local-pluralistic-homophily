.PHONY: core aux lint test

PY=python

core:
	$(PY) core/run_pipeline_core.py --help

aux:
	$(PY) aux/run_pipeline_aux.py --help

lint:
	ruff check .

test:
	pytest -q
