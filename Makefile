.PHONY: install test benchmark sweep lint clean

PYTHON ?= python3

install:
	$(PYTHON) -m pip install -e ".[dev]"

test:
	$(PYTHON) -m pytest tests/

benchmark:
	$(PYTHON) experiments/run_ptq_comparison.py --dim 4096 --batch 128 --outlier-scale 80.0

sweep:
	$(PYTHON) experiments/run_synthetic_bench.py --out results/synthetic_sweep.csv

lint:
	$(PYTHON) -m ruff check src/ tests/ experiments/
	$(PYTHON) -m black --check src/ tests/ experiments/

clean:
	rm -rf build/ dist/ *.egg-info .pytest_cache .ruff_cache
	find . -type d -name __pycache__ -exec rm -rf {} +
