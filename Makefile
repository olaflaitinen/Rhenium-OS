.PHONY: install dev test lint format typecheck clean docs serve

install:
	pip install -e .

dev:
	pip install -e ".[dev]"

test:
	pytest -q

test-cov:
	pytest --cov=rhenium --cov-report=term-missing

lint:
	ruff check src/

format:
	black src/ tests/
	ruff check --fix src/

typecheck:
	mypy src/rhenium

clean:
	rm -rf build/ dist/ *.egg-info .pytest_cache .mypy_cache .ruff_cache
	find . -type d -name __pycache__ -exec rm -rf {} +

docs:
	mkdocs build

serve:
	uvicorn rhenium.server.app:app --reload

synthetic:
	rhenium synthetic --output ./data/synthetic --num 5

benchmark:
	rhenium benchmark --output ./results
