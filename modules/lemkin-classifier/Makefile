.PHONY: install install-dev test lint format type-check clean build upload help

help:
	@echo "Lemkin Document Classifier - Build Commands"
	@echo "=========================================="
	@echo "Development: install install-dev test lint format type-check"
	@echo "Build: build clean upload"

install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"

test:
	pytest --cov=src/lemkin_classifier --cov-report=html --cov-report=term-missing -v

lint:
	ruff check src/ tests/
	mypy src/

format:
	black src/ tests/
	ruff check --fix src/ tests/

type-check:
	mypy src/ --strict

clean:
	rm -rf build/ dist/ *.egg-info/ .pytest_cache/ .coverage htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} +

build: clean
	python -m build

upload: build
	python -m twine upload dist/*

verify-install:
	lemkin-classifier --help
	python -c "import lemkin_classifier; print(f'✅ lemkin-classifier v{lemkin_classifier.__version__} installed')"

quality: format lint type-check test