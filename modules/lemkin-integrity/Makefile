.PHONY: help install test lint format clean build deploy

help:
	@echo "Available commands:"
	@echo "  install     Install development dependencies"
	@echo "  test        Run all tests"
	@echo "  lint        Run linting checks"
	@echo "  format      Format code"
	@echo "  clean       Clean build artifacts"
	@echo "  build       Build distribution packages"
	@echo "  deploy      Deploy to PyPI"

install:
	pip install -e ".[dev]"
	pre-commit install

test:
	pytest --cov=src/lemkin_integrity --cov-report=html --cov-report=term-missing

lint:
	ruff check src/ tests/
	mypy src/
	black --check src/ tests/

format:
	black src/ tests/
	ruff check --fix src/ tests/

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf htmlcov/
	find . -type d -name __pycache__ -delete
	find . -type f -name "*.pyc" -delete

build: clean
	python -m build

deploy: build
	python -m twine upload dist/*

demo:
	python -c "from lemkin_integrity.core import create_sample_evidence; create_sample_evidence()"