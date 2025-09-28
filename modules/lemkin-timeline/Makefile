.PHONY: install install-dev test lint format type-check clean build upload help

# Default target
help:
	@echo "Lemkin Timeline Constructor - Build Commands"
	@echo "============================================"
	@echo ""
	@echo "Development:"
	@echo "  install     Install package in development mode"
	@echo "  install-dev Install with development dependencies"
	@echo "  test        Run tests with coverage"
	@echo "  lint        Run linting checks"
	@echo "  format      Auto-format code"
	@echo "  type-check  Run type checking"
	@echo ""
	@echo "Build & Release:"
	@echo "  build       Build package distributions"
	@echo "  clean       Clean build artifacts"
	@echo "  upload      Upload to PyPI (requires credentials)"
	@echo ""
	@echo "Examples:"
	@echo "  make install-dev  # Set up development environment"
	@echo "  make test         # Run full test suite"
	@echo "  make format lint  # Format and check code"

# Installation
install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"

# Testing
test:
	pytest --cov=src/lemkin_timeline --cov-report=html --cov-report=term-missing -v

test-fast:
	pytest -x --tb=short

# Code Quality
lint:
	ruff check src/ tests/
	mypy src/

format:
	black src/ tests/
	ruff check --fix src/ tests/

type-check:
	mypy src/ --strict

# Build
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

build: clean
	python -m build

upload: build
	python -m twine upload dist/*

# Development workflow
dev-setup: install-dev
	pre-commit install

# Timeline-specific commands
extract-example:
	@echo "Example: Extracting temporal references"
	lemkin-timeline extract-temporal tests/fixtures/sample_document.txt --language en

build-timeline-example:
	@echo "Example: Building timeline from events"
	lemkin-timeline build-timeline tests/fixtures/events.json --output timeline.json

detect-inconsistencies-example:
	@echo "Example: Detecting temporal inconsistencies"
	lemkin-timeline detect-inconsistencies tests/fixtures/timeline.json --threshold 0.8

visualize-example:
	@echo "Example: Creating timeline visualization"
	lemkin-timeline visualize tests/fixtures/timeline.json --format html --output timeline.html

# Batch processing
batch-extract:
	@echo "Example: Batch temporal extraction"
	lemkin-timeline extract-batch tests/fixtures/documents/ --pattern "*.txt" --output results/

batch-analyze:
	@echo "Example: Batch timeline analysis"
	lemkin-timeline analyze-batch tests/fixtures/timelines/ --output analysis/

# Validation and checks
validate-timeline:
	@echo "Validating timeline format"
	lemkin-timeline validate tests/fixtures/timeline.json

check-formats:
	@echo "Supported temporal formats:"
	@echo "- ISO 8601 (2024-01-15T15:30:00Z)"
	@echo "- Natural language (January 15, 2024 at 3:30 PM)"
	@echo "- Relative references (three days ago, last week)"
	@echo "- Durations (for 2 hours, lasted 30 minutes)"
	@echo "- Ranges (from Jan 1 to Jan 15)"

# Performance testing
perf-test:
	@echo "Running performance tests on temporal extraction..."
	time lemkin-timeline extract-temporal tests/fixtures/large_document.txt --output /tmp/temporal_perf.json
	@echo "Running performance tests on timeline construction..."
	time lemkin-timeline build-timeline tests/fixtures/large_events.json --output /tmp/timeline_perf.json

# Installation verification
verify-install:
	@echo "Verifying installation..."
	lemkin-timeline --help
	python -c "import lemkin_timeline; print(f'✅ lemkin-timeline v{lemkin_timeline.__version__} installed successfully')"
	python -c "from lemkin_timeline import TemporalExtractor, TimelineConstructor, InconsistencyDetector; print('✅ All core components imported successfully')"

# Dependencies check
deps-check:
	@echo "Checking critical dependencies..."
	python -c "import dateutil; print('✅ dateutil (temporal parsing)')"
	python -c "import pendulum; print('✅ pendulum (timezone handling)')"
	python -c "import spacy; print('✅ spacy (NLP for temporal extraction)')"
	python -c "import plotly; print('✅ plotly (timeline visualization)')"
	python -c "import pydantic; print('✅ pydantic (data validation)')"

# Security scan
security-scan:
	pip-audit

# All quality checks
quality: format lint type-check test

# CI pipeline simulation
ci: install-dev quality security-scan verify-install

# Release preparation
release-prep: clean quality build
	@echo "Package ready for release!"
	@echo "Files in dist/:"
	@ls -la dist/