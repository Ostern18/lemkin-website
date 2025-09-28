.PHONY: install install-dev test lint format type-check clean build upload help

# Default target
help:
	@echo "Lemkin Digital Forensics Helpers - Build Commands"
	@echo "================================================"
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
	@echo "  make install-dev    # Set up development environment"
	@echo "  make test           # Run full test suite"
	@echo "  make format lint    # Format and check code"

# Installation
install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"

# Testing
test:
	pytest --cov=src/lemkin_forensics --cov-report=html --cov-report=term-missing -v

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

# Forensics examples
file-analysis-example:
	@echo "Example: Analyzing file system"
	lemkin-forensics analyze-filesystem /path/to/evidence --output analysis.json

network-analysis-example:
	@echo "Example: Processing network logs"
	lemkin-forensics process-network-logs tests/fixtures/network.log --output network_analysis.json

authenticity-example:
	@echo "Example: Verifying digital authenticity"
	lemkin-forensics verify-authenticity tests/fixtures/evidence.pdf --output authenticity_report.json

# Batch processing
batch-analysis:
	@echo "Example: Batch file analysis"
	lemkin-forensics batch-process tests/fixtures/files/ results/ --type filesystem

batch-network:
	@echo "Example: Batch network analysis"
	lemkin-forensics batch-process tests/fixtures/logs/ results/ --type network

# Verification
verify-install:
	@echo "Verifying installation..."
	lemkin-forensics --help
	python -c "import lemkin_forensics; print(f'✅ lemkin-forensics v{lemkin_forensics.__version__} installed successfully')"
	python -c "from lemkin_forensics import FileAnalyzer, NetworkProcessor, AuthenticityVerifier, ForensicsReporter; print('✅ All core components imported successfully')"

# Dependencies check
deps-check:
	@echo "Checking critical dependencies..."
	python -c "import hashlib; print('✅ hashlib (cryptographic hashing)')"
	python -c "import sqlite3; print('✅ sqlite3 (database operations)')"
	python -c "import json; print('✅ json (data serialization)')"
	python -c "import pathlib; print('✅ pathlib (file operations)')"

# Performance testing
perf-test:
	@echo "Running performance tests on sample forensic data..."
	time lemkin-forensics analyze-filesystem tests/fixtures/sample_files/ --output /tmp/forensics_perf.json

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