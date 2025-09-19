.PHONY: install install-dev test lint format type-check clean build upload help

# Default target
help:
	@echo "Lemkin Legal Framework Mapper - Build Commands"
	@echo "=============================================="
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
	pytest --cov=src/lemkin_frameworks --cov-report=html --cov-report=term-missing -v

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

# Framework-specific commands
analyze-rome-example:
	@echo "Example: Analyzing evidence against Rome Statute"
	lemkin-frameworks analyze-rome tests/fixtures/evidence.json --output rome_analysis.json

analyze-geneva-example:
	@echo "Example: Checking Geneva Convention violations"
	lemkin-frameworks analyze-geneva tests/fixtures/evidence.json --output geneva_analysis.json

analyze-human-rights-example:
	@echo "Example: Human rights framework analysis"
	lemkin-frameworks analyze-rights tests/fixtures/evidence.json --framework iccpr --output rights_analysis.json

map-elements-example:
	@echo "Example: Mapping evidence to legal elements"
	lemkin-frameworks map-elements tests/fixtures/evidence.json --framework rome-statute --crime war-crimes

comprehensive-analysis:
	@echo "Example: Comprehensive legal framework analysis"
	lemkin-frameworks comprehensive tests/fixtures/evidence.json --all-frameworks --output full_analysis.json

# Batch processing
batch-analyze:
	@echo "Example: Batch framework analysis"
	lemkin-frameworks batch-analyze tests/fixtures/cases/ --framework rome-statute --output results/

# Framework information
list-frameworks:
	@echo "Supported Legal Frameworks:"
	@echo "- Rome Statute (ICC)"
	@echo "  - War crimes"
	@echo "  - Crimes against humanity"
	@echo "  - Genocide"
	@echo "  - Aggression"
	@echo ""
	@echo "- Geneva Conventions"
	@echo "  - Geneva Convention I (Wounded and Sick)"
	@echo "  - Geneva Convention II (Wounded at Sea)"
	@echo "  - Geneva Convention III (Prisoners of War)"
	@echo "  - Geneva Convention IV (Civilians)"
	@echo "  - Additional Protocol I"
	@echo "  - Additional Protocol II"
	@echo ""
	@echo "- Human Rights Instruments"
	@echo "  - ICCPR (International Covenant on Civil and Political Rights)"
	@echo "  - ICESCR (International Covenant on Economic, Social and Cultural Rights)"
	@echo "  - CAT (Convention Against Torture)"
	@echo "  - CRC (Convention on the Rights of the Child)"
	@echo "  - CEDAW (Convention on Elimination of Discrimination Against Women)"

show-elements:
	@echo "Example: Show legal elements for a specific crime"
	lemkin-frameworks show-elements --framework rome-statute --crime genocide

# Validation
validate-analysis:
	@echo "Validating analysis format"
	lemkin-frameworks validate tests/fixtures/analysis.json

check-evidence:
	@echo "Checking evidence sufficiency"
	lemkin-frameworks check-sufficiency tests/fixtures/evidence.json --framework rome-statute --crime war-crimes

# Performance testing
perf-test:
	@echo "Running performance tests on framework analysis..."
	time lemkin-frameworks analyze-rome tests/fixtures/large_evidence.json --output /tmp/rome_perf.json
	time lemkin-frameworks analyze-geneva tests/fixtures/large_evidence.json --output /tmp/geneva_perf.json

# Installation verification
verify-install:
	@echo "Verifying installation..."
	lemkin-frameworks --help
	python -c "import lemkin_frameworks; print(f'✅ lemkin-frameworks v{lemkin_frameworks.__version__} installed successfully')"
	python -c "from lemkin_frameworks import RomeStatuteAnalyzer, GenevaAnalyzer, HumanRightsAnalyzer; print('✅ All framework analyzers imported successfully')"

# Dependencies check
deps-check:
	@echo "Checking critical dependencies..."
	python -c "import pydantic; print('✅ pydantic (data validation)')"
	python -c "import typer; print('✅ typer (CLI framework)')"
	python -c "import rich; print('✅ rich (terminal output)')"
	python -c "import loguru; print('✅ loguru (logging)')"
	python -c "import jinja2; print('✅ jinja2 (report templates)')"

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