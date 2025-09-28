.PHONY: install install-dev test lint format type-check clean build upload help

# Default target
help:
	@echo "Lemkin Geospatial Analysis Suite - Build Commands"
	@echo "================================================="
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
	pytest --cov=src/lemkin_geo --cov-report=html --cov-report=term-missing -v

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

# Geospatial analysis examples
coordinate-example:
	@echo "Example: Converting coordinates"
	lemkin-geo convert-coordinates "40.7128,-74.0060" --format decimal --output coords.json

geofence-example:
	@echo "Example: Creating geofence analysis"
	lemkin-geo correlate-events tests/fixtures/events.json --radius 500 --output geofence_analysis.json

mapping-example:
	@echo "Example: Generating evidence map"
	lemkin-geo generate-map tests/fixtures/evidence.json --output evidence_map.html --format interactive

satellite-example:
	@echo "Example: Analyzing satellite imagery"
	lemkin-geo analyze-satellite tests/fixtures/coordinates.json --date-range "2024-01-01,2024-01-31" --output satellite_analysis.json

# Batch processing
batch-convert:
	@echo "Example: Batch coordinate conversion"
	lemkin-geo batch-process tests/fixtures/coordinates/ results/ --type conversion --format decimal

batch-correlate:
	@echo "Example: Batch event correlation"
	lemkin-geo batch-process tests/fixtures/events/ results/ --type correlation --radius 1000

# Verification
verify-install:
	@echo "Verifying installation..."
	lemkin-geo --help
	python -c "import lemkin_geo; print(f'✅ lemkin-geo v{lemkin_geo.__version__} installed successfully')"
	python -c "from lemkin_geo import CoordinateConverter, GeofenceProcessor, MappingGenerator, SatelliteAnalyzer; print('✅ All core components imported successfully')"

# Dependencies check
deps-check:
	@echo "Checking critical dependencies..."
	python -c "import geopandas; print('✅ geopandas (geospatial operations)')"
	python -c "import pyproj; print('✅ pyproj (coordinate transformations)')"
	python -c "import folium; print('✅ folium (mapping)')"
	python -c "import shapely; print('✅ shapely (geometric operations)')"
	python -c "import geopy; print('✅ geopy (geocoding)')"

# Performance testing
perf-test:
	@echo "Running performance tests on sample geospatial data..."
	time lemkin-geo convert-coordinates "40.7128,-74.0060" --format utm --output /tmp/coord_perf.json
	time lemkin-geo correlate-events tests/fixtures/events.json --radius 1000 --output /tmp/geofence_perf.json

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