.PHONY: install install-dev test lint format type-check clean build upload help

# Default target
help:
	@echo "Lemkin Image Verification Suite - Build Commands"
	@echo "==============================================="
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
	pytest --cov=src/lemkin_images --cov-report=html --cov-report=term-missing -v

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

# Image analysis examples
manipulation-example:
	@echo "Example: Detecting image manipulation"
	lemkin-images detect-manipulation suspicious_photo.jpg --output analysis.json

reverse-search-example:
	@echo "Example: Reverse image search"
	lemkin-images reverse-search evidence_photo.jpg --engines "google,bing" --output search.json

metadata-example:
	@echo "Example: Analyzing image metadata"
	lemkin-images analyze-metadata photo.jpg --show-exif --output metadata.json

geolocation-example:
	@echo "Example: Image geolocation"
	lemkin-images geolocate location_photo.jpg --output geolocation.json

comprehensive-example:
	@echo "Example: Comprehensive image analysis"
	lemkin-images comprehensive-analysis photo.jpg --include-search --include-geolocation --output full_analysis.json

# Batch processing
batch-manipulation:
	@echo "Example: Batch manipulation detection"
	lemkin-images batch-process tests/fixtures/images/ results/ --type manipulation

batch-comprehensive:
	@echo "Example: Batch comprehensive analysis"
	lemkin-images batch-process tests/fixtures/images/ results/ --type comprehensive

# Verification
verify-install:
	@echo "Verifying installation..."
	lemkin-images --help
	python -c "import lemkin_images; print(f'✅ lemkin-images v{lemkin_images.__version__} installed successfully')"
	python -c "from lemkin_images import ManipulationDetector, ReverseImageSearcher, GeolocationHelper, MetadataForensics; print('✅ All core components imported successfully')"

# Dependencies check
deps-check:
	@echo "Checking critical dependencies..."
	python -c "import cv2; print('✅ opencv (image processing)')"
	python -c "import PIL; print('✅ Pillow (image manipulation)')"
	python -c "import numpy; print('✅ numpy (numerical computing)')"
	python -c "import scikit_image; print('✅ scikit-image (image analysis)')"
	python -c "import imagehash; print('✅ imagehash (perceptual hashing)')"

# Performance testing
perf-test:
	@echo "Running performance tests on sample image data..."
	time lemkin-images detect-manipulation tests/fixtures/sample_image.jpg --output /tmp/image_perf.json

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