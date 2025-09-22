.PHONY: install install-dev test lint format type-check clean build upload help

# Default target
help:
	@echo "Lemkin Video Authentication Toolkit - Build Commands"
	@echo "==================================================="
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
	pytest --cov=src/lemkin_video --cov-report=html --cov-report=term-missing -v

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

# Video analysis examples
deepfake-example:
	@echo "Example: Detecting deepfakes"
	lemkin-video detect-deepfake suspicious_video.mp4 --output deepfake_analysis.json

fingerprint-example:
	@echo "Example: Creating video fingerprint"
	lemkin-video fingerprint-video evidence.mp4 --output fingerprint.json

compression-example:
	@echo "Example: Analyzing compression artifacts"
	lemkin-video analyze-compression edited_video.mp4 --output compression_analysis.json

comprehensive-example:
	@echo "Example: Comprehensive video analysis"
	lemkin-video comprehensive-analysis video.mp4 --deepfake --compression --fingerprint --output full_analysis.json

# Batch processing
batch-deepfake:
	@echo "Example: Batch deepfake detection"
	lemkin-video batch-process tests/fixtures/videos/ results/ --type deepfake

batch-comprehensive:
	@echo "Example: Batch comprehensive analysis"
	lemkin-video batch-process tests/fixtures/videos/ results/ --type comprehensive

# Verification
verify-install:
	@echo "Verifying installation..."
	lemkin-video --help
	python -c "import lemkin_video; print(f'✅ lemkin-video v{lemkin_video.__version__} installed successfully')"
	python -c "from lemkin_video import DeepfakeDetector, VideoFingerprinter, CompressionAnalyzer; print('✅ All core components imported successfully')"

# Dependencies check
deps-check:
	@echo "Checking critical dependencies..."
	python -c "import cv2; print('✅ opencv (video processing)')"
	python -c "import numpy; print('✅ numpy (numerical computing)')"
	python -c "import torch; print('✅ torch (deep learning)')"
	python -c "import torchvision; print('✅ torchvision (computer vision)')"

# Performance testing
perf-test:
	@echo "Running performance tests on sample video data..."
	time lemkin-video detect-deepfake tests/fixtures/sample_video.mp4 --output /tmp/video_perf.json

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