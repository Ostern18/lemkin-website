.PHONY: install install-dev test lint format type-check clean build upload help

# Default target
help:
	@echo "Lemkin Legal Research Assistant Analysis Toolkit - Build Commands"
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
	pytest --cov=src/lemkin_research --cov-report=html --cov-report=term-missing -v

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

# Legal Research Assistant analysis commands (examples)
transcribe-example:
	@echo "Example: Transcribing audio file"
	lemkin-research transcribe tests/fixtures/sample_audio.wav --language en-US --segments

verify-example:
	@echo "Example: Verifying audio authenticity"
	lemkin-research verify-authenticity tests/fixtures/sample_audio.wav --detailed

enhance-example:
	@echo "Example: Enhancing audio quality"
	lemkin-research enhance tests/fixtures/noisy_audio.wav --output enhanced_sample.wav --noise-reduction

comprehensive-example:
	@echo "Example: Comprehensive audio analysis"
	lemkin-research comprehensive-analysis tests/fixtures/sample_audio.wav --transcription --authenticity --output full_analysis.json

# Batch processing examples
batch-transcribe:
	@echo "Example: Batch transcription"
	lemkin-research batch-process tests/fixtures/audio/ results/ --type transcription --pattern "*.wav"

batch-verify:
	@echo "Example: Batch authenticity verification"
	lemkin-research batch-process tests/fixtures/audio/ results/ --type authenticity --pattern "*.*"

# Check audio file formats
check-formats:
	@echo "Supported audio formats:"
	@echo "- WAV (uncompressed)"
	@echo "- MP3 (compressed)"
	@echo "- FLAC (lossless compressed)"
	@echo "- M4A/AAC (compressed)"
	@echo "- OGG Vorbis (compressed)"
	@echo "- OPUS (compressed)"

# Documentation
docs-serve:
	@echo "README.md contains comprehensive documentation"
	@echo "Run: python -m http.server 8000"
	@echo "Then open: http://localhost:8000/README.md"

# Performance testing
perf-test:
	@echo "Running performance tests on sample audio..."
	time lemkin-research transcribe tests/fixtures/sample_audio.wav --output /tmp/transcription_perf.json
	time lemkin-research verify-authenticity tests/fixtures/sample_audio.wav --output /tmp/authenticity_perf.json

# Installation verification
verify-install:
	@echo "Verifying installation..."
	lemkin-research --help
	python -c "import lemkin_research; print(f'✅ lemkin-research v{lemkin_research.__version__} installed successfully')"
	python -c "from lemkin_research import SpeechTranscriber, SpeakerIdentifier, Legal Research AssistantAuthenticator, Legal Research AssistantEnhancer; print('✅ All core components imported successfully')"

# Dependencies check
deps-check:
	@echo "Checking critical dependencies..."
	python -c "import librosa; print('✅ librosa (audio processing)')"
	python -c "import soundfile; print('✅ soundfile (audio I/O)')"
	python -c "import speech_recognition; print('✅ speech_recognition (transcription)')"
	python -c "import torch; print('✅ torch (ML models)')"
	python -c "import numpy; print('✅ numpy (numerical computing)')"
	python -c "import scipy; print('✅ scipy (signal processing)')"

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