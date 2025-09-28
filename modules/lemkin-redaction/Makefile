.PHONY: help install test lint format clean build deploy demo

help:
	@echo "Available commands:"
	@echo "  install     Install development dependencies"
	@echo "  test        Run all tests"
	@echo "  lint        Run linting checks"
	@echo "  format      Format code"
	@echo "  clean       Clean build artifacts"
	@echo "  build       Build distribution packages"
	@echo "  deploy      Deploy to PyPI"
	@echo "  demo        Run redaction demo"

install:
	pip install -e ".[dev]"
	pre-commit install
	python -m spacy download en_core_web_sm

test:
	pytest --cov=src/lemkin_redaction --cov-report=html --cov-report=term-missing

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
	python -c "
from lemkin_redaction import PIIRedactor, RedactionConfig, EntityType
import tempfile
from pathlib import Path

# Create sample redaction demo
config = RedactionConfig(
    entity_types=[EntityType.PERSON, EntityType.EMAIL, EntityType.PHONE],
    min_confidence=0.7
)

redactor = PIIRedactor(config)

# Sample text with PII
sample_text = '''
Dr. Sarah Johnson contacted patient John Smith at john.smith@email.com 
regarding his medical appointment. Please call back at 555-123-4567.
The patient's SSN is 123-45-6789 for insurance verification.
'''

print('=== PII Redaction Demo ===')
print('Original text:')
print(sample_text)

result = redactor.redact_text(sample_text.strip())

print(f'\nEntities detected: {result.total_entities}')
print(f'Entities redacted: {result.redacted_count}')
print(f'Processing time: {result.processing_time:.3f}s')

print('\nDetected entities:')
for entity in result.entities_detected:
    print(f'  - {entity.entity_type.value}: \"{entity.text}\" (confidence: {entity.confidence:.2f})')

# Save redacted version to temp file
with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
    redacted_result = redactor.redact_text(sample_text.strip(), Path(f.name))
    redacted_content = Path(f.name).read_text()
    print(f'\nRedacted text:')
    print(redacted_content)
    Path(f.name).unlink()  # Clean up

print('\n=== Demo Complete ===')
"