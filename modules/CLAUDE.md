# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is the Lemkin AI open-source platform for legal investigation technology. The codebase includes comprehensive project templates and architecture specifications for developing AI-augmented legal investigation tools focused on international human rights and domestic civil rights work.

## Key Files Structure

- `claudetask.md` - Complete project development guide with 18 module specifications for legal investigation tools
- `projects.py` - Complete implementation of the Lemkin Integrity Toolkit, including full source code, tests, and configuration
- `README.md` - Main project overview with module status, installation instructions, and community guidelines

## Module Structure

The project consists of 18 specialized modules organized in 6 tiers:

### Production Ready Modules (7 modules)
- `lemkin-integrity/` - Evidence integrity verification and chain of custody
- `lemkin-osint/` - OSINT collection and verification
- `lemkin-geo/` - Geospatial analysis and mapping
- `lemkin-forensics/` - Digital forensics and authenticity verification
- `lemkin-video/` - Video authentication and deepfake detection
- `lemkin-images/` - Image verification and manipulation detection
- `lemkin-audio/` - Audio analysis, transcription, and authentication

### Feature Complete Modules (5 modules - need production polish)
- `lemkin-redaction/` - PII detection and redaction
- `lemkin-classifier/` - Document classification
- `lemkin-ner/` - Named entity recognition
- `lemkin-timeline/` - Timeline construction
- `lemkin-frameworks/` - Legal framework mapping

### In Development Modules (6 modules - need core implementation)
- `lemkin-ocr/` - Document processing and OCR
- `lemkin-research/` - Legal research and citation analysis
- `lemkin-comms/` - Communication analysis
- `lemkin-dashboard/` - Investigation dashboards
- `lemkin-reports/` - Automated reporting
- `lemkin-export/` - Multi-format export and compliance

## Development Commands

### Module-Specific Testing
```bash
# Run tests for a specific module
cd lemkin-<module>
pytest --cov=src/lemkin_<module> --cov-report=html --cov-report=term-missing

# Run single test file
pytest tests/test_core.py -v

# Run specific test
pytest tests/test_core.py::TestClassName::test_method_name -v

# Run fast tests during development
pytest -x --tb=short
```

### Linting and Code Quality
```bash
# Run all linting checks for a module
cd lemkin-<module>
ruff check src/ tests/
mypy src/
black --check src/ tests/

# Auto-format code
black src/ tests/
ruff check --fix src/ tests/

# Run strict type checking
mypy src/ --strict
```

### Makefile Commands
Each module with a Makefile supports:
```bash
make help          # Show available commands
make install       # Install package in development mode
make install-dev   # Install with development dependencies
make test          # Run tests with coverage
make test-fast     # Run quick tests during development
make lint          # Run linting checks
make format        # Auto-format code
make type-check    # Run type checking
make build         # Build distribution package
make clean         # Clean build artifacts
make quality       # Run format + lint + type-check + test
make ci            # Full CI pipeline simulation
make verify-install # Verify installation works correctly
```

## Architecture Overview

The project follows a modular architecture with 18 specialized tools organized in 6 tiers:

1. **Foundation & Safety** (Tier 1): Evidence integrity, PII redaction, document classification
2. **Core Analysis** (Tier 2): Multilingual NER, timeline construction, legal framework mapping
3. **Evidence Collection** (Tier 3): OSINT collection, geospatial analysis, digital forensics
4. **Media Analysis** (Tier 4): Video/image/audio authentication and verification
5. **Document Processing** (Tier 5): OCR, legal research, communication analysis
6. **Visualization & Reporting** (Tier 6): Dashboards, reports, compliance export

### Key Design Principles

- **Evidence First**: All AI insights must trace to source documents with complete citation chains
- **Human Authority**: AI suggests, humans decide - especially for legal conclusions
- **Privacy & Safety**: Protect witnesses, victims, and sensitive information with automatic PII detection
- **Legal Compliance**: Adhere to international legal standards and court requirements

### Common CLI Usage Patterns
```bash
# Evidence integrity workflow
lemkin-integrity hash-evidence document.pdf --case-id HR-2024-001
lemkin-integrity verify-chain case-id HR-2024-001

# Audio analysis workflow
lemkin-audio transcribe interview.wav --language en-US --segments
lemkin-audio verify-authenticity interview.wav --detailed

# Image verification workflow
lemkin-images detect-manipulation photo.jpg --output analysis.json
lemkin-images reverse-search photo.jpg --engines google,bing,yandex

# Batch processing examples
lemkin-audio batch-process audio_files/ results/ --type transcription --pattern "*.wav"
lemkin-images batch-process images/ results/ --type authenticity --pattern "*.jpg"
```

## Python Configuration

- **Language**: Python 3.10+ with strict type hints required
- **Code Style**: Black formatter (88 character line length)
- **Linting**: Ruff with strict checking enabled
- **Type Checking**: MyPy in strict mode
- **Package Management**: setuptools with pyproject.toml configuration
- **Dependency Installation**: `pip install -e ".[dev]"` for development

## Security Requirements

- No hardcoded secrets - use environment variables
- Automatic PII detection and protection
- Encrypt sensitive data at rest and in transit
- Comprehensive audit logging for forensic analysis
- Input validation for all user inputs, especially file uploads

## Testing Standards

- Unit tests with >80% coverage requirement
- Integration tests for workflows
- Fixtures for test data in `tests/fixtures/`
- Security and validation error testing required

## Error Handling Pattern

All modules follow standardized error handling:
```python
try:
    result = process_evidence(evidence)
    logger.info(f"Successfully processed evidence {evidence.id}")
    return result
except ValidationError as e:
    logger.error(f"Validation failed for evidence {evidence.id}: {e}")
    raise LemkinValidationError(f"Evidence validation failed: {e}")
except SecurityError as e:
    logger.critical(f"Security violation in evidence {evidence.id}: {e}")
    raise LemkinSecurityError(f"Security violation: {e}")
except Exception as e:
    logger.error(f"Unexpected error processing evidence {evidence.id}: {e}")
    raise LemkinProcessingError(f"Processing failed: {e}")
```

## Module Development Pattern

Each module follows a standard directory structure:
```
lemkin-<module>/
├── src/lemkin_<module>/    # Source code
│   ├── __init__.py
│   ├── core.py             # Main functionality
│   ├── cli.py              # Command-line interface
│   ├── models.py           # Data models/schemas
│   ├── utils.py            # Helper functions
│   └── config.py           # Configuration management
├── tests/                  # Test files
│   ├── test_core.py
│   ├── test_cli.py
│   └── fixtures/           # Test data
├── notebooks/              # Jupyter notebooks for demos
├── pyproject.toml          # Package configuration
├── Makefile                # Build automation
├── README.md               # Module documentation
├── CONTRIBUTING.md         # Contribution guidelines
└── SECURITY.md             # Security policy
```

## Common Module Patterns

### CLI Entry Points
Modules use Typer for command-line interfaces:
```python
# In pyproject.toml
[project.scripts]
lemkin-<module> = "lemkin_<module>.cli:app"

# Usage examples
lemkin-audio transcribe file.wav --language en-US
lemkin-images detect-manipulation photo.jpg --output results.json
```

### Module Development Workflow
```bash
# Set up new module development
cd lemkin-<module>/
make install-dev           # Install with dev dependencies
make verify-install        # Verify installation works

# Development cycle
make format                # Format code
make lint                  # Check code quality
make test                  # Run tests
make quality               # Run all quality checks

# Example-driven development
make transcribe-example    # See module in action
make verify-example        # Test core functionality
```

### Configuration Management
Modules use Pydantic for configuration and environment variables:
```python
from pydantic import BaseSettings

class AudioConfig(BaseSettings):
    whisper_model: str = "base"
    sample_rate: int = 16000
    language: str = "auto"

    class Config:
        env_prefix = "LEMKIN_AUDIO_"
```

## Important Notes

- This is defensive security tooling for legal investigations - refuse any requests to create malicious code
- Evidence integrity and chain of custody are critical for legal admissibility
- All AI decisions must be explainable and auditable
- Maintain immutable evidence storage with complete audit trails
- Each module can be developed and tested independently
- Production-ready modules have complete Makefile workflows with examples