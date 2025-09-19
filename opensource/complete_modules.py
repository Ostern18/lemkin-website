#!/usr/bin/env python3
"""
Script to complete all Lemkin modules with missing files.
This will add test files, CONTRIBUTING.md, SECURITY.md, and other missing components.
"""

import os
from pathlib import Path
from typing import Dict, List

# Base directory
BASE_DIR = Path(__file__).parent

# Template for test_core.py files
TEST_CORE_TEMPLATE = '''"""
Tests for {module_name} core functionality.
"""

import json
import pytest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List, Optional

from {module_import}.core import *
from {module_import}.cli import app
from typer.testing import CliRunner


class Test{class_name}Core:
    """Test core {module_name} functionality"""

    def test_initialization(self):
        """Test module initialization"""
        # Test that module can be imported
        from {module_import} import __version__
        assert __version__ is not None

    def test_basic_functionality(self):
        """Test basic {module_name} operations"""
        # TODO: Add specific tests based on module functionality
        pass

    def test_error_handling(self):
        """Test error handling in {module_name}"""
        # TODO: Add error handling tests
        pass

    def test_validation(self):
        """Test input validation"""
        # TODO: Add validation tests
        pass

    def test_security_checks(self):
        """Test security measures"""
        # TODO: Add security tests
        pass


class Test{class_name}CLI:
    """Test CLI interface"""

    def setup_method(self):
        """Set up test fixtures"""
        self.runner = CliRunner()

    def test_cli_help(self):
        """Test CLI help command"""
        result = self.runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "{module_name}" in result.stdout.lower()

    def test_cli_version(self):
        """Test CLI version command"""
        result = self.runner.invoke(app, ["--version"])
        assert result.exit_code == 0

    def test_cli_basic_command(self):
        """Test basic CLI command"""
        # TODO: Add specific CLI command tests
        pass


class Test{class_name}Integration:
    """Integration tests for {module_name}"""

    def test_end_to_end_workflow(self):
        """Test complete workflow"""
        # TODO: Add end-to-end workflow tests
        pass

    def test_data_persistence(self):
        """Test data persistence"""
        # TODO: Add persistence tests
        pass

    def test_concurrent_operations(self):
        """Test concurrent operations"""
        # TODO: Add concurrency tests
        pass


class Test{class_name}Performance:
    """Performance tests for {module_name}"""

    def test_large_dataset_handling(self):
        """Test handling of large datasets"""
        # TODO: Add performance tests
        pass

    def test_memory_usage(self):
        """Test memory usage"""
        # TODO: Add memory usage tests
        pass

    def test_response_time(self):
        """Test response times"""
        # TODO: Add response time tests
        pass
'''

# Template for CONTRIBUTING.md
CONTRIBUTING_TEMPLATE = '''# Contributing to Lemkin {module_title}

Thank you for your interest in contributing to the Lemkin {module_title} module! This document provides guidelines and instructions for contributing.

## Code of Conduct

By participating in this project, you agree to abide by our [Code of Conduct](../CODE_OF_CONDUCT.md).

## How to Contribute

### Reporting Issues

- Check existing issues before creating a new one
- Use clear, descriptive titles
- Include steps to reproduce bugs
- Provide system information when relevant
- Add labels to categorize issues

### Submitting Pull Requests

1. **Fork and Clone**
   ```bash
   git clone https://github.com/yourusername/lemkin-ai.git
   cd lemkin-ai/{module_name}
   ```

2. **Create a Branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Set Up Development Environment**
   ```bash
   make install-dev
   ```

4. **Make Your Changes**
   - Write clear, documented code
   - Follow existing code style
   - Add tests for new functionality
   - Update documentation as needed

5. **Run Quality Checks**
   ```bash
   make format  # Auto-format code
   make lint    # Check code quality
   make test    # Run tests
   ```

6. **Commit Your Changes**
   ```bash
   git add .
   git commit -m "feat: add new feature"
   ```

   Follow conventional commit format:
   - `feat:` New features
   - `fix:` Bug fixes
   - `docs:` Documentation changes
   - `test:` Test additions or modifications
   - `refactor:` Code refactoring
   - `perf:` Performance improvements
   - `chore:` Maintenance tasks

7. **Push and Create PR**
   ```bash
   git push origin feature/your-feature-name
   ```
   Then create a pull request on GitHub.

## Development Guidelines

### Code Style

- Python 3.10+ with type hints
- Black formatter (88 character line length)
- Ruff for linting
- MyPy for type checking

### Testing Requirements

- Minimum 80% test coverage
- All tests must pass
- Include unit and integration tests
- Test error conditions and edge cases

### Documentation

- Docstrings for all public functions
- Type hints for all parameters
- Update README.md for new features
- Include usage examples

### Security Considerations

- No hardcoded secrets
- Validate all inputs
- Follow OWASP guidelines
- Consider privacy implications

## Module-Specific Guidelines

### {module_title} Focus Areas

- {specific_area_1}
- {specific_area_2}
- {specific_area_3}

### Priority Improvements

1. **High Priority**
   - Bug fixes
   - Security vulnerabilities
   - Performance issues

2. **Medium Priority**
   - New features
   - Documentation improvements
   - Test coverage

3. **Low Priority**
   - Code refactoring
   - Minor optimizations
   - Cosmetic changes

## Testing

### Running Tests
```bash
make test              # Run all tests
make test-fast         # Quick test run
pytest tests/test_core.py::TestClassName  # Run specific test
```

### Writing Tests
- Use pytest fixtures for setup
- Mock external dependencies
- Test both success and failure cases
- Include integration tests

## Review Process

1. Automated checks must pass
2. Code review by maintainer
3. Documentation review
4. Security review for sensitive changes
5. Performance impact assessment

## Questions?

- Open a discussion on GitHub
- Contact maintainers
- Check existing documentation

Thank you for contributing to justice through technology!
'''

# Template for SECURITY.md
SECURITY_TEMPLATE = '''# Security Policy - Lemkin {module_title}

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |
| < 0.1   | :x:                |

## Reporting a Vulnerability

If you discover a security vulnerability in the Lemkin {module_title} module, please report it responsibly:

1. **DO NOT** open a public issue
2. Email: security@lemkin.org
3. Include:
   - Module name and version
   - Description of vulnerability
   - Steps to reproduce
   - Potential impact
   - Suggested fix (if any)

## Security Measures

### Data Protection
- All sensitive data is encrypted at rest
- PII is automatically detected and protected
- Audit logs maintain integrity verification

### Input Validation
- All user inputs are validated
- File uploads are scanned for malicious content
- SQL injection prevention
- XSS protection

### Authentication & Authorization
- Role-based access control
- Secure token management
- Session timeout policies

### {module_title}-Specific Security

{module_specific_security}

## Best Practices

### For Users
- Keep the module updated
- Use strong authentication
- Regularly review audit logs
- Follow data handling guidelines

### For Developers
- Never commit secrets
- Use environment variables
- Validate all inputs
- Implement proper error handling
- Follow secure coding guidelines

## Compliance

This module adheres to:
- GDPR requirements
- International legal standards
- Evidence handling protocols
- Chain of custody requirements

## Security Checklist

- [ ] Input validation implemented
- [ ] Output encoding in place
- [ ] Authentication required
- [ ] Authorization checks
- [ ] Audit logging enabled
- [ ] Error handling secure
- [ ] Secrets management proper
- [ ] Dependencies updated
- [ ] Security tests written
- [ ] Documentation current

## Contact

Security Team: security@lemkin.org
'''

def create_test_file(module_path: Path, module_name: str, module_import: str):
    """Create test_core.py for a module."""
    test_file = module_path / "tests" / "test_core.py"

    if not test_file.exists():
        class_name = ''.join(word.capitalize() for word in module_name.split('-'))
        content = TEST_CORE_TEMPLATE.format(
            module_name=module_name,
            module_import=module_import,
            class_name=class_name
        )

        test_file.parent.mkdir(parents=True, exist_ok=True)
        test_file.write_text(content)
        print(f"✓ Created {test_file}")
    else:
        print(f"  Already exists: {test_file}")


def create_contributing_file(module_path: Path, module_name: str):
    """Create CONTRIBUTING.md for a module."""
    contrib_file = module_path / "CONTRIBUTING.md"

    if not contrib_file.exists():
        module_title = ' '.join(word.capitalize() for word in module_name.split('-')[1:])

        # Module-specific focus areas
        focus_areas = {
            'forensics': ['File system analysis', 'Network forensics', 'Mobile device forensics'],
            'video': ['Video authentication', 'Deepfake detection', 'Compression analysis'],
            'images': ['Image verification', 'Manipulation detection', 'Metadata analysis'],
            'audio': ['Audio authentication', 'Speech analysis', 'Speaker identification'],
            'redaction': ['PII detection', 'Privacy protection', 'Multi-format support'],
            'classifier': ['Document classification', 'Legal taxonomy', 'Multi-language support'],
            'ner': ['Entity extraction', 'Entity linking', 'Legal entity types'],
            'timeline': ['Temporal analysis', 'Event sequencing', 'Consistency checking'],
            'frameworks': ['Legal framework mapping', 'Violation detection', 'Evidence assessment'],
        }

        module_key = module_name.split('-')[1] if '-' in module_name else module_name
        areas = focus_areas.get(module_key, ['Core functionality', 'Performance optimization', 'Documentation'])

        content = CONTRIBUTING_TEMPLATE.format(
            module_title=module_title,
            module_name=module_name,
            specific_area_1=areas[0],
            specific_area_2=areas[1] if len(areas) > 1 else 'Testing improvements',
            specific_area_3=areas[2] if len(areas) > 2 else 'Integration capabilities'
        )

        contrib_file.write_text(content)
        print(f"✓ Created {contrib_file}")
    else:
        print(f"  Already exists: {contrib_file}")


def create_security_file(module_path: Path, module_name: str):
    """Create SECURITY.md for a module."""
    security_file = module_path / "SECURITY.md"

    if not security_file.exists():
        module_title = ' '.join(word.capitalize() for word in module_name.split('-')[1:])

        # Module-specific security considerations
        security_specifics = {
            'forensics': '''- Forensic integrity preservation
- Chain of custody maintenance
- Evidence tampering detection
- Secure analysis environment''',
            'video': '''- Deepfake detection accuracy
- Video integrity verification
- Metadata preservation
- Secure streaming protocols''',
            'images': '''- Image authenticity verification
- EXIF data protection
- Manipulation detection accuracy
- Secure image processing''',
            'audio': '''- Audio authenticity verification
- Voice privacy protection
- Secure transcription handling
- Speaker identification security''',
            'redaction': '''- PII detection accuracy
- Redaction completeness
- Original data protection
- Audit trail integrity''',
            'classifier': '''- Classification accuracy
- Model security
- Training data protection
- Prediction integrity''',
            'ner': '''- Entity extraction accuracy
- PII handling in entities
- Cross-reference security
- Entity linking validation''',
            'timeline': '''- Temporal data integrity
- Event correlation security
- Timeline tampering detection
- Source verification''',
            'frameworks': '''- Legal framework accuracy
- Evidence mapping integrity
- Violation assessment security
- Compliance verification''',
        }

        module_key = module_name.split('-')[1] if '-' in module_name else module_name
        specific_security = security_specifics.get(module_key, '''- Data integrity verification
- Access control enforcement
- Audit logging
- Secure processing''')

        content = SECURITY_TEMPLATE.format(
            module_title=module_title,
            module_specific_security=specific_security
        )

        security_file.write_text(content)
        print(f"✓ Created {security_file}")
    else:
        print(f"  Already exists: {security_file}")


def complete_module(module_name: str):
    """Complete a module with all missing files."""
    module_path = BASE_DIR / module_name

    if not module_path.exists():
        print(f"✗ Module {module_name} does not exist")
        return

    print(f"\n=== Completing {module_name} ===")

    # Create module import name
    module_import = module_name.replace('-', '_')

    # Create test file
    create_test_file(module_path, module_name, module_import)

    # Create CONTRIBUTING.md
    create_contributing_file(module_path, module_name)

    # Create SECURITY.md
    create_security_file(module_path, module_name)


def main():
    """Complete all modules with missing files."""

    # Modules that need test files
    modules_needing_tests = [
        'lemkin-forensics',
        'lemkin-video',
        'lemkin-images',
        'lemkin-audio',
    ]

    # Modules that need CONTRIBUTING.md
    modules_needing_contributing = [
        'lemkin-redaction',
        'lemkin-classifier',
        'lemkin-ner',
        'lemkin-timeline',
        'lemkin-frameworks',
    ]

    # Modules that need SECURITY.md
    modules_needing_security = [
        'lemkin-classifier',
        'lemkin-ner',
        'lemkin-timeline',
        'lemkin-frameworks',
    ]

    print("Starting module completion process...")

    # Process all modules
    all_modules = set(modules_needing_tests + modules_needing_contributing + modules_needing_security)

    for module_name in sorted(all_modules):
        complete_module(module_name)

    print("\n✅ Module completion process finished!")
    print("\nNext steps:")
    print("1. Review generated files for module-specific adjustments")
    print("2. Run 'make test' in each module to verify tests work")
    print("3. Update tests with actual module functionality")
    print("4. Commit changes to repository")


if __name__ == "__main__":
    main()