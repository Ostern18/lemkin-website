# Contributing to Lemkin Frameworks

Thank you for your interest in contributing to the Lemkin Frameworks module! This document provides guidelines and instructions for contributing.

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
   cd lemkin-ai/lemkin-frameworks
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

### Frameworks Focus Areas

- Legal framework mapping
- Violation detection
- Evidence assessment

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
