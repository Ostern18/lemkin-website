# Contributing to Lemkin Investigation Dashboard Analysis Toolkit

Thank you for your interest in contributing to the Lemkin Investigation Dashboard Analysis Toolkit! This project is part of the Lemkin AI ecosystem designed to democratize legal investigation technology for human rights investigators, prosecutors, and civil rights attorneys.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Environment](#development-environment)
- [Contribution Guidelines](#contribution-guidelines)
- [Pull Request Process](#pull-request-process)
- [Testing](#testing)
- [Documentation](#documentation)
- [Security](#security)
- [Community](#community)

## Code of Conduct

This project adheres to the Lemkin AI Code of Conduct. By participating, you are expected to uphold this code. Please report unacceptable behavior to conduct@lemkin.org.

### Our Pledge

We pledge to make participation in our project a harassment-free experience for everyone, regardless of age, body size, disability, ethnicity, gender identity and expression, level of experience, nationality, personal appearance, race, religion, or sexual identity and orientation.

### Our Standards

Examples of behavior that contributes to creating a positive environment include:

* Using welcoming and inclusive language
* Being respectful of differing viewpoints and experiences
* Gracefully accepting constructive criticism
* Focusing on what is best for the community and human rights work
* Showing empathy towards other community members

## Getting Started

### Prerequisites

- Python 3.10 or higher
- Git
- Basic knowledge of audio processing concepts
- Understanding of legal investigation workflows (helpful but not required)

### First Contribution

1. **Find an Issue**: Look for issues labeled `good first issue` or `help wanted`
2. **Introduce Yourself**: Comment on the issue to let others know you're working on it
3. **Ask Questions**: Don't hesitate to ask for clarification or guidance

### Types of Contributions Welcome

- **Bug Fixes**: Help identify and fix issues in audio processing
- **Feature Development**: Implement new audio analysis capabilities
- **Documentation**: Improve guides, examples, and API documentation
- **Testing**: Add test cases and improve test coverage
- **Performance**: Optimize audio processing algorithms
- **Accessibility**: Make the toolkit more accessible to non-technical users
- **Translation**: Help translate documentation for international users
- **Legal Expertise**: Provide guidance on legal evidence standards

## Development Environment

### Setup

```bash
# Clone the repository
git clone https://github.com/lemkin-org/lemkin-dashboard.git
cd lemkin-dashboard

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install

# Verify installation
make verify-install
```

### Development Dependencies

The `[dev]` extra includes:
- `pytest` - Testing framework
- `pytest-cov` - Code coverage
- `black` - Code formatting
- `ruff` - Linting
- `mypy` - Type checking
- `pre-commit` - Git hooks

### Testing Your Setup

```bash
# Run tests
make test

# Run linting
make lint

# Format code
make format

# Type checking
make type-check

# All checks
make quality
```

## Contribution Guidelines

### Branch Naming

Use descriptive branch names:
- `feature/speaker-identification-improvement`
- `bugfix/transcription-timeout-error`
- `docs/api-reference-update`
- `test/add-authenticity-tests`

### Commit Messages

Follow conventional commit format:
```
type(scope): description

[optional body]

[optional footer]
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `test`: Test additions or changes
- `refactor`: Code refactoring
- `perf`: Performance improvements
- `style`: Code style changes
- `ci`: CI/CD changes

Example:
```
feat(transcription): add multi-language support for Arabic

- Implement Arabic language detection
- Add RTL text handling for transcription output
- Update CLI with Arabic language option

Closes #123
```

### Code Style

We use automated formatting and linting:

```bash
# Format code (runs automatically on commit)
black src/ tests/

# Lint code
ruff check src/ tests/

# Type checking
mypy src/
```

### Code Standards

- **Type Hints**: All functions must have complete type hints
- **Docstrings**: Use Google-style docstrings for all public functions
- **Error Handling**: Implement comprehensive error handling with informative messages
- **Logging**: Use structured logging for debugging and audit trails
- **Security**: Follow security best practices, especially for audio file processing

Example function:

```python
def transcribe_audio(
    audio_path: Path,
    language: Optional[LanguageCode] = None,
    confidence_threshold: float = 0.8
) -> TranscriptionResult:
    """Transcribe audio file to text with confidence scoring.

    Args:
        audio_path: Path to the audio file to transcribe
        language: Target language for transcription (auto-detect if None)
        confidence_threshold: Minimum confidence score for segments

    Returns:
        TranscriptionResult containing text, timestamps, and metadata

    Raises:
        Investigation DashboardProcessingError: If audio file cannot be processed
        ValidationError: If confidence_threshold is out of range

    Example:
        >>> result = transcribe_audio(Path("interview.wav"), LanguageCode.EN_US)
        >>> print(f"Transcribed: {result.full_text}")
    """
```

## Pull Request Process

### Before Submitting

1. **Run Tests**: Ensure all tests pass (`make test`)
2. **Code Quality**: Fix any linting or type checking issues (`make quality`)
3. **Documentation**: Update relevant documentation
4. **Security**: Review code for security implications
5. **Performance**: Consider performance impact of changes

### PR Description Template

```markdown
## Description
Brief description of changes and why they're needed.

## Type of Change
- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update

## Testing
- [ ] Unit tests added/updated
- [ ] Integration tests added/updated
- [ ] Manual testing completed

## Investigation Dashboard Processing Considerations
- [ ] Tested with various audio formats (WAV, MP3, FLAC, etc.)
- [ ] Verified performance with large audio files
- [ ] Checked memory usage and cleanup
- [ ] Tested edge cases (silence, noise, corrupted files)

## Security Considerations
- [ ] No hardcoded secrets or sensitive data
- [ ] Input validation implemented
- [ ] Secure file handling practices followed
- [ ] PII protection measures considered

## Checklist
- [ ] Code follows the style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] Tests added and passing
- [ ] No breaking changes or migration guide provided
```

### Review Process

1. **Automated Checks**: CI/CD pipeline runs tests and quality checks
2. **Peer Review**: At least one maintainer reviews the code
3. **Security Review**: Security-sensitive changes get additional review
4. **Legal Consideration**: Changes affecting legal evidence handling get legal review

## Testing

### Test Structure

```
tests/
├── unit/              # Unit tests for individual components
├── integration/       # Integration tests for workflows
├── performance/       # Performance and load tests
└── fixtures/          # Test data and audio samples
    ├── audio/         # Sample audio files
    └── data/          # Expected results and configurations
```

### Writing Tests

```python
import pytest
from pathlib import Path
from lemkin_dashboard import SpeechTranscriber

class TestSpeechTranscriber:
    def setup_method(self):
        """Set up test fixtures before each test."""
        self.transcriber = SpeechTranscriber()
        self.test_audio = Path("tests/fixtures/audio/sample.wav")

    def test_transcribe_english_audio(self):
        """Test transcription of English audio."""
        result = self.transcriber.transcribe_audio(self.test_audio)

        assert result.full_text
        assert len(result.segments) > 0
        assert result.total_duration > 0

    def test_transcribe_nonexistent_file(self):
        """Test error handling for missing audio file."""
        with pytest.raises(FileNotFoundError):
            self.transcriber.transcribe_audio(Path("nonexistent.wav"))
```

### Test Data Ethics

- **No Real Data**: Never use real investigation audio in tests
- **Synthetic Data**: Use generated or synthetic audio samples
- **Privacy Protection**: Ensure test data doesn't contain PII
- **Legal Compliance**: Follow data protection regulations

## Documentation

### Types of Documentation

1. **API Documentation**: Docstrings in code
2. **User Guides**: Step-by-step tutorials
3. **Examples**: Real-world usage scenarios
4. **Architecture**: Technical design documents

### Documentation Standards

- **Clear Examples**: Include working code examples
- **Real Scenarios**: Use legal investigation use cases
- **Safety Warnings**: Highlight security and privacy considerations
- **Performance Notes**: Include performance characteristics
- **Legal Context**: Explain legal relevance where applicable

### Building Documentation

```bash
# Install documentation dependencies
pip install -e ".[docs]"

# Build documentation locally
make docs-build

# Serve documentation
make docs-serve
```

## Security

### Security-First Development

- **Threat Modeling**: Consider attack vectors for audio processing
- **Input Validation**: Validate all audio file inputs
- **Secure Processing**: Use secure temporary files and cleanup
- **Dependency Scanning**: Regularly update and scan dependencies

### Reporting Security Issues

Please report security vulnerabilities to security@lemkin.org, not through public issues. See [SECURITY.md](SECURITY.md) for details.

## Community

### Communication Channels

- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: General questions and community discussion
- **Discord**: Real-time chat with developers and users
- **Monthly Calls**: Community calls for major updates and planning

### Getting Help

1. **Documentation**: Check existing documentation first
2. **Search Issues**: Look for similar questions or problems
3. **Ask Questions**: Open a discussion or issue if you can't find answers
4. **Community**: Join our Discord for real-time help

### Recognition

We recognize contributions through:
- **Contributor List**: All contributors are listed in the project
- **Release Notes**: Significant contributions are highlighted
- **Community Spotlights**: Regular features of outstanding contributors
- **Conference Talks**: Opportunities to present work at conferences

## Legal and Ethical Considerations

### Human Rights Focus

This project exists to support legitimate legal investigations and human rights work. Contributors should:

- Understand the human rights context of the work
- Consider the impact on vulnerable populations
- Prioritize privacy and security of investigators and subjects
- Follow ethical guidelines for AI development in legal contexts

### Responsible Development

- **Privacy by Design**: Build privacy protection into features
- **Transparency**: Make AI decision processes explainable
- **Fairness**: Consider bias and fairness in audio analysis
- **Accountability**: Enable audit trails for legal proceedings

## Questions?

Don't hesitate to reach out:

- **Technical Questions**: Open a GitHub Discussion
- **Security Concerns**: Email security@lemkin.org
- **Community**: Join our Discord server
- **Legal/Ethical**: Email ethics@lemkin.org

Thank you for helping make legal investigation technology accessible to those who need it most!

---

*By contributing to this project, you help support human rights investigators, civil rights attorneys, and public defenders around the world.*