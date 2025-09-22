# Contributing to Lemkin AI

Thank you for your interest in contributing to Lemkin AI! This project is designed to democratize legal investigation technology for human rights investigators, prosecutors, civil rights attorneys, and organizations working for justice and accountability.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Project Structure](#project-structure)
- [Development Environment](#development-environment)
- [Contribution Guidelines](#contribution-guidelines)
- [Pull Request Process](#pull-request-process)
- [Testing](#testing)
- [Documentation](#documentation)
- [Security](#security)
- [Community](#community)

## Code of Conduct

This project adheres to the Lemkin AI Code of Conduct. By participating, you are expected to uphold this code. Please report unacceptable behavior to conduct@lemkin.ai.

### Our Mission

We are committed to supporting legitimate legal investigations and human rights work through open-source technology. Our tools are designed to:

- Strengthen the capacity of human rights investigators
- Support civil rights attorneys and public defenders
- Enhance legal investigation capabilities for justice-seeking organizations
- Protect vulnerable populations through technology

## Getting Started

### Prerequisites

**For Website Development:**
- Node.js 18+ and npm
- Git
- Basic knowledge of React and TypeScript
- Understanding of legal technology requirements (helpful)

**For Tool Development:**
- Python 3.10+
- Git
- Familiarity with relevant domain (audio, image processing, NLP, etc.)
- Understanding of legal evidence standards (helpful)

### First Contribution

1. **Explore the Project**: Review the README and documentation
2. **Find an Issue**: Look for issues labeled `good first issue` or `help wanted`
3. **Introduce Yourself**: Comment on the issue to let others know you're working on it
4. **Ask Questions**: Don't hesitate to ask for clarification or guidance

### Types of Contributions Welcome

**Website and Documentation:**
- UI/UX improvements for the main website
- Documentation updates and examples
- Accessibility improvements
- Internationalization support
- Performance optimizations

**Core Tools:**
- Bug fixes across any of the analysis tools
- New feature development for existing tools
- Performance optimizations
- Test coverage improvements
- Security enhancements

**Legal and Domain Expertise:**
- Legal evidence standards guidance
- Workflow optimization for investigators
- User experience research for legal professionals
- Compliance and certification guidance

**Community and Ecosystem:**
- Tutorial creation
- Community management
- Translation work
- Outreach to legal organizations

## Project Structure

```
lemkin_website/
├── src/                          # React website source
├── public/                       # Static assets
├── resources/                    # Python tools and libraries
│   ├── lemkin-audio/            # Audio analysis toolkit
│   ├── lemkin-images/           # Image analysis toolkit
│   ├── lemkin-video/            # Video analysis toolkit
│   ├── lemkin-ner/              # Named entity recognition
│   ├── lemkin-forensics/        # Digital forensics tools
│   ├── lemkin-osint/            # OSINT analysis tools
│   ├── lemkin-integrity/        # Evidence integrity tools
│   ├── lemkin-geo/              # Geographic analysis
│   ├── lemkin-redaction/        # Data redaction tools
│   ├── lemkin-classifier/       # Document classification
│   ├── lemkin-comms/            # Communication analysis
│   ├── lemkin-dashboard/        # Investigation dashboard
│   ├── lemkin-export/           # Data export utilities
│   ├── lemkin-frameworks/       # Shared frameworks
│   ├── lemkin-ocr/              # Optical character recognition
│   ├── lemkin-reports/          # Report generation
│   ├── lemkin-research/         # Research tools
│   └── lemkin-timeline/         # Timeline analysis
├── .github/                     # CI/CD workflows
├── docker-compose.yml           # Development environment
└── Dockerfile                   # Production deployment
```

## Development Environment

### Website Development

```bash
# Clone the repository
git clone https://github.com/lemkin-org/lemkin_website.git
cd lemkin_website

# Install dependencies
npm install

# Start development server
npm start

# Run tests
npm test

# Build for production
npm run build
```

### Tool Development

```bash
# Navigate to specific tool (example: audio)
cd resources/lemkin-audio

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install

# Run tests
make test

# Run quality checks
make quality
```

### Docker Development

```bash
# Start full development environment
docker-compose --profile dev up

# Start with documentation server
docker-compose --profile dev --profile docs up

# Run security scanning
docker-compose --profile security run security-scanner
```

## Contribution Guidelines

### Branch Naming

Use descriptive branch names that indicate the type and scope of work:

- `feature/website-accessibility-improvements`
- `feature/audio-transcription-accuracy`
- `bugfix/image-processing-memory-leak`
- `docs/api-reference-updates`
- `security/input-validation-enhancement`

### Commit Messages

Follow conventional commit format for clear, searchable history:

```
type(scope): description

[optional body]

[optional footer]
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `test`: Test additions or changes
- `refactor`: Code refactoring
- `perf`: Performance improvements
- `security`: Security improvements
- `ci`: CI/CD changes
- `style`: Code style changes

**Examples:**
```
feat(website): add dark mode toggle for accessibility

- Implement system preference detection
- Add manual toggle in navigation
- Update all components for dark mode support
- Add accessibility testing for color contrast

Closes #234

fix(audio): resolve memory leak in long audio processing

- Fix buffer cleanup in SpeechTranscriber
- Add automated memory testing
- Update documentation with memory usage guidelines

Fixes #345

security(images): add input validation for image file processing

- Implement file type validation
- Add size limits for uploaded images
- Enhance error handling for malformed files
- Update security documentation

Addresses security advisory SA-2024-001
```

### Code Style

**Website (TypeScript/React):**
- Use TypeScript strict mode
- Follow React best practices and hooks patterns
- Use ESLint and Prettier for code formatting
- Implement comprehensive accessibility features
- Follow the established design system

**Tools (Python):**
- Follow PEP 8 style guidelines
- Use type hints for all functions
- Implement comprehensive error handling
- Use structured logging for audit trails
- Follow security best practices

```bash
# Website formatting
npm run lint
npm run format

# Tool formatting (example: audio)
cd resources/lemkin-audio
make format
make lint
make type-check
```

## Pull Request Process

### Pre-submission Checklist

**For Website Changes:**
- [ ] Tests pass (`npm test`)
- [ ] TypeScript compilation succeeds
- [ ] Accessibility tests pass
- [ ] Performance impact assessed
- [ ] Documentation updated

**For Tool Changes:**
- [ ] Tests pass (`make test`)
- [ ] Quality checks pass (`make quality`)
- [ ] Security scan clean
- [ ] Performance impact assessed
- [ ] Documentation updated

### PR Description Template

```markdown
## Description
Brief description of changes and their purpose in supporting legal investigations.

## Type of Change
- [ ] Website improvement (UI/UX, performance, accessibility)
- [ ] New tool feature (analysis capability, workflow improvement)
- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] Breaking change (fix or feature causing existing functionality changes)
- [ ] Documentation update
- [ ] Security enhancement

## Legal Technology Considerations
- [ ] Maintains evidence integrity standards
- [ ] Preserves audit trail capabilities
- [ ] Considers user privacy and security
- [ ] Supports legal workflow requirements
- [ ] Addresses accessibility for diverse users

## Testing
- [ ] Unit tests added/updated
- [ ] Integration tests verified
- [ ] Manual testing completed
- [ ] Cross-browser testing (website changes)
- [ ] Performance testing completed

## Security Review
- [ ] No hardcoded secrets or sensitive data
- [ ] Input validation implemented where applicable
- [ ] Secure coding practices followed
- [ ] PII protection measures considered
- [ ] Vulnerability scanning completed

## Documentation
- [ ] README updated if needed
- [ ] API documentation updated
- [ ] User guides updated
- [ ] Security implications documented

## Checklist
- [ ] Code follows project style guidelines
- [ ] Self-review completed
- [ ] Peer review requested
- [ ] CI checks passing
- [ ] Breaking changes documented with migration guide
```

### Review Process

1. **Automated Checks**: CI/CD pipeline validates code quality, security, and tests
2. **Peer Review**: At least one maintainer reviews code and approach
3. **Security Review**: Security-sensitive changes receive additional security review
4. **Legal Technology Review**: Changes affecting evidence handling receive domain expert review
5. **Community Feedback**: Significant changes may be discussed in community channels

## Testing

### Website Testing

```bash
# Unit and integration tests
npm test

# E2E testing with Playwright
npm run test:e2e

# Accessibility testing
npm run test:a11y

# Performance testing
npm run test:lighthouse
```

### Tool Testing

```bash
# Example: Testing audio toolkit
cd resources/lemkin-audio

# Unit tests
make test

# Performance tests
make perf-test

# Security tests
make security-scan

# All quality checks
make quality
```

### Test Data Ethics

- **No Real Data**: Never use actual investigation data in tests
- **Synthetic Data**: Use generated or anonymized test data
- **Privacy Protection**: Ensure test data doesn't contain PII
- **Legal Compliance**: Follow data protection regulations
- **Audit Trail**: Maintain records of test data sources and modifications

## Documentation

### Types of Documentation

1. **User Documentation**: How to use tools and website features
2. **Developer Documentation**: API references and architecture guides
3. **Legal Documentation**: Evidence handling standards and compliance
4. **Security Documentation**: Security procedures and threat models

### Documentation Standards

- **Clear Examples**: Include working code examples with real-world context
- **Legal Context**: Explain legal relevance and evidence standards
- **Security Warnings**: Highlight security and privacy considerations
- **Accessibility**: Ensure documentation is accessible to all users
- **Internationalization**: Consider translation needs for global users

## Security

### Security-First Development

All contributions must consider security implications:

- **Threat Modeling**: Consider attack vectors relevant to legal investigations
- **Input Validation**: Validate and sanitize all inputs
- **Secure Processing**: Use secure temporary storage and cleanup
- **Dependency Management**: Keep dependencies updated and scanned
- **Audit Logging**: Maintain comprehensive audit trails

### Reporting Security Issues

**Do not report security vulnerabilities through public issues.**

Report security issues to: security@lemkin.ai

See [SECURITY.md](SECURITY.md) for detailed security procedures.

## Community

### Communication Channels

- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: General questions and community discussion
- **Discord**: Real-time chat with developers and users
- **Monthly Community Calls**: Updates, planning, and collaboration
- **Legal Technology Webinars**: Domain-specific discussions

### Getting Help

1. **Documentation**: Check existing documentation and examples
2. **Search Issues**: Look for similar questions or problems
3. **Community Discussion**: Open a GitHub Discussion
4. **Real-time Help**: Join our Discord community
5. **Expert Consultation**: Request legal technology expert input

### Recognition and Attribution

We recognize contributions through:

- **Contributor Recognition**: All contributors listed in project documentation
- **Release Highlights**: Significant contributions featured in release notes
- **Community Spotlights**: Regular features of outstanding community members
- **Conference Opportunities**: Speaking opportunities at legal technology conferences
- **Research Collaboration**: Academic and research partnership opportunities

## Legal and Ethical Considerations

### Human Rights Mission

This project exists to support legitimate legal investigations and human rights work. Contributors should:

- Understand the human rights context and impact of their work
- Consider implications for vulnerable populations
- Prioritize privacy and security of investigators and subjects
- Follow ethical guidelines for AI/ML development in legal contexts
- Respect cultural and jurisdictional differences in legal systems

### Responsible Technology Development

- **Privacy by Design**: Build privacy protection into all features
- **Algorithmic Transparency**: Make AI decision processes explainable
- **Bias Mitigation**: Consider and address potential bias in analysis tools
- **Accountability**: Enable comprehensive audit trails for legal proceedings
- **User Agency**: Ensure human oversight and control in all automated processes

### Legal Compliance

Contributors should be aware of:

- Data protection regulations (GDPR, CCPA, etc.)
- Evidence handling standards in various jurisdictions
- Cross-border data transfer restrictions
- Legal privilege and confidentiality requirements
- Professional standards for legal technology

## Questions and Support

### Technical Questions
- **GitHub Discussions**: Community-driven Q&A
- **Discord**: Real-time technical support
- **Documentation**: Comprehensive guides and references

### Legal and Ethical Questions
- **Email**: ethics@lemkin.ai
- **Community Calls**: Monthly discussions on ethical considerations
- **Expert Network**: Access to legal technology experts

### Security Concerns
- **Email**: security@lemkin.ai (for security vulnerabilities)
- **Security Documentation**: Detailed security procedures and guidelines

### Community and Collaboration
- **Discord**: Active community chat
- **GitHub Discussions**: Long-form community conversations
- **Social Media**: Follow @lemkin_ai for updates

## Thank You

Your contributions help make legal investigation technology accessible to those who need it most. By working together, we can strengthen the capacity of human rights investigators, civil rights attorneys, and justice-seeking organizations around the world.

Every contribution, whether code, documentation, testing, or community support, makes a meaningful difference in the pursuit of justice and accountability.

---

*By contributing to this project, you join a global community dedicated to using technology for human rights, justice, and accountability.*