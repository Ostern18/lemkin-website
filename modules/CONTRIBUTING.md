# Contributing to Lemkin AI Open Source Legal Investigation Ecosystem

Thank you for your interest in contributing to Lemkin AI! This project exists to democratize legal investigation technology for human rights investigators, prosecutors, civil rights attorneys, and public defenders worldwide.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Environment](#development-environment)
- [Contribution Types](#contribution-types)
- [Module-Specific Guidelines](#module-specific-guidelines)
- [Security & Privacy Requirements](#security--privacy-requirements)
- [Testing Standards](#testing-standards)
- [Documentation Requirements](#documentation-requirements)
- [Pull Request Process](#pull-request-process)
- [Community & Support](#community--support)

## Code of Conduct

This project adheres to the Lemkin AI Code of Conduct. By participating, you are expected to uphold this code. Please report unacceptable behavior to conduct@lemkin.org.

### Our Values

- **Human Rights First**: Every decision prioritizes the safety and effectiveness of human rights work
- **Privacy by Design**: Protect witnesses, victims, and sensitive information in all code
- **Open Collaboration**: Welcome contributions from diverse backgrounds and skill levels
- **Evidence Integrity**: Never compromise the integrity or admissibility of legal evidence
- **Transparency**: Make all algorithms explainable and auditable for legal proceedings

## Getting Started

### Prerequisites

- **Python 3.10+** with type hints
- **Git** for version control
- **Understanding of legal investigation workflows** (helpful but not required)
- **Commitment to ethical technology development**

### Quick Setup

```bash
# Clone the ecosystem
git clone https://github.com/lemkin-org/lemkin-ai.git
cd lemkin-ai

# Choose a module based on your interests
cd lemkin-[module-name]/

# Set up development environment
make install-dev

# Verify setup
make test
make verify-install
```

## Development Environment

### Standard Setup for Any Module

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -e ".[dev]"

# Set up pre-commit hooks
pre-commit install

# Run quality checks
make quality
```

### Development Tools

All modules use standardized tooling:
- **pytest**: Testing framework with >80% coverage requirement
- **black**: Code formatting (88 character line length)
- **ruff**: Fast linting and import sorting
- **mypy**: Strict type checking
- **pre-commit**: Git hooks for quality enforcement

## Contribution Types

### 🔴 **High Priority Contributions**

1. **Complete Core Modules** (Need full implementation):
   - **lemkin-ocr**: Document processing and OCR capabilities
   - **lemkin-research**: Legal research and citation analysis

2. **Production Polish** (Need final touches):
   - Add missing production files (LICENSE, SECURITY.md, CONTRIBUTING.md, Makefile)
   - Comprehensive testing and edge case handling
   - Performance optimization and security hardening

3. **Integration Testing**:
   - Cross-module workflow testing
   - Real-world scenario validation
   - Performance benchmarking

### 🟡 **Medium Priority Contributions**

1. **Enhanced Features**:
   - Multi-language support expansion
   - New platform integrations (within legal/ethical bounds)
   - Advanced analysis algorithms

2. **User Experience**:
   - CLI improvements and user-friendly interfaces
   - Better error messages and debugging
   - Documentation and tutorial improvements

3. **Specialized Modules**:
   - **lemkin-comms**: Communication analysis
   - **lemkin-dashboard**: Web-based investigation dashboards
   - **lemkin-reports**: Automated report generation

### 🟢 **Good First Issues**

- Documentation improvements and examples
- Test case additions and edge case coverage
- Bug fixes in existing implementations
- Translation and internationalization
- Performance optimizations

## Module-Specific Guidelines

### Evidence & Foundation Modules

**lemkin-integrity, lemkin-redaction, lemkin-classifier**

```python
# Example: Evidence handling with complete audit trail
def process_evidence(evidence: EvidenceFile) -> ProcessingResult:
    \"\"\"Process evidence with full chain of custody.

    Args:
        evidence: Evidence file with metadata

    Returns:
        Processing result with audit trail

    Raises:
        EvidenceIntegrityError: If evidence integrity is compromised
    \"\"\"
    # Always verify evidence integrity first
    integrity_check = verify_evidence_integrity(evidence)
    if not integrity_check.is_valid:
        raise EvidenceIntegrityError(f"Evidence integrity compromised: {integrity_check.issues}")

    # Log all processing steps for audit trail
    audit_logger.info(f"Processing evidence {evidence.id}", extra={
        "evidence_id": evidence.id,
        "case_id": evidence.case_id,
        "processing_type": "classification",
        "user_id": get_current_user_id()
    })

    # Process while preserving original
    result = perform_secure_processing(evidence)

    # Update chain of custody
    custody_manager.add_entry(
        evidence_id=evidence.id,
        action="processed",
        user=get_current_user(),
        details=result.processing_details
    )

    return result
```

### Media Analysis Modules

**lemkin-video, lemkin-images, lemkin-audio**

```python
# Example: Media processing with privacy protection
def analyze_media_with_privacy(media_path: Path, privacy_settings: PrivacySettings) -> MediaAnalysis:
    \"\"\"Analyze media while protecting privacy.

    Args:
        media_path: Path to media file
        privacy_settings: Privacy protection requirements

    Returns:
        Analysis results with privacy protections applied
    \"\"\"
    with secure_temp_media(media_path) as temp_media:
        # Apply privacy protections first
        if privacy_settings.blur_faces:
            temp_media = apply_face_anonymization(temp_media)

        if privacy_settings.strip_metadata:
            temp_media = remove_sensitive_metadata(temp_media)

        # Perform analysis on privacy-protected version
        analysis = perform_analysis(temp_media)

        # Ensure no sensitive data in results
        return sanitize_analysis_results(analysis, privacy_settings)
```

### Collection & Analysis Modules

**lemkin-osint, lemkin-geo, lemkin-forensics**

```python
# Example: OSINT collection with ethical boundaries
@rate_limit(calls=100, period=3600)  # Respect platform limits
def collect_osint_data(query: str, platforms: List[str]) -> OSINTCollection:
    \"\"\"Collect OSINT data within ethical and legal boundaries.

    Args:
        query: Search query
        platforms: List of platforms to search

    Returns:
        OSINT collection with metadata

    Raises:
        EthicsViolationError: If collection violates ethical guidelines
    \"\"\"
    # Validate query doesn't target individuals inappropriately
    ethics_check = validate_collection_ethics(query, platforms)
    if not ethics_check.is_ethical:
        raise EthicsViolationError(f"Collection violates ethics: {ethics_check.issues}")

    # Collect data with privacy protections
    collected_data = []
    for platform in platforms:
        platform_data = collect_from_platform(platform, query)

        # Apply automatic PII detection and redaction
        sanitized_data = detect_and_redact_pii(platform_data)
        collected_data.extend(sanitized_data)

    return OSINTCollection(
        query=query,
        platforms=platforms,
        data=collected_data,
        collection_metadata=create_collection_metadata(),
        ethics_compliance=ethics_check
    )
```

## Security & Privacy Requirements

### Critical Security Standards

1. **No Sensitive Data in Code**:
   ```python
   # ❌ NEVER do this
   API_KEY = "sk-abc123..."

   # ✅ Always use environment variables
   API_KEY = os.getenv("API_KEY")
   if not API_KEY:
       raise ConfigurationError("API_KEY environment variable required")
   ```

2. **Secure Data Handling**:
   ```python
   def process_sensitive_file(file_path: Path) -> ProcessingResult:
       \"\"\"Process sensitive file with secure cleanup.\"\"\"
       temp_files = []
       try:
           # Create secure temporary files
           temp_file = create_secure_temp_file(file_path)
           temp_files.append(temp_file)

           # Process data
           result = perform_processing(temp_file)
           return result

       finally:
           # Always securely delete temporary files
           for temp_file in temp_files:
               secure_delete(temp_file)
   ```

3. **Input Validation**:
   ```python
   def validate_evidence_file(file_path: Path) -> ValidationResult:
       \"\"\"Validate evidence file for security threats.\"\"\"
       # Check file size limits
       if file_path.stat().st_size > MAX_FILE_SIZE:
           raise ValidationError(f"File too large: {file_path}")

       # Validate file type
       if not is_supported_file_type(file_path):
           raise ValidationError(f"Unsupported file type: {file_path.suffix}")

       # Scan for malicious content
       security_scan = scan_for_threats(file_path)
       if security_scan.threats_found:
           raise SecurityError(f"Threats detected: {security_scan.threats}")

       return ValidationResult(is_valid=True)
   ```

### Privacy Protection Requirements

1. **Automatic PII Detection**:
   ```python
   from lemkin_redaction import PIIDetector

   def process_content_safely(content: str) -> str:
       \"\"\"Process content with automatic PII protection.\"\"\"
       detector = PIIDetector()

       # Detect PII
       pii_results = detector.detect_pii(content)

       # Redact if PII found
       if pii_results:
           logger.info(f"PII detected: {len(pii_results)} instances")
           content = detector.redact_pii(content, pii_results)

       return content
   ```

2. **Location Data Protection**:
   ```python
   def process_location_data(coordinates: List[Coordinate]) -> LocationAnalysis:
       \"\"\"Process location data with privacy protections.\"\"\"
       # Apply coordinate obfuscation for non-critical analysis
       if should_obfuscate_coordinates():
           coordinates = [obfuscate_coordinate(coord) for coord in coordinates]

       return perform_location_analysis(coordinates)
   ```

## Testing Standards

### Test Structure

```
tests/
├── unit/                 # Unit tests for individual functions
├── integration/          # Integration tests for workflows
├── security/             # Security and privacy tests
├── performance/          # Performance and load tests
└── fixtures/             # Test data (synthetic only!)
    ├── documents/        # Sample documents (no real data)
    ├── media/            # Sample audio/video/images
    └── data/             # JSON/CSV test datasets
```

### Test Requirements

```python
import pytest
from lemkin_[module] import ModuleClass

class TestModuleClass:
    def setup_method(self):
        \"\"\"Set up test environment before each test.\"\"\"
        self.module = ModuleClass()
        self.test_data = load_synthetic_test_data()

    def test_basic_functionality(self):
        \"\"\"Test core functionality with synthetic data.\"\"\"
        result = self.module.process_data(self.test_data)

        assert result.is_valid
        assert result.processing_time > 0
        assert len(result.items) > 0

    def test_privacy_protection(self):
        \"\"\"Test that PII is properly detected and protected.\"\"\"
        content_with_pii = "Contact John Doe at john.doe@example.com"

        result = self.module.process_content(content_with_pii)

        # Ensure PII was detected and removed
        assert "john.doe@example.com" not in result.processed_content
        assert result.pii_detected is True
        assert len(result.pii_locations) > 0

    def test_error_handling(self):
        \"\"\"Test graceful error handling.\"\"\"
        with pytest.raises(ValidationError):
            self.module.process_data(invalid_data)

    def test_security_validation(self):
        \"\"\"Test security validation and threat detection.\"\"\"
        malicious_input = create_malicious_test_input()

        with pytest.raises(SecurityError):
            self.module.process_data(malicious_input)
```

### Test Data Ethics

⚠️ **CRITICAL**: Never use real investigation data in tests

```python
# ✅ Good: Synthetic test data
test_person_data = {
    "name": "Alex Johnson",  # Clearly synthetic
    "email": "test@example.com",
    "phone": "555-0123",
    "address": "123 Test Street, Example City, ST 12345"
}

# ❌ Never: Real personal data
# Don't use real names, emails, phone numbers, or addresses
```

## Documentation Requirements

### Code Documentation

```python
def analyze_legal_document(
    document_path: Path,
    case_id: str,
    analysis_type: AnalysisType = AnalysisType.COMPREHENSIVE,
    privacy_protection: bool = True
) -> DocumentAnalysis:
    \"\"\"Analyze legal document with comprehensive extraction and classification.

    This function processes legal documents to extract entities, classify content,
    and identify relevant legal frameworks while maintaining evidence integrity
    and protecting personally identifiable information.

    Args:
        document_path: Path to the document file to analyze
        case_id: Unique identifier for the legal case
        analysis_type: Type of analysis to perform (basic, comprehensive, custom)
        privacy_protection: Whether to apply automatic PII detection and redaction

    Returns:
        DocumentAnalysis containing:
            - Extracted entities (persons, organizations, dates, locations)
            - Document classification and confidence scores
            - Identified legal frameworks and applicable laws
            - Privacy protection report if enabled
            - Processing metadata and audit trail

    Raises:
        FileNotFoundError: If document_path does not exist
        ValidationError: If document format is unsupported or corrupted
        SecurityError: If document contains potential security threats
        ProcessingError: If analysis fails due to unexpected errors

    Example:
        >>> analysis = analyze_legal_document(
        ...     Path("witness_statement.pdf"),
        ...     case_id="HR-2024-001",
        ...     analysis_type=AnalysisType.COMPREHENSIVE,
        ...     privacy_protection=True
        ... )
        >>> print(f"Entities found: {len(analysis.entities)}")
        >>> print(f"Classification: {analysis.classification.primary_type}")

    Legal Considerations:
        - All processing maintains chain of custody for legal admissibility
        - PII detection helps comply with privacy regulations (GDPR, CCPA)
        - Analysis methods are designed to be explainable in court proceedings
        - Audit trails enable verification of analysis integrity

    Privacy & Security:
        - Automatic PII detection protects witness and victim identities
        - Secure processing prevents data leakage through temporary files
        - Input validation prevents processing of potentially malicious files
        - All sensitive data is encrypted during processing
    \"\"\"
```

### README Standards

Each module README must include:

1. **Purpose** (2-3 sentences)
2. **Safety & Ethics Notice** with clear warnings
3. **Quick Start** with installation and basic usage
4. **Key Features** list
5. **Usage Examples** (5+ real-world scenarios)
6. **API Reference** with code examples
7. **Input/Output specifications** with JSON schemas
8. **Performance metrics** and limitations
9. **Safety guidelines** for evidence handling
10. **Contributing guidelines** specific to the module

## Pull Request Process

### PR Requirements Checklist

- [ ] **Code Quality**:
  - [ ] All tests pass (`make test`)
  - [ ] Code is formatted (`make format`)
  - [ ] Linting passes (`make lint`)
  - [ ] Type checking passes (`make type-check`)

- [ ] **Security & Privacy**:
  - [ ] No hardcoded secrets or API keys
  - [ ] PII protection implemented where applicable
  - [ ] Input validation for all user inputs
  - [ ] Secure handling of temporary files

- [ ] **Legal Compliance**:
  - [ ] Evidence integrity maintained
  - [ ] Chain of custody considerations addressed
  - [ ] Privacy protection measures implemented
  - [ ] Analysis methods are explainable and auditable

- [ ] **Documentation**:
  - [ ] Code is well-documented with docstrings
  - [ ] README updated if needed
  - [ ] API changes documented
  - [ ] Examples provided for new features

- [ ] **Testing**:
  - [ ] Unit tests cover new functionality
  - [ ] Integration tests for workflows
  - [ ] Security tests for sensitive operations
  - [ ] Performance tests for resource-intensive operations

### PR Template

```markdown
## Description
Brief description of changes and their purpose for legal investigations.

## Type of Change
- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update
- [ ] Security enhancement
- [ ] Performance improvement

## Legal Investigation Context
- [ ] Supports human rights investigation workflows
- [ ] Maintains evidence integrity and chain of custody
- [ ] Protects witness and victim privacy
- [ ] Enables explainable analysis for legal proceedings
- [ ] Complies with relevant legal and privacy standards

## Security Checklist
- [ ] No sensitive data hardcoded
- [ ] PII protection implemented
- [ ] Input validation added
- [ ] Secure data handling practices followed
- [ ] Audit logging implemented where appropriate

## Testing
- [ ] Unit tests added/updated
- [ ] Integration tests added/updated
- [ ] Security tests included
- [ ] Performance tests included
- [ ] Manual testing completed

## Documentation
- [ ] Code is self-documenting with clear docstrings
- [ ] README updated if functionality changes
- [ ] API documentation updated
- [ ] Usage examples provided

## Legal & Ethical Considerations
- [ ] Change supports legitimate legal investigation use cases
- [ ] Privacy implications considered and addressed
- [ ] No functionality that could be used for harassment or stalking
- [ ] Complies with platform terms of service where applicable
- [ ] Respects international human rights standards
```

## Community & Support

### Communication Channels

- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: General questions and community discussion
- **Discord**: Real-time chat with developers and users
- **Monthly Community Calls**: Updates, planning, and Q&A sessions

### Specialized Support

- **Technical Questions**: Open GitHub Issues or join Discord
- **Security Vulnerabilities**: security@lemkin.org (confidential)
- **Legal/Ethical Questions**: ethics@lemkin.org
- **Privacy Concerns**: privacy@lemkin.org
- **Partnership Opportunities**: partnerships@lemkin.org

### Recognition & Attribution

Contributors are recognized through:
- **Contributor listings** in project documentation
- **Release notes** highlighting significant contributions
- **Community spotlights** in newsletters and social media
- **Conference speaking opportunities** to present your work
- **Collaboration opportunities** with legal organizations using Lemkin AI

## Getting Help

### New Contributor Onboarding

1. **Join the Community**: Introduce yourself in GitHub Discussions
2. **Find Your Focus**: Review the [current status](README.md#-current-implementation-status) to find modules that match your interests
3. **Start Small**: Look for "good first issue" labels
4. **Ask Questions**: Don't hesitate to ask for clarification or guidance
5. **Pair Programming**: We offer pairing sessions for complex contributions

### Mentorship Program

We offer mentorship for contributors working on:
- **Core module implementations**
- **Security and privacy enhancements**
- **Legal domain expertise integration**
- **Performance optimization projects**

Contact mentorship@lemkin.org to be matched with an experienced contributor.

---

## Legal & Ethical Commitment

By contributing to Lemkin AI, you commit to:

- **Supporting legitimate legal investigations** and human rights work
- **Protecting privacy** and sensitive information at all times
- **Maintaining evidence integrity** and following legal standards
- **Avoiding functionality** that could be used for harassment or illegal activities
- **Respecting platform terms of service** and international law
- **Following ethical AI development** principles

Your contributions help provide equal access to justice technology for those who need it most. Thank you for helping make legal investigation more accessible, accurate, and ethical.

---

*"Technology should accelerate justice, not delay it. Your contributions help ensure that legal investigation technology serves all of humanity."*