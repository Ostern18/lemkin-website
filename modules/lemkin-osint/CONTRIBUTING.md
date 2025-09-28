# Contributing to Lemkin OSINT Collection Toolkit

Thank you for your interest in contributing to the Lemkin OSINT Collection Toolkit! This project is part of the Lemkin AI ecosystem designed to democratize legal investigation technology for human rights investigators, prosecutors, and civil rights attorneys.

## Code of Conduct

This project adheres to the Lemkin AI Code of Conduct. By participating, you are expected to uphold this code. Please report unacceptable behavior to conduct@lemkin.org.

## Getting Started

### Prerequisites

- Python 3.10 or higher
- Understanding of OSINT (Open Source Intelligence) principles
- Knowledge of web scraping and API integration
- Familiarity with legal investigation workflows (helpful)

### Development Environment Setup

```bash
# Clone the repository
git clone https://github.com/lemkin-org/lemkin-osint.git
cd lemkin-osint

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install development dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install

# Verify installation
make verify-install
```

## Contribution Areas

### High Priority Contributions

1. **Platform Integrations**: Add support for new social media and web platforms
2. **Data Extraction**: Improve extraction of structured data from web sources
3. **Privacy Protection**: Enhance PII detection and anonymization features
4. **Rate Limiting**: Implement sophisticated rate limiting to avoid detection
5. **Error Handling**: Improve robustness of web scraping operations

### Platform-Specific Contributions

We welcome contributions for these platforms:
- Social media platforms (within ToS compliance)
- News and media websites
- Public records databases
- Court document systems
- Academic and research databases
- Government transparency portals

### Legal and Compliance Focus

- **Terms of Service Compliance**: Ensure collection methods respect platform ToS
- **Privacy Protection**: Implement privacy-preserving collection techniques
- **Data Minimization**: Reduce collection to investigation-relevant information
- **Audit Trails**: Improve logging and audit capabilities for legal proceedings

## Code Guidelines

### Security Requirements

```python
# Example: Secure credential handling
def initialize_collector(api_key: str) -> OSINTCollector:
    """Initialize OSINT collector with secure credential handling."""
    if not api_key:
        raise ValueError("API key is required")

    # Never log credentials
    logger.info("Initializing OSINT collector")

    return OSINTCollector(api_key=api_key)

# Example: Rate limiting
@rate_limit(calls=100, period=3600)  # 100 calls per hour
def collect_social_media_posts(query: str) -> List[Post]:
    """Collect posts with appropriate rate limiting."""
    pass
```

### Data Privacy Standards

```python
from lemkin_osint import PIIDetector

def process_collected_content(content: str) -> str:
    """Process collected content with PII protection."""
    detector = PIIDetector()

    # Detect and redact PII
    pii_results = detector.detect_pii(content)
    cleaned_content = detector.redact_pii(content, pii_results)

    # Log PII detection for audit
    logger.info(f"Detected {len(pii_results)} PII instances")

    return cleaned_content
```

### Platform Integration Template

```python
class NewPlatformCollector(BasePlatformCollector):
    """Template for adding new platform support."""

    def __init__(self, api_credentials: Dict[str, str]):
        super().__init__()
        self.credentials = api_credentials
        self._setup_rate_limiting()

    def collect_posts(
        self,
        query: str,
        limit: int = 100,
        date_range: Optional[DateRange] = None
    ) -> OSINTCollection:
        """Collect posts from the platform."""
        # Implement platform-specific collection logic
        pass

    def verify_source_credibility(self, source: Source) -> CredibilityScore:
        """Assess source credibility using platform-specific metrics."""
        pass
```

## Testing Requirements

### OSINT Testing Challenges

Testing OSINT collection requires special considerations:

```python
class TestOSINTCollector:
    def setup_method(self):
        """Set up test environment with mock data."""
        # Use mock responses, never real API calls in tests
        self.collector = OSINTCollector(api_key="test-key")
        self.mock_responses = load_mock_responses()

    @mock.patch('requests.get')
    def test_social_media_collection(self, mock_get):
        """Test social media collection with mocked responses."""
        mock_get.return_value.json.return_value = self.mock_responses['posts']

        result = self.collector.collect_social_media_evidence("test query")

        assert result.total_items > 0
        assert all(post.content for post in result.items)

    def test_pii_protection(self):
        """Test that PII is properly detected and protected."""
        content_with_pii = "Contact John Doe at john.doe@email.com"

        result = self.collector.process_content(content_with_pii)

        assert "john.doe@email.com" not in result.processed_content
        assert result.pii_detected is True
```

### Mock Data Guidelines

- **No Real Data**: Never use real people's information in tests
- **Realistic Structure**: Use realistic data structures and formats
- **Edge Cases**: Include edge cases and error conditions
- **Privacy Compliant**: Ensure test data doesn't violate privacy

## Legal and Ethical Considerations

### Platform Terms of Service

When adding platform integrations:

1. **Review ToS**: Carefully review platform terms of service
2. **Respect Limits**: Implement appropriate rate limiting
3. **Attribution**: Provide proper attribution where required
4. **Commercial Use**: Ensure compliance with commercial use restrictions

### Privacy Protection

```python
class PrivacyProtectedCollection:
    """Example of privacy-protecting collection methods."""

    def collect_with_anonymization(self, query: str) -> OSINTCollection:
        """Collect data with built-in anonymization."""
        raw_data = self._collect_raw_data(query)

        # Anonymize sensitive fields
        anonymized_data = self._anonymize_personal_data(raw_data)

        # Remove metadata that could identify investigation
        cleaned_data = self._strip_investigation_metadata(anonymized_data)

        return self._create_collection(cleaned_data)
```

### Legal Compliance Checklist

Before submitting OSINT-related contributions:

- [ ] Reviewed relevant platform terms of service
- [ ] Implemented appropriate rate limiting
- [ ] Added PII detection and protection
- [ ] Ensured no hardcoded credentials or sensitive data
- [ ] Documented legal considerations in code comments
- [ ] Added audit logging for chain of custody
- [ ] Tested with synthetic/mock data only

## Documentation Requirements

### Platform Integration Docs

When adding new platform support, include:

```markdown
## Platform Name Integration

### Legal Considerations
- Terms of service compliance notes
- Rate limiting requirements
- Attribution requirements
- Privacy protection measures

### Setup
```bash
# Installation and setup instructions
```

### Usage Examples
```python
# Code examples for common use cases
```

### Data Formats
```json
// Example output formats
```

### Limitations
- Known limitations and constraints
- Performance characteristics
- Legal restrictions
```

## Pull Request Guidelines

### OSINT-Specific PR Requirements

1. **Legal Review**: OSINT PRs require additional legal compliance review
2. **Privacy Impact**: Assess privacy implications of changes
3. **Platform Compliance**: Verify platform ToS compliance
4. **Testing**: Include comprehensive tests with mock data
5. **Documentation**: Update platform-specific documentation

### PR Template for OSINT Features

```markdown
## OSINT Collection Enhancement

### Platform/Source
- Platform: [Platform name]
- Data types: [Posts, profiles, media, etc.]
- Collection method: [API, scraping, etc.]

### Legal Compliance
- [ ] Reviewed platform Terms of Service
- [ ] Implemented appropriate rate limiting
- [ ] Added PII detection and protection
- [ ] Documented legal considerations

### Privacy Protection
- [ ] Personal data is anonymized
- [ ] Investigation metadata is protected
- [ ] Audit logging implemented
- [ ] Data retention policies followed

### Testing
- [ ] Unit tests with mock data
- [ ] Integration tests completed
- [ ] Performance testing completed
- [ ] Error handling tested

### Security Considerations
- [ ] No hardcoded credentials
- [ ] Secure data handling
- [ ] Input validation implemented
- [ ] Error messages don't leak sensitive info
```

## Community Guidelines

### Responsible OSINT Development

1. **Human Rights Focus**: Remember this tool supports human rights investigations
2. **Ethical Collection**: Prioritize ethical and legal collection methods
3. **Privacy First**: Build privacy protection into all features
4. **Transparency**: Make collection methods auditable and explainable

### Getting Help

- **Technical Questions**: Open GitHub Discussions
- **Legal/Ethical Questions**: Email ethics@lemkin.org
- **Security Concerns**: Email security@lemkin.org
- **Platform-Specific**: Check our Discord channels

## Recognition

Contributors to OSINT capabilities receive special recognition for:
- Expanding platform coverage while maintaining legal compliance
- Improving privacy protection mechanisms
- Enhancing audit and chain of custody capabilities
- Contributing to legal investigation methodology

---

*Thank you for helping make OSINT collection accessible and ethical for human rights investigators worldwide.*