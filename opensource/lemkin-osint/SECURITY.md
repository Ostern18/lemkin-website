# Security Policy

## Supported Versions

We provide security updates for the following versions of lemkin-osint:

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |

## Reporting a Vulnerability

The Lemkin AI team and community take security bugs seriously. We appreciate your efforts to responsibly disclose your findings, and will make every effort to acknowledge your contributions.

### How to Report Security Issues

**Please do not report security vulnerabilities through public GitHub issues.**

Instead, please send email to security@lemkin.org. You should receive a response within 48 hours. If for some reason you do not, please follow up via email to ensure we received your original message.

Please include as much of the following information as possible to help us better understand and resolve the issue:

* Type of issue (e.g. data exposure, injection attacks, authentication bypass, etc.)
* Full paths of source file(s) related to the manifestation of the issue
* The location of the affected source code (tag/branch/commit or direct URL)
* Any special configuration required to reproduce the issue
* Step-by-step instructions to reproduce the issue
* Proof-of-concept or exploit code (if possible)
* Impact of the issue, including how an attacker might exploit the issue

## Security Considerations for OSINT Collection

### Sensitive Information Handling

OSINT collection involves gathering information from public sources that may contain:

- **Personal Information**: Names, addresses, phone numbers, email addresses
- **Location Data**: GPS coordinates, check-ins, travel patterns
- **Social Connections**: Relationships, associations, communications
- **Behavioral Patterns**: Activity timings, preferences, routines
- **Metadata**: Technical information that could reveal investigation methods

### Security Best Practices

When using lemkin-osint, please:

1. **Operational Security**:
   - Use VPNs or Tor networks to mask investigation activities
   - Rotate IP addresses and user agents regularly
   - Use separate, dedicated systems for OSINT collection
   - Implement proper access controls for collected data

2. **Data Protection**:
   - Encrypt collected data both at rest and in transit
   - Use secure storage systems with proper access controls
   - Implement data retention policies and secure deletion
   - Audit and log all data access and processing activities

3. **Source Protection**:
   - Avoid patterns that could reveal investigation targets
   - Respect platform terms of service and rate limits
   - Use authentication carefully to avoid account linking
   - Monitor for detection of automated collection activities

4. **Privacy Protection**:
   - Implement automatic PII detection and redaction
   - Follow data protection regulations (GDPR, CCPA, etc.)
   - Minimize collection to only necessary information
   - Protect witness and source identities

5. **Investigation Security**:
   - Segregate different investigations and cases
   - Use case-specific credentials and collection parameters
   - Implement chain of custody for digital evidence
   - Document all collection activities for legal proceedings

### Known Security Considerations

1. **Platform Detection**: Automated collection may be detected by target platforms

2. **Rate Limiting**: Excessive requests may trigger platform security measures

3. **Account Linking**: Using authenticated accounts may link investigations to organizations

4. **Data Persistence**: Collected data may inadvertently persist in logs or temporary files

5. **Metadata Exposure**: Collection activities may leave digital fingerprints

## OSINT-Specific Security Issues

We are particularly concerned about vulnerabilities in these areas:

### High Priority
- API key exposure or credential leakage
- Unencrypted storage of collected sensitive data
- Injection vulnerabilities in web scraping components
- Authentication bypass in social media collection
- Exposure of investigation targets or methods

### Medium Priority
- Rate limiting bypass that could expose investigations
- Insecure handling of platform cookies and sessions
- Inadequate error handling that reveals sensitive information
- Insufficient logging of collection activities
- Data retention policy violations

### OSINT-Specific Vulnerabilities
- Social media platform API abuse
- Web scraping detection and blocking
- Metadata leakage in collected content
- Cross-contamination between investigations
- Inadequate anonymization of collected data

## Responsible OSINT Collection

### Legal Compliance
- Obtain proper legal authorization for OSINT collection
- Respect international privacy laws and regulations
- Follow platform terms of service where legally required
- Maintain audit trails for legal proceedings
- Protect witness and victim privacy

### Ethical Guidelines
- Minimize collection to investigation-relevant information
- Implement privacy-preserving collection methods
- Respect individual privacy rights where possible
- Avoid collection that could endanger sources
- Follow journalistic and research ethics principles

### Platform Responsibility
- Respect website robots.txt files where legally appropriate
- Use reasonable rate limits to avoid service disruption
- Avoid collection methods that could harm platform performance
- Report security vulnerabilities in target platforms responsibly

## Contact Information

For security-related questions or concerns:

- **Email**: security@lemkin.org
- **PGP Key**: Available on request
- **Response Time**: Within 48 hours for initial acknowledgment

For OSINT-specific questions:
- **GitHub Issues**: https://github.com/lemkin-org/lemkin-osint/issues
- **Documentation**: https://docs.lemkin.org/osint

## Legal Notice

lemkin-osint is designed for legitimate legal investigations and human rights documentation. Users are responsible for:

- Ensuring proper legal authorization for collection activities
- Compliance with all applicable laws and regulations
- Respecting privacy rights and platform terms of service
- Maintaining appropriate security and privacy protections
- Using collected information responsibly and ethically

---

*This security policy is part of the Lemkin AI commitment to protecting human rights investigators and their sensitive work.*