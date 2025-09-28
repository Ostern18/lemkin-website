# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |

## Reporting a Vulnerability

**Please do not report security vulnerabilities through public GitHub issues.**

Send email to security@lemkin.org with detailed information about the vulnerability.

## Security Considerations for PII Redaction

### Highly Sensitive Data Processing

PII redaction involves processing the most sensitive information in legal investigations:

- **Personal Identifiers**: Names, addresses, phone numbers, email addresses, SSNs
- **Biometric Data**: Faces, voices, and other uniquely identifying characteristics
- **Location Information**: Addresses, coordinates, and identifying landmarks
- **Financial Data**: Account numbers, credit card information, financial records
- **Health Information**: Medical records, health conditions, treatment details
- **Legal Information**: Case numbers, court documents, attorney-client communications

### Critical Security Requirements

1. **Data Protection**:
   - Encrypt all PII data both at rest and in transit
   - Use secure, isolated processing environments
   - Implement automatic secure deletion of temporary files
   - Never log or cache personally identifiable information

2. **Redaction Integrity**:
   - Ensure complete removal of PII, not just visual obscuring
   - Verify redaction effectiveness across all data formats
   - Maintain audit logs of all redaction activities
   - Test redaction against recovery attempts

3. **Access Control**:
   - Implement strict role-based access to redaction systems
   - Use multi-factor authentication for system access
   - Monitor and log all access to PII data
   - Restrict redaction operations to authorized personnel only

### High Priority Security Concerns

- **Incomplete Redaction**: PII remaining in processed documents or media
- **Data Leakage**: PII exposed through logs, temporary files, or metadata
- **Recovery Attacks**: Attempts to recover redacted information
- **Access Control Bypass**: Unauthorized access to sensitive PII data
- **Processing Vulnerabilities**: Exploits in PII detection algorithms

### PII-Specific Security Issues

- False negatives in PII detection leading to data exposure
- Metadata containing PII not being properly redacted
- Audio/video redaction that can be reversed or bypassed
- Cross-contamination of PII between different cases
- Inadequate secure deletion of processing artifacts

## Privacy-First Development

### Redaction Standards
```python
def redact_pii_securely(content: str) -> RedactionResult:
    """Redact PII with security-first approach."""
    # Use multiple detection methods for comprehensive coverage
    detected_pii = detect_all_pii_types(content)

    # Apply irreversible redaction, not just masking
    redacted_content = apply_secure_redaction(content, detected_pii)

    # Verify no PII remains
    verification = verify_complete_redaction(redacted_content)

    # Secure cleanup of temporary data
    secure_delete_temporary_data()

    return RedactionResult(
        content=redacted_content,
        verification=verification,
        audit_trail=create_audit_record()
    )
```

### Testing Security
- Never use real PII in development or testing
- Use synthetic PII that matches real patterns
- Test redaction with adversarial recovery attempts
- Verify secure deletion of all processing artifacts

## Contact Information

- **Security Issues**: security@lemkin.org (Critical - 4 hour response time)
- **Privacy Violations**: privacy@lemkin.org (Emergency response)
- **Documentation**: https://docs.lemkin.org/redaction

## Legal Notice

lemkin-redaction processes the most sensitive personal information in legal investigations. Users are responsible for:

- Ensuring proper legal authorization for PII processing
- Compliance with all privacy laws (GDPR, CCPA, HIPAA, etc.)
- Maintaining the highest security standards for PII data
- Verifying complete and irreversible redaction
- Protecting witness, victim, and subject privacy

---

*This security policy protects the most sensitive personal information in human rights work.*