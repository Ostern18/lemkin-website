# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |

## Reporting a Vulnerability

**Please do not report security vulnerabilities through public GitHub issues.**

Send email to security@lemkin.org with detailed information about the vulnerability.

## Security Considerations for Digital Forensics

### Sensitive Evidence Data

Digital forensics involves processing sensitive evidence that may contain:

- **File System Evidence**: Directory structures, deleted files, system logs
- **Network Traffic**: Communication patterns, metadata, connection logs
- **Digital Signatures**: Cryptographic evidence of file authenticity
- **Timestamps**: Critical temporal evidence for legal proceedings
- **Personal Data**: User files, documents, and private information

### Security Best Practices

1. **Evidence Integrity**:
   - Maintain immutable records of original evidence
   - Use cryptographic hashing for evidence verification
   - Implement secure chain of custody procedures
   - Never modify original evidence files

2. **Secure Processing**:
   - Process evidence in isolated, air-gapped environments
   - Use encrypted storage for all forensic analysis
   - Implement proper access controls and authentication
   - Secure deletion of temporary analysis files

3. **Legal Compliance**:
   - Follow applicable laws for evidence handling
   - Maintain detailed audit logs of all analysis
   - Ensure analysis methods are legally admissible
   - Document all forensic procedures and results

### High Priority Security Concerns

- Evidence contamination or modification
- Unauthorized access to sensitive forensic data
- Cryptographic vulnerabilities in evidence verification
- Improper handling of personally identifiable information
- Chain of custody integrity violations

## Contact Information

- **Security Issues**: security@lemkin.org
- **Emergency**: For evidence integrity threats, contact immediately
- **Documentation**: https://docs.lemkin.org/forensics

---

*This security policy is part of the Lemkin AI commitment to protecting digital evidence integrity.*