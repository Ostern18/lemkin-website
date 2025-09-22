# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |

## Reporting a Vulnerability

**Please do not report security vulnerabilities through public GitHub issues.**

Send email to security@lemkin.org with detailed information about the vulnerability.

## Security Considerations for Video Authentication

### Sensitive Video Content

Video authentication involves processing sensitive media that may contain:

- **Witness Testimonies**: Recorded statements and interviews
- **Surveillance Footage**: Security cameras, body cameras, phone recordings
- **Location Information**: Background details revealing sensitive locations
- **Personal Identifiers**: Faces, voices, and other biometric data
- **Evidence Metadata**: Technical details about recording equipment and settings

### Security Best Practices

1. **Video Processing Security**:
   - Process videos in secure, isolated environments
   - Encrypt video files both at rest and in transit
   - Use secure temporary storage with automatic cleanup
   - Implement proper access controls for video analysis

2. **Privacy Protection**:
   - Detect and protect personally identifiable information
   - Implement face blurring for non-relevant individuals
   - Secure voice data and speaker identification results
   - Maintain strict access controls for sensitive footage

3. **Authentication Integrity**:
   - Preserve original video evidence without modification
   - Use cryptographic methods to verify authenticity
   - Maintain detailed audit logs of all analysis
   - Document all processing steps for legal admissibility

### High Priority Security Concerns

- Unauthorized access to sensitive video content
- Privacy violations through inadequate anonymization
- Video tampering detection false positives/negatives
- Deepfake detection bypass methods
- Metadata exposure revealing investigation details

### Video-Specific Vulnerabilities

- Buffer overflows in video parsing libraries
- Malicious video files designed to exploit processing
- Privacy leaks through video fingerprinting
- Compression analysis revealing processing history
- Inadequate security for machine learning models

## Contact Information

- **Security Issues**: security@lemkin.org
- **Privacy Concerns**: For video privacy issues
- **Documentation**: https://docs.lemkin.org/video

---

*This security policy protects sensitive video evidence in human rights investigations.*