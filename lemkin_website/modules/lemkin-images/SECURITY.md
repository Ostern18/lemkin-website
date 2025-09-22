# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |

## Reporting a Vulnerability

**Please do not report security vulnerabilities through public GitHub issues.**

Send email to security@lemkin.org with detailed information about the vulnerability.

## Security Considerations for Image Verification

### Sensitive Image Content

Image verification involves processing sensitive visual evidence that may contain:

- **Witness Documentation**: Photos of witnesses, victims, and sensitive locations
- **Evidence Photos**: Crime scenes, document photos, surveillance images
- **Location Information**: GPS coordinates, landmarks, and identifying features
- **Personal Data**: Faces, license plates, addresses, and other identifiers
- **Metadata**: Camera information, timestamps, and processing history

### Security Best Practices

1. **Image Processing Security**:
   - Process images in secure, air-gapped environments
   - Encrypt image files and analysis results
   - Use secure temporary storage with automatic cleanup
   - Implement strict access controls for sensitive images

2. **Privacy Protection**:
   - Automatically detect and redact personally identifiable information
   - Implement face detection and blurring capabilities
   - Protect location data from GPS metadata
   - Secure reverse image search results

3. **Evidence Integrity**:
   - Preserve original images without modification
   - Use cryptographic hashing for authenticity verification
   - Maintain immutable audit logs of all processing
   - Document manipulation detection methods for court

### High Priority Security Concerns

- Unauthorized access to sensitive photographic evidence
- Privacy violations through inadequate PII protection
- Image manipulation detection false positives/negatives
- Reverse image search exposing investigation details
- GPS and metadata exposure revealing sensitive locations

### Image-Specific Vulnerabilities

- Malicious image files exploiting processing libraries
- EXIF data injection attacks
- Privacy leaks through reverse image searches
- Inadequate sanitization of image metadata
- Buffer overflows in image parsing components

## Contact Information

- **Security Issues**: security@lemkin.org
- **Privacy Concerns**: For image privacy and PII issues
- **Documentation**: https://docs.lemkin.org/images

---

*This security policy protects sensitive photographic evidence in legal investigations.*