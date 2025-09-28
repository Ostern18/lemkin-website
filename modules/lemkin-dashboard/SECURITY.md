# Security Policy

## Supported Versions

We provide security updates for the following versions of lemkin-dashboard:

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |

## Reporting a Vulnerability

The Lemkin AI team and community take security bugs seriously. We appreciate your efforts to responsibly disclose your findings, and will make every effort to acknowledge your contributions.

### How to Report Security Issues

**Please do not report security vulnerabilities through public GitHub issues.**

Instead, please send email to security@lemkin.org. You should receive a response within 48 hours. If for some reason you do not, please follow up via email to ensure we received your original message.

Please include as much of the following information as possible to help us better understand and resolve the issue:

* Type of issue (e.g. buffer overflow, SQL injection, cross-site scripting, etc.)
* Full paths of source file(s) related to the manifestation of the issue
* The location of the affected source code (tag/branch/commit or direct URL)
* Any special configuration required to reproduce the issue
* Step-by-step instructions to reproduce the issue
* Proof-of-concept or exploit code (if possible)
* Impact of the issue, including how an attacker might exploit the issue

This information will help us triage your report more quickly.

### What to Expect

After you submit a report, here's what happens:

1. **Acknowledgment**: We'll acknowledge receipt of your vulnerability report within 2 business days.

2. **Initial Assessment**: We'll provide an initial assessment of the reported vulnerability within 5 business days, including:
   - Whether we can reproduce the issue
   - Severity assessment using CVSS scoring
   - Preliminary timeline for resolution

3. **Investigation**: Our security team will investigate and work on a fix:
   - Critical/High severity: Target resolution within 30 days
   - Medium severity: Target resolution within 60 days
   - Low severity: Target resolution within 90 days

4. **Disclosure**: Once a fix is available:
   - We'll release security patches
   - Credit will be given to reporters (unless anonymity is requested)
   - CVE numbers will be assigned for significant vulnerabilities
   - Public disclosure will occur 7 days after patch release

## Security Considerations for Investigation Dashboard Analysis

### Sensitive Investigation Dashboard Data

Investigation Dashboard files processed by lemkin-dashboard may contain:

- **Personal conversations**: Private discussions, witness testimonies, confidential communications
- **Biometric data**: Voice prints that can uniquely identify individuals
- **Location information**: Background noise that may reveal locations
- **Metadata**: EXIF data, GPS coordinates, device information

### Security Best Practices

When using lemkin-dashboard, please:

1. **Secure Storage**:
   - Store audio files in encrypted storage systems
   - Use secure file transfer protocols (SFTP, HTTPS)
   - Implement proper access controls and authentication

2. **Data Processing**:
   - Process sensitive audio in isolated environments
   - Use temporary directories that are securely wiped after processing
   - Avoid logging sensitive content or audio metadata

3. **Network Security**:
   - Use VPNs or secure networks for audio processing
   - Encrypt network traffic when transferring audio files
   - Monitor network access and file transfers

4. **Access Control**:
   - Implement role-based access control for audio analysis systems
   - Use multi-factor authentication for system access
   - Audit and log all audio file access and processing activities

5. **Data Retention**:
   - Follow legal requirements for audio evidence retention
   - Securely delete temporary files and processing artifacts
   - Implement automated data lifecycle management

### Known Security Considerations

1. **Investigation Dashboard Enhancement**: Enhanced audio may reveal previously inaudible content that could be privacy-sensitive

2. **Speaker Identification**: Voice profiles can be used for tracking individuals across multiple recordings

3. **Transcription**: Automatic transcription may capture sensitive information that should be redacted

4. **Metadata Extraction**: Investigation Dashboard metadata may contain personally identifiable information

5. **Authenticity Verification**: False positives/negatives in tampering detection could affect legal proceedings

## Vulnerability Categories

We are particularly concerned about vulnerabilities in these areas:

### High Priority
- Remote code execution vulnerabilities
- SQL injection or NoSQL injection
- Authentication/authorization bypass
- Sensitive data exposure
- Cryptographic vulnerabilities

### Medium Priority
- Cross-site scripting (XSS)
- Cross-site request forgery (CSRF)
- Insecure direct object references
- Security misconfigurations
- Insufficient logging and monitoring

### Investigation Dashboard-Specific Security Issues
- Investigation Dashboard file parsing vulnerabilities
- Buffer overflows in audio processing libraries
- Metadata injection attacks
- Voice synthesis/spoofing detection bypass
- Speaker identification privacy leaks

## Security Development Lifecycle

### Code Review
- All code changes undergo security-focused peer review
- Automated static analysis tools scan for common vulnerabilities
- Dependencies are regularly scanned for known vulnerabilities

### Testing
- Security test cases are included in our test suite
- Fuzzing is performed on audio file parsing code
- Penetration testing is conducted on major releases

### Dependencies
- We use automated tools to monitor dependency vulnerabilities
- Security updates are prioritized and released promptly
- We maintain an inventory of all third-party components

## Responsible Disclosure Policy

We believe that coordinated disclosure of security vulnerabilities is in everyone's best interest. We commit to:

- Working with security researchers to understand and resolve issues
- Providing credit to researchers who report vulnerabilities responsibly
- Not taking legal action against researchers who follow our responsible disclosure process
- Maintaining transparency about security issues while protecting users

## Contact Information

For security-related questions or concerns:

- **Email**: security@lemkin.org
- **PGP Key**: Available on request
- **Response Time**: Within 48 hours for initial acknowledgment

For general questions about lemkin-dashboard:
- **GitHub Issues**: https://github.com/lemkin-org/lemkin-dashboard/issues
- **Documentation**: https://docs.lemkin.org

---

*This security policy is part of the Lemkin AI commitment to protecting human rights investigators and their sensitive work.*