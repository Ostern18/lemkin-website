# Security Policy

## Supported Versions

We provide security updates for the following versions:

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |
| < 0.1   | :x:                |

## Reporting a Vulnerability

**DO NOT** report security vulnerabilities through public GitHub issues.

Instead, please report security vulnerabilities to: **security@lemkin.org**

Include the following information:
- Description of the vulnerability
- Steps to reproduce
- Potential impact assessment
- Suggested remediation (if any)

### What to Expect

1. **Acknowledgment**: Within 24 hours
2. **Initial Assessment**: Within 72 hours  
3. **Regular Updates**: Weekly status updates
4. **Resolution Timeline**: Depends on severity
   - Critical: 7 days
   - High: 14 days
   - Medium: 30 days
   - Low: 90 days

## Security Considerations

### Evidence Integrity
- **Cryptographic Hashes**: SHA-256 and SHA-512 for evidence verification
- **Digital Signatures**: RSA-2048 for chain of custody authentication
- **Immutable Storage**: Original evidence never modified
- **Audit Trails**: Complete logging of all evidence access

### Data Protection
- **Encryption at Rest**: Database encryption recommended
- **Encryption in Transit**: TLS 1.3 for all communications
- **Key Management**: Secure storage of cryptographic keys
- **Access Controls**: Role-based access to evidence

### Database Security
- **SQL Injection**: Parameterized queries throughout
- **Input Validation**: All inputs validated and sanitized
- **Access Logging**: All database operations logged
- **Backup Security**: Encrypted backups with secure storage

### Privacy Protection
- **PII Handling**: Minimal PII storage with protection measures
- **Data Retention**: Configurable retention policies
- **Right to Deletion**: Support for evidence disposal
- **Access Monitoring**: Complete audit trails

## Security Best Practices

### For Users
1. **Database Security**
   - Store databases on encrypted filesystems
   - Use strong access controls
   - Regular security backups
   - Monitor for unauthorized access

2. **Key Management**
   - Protect private keys with appropriate measures
   - Use hardware security modules when possible
   - Regular key rotation for long-term deployments
   - Secure key backup and recovery

3. **Network Security**
   - Use VPNs for remote access
   - Implement network segmentation
   - Monitor network traffic
   - Regular security assessments

### For Developers
1. **Secure Coding**
   - Input validation on all data
   - Parameterized database queries
   - Proper error handling
   - Security-focused code reviews

2. **Dependencies**
   - Regular dependency updates
   - Security scanning of dependencies
   - Minimal dependency footprint
   - Trusted sources only

3. **Testing**
   - Security test cases
   - Penetration testing
   - Vulnerability scanning
   - Code security analysis

## Known Security Considerations

### Current Limitations
1. **Local Database**: SQLite provides limited multi-user security
2. **Key Storage**: Private keys stored in filesystem (consider HSM)
3. **Network Security**: No built-in network security features
4. **Audit Immutability**: Audit logs stored in same database

### Recommended Mitigations
1. **Database Encryption**: Use full-disk encryption
2. **Access Controls**: Implement OS-level access controls
3. **Network Security**: Deploy behind secure networks
4. **Monitoring**: Implement security monitoring and alerting

## Compliance Framework

### Standards Adherence
- **NIST Cybersecurity Framework**: Risk management approach
- **ISO 27001**: Information security management
- **SOC 2**: Security, availability, and confidentiality
- **GDPR**: Data protection and privacy (where applicable)

### Legal Compliance
- **Chain of Custody**: Meets legal requirements for evidence
- **Evidence Integrity**: Cryptographically verifiable
- **Audit Trails**: Complete and tamper-evident
- **Court Admissibility**: Designed for legal proceedings

## Incident Response

### Security Incident Classification
- **P0 - Critical**: Evidence integrity compromised
- **P1 - High**: Data exposure or unauthorized access
- **P2 - Medium**: System vulnerability or partial compromise
- **P3 - Low**: Minor security issues or policy violations

### Response Procedures
1. **Immediate**: Contain and assess impact
2. **Investigation**: Determine scope and cause
3. **Notification**: Inform affected users and authorities
4. **Remediation**: Fix vulnerabilities and restore security
5. **Post-Incident**: Review and improve security measures

## Security Contact

For security-related questions or concerns:
- **Email**: security@lemkin.org
- **PGP Key**: Available at keybase.io/lemkin
- **Response Time**: 24 hours for initial response

## Updates

This security policy is reviewed quarterly and updated as needed.
Last updated: [Current Date]