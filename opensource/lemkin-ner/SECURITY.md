# Security Policy - Lemkin Ner

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |
| < 0.1   | :x:                |

## Reporting a Vulnerability

If you discover a security vulnerability in the Lemkin Ner module, please report it responsibly:

1. **DO NOT** open a public issue
2. Email: security@lemkin.org
3. Include:
   - Module name and version
   - Description of vulnerability
   - Steps to reproduce
   - Potential impact
   - Suggested fix (if any)

## Security Measures

### Data Protection
- All sensitive data is encrypted at rest
- PII is automatically detected and protected
- Audit logs maintain integrity verification

### Input Validation
- All user inputs are validated
- File uploads are scanned for malicious content
- SQL injection prevention
- XSS protection

### Authentication & Authorization
- Role-based access control
- Secure token management
- Session timeout policies

### Ner-Specific Security

- Entity extraction accuracy
- PII handling in entities
- Cross-reference security
- Entity linking validation

## Best Practices

### For Users
- Keep the module updated
- Use strong authentication
- Regularly review audit logs
- Follow data handling guidelines

### For Developers
- Never commit secrets
- Use environment variables
- Validate all inputs
- Implement proper error handling
- Follow secure coding guidelines

## Compliance

This module adheres to:
- GDPR requirements
- International legal standards
- Evidence handling protocols
- Chain of custody requirements

## Security Checklist

- [ ] Input validation implemented
- [ ] Output encoding in place
- [ ] Authentication required
- [ ] Authorization checks
- [ ] Audit logging enabled
- [ ] Error handling secure
- [ ] Secrets management proper
- [ ] Dependencies updated
- [ ] Security tests written
- [ ] Documentation current

## Contact

Security Team: security@lemkin.org
