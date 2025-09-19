# Contributing to Lemkin Evidence Integrity Toolkit

Thank you for your interest in contributing to the Lemkin Evidence Integrity Toolkit! This project is part of the broader Lemkin AI ecosystem focused on democratizing legal technology for human rights investigators and public interest lawyers.

## Code of Conduct

This project adheres to a strict code of conduct focused on professional, respectful communication and the protection of human rights. By participating, you agree to:

- Maintain professional and respectful communication
- Focus on facts and evidence in all discussions
- Respect diverse legal traditions and perspectives
- Protect the privacy and safety of vulnerable populations
- Use the toolkit only for legitimate legal purposes

## How to Contribute

### Reporting Issues

Before creating an issue, please:
1. Check existing issues to avoid duplicates
2. Use the issue template when available
3. Provide clear, reproducible steps for bugs
4. Include relevant system information

### Legal and Technical Standards

All contributions must meet these standards:

#### Legal Content Review
- Any changes affecting legal processes require expert review
- New legal frameworks must be validated by qualified professionals
- Evidence handling must comply with international standards
- Privacy protections must be maintained or enhanced

#### Technical Standards
- Python 3.10+ with full type hints
- Comprehensive unit tests (>80% coverage)
- Security review for all contributions
- Documentation for all public APIs
- Error handling with informative messages

### Development Process

1. **Fork and Clone**
   ```bash
   git clone https://github.com/your-username/lemkin-integrity.git
   cd lemkin-integrity
   ```

2. **Set up Development Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\\Scripts\\activate
   pip install -e ".[dev]"
   pre-commit install
   ```

3. **Create Feature Branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

4. **Develop and Test**
   ```bash
   # Make your changes
   # Run tests
   pytest
   
   # Check code quality
   make lint
   
   # Format code
   make format
   ```

5. **Submit Pull Request**
   - Use clear, descriptive title
   - Reference related issues
   - Include test coverage
   - Update documentation if needed

### Testing Requirements

#### Unit Tests
```python
# Example test structure
def test_evidence_integrity_feature():
    """Test description following legal compliance requirements"""
    # Arrange
    manager = EvidenceIntegrityManager(temp_db)
    
    # Act  
    result = manager.some_operation()
    
    # Assert
    assert result.meets_legal_standard()
    assert result.preserves_chain_of_custody()
```

#### Integration Tests
- Test complete workflows end-to-end
- Verify legal compliance at each step
- Test error handling and recovery
- Validate export formats for court submission

#### Security Tests
- Test input validation and sanitization
- Verify cryptographic operations
- Test access controls and permissions
- Validate audit logging completeness

### Documentation Standards

#### Code Documentation
```python
def generate_evidence_hash(self, file_path: Path, metadata: EvidenceMetadata) -> EvidenceHash:
    """
    Generate cryptographic hash of evidence file with metadata.
    
    This function creates tamper-evident hashes suitable for legal proceedings
    and maintains chain of custody from point of creation.
    
    Args:
        file_path: Path to evidence file (must exist and be readable)
        metadata: Complete evidence metadata including case info
        
    Returns:
        EvidenceHash object containing SHA-256/SHA-512 hashes and metadata
        
    Raises:
        FileNotFoundError: If evidence file doesn't exist
        PermissionError: If file cannot be read
        ValidationError: If metadata is incomplete or invalid
        
    Legal Compliance:
        - Creates immutable hash record suitable for court admission
        - Maintains chain of custody from point of creation
        - Generates audit trail entry automatically
        
    Security:
        - Uses cryptographically secure hash algorithms
        - Validates input parameters to prevent injection
        - Logs all operations for forensic review
    """
```

#### User Documentation
- Clear step-by-step instructions
- Real-world examples with sample data
- Safety warnings and legal considerations
- Troubleshooting guides

### Security Guidelines

#### Data Protection
- Never log sensitive information
- Encrypt sensitive data at rest
- Use secure communication channels
- Implement proper access controls

#### Code Security
- Validate all inputs rigorously
- Use parameterized queries for database access
- Implement proper error handling
- Follow secure coding practices

#### Cryptographic Standards
- Use well-established algorithms (SHA-256, RSA)
- Implement proper key management
- Follow current security best practices
- Regular security reviews

### Legal Compliance Requirements

#### Evidence Standards
- Maintain immutable original evidence
- Create complete audit trails
- Support chain of custody requirements
- Generate court-admissible reports

#### Privacy Protection
- Implement PII detection and protection
- Provide data anonymization capabilities
- Support right-to-deletion where applicable
- Follow data minimization principles

#### International Standards
- Berkeley Protocol compliance for digital investigations
- ICC evidence standards for international cases
- Regional court requirements (ECHR, etc.)
- Domestic legal framework support

### Review Process

#### Code Review Checklist
- [ ] Functionality works as specified
- [ ] All tests pass with good coverage
- [ ] Security review completed
- [ ] Legal compliance verified
- [ ] Documentation updated
- [ ] Performance acceptable
- [ ] No breaking changes (or properly versioned)

#### Legal Review Triggers
- Changes to evidence handling procedures
- New legal framework support
- Modifications to chain of custody
- Export format changes
- Privacy/security modifications

### Release Process

1. **Version Bumping**: Follow semantic versioning
2. **Changelog**: Update with all changes
3. **Testing**: Full test suite on multiple Python versions
4. **Documentation**: Ensure all docs are current
5. **Security**: Final security review
6. **Legal**: Legal compliance verification
7. **Release**: Tagged release with signed commits

### Getting Help

- **Technical Questions**: GitHub Discussions
- **Security Issues**: security@lemkin.org (private)
- **Legal Questions**: legal@lemkin.org
- **General Support**: Discord community

### Recognition

Contributors will be recognized in:
- CHANGELOG.md for each release
- README.md contributors section
- Annual contributor recognition
- Conference presentations (with permission)

## Special Considerations for Legal Technology

### Ethical Responsibilities
As contributors to legal technology, we have special responsibilities:
- Ensure technology serves justice and human rights
- Protect vulnerable populations from harm
- Maintain evidence integrity above all else
- Support legitimate legal processes
- Refuse to enable surveillance or oppression

### Quality Standards
Legal technology requires exceptional quality:
- Zero tolerance for evidence corruption
- Comprehensive testing of all features
- Clear documentation of limitations
- Transparent about accuracy and reliability
- Regular audits and validation

Thank you for helping to democratize access to justice through technology!