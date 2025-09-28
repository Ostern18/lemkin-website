# Contributing to Lemkin Digital Forensics Helpers

Thank you for contributing to digital forensics capabilities for legal investigations!

## Development Setup

```bash
git clone https://github.com/lemkin-org/lemkin-forensics.git
cd lemkin-forensics
pip install -e ".[dev]"
pre-commit install
```

## Key Contribution Areas

- **File Analysis**: Enhanced file system and metadata analysis
- **Network Forensics**: Improved network traffic and log analysis
- **Authenticity Verification**: Advanced digital signature and tampering detection
- **Chain of Custody**: Robust evidence tracking and audit capabilities

## Forensics-Specific Guidelines

### Evidence Integrity
```python
def analyze_digital_evidence(
    evidence_path: Path,
    preserve_original: bool = True
) -> ForensicsResult:
    \"\"\"Analyze digital evidence while preserving integrity.

    Args:
        evidence_path: Path to evidence file
        preserve_original: Whether to preserve original evidence

    Returns:
        Forensics analysis with full audit trail
    \"\"\"
    if preserve_original:
        evidence_hash = calculate_evidence_hash(evidence_path)
        logger.info(f"Evidence hash: {evidence_hash}")

    # Work with forensic copy, never modify original
    working_copy = create_forensic_copy(evidence_path)
    return perform_analysis(working_copy)
```

### Security Requirements
- Never modify original evidence files
- Use cryptographic hashing for integrity verification
- Implement secure chain of custody tracking
- Maintain detailed audit logs for legal proceedings

### Testing Standards
- Use synthetic test evidence, never real case data
- Test evidence preservation and integrity verification
- Verify audit trail completeness
- Include performance tests for large evidence files

## Legal Compliance Focus

- **Chain of Custody**: Maintain unbroken evidence trails
- **Integrity**: Ensure evidence authenticity and completeness
- **Admissibility**: Follow legal standards for digital evidence
- **Privacy**: Protect sensitive information in evidence

## Contact

- **Technical**: Open GitHub Issues
- **Security**: security@lemkin.org
- **Legal**: legal@lemkin.org

---

*Help ensure digital evidence integrity in legal investigations worldwide.*