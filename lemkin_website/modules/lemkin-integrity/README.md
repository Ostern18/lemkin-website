# Lemkin Evidence Integrity Toolkit

## Purpose

The Lemkin Evidence Integrity Toolkit provides cryptographic integrity verification and chain of custody management for legal evidence. This toolkit ensures that all evidence meets legal admissibility standards with complete audit trails, making it suitable for use in court proceedings and investigations.

## Safety & Ethics Notice

⚠️ **IMPORTANT**: This toolkit is designed for legitimate legal investigations and human rights work. Users must:
- Ensure proper legal authorization for evidence handling
- Protect witness and victim privacy
- Follow all applicable laws and regulations
- Maintain evidence confidentiality
- Use only for lawful purposes

## Key Features

- **Cryptographic Integrity**: SHA-256 and SHA-512 hashing with verification
- **Chain of Custody**: Complete audit trail with digital signatures
- **Court-Ready**: Generate manifests and packages for legal submission
- **Database Storage**: SQLite database for reliability and portability
- **CLI Interface**: Easy-to-use command-line tools
- **Export Capabilities**: Court-ready evidence packages

## Quick Start

```bash
# Install the toolkit
pip install lemkin-integrity

# Generate hash for evidence file
lemkin-integrity hash-evidence evidence.pdf \
    --case-id CASE-2024-001 \
    --collector "Investigator Name" \
    --source "Interview Recording"

# Add custody entry
lemkin-integrity add-custody <evidence-id> accessed "Legal Assistant" \
    --location "Legal Office"

# Verify integrity
lemkin-integrity verify <evidence-id> --file-path evidence.pdf

# Generate court manifest
lemkin-integrity generate-manifest CASE-2024-001 \
    --output-file manifest.json
```

## Usage Examples

### 1. Processing Evidence File

```bash
# Hash a witness statement
lemkin-integrity hash-evidence witness_statement.pdf \
    --case-id HR-2024-003 \
    --collector "Human Rights Investigator" \
    --source "Witness Interview" \
    --location "Field Office" \
    --description "Statement from civilian witness" \
    --tags "witness,civilian,testimony"
```

### 2. Managing Chain of Custody

```bash
# Record evidence access
lemkin-integrity add-custody abc-123-def accessed "Legal Analyst" \
    --location "Evidence Room" \
    --notes "Reviewed for case preparation"

# Record evidence transfer
lemkin-integrity add-custody abc-123-def transferred "Court Clerk" \
    --location "Courthouse" \
    --notes "Submitted for trial proceedings"
```

### 3. Verification and Reporting

```bash
# Verify evidence integrity
lemkin-integrity verify abc-123-def --file-path /path/to/current/file.pdf

# View custody chain
lemkin-integrity custody-chain abc-123-def

# Export complete evidence package
lemkin-integrity export-package HR-2024-003 ./evidence_package/
```

## Input/Output Specifications

### Evidence Metadata Structure
```python
{
    "filename": "witness_statement.pdf",
    "file_size": 2048576,
    "mime_type": "application/pdf", 
    "created_date": "2024-01-15T10:30:00Z",
    "source": "Interview Recording",
    "case_id": "HR-2024-003",
    "collector": "Investigator Name",
    "location": "Field Office",
    "description": "Statement from civilian witness",
    "tags": ["witness", "civilian", "testimony"]
}
```

### Integrity Report Format
```python
{
    "evidence_id": "abc-123-def",
    "status": "verified",
    "hash_verified": true,
    "custody_verified": true,
    "admissible": true,
    "timestamp": "2024-01-15T14:30:00Z",
    "issues": [],
    "recommendations": []
}
```

## Evaluation & Limitations

### Performance Metrics
- Hash generation: ~50MB/sec for SHA-256
- Database operations: <100ms for typical queries
- Integrity verification: <500ms for most files

### Known Limitations
- SQLite database may not scale beyond 10,000 evidence items
- Digital signatures require secure key management
- File modifications after hashing will fail integrity checks
- No built-in encryption of evidence files themselves

### Failure Modes
- Database corruption: Use backup and recovery procedures
- Key loss: Digital signatures cannot be verified
- File system errors: May prevent hash calculation
- Network issues: May affect timestamp synchronization

## Safety Guidelines

### Evidence Handling
1. **Always maintain original files**: Never modify evidence files
2. **Secure storage**: Store evidence in secure, access-controlled locations
3. **Key management**: Protect cryptographic keys used for signatures
4. **Regular verification**: Periodically verify evidence integrity
5. **Backup procedures**: Maintain secure backups of database

### Privacy Protection
1. **PII handling**: Be aware that metadata may contain sensitive information
2. **Access controls**: Limit database access to authorized personnel only
3. **Audit logging**: All evidence access is logged and cannot be deleted
4. **Data retention**: Follow legal requirements for evidence retention
5. **Disposal**: Securely dispose of evidence when legally permitted

### Legal Compliance
- Designed to meet international evidence standards
- Compatible with ICC, ECHR, and domestic court requirements
- Follows Berkeley Protocol for digital investigations
- Maintains chain of custody as required by law

## API Reference

### Core Classes

#### EvidenceIntegrityManager
Main class for managing evidence integrity and chain of custody.

```python
from lemkin_integrity import EvidenceIntegrityManager, EvidenceMetadata

# Initialize manager
manager = EvidenceIntegrityManager("evidence.db")

# Generate evidence hash
metadata = EvidenceMetadata(
    filename="evidence.pdf",
    file_size=1024,
    mime_type="application/pdf",
    created_date=datetime.now(),
    source="Investigation",
    case_id="CASE-001",
    collector="Investigator"
)
evidence_hash = manager.generate_evidence_hash("evidence.pdf", metadata)

# Verify integrity
report = manager.verify_integrity(evidence_hash.evidence_id)
```

#### Key Methods
- `generate_evidence_hash()`: Create hash for evidence file
- `create_custody_entry()`: Add chain of custody entry
- `verify_integrity()`: Verify evidence integrity
- `get_custody_chain()`: Retrieve custody history
- `generate_court_manifest()`: Create court submission manifest
- `export_evidence_package()`: Export complete evidence package

## Installation

### Requirements
- Python 3.10 or higher
- cryptography library for digital signatures
- SQLite for database storage

### Install from PyPI
```bash
pip install lemkin-integrity
```

### Install for Development
```bash
git clone https://github.com/lemkin-org/lemkin-integrity.git
cd lemkin-integrity
pip install -e ".[dev]"
```

## Contributing

We welcome contributions from the legal technology and human rights communities!

### Getting Started
1. Fork the repository
2. Create a feature branch
3. Make your changes with tests
4. Submit a pull request

### Development Setup
```bash
# Clone repository
git clone https://github.com/lemkin-org/lemkin-integrity.git
cd lemkin-integrity

# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run linting
black src/ tests/
ruff check src/ tests/
mypy src/
```

### Testing Requirements
- All new features must have unit tests
- Maintain >80% code coverage
- Test both success and failure cases
- Include CLI integration tests

### Code Standards
- Use type hints for all functions
- Follow PEP 8 style guidelines
- Write comprehensive docstrings
- Handle errors gracefully
- Log important operations

## License

Apache License 2.0 - see LICENSE file for details.

This toolkit is designed for legitimate legal investigations and human rights work. Users are responsible for ensuring proper legal authorization and compliance with applicable laws.

## Support

- GitHub Issues: Report bugs and request features
- Documentation: Full API docs at docs.lemkin.org
- Security Issues: security@lemkin.org
- Community: Join our Discord for discussions

---

*Part of the Lemkin AI open-source legal technology ecosystem.*