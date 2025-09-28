# Lemkin Digital Forensics Helpers

## Purpose

The Lemkin Digital Forensics Helpers provide accessible digital evidence analysis and authentication tools for non-technical legal investigators. This toolkit enables investigators to analyze file systems, network logs, mobile device backups, and verify digital evidence authenticity without requiring deep forensic expertise.

## Safety & Ethics Notice

⚠️ **IMPORTANT**: This toolkit is designed for legitimate legal investigations and human rights documentation. Users must:
- Obtain proper legal authorization before analyzing digital evidence
- Respect privacy rights and follow applicable laws
- Maintain chain of custody for all digital evidence
- Use forensically sound methods for evidence preservation
- Document all analysis procedures for court admissibility

## Key Features

- **File System Analysis**: Scan directories and disk images for forensic evidence
- **Deleted File Recovery**: Identify and recover deleted files (simulation)
- **Network Log Analysis**: Process network logs for suspicious activities
- **Mobile Data Extraction**: Extract data from mobile device backups
- **Authenticity Verification**: Verify digital evidence integrity and authenticity
- **Hash Calculation**: Calculate cryptographic hashes for evidence integrity

## Quick Start

```bash
# Install the toolkit
pip install lemkin-forensics

# Analyze file system
lemkin-forensics analyze-files /path/to/evidence --output analysis.json

# Analyze network logs
lemkin-forensics analyze-network "server.log,firewall.log" --output network_analysis.json

# Extract mobile data
lemkin-forensics extract-mobile /path/to/backup --output mobile_data.json

# Verify evidence authenticity
lemkin-forensics verify-authenticity evidence.pdf --evidence-id "DOC001"

# Calculate file hashes
lemkin-forensics hash-file important_document.pdf --algorithms "md5,sha256,sha512"
```

## Usage Examples

### 1. File System Forensic Analysis

```bash
# Comprehensive file system analysis
lemkin-forensics analyze-files /media/evidence-drive \
    --output filesystem_analysis.json \
    --include-deleted true \
    --show-hidden true
```

### 2. Network Traffic Analysis

```bash
# Analyze multiple network log files
lemkin-forensics analyze-network "access.log,error.log,firewall.log" \
    --output network_forensics.json \
    --show-suspicious true
```

### 3. Mobile Device Data Extraction

```bash
# Extract data from mobile backup
lemkin-forensics extract-mobile /path/to/ios_backup \
    --output mobile_extraction.json \
    --include-location false \
    --include-messages true
```

### 4. Evidence Integrity Verification

```bash
# Verify document authenticity
lemkin-forensics verify-authenticity witness_statement.pdf \
    --evidence-id "WIT-2024-001" \
    --evidence-type "document" \
    --output authenticity_report.json
```

### 5. Comprehensive Evidence Processing

```bash
# Calculate multiple hashes for chain of custody
lemkin-forensics hash-file critical_evidence.zip \
    --algorithms "md5,sha1,sha256,sha512"
```

## Input/Output Specifications

### File Metadata Format
```python
{
    "file_path": "/path/to/file.pdf",
    "file_name": "document.pdf",
    "file_size": 2048576,
    "mime_type": "application/pdf",
    "file_type": "document",
    "created_date": "2024-01-15T10:30:00Z",
    "modified_date": "2024-01-15T10:35:00Z",
    "md5_hash": "abc123...",
    "sha256_hash": "def456...",
    "is_hidden": false,
    "is_encrypted": false
}
```

### Network Analysis Format
```python
{
    "analysis_id": "uuid",
    "total_entries": 50000,
    "date_range": {
        "start": "2024-01-01T00:00:00Z",
        "end": "2024-01-31T23:59:59Z"
    },
    "top_sources": [
        {
            "ip": "192.168.1.100",
            "connections": 1500,
            "total_bytes": 10485760
        }
    ],
    "suspicious_activities": [
        {
            "type": "port_scanning",
            "source_ip": "10.0.0.50",
            "description": "Possible port scanning detected"
        }
    ]
}
```

### Mobile Extraction Format
```python
{
    "extraction_id": "uuid",
    "device_info": {
        "device_type": "smartphone",
        "os_version": "iOS 17.0",
        "model": "iPhone 14"
    },
    "contacts": [
        {
            "name": "John Doe",
            "phone": "+1234567890",
            "email": "john@example.com"
        }
    ],
    "messages": [
        {
            "sender": "+1234567890",
            "recipient": "+0987654321",
            "content": "Message content",
            "timestamp": "2024-01-15T14:30:00Z",
            "type": "SMS"
        }
    ]
}
```

## API Reference

### Core Classes

#### FileAnalyzer
Analyzes file systems and identifies forensic artifacts.

```python
from lemkin_forensics import FileAnalyzer
from pathlib import Path

analyzer = FileAnalyzer()
analysis = analyzer.analyze_file_system(Path("/evidence"))

print(f"Found {len(analysis.suspicious_files)} suspicious files")
print(f"Recovered {len(analysis.deleted_files)} deleted files")
```

#### NetworkProcessor
Processes network logs for forensic analysis.

```python
from lemkin_forensics import NetworkProcessor
from pathlib import Path

processor = NetworkProcessor()
log_files = [Path("access.log"), Path("error.log")]
analysis = processor.process_network_logs(log_files)

print(f"Analyzed {analysis.total_entries} network entries")
for activity in analysis.suspicious_activities:
    print(f"Suspicious: {activity['description']}")
```

#### MobileAnalyzer
Extracts data from mobile device backups.

```python
from lemkin_forensics import MobileAnalyzer
from pathlib import Path

analyzer = MobileAnalyzer()
extraction = analyzer.extract_mobile_data(Path("/backup"))

print(f"Found {len(extraction.contacts)} contacts")
print(f"Found {len(extraction.messages)} messages")
```

#### AuthenticityVerifier
Verifies digital evidence authenticity.

```python
from lemkin_forensics import AuthenticityVerifier, DigitalEvidence
from pathlib import Path

evidence = DigitalEvidence(
    evidence_id="DOC001",
    evidence_type="document",
    file_path=Path("evidence.pdf")
)

verifier = AuthenticityVerifier()
report = verifier.verify_digital_authenticity(evidence)

print(f"Authenticity: {report.status.value}")
print(f"Confidence: {report.confidence_score:.1%}")
```

## Evaluation & Limitations

### Performance Metrics
- File system scanning: ~1000 files/minute
- Network log processing: ~10,000 entries/minute
- Hash calculation: ~50MB/second for SHA-256
- Mobile data extraction: Depends on backup size

### Known Limitations
- Deleted file recovery is simulated (requires forensic tools integration)
- Mobile extraction supports common backup formats only
- Network log parsing supports standard formats
- Cannot decrypt encrypted files without keys
- Some metadata may be filesystem-dependent

### Failure Modes
- Large file systems: Use progress indicators, consider chunking
- Corrupted files: Skip with error logging
- Permission errors: Run with appropriate privileges
- Memory limitations: Stream processing for large files

## Safety Guidelines

### Evidence Handling
1. **Chain of Custody**: Document all evidence handling steps
2. **Write Protection**: Use write-blockers for original evidence
3. **Working Copies**: Always work with forensic copies
4. **Integrity Verification**: Calculate hashes before and after analysis
5. **Secure Storage**: Store evidence in secure, controlled environments

### Legal Compliance
- Obtain proper search warrants or authorization
- Follow jurisdiction-specific evidence rules
- Maintain detailed logs of all analysis steps
- Use forensically sound methodologies
- Prepare for legal testimony requirements

### Privacy Protection
1. **PII Handling**: Protect personally identifiable information
2. **Privileged Content**: Respect attorney-client privilege
3. **Victim Privacy**: Protect sensitive victim information
4. **Data Minimization**: Extract only relevant evidence
5. **Secure Disposal**: Properly dispose of evidence copies when authorized

## Contributing

We welcome contributions that enhance digital forensics capabilities for legal investigations.

### Development Setup
```bash
# Clone repository
git clone https://github.com/lemkin-org/lemkin-forensics.git
cd lemkin-forensics

# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Format code
black src/ tests/
```

### Testing Guidelines
- Test with various file systems and formats
- Include edge cases and error conditions
- Verify hash calculations against known values
- Test with real (anonymized) mobile backups
- Validate network log parsing accuracy

## License

Apache License 2.0 - see LICENSE file for details.

This toolkit is designed for legitimate legal investigations and human rights documentation. Users are responsible for ensuring compliance with all applicable laws regarding digital evidence handling and analysis.

---

*Part of the Lemkin AI open-source legal technology ecosystem.*