# Lemkin Digital Forensics Toolkit

A comprehensive digital forensics analysis toolkit designed for legal professionals investigating human rights violations, war crimes, and other legal matters requiring digital evidence analysis.

## Overview

The Lemkin Digital Forensics Toolkit provides professional-grade digital forensics capabilities while maintaining the highest standards for evidence handling, chain of custody, and legal admissibility. It supports comprehensive analysis of disk images, network traffic, mobile devices, and evidence authenticity verification.

## Features

### 🔍 **Comprehensive Evidence Analysis**
- **File System Analysis**: Complete disk image analysis with deleted file recovery
- **Network Traffic Analysis**: PCAP analysis, flow reconstruction, and communication patterns
- **Mobile Device Forensics**: iOS and Android backup analysis with app data extraction
- **Evidence Verification**: Cryptographic integrity checking and authenticity verification

### ⚖️ **Legal Compliance**
- **Chain of Custody**: Automated logging and verification of evidence handling
- **ISO 27037 Compliance**: Meets international standards for digital evidence handling
- **Court-Ready Reports**: Professional documentation suitable for legal proceedings
- **Hash Verification**: Multi-algorithm integrity checking (MD5, SHA1, SHA256, SHA512)

### 🛠️ **Professional Tools**
- **Timeline Generation**: Unified forensic timelines from multiple evidence sources
- **Case Management**: Structured case organization and documentation
- **Export Capabilities**: Multiple output formats (JSON, CSV, HTML)
- **CLI Interface**: User-friendly command-line tools for all operations

## Installation

### Prerequisites

```bash
# Python 3.8 or higher
python --version

# Required system tools (optional but recommended)
# The Sleuth Kit for disk image analysis
sudo apt-get install sleuthkit  # Ubuntu/Debian
brew install sleuthkit          # macOS

# libmagic for file type detection
sudo apt-get install libmagic1  # Ubuntu/Debian
brew install libmagic           # macOS
```

### Install from PyPI

```bash
pip install lemkin-forensics
```

### Install from Source

```bash
git clone https://github.com/lemkin-org/lemkin-frameworks
cd lemkin-frameworks/lemkin-forensics
pip install -e .
```

## Quick Start

### 1. Create a New Case

```bash
# Create a new forensics case
lemkin-forensics create-case "CASE-2024-001" "Human Rights Investigation" "John Investigator" \
  --client "Legal Aid Organization" \
  --legal-matter "Documentation of digital evidence for war crimes investigation"
```

### 2. Analyze Disk Image

```bash
# Comprehensive disk image analysis
lemkin-forensics analyze-filesystem /path/to/disk_image.E01 \
  --output-dir ./analysis_results \
  --recover-deleted \
  --timeline \
  --carve-files
```

### 3. Process Network Traffic

```bash
# Analyze network packet capture
lemkin-forensics process-network /path/to/capture.pcap \
  --output-dir ./network_analysis \
  --extract-files
```

### 4. Extract Mobile Data

```bash
# Extract data from mobile backup
lemkin-forensics extract-mobile /path/to/mobile_backup \
  --output-dir ./mobile_data \
  --backup-type ios \
  --extract-media
```

### 5. Verify Evidence Authenticity

```bash
# Comprehensive authenticity verification
lemkin-forensics verify-authenticity /path/to/evidence.pdf \
  --verification-level comprehensive \
  --expected-hash sha256:abc123... \
  --output-dir ./verification_report
```

### 6. Generate Timeline

```bash
# Create unified forensic timeline
lemkin-forensics generate-timeline ./case_directory \
  --output-dir ./timeline \
  --format html \
  --start-date 2024-01-01 \
  --end-date 2024-12-31
```

## Core Components

### DigitalForensicsAnalyzer

The main coordinator class that provides a unified interface for all forensics operations:

```python
from lemkin_forensics import DigitalForensicsAnalyzer, ForensicsConfig, EvidenceType

# Initialize analyzer
config = ForensicsConfig(
    enable_deleted_file_recovery=True,
    maintain_chain_of_custody=True,
    generate_detailed_reports=True
)
analyzer = DigitalForensicsAnalyzer(config)

# Create a case
case = analyzer.create_case(
    case_number="CASE-2024-001",
    case_name="Investigation Case",
    investigator="Digital Forensics Analyst"
)

# Add evidence
evidence = analyzer.add_evidence(
    file_path="/path/to/evidence.E01",
    evidence_type=EvidenceType.DISK_IMAGE,
    name="Seized Hard Drive",
    description="Primary storage device from suspect computer"
)

# Perform analysis
result = analyzer.analyze_evidence(evidence)
```

### FileAnalyzer

Comprehensive file system analysis using The Sleuth Kit and custom algorithms:

```python
from lemkin_forensics import FileAnalyzer

analyzer = FileAnalyzer()
analysis = analyzer.analyze_disk_image(evidence, output_dir="./recovered_files")

print(f"Files found: {len(analysis.file_artifacts)}")
print(f"Deleted files recovered: {analysis.deleted_files_recovered}")
print(f"Timeline events: {len(analysis.timeline_events)}")
```

### NetworkProcessor

Network traffic analysis and communication pattern detection:

```python
from lemkin_forensics import NetworkProcessor

processor = NetworkProcessor()
analysis = processor.analyze_pcap(evidence, output_dir="./extracted_files")

print(f"Network flows: {len(analysis.flows)}")
print(f"HTTP transactions: {len(analysis.http_transactions)}")
print(f"Suspicious activities: {len(analysis.suspicious_activities)}")
```

### MobileAnalyzer

Mobile device forensics for iOS and Android:

```python
from lemkin_forensics import MobileAnalyzer

analyzer = MobileAnalyzer()
extraction = analyzer.analyze_mobile_backup(evidence, backup_type="ios")

print(f"Contacts: {len(extraction.contacts)}")
print(f"Messages: {len(extraction.messages)}")
print(f"Location points: {len(extraction.location_data)}")
```

### AuthenticityVerifier

Evidence integrity and authenticity verification:

```python
from lemkin_forensics import AuthenticityVerifier

verifier = AuthenticityVerifier()
report = verifier.verify_evidence_authenticity(evidence, "comprehensive")

print(f"Authentic: {report.overall_authentic}")
print(f"Confidence: {report.confidence_score}/10.0")
print(f"Admissibility: {report.admissibility_assessment}")
```

## Evidence Types Supported

### Disk Images
- **EnCase (E01)**: Industry standard forensic imaging format
- **Raw (DD)**: Bit-for-bit disk copies
- **AFF (Advanced Forensic Format)**: Open source forensic format
- **VMDK**: Virtual machine disk images

### Network Captures
- **PCAP/PCAPNG**: Wireshark and tcpdump packet captures
- **Network Logs**: Apache, IIS, firewall logs
- **Flow Records**: NetFlow and sFlow data

### Mobile Backups
- **iOS**: iTunes/Finder backups, logical extractions
- **Android**: ADB backups, TWRP images
- **App Data**: WhatsApp, Signal, social media apps

### Documents & Media
- **Office Documents**: Word, Excel, PowerPoint with metadata
- **PDF Files**: Including digital signatures and forms
- **Images**: JPEG, PNG, TIFF with EXIF data
- **Videos**: MP4, AVI, MOV with embedded metadata

## Legal Considerations

### Chain of Custody

The toolkit automatically maintains detailed chain of custody logs for all evidence:

```python
# All evidence handling is automatically logged
evidence.add_custody_entry(
    action="Evidence received from client",
    operator="John Investigator",
    location="Forensics Lab A",
    hash_before="abc123...",
    notes="Evidence received in sealed container"
)
```

### Evidence Standards

The toolkit complies with international evidence handling standards:

- **ISO/IEC 27037**: Guidelines for identification, collection, acquisition and preservation of digital evidence
- **NIST SP 800-86**: Guide to Integrating Forensic Techniques into Incident Response
- **RFC 3227**: Guidelines for Evidence Collection and Archiving

### Admissibility Requirements

All analysis maintains requirements for legal admissibility:

1. **Integrity Verification**: Cryptographic hashes verify evidence hasn't been altered
2. **Chain of Custody**: Complete documentation of evidence handling
3. **Reproducible Results**: Detailed logging allows analysis reproduction
4. **Expert Documentation**: Professional reports suitable for court proceedings

## Configuration

### Configuration File

Create a `forensics_config.json` file:

```json
{
  "enable_deleted_file_recovery": true,
  "enable_metadata_extraction": true,
  "enable_timeline_generation": true,
  "maintain_chain_of_custody": true,
  "generate_detailed_reports": true,
  "max_file_size_mb": 1024,
  "analysis_timeout_minutes": 60,
  "sleuthkit_path": "/usr/local/bin",
  "evidence_handling_standard": "ISO_27037"
}
```

Use with CLI:

```bash
lemkin-forensics --config forensics_config.json analyze-filesystem disk.E01
```

### Environment Variables

```bash
export LEMKIN_FORENSICS_TSK_PATH="/usr/local/bin"
export LEMKIN_FORENSICS_OUTPUT_DIR="./forensics_output"
export LEMKIN_FORENSICS_LOG_LEVEL="INFO"
```

## Advanced Usage

### Custom Analysis Workflows

```python
from lemkin_forensics import DigitalForensicsAnalyzer

analyzer = DigitalForensicsAnalyzer()

# Create case and add multiple evidence items
case = analyzer.create_case("CASE-001", "Investigation", "Analyst")
disk_evidence = analyzer.add_evidence("disk.E01", EvidenceType.DISK_IMAGE, "Hard Drive")
network_evidence = analyzer.add_evidence("traffic.pcap", EvidenceType.NETWORK_CAPTURE, "Network Logs")
mobile_evidence = analyzer.add_evidence("mobile_backup", EvidenceType.MOBILE_BACKUP, "Phone Backup")

# Analyze all evidence
results = []
for evidence in case.evidence_items:
    result = analyzer.analyze_evidence(evidence)
    results.append(result)

# Generate comprehensive timeline
timeline = analyzer.generate_timeline(case)

# Export complete case
analyzer.export_case(case, Path("./case_export.json"))
```

### Integration with External Tools

The toolkit can integrate with external forensics tools:

```python
# Custom tool integration
config = ForensicsConfig(
    sleuthkit_path="/custom/path/to/tsk",
    volatility_path="/path/to/volatility",
    autopsy_path="/path/to/autopsy"
)
```

### Batch Processing

Process multiple evidence files:

```bash
# Process all PCAP files in directory
for pcap in *.pcap; do
    lemkin-forensics process-network "$pcap" --output-dir "./analysis_$(basename $pcap .pcap)"
done

# Analyze multiple disk images
find ./evidence -name "*.E01" -exec lemkin-forensics analyze-filesystem {} --output-dir "./analysis_{}" \;
```

## Troubleshooting

### Common Issues

#### Missing Dependencies

```bash
# Install missing system dependencies
sudo apt-get install sleuthkit libmagic1 python3-dev

# Install Python packages
pip install --upgrade lemkin-forensics
```

#### Permission Issues

```bash
# Ensure proper permissions for evidence files
chmod 444 evidence_file.E01  # Read-only to prevent modification
```

#### Large File Processing

```bash
# For large files, increase timeout and memory limits
lemkin-forensics analyze-filesystem large_image.E01 \
  --config config_large_files.json
```

### Debugging

Enable verbose logging:

```bash
lemkin-forensics --verbose analyze-filesystem disk.E01
```

Check log files:

```bash
tail -f ~/.lemkin_forensics/logs/forensics.log
```

### Performance Optimization

For large cases:

1. **Increase memory allocation**: Set `LEMKIN_MAX_MEMORY=8GB`
2. **Use SSD storage**: Store working files on fast storage
3. **Parallel processing**: Use `--max-concurrent 4` for multiple evidence items
4. **Selective analysis**: Use specific analysis types instead of comprehensive

## API Reference

### Core Classes

#### DigitalForensicsAnalyzer
- `create_case()`: Create new forensics case
- `add_evidence()`: Add evidence to case
- `analyze_evidence()`: Perform comprehensive analysis
- `generate_timeline()`: Create unified timeline
- `export_case()`: Export case data

#### FileAnalyzer
- `analyze_disk_image()`: Comprehensive disk analysis
- `recover_deleted_files()`: Deleted file recovery
- `generate_file_listing()`: File system enumeration

#### NetworkProcessor
- `analyze_pcap()`: PCAP file analysis
- `analyze_log_file()`: Network log analysis
- `detect_suspicious_activities()`: Threat detection

#### MobileAnalyzer
- `analyze_mobile_backup()`: Mobile data extraction
- `extract_messages()`: Communication analysis
- `analyze_location_data()`: Geospatial analysis

#### AuthenticityVerifier
- `verify_evidence_authenticity()`: Comprehensive verification
- `verify_file_hashes()`: Integrity checking
- `validate_chain_of_custody()`: Custody verification

### Data Models

#### DigitalEvidence
- Evidence metadata and chain of custody
- File integrity verification
- Temporal tracking

#### AnalysisResult
- Comprehensive analysis results
- Artifact collections
- Performance metrics

#### TimelineEvent
- Temporal event representation
- Cross-evidence correlation
- Legal significance tracking

## Contributing

We welcome contributions to the Lemkin Digital Forensics Toolkit. Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Setup

```bash
git clone https://github.com/lemkin-org/lemkin-frameworks
cd lemkin-frameworks/lemkin-forensics
pip install -e ".[dev]"
pytest tests/
```

### Code Standards

- Follow PEP 8 style guidelines
- Include comprehensive docstrings
- Add unit tests for new features
- Maintain chain of custody logging
- Document legal considerations

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

- **Documentation**: [docs.lemkin.org/forensics](https://docs.lemkin.org/forensics)
- **Issues**: [GitHub Issues](https://github.com/lemkin-org/lemkin-frameworks/issues)
- **Discussions**: [GitHub Discussions](https://github.com/lemkin-org/lemkin-frameworks/discussions)
- **Email**: support@lemkin.org

## Legal Disclaimer

This toolkit is designed for legitimate digital forensics investigations conducted by qualified professionals. Users are responsible for:

- Compliance with applicable laws and regulations
- Proper authorization for evidence collection and analysis
- Maintaining appropriate chain of custody procedures
- Professional and ethical use of the toolkit

The Lemkin Project and contributors are not responsible for misuse of this toolkit or any legal consequences arising from its use.

## Acknowledgments

- **The Sleuth Kit**: Foundation for disk image analysis
- **Wireshark**: Network protocol analysis inspiration
- **Digital Forensics Community**: Standards and best practices
- **Human Rights Organizations**: Requirements and use case guidance

---

*The Lemkin Digital Forensics Toolkit is part of the Lemkin Project's mission to support justice and accountability through technology.*