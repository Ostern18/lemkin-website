# Digital Forensics & Metadata Analyst Agent

The Digital Forensics & Metadata Analyst Agent analyzes digital evidence and metadata for legal investigations. It extracts, authenticates, and interprets digital artifacts while maintaining chain of custody and ensuring evidence admissibility in legal proceedings.

## Overview

This agent specializes in analyzing digital evidence including file metadata, communication records, timestamps, and authentication markers. It provides expert analysis of digital artifacts for human rights investigations and legal proceedings while maintaining forensic integrity and chain of custody.

## Core Capabilities

### Metadata Analysis
- **EXIF data extraction** from images and documents
- **File modification history** tracking and analysis
- **Creation and access timestamps** correlation and verification
- **Geolocation data** extraction and validation
- **Device fingerprinting** from metadata artifacts
- **Author and editor tracking** across multiple versions

### Digital Authentication
- **Tampering detection** through metadata inconsistencies
- **Hash verification** for file integrity checking
- **Digital signature validation** and certificate analysis
- **Modification history** reconstruction and analysis
- **Authentication markers** identification and assessment
- **Forensic timeline** construction from digital artifacts

### Communication Analysis
- **Email header analysis** for routing and authentication
- **Messaging metadata** extraction and verification
- **Social media timestamps** and location data analysis
- **Communication pattern** identification and mapping
- **Network analysis** from digital interactions
- **Coordination indicators** from communication timing

## Key Features

### Forensic Standards Compliance
- Chain of custody maintenance throughout analysis
- Evidence handling according to digital forensics best practices
- Admissibility assessment for legal proceedings
- Integrity verification with cryptographic hashing
- Reproducible analysis methodology
- Expert report generation for court testimony

### Technical Expertise
- Multiple file format analysis (images, documents, media)
- Cross-platform metadata interpretation
- Timezone and timestamp standardization
- Data correlation across multiple artifacts
- Pattern recognition in digital behavior
- Technical explanation for non-expert audiences

### Legal Integration
- Evidence authentication for legal admissibility
- Expert opinion formulation with confidence levels
- Technical findings translation to legal context
- Support for expert witness testimony
- Compliance with evidentiary rules
- Documentation suitable for legal proceedings

## Configuration Options

### Default Configuration
- Comprehensive forensic analysis
- High-confidence threshold for authentication conclusions
- Detailed technical reporting with evidence preservation
- Chain of custody tracking enabled
- Human review for critical authenticity determinations

### Analysis Parameters
- `metadata_extraction_depth`: Level of metadata detail to extract
- `authentication_threshold`: Confidence level required for authentication conclusions
- `timeline_precision`: Granularity of timeline reconstruction
- `tampering_sensitivity`: Threshold for flagging potential tampering
- `technical_detail_level`: Amount of technical detail in reports

## Input Format

The agent accepts:

```python
input_data = {
    'digital_files': [
        {
            'file_path': 'path/to/file.jpg',
            'file_data': '...',  # or base64 encoded
            'known_metadata': {...}
        }
    ],
    'file_metadata': {
        'extraction_method': 'exiftool|manual',
        'collection_date': '2024-01-15',
        'collector': 'Investigator Name'
    },
    'communication_data': {
        'email_headers': [...],
        'message_metadata': [...]
    },
    'timeline_data': {
        'events': [...],
        'reference_timezone': 'UTC'
    },
    'comparison_files': [...],  # Files to compare
    'case_id': 'CASE-2024-001',
    'analysis_focus': ['authentication', 'timeline', 'tampering'],
    'chain_of_custody_info': {...}
}
```

## Output Format

Returns structured analysis:

```python
{
    'analysis_id': 'uuid',
    'case_id': 'CASE-2024-001',
    'files_analyzed': 15,
    'metadata_extraction': {
        'file_id': {
            'exif_data': {...},
            'timestamps': {...},
            'geolocation': {...},
            'device_info': {...}
        }
    },
    'authentication_assessment': {
        'overall_authenticity': 'high|medium|low',
        'confidence_score': 0.85,
        'tampering_indicators': [...],
        'integrity_verification': {...}
    },
    'timeline_reconstruction': {
        'events': [...],
        'confidence_levels': {...},
        'gaps_identified': [...]
    },
    'communication_analysis': {
        'patterns': [...],
        'network_relationships': {...},
        'coordination_indicators': [...]
    },
    'findings_summary': "...",
    'technical_report': "...",
    'legal_admissibility': {
        'assessment': 'admissible|questionable|inadmissible',
        'concerns': [...],
        'recommendations': [...]
    },
    'chain_of_custody': {
        'evidence_tracking': [...],
        'integrity_maintained': true
    },
    'recommendations': [...]
}
```

## Usage Examples

### Basic Metadata Analysis

```python
from agents.digital_forensics_analyst.agent import DigitalForensicsAnalystAgent

# Initialize agent
analyst = DigitalForensicsAnalystAgent()

# Analyze digital evidence
result = analyst.process({
    'digital_files': [
        {'file_path': 'evidence/photo.jpg'},
        {'file_path': 'evidence/document.pdf'}
    ],
    'file_metadata': {
        'collection_date': '2024-01-15',
        'collector': 'Investigator Smith'
    },
    'case_id': 'CASE-2024-001',
    'analysis_focus': ['metadata', 'authentication', 'timeline']
})

print(f"Authentication: {result['authentication_assessment']['overall_authenticity']}")
print(f"Files analyzed: {result['files_analyzed']}")
```

### Timeline Reconstruction

```python
# Reconstruct timeline from multiple digital artifacts
result = analyst.process({
    'digital_files': [
        {'file_path': 'evidence/email.eml'},
        {'file_path': 'evidence/photo1.jpg'},
        {'file_path': 'evidence/photo2.jpg'}
    ],
    'timeline_data': {
        'reference_timezone': 'UTC',
        'known_events': [
            {'timestamp': '2024-01-15T14:30:00Z', 'description': 'Incident occurred'}
        ]
    },
    'case_id': 'CASE-2024-001',
    'analysis_focus': ['timeline', 'correlation']
})

for event in result['timeline_reconstruction']['events']:
    print(f"{event['timestamp']}: {event['description']}")
```

### Communication Pattern Analysis

```python
# Analyze communication patterns
result = analyst.process({
    'communication_data': {
        'email_headers': [
            {'from': 'suspect@domain.com', 'to': 'target@domain.com', 'date': '...'},
            # ... more emails
        ],
        'message_metadata': [...]
    },
    'case_id': 'CASE-2024-001',
    'analysis_focus': ['communication', 'patterns', 'coordination']
})

print(f"Patterns identified: {len(result['communication_analysis']['patterns'])}")
print(f"Coordination indicators: {result['communication_analysis']['coordination_indicators']}")
```

### Multi-Agent Workflow Integration

```python
from shared import AuditLogger, EvidenceHandler
from agents.document_parser.agent import DocumentParserAgent
from agents.digital_forensics_analyst.agent import DigitalForensicsAnalystAgent

# Shared infrastructure
shared_infra = {
    'audit_logger': AuditLogger(),
    'evidence_handler': EvidenceHandler()
}

# Initialize agents
parser = DocumentParserAgent(**shared_infra)
forensics = DigitalForensicsAnalystAgent(**shared_infra)

# Parse document
parsed = parser.parse_document("evidence/report.pdf", case_id="CASE-001")

# Analyze metadata
forensics_result = forensics.process({
    'digital_files': [{'file_path': 'evidence/report.pdf'}],
    'file_metadata': {'document_id': parsed['evidence_id']},
    'case_id': 'CASE-001',
    'analysis_focus': ['authentication', 'timeline', 'tampering']
})

# Verify chain of custody
chain = shared_infra['audit_logger'].get_evidence_chain(parsed['evidence_id'])
```

## Integration with Other Agents

### Document Parser
Provides digital forensics analysis of parsed documents, verifying authenticity and detecting tampering.

### Comparative Analyzer
Compares metadata across multiple versions to identify modifications and track document history.

### OSINT Synthesis
Validates digital artifacts collected from open sources, verifying timestamps and authentication markers.

### Evidence Gap Identifier
Identifies missing metadata or authentication data needed for evidence admissibility.

## Evidentiary Standards

The agent maintains strict evidentiary standards:

- **Chain of Custody**: All digital evidence handling logged with immutable audit trail
- **Integrity Verification**: SHA-256 hashing of all analyzed files
- **Expert Standards**: Analysis follows digital forensics best practices
- **Admissibility Assessment**: Evidence evaluated for legal admissibility
- **Reproducibility**: All analysis steps documented for replication
- **Human Review**: High-stakes authentication determinations flagged for expert review

## Technical Requirements

- Python 3.9+
- Anthropic API key (Claude Sonnet 4.5)
- Access to digital evidence files
- File metadata extraction capabilities

## Limitations

- Cannot recover deleted or encrypted data
- Relies on available metadata (may be incomplete)
- Authentication conclusions are probabilistic, not absolute
- Technical analysis requires human expert review for court testimony
- Some proprietary formats may have limited metadata access

## Use Cases

- Authentication of digital evidence for legal proceedings
- Timeline reconstruction from digital artifacts
- Tampering detection in submitted documents
- Email and communication analysis
- Social media evidence verification
- File integrity verification
- Chain of custody documentation for digital evidence
