

# Multi-Format Document Parser Agent

Extracts and structures content from PDFs, images, and scanned documents for legal and investigative purposes.

## Purpose

The Document Parser Agent is the critical first step in evidence processing. It transforms unstructured documents into machine-readable, structured data while maintaining strict evidentiary standards through comprehensive audit logging and chain-of-custody tracking.

## Capabilities

- **Multi-Format Support**: Process PDFs (native and scanned), images (JPG, PNG), and mixed-content documents
- **Text Extraction**: Extract all readable text while preserving formatting and layout
- **Document Classification**: Automatically identify document type (witness statement, medical record, contract, etc.)
- **Key Field Extraction**: Extract dates, names, signatures, identifiers, and other critical metadata
- **Structure Recognition**: Parse headers, tables, lists, and complex layouts
- **Quality Assessment**: Identify illegible sections, OCR errors, and quality issues
- **Multi-Language Support**: Process documents in 15+ languages
- **Handwriting Detection**: Identify and flag handwritten content
- **Confidence Scoring**: Provide confidence levels for all extractions
- **Chain of Custody**: Full audit trail of all processing steps

## Usage

### Basic Usage

```python
from agents.document_parser import DocumentParserAgent

# Initialize agent
agent = DocumentParserAgent()

# Parse a document
result = agent.parse_document(
    file_path="evidence/witness_statement.pdf",
    source="Police Department",
    case_id="CASE-2024-001",
    tags=["witness", "statement"]
)

# Access extracted content
print(result['extracted_text']['full_text'])
print(result['key_fields']['dates'])
print(result['confidence_scores']['overall'])
```

### Advanced Configuration

```python
from agents.document_parser import DocumentParserAgent
from agents.document_parser.config import HIGH_ACCURACY_CONFIG

# Use high-accuracy configuration for critical evidence
agent = DocumentParserAgent(config=HIGH_ACCURACY_CONFIG)

result = agent.parse_document(
    file_path="critical_evidence.pdf",
    source="Crime Scene",
    case_id="CASE-2024-001",
    custodian="Detective Smith"
)

# Check if human review was triggered
if 'human_review_requested' in result:
    print(f"Human review needed: {result['human_review_requested']}")
```

### Batch Processing

```python
# Process multiple documents
file_paths = [
    "evidence/doc1.pdf",
    "evidence/doc2.pdf",
    "evidence/doc3.jpg"
]

results = agent.batch_process(
    file_paths=file_paths,
    source="Investigation Team",
    case_id="CASE-2024-001"
)

# Process results
for result in results:
    if 'error' in result:
        print(f"Failed: {result['file_path']}")
    else:
        print(f"Processed: {result['evidence_id']}")
```

### Custom Processing

```python
# Process with custom configuration
from agents.document_parser.config import ParserConfig

custom_config = ParserConfig(
    temperature=0.0,  # Maximum accuracy
    min_confidence_threshold=0.9,
    enable_table_extraction=True,
    human_review_threshold=0.8
)

agent = DocumentParserAgent(config=custom_config)

# Process from bytes
with open("document.pdf", "rb") as f:
    file_data = f.read()

result = agent.process({
    'file_data': file_data,
    'file_type': 'pdf',
    'source': 'Court',
    'filename': 'filing.pdf'
})
```

## Output Format

The agent returns structured JSON with the following schema:

```json
{
  "evidence_id": "UUID",
  "document_type": "witness_statement",
  "language": "en",
  "metadata": {
    "total_pages": 5,
    "has_handwriting": true,
    "quality_score": 0.92,
    "processing_notes": []
  },
  "extracted_text": {
    "full_text": "Complete extracted text...",
    "structured_content": {
      "pages": [...]
    }
  },
  "key_fields": {
    "dates": ["2024-01-15", "2024-01-20"],
    "names": ["John Smith (Witness)", "Jane Doe (Officer)"],
    "signatures": ["Handwritten signature present on page 5"],
    "identifiers": {
      "case_number": "CASE-2024-001",
      "statement_id": "WS-2024-123"
    },
    "locations": ["123 Main Street, Cityville"]
  },
  "tables": [],
  "quality_flags": [
    {
      "type": "warning",
      "location": "Page 3, bottom paragraph",
      "description": "Slight blurring, confidence 0.75",
      "severity": "low"
    }
  ],
  "confidence_scores": {
    "text_extraction": 0.95,
    "document_classification": 0.98,
    "key_field_extraction": 0.87,
    "overall": 0.92
  },
  "recommendations": [
    "Consider professional review of page 3 due to image quality"
  ],
  "_metadata": {
    "agent_id": "document_parser",
    "output_id": "...",
    "evidence_ids": ["..."],
    "audit_session": "..."
  }
}
```

## Configuration Options

### ParserConfig

- `model`: Claude model to use (default: claude-sonnet-4-5)
- `max_tokens`: Maximum response tokens (default: 8192)
- `temperature`: Sampling temperature (default: 0.1 for accuracy)
- `min_confidence_threshold`: Minimum confidence for auto-approval (default: 0.7)
- `enable_table_extraction`: Extract tables (default: True)
- `enable_handwriting_detection`: Detect handwriting (default: True)
- `human_review_threshold`: Trigger human review below this (default: 0.5)

### Pre-configured Profiles

- `DEFAULT_CONFIG`: Balanced accuracy and speed
- `HIGH_ACCURACY_CONFIG`: Maximum accuracy for critical evidence
- `FAST_CONFIG`: Quick processing for initial triage

## Chain of Custody

All operations are automatically logged for chain-of-custody compliance:

```python
# Retrieve chain of custody for evidence
chain = agent.get_chain_of_custody("evidence-uuid")

for event in chain:
    print(f"{event['timestamp']}: {event['event_type']}")
    print(f"  Details: {event['details']}")

# Verify integrity
is_valid = agent.verify_integrity()
print(f"Audit trail intact: {is_valid}")
```

## Human-in-the-Loop

The agent automatically requests human review when:
- Overall confidence falls below threshold (default: 0.5)
- High-severity quality issues are detected
- Critical evidence is being processed (configurable)

```python
# Manual human review request
review = agent.request_human_review(
    item_for_review=result,
    review_type="evidence_verification",
    priority="high"
)

# Later: record review completion
agent.complete_human_review(
    review_request_id=review['review_request_id'],
    decision="approved",
    reviewer_notes="Verified against original document",
    reviewer_id="Investigator-123"
)
```

## Error Handling

The agent includes comprehensive error handling:

```python
try:
    result = agent.parse_document("document.pdf")
except FileNotFoundError:
    print("Document not found")
except ValueError as e:
    print(f"Invalid input: {e}")
except Exception as e:
    print(f"Processing error: {e}")
    # Error is automatically logged to audit trail
```

## Integration

The Document Parser integrates with:
- **Evidence Handler**: Automatic evidence ingestion and tracking
- **Audit Logger**: Complete chain-of-custody logging
- **Output Formatter**: Standardized report generation

```python
from shared import EvidenceHandler, AuditLogger

# Use shared evidence handler
evidence_handler = EvidenceHandler()
audit_logger = AuditLogger()

agent = DocumentParserAgent(
    evidence_handler=evidence_handler,
    audit_logger=audit_logger
)

# All operations share the same evidence store and audit log
```

## Best Practices

1. **Always specify source and case_id** for proper evidence tracking
2. **Use HIGH_ACCURACY_CONFIG** for critical evidence (confessions, key documents)
3. **Review quality_flags** in output to identify potential issues
4. **Implement human review** for low-confidence extractions
5. **Preserve original files** - this agent creates structured derivatives
6. **Batch process when possible** for efficiency
7. **Check confidence_scores** before using extracted data in analysis
8. **Tag documents appropriately** for easy retrieval

## Testing

Run unit tests:
```bash
pytest agents/document-parser/tests/
```

## Requirements

- Python 3.9+
- anthropic >= 0.42.0
- PIL (Pillow) >= 10.0.0
- pypdf >= 5.0.0

See `requirements.txt` for complete dependencies.

## License

Part of the LemkinAI project - open source tools for human rights investigations.
