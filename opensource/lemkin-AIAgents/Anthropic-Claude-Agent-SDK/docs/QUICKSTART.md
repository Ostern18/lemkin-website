# LemkinAI Agents - Quick Start Guide

Get up and running with LemkinAI document processing agents in minutes.

## Installation

### Prerequisites

- Python 3.9 or higher
- Anthropic API key ([get one here](https://console.anthropic.com/))

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd Lemkin_Agents
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set your API key:
```bash
export ANTHROPIC_API_KEY='your-api-key-here'
```

Or create a `.env` file:
```
ANTHROPIC_API_KEY=your-api-key-here
```

## Quick Examples

### Example 1: Parse a Document

```python
from agents.document_parser.agent import DocumentParserAgent

# Initialize agent
parser = DocumentParserAgent()

# Parse a document
result = parser.parse_document(
    file_path="path/to/document.pdf",
    source="Police Department",
    case_id="CASE-2024-001"
)

# Access results
print(f"Document type: {result['document_type']}")
print(f"Extracted text: {result['extracted_text']['full_text'][:500]}")
print(f"Key dates: {result['key_fields']['dates']}")
print(f"Confidence: {result['confidence_scores']['overall']}")
```

### Example 2: Analyze Medical Evidence

```python
from agents.medical_forensic_analyst.agent import MedicalForensicAnalystAgent

# Initialize agent
medical = MedicalForensicAnalystAgent()

# Analyze for torture indicators
result = medical.analyze_for_torture(
    medical_record="""
    Patient presents with multiple contusions on back and shoulders,
    consistent with blunt force trauma. Patient reports being beaten
    while in detention...
    """,
    case_id="CASE-2024-002"
)

# Check findings
if result['torture_indicators']['present']:
    print("Torture indicators found:")
    for indicator in result['torture_indicators']['indicators_found']:
        print(f"  - {indicator['indicator']}")
```

### Example 3: Compare Documents

```python
from agents.comparative_analyzer.agent import ComparativeAnalyzerAgent

# Initialize agent
comparator = ComparativeAnalyzerAgent()

# Compare two versions
result = comparator.compare_versions(
    original_doc={"text": "Original contract dated Jan 1, 2024..."},
    modified_doc={"text": "Modified contract dated Jan 1, 2024..."},
    case_id="CASE-2024-003"
)

# Review differences
print(f"Similarity: {result['comparison_results']['overall_similarity']}")
print(f"Differences found: {len(result['differences_identified'])}")

for diff in result['differences_identified'][:5]:
    print(f"\n{diff['type']}: {diff['description']}")
```

### Example 4: Identify Evidence Gaps

```python
from agents.evidence_gap_identifier.agent import EvidenceGapIdentifierAgent

# Initialize agent
gap_finder = EvidenceGapIdentifierAgent()

# Analyze case gaps
result = gap_finder.process({
    'charges': ['torture', 'unlawful_detention'],
    'available_evidence': [
        "Medical report showing injuries",
        "Witness statement from victim",
        "Detention facility photographs"
    ],
    'case_theory': "Systematic torture during unlawful military detention",
    'case_id': "CASE-2024-004"
})

# Review gaps and recommendations
print(f"Total gaps: {result['gap_summary']['total_gaps']}")
print(f"Critical gaps: {result['gap_summary']['critical']}")

print("\nTop 5 Priority Actions:")
for i, action in enumerate(result['critical_next_steps'][:5], 1):
    print(f"{i}. [{action['priority']}] {action['description']}")
```

## Multi-Agent Workflow

Combine agents for complete investigation:

```python
from agents.document_parser.agent import DocumentParserAgent
from agents.medical_forensic_analyst.agent import MedicalForensicAnalystAgent
from agents.evidence_gap_identifier.agent import EvidenceGapIdentifierAgent
from shared import AuditLogger, EvidenceHandler

# Setup shared infrastructure
audit_logger = AuditLogger()
evidence_handler = EvidenceHandler()

# Initialize agents with shared infrastructure
parser = DocumentParserAgent(
    audit_logger=audit_logger,
    evidence_handler=evidence_handler
)

medical = MedicalForensicAnalystAgent(
    audit_logger=audit_logger,
    evidence_handler=evidence_handler
)

gap_finder = EvidenceGapIdentifierAgent(
    audit_logger=audit_logger,
    evidence_handler=evidence_handler
)

# Step 1: Parse medical report
parse_result = parser.parse_document(
    file_path="medical_report.pdf",
    source="Hospital",
    case_id="CASE-2024-005"
)

evidence_id = parse_result['evidence_id']
extracted_text = parse_result['extracted_text']['full_text']

# Step 2: Analyze medical content
medical_result = medical.analyze_for_torture(
    medical_record=extracted_text,
    evidence_id=evidence_id,
    case_id="CASE-2024-005"
)

# Step 3: Identify gaps
gap_result = gap_finder.process({
    'charges': ['torture'],
    'available_evidence': [
        {
            'evidence_id': evidence_id,
            'type': 'medical_report',
            'findings': medical_result['key_findings']
        }
    ],
    'case_id': "CASE-2024-005"
})

# Step 4: Review complete chain of custody
chain = audit_logger.get_evidence_chain(evidence_id)
print(f"\nChain of custody events: {len(chain)}")

for event in chain:
    print(f"  {event['timestamp']}: {event['event_type']} by {event['agent_id']}")

# Verify integrity
integrity_ok = audit_logger.verify_chain_integrity()
print(f"\nChain integrity verified: {'✓' if integrity_ok else '✗'}")
```

## Configuration

Each agent can be configured:

```python
from agents.document_parser.agent import DocumentParserAgent
from agents.document_parser.config import ParserConfig

# Custom configuration
config = ParserConfig(
    temperature=0.0,  # Maximum accuracy
    min_confidence_threshold=0.85,
    enable_table_extraction=True,
    human_review_threshold=0.7
)

parser = DocumentParserAgent(config=config)
```

Pre-configured profiles available:
- `DEFAULT_CONFIG`: Balanced settings
- `HIGH_ACCURACY_CONFIG`: Maximum accuracy for critical evidence
- `FAST_CONFIG`: Quick processing for triage

## Best Practices

1. **Always use shared infrastructure** for multi-agent workflows
2. **Specify case_id** to track evidence by case
3. **Use tags** for easy evidence retrieval
4. **Check confidence scores** before using results
5. **Review audit logs** regularly to ensure chain of custody
6. **Enable human review** for critical decisions
7. **Test with sample data** before processing real evidence

## Troubleshooting

### API Key Issues
```python
# Explicitly pass API key
parser = DocumentParserAgent(api_key="your-key-here")
```

### Import Errors
```python
# Add parent directory to path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
```

### File Not Found
```python
# Use absolute paths
from pathlib import Path
file_path = Path("/absolute/path/to/document.pdf")
```

## Next Steps

- Read full documentation in `/docs`
- Review agent-specific READMEs in each agent directory
- Explore workflow examples in `/examples/workflows`
- Run integration tests: `pytest tests/integration`

## Getting Help

- Check agent-specific documentation in each `/agents/*/README.md`
- Review test files for usage examples
- See complete workflows in `/examples/workflows`

## API Reference

See individual agent documentation:
- [Document Parser](../agents/document-parser/README.md)
- [Comparative Analyzer](../agents/comparative-analyzer/README.md)
- [Medical/Forensic Analyst](../agents/medical-forensic-analyst/README.md)
- [Gap Identifier](../agents/evidence-gap-identifier/README.md)
