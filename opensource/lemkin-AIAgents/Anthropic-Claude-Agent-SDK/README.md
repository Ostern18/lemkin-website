# LemkinAI Anthropic Claude Agent SDK

**Open-source AI agents for human rights investigations and legal documentation**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

## What LemkinAI Enables

LemkinAI provides AI-powered automation for the time-consuming, labor-intensive work of processing evidence in human rights cases, criminal investigations, and legal proceedings. The agents handle document parsing, pattern detection across large datasets, medical record interpretation, satellite imagery analysis, and evidence gap identification—tasks that traditionally require weeks of manual review. Each agent maintains cryptographic audit trails and chain-of-custody logs, ensuring outputs meet evidentiary standards for legal proceedings while accelerating investigations from months to days.

**Key capabilities:** Parse multilingual documents at scale, identify torture indicators in medical records using Istanbul Protocol standards, analyze satellite imagery for mass graves or detention facilities, cross-reference thousands of witness statements for patterns, and automatically flag missing evidence against legal charge elements.

## Overview

LemkinAI provides specialized AI agents built on **Claude Sonnet 4.5** for processing documents, analyzing evidence, and supporting legal investigations. Each agent is designed for specific investigative tasks with built-in:

- **Chain-of-custody tracking** - Immutable audit logs for every operation
- **Human-in-the-loop gates** - Automatic review triggers for high-stakes decisions
- **Evidentiary compliance** - Output formats suitable for legal proceedings
- **Multi-agent workflows** - Agents work together seamlessly

## Features

**18 Specialized Agents** across three domains:
- Investigative Research & Intelligence (5 agents)
- Document Processing & Analysis (4 agents)
- Crime-Specific Analysis (9 agents)

**Production-Ready** with comprehensive error handling, batch processing, and testing

**Evidentiary Standards** including verifiability, source provenance, and audit trails

**Multi-Language Support** for international investigations (15+ languages)

## Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd Anthropic-Claude-Agent-SDK

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set your Anthropic API key
export ANTHROPIC_API_KEY='your-api-key-here'
```

### Basic Usage

```python
from agents.document_parser.agent import DocumentParserAgent
from shared import AuditLogger, EvidenceHandler

# Initialize agent with evidentiary infrastructure
audit_logger = AuditLogger()
evidence_handler = EvidenceHandler()

parser = DocumentParserAgent(
    audit_logger=audit_logger,
    evidence_handler=evidence_handler
)

# Parse a document
result = parser.process({
    'file_path': 'evidence/document.pdf',
    'source': 'Hospital Records',
    'case_id': 'CASE-2024-001'
})

print(f"Document Type: {result['document_type']}")
print(f"Confidence: {result['confidence_scores']['overall']}")
print(f"Extracted Text: {result['extracted_text']['full_text'][:200]}...")

# Verify chain of custody
integrity = audit_logger.verify_chain_integrity()
print(f"Audit trail verified: {integrity}")
```

## Available Agents

### Domain 1: Investigative Research & Intelligence

| Agent | Purpose | Status |
|-------|---------|--------|
| **OSINT Synthesis** | Aggregate and analyze publicly available information | Implemented |
| **Satellite Imagery Analyst** | Interpret satellite/aerial imagery for evidence | Implemented |
| **Social Media Harvester** | Collect and contextualize social media evidence | Implemented |
| **Historical Researcher** | Provide deep background on conflicts and regions | Implemented |
| **Legal Advisor** | Legal framework and jurisdictional guidance | Implemented |

### Domain 2: Document Processing & Analysis

| Agent | Purpose | Status |
|-------|---------|--------|
| **Document Parser** | Extract and structure content from documents | Implemented |
| **Comparative Analyzer** | Identify patterns across multiple documents | Implemented |
| **Medical/Forensic Analyst** | Interpret medical records and forensic reports | Implemented |
| **Evidence Gap Identifier** | Identify missing evidence and recommend next steps | Implemented |

### Domain 3: Crime-Specific Analysis

| Agent | Purpose | Status |
|-------|---------|--------|
| **Torture Analyst** | Document torture using Istanbul Protocol standards | Implemented |
| **Genocide Intent Analyzer** | Assess genocide indicators and intent | Implemented |
| **Disappearance Investigator** | Track enforced disappearances | Implemented |
| **NGO/UN Reporter** | Generate reports for international bodies | Implemented |
| **Digital Forensics Analyst** | Analyze digital evidence and metadata | Implemented |
| **Forensic Analysis Reviewer** | Review and validate forensic findings | Implemented |
| **Siege/Starvation Analyst** | Document siege warfare and starvation tactics | Implemented |
| **Ballistics/Weapons Identifier** | Identify weapons and munitions | Implemented |
| **Military Structure Analyst** | Analyze military units and command structures | Implemented |

## Multi-Agent Workflows

Agents share infrastructure for seamless collaboration:

```python
from shared import AuditLogger, EvidenceHandler
from agents.document_parser.agent import DocumentParserAgent
from agents.medical_forensic_analyst.agent import MedicalForensicAnalystAgent
from agents.evidence_gap_identifier.agent import EvidenceGapIdentifierAgent

# Shared infrastructure tracks evidence across all agents
shared_infra = {
    'audit_logger': AuditLogger(),
    'evidence_handler': EvidenceHandler()
}

# Initialize agents
parser = DocumentParserAgent(**shared_infra)
medical = MedicalForensicAnalystAgent(**shared_infra)
gap_finder = EvidenceGapIdentifierAgent(**shared_infra)

# Step 1: Parse medical document
parsed = parser.process({
    'file_path': 'medical_report.pdf',
    'case_id': 'CASE-001'
})

# Step 2: Analyze for torture indicators
analysis = medical.process({
    'record_text': parsed['extracted_text']['full_text'],
    'evidence_id': parsed['evidence_id'],
    'case_id': 'CASE-001'
})

# Step 3: Identify evidence gaps
gaps = gap_finder.process({
    'charges': ['torture'],
    'available_evidence': [analysis],
    'case_id': 'CASE-001'
})

# Complete audit trail maintained automatically
chain = shared_infra['audit_logger'].get_evidence_chain(parsed['evidence_id'])
```

## Architecture

### Shared Infrastructure

All agents use common components from `agents/shared/`:

- **BaseAgent** - Core functionality for all agents
- **VisionCapableAgent** - Extends BaseAgent for PDF/image processing
- **AuditLogger** - Blockchain-style immutable audit trails
- **EvidenceHandler** - Evidence ingestion and integrity verification
- **OutputFormatter** - Standardized legal report templates

### Agent Structure

```
agents/{agent-name}/
├── agent.py           # Main agent implementation
├── system_prompt.py   # Agent's specialized instructions
├── config.py         # Configuration settings
└── README.md         # Agent-specific documentation
```

## Testing

```bash
# Run all tests
pytest

# Run integration tests
pytest tests/integration/ -v

# Run agent-specific tests
pytest agents/document-parser/tests/ -v

# Run with coverage
pytest --cov=agents --cov-report=html
```

## Use Cases

### Human Rights Investigations
- Process witness statements at scale
- Document torture with Istanbul Protocol standards
- Build evidence chains for atrocity prosecutions

### International Criminal Justice
- Analyze evidence for ICC/hybrid tribunal cases
- Track command responsibility across military structures
- Generate court-ready documentation

### Civil Rights Litigation
- Process large document sets for pattern litigation
- Identify systemic evidence across cases
- Track policy document modifications

### Public Defenders & Legal Aid
- Quick case assessment and gap analysis
- Automated document processing for high caseloads
- Interview question generation

## Requirements

- Python 3.9 or higher
- Anthropic API key (Claude Sonnet 4.5 access)
- See `requirements.txt` for full dependencies

## Documentation

- **[Agent Specifications](Agents.md)** - Detailed specifications for all 18 agents
- **[CLAUDE.md](CLAUDE.md)** - Developer guide for working with this codebase
- Individual agent READMEs in each agent directory

## Project Principles

1. **Accuracy over novelty** - Evidentiary standards beat innovation
2. **Human-in-the-loop by default** - AI augments, doesn't replace investigators
3. **Verifiability and auditability** - Legal requirements first
4. **Open models, open methods** - Transparency and reproducibility
5. **Repeatable pipelines** - Consistent, reliable processing

## Contributing

Contributions are welcome! Areas of particular need:
- Additional agent implementations
- Improved test coverage
- Documentation improvements
- Integration examples
- Performance optimizations

## License

MIT License - see LICENSE file for details

## Citation

If you use LemkinAI in research or legal work, please cite:

```
LemkinAI Anthropic Claude Agent SDK (2024)
https://github.com/[your-organization]/LemkinAI-Anthropic-Claude-SDK
```

## Support

- **Issues**: [GitHub Issues](https://github.com/[your-organization]/LemkinAI-Anthropic-Claude-SDK/issues)
- **Documentation**: See individual agent READMEs and CLAUDE.md
- **API Reference**: https://docs.anthropic.com/claude/reference

---

**LemkinAI**: Turning messy evidence into court-ready insight.

Built for human rights investigators, public defenders, and justice advocates worldwide.
