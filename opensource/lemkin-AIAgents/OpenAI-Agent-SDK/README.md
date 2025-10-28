# LemkinAI OpenAI Agents SDK

**Production-ready AI agents for human rights investigations using OpenAI's multi-agent framework**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![OpenAI Agents SDK](https://img.shields.io/badge/OpenAI-Agents%20SDK-00A67E)](https://openai.github.io/openai-agents-python/)

## What LemkinAI Enables

LemkinAI provides AI-powered automation for the time-consuming, labor-intensive work of processing evidence in human rights cases, criminal investigations, and legal proceedings. The agents handle document parsing, pattern detection across large datasets, medical record interpretation, satellite imagery analysis, and evidence gap identification—tasks that traditionally require weeks of manual review. Each agent maintains cryptographic audit trails and chain-of-custody logs, ensuring outputs meet evidentiary standards for legal proceedings while accelerating investigations from months to days.

**Key capabilities:** Parse multilingual documents at scale, identify torture indicators in medical records using Istanbul Protocol standards, analyze satellite imagery for mass graves or detention facilities, cross-reference thousands of witness statements for patterns, and automatically flag missing evidence against legal charge elements.

## Overview

LemkinAI provides specialized AI agents built on the **OpenAI Agents SDK** for processing documents, analyzing evidence, and supporting legal investigations. This implementation leverages OpenAI's production-ready multi-agent framework with:

- **Built-in Multi-Agent Handoffs** - Agents automatically delegate to specialists
- **Provider-Agnostic Architecture** - Use OpenAI, Azure OpenAI, or 100+ other LLMs
- **Automatic Session Management** - Conversation history maintained automatically
- **Built-in Tracing** - Debug and monitor agent interactions
- **Evidentiary Compliance** - Chain-of-custody tracking and audit trails

## Why OpenAI Agents SDK?

The OpenAI Agents SDK is a lightweight, powerful framework for building multi-agent workflows:

| Feature | Benefit |
|---------|---------|
| **Multi-Agent Handoffs** | Agents can delegate to specialists automatically |
| **Provider-Agnostic** | Switch between OpenAI, Azure, local models seamlessly |
| **Session Management** | Conversation state handled automatically |
| **Built-in Tracing** | Full visibility into agent interactions |
| **Function Tools** | Easy integration of custom capabilities |
| **Production-Ready** | Battle-tested framework from OpenAI |

## Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd OpenAI-Agent-SDK

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set your OpenAI API key
export OPENAI_API_KEY='your-api-key-here'
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
    'message': 'Parse this document and extract key information',
    'file_data': pdf_bytes,
    'source': 'Hospital Records',
    'case_id': 'CASE-2024-001'
})

print(f"Analysis: {result['final_output']}")
print(f"Evidence ID: {result.get('evidence_id')}")

# Verify chain of custody
integrity = audit_logger.verify_chain_integrity()
print(f"Audit trail verified: {integrity}")
```

## Multi-Agent Workflows with Handoffs

One of the key features is automatic agent handoffs:

```python
from shared import LemkinAgent, AuditLogger

# Shared infrastructure
audit_logger = AuditLogger()

# Create specialized agents
document_parser = LemkinAgent(
    agent_id="document_parser",
    name="Document Parser",
    instructions="Parse and extract content from documents...",
    audit_logger=audit_logger
)

medical_analyst = LemkinAgent(
    agent_id="medical_analyst",
    name="Medical Analyst",
    instructions="Analyze medical records for torture indicators...",
    audit_logger=audit_logger
)

legal_advisor = LemkinAgent(
    agent_id="legal_advisor",
    name="Legal Advisor",
    instructions="Provide legal analysis and jurisdictional guidance...",
    audit_logger=audit_logger
)

# Create triage agent with handoffs
triage = LemkinAgent(
    agent_id="triage",
    name="Investigation Triage",
    instructions="""
    Analyze the evidence and route to the appropriate specialist:
    - Document Parser: for PDFs, images, scanned documents
    - Medical Analyst: for medical records, autopsy reports
    - Legal Advisor: for legal questions, jurisdiction issues
    """,
    handoffs=[
        document_parser.agent,
        medical_analyst.agent,
        legal_advisor.agent
    ],
    audit_logger=audit_logger
)

# Run - automatically hands off to appropriate agent
result = triage.run(
    message="I have a medical report from a detention facility that needs analysis",
    evidence_ids=[evidence_id]
)

print(f"Handled by: {result['final_agent']}")
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

## Function Tools

Add custom tools to agents for specialized capabilities:

```python
from agents import Agent

def verify_document_hash(document_id: str) -> dict:
    """
    Verify document integrity using SHA-256 hash.

    Args:
        document_id: ID of document to verify

    Returns:
        dict: Verification results
    """
    # Implementation
    return {"verified": True, "hash": "..."}

def search_case_law(query: str, jurisdiction: str) -> list:
    """
    Search legal databases for relevant case law.

    Args:
        query: Search query
        jurisdiction: Legal jurisdiction

    Returns:
        list: Matching cases
    """
    # Implementation
    return [...]

# Create agent with custom tools
legal_agent = LemkinAgent(
    agent_id="legal_advisor",
    name="Legal Advisor",
    instructions="Provide legal analysis using case law database...",
    tools=[verify_document_hash, search_case_law]
)
```

## Architecture

### LemkinAgent Wrapper

We wrap OpenAI's `Agent` class with `LemkinAgent` to add evidentiary compliance:

```python
from agents import Agent  # OpenAI Agents SDK
from shared import LemkinAgent, AuditLogger, EvidenceHandler

# LemkinAgent wraps Agent with evidentiary features
agent = LemkinAgent(
    agent_id="document_parser",  # For audit logs
    name="Document Parser",      # For OpenAI SDK
    instructions=SYSTEM_PROMPT,  # Agent's role
    model="gpt-4o",
    audit_logger=audit_logger    # Chain-of-custody tracking
)

# Run with automatic history and tracing
result = agent.run(
    message="Parse this document...",
    evidence_ids=[evidence_id]
)
```

### Shared Infrastructure

All agents use common components from `agents/shared/`:

- **LemkinAgent** - Wraps OpenAI Agent with evidentiary compliance
- **AuditLogger** - Blockchain-style immutable audit trails
- **EvidenceHandler** - Evidence ingestion and integrity verification
- **OutputFormatter** - Standardized legal report templates

### Agent Structure

```
agents/{agent-name}/
├── agent.py           # Agent wrapper class
├── system_prompt.py   # Agent's specialized instructions
├── config.py         # Configuration settings
└── README.md         # Agent-specific documentation
```

## Key Differences from Claude SDK

| Feature | OpenAI Agents SDK | Claude SDK |
|---------|------------------|------------|
| **Framework** | OpenAI Agents SDK | Direct Anthropic API |
| **Agent Creation** | `LemkinAgent(name=..., instructions=...)` | `MyAgent(BaseAgent)` |
| **Running** | `agent.run(message=...)` | `agent.process(input_data=...)` |
| **Multi-Agent** | Built-in handoffs | Manual orchestration |
| **PDF Processing** | Requires conversion | Native support |
| **Context Window** | 128K (GPT-4o) | 200K (Claude Sonnet) |
| **Tracing** | Built-in | Manual logging |
| **Model Flexibility** | 100+ models | Claude only |
| **Session Management** | Automatic | Manual |

## Testing

```bash
# Run all tests
pytest

# Run integration tests
pytest tests/integration/ -v

# Run with coverage
pytest --cov=agents --cov-report=html
```

## Evidentiary Compliance

All agents maintain strict evidentiary standards:

```python
# Every operation is logged automatically
agent.run(message="...", evidence_ids=[evidence_id])
# → Logs: TOOL_EXECUTED, ANALYSIS_PERFORMED

# Retrieve complete chain of custody
chain = agent.get_chain_of_custody(evidence_id)

# Verify audit integrity
integrity = agent.verify_integrity()
assert integrity == True

# Get session summary
summary = agent.get_session_summary()
print(f"Total events: {summary['total_events']}")
print(f"Evidence items: {summary['unique_evidence_items']}")
```

## Use Cases

### Human Rights Investigations
- Process witness statements at scale
- Document torture with Istanbul Protocol standards
- Build evidence chains for atrocity prosecutions

### International Criminal Justice
- Analyze evidence for ICC/hybrid tribunal cases
- Multi-agent workflows for complex investigations
- Generate court-ready documentation

### Civil Rights Litigation
- Process large document sets for pattern litigation
- Automatic routing to specialist agents
- Track policy document modifications

### Public Defenders & Legal Aid
- Quick case assessment with automatic handoffs
- Automated document processing
- Evidence gap identification

## Requirements

- Python 3.9 or higher
- OpenAI API key (or compatible provider)
- See `requirements.txt` for full dependencies

## Documentation

- **[CLAUDE.md](CLAUDE.md)** - Developer guide for working with this codebase
- **[OpenAI Agents SDK Docs](https://openai.github.io/openai-agents-python/)** - Official SDK documentation
- Individual agent READMEs in each agent directory

## Project Principles

1. **Accuracy over novelty** - Evidentiary standards beat innovation
2. **Human-in-the-loop by default** - AI augments, doesn't replace investigators
3. **Verifiability and auditability** - Legal requirements first
4. **Open models, open methods** - Transparency and reproducibility
5. **Provider-agnostic** - No vendor lock-in

## Contributing

Contributions are welcome! Areas of particular need:
- Additional agent implementations
- Custom function tools
- Multi-agent workflow examples
- Integration tests
- Documentation improvements

## License

MIT License - see LICENSE file for details

## Citation

If you use LemkinAI in research or legal work, please cite:

```
LemkinAI OpenAI Agents SDK Implementation (2024)
https://github.com/[your-organization]/LemkinAI-OpenAI-Agent-SDK
```

## Support

- **Issues**: [GitHub Issues](https://github.com/[your-organization]/LemkinAI-OpenAI-Agent-SDK/issues)
- **Documentation**: See CLAUDE.md and individual agent READMEs
- **OpenAI SDK Reference**: https://openai.github.io/openai-agents-python/

## Related Projects

- **[LemkinAI Anthropic Claude SDK](../Anthropic-Claude-Agent-SDK/)** - Claude-based implementation with native PDF processing
- **Main LemkinAI Repository** - Complete documentation and specifications

---

**LemkinAI**: Turning messy evidence into court-ready insight.

Built for human rights investigators, public defenders, and justice advocates worldwide.

**Powered by OpenAI Agents SDK** - Production-ready multi-agent framework with evidentiary compliance.
