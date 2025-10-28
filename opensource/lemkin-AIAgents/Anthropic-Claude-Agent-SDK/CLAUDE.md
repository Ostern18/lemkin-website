# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**LemkinAI** - AI agents for human rights investigations and legal documentation. Turns messy evidence into court-ready insight with strict evidentiary standards (chain-of-custody, audit trails, human-in-the-loop gates).

Built on Claude Sonnet 4.5 with Vision API for PDF/image processing, extended context (200K tokens) for multi-document analysis.

## Common Development Commands

### Environment Setup
```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY='your-api-key-here'
```

### Testing
```bash
# Run all integration tests
pytest tests/integration/ -v

# Run specific test file
pytest tests/integration/test_document_processing_workflow.py -v

# Run single test
pytest tests/integration/test_document_processing_workflow.py::TestDocumentProcessingWorkflow::test_medical_record_analysis -v

# Run with coverage
pytest tests/integration/ --cov=agents --cov-report=html
```

### Code Quality
```bash
# Format code (run before committing)
black agents/ tests/ examples/

# Lint
flake8 agents/ tests/ examples/

# Type checking
mypy agents/

# Sort imports
isort agents/ tests/ examples/
```

### Run Example Workflows
```bash
# Multi-domain investigation workflow
python examples/workflows/complete_investigation_workflow.py

# Document processing only
python examples/workflows/torture_investigation_workflow.py
```

## Repository Structure

### Python Agents (Primary Codebase)
**18 production-ready agents** organized in 3 domains:
- **6 committed agents**: osint-synthesis, satellite-imagery-analyst, document-parser, comparative-analyzer, medical-forensic-analyst, evidence-gap-identifier
- **12 uncommitted agents**: social-media-harvester, historical-researcher, legal-advisor, torture-analyst, genocide-intent-analyzer, disappearance-investigator, ngo-un-reporter, digital-forensics-analyst, siege-starvation-analyst, forensic-analysis-reviewer, ballistics-weapons-identifier, military-structure-analyst
- **Shared infrastructure**: BaseAgent, VisionCapableAgent, AuditLogger, EvidenceHandler, OutputFormatter
- **Testing**: 2 integration test files, 2 workflow examples

### Node.js/Remotion Component
The `my-video/` directory and root `package.json` contain a Remotion video generation setup (separate from Python agents). This is for programmatic video creation but is not integrated with the core agent system.

**See PROJECT_STATUS.md for detailed agent status and README.md for comprehensive documentation.**

## Code Architecture

### Shared Infrastructure Pattern

All agents share common components from `agents/shared/`:

**BaseAgent** (`base_agent.py`)
- Abstract base class that all agents inherit from
- Provides Claude API client, audit logging, human-in-the-loop gates
- Initialization: `super().__init__(agent_id="...", system_prompt=SYSTEM_PROMPT, ...)`
- Key methods: `process()`, `call_claude()`, `request_human_review()`

**VisionCapableAgent** (extends BaseAgent)
- For agents processing images/PDFs (satellite imagery, document parser, etc.)
- Additional methods: `process_image()`, `process_pdf()`
- Handles base64 encoding and proper message formatting for Claude Vision API

**AuditLogger** (`audit_logger.py`)
- Blockchain-style immutable audit trail (each event hashes previous event)
- Log operations: `log_event()` with AuditEventType enum
- Verify integrity: `verify_chain_integrity()` - checks all hash chains
- Export: `get_session_summary()`, `get_evidence_chain()` for legal documentation

**EvidenceHandler** (`evidence_handler.py`)
- Evidence ingestion with SHA-256 integrity verification
- Track evidence through multi-agent workflows
- Methods: `ingest_evidence()`, `get_evidence()`, `get_metadata()`, `verify_integrity()`
- Supports multiple evidence types via EvidenceType enum

**OutputFormatter** (`output_formatter.py`)
- Standardized legal report templates
- Formats: executive summaries, detailed findings, evidence chains, recommendations
- Ensures consistent output structure across all agents

### Agent Directory Structure

```
agents/{agent-name}/
├── agent.py           # Main agent class (extends BaseAgent or VisionCapableAgent)
├── system_prompt.py   # Agent's specialized instructions for Claude
├── config.py         # Settings (model, temperature, thresholds)
└── README.md         # Agent-specific documentation
```

### Creating a New Agent

**Minimum viable agent implementation:**

```python
# agents/new-agent/agent.py
from shared import BaseAgent, AuditLogger
from .system_prompt import SYSTEM_PROMPT
from .config import DEFAULT_CONFIG

class NewAgent(BaseAgent):
    def __init__(self, audit_logger=None, evidence_handler=None, **kwargs):
        super().__init__(
            agent_id="new_agent",
            system_prompt=SYSTEM_PROMPT,
            model=DEFAULT_CONFIG.model,
            audit_logger=audit_logger,
            **kwargs
        )
        self.evidence_handler = evidence_handler or EvidenceHandler()

    def process(self, input_data: dict) -> dict:
        """Main processing method - customize per agent."""
        self.audit_logger.log_event(
            event_type=AuditEventType.ANALYSIS_PERFORMED,
            agent_id=self.agent_id,
            evidence_ids=input_data.get('evidence_ids', [])
        )

        result = self.call_claude(
            messages=[{"role": "user", "content": self._prepare_input(input_data)}]
        )

        return self._format_output(result)
```

**For vision-capable agents (images/PDFs):**
- Extend `VisionCapableAgent` instead of `BaseAgent`
- Use `self.process_image()` or `self.process_pdf()` methods
- See `agents/satellite-imagery-analyst/agent.py` or `agents/document-parser/agent.py` for examples

**Required files:**
1. `system_prompt.py`: Define agent's specialized instructions
2. `config.py`: Set model parameters (use Sonnet 4.5, temp 0.2)
3. `README.md`: Document capabilities and usage examples

**Integration checklist:**
- Inherit from BaseAgent or VisionCapableAgent
- Accept `audit_logger` and `evidence_handler` in `__init__`
- Use `audit_logger.log_event()` for all evidence operations
- Call `self.call_claude()` (not `self.client.messages.create()`) for automatic logging
- Add to integration tests showing multi-agent collaboration
- Update README.md and PROJECT_STATUS.md

### Multi-Agent Workflow Pattern

Agents share infrastructure for cross-agent evidence tracking:

```python
from shared import AuditLogger, EvidenceHandler
from agents.document_parser.agent import DocumentParserAgent
from agents.medical_forensic_analyst.agent import MedicalForensicAnalystAgent

# Shared infrastructure
audit_logger = AuditLogger()
evidence_handler = EvidenceHandler()
shared_infra = {
    'audit_logger': audit_logger,
    'evidence_handler': evidence_handler
}

# Initialize agents with shared infrastructure
parser = DocumentParserAgent(**shared_infra)
analyst = MedicalForensicAnalystAgent(**shared_infra)

# Process through pipeline - evidence tracked automatically
parsed = parser.process({
    'file_data': file_bytes,
    'file_type': 'pdf',
    'source': 'Hospital',
    'case_id': 'CASE-001'
})

analysis = analyst.process({
    'record_text': parsed['extracted_text']['full_text'],
    'evidence_id': parsed['evidence_id'],
    'case_id': 'CASE-001'
})

# Verify complete chain of custody
chain = audit_logger.get_evidence_chain(parsed['evidence_id'])
integrity = audit_logger.verify_chain_integrity()
```

### Key Implementation Notes

**Evidentiary requirements:**
- ALL evidence operations MUST be logged via AuditLogger
- Evidence integrity verified with SHA-256 hashing
- Human review gates for high-confidence thresholds (configurable in config.py)
- Source provenance maintained through entire pipeline

**Model configuration:**
- Use `claude-sonnet-4-5-20250929` (Sonnet 4.5)
- Temperature: 0.2 for focused, consistent output
- Max tokens: 4096 for detailed analysis
- Extended context (200K) enables multi-document analysis

**Import pattern for agents:**
```python
import sys
from pathlib import Path
# Add parent directory to path for imports (agents/ is not a package)
sys.path.insert(0, str(Path(__file__).parent.parent))
from shared import BaseAgent, VisionCapableAgent, AuditLogger, EvidenceHandler, OutputFormatter
from .system_prompt import SYSTEM_PROMPT
from .config import DEFAULT_CONFIG
```

**Import pattern for tests/examples:**
```python
import sys
from pathlib import Path
# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from agents.document_parser.agent import DocumentParserAgent
from shared import AuditLogger, EvidenceHandler
```

**Testing pattern:**
All integration tests use shared infrastructure fixture:
```python
@pytest.fixture
def shared_infrastructure(self):
    return {
        'audit_logger': AuditLogger(),
        'evidence_handler': EvidenceHandler()
    }
```

## Agent Categories

### Domain 1: Investigative Research & Intelligence (5 agents)
- OSINT Synthesis Agent (committed)
- Satellite Imagery Analyst (committed)
- Social Media Evidence Harvester (uncommitted)
- Historical Context & Background Researcher (uncommitted)
- Legal Framework & Jurisdiction Advisor (uncommitted)

### Domain 2: Document Processing & Analysis (4 agents)
- Multi-Format Document Parser (committed)
- Comparative Document Analyzer (committed)
- Medical & Forensic Record Analyst (committed)
- Evidence Gap & Next Steps Identifier (committed)

### Domain 3: Crime-Specific & Specialized Analysis (9 agents)
All uncommitted:
- Torture & Ill-Treatment Analyst
- Genocide Intent Analyzer
- Enforced Disappearance Investigator
- Siege & Starvation Warfare Analyst
- Digital Forensics & Metadata Analyst
- Forensic Analysis Reviewer
- Ballistics & Weapons Identifier
- Military Structure & Tactics Analyst
- NGO & UN Reporting Specialist
