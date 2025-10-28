# Medical & Forensic Record Analyst (OpenAI Agents SDK)

Interprets medical reports and forensic documentation.

## Usage

```python
from agents.medical-forensic-analyst.agent import MedicalForensicAnalystAgent
from shared import AuditLogger, EvidenceHandler

# Initialize
audit_logger = AuditLogger()
evidence_handler = EvidenceHandler()

agent = MedicalForensicAnalystAgent(
    audit_logger=audit_logger,
    evidence_handler=evidence_handler
)

# Process
result = agent.process({
    'message': 'Analyze this medical record for torture indicators',
    'case_id': 'CASE-001',
    'evidence_ids': [evidence_id]
})

print(result['analysis'])

# Verify audit trail
integrity = audit_logger.verify_chain_integrity()
```

## Architecture

Built on OpenAI Agents SDK with `LemkinAgent` wrapper for evidentiary compliance.

## Configuration

Customize in `config.py`:

```python
config = MedicalConfig(
    model="gpt-4o",
    temperature=0.2
)

agent = MedicalForensicAnalystAgent(config=config)
```

## Multi-Agent Workflows

This agent can participate in handoff workflows with other LemkinAI agents.

---

**OpenAI Agents SDK Implementation** - Part of the LemkinAI multi-agent framework.
