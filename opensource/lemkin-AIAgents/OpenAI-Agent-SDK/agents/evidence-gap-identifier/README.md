# Evidence Gap & Next Steps Identifier (OpenAI Agents SDK)

Analyzes investigations to identify missing evidence.

## Usage

```python
from agents.evidence-gap-identifier.agent import EvidenceGapIdentifierAgent
from shared import AuditLogger, EvidenceHandler

# Initialize
audit_logger = AuditLogger()
evidence_handler = EvidenceHandler()

agent = EvidenceGapIdentifierAgent(
    audit_logger=audit_logger,
    evidence_handler=evidence_handler
)

# Process
result = agent.process({
    'message': 'Identify evidence gaps for torture charges',
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
config = GapConfig(
    model="gpt-4o",
    temperature=0.2
)

agent = EvidenceGapIdentifierAgent(config=config)
```

## Multi-Agent Workflows

This agent can participate in handoff workflows with other LemkinAI agents.

---

**OpenAI Agents SDK Implementation** - Part of the LemkinAI multi-agent framework.
