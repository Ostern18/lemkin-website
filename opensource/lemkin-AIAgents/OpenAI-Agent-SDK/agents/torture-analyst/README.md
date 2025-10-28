# Torture & Ill-Treatment Analyst (OpenAI Agents SDK)

Analyzes evidence for torture and ill-treatment.

## Usage

```python
from agents.torture-analyst.agent import TortureAnalystAgent
from shared import AuditLogger, EvidenceHandler

# Initialize
audit_logger = AuditLogger()
evidence_handler = EvidenceHandler()

agent = TortureAnalystAgent(
    audit_logger=audit_logger,
    evidence_handler=evidence_handler
)

# Process
result = agent.process({
    'message': 'Assess this evidence for Istanbul Protocol indicators',
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
config = TortureConfig(
    model="gpt-4o",
    temperature=0.2
)

agent = TortureAnalystAgent(config=config)
```

## Multi-Agent Workflows

This agent can participate in handoff workflows with other LemkinAI agents.

---

**OpenAI Agents SDK Implementation** - Part of the LemkinAI multi-agent framework.
