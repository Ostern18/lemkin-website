# Enforced Disappearance Investigator (OpenAI Agents SDK)

Investigates enforced disappearances.

## Usage

```python
from agents.disappearance-investigator.agent import DisappearanceInvestigatorAgent
from shared import AuditLogger, EvidenceHandler

# Initialize
audit_logger = AuditLogger()
evidence_handler = EvidenceHandler()

agent = DisappearanceInvestigatorAgent(
    audit_logger=audit_logger,
    evidence_handler=evidence_handler
)

# Process
result = agent.process({
    'message': 'Investigate this enforced disappearance case',
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
config = DisappearanceConfig(
    model="gpt-4o",
    temperature=0.2
)

agent = DisappearanceInvestigatorAgent(config=config)
```

## Multi-Agent Workflows

This agent can participate in handoff workflows with other LemkinAI agents.

---

**OpenAI Agents SDK Implementation** - Part of the LemkinAI multi-agent framework.
