# Ballistics & Weapons Identifier (OpenAI Agents SDK)

Identifies weapons and analyzes ballistics evidence.

## Usage

```python
from agents.ballistics-weapons-identifier.agent import BallisticsWeaponsIdentifierAgent
from shared import AuditLogger, EvidenceHandler

# Initialize
audit_logger = AuditLogger()
evidence_handler = EvidenceHandler()

agent = BallisticsWeaponsIdentifierAgent(
    audit_logger=audit_logger,
    evidence_handler=evidence_handler
)

# Process
result = agent.process({
    'message': 'Identify the weapon type from this ballistics evidence',
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
config = BallisticsConfig(
    model="gpt-4o",
    temperature=0.2
)

agent = BallisticsWeaponsIdentifierAgent(config=config)
```

## Multi-Agent Workflows

This agent can participate in handoff workflows with other LemkinAI agents.

---

**OpenAI Agents SDK Implementation** - Part of the LemkinAI multi-agent framework.
