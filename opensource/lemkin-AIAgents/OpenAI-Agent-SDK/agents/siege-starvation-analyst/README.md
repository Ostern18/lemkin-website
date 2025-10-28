# Siege & Starvation Warfare Analyst (OpenAI Agents SDK)

Analyzes siege and starvation warfare tactics.

## Usage

```python
from agents.siege-starvation-analyst.agent import SiegeStarvationAnalystAgent
from shared import AuditLogger, EvidenceHandler

# Initialize
audit_logger = AuditLogger()
evidence_handler = EvidenceHandler()

agent = SiegeStarvationAnalystAgent(
    audit_logger=audit_logger,
    evidence_handler=evidence_handler
)

# Process
result = agent.process({
    'message': 'Analyze supply flow restrictions and humanitarian impact',
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
config = SiegeConfig(
    model="gpt-4o",
    temperature=0.2
)

agent = SiegeStarvationAnalystAgent(config=config)
```

## Multi-Agent Workflows

This agent can participate in handoff workflows with other LemkinAI agents.

---

**OpenAI Agents SDK Implementation** - Part of the LemkinAI multi-agent framework.
