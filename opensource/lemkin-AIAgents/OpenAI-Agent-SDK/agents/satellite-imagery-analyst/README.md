# Satellite Imagery Analyst (OpenAI Agents SDK)

Interprets satellite and aerial imagery for evidence documentation.

## Usage

```python
from agents.satellite-imagery-analyst.agent import SatelliteImageryAnalystAgent
from shared import AuditLogger, EvidenceHandler

# Initialize
audit_logger = AuditLogger()
evidence_handler = EvidenceHandler()

agent = SatelliteImageryAnalystAgent(
    audit_logger=audit_logger,
    evidence_handler=evidence_handler
)

# Process
result = agent.process({
    'message': 'Analyze this satellite image for infrastructure damage',
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
config = ImageryConfig(
    model="gpt-4o",
    temperature=0.2
)

agent = SatelliteImageryAnalystAgent(config=config)
```

## Multi-Agent Workflows

This agent can participate in handoff workflows with other LemkinAI agents.

---

**OpenAI Agents SDK Implementation** - Part of the LemkinAI multi-agent framework.
