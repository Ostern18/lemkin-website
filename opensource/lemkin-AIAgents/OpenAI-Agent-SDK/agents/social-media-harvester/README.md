# Social Media Evidence Harvester (OpenAI Agents SDK)

Analyzes social media screenshots and posts for evidence collection.

## Usage

```python
from agents.social-media-harvester.agent import SocialMediaHarvesterAgent
from shared import AuditLogger, EvidenceHandler

# Initialize
audit_logger = AuditLogger()
evidence_handler = EvidenceHandler()

agent = SocialMediaHarvesterAgent(
    audit_logger=audit_logger,
    evidence_handler=evidence_handler
)

# Process
result = agent.process({
    'message': 'Extract metadata and assess authenticity of this social media post',
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
config = HarvesterConfig(
    model="gpt-4o",
    temperature=0.2
)

agent = SocialMediaHarvesterAgent(config=config)
```

## Multi-Agent Workflows

This agent can participate in handoff workflows with other LemkinAI agents.

---

**OpenAI Agents SDK Implementation** - Part of the LemkinAI multi-agent framework.
