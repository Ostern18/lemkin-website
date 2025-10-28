# Genocide Intent Analyzer (OpenAI Agents SDK)

Analyzes evidence for genocide intent.

## Usage

```python
from agents.genocide-intent-analyzer.agent import GenocideIntentAnalyzerAgent
from shared import AuditLogger, EvidenceHandler

# Initialize
audit_logger = AuditLogger()
evidence_handler = EvidenceHandler()

agent = GenocideIntentAnalyzerAgent(
    audit_logger=audit_logger,
    evidence_handler=evidence_handler
)

# Process
result = agent.process({
    'message': 'Analyze these statements for genocidal intent',
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
config = GenocideConfig(
    model="gpt-4o",
    temperature=0.2
)

agent = GenocideIntentAnalyzerAgent(config=config)
```

## Multi-Agent Workflows

This agent can participate in handoff workflows with other LemkinAI agents.

---

**OpenAI Agents SDK Implementation** - Part of the LemkinAI multi-agent framework.
