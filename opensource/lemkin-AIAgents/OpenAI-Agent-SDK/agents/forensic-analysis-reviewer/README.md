# Forensic Analysis Reviewer (OpenAI Agents SDK)

Reviews forensic analysis reports.

## Usage

```python
from agents.forensic-analysis-reviewer.agent import ForensicAnalysisReviewerAgent
from shared import AuditLogger, EvidenceHandler

# Initialize
audit_logger = AuditLogger()
evidence_handler = EvidenceHandler()

agent = ForensicAnalysisReviewerAgent(
    audit_logger=audit_logger,
    evidence_handler=evidence_handler
)

# Process
result = agent.process({
    'message': 'Review this DNA analysis report for legal standards',
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
config = ReviewerConfig(
    model="gpt-4o",
    temperature=0.2
)

agent = ForensicAnalysisReviewerAgent(config=config)
```

## Multi-Agent Workflows

This agent can participate in handoff workflows with other LemkinAI agents.

---

**OpenAI Agents SDK Implementation** - Part of the LemkinAI multi-agent framework.
