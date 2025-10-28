# OSINT Synthesis Agent (OpenAI Agents SDK)

**Example implementation demonstrating the OpenAI Agents SDK pattern for LemkinAI.**

## Overview

This agent demonstrates how to implement a LemkinAI agent using the OpenAI Agents SDK framework with full evidentiary compliance.

## Features

- Built on OpenAI Agents SDK with LemkinAgent wrapper
- Automatic audit logging and chain-of-custody tracking
- Human-in-the-loop gates for low-credibility findings
- Structured JSON output
- Evidence integrity verification

## Usage

```python
from agents.osint_synthesis.agent import OSINTSynthesisAgent
from shared import AuditLogger, EvidenceHandler

# Initialize
audit_logger = AuditLogger()
evidence_handler = EvidenceHandler()

agent = OSINTSynthesisAgent(
    audit_logger=audit_logger,
    evidence_handler=evidence_handler
)

# Verify a claim
result = agent.verify_claim(
    claim="Torture occurred at detention facility",
    sources=[
        {'url': 'https://...', 'content': '...'},
        {'url': 'https://...', 'content': '...'}
    ],
    case_id='CASE-001'
)

print(result['analysis'])

# Verify audit trail
integrity = audit_logger.verify_chain_integrity()
print(f"Audit chain verified: {integrity}")
```

## Architecture

### LemkinAgent Wrapper

The agent uses `LemkinAgent` which wraps OpenAI's `Agent` class:

```python
self.agent = LemkinAgent(
    agent_id="osint_synthesis",  # For audit logs
    name="OSINT Synthesis Agent",  # For OpenAI SDK
    instructions=SYSTEM_PROMPT,  # Agent instructions
    model="gpt-4o",
    audit_logger=audit_logger
)
```

### Running the Agent

```python
result = self.agent.run(
    message="Verify this claim...",
    context_variables={
        "sources": [...],
        "case_id": "CASE-001"
    },
    evidence_ids=[evidence_id]
)
```

### Audit Logging

Every operation is automatically logged:
- Agent initialization
- Processing start
- Analysis performed
- Human review requests
- Output generation

## Configuration

Customize behavior in `config.py`:

```python
config = OSINTConfig(
    model="gpt-4o",
    temperature=0.2,
    high_credibility_threshold=0.8,
    minimum_sources_for_verification=2
)

agent = OSINTSynthesisAgent(config=config)
```

## Multi-Agent Workflows

This agent can participate in handoff workflows:

```python
from shared import LemkinAgent
from agents.osint_synthesis.agent import OSINTSynthesisAgent
from agents.legal_advisor.agent import LegalAdvisorAgent

# Create agents
osint = OSINTSynthesisAgent()
legal = LegalAdvisorAgent()

# Create triage agent with handoffs
triage = LemkinAgent(
    agent_id="triage",
    name="Investigation Triage",
    instructions="Route to appropriate specialist...",
    handoffs=[osint.agent.agent, legal.agent.agent]
)

# Automatically hands off to correct agent
result = triage.run("Verify this torture claim...")
```

## Pattern for Other Agents

This serves as a template for implementing all 18 LemkinAI agents:

1. **system_prompt.py**: Define agent's role and capabilities
2. **config.py**: Configure model and thresholds
3. **agent.py**:
   - Create wrapper class
   - Initialize `LemkinAgent` in `__init__`
   - Implement `process()` method
   - Use `agent.run()` to execute

## Testing

```python
def test_osint_agent(shared_infrastructure):
    agent = OSINTSynthesisAgent(**shared_infrastructure)

    result = agent.verify_claim(
        claim="Test claim",
        sources=[{'url': 'test', 'content': 'test'}],
        case_id='TEST-001'
    )

    assert result['analysis'] is not None
    assert shared_infrastructure['audit_logger'].verify_chain_integrity()
```

## Next Steps

To implement other agents, follow this pattern:
1. Copy this directory structure
2. Customize `system_prompt.py` for agent's specialty
3. Update `config.py` with appropriate thresholds
4. Implement `agent.py` following the same wrapper pattern
5. Add specialized methods as needed

---

**Example Agent** - Demonstrates OpenAI Agents SDK integration with LemkinAI evidentiary compliance.
