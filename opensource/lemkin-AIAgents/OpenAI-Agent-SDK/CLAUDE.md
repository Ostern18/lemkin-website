# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with the OpenAI Agents SDK implementation.

## Project Overview

**LemkinAI OpenAI Agents SDK Implementation** - Multi-agent framework for human rights investigations using OpenAI's production-ready Agents SDK.

Built on OpenAI Agents SDK with GPT-4o, featuring built-in handoffs, tracing, and provider-agnostic architecture.

## Common Development Commands

### Environment Setup
```bash
pip install -r requirements.txt
export OPENAI_API_KEY='your-api-key-here'
```

### Testing
```bash
# Run all integration tests
pytest tests/integration/ -v

# Run specific test file
pytest tests/integration/test_multi_agent_workflow.py -v

# Run with coverage
pytest tests/integration/ --cov=agents --cov-report=html
```

### Code Quality
```bash
# Format code
black agents/ tests/ examples/

# Lint
flake8 agents/ tests/ examples/

# Type checking
mypy agents/

# Sort imports
isort agents/ tests/ examples/
```

## OpenAI Agents SDK Architecture

### Key Imports

```python
from agents import Agent, Runner, handoff  # OpenAI Agents SDK
from shared import LemkinAgent, AuditLogger, EvidenceHandler
```

### Agent Implementation Pattern

**Every agent follows this pattern:**

```python
# agents/my-agent/agent.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from shared import LemkinAgent, AuditLogger, EvidenceHandler
from .system_prompt import SYSTEM_PROMPT
from .config import DEFAULT_CONFIG


class MyAgentWrapper:
    """
    Wrapper class for the LemkinAgent.
    Maintains API compatibility with Claude SDK implementation.
    """

    def __init__(self, audit_logger=None, evidence_handler=None, **kwargs):
        self.audit_logger = audit_logger or AuditLogger()
        self.evidence_handler = evidence_handler or EvidenceHandler()

        # Create LemkinAgent (wraps OpenAI Agent)
        self.agent = LemkinAgent(
            agent_id="my_agent",
            name="My Agent",
            instructions=SYSTEM_PROMPT,
            model=DEFAULT_CONFIG.model,
            audit_logger=self.audit_logger,
            **kwargs
        )

    def process(self, input_data):
        """Main processing method."""
        result = self.agent.run(
            message=input_data.get('message'),
            context_variables=input_data,
            evidence_ids=input_data.get('evidence_ids', [])
        )
        return result
```

### Configuration Pattern

```python
# agents/my-agent/config.py
from dataclasses import dataclass

@dataclass
class MyAgentConfig:
    model: str = "gpt-4o"
    temperature: float = 0.2
    max_tokens: int = 4096

DEFAULT_CONFIG = MyAgentConfig()
```

## Key Differences from Claude SDK

1. **Agent Creation**:
   - Claude: Inherit from `BaseAgent` or `VisionCapableAgent`
   - OpenAI: Use `LemkinAgent` wrapper around `Agent`

2. **Running Agents**:
   - Claude: `agent.call_claude(messages=[...])`
   - OpenAI: `agent.run(message=..., context_variables=...)`

3. **Multi-Agent Workflows**:
   - Claude: Manual orchestration
   - OpenAI: Built-in handoffs

4. **PDF Processing**:
   - Claude: Native `process_pdf()` method
   - OpenAI: Requires pdf2image conversion

## Multi-Agent Handoffs

OpenAI Agents SDK provides built-in handoff support:

```python
# Create specialized agents
parser_agent = LemkinAgent(
    agent_id="document_parser",
    name="Document Parser",
    instructions="Parse documents..."
)

analyst_agent = LemkinAgent(
    agent_id="medical_analyst",
    name="Medical Analyst",
    instructions="Analyze medical records..."
)

# Create triage agent with handoffs
triage = LemkinAgent(
    agent_id="triage",
    name="Triage Agent",
    instructions="Route to appropriate specialist...",
    handoffs=[parser_agent.agent, analyst_agent.agent]
)

# Run - automatically hands off to correct agent
result = triage.run("I have a medical record to analyze")
print(f"Handled by: {result['final_agent']}")
```

## Evidentiary Compliance

All agents maintain chain-of-custody:

```python
# Every operation logged
result = agent.run(message="...", evidence_ids=[evidence_id])

# Retrieve chain of custody
chain = agent.get_chain_of_custody(evidence_id)

# Verify integrity
integrity = agent.verify_integrity()
assert integrity == True
```

## Function Tools

Add custom tools to agents:

```python
def verify_hash(evidence_id: str) -> dict:
    """Verify evidence integrity."""
    return {"verified": True}

def search_database(query: str) -> list:
    """Search evidence database."""
    return [...]

agent = LemkinAgent(
    agent_id="investigator",
    name="Investigator",
    instructions="...",
    tools=[verify_hash, search_database]
)
```

## Testing Pattern

```python
@pytest.fixture
def shared_infrastructure():
    return {
        'audit_logger': AuditLogger(),
        'evidence_handler': EvidenceHandler()
    }

def test_agent_workflow(shared_infrastructure):
    agent = MyAgent(**shared_infrastructure)

    result = agent.process({
        'message': 'Test input',
        'evidence_ids': [evidence_id]
    })

    assert result['final_output'] is not None

    # Verify audit trail
    integrity = shared_infrastructure['audit_logger'].verify_chain_integrity()
    assert integrity == True
```

## Important Notes

1. **Provider-Agnostic**: Can use OpenAI, Azure OpenAI, or 100+ other LLMs
2. **Automatic Sessions**: Conversation history maintained automatically
3. **Built-in Tracing**: Debug and monitor agent interactions
4. **Guardrails**: Add input/output validation as needed
5. **Model Configuration**: Use `gpt-4o` by default, configurable per agent

## Resources

- **OpenAI Agents SDK Docs**: https://openai.github.io/openai-agents-python/
- **Main README**: ../README.md
- **OpenAI SDK README**: ./README.md
- **Claude SDK for comparison**: ../Anthropic-Claude-Agent-SDK/

## Agent Implementation Checklist

When creating a new agent:
- [ ] Create `agent.py` with wrapper class
- [ ] Create `system_prompt.py` with instructions
- [ ] Create `config.py` with model settings
- [ ] Use `LemkinAgent` wrapper (not direct `Agent`)
- [ ] Accept `audit_logger` and `evidence_handler` in `__init__`
- [ ] Implement `process()` method calling `agent.run()`
- [ ] Add to integration tests
- [ ] Document in README.md

---

**Built with OpenAI Agents SDK** - Production-ready multi-agent framework with evidentiary compliance.
