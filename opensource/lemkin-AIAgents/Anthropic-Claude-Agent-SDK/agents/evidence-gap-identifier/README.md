# Evidence Gap & Next Steps Identifier Agent

Analyzes investigations to identify missing evidence and recommend concrete next steps.

## Purpose

Strategic planning agent that ensures investigations don't miss critical evidence. Compares available evidence to legal requirements, prioritizes investigative actions, and suggests alternative approaches.

## Capabilities

- **Gap Analysis**: Compare evidence to legal elements for each charge
- **Action Prioritization**: Recommend specific, prioritized next steps
- **Interview Planning**: Generate follow-up questions for witnesses
- **Document Requests**: Identify needed documents and how to obtain them
- **Expert Consultation**: Determine what expert help is needed
- **Alternative Strategies**: Suggest backup approaches if direct evidence unavailable
- **Risk Assessment**: Flag time-sensitive evidence and availability risks

## Usage

### Basic Gap Analysis

```python
from agents.evidence_gap_identifier import EvidenceGapIdentifierAgent

agent = EvidenceGapIdentifierAgent()

result = agent.process({
    'charges': ['torture', 'war_crimes'],
    'available_evidence': [
        "Medical report showing injuries",
        "Witness statement from victim",
        "Photos of detention facility"
    ],
    'case_theory': "Systematic torture at military detention center",
    'case_id': 'CASE-2024-001'
})

# View critical gaps
print(result['gap_summary'])
print(result['critical_next_steps'])

# View detailed recommendations
for action in result['priority_actions']:
    print(f"{action['priority']}: {action['description']}")
```

### Quick Gap Check

```python
# Quick assessment for single charge
result = agent.quick_gap_check(
    charge="murder",
    available_evidence_summary="Autopsy report, circumstantial evidence, no eyewitnesses",
    case_id="CASE-2024-002"
)

print(result['evidence_gaps'])
print(result['priority_actions'][:3])  # Top 3 actions
```

### Interview Planning

```python
# Generate follow-up questions
result = agent.process({
    'charges': ['genocide'],
    'available_evidence': {...},
    'witnesses': [{
        'witness_id': 'WITNESS-123',
        'prior_statement': 'Initial witness statement text...'
    }]
})

# Extract interview questions
interview_plan = result['witness_interview_questions']
for witness in interview_plan['existing_witnesses']:
    print(f"Questions for {witness['witness_id']}:")
    for q in witness['follow_up_questions']:
        print(f"  - {q}")
```

## Output Format

Returns structured JSON with:
- Legal elements assessed (status, gaps, confidence)
- Evidence gaps (severity, impact, alternatives)
- Priority actions (specific steps, rationale, effort estimate)
- Witness interview questions
- Document requests
- Expert consultations needed
- Alternative strategies
- Timeline recommendations
- Resource requirements
- Risk assessment

## Configuration

```python
from agents.evidence_gap_identifier.config import GapIdentifierConfig

config = GapIdentifierConfig(
    max_priority_actions=5,  # Limit to top 5 actions
    focus_on_critical_gaps=True,
    generate_interview_questions=True,
    assess_risks=True
)

agent = EvidenceGapIdentifierAgent(config=config)
```

## Use Cases

- **Case Assessment**: Determine if you have enough evidence to prosecute
- **Investigation Planning**: Prioritize investigative resources
- **Witness Preparation**: Generate targeted follow-up questions
- **Legal Strategy**: Identify alternative charges if primary charges lack evidence
- **Resource Allocation**: Understand what expertise and resources are needed
- **Deadline Management**: Flag time-sensitive evidence collection needs
