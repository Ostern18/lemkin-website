# Military Structure & Tactics Analyst Agent

The Military Structure & Tactics Analyst Agent analyzes military organizational structures, command hierarchies, tactical operations, and military doctrine for legal investigations. It provides expert military analysis to support accountability efforts and IHL compliance assessment.

## Overview

This agent bridges military expertise and legal investigation by explaining military structures, mapping command relationships, analyzing tactical operations, and assessing IHL compliance. It supports command responsibility investigations and helps legal teams understand military operations.

## Core Capabilities

### Military Structure Analysis
- **Unit organization** - Hierarchy from strategic to tactical levels
- **Command structures** - Chain of command mapping
- **Unit types** - Infantry, armor, artillery, special forces identification
- **Rank structures** - Authority relationships and responsibilities
- **Force composition** - Capabilities and organization
- **Organizational changes** - Tracking structure evolution over time

### Command Hierarchy Mapping
- **Chain of command** - Strategic, operational, tactical level mapping
- **Commander identification** - Names, positions, authority scope
- **Command relationships** - Operational vs. administrative control
- **Command succession** - Changes in leadership over time
- **Parallel structures** - Intelligence, political, logistical command
- **Effective control** - Actual vs. formal authority assessment

### Tactical Operations Analysis
- **Tactic explanation** - Military tactics and their objectives
- **Attack patterns** - Operational methods and patterns
- **Decision-making** - Tactical decision analysis
- **Unit coordination** - Multi-unit operation coordination
- **Operational indicators** - Evidence of specific operation types
- **Standard procedures** - SOP identification and application

### Military Doctrine Assessment
- **Doctrine explanation** - Relevant military doctrine and training
- **Doctrinal influence** - How doctrine shapes operations
- **Deviations** - Non-standard practices and their significance
- **Institutional practices** - Systematic patterns across units
- **Training indicators** - Evidence of formal training
- **Doctrinal knowledge** - Commander understanding assessment

### IHL Compliance Analysis
- **Military necessity** - Assessment of operational necessity
- **Proportionality** - Proportionality of attacks evaluation
- **Precautions** - Analysis of civilian protection measures
- **Distinction** - Civilian vs. military target distinction
- **Prohibited methods** - Identification of IHL violations
- **Geneva Conventions** - Compliance assessment

### Command Responsibility Assessment
- **Responsibility mapping** - Linking operations to commanders
- **Knowledge assessment** - What commanders knew or should have known
- **Orders analysis** - Documentation of orders and policies
- **Failure to prevent** - Assessment of prevention failures
- **Failure to punish** - Investigation and punishment failures
- **Effective control** - Authority and control documentation

## Configuration Options

**Default**: Comprehensive military analysis across all domains

**Command Responsibility** (`COMMAND_RESPONSIBILITY_CONFIG`):
- Focus on command hierarchies and responsibility
- Commander knowledge assessment
- Higher confidence threshold (0.7)
- Extended analysis (16,000 tokens)

**Tactical Analysis** (`TACTICAL_ANALYSIS_CONFIG`):
- Deep tactical operations analysis
- Military doctrine assessment
- IHL compliance focus
- Maximum precision (temperature 0.05)

## Usage Examples

### Basic Military Structure Analysis

```python
from agents.military_structure_analyst.agent import MilitaryStructureAnalystAgent

agent = MilitaryStructureAnalystAgent()

result = agent.process({
    'military_force': '5th Army Corps',
    'organizational_data': {
        'units': [
            {'designation': '1st Brigade', 'type': 'infantry', 'strength': 3000},
            {'designation': '2nd Brigade', 'type': 'armor', 'strength': 2500}
        ],
        'area_of_operations': 'Eastern Region'
    },
    'command_data': {
        'overall_commander': 'General John Smith',
        'subordinate_commanders': [...]
    },
    'case_id': 'CASE-2024-001'
})

print(f"Force structure: {result['unit_structure_analysis']}")
print(f"Command responsibility: {result['command_responsibility_analysis']}")
```

### Command Responsibility Analysis

```python
result = agent.analyze_command_structure(
    military_force='Special Operations Brigade',
    command_data={
        'commanders': [
            {'name': 'Colonel Smith', 'position': 'Brigade Commander', 'authority': 'operational'},
            {'name': 'Major Jones', 'position': 'Battalion Commander', 'reports_to': 'Colonel Smith'}
        ],
        'orders': [
            {'date': '2024-01-15', 'content': 'Clear area of insurgents', 'issued_by': 'Colonel Smith'}
        ],
        'operations': [
            {'date': '2024-01-16', 'location': 'Village A', 'civilian_casualties': 15}
        ]
    },
    case_id='CASE-2024-001'
})

# Extract command responsibility findings
responsibility = result['command_responsibility_analysis']
commanders = responsibility['decision_makers']
```

### Tactical Operations & IHL Analysis

```python
from agents.military_structure_analyst.config import TACTICAL_ANALYSIS_CONFIG

agent = MilitaryStructureAnalystAgent(config=TACTICAL_ANALYSIS_CONFIG)

result = agent.process({
    'military_force': '3rd Infantry Division',
    'operational_data': {
        'operations': [
            {
                'name': 'Operation Clear Sky',
                'date': '2024-02-20',
                'objectives': ['Neutralize artillery positions'],
                'methods': ['Artillery barrage', 'Infantry assault'],
                'civilian_impact': '12 civilians killed, hospital damaged'
            }
        ]
    },
    'case_id': 'CASE-2024-001',
    'analysis_focus': ['ihl_compliance', 'proportionality', 'precautions']
})

# Extract IHL findings
ihl = result['ihl_compliance_assessment']
proportionality = ihl['proportionality_analysis']
precautions = ihl['precautions_assessment']
```

### Multi-Agent Workflow

```python
from shared import AuditLogger, EvidenceHandler
from agents.osint_synthesis.agent import OSINTSynthesisAgent
from agents.military_structure_analyst.agent import MilitaryStructureAnalystAgent

shared_infra = {
    'audit_logger': AuditLogger(),
    'evidence_handler': EvidenceHandler()
}

osint = OSINTSynthesisAgent(**shared_infra)
military = MilitaryStructureAnalystAgent(**shared_infra)

# Gather OSINT on military unit
osint_result = osint.process({
    'monitoring_query': '5th Army Corps operations',
    'case_id': 'CASE-001'
})

# Analyze military structure based on OSINT
military_result = military.process({
    'military_force': '5th Army Corps',
    'organizational_data': osint_result['intelligence_assessment'],
    'operational_data': osint_result['events_timeline'],
    'case_id': 'CASE-001'
})
```

## Integration with Other Agents

### OSINT Synthesis
Provides intelligence on military units, operations, and command structures.

### Satellite Imagery Analyst
Documents military positions, movements, and infrastructure.

### Ballistics & Weapons Identifier
Links weapons to military units and operations.

### Torture Analyst
Assesses command responsibility for systematic torture.

### Evidence Gap Identifier
Identifies missing military records or command documentation needed.

## Evidentiary Standards

- **Military Expertise**: Applies professional military knowledge
- **Objectivity**: Assesses operations without bias
- **Evidence-Based**: Grounds conclusions in documentary/testimonial evidence
- **Legal Relevance**: Focuses on IHL compliance and accountability
- **Clear Explanation**: Translates military concepts for legal audience
- **Chain of Custody**: All evidence handling logged

## Technical Requirements

- Python 3.9+
- Anthropic API key (Claude Sonnet 4.5)
- Access to military organizational data, operational reports, orders

## Limitations

- Analysis based on available evidence (may be incomplete)
- Cannot replace military expert witnesses for court
- Command responsibility requires detailed evidence
- Some military operations require classified context
- Tactical assessments are analytical, not definitive

## Use Cases

- Command responsibility investigations
- IHL compliance assessment
- War crimes investigation (military operations analysis)
- Proportionality and military necessity evaluation
- Expert witness preparation
- Military tribunal proceedings
- Training legal professionals on military operations
- Strategic accountability planning
