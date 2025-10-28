# Siege & Starvation Warfare Analyst Agent

The Siege & Starvation Warfare Analyst Agent documents and analyzes violations of international humanitarian law related to sieges, blockades, and deliberate starvation of civilian populations. It provides comprehensive analysis of humanitarian access restrictions, population impacts, and legal violations for accountability efforts.

## Overview

This agent specializes in analyzing siege warfare tactics, documenting denial of humanitarian access, assessing population health and nutrition impacts, and evaluating violations of IHL prohibitions on starvation as a method of warfare. It supports accountability for war crimes and crimes against humanity related to siege tactics.

## Core Capabilities

### Supply Flow & Access Analysis
- **Humanitarian access documentation** - Track aid convoy attempts, denials, and patterns
- **Supply deficit calculation** - Compare population needs to actual deliveries (food, water, medicine)
- **Checkpoint mapping** - Document access points, restrictions, and control mechanisms
- **Aid convoy attacks** - Document attacks on humanitarian workers and supplies
- **Evacuation analysis** - Track civilian evacuation attempts and denials
- **Systematic denial patterns** - Identify deliberate policies vs. ad hoc restrictions

### Population Impact Assessment
- **Malnutrition rates** - Calculate acute and chronic malnutrition across demographics
- **Mortality analysis** - Document starvation deaths, excess mortality, preventable deaths
- **Disease tracking** - Monitor disease outbreaks related to siege conditions
- **Vulnerable populations** - Assess specific impacts on children, elderly, pregnant women
- **Health infrastructure** - Document attacks on hospitals and medical facilities
- **Long-term impacts** - Assess developmental, economic, and psychosocial effects

### Siege Infrastructure Mapping
- **Territorial control** - Map areas under siege and besieging forces
- **Military positions** - Document siege lines, artillery positions, sniper locations
- **Physical barriers** - Identify walls, trenches, checkpoints, obstacles
- **Access routes** - Map potential and actual supply and evacuation routes
- **Infrastructure attacks** - Document destruction of civilian infrastructure
- **Timeline tracking** - Monitor siege progression and escalation patterns

### Legal Element Analysis
- **IHL violations** - Assess starvation as method of warfare (Additional Protocol I, Art. 54)
- **War crimes** - Analyze Rome Statute violations (Art. 8(2)(b)(xxv))
- **Crimes against humanity** - Evaluate systematic/widespread attacks on civilians
- **Genocide indicators** - Assess intent to destroy protected groups
- **Command responsibility** - Map decision-making and accountability
- **Military necessity** - Evaluate proportionality and necessity defenses

## Key Features

### Evidence-Based Analysis
- Multiple source corroboration requirements
- Data reliability and bias assessment
- Distinction between documented facts and allegations
- Alternative explanation consideration
- Confidence level scoring for all conclusions
- Temporal and geographic pattern analysis

### Legal Rigor
- Application of Geneva Conventions and Additional Protocols
- Rome Statute war crimes analysis
- Customary international humanitarian law
- Command responsibility jurisprudence
- Proportionality and military necessity assessment
- Linkage of acts to perpetrators

### Humanitarian Standards
- WHO/UNICEF malnutrition assessment standards
- Sphere Standards for humanitarian response
- WFP food security methodologies
- WASH cluster water/sanitation standards
- Protection of civilians principles
- Victim-centered approach

## Configuration Options

### Default Configuration
- Comprehensive analysis across all domains
- Evidence confidence threshold: 0.5
- Corroboration required for major findings
- Legal analysis enabled
- Pattern detection enabled
- Command responsibility assessment enabled

### Specialized Configurations

**Humanitarian Assessment (`HUMANITARIAN_CONFIG`)**
- Focus on population impact and needs assessment
- Extended token limit (20,000) for detailed health data
- Emphasis on nutrition, health, and infrastructure
- Suitable for UN agencies and humanitarian organizations

**Legal Proceedings (`LEGAL_PROCEEDINGS_CONFIG`)**
- Focus on war crimes and crimes against humanity
- Higher evidence threshold (0.7)
- Strict corroboration requirements
- Command responsibility analysis
- Suitable for prosecutors and legal teams

**Policy Analysis (`POLICY_ANALYSIS_CONFIG`)**
- Focus on systematic patterns and deliberate policies
- Policy indicator identification
- Strategic recommendations
- Suitable for advocacy and policy work

## Input Format

```python
input_data = {
    'location': 'City Name / Coordinates',
    'siege_start_date': '2024-01-01',
    'siege_end_date': '2024-06-01' or 'ongoing',
    'population_affected': '500,000',

    'supply_data': {
        'food': {
            'population_needs': 'daily calories needed',
            'actual_deliveries': 'actual supply data',
            'deficit_percentage': 75
        },
        'water': {...},
        'medicine': {...}
    },

    'humanitarian_access_reports': [
        {
            'date': '2024-02-15',
            'convoy_description': '...',
            'access_denied': True,
            'justification_given': '...',
            'evidence': [...]
        }
    ],

    'health_nutrition_data': {
        'malnutrition_rates': {
            'children_under_5_SAM': 15.2,  # percentage
            'overall_GAM': 28.5
        },
        'mortality_data': {...},
        'disease_outbreaks': [...]
    },

    'geographic_data': {
        'siege_lines': [...],
        'checkpoints': [...],
        'military_positions': [...],
        'infrastructure_attacks': [...]
    },

    'witness_testimony': [...],
    'policy_documents': [...],
    'case_id': 'CASE-2024-001',
    'analysis_focus': ['legal', 'humanitarian', 'patterns']
}
```

## Output Format

Returns structured JSON with:
- Siege characteristics and infrastructure
- Humanitarian access analysis with denial patterns
- Supply flow analysis across food, water, medicine, fuel
- Population impact (nutrition, mortality, health, infrastructure)
- Legal analysis (IHL violations, crimes against humanity, genocide indicators)
- Command responsibility assessment
- Evidence quality and alternative explanations
- Recommendations (humanitarian, investigative, accountability)
- Confidence assessments

## Usage Examples

### Basic Siege Analysis

```python
from agents.siege_starvation_analyst.agent import SiegeStarvationAnalystAgent

# Initialize agent
analyst = SiegeStarvationAnalystAgent()

# Analyze siege
result = analyst.process({
    'location': 'Besieged City',
    'siege_start_date': '2024-01-01',
    'siege_end_date': 'ongoing',
    'population_affected': '300,000',
    'supply_data': {
        'food': {
            'population_needs': '2000 kcal/person/day',
            'actual_deliveries': '400 kcal/person/day',
            'deficit_percentage': 80
        }
    },
    'humanitarian_access_reports': [
        {
            'date': '2024-02-15',
            'convoy_description': 'WFP food convoy',
            'access_denied': True,
            'justification_given': 'Security concerns'
        }
    ],
    'case_id': 'CASE-2024-001'
})

print(f"Starvation violation identified: {result['legal_analysis']['ihl_violations']['starvation_as_warfare_method']['violation_identified']}")
print(f"Confidence: {result['confidence_assessment']['overall_confidence']}")
```

### Humanitarian Impact Assessment

```python
# Focus on population impact
result = analyst.assess_humanitarian_impact(
    location='Besieged City',
    population_data={
        'malnutrition_rates': {
            'children_under_5_SAM': 18.5,
            'pregnant_women_GAM': 32.0
        },
        'mortality_data': {
            'starvation_deaths_documented': 450,
            'excess_mortality_estimated': 2300
        },
        'disease_outbreaks': [
            {'disease': 'cholera', 'cases': 1200},
            {'disease': 'measles', 'cases': 890}
        ]
    },
    case_id='CASE-2024-001'
)

print(f"Population impact severity: {result['population_impact_assessment']}")
```

### Legal Proceedings Analysis

```python
from agents.siege_starvation_analyst.config import LEGAL_PROCEEDINGS_CONFIG

# Use legal proceedings configuration
analyst = SiegeStarvationAnalystAgent(config=LEGAL_PROCEEDINGS_CONFIG)

result = analyst.process({
    'location': 'Besieged City',
    'siege_start_date': '2024-01-01',
    # ... comprehensive evidence data
    'policy_documents': [
        {
            'document_type': 'Military order',
            'date': '2024-01-05',
            'content': 'Order restricting all civilian movement...',
            'issuing_authority': 'Commander Name'
        }
    ],
    'case_id': 'CASE-2024-001',
    'analysis_focus': ['legal', 'command_responsibility']
})

# Extract legal findings
legal_findings = result['legal_analysis']
command_responsibility = result['command_responsibility']
```

### Multi-Agent Workflow Integration

```python
from shared import AuditLogger, EvidenceHandler
from agents.osint_synthesis.agent import OSINTSynthesisAgent
from agents.satellite_imagery_analyst.agent import SatelliteImageryAnalystAgent
from agents.siege_starvation_analyst.agent import SiegeStarvationAnalystAgent

# Shared infrastructure
shared_infra = {
    'audit_logger': AuditLogger(),
    'evidence_handler': EvidenceHandler()
}

# Initialize agents
osint = OSINTSynthesisAgent(**shared_infra)
imagery = SatelliteImageryAnalystAgent(**shared_infra)
siege_analyst = SiegeStarvationAnalystAgent(**shared_infra)

# Workflow: OSINT → Imagery → Siege Analysis
# 1. Monitor social media for siege claims
osint_result = osint.process({
    'monitoring_query': 'siege conditions in City',
    'case_id': 'CASE-2024-001'
})

# 2. Analyze satellite imagery of siege area
with open('siege_area.jpg', 'rb') as f:
    imagery_result = imagery.assess_site(
        image_data=f.read(),
        site_type='siege_infrastructure',
        case_id='CASE-2024-001'
    )

# 3. Comprehensive siege analysis
siege_result = siege_analyst.process({
    'location': 'Besieged City',
    'siege_start_date': '2024-01-01',
    'geographic_data': imagery_result['geographic_analysis'],
    'witness_testimony': osint_result['claims_extracted'],
    # ... additional data
    'case_id': 'CASE-2024-001'
})

# Verify chain of custody across all agents
chain = shared_infra['audit_logger'].verify_chain_integrity()
```

## Integration with Other Agents

### OSINT Synthesis
Provides open-source reporting on siege conditions, witness testimony, and social media evidence of humanitarian impact.

### Satellite Imagery Analyst
Documents siege infrastructure, territorial control, infrastructure destruction, and checkpoint locations from satellite/aerial imagery.

### Medical Forensic Analyst
Analyzes medical records documenting malnutrition, preventable deaths, and health impacts of siege conditions.

### Evidence Gap Identifier
Identifies missing evidence needed to strengthen legal case (supply records, mortality data, policy documents).

### NGO/UN Reporter
Transforms siege analysis into reports for UN mechanisms, humanitarian organizations, and advocacy purposes.

## Evidentiary Standards

- **Chain of Custody**: All evidence handling logged with immutable audit trail
- **Multi-Source Corroboration**: Major findings require multiple independent sources
- **Data Reliability**: Assessment of each source's reliability and potential bias
- **Causation Analysis**: Distinguish between siege-caused harm and general conflict impacts
- **Alternative Explanations**: Consider and evaluate alternative causes for conditions
- **Confidence Scoring**: All major conclusions include confidence levels
- **Human Review**: High-stakes findings flagged for expert review

## Technical Requirements

- Python 3.9+
- Anthropic API key (Claude Sonnet 4.5)
- Access to humanitarian data sources
- Geographic/mapping data capabilities

## Limitations

- Cannot independently verify all data (relies on provided evidence)
- Causation assessment is analytical, not definitive
- Military necessity assessments require context not always available
- Some data (nutrition surveys, mortality) may be estimates
- Access restrictions may limit available evidence

## Use Cases

- War crimes documentation for ICC or hybrid tribunals
- UN Human Rights Council reporting
- Humanitarian needs assessment and advocacy
- Policy analysis on siege warfare tactics
- Command responsibility investigations
- NGO reporting on IHL violations
- Academic research on modern siege warfare
- Early warning for genocide prevention

## Legal Framework References

- Geneva Convention IV (1949)
- Additional Protocol I (1977), Article 54
- Additional Protocol II (1977), Article 14
- Rome Statute, Article 8(2)(b)(xxv) - War crime of starvation
- Rome Statute, Article 7 - Crimes against humanity
- Customary International Humanitarian Law Rules 53-56
- UN Security Council Resolutions on protection of civilians
