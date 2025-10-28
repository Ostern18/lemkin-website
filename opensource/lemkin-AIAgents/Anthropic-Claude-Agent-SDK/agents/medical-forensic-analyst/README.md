# Medical & Forensic Record Analyst Agent

Interprets medical reports, autopsy records, and forensic documentation for legal and investigative purposes.

## Purpose

Translates complex medical and forensic evidence into legally relevant findings, with special emphasis on torture documentation, death investigations, and consistency analysis.

## Capabilities

- **Medical Record Interpretation**: Extract and explain diagnoses, injuries, treatments
- **Istanbul Protocol Application**: Systematic torture indicator assessment
- **Autopsy Analysis**: Interpret cause/manner of death, forensic findings
- **Consistency Checking**: Identify inconsistencies in medical narratives
- **Legal Mapping**: Link medical findings to criminal elements

## Usage

### Basic Medical Analysis

```python
from agents.medical_forensic_analyst import MedicalForensicAnalystAgent

agent = MedicalForensicAnalystAgent()

result = agent.process({
    'record_text': "Medical examination text...",
    'record_type': 'medical_report',
    'evidence_id': 'evidence-uuid'
})

print(result['key_findings']['injuries'])
print(result['layperson_summary'])
```

### Torture Assessment

```python
result = agent.analyze_for_torture(
    medical_record="Patient presents with...",
    case_id="CASE-2024-001"
)

if result['torture_indicators']['present']:
    print("Torture indicators found:")
    for indicator in result['torture_indicators']['indicators_found']:
        print(f"- {indicator['indicator']}: {indicator['evidence']}")
```

### Autopsy Analysis

```python
result = agent.analyze_autopsy(
    autopsy_report="Autopsy report text...",
    case_id="CASE-2024-002"
)

print(f"Cause of death: {result['key_findings']['cause_of_death']}")
print(f"Inconsistencies: {result['inconsistencies_identified']}")
```

## Output Format

Returns structured JSON with:
- Key medical findings (injuries, diagnoses, treatments, outcomes)
- Torture indicators (Istanbul Protocol assessment)
- Medical timeline
- Inconsistencies
- Legal relevance (charges supported, elements proven)
- Layperson summary
- Expert consultation recommendations

## Configuration

```python
from agents.medical_forensic_analyst.config import MedicalForensicConfig

config = MedicalForensicConfig(
    apply_istanbul_protocol=True,
    require_review_for_torture_findings=True,
    redact_patient_identifiers=True
)

agent = MedicalForensicAnalystAgent(config=config)
```

## Use Cases

- **Torture Documentation**: Systematically assess medical evidence of torture
- **Death Investigations**: Interpret autopsy findings for legal proceedings
- **Injury Consistency**: Verify if injuries match claimed causes
- **Expert Preparation**: Generate summaries for medical experts
- **Legal Strategy**: Identify which charges medical evidence supports
