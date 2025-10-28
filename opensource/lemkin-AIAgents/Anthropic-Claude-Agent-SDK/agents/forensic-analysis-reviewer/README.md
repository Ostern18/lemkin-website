# Forensic Analysis Reviewer Agent

The Forensic Analysis Reviewer Agent interprets technical forensic reports (DNA, ballistics, autopsy, toxicology, trace evidence) for legal teams and non-expert audiences. It translates complex forensic findings into accessible language, maps evidence to legal elements, and assesses methodology and expert opinions.

## Overview

This agent bridges the gap between forensic science and legal practice by reviewing technical reports, identifying legally relevant findings, evaluating forensic methodologies, and generating plain-language summaries that legal professionals can use without forensic expertise.

## Core Capabilities

### Forensic Report Interpretation
- **DNA analysis** - Match interpretation, statistical significance, inclusion/exclusion
- **Ballistics** - Weapon identification, bullet matching, trajectory analysis
- **Autopsy reports** - Cause/manner of death, injury patterns, time of death
- **Toxicology** - Substance identification, concentrations, significance
- **Trace evidence** - Fibers, hair, soil, glass analysis and significance
- **Pathology** - Injury causation, timing, pattern analysis

### Legal Element Mapping
- Map forensic findings to specific charges and legal elements
- Identify evidence supporting prosecution or defense
- Assess evidence strength for legal arguments
- Highlight exculpatory and inculpatory evidence
- Evaluate alternative explanations and weaknesses

### Methodology Assessment
- Explain forensic techniques and their appropriateness
- Identify limitations and potential issues
- Assess evidence collection quality
- Flag contamination or chain of custody concerns
- Evaluate statistical methods and conclusions

### Expert Opinion Evaluation
- Assess expert qualifications and credibility
- Evaluate strength and certainty of conclusions
- Identify areas of uncertainty vs. certainty
- Flag unsupported conclusions or overreach
- Generate follow-up questions for experts

### Non-Expert Translation
- Translate technical terminology to plain language
- Explain statistical significance and probabilities
- Clarify complex forensic concepts
- Provide context for findings
- Generate executive summaries for legal teams

## Configuration Options

**Default**: Comprehensive review across all forensic disciplines

**Legal Proceedings** (`LEGAL_PROCEEDINGS_CONFIG`):
- Focus on legal element mapping and admissibility
- Higher evidence confidence threshold (0.7)
- Detailed non-expert summaries
- Extended analysis (16,000 tokens)

**Technical Review** (`TECHNICAL_REVIEW_CONFIG`):
- Deep methodology assessment
- Technical issue identification
- Follow-up question generation
- Maximum precision (temperature 0.05)

## Usage Examples

### Basic Forensic Report Review

```python
from agents.forensic_analysis_reviewer.agent import ForensicAnalysisReviewerAgent

agent = ForensicAnalysisReviewerAgent()

result = agent.process({
    'forensic_report': """
    DNA Analysis Report
    Sample ID: EVD-2024-001
    Analysis Date: 2024-01-15
    ...
    """,
    'report_type': 'DNA',
    'charges': ['murder', 'assault'],
    'case_id': 'CASE-2024-001'
})

print(f"Key findings: {len(result['key_findings'])}")
print(f"Summary: {result['executive_summary']}")
```

### Autopsy Report Review

```python
result = agent.review_autopsy_report(
    report="""
    Autopsy Report - Medical Examiner
    Decedent: John Doe
    Cause of Death: Gunshot wound to chest
    Manner: Homicide
    ...
    """,
    case_id='CASE-2024-001'
)

# Access specific findings
cause_of_death = result['forensic_evidence_details']['autopsy_findings']['cause_of_death']
legal_relevance = result['legal_analysis']
```

### Multi-Agent Workflow

```python
from shared import AuditLogger, EvidenceHandler
from agents.document_parser.agent import DocumentParserAgent
from agents.forensic_analysis_reviewer.agent import ForensicAnalysisReviewerAgent

# Shared infrastructure
shared_infra = {
    'audit_logger': AuditLogger(),
    'evidence_handler': EvidenceHandler()
}

parser = DocumentParserAgent(**shared_infra)
forensics = ForensicAnalysisReviewerAgent(**shared_infra)

# Parse forensic report document
parsed = parser.parse_document("autopsy_report.pdf", case_id="CASE-001")

# Review forensic findings
review = forensics.process({
    'forensic_report': parsed['extracted_text']['full_text'],
    'report_type': 'autopsy',
    'charges': ['murder'],
    'case_id': 'CASE-001'
})

# Chain of custody verification
chain = shared_infra['audit_logger'].verify_chain_integrity()
```

## Integration with Other Agents

### Document Parser
Parses forensic report PDFs/images before forensic review.

### Medical Forensic Analyst
Provides medical expertise that complements forensic report interpretation.

### Evidence Gap Identifier
Identifies missing forensic evidence needed for case.

### Comparative Analyzer
Compares multiple forensic reports for consistency.

## Evidentiary Standards

- **Chain of Custody**: All evidence handling logged
- **Expert Standards**: Applies established forensic science standards
- **Methodology Rigor**: Assesses appropriateness of forensic techniques
- **Objectivity**: Presents both inculpatory and exculpatory evidence
- **Limitations**: Clearly states uncertainties and limitations
- **Accessibility**: Translates technical findings for non-experts

## Technical Requirements

- Python 3.9+
- Anthropic API key (Claude Sonnet 4.5)
- Access to forensic reports

## Limitations

- Cannot perform original forensic analysis (reviews existing reports only)
- Relies on quality of underlying forensic work
- Cannot replace expert witness testimony
- Technical assessments are analytical, not definitive
- Some forensic fields require specialized human expertise

## Use Cases

- Pre-trial forensic evidence review
- Expert witness preparation
- Forensic evidence admissibility assessment
- Jury presentation preparation
- Defense forensic review
- Forensic methodology critique
- Training for legal professionals on forensic evidence
