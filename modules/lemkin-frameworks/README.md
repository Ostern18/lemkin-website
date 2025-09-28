# Lemkin Frameworks

A comprehensive Python library for analyzing evidence against international legal frameworks including the Rome Statute, Geneva Conventions, and human rights instruments.

## Overview

The Lemkin Frameworks library provides sophisticated tools for mapping evidence to legal framework elements, conducting confidence-based assessments, and generating comprehensive legal analyses across multiple international law domains.

### Named After Raphael Lemkin

This library is named in honor of Raphael Lemkin (1900-1959), the Polish-Jewish lawyer who coined the term "genocide" and was instrumental in the development of international criminal law. His pioneering work laid the foundation for the legal frameworks this library analyzes.

## Features

- **Multi-Framework Analysis**: Support for Rome Statute (ICC), Geneva Conventions, ICCPR, ECHR, ACHR, ACHPR, and UDHR
- **Advanced Evidence Processing**: Sophisticated algorithms for evidence-to-element mapping
- **Confidence Scoring**: Detailed confidence breakdowns with multiple scoring factors
- **Gap Analysis**: Identification of evidence gaps and investigation recommendations  
- **Cross-Framework Assessment**: Comprehensive analysis across multiple legal frameworks
- **Rich CLI Interface**: Complete command-line interface with multiple analysis modes
- **Extensible Architecture**: Modular design for easy addition of new legal frameworks

## Installation

```bash
pip install lemkin-frameworks
```

For development installation:

```bash
git clone https://github.com/your-org/lemkin-frameworks.git
cd lemkin-frameworks
pip install -e ".[dev]"
```

## Quick Start

### Basic Usage

```python
from lemkin_frameworks import LegalFrameworkMapper, Evidence, LegalFramework, EvidenceType
from datetime import datetime

# Create evidence
evidence = [
    Evidence(
        title="Witness Testimony - Mass Killing",
        content="Witness observed systematic killing of civilians belonging to specific ethnic group...",
        evidence_type=EvidenceType.TESTIMONY,
        source="Investigation Report #1",
        reliability_score=0.85,
        incident_date=datetime(2023, 6, 15)
    )
]

# Initialize mapper
mapper = LegalFrameworkMapper()

# Analyze against Rome Statute
analysis = mapper.map_to_legal_framework(evidence, LegalFramework.ROME_STATUTE)

print(f"Overall Confidence: {analysis.overall_confidence:.1%}")
print(f"Violations Found: {len(analysis.violations_identified)}")
```

### Command Line Interface

```bash
# Analyze evidence against Rome Statute
lemkin-frameworks analyze-rome-statute evidence.json -o results.json

# Geneva Conventions analysis
lemkin-frameworks analyze-geneva evidence.json --conflict-type international

# Human rights analysis
lemkin-frameworks analyze-human-rights evidence.json iccpr

# Comprehensive multi-framework assessment
lemkin-frameworks generate-assessment evidence.json \
  --frameworks rome_statute geneva_conventions iccpr \
  --title "Case Assessment" \
  --output comprehensive_assessment.json

# List available frameworks
lemkin-frameworks list-frameworks --detailed
```

## Legal Frameworks Supported

### 1. Rome Statute of the International Criminal Court

Analyzes evidence against ICC crimes:
- **Genocide (Article 6)**: All five acts of genocide with detailed element analysis
- **Crimes Against Humanity (Article 7)**: Murder, extermination, deportation, torture, rape, persecution, enforced disappearance, apartheid, and other inhumane acts
- **War Crimes (Article 8)**: Grave breaches and serious violations in international and non-international armed conflicts
- **Jurisdictional Elements**: Temporal, territorial, personal, and complementarity assessments

```python
from lemkin_frameworks import RomeStatuteAnalyzer

analyzer = RomeStatuteAnalyzer()
rome_analysis = analyzer.analyze(evidence)

# Access specific crime findings
print("Genocide findings:", len(rome_analysis.genocide_findings))
print("Crimes against humanity:", len(rome_analysis.crimes_against_humanity_findings))
print("War crimes:", len(rome_analysis.war_crimes_findings))
```

### 2. Geneva Conventions and Additional Protocols

International Humanitarian Law analysis:
- **Geneva Convention I**: Protection of wounded and sick in armed forces
- **Geneva Convention II**: Protection of wounded, sick, and shipwrecked at sea  
- **Geneva Convention III**: Treatment of prisoners of war
- **Geneva Convention IV**: Protection of civilian persons in time of war
- **Additional Protocol I**: Protection of victims of international armed conflicts
- **Additional Protocol II**: Protection of victims of non-international armed conflicts

```python
from lemkin_frameworks import GenevaAnalyzer

analyzer = GenevaAnalyzer()
geneva_analysis = analyzer.analyze(evidence)

# Access IHL-specific analysis
print("Grave breaches:", len(geneva_analysis.grave_breaches_found))
print("Protected persons violations:", len(geneva_analysis.protected_persons_analysis))
```

### 3. Human Rights Frameworks

Comprehensive human rights analysis across multiple instruments:

#### Universal Instruments
- **UDHR**: Universal Declaration of Human Rights
- **ICCPR**: International Covenant on Civil and Political Rights

#### Regional Instruments  
- **ECHR**: European Convention on Human Rights
- **ACHR**: American Convention on Human Rights
- **ACHPR**: African Charter on Human and Peoples' Rights

```python
from lemkin_frameworks import HumanRightsAnalyzer

# Analyze against ICCPR
analyzer = HumanRightsAnalyzer(LegalFramework.ICCPR)
hr_analysis = analyzer.analyze(evidence)

# Access human rights specific analysis
print("State responsibility:", hr_analysis.state_responsibility_assessment)
print("Individual remedies:", hr_analysis.individual_remedies)
```

## Evidence Types and Processing

The library supports multiple evidence types with specialized processing:

### Evidence Types
- **Documents**: Official documents, reports, orders, policies
- **Testimony**: Witness statements, victim accounts, expert testimony
- **Physical Evidence**: Forensic evidence, weapons, physical objects
- **Digital Evidence**: Electronic documents, communications, digital media
- **Photos**: Visual evidence, documentation of damage/harm
- **Video**: Moving visual evidence, documentation of events
- **Audio**: Recorded statements, communications, ambient sound
- **Geospatial**: Location data, maps, satellite imagery
- **Forensic**: Scientific analysis results, DNA evidence
- **Expert Reports**: Professional analysis, technical assessments

### Evidence Analysis Features

```python
# Evidence with metadata
evidence = Evidence(
    title="Medical Report - Torture Evidence",
    content="Medical examination reveals injuries consistent with torture...",
    evidence_type=EvidenceType.EXPERT_REPORT,
    source="Medical Expert Dr. Smith",
    location="Detention Center A",
    tags=["torture", "medical", "expert_analysis"],
    metadata={"expert_credentials": "Forensic medical specialist"},
    reliability_score=0.9
)

# Advanced confidence scoring
from lemkin_frameworks import ElementAnalyzer

analyzer = ElementAnalyzer()
satisfaction = analyzer.analyze_element_satisfaction(evidence_list, legal_element)

# Access detailed scoring breakdown
print("Keyword match score:", satisfaction.score_breakdown["keyword_match"])
print("Semantic similarity:", satisfaction.score_breakdown["semantic_similarity"])
print("Evidence strength:", satisfaction.score_breakdown["evidence_strength"])
```

## Analysis Methodology

### Confidence Scoring Algorithm

The library uses a sophisticated multi-factor confidence scoring algorithm:

1. **Keyword Matching (25%)**: Exact and fuzzy matching of legal keywords
2. **Semantic Similarity (20%)**: TF-IDF vectorization and cosine similarity
3. **Evidence Strength (20%)**: Relevance weighted by reliability
4. **Evidence Quantity (15%)**: Logarithmic scaling for optimal evidence count
5. **Evidence Diversity (10%)**: Variety of evidence types and sources
6. **Temporal Relevance (5%)**: Recency and temporal context
7. **Reliability Factor (10%)**: Source credibility and evidence quality
8. **Completeness (10%)**: Coverage of legal element requirements

### Element Satisfaction Levels

- **Satisfied** (≥80% confidence): Strong evidence supporting the element
- **Partially Satisfied** (50-79% confidence): Moderate evidence with some gaps
- **Insufficient Evidence** (20-49% confidence): Some relevant evidence but significant gaps
- **Not Satisfied** (<20% confidence): Little or no relevant evidence

### Gap Analysis

The system identifies specific evidence needs:

```python
# Access gap analysis
gaps = analysis.gap_analysis

print("Missing elements:", gaps.missing_elements)
print("Weak elements:", gaps.weak_elements)
print("Evidence needs:", gaps.evidence_needs)
print("Recommendations:", gaps.recommendations)
print("Priority score:", gaps.priority_score)
```

## Configuration and Customization

### Framework Configuration

```python
from lemkin_frameworks import FrameworkConfig

config = FrameworkConfig(
    confidence_threshold=0.7,           # Higher threshold for violations
    include_weak_evidence=False,        # Exclude low-reliability evidence
    require_corroboration=True,         # Require multiple sources
    enable_advanced_analytics=True,     # Enable ML features
    keyword_weight=0.3,                 # Increase keyword importance
    semantic_weight=0.25,               # Increase semantic analysis
    output_detailed_reasoning=True,     # Include full reasoning
    generate_visualizations=False       # Disable visualizations
)

mapper = LegalFrameworkMapper(config=config)
```

### Custom Evidence Templates

```bash
# Create evidence templates for testing
lemkin-frameworks create-evidence-template rome_statute --count 10 -o test_evidence.json
lemkin-frameworks create-evidence-template geneva --count 5 -o ihl_evidence.json
lemkin-frameworks create-evidence-template human_rights --count 8 -o hr_evidence.json
```

## Advanced Features

### Cross-Framework Analysis

```python
# Multi-framework assessment
assessment = mapper.generate_legal_assessment(
    evidence=evidence_list,
    frameworks=[
        LegalFramework.ROME_STATUTE,
        LegalFramework.GENEVA_CONVENTIONS,
        LegalFramework.ICCPR
    ],
    title="Comprehensive Legal Assessment",
    description="Multi-framework analysis of incident"
)

# Access cross-framework findings
cross_findings = assessment.cross_framework_findings
print("Common violations:", cross_findings["common_violations"])
print("Overlapping evidence:", cross_findings["overlapping_evidence"])
```

### Machine Learning Features

```python
from lemkin_frameworks import AdvancedElementAnalyzer

# Advanced analyzer with learning capabilities
analyzer = AdvancedElementAnalyzer()

# Provide feedback for continuous learning
analyzer.learn_from_analysis(
    evidence=evidence_list,
    legal_element=element,
    satisfaction=satisfaction_result,
    feedback_score=0.85  # Expert assessment of analysis quality
)

# Get insights from similar analyses
patterns = analyzer.get_similar_element_patterns(legal_element)
```

### Batch Processing

```bash
# Process multiple evidence files
lemkin-frameworks process-evidence file1.json file2.csv file3.txt \
  --output-dir processed_evidence \
  --format json \
  --merge
```

## Output Formats

### Analysis Results

```json
{
  "framework_analysis": {
    "framework": "rome_statute",
    "analysis_date": "2024-01-15T10:30:00Z",
    "evidence_count": 25,
    "overall_confidence": 0.78,
    "violations_identified": [
      "Article 6(a): Genocide by killing members of the group",
      "Article 7(1)(a): Murder as a crime against humanity"
    ],
    "element_satisfactions": [...],
    "gap_analysis": {...},
    "recommendations": [...]
  }
}
```

### Legal Assessment

```json
{
  "legal_assessment": {
    "title": "Comprehensive Legal Assessment",
    "assessment_date": "2024-01-15T10:30:00Z",
    "frameworks_analyzed": ["rome_statute", "geneva_conventions", "iccpr"],
    "strength_of_case": "high",
    "overall_assessment": "Analysis reveals strong evidence of multiple violations...",
    "jurisdiction_recommendations": [
      "International Criminal Court (ICC) - Strong case for jurisdiction"
    ],
    "next_steps": [
      "Review evidence gaps identified in individual framework analyses",
      "Consult with legal experts specializing in relevant jurisdictions"
    ]
  }
}
```

## Integration Examples

### With Legal Research Workflows

```python
# Integration with document analysis pipeline
from lemkin_frameworks import LegalFrameworkMapper, Evidence

def analyze_case_documents(document_paths, frameworks):
    evidence_list = []
    
    for doc_path in document_paths:
        # Extract content using your preferred method
        content = extract_document_content(doc_path)
        
        evidence = Evidence(
            title=f"Document: {doc_path}",
            content=content,
            evidence_type=EvidenceType.DOCUMENT,
            source=doc_path,
            reliability_score=assess_document_reliability(doc_path)
        )
        evidence_list.append(evidence)
    
    mapper = LegalFrameworkMapper()
    return mapper.generate_legal_assessment(
        evidence=evidence_list,
        frameworks=frameworks,
        title="Case Document Analysis"
    )
```

### With Investigation Management Systems

```python
# Integration with case management
class CaseAnalyzer:
    def __init__(self, case_id):
        self.case_id = case_id
        self.mapper = LegalFrameworkMapper()
        
    def analyze_new_evidence(self, evidence_item):
        """Analyze single piece of evidence and update case assessment"""
        # Add to existing evidence pool
        self.evidence_pool.append(evidence_item)
        
        # Re-run analysis
        return self.mapper.generate_legal_assessment(
            evidence=self.evidence_pool,
            frameworks=self.applicable_frameworks,
            title=f"Case {self.case_id} Assessment"
        )
    
    def generate_court_submission(self, format="detailed"):
        """Generate legal analysis for court submission"""
        assessment = self.mapper.generate_legal_assessment(
            evidence=self.evidence_pool,
            frameworks=self.applicable_frameworks
        )
        
        if format == "summary":
            return self._create_executive_summary(assessment)
        else:
            return self._create_detailed_analysis(assessment)
```

## Development and Testing

### Running Tests

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/

# Run with coverage
pytest tests/ --cov=lemkin_frameworks --cov-report=html
```

### Contributing New Legal Frameworks

```python
from lemkin_frameworks.core import LegalElement, LegalFramework
from lemkin_frameworks.element_analyzer import ElementAnalyzer

class CustomFrameworkAnalyzer:
    def __init__(self):
        self.element_analyzer = ElementAnalyzer()
        self.legal_elements = self._load_framework_elements()
    
    def _load_framework_elements(self):
        """Load elements for your custom framework"""
        elements = {}
        
        elements["custom_element_1"] = LegalElement(
            id="custom_element_1",
            framework=LegalFramework.CUSTOM,  # Add to enum
            article="Article 1",
            title="Custom Legal Element",
            description="Description of the legal requirement",
            requirements=[
                "First requirement",
                "Second requirement"
            ],
            keywords=["keyword1", "keyword2"],
            citation="Custom Framework, Article 1"
        )
        
        return elements
    
    def analyze(self, evidence):
        """Implement analysis logic"""
        # Follow the pattern from existing analyzers
        pass
```

## Performance Considerations

### Optimization Tips

- **Evidence Preprocessing**: Clean and normalize evidence text before analysis
- **Batch Analysis**: Process multiple evidence items together for efficiency
- **Configuration Tuning**: Adjust weights and thresholds based on use case
- **Caching**: Enable caching for repeated analyses of similar evidence

```python
# Performance optimization example
config = FrameworkConfig(
    max_elements_analyzed=50,          # Limit elements for faster processing
    enable_advanced_analytics=False,   # Disable ML for speed
    generate_visualizations=False      # Skip visualization generation
)

# Use batch processing for large evidence sets
def process_large_evidence_set(evidence_items, batch_size=100):
    results = []
    for i in range(0, len(evidence_items), batch_size):
        batch = evidence_items[i:i+batch_size]
        result = mapper.analyze_evidence_batch(batch)
        results.extend(result)
    return results
```

## Security and Privacy

### Data Handling

The library is designed with security in mind:

- **No External API Calls**: All processing is done locally
- **Evidence Anonymization**: Support for anonymizing sensitive evidence
- **Audit Logging**: Configurable logging for compliance requirements
- **Secure Configuration**: Secure handling of configuration parameters

```python
# Anonymize sensitive evidence
from lemkin_frameworks.utils import anonymize_evidence

anonymized_evidence = anonymize_evidence(
    evidence,
    anonymize_names=True,
    anonymize_locations=True,
    preserve_legal_terms=True
)
```

## Troubleshooting

### Common Issues

1. **Low Confidence Scores**
   - Check evidence quality and relevance
   - Verify legal element keywords are present
   - Consider adjusting confidence threshold

2. **Missing Evidence Types**
   - Ensure evidence type is correctly specified
   - Check evidence content is properly formatted
   - Verify reliability scores are reasonable

3. **Performance Issues**
   - Reduce max_elements_analyzed in configuration
   - Disable advanced analytics for faster processing
   - Use batch processing for large datasets

### Debug Mode

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Enable detailed logging
from loguru import logger
logger.add("debug.log", level="DEBUG")

# Analyze with debug information
analysis = mapper.map_to_legal_framework(evidence, framework)
```

## API Reference

### Core Classes

- `LegalFrameworkMapper`: Main coordination class
- `Evidence`: Evidence data model with metadata
- `LegalElement`: Legal framework element definition
- `ElementSatisfaction`: Element analysis results
- `FrameworkAnalysis`: Complete framework analysis results
- `LegalAssessment`: Multi-framework assessment results

### Analyzers

- `RomeStatuteAnalyzer`: ICC crimes analysis
- `GenevaAnalyzer`: IHL violations analysis  
- `HumanRightsAnalyzer`: Human rights violations analysis
- `ElementAnalyzer`: Core element satisfaction analysis
- `AdvancedElementAnalyzer`: ML-enhanced analysis

### Configuration

- `FrameworkConfig`: Analysis configuration options
- `ConfidenceLevel`: Confidence level enumeration
- `ElementStatus`: Element satisfaction status
- `ViolationType`: Legal violation classification

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this library in academic research, please cite:

```bibtex
@software{lemkin_frameworks,
  title={Lemkin Frameworks: A Comprehensive Library for International Legal Framework Analysis},
  author={Your Name},
  year={2024},
  url={https://github.com/your-org/lemkin-frameworks}
}
```

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Areas for Contribution

- Additional legal frameworks (regional instruments, specialized treaties)
- Enhanced natural language processing capabilities
- Machine learning model improvements
- Documentation and examples
- Performance optimizations
- Visualization features

## Support

- **Documentation**: [https://lemkin-frameworks.readthedocs.io](https://lemkin-frameworks.readthedocs.io)
- **Issues**: [GitHub Issues](https://github.com/your-org/lemkin-frameworks/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-org/lemkin-frameworks/discussions)
- **Email**: support@lemkin-frameworks.org

## Acknowledgments

- Named in honor of Raphael Lemkin, pioneer of international criminal law
- Inspired by the work of international criminal tribunals and human rights bodies
- Built with support from the legal tech and human rights communities
- Special thanks to all contributors and legal experts who provided guidance

## Disclaimer

This software is provided for research and educational purposes. Legal analysis results should be reviewed by qualified legal professionals. The software does not constitute legal advice and should not be relied upon as the sole basis for legal decisions.