# Comparative Document Analyzer Agent

Identifies similarities, differences, and patterns across multiple documents for legal and investigative purposes.

## Purpose

Analyzes relationships between documents to detect versions, patterns, forgeries, and coordinated content. Critical for proving document tampering, identifying coached testimony, and establishing authenticity chains.

## Capabilities

- **Version Comparison**: Track changes between document versions
- **Pattern Detection**: Identify recurring content across document sets
- **Similarity Analysis**: Measure document relationships quantitatively
- **Forgery Detection**: Flag suspicious metadata and content
- **Redaction Analysis**: Compare redacted vs. unredacted versions

## Usage

### Basic Comparison

```python
from agents.comparative_analyzer import ComparativeAnalyzerAgent

agent = ComparativeAnalyzerAgent()

# Compare two documents
result = agent.compare_versions(
    original_doc={"text": "Original contract text..."},
    modified_doc={"text": "Modified contract text..."},
    case_id="CASE-2024-001"
)

print(result['comparison_results']['overall_similarity'])
print(result['differences_identified'])
```

### Pattern Detection

```python
# Analyze 50 witness statements for patterns
documents = [
    {"text": "Statement 1..."},
    {"text": "Statement 2..."},
    # ... more documents
]

result = agent.detect_patterns(
    documents=documents,
    case_id="CASE-2024-002"
)

# Check for suspicious patterns
for pattern in result['patterns_detected']:
    if pattern['pattern_type'] == 'copy_paste':
        print(f"Copy-paste detected: {pattern['description']}")
```

## Output Format

Returns structured JSON with:
- Similarity scores (overall, structural, content, metadata)
- Detailed differences (additions, deletions, modifications)
- Detected patterns (boilerplate, templates, coordination)
- Red flags (forgery indicators, inconsistencies)
- Comparison matrix
- Timeline analysis

## Configuration

```python
from agents.comparative_analyzer.config import ComparativeAnalyzerConfig

config = ComparativeAnalyzerConfig(
    max_tokens=16384,  # Large context for many documents
    high_similarity_threshold=0.90,
    detect_patterns=True,
    analyze_metadata=True
)

agent = ComparativeAnalyzerAgent(config=config)
```

## Use Cases

- **Contract Analysis**: Identify unauthorized changes to agreements
- **Witness Statement Comparison**: Detect coached or coordinated testimony
- **Forgery Detection**: Flag suspicious metadata and timing
- **Redaction Review**: Understand what was hidden and why
- **Document Family Analysis**: Group related documents by similarity
