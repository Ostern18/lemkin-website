# Lemkin NER - Multilingual Named Entity Recognition for Legal Documents

A comprehensive multilingual named entity recognition (NER) system optimized for legal investigations and document analysis. Lemkin NER provides advanced entity extraction, cross-document linking, validation workflows, and support for multiple languages commonly found in legal contexts.

## Features

### 🌍 Multilingual Support
- **Languages**: English, Spanish, French, German, Italian, Portuguese, Russian, Chinese, Arabic, Japanese
- **Automatic language detection** with confidence scoring
- **Cross-language entity matching** with transliteration support
- **Language-specific processing** using specialized models

### ⚖️ Legal Domain Optimization
- **Legal entity types**: Courts, statutes, case names, legal documents, contracts
- **Legal terminology recognition** with domain-specific dictionaries
- **Citation extraction** for statutes, cases, and regulations
- **Legal role identification** (judges, attorneys, prosecutors, etc.)

### 🔗 Entity Linking & Resolution
- **Cross-document entity resolution** to identify the same entities across documents
- **Fuzzy matching** for name variations and aliases
- **Graph-based relationship mapping** between entities
- **Co-reference resolution** for entity mentions

### ✅ Validation & Quality Assurance
- **Automated quality scoring** based on multiple metrics
- **Human validation workflows** with review task generation
- **Confidence-based filtering** and validation thresholds
- **Interactive validation interfaces** for manual review

### 🛠️ Developer-Friendly
- **Command-line interface** for batch processing
- **Python API** for programmatic access
- **Multiple export formats** (JSON, CSV, XML, GEXF)
- **Comprehensive logging** and error handling

## Installation

### Requirements
- Python 3.10 or higher
- 4GB+ RAM recommended for optimal performance
- GPU support optional but recommended for transformer models

### Install from PyPI
```bash
pip install lemkin-ner
```

### Install from Source
```bash
git clone https://github.com/lemkin-ai/lemkin-ner.git
cd lemkin-ner
pip install -e .
```

### Install Language Models
```bash
# Install spaCy models for supported languages
python -m spacy download en_core_web_sm
python -m spacy download es_core_news_sm
python -m spacy download fr_core_news_sm
python -m spacy download de_core_news_sm

# Install Stanza models (automatic download on first use)
python -c "import stanza; stanza.download('en'); stanza.download('es')"
```

## Quick Start

### Command Line Usage

#### Extract entities from a document:
```bash
lemkin-ner extract-entities document.txt --output results.json --language en
```

#### Process multiple files:
```bash
lemkin-ner extract-entities ./documents/ --batch --output-dir ./results/
```

#### Link entities across documents:
```bash
lemkin-ner link-entities ./results/ --output entity_graph.json
```

#### Validate extracted entities:
```bash
lemkin-ner validate-entities results.json --output-dir ./validation/
```

### Python API Usage

#### Basic Entity Extraction
```python
from lemkin_ner import LegalNERProcessor, NERConfig, LanguageCode

# Create configuration
config = NERConfig(
    primary_language=LanguageCode.EN,
    min_confidence=0.7,
    enable_legal_ner=True
)

# Initialize processor
processor = LegalNERProcessor(config)

# Process text
text = """
In the case of Smith v. ABC Corporation, Judge Williams 
presided over the hearing at the Southern District Court.
"""

result = processor.process_text(text, document_id="case_001")

# Access extracted entities
for entity in result['entities']:
    print(f"{entity['text']} ({entity['entity_type']}) - {entity['confidence']:.3f}")
```

#### Cross-Document Entity Linking
```python
# Process multiple documents
documents = [
    "John Smith filed a lawsuit against ABC Corp.",
    "Mr. Smith's attorney argued the case.",
    "ABC Corporation denied the allegations."
]

results = []
for i, doc in enumerate(documents):
    result = processor.process_text(doc, f"doc_{i}")
    results.append(result)

# Create entity graph
entity_graph = processor.link_entities_across_documents(results)

# Access relationships
for relationship in entity_graph.relationships:
    source = entity_graph.entities[relationship['source_id']]
    target = entity_graph.entities[relationship['target_id']]
    print(f"{source.text} --[{relationship['relationship_type']}]--> {target.text}")
```

#### Entity Validation
```python
from lemkin_ner import EntityValidator, Entity

validator = EntityValidator(config)

# Create entity objects from results
entities = [Entity.model_validate(e) for e in result['entities']]

# Validate entities
validation_results = validator.validate_batch(entities)

# Generate quality report
quality_report = validator.generate_quality_report(validation_results)
print(f"Validity rate: {quality_report['summary']['validity_rate']:.2%}")

# Create human review tasks
review_summary = validator.create_human_review_tasks(
    validation_results, 
    output_dir="./review_tasks"
)
```

## Configuration

### Configuration File
Create a configuration file to customize behavior:

```yaml
# config.yaml
primary_language: "en"
supported_languages: ["en", "es", "fr"]
auto_detect_language: true

entity_types:
  - "PERSON"
  - "ORGANIZATION"
  - "LOCATION"
  - "DATE"
  - "LEGAL_ENTITY"
  - "COURT"

min_confidence: 0.6
similarity_threshold: 0.8
validation_threshold: 0.7

enable_legal_ner: true
enable_entity_linking: true
enable_validation: true
require_human_review: false

use_transformers: true
transformer_model: "dbmdz/bert-large-cased-finetuned-conll03-english"

output_format: "json"
include_context: true
normalize_entities: true
```

### Use configuration file:
```bash
lemkin-ner extract-entities document.txt --config config.yaml
```

## Entity Types

### Standard Entity Types
- **PERSON**: Individual names (judges, attorneys, plaintiffs, defendants)
- **ORGANIZATION**: Companies, law firms, government agencies
- **LOCATION**: Cities, states, countries, addresses, jurisdictions
- **DATE**: Dates and time expressions
- **TIME**: Time-specific expressions
- **EVENT**: Named events, incidents, proceedings

### Legal-Specific Entity Types
- **LEGAL_ENTITY**: Legal roles and positions
- **COURT**: Court names and judicial bodies
- **STATUTE**: Laws, regulations, statutes, codes
- **CASE_NAME**: Legal case names and citations
- **CONTRACT**: Contract types and agreement names
- **LEGAL_DOCUMENT**: Legal document types (motions, briefs, orders)

## Supported Languages

| Language | Code | spaCy Model | Stanza | Legal Terms |
|----------|------|-------------|---------|-------------|
| English | `en` | ✅ | ✅ | ✅ |
| Spanish | `es` | ✅ | ✅ | ✅ |
| French | `fr` | ✅ | ✅ | ✅ |
| German | `de` | ✅ | ✅ | ✅ |
| Italian | `it` | ✅ | ✅ | ⚠️ |
| Portuguese | `pt` | ✅ | ✅ | ⚠️ |
| Russian | `ru` | ⚠️ | ✅ | ✅ |
| Chinese | `zh` | ✅ | ✅ | ⚠️ |
| Arabic | `ar` | ⚠️ | ✅ | ✅ |
| Japanese | `ja` | ✅ | ✅ | ⚠️ |

✅ Full support | ⚠️ Basic support

## Advanced Features

### Custom Legal Terminology
Add domain-specific terms by creating a terminology file:

```json
{
  "en": {
    "legal_entities": ["attorney", "counsel", "prosecutor"],
    "legal_documents": ["motion", "brief", "order", "judgment"],
    "courts": ["district court", "appeals court", "supreme court"]
  },
  "es": {
    "legal_entities": ["abogado", "fiscal", "juez"],
    "legal_documents": ["demanda", "sentencia", "auto"]
  }
}
```

### Human Validation Workflow
1. **Extract entities** with automated processing
2. **Validate quality** using confidence and rule-based checks
3. **Generate review tasks** for human validation
4. **Review entities** using generated CSV/JSON files
5. **Process feedback** to improve future extractions

#### Review Task Format
```csv
entity_id,text,entity_type,confidence,issues,decision,corrected_text,reviewer
uuid-1234,John Smith,PERSON,0.95,"","approved","",reviewer@example.com
uuid-5678,ABC Corp,ORGANIZATION,0.75,"Boundary issue","corrected","ABC Corporation",reviewer@example.com
```

### Entity Graph Analysis
Analyze entity relationships using NetworkX integration:

```bash
lemkin-ner analyze-graph entity_graph.json --metrics centrality clusters components
```

Export to graph formats for visualization:
```bash
lemkin-ner link-entities results.json --format gexf --output graph.gexf
```

## Performance Optimization

### Memory Usage
- Use `max_entities_per_document` to limit memory usage
- Process large document collections in batches
- Clear entity cache periodically for long-running processes

### Speed Optimization
- Disable transformer models for faster processing: `use_transformers: false`
- Reduce context window size: `context_window: 25`
- Use specific entity types: `entity_types: ["PERSON", "ORGANIZATION"]`

### GPU Acceleration
Enable GPU support for transformer models:
```python
config = NERConfig(
    use_transformers=True,
    transformer_model="dbmdz/bert-large-cased-finetuned-conll03-english"
)
```

## Error Handling and Logging

Lemkin NER uses structured logging with loguru:

```python
from loguru import logger

# Configure logging level
logger.remove()
logger.add("ner_processing.log", level="INFO")
logger.add(lambda msg: print(msg), level="WARNING")  # Console warnings

# Enable debug logging
config.debug_mode = True
```

Common error scenarios and solutions:

| Error | Cause | Solution |
|-------|-------|----------|
| `ModelNotFoundError` | Missing spaCy model | Install required language model |
| `LanguageDetectionError` | Text too short | Provide language explicitly |
| `ValidationError` | Invalid configuration | Check configuration constraints |
| `OutOfMemoryError` | Large document/batch | Reduce batch size or use chunking |

## API Reference

### Core Classes

#### `LegalNERProcessor`
Main processor class for entity extraction and linking.

**Methods:**
- `process_text(text, document_id, language=None)` - Process single text
- `process_document(file_path)` - Process document file
- `process_batch(texts, document_ids=None)` - Process multiple texts
- `link_entities_across_documents(results)` - Create entity graph
- `export_results(results, output_path, format)` - Export results

#### `EntityValidator`
Handles entity validation and quality assurance.

**Methods:**
- `validate_entity(entity, context_entities=None)` - Validate single entity
- `validate_batch(entities)` - Validate multiple entities
- `generate_quality_report(validation_results)` - Generate quality metrics
- `create_human_review_tasks(results, output_dir)` - Create review workflows

#### `Entity`
Represents an extracted named entity.

**Attributes:**
- `text` - Entity text as it appears
- `entity_type` - Type of entity (EntityType enum)
- `confidence` - Extraction confidence (0.0-1.0)
- `start_pos` / `end_pos` - Position in original text
- `language` - Language of the entity
- `normalized_form` - Canonical form of entity
- `aliases` - Alternative names/forms

### Configuration Options

#### `NERConfig`
Configuration class for customizing behavior.

**Key Parameters:**
- `primary_language` - Default language for processing
- `entity_types` - Types of entities to extract
- `min_confidence` - Minimum confidence threshold
- `enable_legal_ner` - Enable legal-specific processing
- `similarity_threshold` - Threshold for entity linking
- `validation_threshold` - Threshold for validation

## Integration Examples

### Elasticsearch Integration
```python
from elasticsearch import Elasticsearch

es = Elasticsearch()

def index_entities(doc_id, entities):
    for entity in entities:
        es.index(
            index="legal_entities",
            body={
                "document_id": doc_id,
                "entity_text": entity.text,
                "entity_type": entity.entity_type.value,
                "confidence": entity.confidence,
                "language": entity.language.value
            }
        )
```

### Database Integration
```python
import sqlite3

def store_entities(entities, db_path="entities.db"):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS entities (
            id TEXT PRIMARY KEY,
            text TEXT,
            entity_type TEXT,
            confidence REAL,
            document_id TEXT,
            language TEXT
        )
    """)
    
    for entity in entities:
        cursor.execute(
            "INSERT INTO entities VALUES (?, ?, ?, ?, ?, ?)",
            (entity.entity_id, entity.text, entity.entity_type.value,
             entity.confidence, entity.document_id, entity.language.value)
        )
    
    conn.commit()
    conn.close()
```

### Web API Integration
```python
from flask import Flask, request, jsonify
from lemkin_ner import LegalNERProcessor, create_default_config

app = Flask(__name__)
processor = LegalNERProcessor(create_default_config())

@app.route('/extract', methods=['POST'])
def extract_entities():
    data = request.json
    text = data.get('text', '')
    
    result = processor.process_text(text, "api_request")
    return jsonify(result)

@app.route('/validate', methods=['POST'])
def validate_entities():
    # Validation endpoint implementation
    pass
```

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup
```bash
git clone https://github.com/lemkin-ai/lemkin-ner.git
cd lemkin-ner
pip install -e ".[dev]"
pre-commit install
```

### Running Tests
```bash
pytest tests/ -v
pytest tests/test_integration.py -v  # Integration tests
```

### Code Quality
```bash
black src/  # Format code
ruff check src/  # Lint code
mypy src/  # Type checking
```

## License

Apache License 2.0. See [LICENSE](LICENSE) for details.

## Citation

If you use Lemkin NER in your research, please cite:

```bibtex
@software{lemkin_ner,
  title={Lemkin NER: Multilingual Named Entity Recognition for Legal Documents},
  author={Lemkin AI Contributors},
  year={2024},
  url={https://github.com/lemkin-ai/lemkin-ner}
}
```

## Support

- **Documentation**: [https://lemkin-ner.readthedocs.io](https://lemkin-ner.readthedocs.io)
- **Issues**: [GitHub Issues](https://github.com/lemkin-ai/lemkin-ner/issues)
- **Discussions**: [GitHub Discussions](https://github.com/lemkin-ai/lemkin-ner/discussions)
- **Email**: contributors@lemkin.org

## Changelog

### v0.1.0 (2024-01-XX)
- Initial release
- Multilingual NER support for 10 languages
- Legal domain optimization
- Entity linking and validation
- Command-line interface
- Human validation workflows

---

Built with ❤️ by the Lemkin AI team for the legal technology community.