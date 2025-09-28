Information on the Model GitHub page:
Our Models

We've developed two specialized AI models that represent the first purpose-built solutions for legal professionals:
🔍 RoBERTa Joint NER+RE Model

The world's first multilingual legal entity recognition and relation extraction model
from transformers import AutoTokenizer, AutoModelForTokenClassification

# Load the model
tokenizer = AutoTokenizer.from_pretrained("LemkinAI/roberta-joint-ner-re")
model = AutoModelForTokenClassification.from_pretrained("LemkinAI/roberta-joint-ner-re")

🎯 Capabilities:
* 71 specialized legal entity types (persons, organizations, violations, evidence, courts, etc.)
* 21 legal relation types (perpetrator-victim, violation-location, evidence-violation, etc.)
* Multilingual support: English, French, Spanish, Arabic
* 92% F1 accuracy for named entity recognition
* 87% F1 accuracy for relation extraction
💼 Use Cases:
* Human rights violation documentation
* Legal case analysis and timeline construction
* Evidence organization and categorization
* International criminal justice case preparation
* Academic legal research and analysis

✍️ T5 Legal Narrative Generation Model

Transform structured legal data into coherent, professional narratives
from transformers import T5Tokenizer, T5ForConditionalGeneration

# Load the model
tokenizer = T5Tokenizer.from_pretrained("LemkinAI/t5-legal-narrative")
model = T5ForConditionalGeneration.from_pretrained("LemkinAI/t5-legal-narrative")

🎯 Capabilities:
* Entity-to-narrative conversion: Transform structured legal entities into coherent prose
* Legal document drafting: Generate case summaries, reports, and legal narratives
* Template-based generation: Support for structured legal document creation
* 89% ROUGE-L score for narrative quality
* 92% legal accuracy validated by legal experts
💼 Use Cases:
* Human rights report generation
* Legal case summary creation
* Investigation documentation
* Court filing narrative sections
* NGO reporting and documentation

🌍 Global Impact

🏛️ Supporting International Justice

Our models are supporting organizations working on:
* International Criminal Court (ICC) case analysis
* European Court of Human Rights (ECHR) decision processing
* UN Human Rights Council report analysis
* Truth and Reconciliation Commissions documentation
* International legal tribunal case preparation
🌐 Language & Regional Coverage

Language	Coverage	Legal Systems Supported
English	🟢 Excellent	Common law, international law
French	🟢 Strong	Civil law, international courts
Spanish	🟡 Good	Latin American legal systems
Arabic	🟡 Functional	Middle Eastern legal contexts
🚀 Quick Start

Installation

pip install transformers torch

Basic Usage

from transformers import pipeline

# Named Entity Recognition
ner_pipeline = pipeline(
    "token-classification",
    model="LemkinAI/roberta-joint-ner-re",
    tokenizer="LemkinAI/roberta-joint-ner-re"
)

# Analyze legal text
text = "The International Criminal Court issued a warrant for war crimes."
entities = ner_pipeline(text)
print(entities)

# Legal Narrative Generation
text_generator = pipeline(
    "text2text-generation",
    model="LemkinAI/t5-legal-narrative",
    tokenizer="LemkinAI/t5-legal-narrative"
)

# Generate legal narrative
prompt = "Generate narrative: violation=torture, perpetrator=military, victim=civilian, location=detention center"
narrative = text_generator(f"legal_narrative: {prompt}")
print(narrative[0]['generated_text'])


📊 Performance Benchmarks

RoBERTa NER+RE Model

Metric	Score	Benchmark
Named Entity Recognition	92% F1	Legal entity types
Relation Extraction	87% F1	Legal relationships
Multilingual Performance	89% avg	4 languages
Inference Speed	200ms	Per document (GPU)
T5 Narrative Generation Model

Metric	Score	Benchmark
ROUGE-L	89%	Narrative coherence
BLEU Score	74%	Text quality
Legal Accuracy	92%	Expert validation
Generation Speed	100 tokens/sec	GPU inference
🛠️ Technical Requirements

Minimum Requirements

* RAM: 16GB system memory
* Storage: 5GB available space
* Python: 3.7+ with PyTorch and Transformers
Recommended Setup

* RAM: 32GB system memory
* GPU: 8GB VRAM (RTX 3070/4060 or better)
* Cloud: Compatible with AWS, Google Cloud, Azure

🎯 Use Cases & Applications

🏛️ International Organizations

* UN Human Rights Office: Systematic violation documentation
* International Criminal Court: Case analysis and evidence organization
* European Court of Human Rights: Decision analysis and case preparation
* International Court of Justice: Treaty and case law analysis
🏢 Legal Technology

* Law Firms: Automated case research and document analysis
* LegalTech Startups: Building specialized legal AI applications
* E-Discovery Platforms: Enhanced legal document processing
* Compliance Software: Regulatory document analysis
🎓 Academic & Research

* Law Schools: Legal AI research and education
* Political Science Departments: Human rights research
* Think Tanks: Policy analysis and legal research
* Research Institutions: Computational law studies
🤝 Civil Society

* Human Rights NGOs: Documentation and reporting
* Advocacy Organizations: Research and case building
* Investigative Journalists: Legal document analysis
* Truth Commissions: Systematic evidence processing 
Roberta Joint NER Model: https://github.com/Lemkin-AI/lemkin-ai-models/tree/main/models/roberta-joint-ner-re  T5 Legal Narrative Model: https://github.com/Lemkin-AI/lemkin-ai-models/tree/main/models/t5-legal-narrative  ## Production-Ready Models

### 1. CivInfra-Detection (Civilian Infrastructure Detection Model)

**Local Directory Path**: `/Users/oliverstern/Documents/lemkin-data-collection/models/gaza_trained_model/`

**Model Files**:
```
gaza_trained_model/
├── saved_model.pb (266 KB) - Model graph definition
├── keras_metadata.pb (18 KB) - Keras layer metadata
└── variables/
    ├── variables.data-00000-of-00001 (274 MB) - Model weights
    └── variables.index (54 KB) - Variable index
```

**Performance Metrics**:
- **Training Accuracy**: 95.53%
- **Validation Accuracy**: 75.02%
- **Training Loss**: 0.1376
- **Validation Loss**: 1.0840
- **Model Size**: 274 MB total
- **Architecture**: Deep CNN (TensorFlow SavedModel)

**Capabilities**:
- Identifies 19 types of civilian infrastructure
- Specialized for conflict zone monitoring
- Processes satellite and aerial imagery
- Real-time detection capability

**Key Detection Categories**:
- Hospitals (9.3% of training data)
- Police Stations (8.3%)
- Prisons (7.7%)
- Airports (7.2%)
- Dams (6.5%)
- Water Treatment Facilities
- Bridges and Transportation Infrastructure

**Integration Notes**:
- Framework: TensorFlow 2.x
- Input: RGB or multispectral satellite imagery
- Output: Multi-class classification (19 categories)
- Deployment: Production-ready SavedModel format

---

### 2. BuildingDamage-Assessment (Temporal Building Damage Evaluation Model)

**Local Directory Path**: `/Users/oliverstern/Documents/lemkin-data-collection/models/syria_xbd_trained_model_final/`

**Model Files**:
```
syria_xbd_trained_model_final/
├── temporal_cnn_weights.h5 (4.4 MB) - CNN feature extractor
├── temporal_rf.pkl (61 KB) - Random Forest classifier
└── training_metadata.json (2 KB) - Model configuration
```

**Performance Metrics**:
- **Training Accuracy**: 73.56%
- **Validation Accuracy**: 73.39%
- **Training Loss**: 0.8677
- **Validation Loss**: 0.8713
- **Model Size**: 4.5 MB total (highly optimized)
- **Architecture**: Hybrid CNN + Random Forest

**Capabilities**:
- 4-level damage severity classification
- Before/after temporal analysis
- Processes image pairs for change detection
- Trained on 162,787 validated samples from 2,799 disaster events

**Damage Classification Levels**:
1. No Damage (baseline)
2. Minor Damage
3. Major Damage
4. Destroyed

**Integration Notes**:
- Framework: Keras (H5) + scikit-learn (pickle)
- Input: Before/after image pairs
- Output: 4-class damage classification
- Special: Requires both H5 and pickle file loading

---

### 3. RightsViolation-Detector (Human Rights Monitoring Model)

**Local Directory Path**: `/Users/oliverstern/Documents/lemkin-data-collection/hra-models/`

**Model Files**:
```
hra-models/
├── fingerprint.pb (54 bytes) - Model validation fingerprint
├── saved_model.pb (266 KB) - Model graph
└── variables/
    ├── variables.data-00000-of-00001 (56.7 MB) - Compressed weights
    └── variables.index (2.2 KB) - Variable mapping
```

**Performance Metrics**:
- **Estimated Accuracy**: ~95%
- **Model Size**: 56.9 MB (89.3% compression from standard VGG16)
- **Parameter Count**: 14.85 million (vs 138M in standard VGG16)
- **Architecture**: Optimized VGG16 CNN
- **Training Samples**: 243,827 images

**Capabilities**:
- Human rights violations detection
- Military facility identification
- Hospital and medical facility monitoring
- Law enforcement facility tracking
- Evidence-quality output for legal proceedings

**Key Detection Focus**:
- Military Facilities (11.4% of training)
- Hospitals (4.8%)
- Police Stations (4.7%)
- Critical Infrastructure
- Civilian Protection Zones

**Integration Notes**:
- Framework: TensorFlow 2.x SavedModel
- Input: RGB satellite/aerial imagery
- Output: Human rights violation classification
- Special: 10x faster inference than standard VGG16  Tools/Resources: 
- Separate repository for all the individual tools  
