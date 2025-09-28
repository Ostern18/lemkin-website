# Lemkin AI Open Source Development Guide for Claude Code

## Project Overview

Lemkin AI is an open-source platform designed to democratize legal investigation technology for international human rights investigators, prosecutors, civil rights attorneys, and public defenders. The platform transforms complex legal investigation from manual document review into intelligent, AI-augmented analysis while maintaining the highest standards of evidence integrity and legal ethics.

### Core Mission
- **Justice Access**: Provide every public interest lawyer with analytical power equivalent to a 20-person research unit
- **Evidence Integrity**: Maintain immutable evidence storage with complete chain of custody
- **Human Authority**: AI suggests, humans decide - especially for legal conclusions
- **Global Impact**: Support both domestic civil rights and international human rights work
- **Professional Grade**: Meet quality standards expected in courtrooms and tribunals

### Key Principles
1. **Evidence First**: Every AI insight must trace back to source documents with complete citation chains
2. **Transparency**: All algorithms must be explainable and auditable
3. **Privacy & Safety**: Protect witnesses, victims, and sensitive information
4. **Legal Compliance**: Adhere to international legal standards and court requirements
5. **Open Source**: Ensure community ownership and collaborative development

## Development Standards & Requirements

### Code Quality Standards
- **Language**: Python 3.10+ with type hints required
- **Documentation**: Comprehensive docstrings, README files, and inline comments
- **Testing**: Unit tests with >80% coverage, integration tests for workflows
- **Error Handling**: Graceful error handling with informative messages
- **Logging**: Structured logging for audit trails and debugging
- **Security**: No hardcoded secrets, secure data handling, encryption at rest/transit

### Repository Structure Template
```
lemkin-<module>/
├── src/
│   └── lemkin_<module>/
│       ├── __init__.py
│       ├── core.py              # Main functionality
│       ├── utils.py             # Helper functions
│       ├── models.py            # Data models/schemas
│       ├── config.py            # Configuration management
│       └── cli.py               # Command-line interface
├── tests/
│   ├── __init__.py
│   ├── test_core.py
│   ├── test_utils.py
│   └── fixtures/                # Test data
├── notebooks/
│   ├── <module>_demo.ipynb      # Interactive demonstration
│   ├── <module>_tutorial.ipynb  # Step-by-step guide
│   └── <module>_evaluation.ipynb # Performance assessment
├── data/
│   ├── README.md                # Data description and ethics
│   ├── sample/                  # Sanitized example data
│   └── schemas/                 # Data schemas and validation
├── docs/
│   ├── README.md
│   ├── user_guide.md
│   ├── api_reference.md
│   └── safety_guidelines.md
├── scripts/
│   ├── setup.sh                 # Environment setup
│   ├── test.sh                  # Run all tests
│   └── demo.sh                  # Quick demonstration
├── .github/
│   └── workflows/
│       └── ci.yml               # Continuous integration
├── pyproject.toml               # Dependencies and build config
├── requirements.txt             # Runtime dependencies
├── requirements-dev.txt         # Development dependencies
├── README.md                    # Project overview and quickstart
├── LICENSE                      # Apache 2.0 or MIT
├── CONTRIBUTING.md              # Contribution guidelines
├── CODE_OF_CONDUCT.md           # Community standards
├── SECURITY.md                  # Security policy and reporting
├── GOVERNANCE.md                # Project governance
├── CHANGELOG.md                 # Version history
├── Dockerfile                   # Container configuration
├── docker-compose.yml           # Multi-service setup
└── Makefile                     # Common tasks automation
```

### Required Files Content Structure

#### README.md Template
```markdown
# Lemkin <Module Name>

## Purpose
[2-3 sentence description of what this module does and why it matters for human rights investigations]

## Safety & Ethics Notice
[Warning about responsible use, PII protection, and legal compliance]

## Quick Start
```bash
pip install lemkin-<module>
lemkin-<module> --help
```

## Key Features
- [Feature 1 with safety considerations]
- [Feature 2 with legal compliance notes]
- [Feature 3 with accuracy limitations]

## Usage Examples
[3-4 concrete examples with sample commands and expected outputs]

## Input/Output Specifications
[Detailed schemas for inputs and outputs]

## Evaluation & Limitations
[Performance metrics, known limitations, failure modes]

## Safety Guidelines
[Specific safety considerations for this module]

## Contributing
[How to contribute, testing requirements, review process]

## License
[License information and attribution requirements]
```

#### pyproject.toml Template
```toml
[build-system]
requires = ["setuptools>=61.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "lemkin-<module>"
version = "0.1.0"
description = "[Module description]"
readme = "README.md"
license = {text = "Apache-2.0"}
authors = [
    {name = "Lemkin AI Contributors", email = "contributors@lemkin.org"}
]
classifiers = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Legal Industry",
    "License :: OSI Approved :: Apache Software License",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Topic :: Scientific/Engineering :: Information Analysis",
]
requires-python = ">=3.10"
dependencies = [
    "pydantic>=2.0.0",
    "typer>=0.9.0",
    "rich>=13.0.0",
    "loguru>=0.7.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0.0",
    "pytest-cov>=4.0.0",
    "black>=23.0.0",
    "ruff>=0.1.0",
    "mypy>=1.0.0",
]

[project.scripts]
lemkin-<module> = "lemkin_<module>.cli:app"

[tool.setuptools.packages.find]
where = ["src"]

[tool.black]
line-length = 88
target-version = ['py310']

[tool.ruff]
line-length = 88
target-version = "py310"

[tool.mypy]
python_version = "3.10"
strict = true
```

## Tier 1: Foundation & Safety Components

### 1. Evidence Integrity Toolkit (`lemkin-integrity`)

**Purpose**: Ensure evidence admissibility through cryptographic integrity verification and chain of custody management.

**Core Components**:
- `hash_generator.py`: SHA-256 hashing with metadata preservation
- `chain_of_custody.py`: Audit trail management with timestamps and digital signatures
- `integrity_checker.py`: Validation of evidence integrity over time
- `manifest_generator.py`: Evidence collection manifests for court submission

**Key Functions**:
```python
def generate_evidence_hash(file_path: Path, metadata: Dict) -> EvidenceHash
def create_custody_entry(evidence_id: str, action: str, actor: str) -> CustodyEntry
def verify_integrity(evidence_id: str) -> IntegrityReport
def generate_court_manifest(case_id: str) -> CourtManifest
```

**Notebook**: `integrity_demo.ipynb` - Interactive evidence integrity verification workflow

**End State**: Production-ready toolkit that ensures all evidence meets legal admissibility standards with complete audit trails.

### 2. PII Redaction Pipeline (`lemkin-redaction`)

**Purpose**: Protect witness and victim privacy through automated redaction of personally identifiable information.

**Core Components**:
- `text_redactor.py`: NER-based PII detection and masking in documents
- `image_redactor.py`: Face, license plate, and identifying information blurring
- `audio_redactor.py`: Voice anonymization and sensitive audio content removal
- `video_redactor.py`: Automated video redaction combining image and audio techniques

**Key Functions**:
```python
def redact_text_pii(text: str, entities: List[str]) -> RedactedText
def blur_faces_and_plates(image: np.ndarray) -> RedactedImage
def anonymize_voice(audio: np.ndarray, sample_rate: int) -> AnonymizedAudio
def redact_video_content(video_path: Path, redaction_config: Config) -> RedactedVideo
```

**Notebooks**: 
- `redaction_demo.ipynb` - Interactive redaction workflow
- `redaction_evaluation.ipynb` - Accuracy assessment and bias testing

**End State**: Comprehensive redaction pipeline that automatically protects PII while preserving evidentiary value.

### 3. Legal Document Classifier (`lemkin-classifier`)

**Purpose**: Automatically categorize legal documents to accelerate evidence triage and case organization.

**Core Components**:
- `doc_classifier.py`: Multi-class legal document classification using fine-tuned BERT
- `legal_taxonomy.py`: Standardized legal document categories and hierarchies
- `confidence_scorer.py`: Classification confidence assessment with human review triggers
- `batch_processor.py`: High-volume document processing capabilities

**Document Categories**:
- Witness statements, police reports, medical records, court filings
- Government documents, military reports, communication records
- Expert testimony, forensic reports, media evidence

**Key Functions**:
```python
def classify_document(doc_text: str, language: str) -> DocumentClassification
def batch_classify(doc_paths: List[Path]) -> List[DocumentClassification]
def update_classification(doc_id: str, human_label: str) -> None
def get_classification_confidence(doc_text: str) -> float
```

**Notebooks**:
- `classifier_training.ipynb` - Model fine-tuning workflow
- `classifier_evaluation.ipynb` - Performance metrics and error analysis

**End State**: Robust document classifier with high accuracy across multiple languages and legal domains.

## Tier 2: Core Analysis Tools

### 4. Multilingual Entity Extraction (`lemkin-ner`)

**Purpose**: Extract and link named entities across documents in multiple languages for international investigations.

**Core Components**:
- `legal_ner.py`: Specialized NER for legal entities (persons, organizations, locations, dates)
- `entity_linking.py`: Cross-document entity resolution and deduplication
- `multilingual_processor.py`: Language-specific processing optimizations
- `entity_validator.py`: Human-in-the-loop entity verification

**Entity Types**:
- PERSON (names, aliases, roles), ORGANIZATION (agencies, groups, military units)
- LOCATION (addresses, coordinates, jurisdictions), EVENT (incidents, operations)
- DATE/TIME (temporal references), LEGAL_ENTITY (statutes, cases, courts)

**Key Functions**:
```python
def extract_entities(text: str, language: str) -> List[Entity]
def link_entities_across_documents(entities: List[Entity]) -> EntityGraph
def validate_entity_extraction(entities: List[Entity]) -> ValidationResult
def merge_duplicate_entities(entities: List[Entity]) -> List[Entity]
```

**Notebooks**:
- `ner_demo.ipynb` - Interactive entity extraction workflow
- `ner_evaluation.ipynb` - Multi-language performance assessment

**End State**: Highly accurate multilingual NER system optimized for legal documents.

### 5. Timeline Constructor (`lemkin-timeline`)

**Purpose**: Extract temporal information and construct chronological narratives from evidence.

**Core Components**:
- `temporal_extractor.py`: Date/time extraction with normalization
- `event_sequencer.py`: Chronological ordering with uncertainty handling
- `timeline_visualizer.py`: Interactive timeline generation
- `temporal_validator.py`: Consistency checking across sources

**Key Functions**:
```python
def extract_temporal_references(text: str) -> List[TemporalEntity]
def sequence_events(events: List[Event]) -> Timeline
def detect_temporal_inconsistencies(timeline: Timeline) -> List[Inconsistency]
def generate_interactive_timeline(timeline: Timeline) -> TimelineVisualization
```

**Notebooks**:
- `timeline_demo.ipynb` - Interactive timeline construction
- `temporal_analysis.ipynb` - Temporal pattern detection

**End State**: Sophisticated timeline analysis tool that handles complex temporal relationships.

### 6. Legal Framework Mapper (`lemkin-frameworks`)

**Purpose**: Map evidence to specific legal framework elements for violation assessment.

**Core Components**:
- `rome_statute.py`: ICC crimes analysis (war crimes, crimes against humanity, genocide)
- `geneva_conventions.py`: International humanitarian law violations
- `human_rights_frameworks.py`: Universal Declaration, ICCPR, regional conventions
- `element_analyzer.py`: Legal element satisfaction assessment

**Supported Frameworks**:
- Rome Statute of the International Criminal Court
- Geneva Conventions and Additional Protocols
- International Covenant on Civil and Political Rights
- Regional human rights instruments (ECHR, ACHR, ACHPR)

**Key Functions**:
```python
def analyze_rome_statute_elements(evidence: List[Evidence]) -> RomeStatuteAnalysis
def assess_geneva_violations(evidence: List[Evidence]) -> GenevaAnalysis
def map_to_legal_framework(evidence: List[Evidence], framework: str) -> FrameworkAnalysis
def generate_legal_assessment(analysis: FrameworkAnalysis) -> LegalAssessment
```

**Notebooks**:
- `framework_analysis.ipynb` - Interactive legal framework mapping
- `element_satisfaction.ipynb` - Legal element assessment workflow

**End State**: Comprehensive legal framework analysis tool for international law violations.

## Tier 3: Evidence Collection & Verification

### 7. OSINT Collection Toolkit (`lemkin-osint`)

**Purpose**: Systematic open-source intelligence gathering while respecting platform terms of service.

**Core Components**:
- `social_scraper.py`: Ethical social media data collection within ToS limits
- `web_archiver.py`: Web content preservation using Wayback Machine API
- `metadata_extractor.py`: EXIF/XMP extraction from images and videos
- `source_verifier.py`: Source credibility assessment and verification

**Key Functions**:
```python
def collect_social_media_evidence(query: str, platforms: List[str]) -> OSINTCollection
def archive_web_content(urls: List[str]) -> ArchiveCollection
def extract_media_metadata(file_path: Path) -> MediaMetadata
def verify_source_credibility(source: Source) -> CredibilityAssessment
```

**Notebooks**:
- `osint_workflow.ipynb` - Complete OSINT investigation methodology
- `source_verification.ipynb` - Source credibility assessment

**End State**: Comprehensive OSINT toolkit following Berkeley Protocol standards.

### 8. Geospatial Analysis Suite (`lemkin-geo`)

**Purpose**: Geographic analysis of evidence without requiring GIS expertise.

**Core Components**:
- `coordinate_converter.py`: GPS format standardization and projection handling
- `satellite_analyzer.py`: Satellite imagery analysis using public datasets
- `geofence_processor.py`: Location-based event correlation
- `mapping_generator.py`: Interactive map creation with evidence overlay

**Key Functions**:
```python
def standardize_coordinates(coords: str, format: str) -> StandardCoordinate
def analyze_satellite_imagery(bbox: BoundingBox, date_range: DateRange) -> SatelliteAnalysis
def correlate_events_by_location(events: List[Event], radius: float) -> LocationCorrelation
def generate_evidence_map(evidence: List[Evidence]) -> InteractiveMap
```

**Notebooks**:
- `geospatial_demo.ipynb` - Interactive geospatial analysis workflow
- `satellite_analysis.ipynb` - Satellite imagery change detection

**End State**: User-friendly geospatial analysis tools for non-GIS experts.

### 9. Digital Forensics Helpers (`lemkin-forensics`)

**Purpose**: Digital evidence analysis and authentication for non-technical investigators.

**Core Components**:
- `file_analyzer.py`: File system analysis and deleted file recovery
- `network_processor.py`: Network log analysis and communication pattern detection
- `mobile_analyzer.py`: Mobile device data extraction and analysis
- `authenticity_verifier.py`: Digital evidence authenticity verification

**Key Functions**:
```python
def analyze_file_system(disk_image: Path) -> FileSystemAnalysis
def process_network_logs(log_files: List[Path]) -> NetworkAnalysis
def extract_mobile_data(backup_path: Path) -> MobileDataExtraction
def verify_digital_authenticity(evidence: DigitalEvidence) -> AuthenticityReport
```

**Notebooks**:
- `forensics_workflow.ipynb` - Digital forensics investigation process
- `authenticity_verification.ipynb` - Digital evidence authentication

**End State**: Accessible digital forensics tools for legal investigators.

## Tier 4: Media Analysis & Authentication

### 10. Video Authentication Toolkit (`lemkin-video`)

**Purpose**: Verify video authenticity and detect manipulation.

**Core Components**:
- `deepfake_detector.py`: Integration with deepfake detection models
- `video_fingerprinter.py`: Content-based video duplicate detection
- `compression_analyzer.py`: Video compression analysis for authenticity
- `frame_analyzer.py`: Frame-level analysis and key frame extraction

**Key Functions**:
```python
def detect_deepfake(video_path: Path) -> DeepfakeAnalysis
def fingerprint_video(video_path: Path) -> VideoFingerprint
def analyze_compression_artifacts(video_path: Path) -> CompressionAnalysis
def extract_key_frames(video_path: Path) -> List[KeyFrame]
```

**Notebooks**:
- `video_authentication.ipynb` - Complete video verification workflow
- `deepfake_detection.ipynb` - Deepfake detection and analysis

**End State**: Comprehensive video authentication system for legal evidence.

### 11. Image Verification Suite (`lemkin-images`)

**Purpose**: Verify image authenticity and detect manipulation.

**Core Components**:
- `reverse_search.py`: Multi-engine reverse image search
- `manipulation_detector.py`: Image manipulation detection algorithms
- `geolocation_helper.py`: Image geolocation from visual content
- `metadata_forensics.py`: EXIF metadata analysis for authenticity

**Key Functions**:
```python
def reverse_search_image(image_path: Path) -> ReverseSearchResults
def detect_image_manipulation(image_path: Path) -> ManipulationAnalysis
def geolocate_image(image_path: Path) -> GeolocationResult
def analyze_image_metadata(image_path: Path) -> MetadataForensics
```

**Notebooks**:
- `image_verification.ipynb` - Complete image authentication workflow
- `geolocation_analysis.ipynb` - Image geolocation techniques

**End State**: Professional-grade image verification tools for investigators.

### 12. Audio Analysis Toolkit (`lemkin-audio`)

**Purpose**: Audio evidence processing and authentication.

**Core Components**:
- `speech_transcriber.py`: Multi-language speech-to-text with Whisper
- `speaker_analyzer.py`: Speaker identification and verification
- `audio_enhancer.py`: Audio quality enhancement and noise reduction
- `authenticity_detector.py`: Audio manipulation detection

**Key Functions**:
```python
def transcribe_audio(audio_path: Path, language: str) -> AudioTranscription
def identify_speakers(audio_path: Path) -> SpeakerAnalysis
def enhance_audio_quality(audio_path: Path) -> EnhancedAudio
def detect_audio_manipulation(audio_path: Path) -> AudioAuthenticity
```

**Notebooks**:
- `audio_processing.ipynb` - Complete audio analysis workflow
- `speaker_identification.ipynb` - Speaker analysis techniques

**End State**: Comprehensive audio analysis tools for legal investigations.

## Tier 5: Document Processing & Analysis

### 13. OCR & Document Digitization (`lemkin-ocr`)

**Purpose**: Convert physical documents to searchable digital format.

**Core Components**:
- `multilingual_ocr.py`: Multi-language OCR with quality assessment
- `layout_analyzer.py`: Document layout analysis and structure extraction
- `handwriting_processor.py`: Handwritten text recognition
- `quality_assessor.py`: OCR quality evaluation and improvement

**Key Functions**:
```python
def ocr_document(image_path: Path, language: str) -> OCRResult
def analyze_document_layout(image_path: Path) -> LayoutAnalysis
def process_handwriting(image_path: Path) -> HandwritingResult
def assess_ocr_quality(ocr_result: OCRResult) -> QualityAssessment
```

**Notebooks**:
- `ocr_workflow.ipynb` - Document digitization process
- `ocr_evaluation.ipynb` - OCR accuracy assessment

**End State**: High-quality document digitization with accuracy validation.

### 14. Legal Research Assistant (`lemkin-research`)

**Purpose**: Accelerate legal research and precedent analysis.

**Core Components**:
- `case_law_searcher.py`: Legal database search and retrieval
- `precedent_analyzer.py`: Case law similarity and relevance analysis
- `citation_processor.py`: Legal citation parsing and validation
- `research_aggregator.py`: Multi-source legal research compilation

**Key Functions**:
```python
def search_case_law(query: str, jurisdiction: str) -> CaseLawResults
def find_similar_precedents(case_facts: str) -> List[SimilarCase]
def parse_legal_citations(text: str) -> List[LegalCitation]
def aggregate_research(query: str, sources: List[str]) -> ResearchSummary
```

**Notebooks**:
- `legal_research.ipynb` - Interactive legal research workflow
- `precedent_analysis.ipynb` - Case law analysis techniques

**End State**: Powerful legal research tools for case building.

### 15. Communication Analysis (`lemkin-comms`)

**Purpose**: Analyze seized communications for patterns and evidence.

**Core Components**:
- `chat_processor.py`: WhatsApp/Telegram export analysis
- `email_analyzer.py`: Email thread reconstruction and analysis
- `network_mapper.py`: Communication network visualization
- `pattern_detector.py`: Communication pattern and anomaly detection

**Key Functions**:
```python
def process_chat_exports(export_path: Path) -> ChatAnalysis
def analyze_email_threads(email_data: List[Email]) -> EmailAnalysis
def map_communication_network(communications: List[Communication]) -> NetworkGraph
def detect_communication_patterns(communications: List[Communication]) -> PatternAnalysis
```

**Notebooks**:
- `communication_analysis.ipynb` - Communication data analysis workflow
- `network_analysis.ipynb` - Communication network visualization

**End State**: Comprehensive communication analysis tools for investigations.

## Tier 6: Visualization & Reporting

### 16. Evidence Dashboard Generator (`lemkin-dashboard`)

**Purpose**: Create professional dashboards for case presentation.

**Core Components**:
- `case_dashboard.py`: Streamlit-based case overview dashboard
- `timeline_visualizer.py`: Interactive timeline visualization
- `network_grapher.py`: Entity relationship network graphs
- `metrics_tracker.py`: Investigation progress tracking

**Key Functions**:
```python
def generate_case_dashboard(case_id: str) -> Dashboard
def create_interactive_timeline(events: List[Event]) -> TimelineVisualization
def visualize_entity_network(entities: List[Entity], relationships: List[Relationship]) -> NetworkGraph
def track_investigation_metrics(case_id: str) -> MetricsDashboard
```

**Notebooks**:
- `dashboard_demo.ipynb` - Interactive dashboard creation
- `visualization_gallery.ipynb` - Visualization examples and templates

**End State**: Professional visualization tools for case presentation.

### 17. Report Generator Suite (`lemkin-reports`)

**Purpose**: Generate standardized legal reports and documentation.

**Core Components**:
- `fact_sheet_generator.py`: Standardized fact sheet creation
- `evidence_cataloger.py`: Comprehensive evidence inventory
- `legal_brief_formatter.py`: Auto-populated legal brief templates
- `export_manager.py`: Multi-format report export (PDF, Word, LaTeX)

**Key Functions**:
```python
def generate_fact_sheet(case_data: CaseData) -> FactSheet
def catalog_evidence(evidence_list: List[Evidence]) -> EvidenceCatalog
def format_legal_brief(case_data: CaseData, template: str) -> LegalBrief
def export_report(report: Report, format: str) -> ExportedReport
```

**Notebooks**:
- `report_generation.ipynb` - Report creation workflow
- `template_customization.ipynb` - Custom report template creation

**End State**: Automated report generation system for legal documentation.

### 18. Data Export & Compliance (`lemkin-export`)

**Purpose**: Ensure compliance with international court submission requirements.

**Core Components**:
- `icc_formatter.py`: ICC submission format compliance
- `court_packager.py`: Court-ready evidence package creation
- `privacy_compliance.py`: GDPR/privacy-compliant data handling
- `format_validator.py`: Submission format validation

**Key Functions**:
```python
def format_for_icc(case_data: CaseData) -> ICCSubmission
def create_court_package(evidence: List[Evidence]) -> CourtPackage
def ensure_privacy_compliance(data: PersonalData) -> ComplianceReport
def validate_submission_format(submission: Submission, court: str) -> ValidationResult
```

**Notebooks**:
- `export_compliance.ipynb` - Court submission preparation
- `privacy_compliance.ipynb` - Privacy protection workflow

**End State**: Compliant data export system for international courts.

## Development Workflow Guidelines

### Phase 1: Setup and Foundation (Days 1-5)
1. **Project Structure**: Create repository structure following template
2. **CI/CD Pipeline**: Set up GitHub Actions for testing and deployment
3. **Core Dependencies**: Establish shared dependencies and configurations
4. **Documentation**: Create comprehensive README and contribution guidelines
5. **Safety Framework**: Implement PII protection and ethical guidelines

### Phase 2: Evidence Handling (Days 6-15)
1. **Evidence Integrity**: Implement cryptographic verification
2. **PII Redaction**: Build automated redaction pipeline
3. **Document Classification**: Create legal document classifier
4. **Quality Assurance**: Implement testing and validation frameworks
5. **Integration Testing**: Ensure components work together

### Phase 3: Analysis Tools (Days 16-30)
1. **Entity Extraction**: Build multilingual NER system
2. **Timeline Construction**: Implement temporal analysis
3. **Legal Framework Mapping**: Create framework analysis tools
4. **OSINT Collection**: Build ethical data collection tools
5. **Geospatial Analysis**: Implement location-based analysis

### Phase 4: Media Processing (Days 31-45)
1. **Video Authentication**: Build video verification tools
2. **Image Verification**: Implement image authenticity detection
3. **Audio Analysis**: Create audio processing pipeline
4. **Digital Forensics**: Build forensics analysis tools
5. **Media Integration**: Integrate all media processing components

### Phase 5: Advanced Features (Days 46-60)
1. **Dashboard Generation**: Create visualization tools
2. **Report Generation**: Build automated reporting
3. **Export Compliance**: Implement court submission formats
4. **Communication Analysis**: Build communication processing
5. **Legal Research**: Create research assistant tools

### Error Handling Strategy
```python
# Standard error handling pattern for all modules
try:
    result = process_evidence(evidence)
    logger.info(f"Successfully processed evidence {evidence.id}")
    return result
except ValidationError as e:
    logger.error(f"Validation failed for evidence {evidence.id}: {e}")
    raise LemkinValidationError(f"Evidence validation failed: {e}")
except SecurityError as e:
    logger.critical(f"Security violation in evidence {evidence.id}: {e}")
    raise LemkinSecurityError(f"Security violation: {e}")
except Exception as e:
    logger.error(f"Unexpected error processing evidence {evidence.id}: {e}")
    raise LemkinProcessingError(f"Processing failed: {e}")
```

### Security Requirements
1. **Input Validation**: Validate all inputs, especially file uploads
2. **Encryption**: Encrypt sensitive data at rest and in transit
3. **Access Control**: Implement role-based access control
4. **Audit Logging**: Log all operations for forensic analysis
5. **Secret Management**: Use environment variables for secrets

### Testing Strategy
```python
# Testing patterns for all modules
import pytest
from lemkin_module import core, utils
from lemkin_module.models import Evidence, Analysis

class TestCore:
    def test_process_evidence_success(self):
        """Test successful evidence processing"""
        evidence = Evidence(id="test-1", content="test content")
        result = core.process_evidence(evidence)
        assert result.status == "success"
        assert result.evidence_id == "test-1"
    
    def test_process_evidence_validation_error(self):
        """Test evidence validation failure"""
        invalid_evidence = Evidence(id="", content="")
        with pytest.raises(ValidationError):
            core.process_evidence(invalid_evidence)
    
    def test_process_evidence_security_error(self):
        """Test security violation handling"""
        malicious_evidence = Evidence(id="test-2", content="<script>alert('xss')</script>")
        with pytest.raises(SecurityError):
            core.process_evidence(malicious_evidence)
```

### Documentation Requirements
Each module must include:
1. **API Documentation**: Complete function and class documentation
2. **User Guide**: Step-by-step usage instructions
3. **Safety Guidelines**: Ethical use and safety considerations
4. **Evaluation Reports**: Performance metrics and limitations
5. **Contribution Guide**: How to contribute and extend functionality

### Ethical Guidelines Integration
All modules must implement:
1. **Consent Verification**: Ensure data usage consent
2. **PII Protection**: Automatic PII detection and protection
3. **Bias Monitoring**: Regular bias assessment and mitigation
4. **Transparency**: Clear explanations of AI decisions
5. **Human Oversight**: Human review for critical decisions

This comprehensive guide provides Claude Code with all the information needed to develop the complete Lemkin AI open-source platform while maintaining the highest standards of quality, safety, and legal compliance.