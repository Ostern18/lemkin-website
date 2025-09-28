"""
Core functionality for multilingual named entity recognition and linking.
"""

import json
import uuid
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any, Union
from dataclasses import dataclass, field
from pydantic import BaseModel, Field, ConfigDict
import logging
from loguru import logger

# Configure logging
logging.basicConfig(level=logging.INFO)


class EntityType(Enum):
    """Types of entities that can be recognized"""
    PERSON = "PERSON"
    ORGANIZATION = "ORGANIZATION"  
    LOCATION = "LOCATION"
    EVENT = "EVENT"
    DATE = "DATE"
    TIME = "TIME"
    LEGAL_ENTITY = "LEGAL_ENTITY"
    COURT = "COURT"
    STATUTE = "STATUTE"
    CASE_NAME = "CASE_NAME"
    CONTRACT = "CONTRACT"
    LEGAL_DOCUMENT = "LEGAL_DOCUMENT"


class LanguageCode(Enum):
    """Supported language codes"""
    EN = "en"  # English
    ES = "es"  # Spanish  
    FR = "fr"  # French
    AR = "ar"  # Arabic
    RU = "ru"  # Russian
    ZH = "zh"  # Chinese
    DE = "de"  # German
    IT = "it"  # Italian
    PT = "pt"  # Portuguese
    JA = "ja"  # Japanese


class Entity(BaseModel):
    """Represents a named entity"""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    entity_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    text: str = Field(..., description="The entity text as it appears in the document")
    entity_type: EntityType = Field(..., description="Type of entity")
    start_pos: int = Field(..., description="Start position in original text")
    end_pos: int = Field(..., description="End position in original text")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    language: LanguageCode = Field(..., description="Language of the entity")
    document_id: str = Field(..., description="Source document identifier")
    context: Optional[str] = Field(default=None, description="Surrounding context")
    normalized_form: Optional[str] = Field(default=None, description="Normalized/canonical form")
    aliases: List[str] = Field(default_factory=list, description="Alternative names/aliases")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "entity_id": self.entity_id,
            "text": self.text,
            "entity_type": self.entity_type.value,
            "start_pos": self.start_pos,
            "end_pos": self.end_pos,
            "confidence": self.confidence,
            "language": self.language.value,
            "document_id": self.document_id,
            "context": self.context,
            "normalized_form": self.normalized_form,
            "aliases": self.aliases,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat()
        }


class EntityGraph(BaseModel):
    """Represents relationships between entities"""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    graph_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    entities: Dict[str, Entity] = Field(default_factory=dict)
    relationships: List[Dict[str, Any]] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    
    def add_entity(self, entity: Entity) -> None:
        """Add entity to the graph"""
        self.entities[entity.entity_id] = entity
    
    def add_relationship(self, source_id: str, target_id: str, 
                        relationship_type: str, confidence: float = 1.0,
                        metadata: Optional[Dict[str, Any]] = None) -> None:
        """Add relationship between entities"""
        relationship = {
            "source_id": source_id,
            "target_id": target_id, 
            "relationship_type": relationship_type,
            "confidence": confidence,
            "metadata": metadata or {},
            "created_at": datetime.now(timezone.utc).isoformat()
        }
        self.relationships.append(relationship)
    
    def get_connected_entities(self, entity_id: str) -> List[Entity]:
        """Get all entities connected to the given entity"""
        connected = []
        for rel in self.relationships:
            if rel["source_id"] == entity_id and rel["target_id"] in self.entities:
                connected.append(self.entities[rel["target_id"]])
            elif rel["target_id"] == entity_id and rel["source_id"] in self.entities:
                connected.append(self.entities[rel["source_id"]])
        return connected


class EntityLinkResult(BaseModel):
    """Result of entity linking operation"""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    source_entity: Entity
    target_entities: List[Entity] = Field(default_factory=list)
    similarity_scores: Dict[str, float] = Field(default_factory=dict)
    link_confidence: float = Field(default=0.0)
    link_type: str = Field(default="SIMILAR")
    explanation: Optional[str] = Field(default=None)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ValidationResult(BaseModel):
    """Result of entity validation"""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    entity: Entity
    is_valid: bool = Field(default=True)
    validation_confidence: float = Field(default=1.0)
    issues: List[str] = Field(default_factory=list)
    suggestions: List[str] = Field(default_factory=list)
    human_reviewed: bool = Field(default=False)
    reviewer: Optional[str] = Field(default=None)
    review_timestamp: Optional[datetime] = Field(default=None)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class NERConfig(BaseModel):
    """Configuration for NER processing"""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    # Language settings
    primary_language: LanguageCode = Field(default=LanguageCode.EN)
    supported_languages: List[LanguageCode] = Field(default_factory=lambda: [LanguageCode.EN])
    auto_detect_language: bool = Field(default=True)
    
    # Model settings  
    model_name: str = Field(default="en_core_web_sm")
    use_transformers: bool = Field(default=True)
    transformer_model: str = Field(default="dbmdz/bert-large-cased-finetuned-conll03-english")
    
    # Entity types to extract
    entity_types: List[EntityType] = Field(
        default_factory=lambda: [
            EntityType.PERSON, EntityType.ORGANIZATION, 
            EntityType.LOCATION, EntityType.DATE, EntityType.TIME
        ]
    )
    
    # Legal-specific settings
    enable_legal_ner: bool = Field(default=True)
    legal_model_path: Optional[str] = Field(default=None)
    legal_terminology_file: Optional[str] = Field(default=None)
    
    # Processing settings
    min_confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    max_entities_per_document: int = Field(default=1000)
    context_window: int = Field(default=50)
    
    # Entity linking settings
    enable_entity_linking: bool = Field(default=True)
    similarity_threshold: float = Field(default=0.8, ge=0.0, le=1.0)
    max_link_distance: int = Field(default=10)
    
    # Validation settings
    enable_validation: bool = Field(default=True)
    validation_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    require_human_review: bool = Field(default=False)
    
    # Output settings
    output_format: str = Field(default="json")
    include_context: bool = Field(default=True)
    normalize_entities: bool = Field(default=True)


class LegalNERProcessor:
    """
    Main processor for multilingual named entity recognition and linking
    optimized for legal documents
    """
    
    def __init__(self, config: NERConfig):
        """
        Initialize the LegalNER processor
        
        Args:
            config: NER configuration settings
        """
        self.config = config
        self.entity_recognizer = None
        self.entity_linker = None
        self.multilingual_processor = None
        self.entity_validator = None
        
        # Initialize components
        self._initialize_components()
        
        logger.info("LegalNERProcessor initialized with config: {}", config.model_dump())
    
    def _initialize_components(self) -> None:
        """Initialize NER pipeline components"""
        try:
            # Import and initialize components
            from .legal_ner import LegalEntityRecognizer
            from .entity_linking import EntityLinker
            from .multilingual_processor import MultilingualProcessor
            from .entity_validator import EntityValidator
            
            self.entity_recognizer = LegalEntityRecognizer(self.config)
            self.entity_linker = EntityLinker(self.config)
            self.multilingual_processor = MultilingualProcessor(self.config)
            self.entity_validator = EntityValidator(self.config)
            
            logger.info("All NER components initialized successfully")
            
        except ImportError as e:
            logger.error("Failed to import NER components: {}", e)
            raise
        except Exception as e:
            logger.error("Failed to initialize NER components: {}", e)
            raise
    
    def process_text(self, text: str, document_id: Optional[str] = None,
                    language: Optional[LanguageCode] = None) -> Dict[str, Any]:
        """
        Process text for named entity recognition
        
        Args:
            text: Input text to process
            document_id: Optional document identifier
            language: Optional language code (auto-detected if not provided)
            
        Returns:
            Dictionary containing extracted entities and metadata
        """
        if not text or not text.strip():
            logger.warning("Empty or whitespace-only text provided")
            return {"entities": [], "metadata": {"status": "empty_text"}}
        
        if not document_id:
            document_id = str(uuid.uuid4())
        
        logger.info("Processing text document: {} (length: {} chars)", document_id, len(text))
        
        try:
            # Step 1: Language detection
            if not language and self.config.auto_detect_language:
                language = self.multilingual_processor.detect_language(text)
                logger.info("Detected language: {}", language.value)
            elif not language:
                language = self.config.primary_language
            
            # Step 2: Extract entities using appropriate models
            entities = []
            
            # Standard NER extraction
            standard_entities = self.entity_recognizer.extract_entities(
                text, document_id, language
            )
            entities.extend(standard_entities)
            
            # Legal-specific NER if enabled
            if self.config.enable_legal_ner:
                legal_entities = self.entity_recognizer.extract_legal_entities(
                    text, document_id, language
                )
                entities.extend(legal_entities)
            
            # Step 3: Filter by confidence threshold
            filtered_entities = [
                entity for entity in entities 
                if entity.confidence >= self.config.min_confidence
            ]
            
            # Step 4: Limit number of entities
            if len(filtered_entities) > self.config.max_entities_per_document:
                filtered_entities = sorted(
                    filtered_entities, key=lambda x: x.confidence, reverse=True
                )[:self.config.max_entities_per_document]
                logger.warning(
                    "Limited entities to {} (original: {})",
                    self.config.max_entities_per_document,
                    len(entities)
                )
            
            # Step 5: Entity normalization
            if self.config.normalize_entities:
                normalized_entities = []
                for entity in filtered_entities:
                    normalized = self.multilingual_processor.normalize_entity(entity)
                    normalized_entities.append(normalized)
                filtered_entities = normalized_entities
            
            # Step 6: Entity validation
            validation_results = []
            if self.config.enable_validation:
                for entity in filtered_entities:
                    validation = self.entity_validator.validate_entity(entity)
                    validation_results.append(validation)
            
            result = {
                "entities": [entity.to_dict() for entity in filtered_entities],
                "validation_results": [vr.model_dump() for vr in validation_results],
                "metadata": {
                    "document_id": document_id,
                    "language": language.value,
                    "total_entities_found": len(entities),
                    "entities_after_filtering": len(filtered_entities),
                    "processing_timestamp": datetime.now(timezone.utc).isoformat(),
                    "config_hash": hash(str(self.config.model_dump()))
                }
            }
            
            logger.info(
                "Text processing completed: {} entities extracted", 
                len(filtered_entities)
            )
            return result
            
        except Exception as e:
            logger.error("Error processing text: {}", e)
            return {
                "entities": [],
                "validation_results": [],
                "metadata": {
                    "document_id": document_id,
                    "error": str(e),
                    "status": "error",
                    "processing_timestamp": datetime.now(timezone.utc).isoformat()
                }
            }
    
    def process_document(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Process document file for named entity recognition
        
        Args:
            file_path: Path to document file
            
        Returns:
            Dictionary containing extracted entities and metadata
        """
        file_path = Path(file_path)
        if not file_path.exists():
            logger.error("Document file not found: {}", file_path)
            raise FileNotFoundError(f"Document file not found: {file_path}")
        
        logger.info("Processing document file: {}", file_path)
        
        try:
            # Extract text from document
            text = self._extract_text_from_document(file_path)
            document_id = file_path.stem
            
            # Process extracted text
            result = self.process_text(text, document_id)
            
            # Add file metadata
            result["metadata"].update({
                "file_path": str(file_path),
                "file_size": file_path.stat().st_size,
                "file_type": file_path.suffix.lower(),
                "text_length": len(text)
            })
            
            return result
            
        except Exception as e:
            logger.error("Error processing document {}: {}", file_path, e)
            raise
    
    def process_batch(self, texts: List[str], document_ids: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Process multiple texts in batch
        
        Args:
            texts: List of texts to process
            document_ids: Optional list of document identifiers
            
        Returns:
            List of processing results
        """
        if not texts:
            return []
        
        if document_ids and len(document_ids) != len(texts):
            raise ValueError("document_ids length must match texts length")
        
        if not document_ids:
            document_ids = [str(uuid.uuid4()) for _ in texts]
        
        logger.info("Processing batch of {} texts", len(texts))
        
        results = []
        for i, text in enumerate(texts):
            try:
                result = self.process_text(text, document_ids[i])
                results.append(result)
            except Exception as e:
                logger.error("Error processing batch item {}: {}", i, e)
                results.append({
                    "entities": [],
                    "validation_results": [],
                    "metadata": {
                        "document_id": document_ids[i],
                        "error": str(e),
                        "status": "error",
                        "batch_index": i
                    }
                })
        
        logger.info("Batch processing completed: {}/{} successful", 
                   sum(1 for r in results if "error" not in r["metadata"]), len(results))
        return results
    
    def link_entities_across_documents(self, document_results: List[Dict[str, Any]]) -> EntityGraph:
        """
        Link entities across multiple documents
        
        Args:
            document_results: List of document processing results
            
        Returns:
            EntityGraph with linked entities
        """
        if not self.config.enable_entity_linking:
            logger.warning("Entity linking is disabled in configuration")
            return EntityGraph()
        
        logger.info("Linking entities across {} documents", len(document_results))
        
        # Collect all entities
        all_entities = []
        for result in document_results:
            for entity_dict in result.get("entities", []):
                entity = Entity.model_validate(entity_dict)
                all_entities.append(entity)
        
        # Use entity linker to create graph
        entity_graph = self.entity_linker.create_entity_graph(all_entities)
        
        logger.info(
            "Entity linking completed: {} entities, {} relationships",
            len(entity_graph.entities),
            len(entity_graph.relationships)
        )
        
        return entity_graph
    
    def export_results(self, results: Union[Dict[str, Any], List[Dict[str, Any]]], 
                      output_path: Union[str, Path], format: str = "json") -> None:
        """
        Export processing results to file
        
        Args:
            results: Processing results to export
            output_path: Output file path
            format: Export format ("json", "csv", "xml")
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info("Exporting results to: {} (format: {})", output_path, format)
        
        try:
            if format.lower() == "json":
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(results, f, indent=2, ensure_ascii=False, default=str)
            
            elif format.lower() == "csv":
                import pandas as pd
                
                # Flatten entities for CSV export
                if isinstance(results, dict):
                    entities = results.get("entities", [])
                else:
                    entities = []
                    for result in results:
                        entities.extend(result.get("entities", []))
                
                df = pd.DataFrame(entities)
                df.to_csv(output_path, index=False)
            
            elif format.lower() == "xml":
                # Simple XML export
                import xml.etree.ElementTree as ET
                
                root = ET.Element("ner_results")
                
                if isinstance(results, dict):
                    doc_results = [results]
                else:
                    doc_results = results
                
                for doc_result in doc_results:
                    doc_elem = ET.SubElement(root, "document")
                    doc_elem.set("id", doc_result.get("metadata", {}).get("document_id", "unknown"))
                    
                    entities_elem = ET.SubElement(doc_elem, "entities")
                    for entity in doc_result.get("entities", []):
                        entity_elem = ET.SubElement(entities_elem, "entity")
                        for key, value in entity.items():
                            if isinstance(value, (str, int, float, bool)):
                                entity_elem.set(key, str(value))
                
                tree = ET.ElementTree(root)
                tree.write(output_path, encoding='utf-8', xml_declaration=True)
            
            else:
                raise ValueError(f"Unsupported export format: {format}")
            
            logger.info("Results exported successfully to: {}", output_path)
            
        except Exception as e:
            logger.error("Error exporting results: {}", e)
            raise
    
    def _extract_text_from_document(self, file_path: Path) -> str:
        """Extract text from various document formats"""
        suffix = file_path.suffix.lower()
        
        try:
            if suffix == '.txt':
                return file_path.read_text(encoding='utf-8')
            
            elif suffix == '.pdf':
                try:
                    import PyPDF2
                    with open(file_path, 'rb') as f:
                        reader = PyPDF2.PdfReader(f)
                        text = ""
                        for page in reader.pages:
                            text += page.extract_text() + "\n"
                        return text
                except ImportError:
                    logger.error("PyPDF2 not installed, cannot process PDF files")
                    raise
            
            elif suffix in ['.doc', '.docx']:
                try:
                    from docx import Document
                    doc = Document(file_path)
                    text = ""
                    for paragraph in doc.paragraphs:
                        text += paragraph.text + "\n"
                    return text
                except ImportError:
                    logger.error("python-docx not installed, cannot process Word files")
                    raise
            
            elif suffix in ['.html', '.htm']:
                try:
                    from bs4 import BeautifulSoup
                    with open(file_path, 'r', encoding='utf-8') as f:
                        soup = BeautifulSoup(f.read(), 'html.parser')
                        return soup.get_text()
                except ImportError:
                    logger.error("beautifulsoup4 not installed, cannot process HTML files")
                    raise
            
            else:
                logger.warning("Unsupported file format: {}, treating as plain text", suffix)
                return file_path.read_text(encoding='utf-8', errors='ignore')
                
        except Exception as e:
            logger.error("Error extracting text from {}: {}", file_path, e)
            raise


def create_default_config() -> NERConfig:
    """Create default NER configuration"""
    return NERConfig()


def validate_config(config: NERConfig) -> List[str]:
    """Validate NER configuration and return list of issues"""
    issues = []
    
    if config.min_confidence < 0.0 or config.min_confidence > 1.0:
        issues.append("min_confidence must be between 0.0 and 1.0")
    
    if config.similarity_threshold < 0.0 or config.similarity_threshold > 1.0:
        issues.append("similarity_threshold must be between 0.0 and 1.0")
    
    if config.validation_threshold < 0.0 or config.validation_threshold > 1.0:
        issues.append("validation_threshold must be between 0.0 and 1.0")
    
    if config.max_entities_per_document <= 0:
        issues.append("max_entities_per_document must be positive")
    
    if config.context_window < 0:
        issues.append("context_window must be non-negative")
    
    if not config.entity_types:
        issues.append("At least one entity type must be specified")
    
    return issues