"""
Specialized named entity recognition for legal documents.
"""

import re
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any, Union
import spacy
from spacy.tokens import Doc, Span
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import torch
import numpy as np
from loguru import logger

from .core import Entity, EntityType, LanguageCode, NERConfig


class LegalEntityRecognizer:
    """
    Specialized NER for legal entities and terminology
    """
    
    def __init__(self, config: NERConfig):
        """
        Initialize legal entity recognizer
        
        Args:
            config: NER configuration
        """
        self.config = config
        self.spacy_models = {}
        self.transformer_pipelines = {}
        self.legal_patterns = {}
        self.legal_terminology = {}
        
        # Initialize models
        self._initialize_models()
        self._load_legal_patterns()
        self._load_legal_terminology()
        
        logger.info("LegalEntityRecognizer initialized")
    
    def _initialize_models(self) -> None:
        """Initialize spaCy and transformer models for supported languages"""
        # SpaCy model mappings for different languages
        spacy_model_map = {
            LanguageCode.EN: "en_core_web_sm",
            LanguageCode.ES: "es_core_news_sm", 
            LanguageCode.FR: "fr_core_news_sm",
            LanguageCode.DE: "de_core_news_sm",
            LanguageCode.IT: "it_core_news_sm",
            LanguageCode.PT: "pt_core_news_sm",
            LanguageCode.ZH: "zh_core_web_sm",
            LanguageCode.JA: "ja_core_news_sm"
        }
        
        # Transformer model mappings
        transformer_model_map = {
            LanguageCode.EN: "dbmdz/bert-large-cased-finetuned-conll03-english",
            LanguageCode.ES: "mrm8488/bert-spanish-cased-finetuned-ner",
            LanguageCode.FR: "dbmdz/bert-base-french-europeana-cased",
            LanguageCode.DE: "dbmdz/bert-large-cased-finetuned-conll03-german",
            LanguageCode.ZH: "ckiplab/bert-base-chinese-ner",
            LanguageCode.AR: "CAMeL-Lab/bert-base-arabic-camelbert-mix-ner",
            LanguageCode.RU: "wietsedv/bert-base-multilingual-cased-finetuned-conll2002-ner"
        }
        
        # Load spaCy models for supported languages
        for language in self.config.supported_languages:
            if language in spacy_model_map:
                try:
                    model_name = spacy_model_map[language]
                    self.spacy_models[language] = spacy.load(model_name)
                    logger.info("Loaded spaCy model: {} for {}", model_name, language.value)
                except OSError:
                    logger.warning("spaCy model {} not found for {}", spacy_model_map[language], language.value)
                    # Fall back to basic model if specific language model not available
                    try:
                        self.spacy_models[language] = spacy.load("xx_core_web_sm")
                    except OSError:
                        logger.error("No fallback spaCy model available for {}", language.value)
        
        # Load transformer models if enabled
        if self.config.use_transformers:
            for language in self.config.supported_languages:
                if language in transformer_model_map:
                    try:
                        model_name = transformer_model_map[language]
                        self.transformer_pipelines[language] = pipeline(
                            "ner",
                            model=model_name,
                            tokenizer=model_name,
                            aggregation_strategy="simple",
                            device=0 if torch.cuda.is_available() else -1
                        )
                        logger.info("Loaded transformer model: {} for {}", model_name, language.value)
                    except Exception as e:
                        logger.warning("Failed to load transformer model for {}: {}", language.value, e)
    
    def _load_legal_patterns(self) -> None:
        """Load legal-specific regex patterns"""
        # Court patterns
        court_patterns = {
            LanguageCode.EN: [
                r'\b(?:Supreme Court|Court of Appeals|District Court|Circuit Court|High Court|Magistrate Court)\b',
                r'\b(?:U\.S\.|United States) (?:Supreme Court|Court of Appeals|District Court)\b',
                r'\b(?:Federal|State|County|Municipal) Court\b',
                r'\b[A-Z][a-z]+\s+(?:County|District)\s+Court\b'
            ],
            LanguageCode.ES: [
                r'\b(?:Tribunal Supremo|Audiencia Nacional|Tribunal Superior|Juzgado)\b',
                r'\b(?:Corte|Tribunal) (?:Suprema?|Superior|de Justicia)\b'
            ],
            LanguageCode.FR: [
                r'\b(?:Cour de cassation|Conseil d\'État|Tribunal|Cour d\'appel)\b',
                r'\b(?:Tribunal de grande instance|Tribunal correctionnel)\b'
            ]
        }
        
        # Statute patterns
        statute_patterns = {
            LanguageCode.EN: [
                r'\b(?:USC|U\.S\.C\.|CFR)\s+(?:§\s*)?(\d+)(?:[-–]\d+)*\b',
                r'\b\d+\s+U\.S\.C\.?\s+(?:§\s*)?(\d+)\b',
                r'\b(?:Section|Sec\.|§)\s*(\d+(?:\.\d+)*)\b',
                r'\b(?:Title|Tit\.)\s+(\d+)\b',
                r'\b(?:Rule|Fed\.?\s*R\.)\s*(\d+(?:\.\d+)*)\b'
            ],
            LanguageCode.ES: [
                r'\b(?:Artículo|Art\.|Ley)\s+(\d+(?:\.\d+)*)\b',
                r'\b(?:Código Civil|Código Penal|CP|CC)\s+(?:Art\.|Artículo)\s*(\d+)\b'
            ],
            LanguageCode.FR: [
                r'\b(?:Article|Art\.|Code)\s+(\d+(?:-\d+)*)\b',
                r'\b(?:Code civil|Code pénal|CP|CC)\s+(?:Art\.|Article)\s*(\d+)\b'
            ]
        }
        
        # Case name patterns
        case_patterns = {
            LanguageCode.EN: [
                r'\b[A-Z][a-zA-Z\s,\.]+\s+v\.?\s+[A-Z][a-zA-Z\s,\.]+\b',
                r'\b(?:In re|Ex parte)\s+[A-Z][a-zA-Z\s,\.]+\b',
                r'\b[A-Z][a-zA-Z\s&,\.]+\s+(?:Inc\.|Corp\.|LLC|Ltd\.)\s+v\.?\s+[A-Z][a-zA-Z\s,\.]+\b'
            ],
            LanguageCode.ES: [
                r'\b[A-Z][a-zA-Z\s,\.]+\s+(?:contra|c\.)\s+[A-Z][a-zA-Z\s,\.]+\b',
                r'\b(?:Sentencia|STS|SAP)\s+\d+/\d{4}\b'
            ],
            LanguageCode.FR: [
                r'\b[A-Z][a-zA-Z\s,\.]+\s+(?:contre|c\.)\s+[A-Z][a-zA-Z\s,\.]+\b',
                r'\b(?:Arrêt|Cass\.|CE)\s+\d+\s+\w+\s+\d{4}\b'
            ]
        }
        
        # Contract patterns
        contract_patterns = {
            LanguageCode.EN: [
                r'\b(?:Agreement|Contract|Lease|License|NDA|MOU|LOI)\b',
                r'\b(?:Employment|Service|Purchase|Sales|Non-disclosure)\s+Agreement\b',
                r'\b(?:Memorandum of Understanding|Letter of Intent)\b'
            ],
            LanguageCode.ES: [
                r'\b(?:Contrato|Acuerdo|Convenio|Arrendamiento)\b',
                r'\b(?:Contrato de|Acuerdo de)\s+\w+\b'
            ],
            LanguageCode.FR: [
                r'\b(?:Contrat|Accord|Convention|Bail|Licence)\b',
                r'\b(?:Contrat de|Accord de)\s+\w+\b'
            ]
        }
        
        self.legal_patterns = {
            EntityType.COURT: court_patterns,
            EntityType.STATUTE: statute_patterns, 
            EntityType.CASE_NAME: case_patterns,
            EntityType.CONTRACT: contract_patterns
        }
    
    def _load_legal_terminology(self) -> None:
        """Load legal terminology dictionaries"""
        # Load from file if specified
        if self.config.legal_terminology_file:
            try:
                terminology_path = Path(self.config.legal_terminology_file)
                if terminology_path.exists():
                    import json
                    with open(terminology_path, 'r', encoding='utf-8') as f:
                        self.legal_terminology = json.load(f)
                    logger.info("Loaded legal terminology from: {}", terminology_path)
                    return
            except Exception as e:
                logger.warning("Failed to load legal terminology file: {}", e)
        
        # Default legal terminology
        self.legal_terminology = {
            LanguageCode.EN.value: {
                "legal_entities": [
                    "attorney", "lawyer", "counsel", "barrister", "solicitor",
                    "judge", "magistrate", "justice", "plaintiff", "defendant", 
                    "prosecutor", "district attorney", "public defender",
                    "jury", "juror", "witness", "expert witness", "court reporter",
                    "bailiff", "clerk", "sheriff", "marshal"
                ],
                "legal_documents": [
                    "brief", "motion", "pleading", "complaint", "answer", "counterclaim",
                    "subpoena", "warrant", "injunction", "decree", "order", "judgment",
                    "verdict", "sentence", "affidavit", "deposition", "testimony",
                    "evidence", "exhibit", "discovery", "interrogatory"
                ],
                "legal_concepts": [
                    "due process", "probable cause", "reasonable doubt", "burden of proof",
                    "habeas corpus", "res judicata", "stare decisis", "jurisdiction",
                    "venue", "standing", "liability", "negligence", "tort", "contract",
                    "statute of limitations", "chain of custody", "miranda rights"
                ]
            },
            LanguageCode.ES.value: {
                "legal_entities": [
                    "abogado", "letrado", "juez", "magistrado", "fiscal", "procurador",
                    "demandante", "demandado", "acusado", "testigo", "perito",
                    "secretario judicial", "ujier", "alguacil"
                ],
                "legal_documents": [
                    "demanda", "contestación", "recurso", "sentencia", "auto", "decreto",
                    "citación", "emplazamiento", "mandamiento", "orden", "resolución",
                    "testimonio", "declaración", "prueba", "diligencia"
                ],
                "legal_concepts": [
                    "debido proceso", "presunción de inocencia", "carga de la prueba",
                    "cosa juzgada", "jurisdicción", "competencia", "responsabilidad",
                    "negligencia", "prescripción", "cadena de custodia"
                ]
            },
            LanguageCode.FR.value: {
                "legal_entities": [
                    "avocat", "juge", "magistrat", "procureur", "huissier", "greffier",
                    "demandeur", "défendeur", "accusé", "témoin", "expert", "partie"
                ],
                "legal_documents": [
                    "assignation", "requête", "mémoire", "jugement", "arrêt", "ordonnance",
                    "citation", "convocation", "signification", "exploit", "procès-verbal"
                ],
                "legal_concepts": [
                    "due process", "présomption d'innocence", "charge de la preuve",
                    "chose jugée", "juridiction", "compétence", "responsabilité",
                    "négligence", "prescription", "chaîne de possession"
                ]
            }
        }
    
    def extract_entities(self, text: str, document_id: str, 
                        language: LanguageCode) -> List[Entity]:
        """
        Extract standard named entities from text
        
        Args:
            text: Input text
            document_id: Document identifier
            language: Language of the text
            
        Returns:
            List of extracted entities
        """
        entities = []
        
        try:
            # SpaCy-based extraction
            if language in self.spacy_models:
                spacy_entities = self._extract_spacy_entities(text, document_id, language)
                entities.extend(spacy_entities)
            
            # Transformer-based extraction
            if self.config.use_transformers and language in self.transformer_pipelines:
                transformer_entities = self._extract_transformer_entities(text, document_id, language)
                entities.extend(transformer_entities)
            
            # Remove duplicates and merge overlapping entities
            entities = self._merge_overlapping_entities(entities)
            
            logger.info(
                "Extracted {} standard entities from document {} ({})",
                len(entities), document_id, language.value
            )
            
        except Exception as e:
            logger.error("Error extracting standard entities: {}", e)
        
        return entities
    
    def extract_legal_entities(self, text: str, document_id: str,
                              language: LanguageCode) -> List[Entity]:
        """
        Extract legal-specific entities from text
        
        Args:
            text: Input text
            document_id: Document identifier  
            language: Language of the text
            
        Returns:
            List of extracted legal entities
        """
        entities = []
        
        try:
            # Pattern-based extraction
            pattern_entities = self._extract_pattern_entities(text, document_id, language)
            entities.extend(pattern_entities)
            
            # Terminology-based extraction
            terminology_entities = self._extract_terminology_entities(text, document_id, language)
            entities.extend(terminology_entities)
            
            # Legal model extraction (if available)
            if self.config.legal_model_path:
                legal_model_entities = self._extract_legal_model_entities(text, document_id, language)
                entities.extend(legal_model_entities)
            
            # Remove duplicates and merge overlapping entities
            entities = self._merge_overlapping_entities(entities)
            
            logger.info(
                "Extracted {} legal entities from document {} ({})",
                len(entities), document_id, language.value
            )
            
        except Exception as e:
            logger.error("Error extracting legal entities: {}", e)
        
        return entities
    
    def _extract_spacy_entities(self, text: str, document_id: str,
                               language: LanguageCode) -> List[Entity]:
        """Extract entities using spaCy"""
        entities = []
        
        try:
            nlp = self.spacy_models[language]
            doc = nlp(text)
            
            for ent in doc.ents:
                # Map spaCy labels to our entity types
                entity_type = self._map_spacy_label(ent.label_)
                if entity_type and entity_type in self.config.entity_types:
                    # Get context
                    context = self._get_entity_context(text, ent.start_char, ent.end_char)
                    
                    entity = Entity(
                        text=ent.text,
                        entity_type=entity_type,
                        start_pos=ent.start_char,
                        end_pos=ent.end_char,
                        confidence=0.8,  # Default confidence for spaCy
                        language=language,
                        document_id=document_id,
                        context=context,
                        metadata={"source": "spacy", "label": ent.label_}
                    )
                    entities.append(entity)
                    
        except Exception as e:
            logger.error("Error in spaCy extraction: {}", e)
        
        return entities
    
    def _extract_transformer_entities(self, text: str, document_id: str,
                                    language: LanguageCode) -> List[Entity]:
        """Extract entities using transformer models"""
        entities = []
        
        try:
            pipeline = self.transformer_pipelines[language]
            
            # Process text in chunks to handle length limits
            max_length = 512
            chunks = [text[i:i+max_length] for i in range(0, len(text), max_length)]
            
            offset = 0
            for chunk in chunks:
                results = pipeline(chunk)
                
                for result in results:
                    entity_type = self._map_transformer_label(result['entity_group'])
                    if entity_type and entity_type in self.config.entity_types:
                        # Calculate absolute positions
                        start_pos = offset + result['start']
                        end_pos = offset + result['end']
                        
                        # Get context
                        context = self._get_entity_context(text, start_pos, end_pos)
                        
                        entity = Entity(
                            text=result['word'],
                            entity_type=entity_type,
                            start_pos=start_pos,
                            end_pos=end_pos,
                            confidence=result['score'],
                            language=language,
                            document_id=document_id,
                            context=context,
                            metadata={"source": "transformer", "label": result['entity_group']}
                        )
                        entities.append(entity)
                
                offset += len(chunk)
                
        except Exception as e:
            logger.error("Error in transformer extraction: {}", e)
        
        return entities
    
    def _extract_pattern_entities(self, text: str, document_id: str,
                                 language: LanguageCode) -> List[Entity]:
        """Extract entities using regex patterns"""
        entities = []
        
        try:
            for entity_type, language_patterns in self.legal_patterns.items():
                if entity_type not in self.config.entity_types:
                    continue
                
                patterns = language_patterns.get(language, [])
                for pattern in patterns:
                    matches = re.finditer(pattern, text, re.IGNORECASE)
                    
                    for match in matches:
                        start_pos = match.start()
                        end_pos = match.end()
                        entity_text = match.group()
                        
                        # Skip very short matches
                        if len(entity_text.strip()) < 3:
                            continue
                        
                        # Get context
                        context = self._get_entity_context(text, start_pos, end_pos)
                        
                        entity = Entity(
                            text=entity_text.strip(),
                            entity_type=entity_type,
                            start_pos=start_pos,
                            end_pos=end_pos,
                            confidence=0.7,  # Lower confidence for pattern matching
                            language=language,
                            document_id=document_id,
                            context=context,
                            metadata={"source": "pattern", "pattern": pattern}
                        )
                        entities.append(entity)
                        
        except Exception as e:
            logger.error("Error in pattern extraction: {}", e)
        
        return entities
    
    def _extract_terminology_entities(self, text: str, document_id: str,
                                    language: LanguageCode) -> List[Entity]:
        """Extract entities using legal terminology dictionaries"""
        entities = []
        
        try:
            terminology = self.legal_terminology.get(language.value, {})
            
            # Combine all terminology lists
            all_terms = []
            for category, terms in terminology.items():
                all_terms.extend(terms)
            
            # Case-insensitive search for terms
            text_lower = text.lower()
            for term in all_terms:
                term_lower = term.lower()
                
                # Find all occurrences
                start = 0
                while True:
                    pos = text_lower.find(term_lower, start)
                    if pos == -1:
                        break
                    
                    # Check word boundaries
                    if (pos == 0 or not text[pos-1].isalnum()) and \
                       (pos + len(term) >= len(text) or not text[pos + len(term)].isalnum()):
                        
                        start_pos = pos
                        end_pos = pos + len(term)
                        entity_text = text[start_pos:end_pos]
                        
                        # Get context
                        context = self._get_entity_context(text, start_pos, end_pos)
                        
                        entity = Entity(
                            text=entity_text,
                            entity_type=EntityType.LEGAL_ENTITY,
                            start_pos=start_pos,
                            end_pos=end_pos,
                            confidence=0.6,  # Lower confidence for terminology matching
                            language=language,
                            document_id=document_id,
                            context=context,
                            metadata={"source": "terminology", "term": term}
                        )
                        entities.append(entity)
                    
                    start = pos + 1
                    
        except Exception as e:
            logger.error("Error in terminology extraction: {}", e)
        
        return entities
    
    def _extract_legal_model_entities(self, text: str, document_id: str,
                                    language: LanguageCode) -> List[Entity]:
        """Extract entities using specialized legal models"""
        entities = []
        
        # Placeholder for legal model extraction
        # This would load and use a specialized legal NER model
        logger.info("Legal model extraction not implemented yet")
        
        return entities
    
    def _map_spacy_label(self, label: str) -> Optional[EntityType]:
        """Map spaCy entity labels to our EntityType enum"""
        label_mapping = {
            "PERSON": EntityType.PERSON,
            "ORG": EntityType.ORGANIZATION,
            "GPE": EntityType.LOCATION,
            "LOC": EntityType.LOCATION,
            "DATE": EntityType.DATE,
            "TIME": EntityType.TIME,
            "EVENT": EntityType.EVENT,
            "LAW": EntityType.LEGAL_ENTITY,
            "NORP": EntityType.ORGANIZATION  # Nationalities, religious/political groups
        }
        return label_mapping.get(label)
    
    def _map_transformer_label(self, label: str) -> Optional[EntityType]:
        """Map transformer entity labels to our EntityType enum"""
        # Clean up common transformer label prefixes
        clean_label = label.replace("B-", "").replace("I-", "")
        
        label_mapping = {
            "PER": EntityType.PERSON,
            "PERSON": EntityType.PERSON,
            "ORG": EntityType.ORGANIZATION,
            "LOC": EntityType.LOCATION,
            "LOCATION": EntityType.LOCATION,
            "MISC": EntityType.LEGAL_ENTITY,
            "DATE": EntityType.DATE,
            "TIME": EntityType.TIME
        }
        return label_mapping.get(clean_label)
    
    def _get_entity_context(self, text: str, start_pos: int, end_pos: int) -> str:
        """Get surrounding context for an entity"""
        if not self.config.include_context:
            return ""
        
        window = self.config.context_window
        context_start = max(0, start_pos - window)
        context_end = min(len(text), end_pos + window)
        
        return text[context_start:context_end]
    
    def _merge_overlapping_entities(self, entities: List[Entity]) -> List[Entity]:
        """Merge overlapping entities, keeping the one with higher confidence"""
        if not entities:
            return entities
        
        # Sort by start position
        sorted_entities = sorted(entities, key=lambda e: e.start_pos)
        merged = []
        
        for entity in sorted_entities:
            # Check for overlap with last merged entity
            if merged and self._entities_overlap(merged[-1], entity):
                # Keep the entity with higher confidence
                if entity.confidence > merged[-1].confidence:
                    merged[-1] = entity
            else:
                merged.append(entity)
        
        return merged
    
    def _entities_overlap(self, entity1: Entity, entity2: Entity) -> bool:
        """Check if two entities overlap"""
        return (entity1.start_pos < entity2.end_pos and 
                entity2.start_pos < entity1.end_pos)
    
    def get_supported_languages(self) -> List[LanguageCode]:
        """Get list of supported languages"""
        return list(self.spacy_models.keys())
    
    def validate_language_support(self, language: LanguageCode) -> bool:
        """Check if a language is supported"""
        return language in self.spacy_models or language in self.transformer_pipelines