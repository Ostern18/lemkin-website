"""
Text-based PII redaction using Named Entity Recognition (NER).

This module provides specialized NER-based PII detection and redaction
for text documents with support for multiple languages.
"""

import re
import hashlib
import time
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

import spacy
from spacy.matcher import Matcher
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification

from .core import (
    PIIEntity, EntityType, ConfidenceLevel, RedactionConfig, 
    RedactionResult, RedactionType
)
from loguru import logger


class TextRedactor:
    """Text-based PII redaction using NER and pattern matching."""
    
    def __init__(self, config: RedactionConfig):
        """Initialize text redactor with configuration."""
        self.config = config
        self.logger = logger
        
        # Initialize NLP models
        self._spacy_nlp = None
        self._transformers_pipeline = None
        self._matcher = None
        
        # Predefined patterns for common PII
        self.patterns = {
            EntityType.EMAIL: r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            EntityType.PHONE: r'(\+\d{1,3}[-.\s]?)?\(?\d{1,4}\)?[-.\s]?\d{1,4}[-.\s]?\d{1,9}',
            EntityType.SSN: r'\b\d{3}-?\d{2}-?\d{4}\b',
            EntityType.CREDIT_CARD: r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',
            EntityType.IP_ADDRESS: r'\b(?:\d{1,3}\.){3}\d{1,3}\b',
        }
        
        # Update with custom patterns
        self.patterns.update(self.config.custom_patterns)
        
        self.logger.info("TextRedactor initialized")
    
    @property
    def spacy_nlp(self):
        """Lazy loading of spaCy NLP pipeline."""
        if self._spacy_nlp is None:
            try:
                model_name = f"{self.config.language}_core_web_sm"
                self._spacy_nlp = spacy.load(model_name)
                self.logger.info(f"Loaded spaCy model: {model_name}")
            except OSError:
                # Fallback to English if language model not available
                self.logger.warning(f"Language model for {self.config.language} not found, using English")
                self._spacy_nlp = spacy.load("en_core_web_sm")
                
        return self._spacy_nlp
    
    @property
    def transformers_pipeline(self):
        """Lazy loading of Transformers NER pipeline."""
        if self._transformers_pipeline is None:
            try:
                self._transformers_pipeline = pipeline(
                    "ner",
                    model="dbmdz/bert-large-cased-finetuned-conll03-english",
                    tokenizer="dbmdz/bert-large-cased-finetuned-conll03-english",
                    aggregation_strategy="simple"
                )
                self.logger.info("Loaded Transformers NER pipeline")
            except Exception as e:
                self.logger.warning(f"Failed to load Transformers pipeline: {e}")
                
        return self._transformers_pipeline
    
    @property
    def matcher(self):
        """Lazy loading of spaCy Matcher for pattern matching."""
        if self._matcher is None:
            self._matcher = Matcher(self.spacy_nlp.vocab)
            
            # Add pattern matching rules
            email_pattern = [{"LIKE_EMAIL": True}]
            self._matcher.add("EMAIL", [email_pattern])
            
            self.logger.info("Initialized spaCy Matcher")
            
        return self._matcher
    
    def detect_entities_spacy(self, text: str) -> List[PIIEntity]:
        """Detect PII entities using spaCy NER."""
        entities = []
        
        try:
            doc = self.spacy_nlp(text)
            
            for ent in doc.ents:
                # Map spaCy labels to our entity types
                entity_type = self._map_spacy_label(ent.label_)
                
                if entity_type and entity_type in self.config.entity_types:
                    confidence = 0.8  # spaCy doesn't provide confidence scores
                    
                    entity = PIIEntity(
                        entity_type=entity_type,
                        text=ent.text,
                        start_pos=ent.start_char,
                        end_pos=ent.end_char,
                        confidence=confidence,
                        confidence_level=self._get_confidence_level(confidence),
                        metadata={"source": "spacy", "label": ent.label_}
                    )
                    entities.append(entity)
                    
        except Exception as e:
            self.logger.error(f"spaCy NER failed: {e}")
            
        return entities
    
    def detect_entities_transformers(self, text: str) -> List[PIIEntity]:
        """Detect PII entities using Transformers NER."""
        entities = []
        
        if not self.transformers_pipeline:
            return entities
        
        try:
            # Process in chunks to handle long texts
            max_length = 512
            text_chunks = [text[i:i+max_length] for i in range(0, len(text), max_length)]
            
            offset = 0
            for chunk in text_chunks:
                results = self.transformers_pipeline(chunk)
                
                for result in results:
                    entity_type = self._map_transformers_label(result['entity_group'])
                    
                    if entity_type and entity_type in self.config.entity_types:
                        entity = PIIEntity(
                            entity_type=entity_type,
                            text=result['word'],
                            start_pos=result['start'] + offset,
                            end_pos=result['end'] + offset,
                            confidence=result['score'],
                            confidence_level=self._get_confidence_level(result['score']),
                            metadata={"source": "transformers", "label": result['entity_group']}
                        )
                        entities.append(entity)
                
                offset += len(chunk)
                
        except Exception as e:
            self.logger.error(f"Transformers NER failed: {e}")
            
        return entities
    
    def detect_entities_patterns(self, text: str) -> List[PIIEntity]:
        """Detect PII entities using regex patterns."""
        entities = []
        
        for entity_type, pattern in self.patterns.items():
            if entity_type not in self.config.entity_types:
                continue
                
            try:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                
                for match in matches:
                    confidence = 0.9  # High confidence for pattern matching
                    
                    entity = PIIEntity(
                        entity_type=entity_type,
                        text=match.group(),
                        start_pos=match.start(),
                        end_pos=match.end(),
                        confidence=confidence,
                        confidence_level=self._get_confidence_level(confidence),
                        metadata={"source": "pattern", "pattern": pattern}
                    )
                    entities.append(entity)
                    
            except Exception as e:
                self.logger.error(f"Pattern matching for {entity_type} failed: {e}")
                
        return entities
    
    def merge_entities(self, entity_lists: List[List[PIIEntity]]) -> List[PIIEntity]:
        """Merge entities from different detection methods, removing duplicates."""
        all_entities = []
        for entities in entity_lists:
            all_entities.extend(entities)
        
        # Sort by position
        all_entities.sort(key=lambda x: (x.start_pos, x.end_pos))
        
        # Remove overlapping entities, keeping highest confidence
        merged = []
        for entity in all_entities:
            # Check for overlap with existing entities
            overlaps = False
            for existing in merged:
                if self._entities_overlap(entity, existing):
                    overlaps = True
                    # Replace if new entity has higher confidence
                    if entity.confidence > existing.confidence:
                        merged.remove(existing)
                        merged.append(entity)
                    break
            
            if not overlaps:
                merged.append(entity)
        
        return merged
    
    def apply_redaction(self, text: str, entities: List[PIIEntity]) -> Tuple[str, List[PIIEntity]]:
        """Apply redaction to text based on detected entities."""
        redacted_text = text
        redacted_entities = []
        
        # Sort entities by position (reverse order to maintain positions)
        entities.sort(key=lambda x: x.start_pos, reverse=True)
        
        for entity in entities:
            # Check if entity meets confidence threshold
            if entity.confidence < self.config.min_confidence:
                continue
            
            # Get redaction method for this entity type
            redaction_method = self.config.redaction_methods.get(
                entity.entity_type, RedactionType.MASK
            )
            
            # Generate replacement text
            replacement = self._generate_replacement(entity, redaction_method)
            entity.replacement = replacement
            
            # Apply redaction
            redacted_text = (
                redacted_text[:entity.start_pos] + 
                replacement + 
                redacted_text[entity.end_pos:]
            )
            
            redacted_entities.append(entity)
        
        return redacted_text, redacted_entities
    
    def redact(self, text: str, output_path: Optional[Path] = None) -> RedactionResult:
        """
        Main redaction method for text content.
        
        Args:
            text: Text content to redact
            output_path: Optional path to save redacted text
            
        Returns:
            RedactionResult with processing details
        """
        start_time = time.time()
        
        # Generate content hash for integrity
        content_hash = hashlib.sha256(text.encode()).hexdigest()
        
        # Detect entities using multiple methods
        spacy_entities = self.detect_entities_spacy(text)
        transformers_entities = self.detect_entities_transformers(text)
        pattern_entities = self.detect_entities_patterns(text)
        
        # Merge and deduplicate entities
        all_entities = self.merge_entities([
            spacy_entities, 
            transformers_entities, 
            pattern_entities
        ])
        
        # Apply redaction
        redacted_text, redacted_entities = self.apply_redaction(text, all_entities)
        
        # Save redacted text if output path provided
        if output_path:
            output_path.write_text(redacted_text, encoding='utf-8')
        
        # Calculate statistics
        processing_time = time.time() - start_time
        confidence_scores = self._calculate_confidence_scores(all_entities)
        
        # Create result
        result = RedactionResult(
            original_content_hash=content_hash,
            content_type="text",
            entities_detected=all_entities,
            entities_redacted=redacted_entities,
            total_entities=len(all_entities),
            redacted_count=len(redacted_entities),
            confidence_scores=confidence_scores,
            processing_time=processing_time,
            config_used=self.config,
            redacted_content_path=str(output_path) if output_path else None
        )
        
        return result
    
    def _map_spacy_label(self, label: str) -> Optional[EntityType]:
        """Map spaCy entity labels to our EntityType enum."""
        mapping = {
            "PERSON": EntityType.PERSON,
            "PER": EntityType.PERSON,
            "ORG": EntityType.ORGANIZATION,
            "GPE": EntityType.LOCATION,
            "LOC": EntityType.LOCATION,
            "DATE": EntityType.DATE,
            "TIME": EntityType.DATE,
        }
        return mapping.get(label.upper())
    
    def _map_transformers_label(self, label: str) -> Optional[EntityType]:
        """Map Transformers entity labels to our EntityType enum."""
        mapping = {
            "PER": EntityType.PERSON,
            "PERSON": EntityType.PERSON,
            "ORG": EntityType.ORGANIZATION,
            "LOC": EntityType.LOCATION,
            "MISC": EntityType.CUSTOM,
        }
        return mapping.get(label.upper())
    
    def _get_confidence_level(self, confidence: float) -> ConfidenceLevel:
        """Convert numeric confidence to confidence level."""
        if confidence >= 0.9:
            return ConfidenceLevel.VERY_HIGH
        elif confidence >= 0.75:
            return ConfidenceLevel.HIGH
        elif confidence >= 0.5:
            return ConfidenceLevel.MEDIUM
        else:
            return ConfidenceLevel.LOW
    
    def _entities_overlap(self, entity1: PIIEntity, entity2: PIIEntity) -> bool:
        """Check if two entities overlap in text position."""
        return not (entity1.end_pos <= entity2.start_pos or entity2.end_pos <= entity1.start_pos)
    
    def _generate_replacement(self, entity: PIIEntity, method: RedactionType) -> str:
        """Generate replacement text based on redaction method."""
        if method == RedactionType.MASK:
            return "*" * len(entity.text)
        
        elif method == RedactionType.REPLACE:
            replacements = {
                EntityType.PERSON: "[PERSON]",
                EntityType.ORGANIZATION: "[ORGANIZATION]",
                EntityType.LOCATION: "[LOCATION]",
                EntityType.EMAIL: "[EMAIL]",
                EntityType.PHONE: "[PHONE]",
                EntityType.SSN: "[SSN]",
                EntityType.CREDIT_CARD: "[CREDIT_CARD]",
                EntityType.IP_ADDRESS: "[IP_ADDRESS]",
                EntityType.DATE: "[DATE]",
                EntityType.ADDRESS: "[ADDRESS]",
            }
            return replacements.get(entity.entity_type, "[REDACTED]")
        
        elif method == RedactionType.DELETE:
            return ""
        
        elif method == RedactionType.ANONYMIZE:
            # Generate synthetic replacement
            anonymized = {
                EntityType.PERSON: "John Doe",
                EntityType.ORGANIZATION: "Organization Inc.",
                EntityType.LOCATION: "City, State",
                EntityType.EMAIL: "example@email.com",
                EntityType.PHONE: "555-0000",
            }
            return anonymized.get(entity.entity_type, "[ANONYMIZED]")
        
        else:
            return "[REDACTED]"
    
    def _calculate_confidence_scores(self, entities: List[PIIEntity]) -> Dict[str, float]:
        """Calculate average confidence scores by entity type."""
        scores = {}
        type_counts = {}
        
        for entity in entities:
            entity_type = entity.entity_type.value
            if entity_type not in scores:
                scores[entity_type] = 0.0
                type_counts[entity_type] = 0
            
            scores[entity_type] += entity.confidence
            type_counts[entity_type] += 1
        
        # Calculate averages
        for entity_type in scores:
            if type_counts[entity_type] > 0:
                scores[entity_type] /= type_counts[entity_type]
        
        return scores