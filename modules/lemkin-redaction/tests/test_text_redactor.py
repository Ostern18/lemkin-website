import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import hashlib

from lemkin_redaction.text_redactor import TextRedactor
from lemkin_redaction.core import (
    RedactionConfig,
    PIIEntity,
    EntityType,
    RedactionType,
    ConfidenceLevel
)

@pytest.fixture
def sample_config():
    """Sample redaction configuration for text redaction"""
    return RedactionConfig(
        entity_types=[EntityType.PERSON, EntityType.EMAIL, EntityType.PHONE],
        redaction_methods={
            EntityType.PERSON: RedactionType.REPLACE,
            EntityType.EMAIL: RedactionType.MASK,
            EntityType.PHONE: RedactionType.MASK
        },
        min_confidence=0.7,
        language="en",
        custom_patterns={
            "custom_id": r"ID-\d{6}"
        }
    )

@pytest.fixture
def sample_text():
    """Sample text with various PII entities"""
    return "John Smith contacted jane.doe@email.com at 555-123-4567 regarding case ID-123456."

@pytest.fixture
def sample_entities():
    """Sample PII entities for testing"""
    return [
        PIIEntity(
            entity_type=EntityType.PERSON,
            text="John Smith",
            start_pos=0,
            end_pos=10,
            confidence=0.95,
            confidence_level=ConfidenceLevel.VERY_HIGH
        ),
        PIIEntity(
            entity_type=EntityType.EMAIL,
            text="jane.doe@email.com",
            start_pos=21,
            end_pos=40,
            confidence=0.99,
            confidence_level=ConfidenceLevel.VERY_HIGH
        ),
        PIIEntity(
            entity_type=EntityType.PHONE,
            text="555-123-4567",
            start_pos=44,
            end_pos=56,
            confidence=0.88,
            confidence_level=ConfidenceLevel.HIGH
        )
    ]

class TestTextRedactor:
    
    def test_initialization(self, sample_config):
        """Test TextRedactor initialization"""
        redactor = TextRedactor(sample_config)
        
        assert redactor.config == sample_config
        assert redactor._spacy_nlp is None  # Lazy loaded
        assert redactor._transformers_pipeline is None  # Lazy loaded
        assert redactor._matcher is None  # Lazy loaded
        
        # Check that custom patterns are included
        assert "custom_id" in redactor.patterns
        assert redactor.patterns["custom_id"] == r"ID-\d{6}"
    
    def test_predefined_patterns(self, sample_config):
        """Test predefined regex patterns"""
        redactor = TextRedactor(sample_config)
        
        # Test email pattern
        assert EntityType.EMAIL in redactor.patterns
        email_pattern = redactor.patterns[EntityType.EMAIL]
        import re
        assert re.search(email_pattern, "test@example.com")
        assert re.search(email_pattern, "user.name+tag@domain.co.uk")
        
        # Test phone pattern
        assert EntityType.PHONE in redactor.patterns
        phone_pattern = redactor.patterns[EntityType.PHONE]
        assert re.search(phone_pattern, "555-123-4567")
        assert re.search(phone_pattern, "(555) 123-4567")
        assert re.search(phone_pattern, "+1-555-123-4567")
        
        # Test SSN pattern
        assert EntityType.SSN in redactor.patterns
        ssn_pattern = redactor.patterns[EntityType.SSN]
        assert re.search(ssn_pattern, "123-45-6789")
        assert re.search(ssn_pattern, "123456789")
    
    @patch('lemkin_redaction.text_redactor.spacy.load')
    def test_spacy_nlp_loading(self, mock_spacy_load, sample_config):
        """Test spaCy NLP model loading"""
        mock_nlp = Mock()
        mock_spacy_load.return_value = mock_nlp
        
        redactor = TextRedactor(sample_config)
        nlp = redactor.spacy_nlp
        
        assert nlp is mock_nlp
        mock_spacy_load.assert_called_once_with("en_core_web_sm")
        
        # Second access should return cached instance
        nlp2 = redactor.spacy_nlp
        assert nlp2 is mock_nlp
        assert mock_spacy_load.call_count == 1
    
    @patch('lemkin_redaction.text_redactor.spacy.load')
    def test_spacy_nlp_fallback_to_english(self, mock_spacy_load, sample_config):
        """Test spaCy fallback to English when language model not available"""
        # First call raises OSError, second call succeeds
        mock_spacy_load.side_effect = [OSError("Model not found"), Mock()]
        
        sample_config.language = "es"  # Spanish
        redactor = TextRedactor(sample_config)
        
        nlp = redactor.spacy_nlp
        
        assert mock_spacy_load.call_count == 2
        mock_spacy_load.assert_any_call("es_core_web_sm")
        mock_spacy_load.assert_any_call("en_core_web_sm")
    
    @patch('lemkin_redaction.text_redactor.pipeline')
    def test_transformers_pipeline_loading(self, mock_pipeline, sample_config):
        """Test Transformers NER pipeline loading"""
        mock_ner_pipeline = Mock()
        mock_pipeline.return_value = mock_ner_pipeline
        
        redactor = TextRedactor(sample_config)
        pipeline_obj = redactor.transformers_pipeline
        
        assert pipeline_obj is mock_ner_pipeline
        mock_pipeline.assert_called_once_with(
            "ner",
            model="dbmdz/bert-large-cased-finetuned-conll03-english",
            tokenizer="dbmdz/bert-large-cased-finetuned-conll03-english",
            aggregation_strategy="simple"
        )
    
    @patch('lemkin_redaction.text_redactor.pipeline')
    def test_transformers_pipeline_loading_failure(self, mock_pipeline, sample_config):
        """Test Transformers pipeline loading failure handling"""
        mock_pipeline.side_effect = Exception("Model loading failed")
        
        redactor = TextRedactor(sample_config)
        pipeline_obj = redactor.transformers_pipeline
        
        assert pipeline_obj is None
    
    def test_detect_entities_spacy(self, sample_config, sample_text):
        """Test spaCy entity detection"""
        redactor = TextRedactor(sample_config)
        
        # Mock spaCy NLP and entities
        mock_ent1 = Mock()
        mock_ent1.text = "John Smith"
        mock_ent1.label_ = "PERSON"
        mock_ent1.start_char = 0
        mock_ent1.end_char = 10
        
        mock_ent2 = Mock()
        mock_ent2.text = "Organization Inc"
        mock_ent2.label_ = "ORG"
        mock_ent2.start_char = 60
        mock_ent2.end_char = 76
        
        mock_doc = Mock()
        mock_doc.ents = [mock_ent1, mock_ent2]
        
        with patch.object(redactor, 'spacy_nlp') as mock_nlp:
            mock_nlp.return_value = mock_doc
            
            entities = redactor.detect_entities_spacy(sample_text)
        
        assert len(entities) == 2
        
        # Check first entity (PERSON)
        person_entity = entities[0]
        assert person_entity.entity_type == EntityType.PERSON
        assert person_entity.text == "John Smith"
        assert person_entity.start_pos == 0
        assert person_entity.end_pos == 10
        assert person_entity.confidence == 0.8  # Default spaCy confidence
        assert person_entity.metadata["source"] == "spacy"
        
        # Check second entity (ORG)
        org_entity = entities[1]
        assert org_entity.entity_type == EntityType.ORGANIZATION
        assert org_entity.text == "Organization Inc"
    
    def test_detect_entities_spacy_exception_handling(self, sample_config, sample_text):
        """Test spaCy entity detection exception handling"""
        redactor = TextRedactor(sample_config)
        
        with patch.object(redactor, 'spacy_nlp') as mock_nlp:
            mock_nlp.side_effect = Exception("Processing failed")
            
            entities = redactor.detect_entities_spacy(sample_text)
        
        assert entities == []
    
    def test_detect_entities_transformers(self, sample_config, sample_text):
        """Test Transformers entity detection"""
        redactor = TextRedactor(sample_config)
        
        # Mock Transformers pipeline results
        mock_results = [
            {
                'entity_group': 'PER',
                'word': 'John Smith',
                'start': 0,
                'end': 10,
                'score': 0.95
            },
            {
                'entity_group': 'ORG', 
                'word': 'Organization',
                'start': 60,
                'end': 72,
                'score': 0.88
            }
        ]
        
        mock_pipeline = Mock()
        mock_pipeline.return_value = mock_results
        
        with patch.object(redactor, 'transformers_pipeline', mock_pipeline):
            entities = redactor.detect_entities_transformers(sample_text)
        
        assert len(entities) == 2
        
        # Check first entity
        person_entity = entities[0]
        assert person_entity.entity_type == EntityType.PERSON
        assert person_entity.text == "John Smith"
        assert person_entity.confidence == 0.95
        assert person_entity.metadata["source"] == "transformers"
    
    def test_detect_entities_transformers_no_pipeline(self, sample_config, sample_text):
        """Test Transformers detection when pipeline is None"""
        redactor = TextRedactor(sample_config)
        
        with patch.object(redactor, 'transformers_pipeline', None):
            entities = redactor.detect_entities_transformers(sample_text)
        
        assert entities == []
    
    def test_detect_entities_patterns(self, sample_config, sample_text):
        """Test pattern-based entity detection"""
        redactor = TextRedactor(sample_config)
        
        entities = redactor.detect_entities_patterns(sample_text)
        
        # Should detect email and phone
        entity_types = [e.entity_type for e in entities]
        assert EntityType.EMAIL in entity_types
        assert EntityType.PHONE in entity_types
        
        # Check email entity
        email_entities = [e for e in entities if e.entity_type == EntityType.EMAIL]
        assert len(email_entities) == 1
        email_entity = email_entities[0]
        assert email_entity.text == "jane.doe@email.com"
        assert email_entity.confidence == 0.9  # Pattern matching confidence
        assert email_entity.metadata["source"] == "pattern"
        
        # Check phone entity
        phone_entities = [e for e in entities if e.entity_type == EntityType.PHONE]
        assert len(phone_entities) == 1
        phone_entity = phone_entities[0]
        assert phone_entity.text == "555-123-4567"
    
    def test_detect_entities_patterns_custom(self, sample_config):
        """Test custom pattern detection"""
        redactor = TextRedactor(sample_config)
        text_with_custom = "Please reference case ID-123456 in your response."
        
        # Add custom entity type to config
        redactor.config.entity_types.append(EntityType.CUSTOM)
        redactor.patterns[EntityType.CUSTOM] = redactor.patterns["custom_id"]
        
        entities = redactor.detect_entities_patterns(text_with_custom)
        
        custom_entities = [e for e in entities if e.entity_type == EntityType.CUSTOM]
        assert len(custom_entities) == 1
        assert custom_entities[0].text == "ID-123456"
    
    def test_merge_entities_no_overlap(self, sample_config):
        """Test merging entities with no overlaps"""
        redactor = TextRedactor(sample_config)
        
        entities1 = [
            PIIEntity(EntityType.PERSON, "John", 0, 4, 0.9, ConfidenceLevel.VERY_HIGH)
        ]
        entities2 = [
            PIIEntity(EntityType.EMAIL, "test@email.com", 10, 24, 0.95, ConfidenceLevel.VERY_HIGH)
        ]
        
        merged = redactor.merge_entities([entities1, entities2])
        
        assert len(merged) == 2
        assert merged[0].entity_type == EntityType.PERSON
        assert merged[1].entity_type == EntityType.EMAIL
    
    def test_merge_entities_with_overlap(self, sample_config):
        """Test merging entities with overlaps"""
        redactor = TextRedactor(sample_config)
        
        # Overlapping entities - same text span but different confidence
        entities1 = [
            PIIEntity(EntityType.PERSON, "John Smith", 0, 10, 0.8, ConfidenceLevel.HIGH)
        ]
        entities2 = [
            PIIEntity(EntityType.PERSON, "John Smith", 0, 10, 0.95, ConfidenceLevel.VERY_HIGH)
        ]
        
        merged = redactor.merge_entities([entities1, entities2])
        
        assert len(merged) == 1
        assert merged[0].confidence == 0.95  # Higher confidence entity kept
    
    def test_apply_redaction_replace_method(self, sample_config):
        """Test redaction with REPLACE method"""
        redactor = TextRedactor(sample_config)
        text = "John Smith called yesterday."
        
        entity = PIIEntity(
            entity_type=EntityType.PERSON,
            text="John Smith",
            start_pos=0,
            end_pos=10,
            confidence=0.9,
            confidence_level=ConfidenceLevel.VERY_HIGH
        )
        
        redacted_text, redacted_entities = redactor.apply_redaction(text, [entity])
        
        assert redacted_text == "[PERSON] called yesterday."
        assert len(redacted_entities) == 1
        assert redacted_entities[0].replacement == "[PERSON]"
    
    def test_apply_redaction_mask_method(self, sample_config):
        """Test redaction with MASK method"""
        sample_config.redaction_methods[EntityType.PERSON] = RedactionType.MASK
        
        redactor = TextRedactor(sample_config)
        text = "John Smith called yesterday."
        
        entity = PIIEntity(
            entity_type=EntityType.PERSON,
            text="John Smith",
            start_pos=0,
            end_pos=10,
            confidence=0.9,
            confidence_level=ConfidenceLevel.VERY_HIGH
        )
        
        redacted_text, redacted_entities = redactor.apply_redaction(text, [entity])
        
        assert redacted_text == "********** called yesterday."
        assert len(redacted_entities) == 1
        assert redacted_entities[0].replacement == "**********"
    
    def test_apply_redaction_delete_method(self, sample_config):
        """Test redaction with DELETE method"""
        sample_config.redaction_methods[EntityType.PERSON] = RedactionType.DELETE
        
        redactor = TextRedactor(sample_config)
        text = "John Smith called yesterday."
        
        entity = PIIEntity(
            entity_type=EntityType.PERSON,
            text="John Smith",
            start_pos=0,
            end_pos=10,
            confidence=0.9,
            confidence_level=ConfidenceLevel.VERY_HIGH
        )
        
        redacted_text, redacted_entities = redactor.apply_redaction(text, [entity])
        
        assert redacted_text == " called yesterday."
        assert len(redacted_entities) == 1
        assert redacted_entities[0].replacement == ""
    
    def test_apply_redaction_confidence_threshold(self, sample_config):
        """Test redaction respects confidence threshold"""
        sample_config.min_confidence = 0.8
        
        redactor = TextRedactor(sample_config)
        text = "John Smith called yesterday."
        
        # Entity with confidence below threshold
        low_confidence_entity = PIIEntity(
            entity_type=EntityType.PERSON,
            text="John Smith",
            start_pos=0,
            end_pos=10,
            confidence=0.7,  # Below threshold
            confidence_level=ConfidenceLevel.MEDIUM
        )
        
        redacted_text, redacted_entities = redactor.apply_redaction(text, [low_confidence_entity])
        
        assert redacted_text == text  # No redaction applied
        assert len(redacted_entities) == 0
    
    def test_apply_redaction_multiple_entities(self, sample_config):
        """Test redaction with multiple entities"""
        redactor = TextRedactor(sample_config)
        text = "John Smith called jane@email.com at 555-1234."
        
        entities = [
            PIIEntity(EntityType.PERSON, "John Smith", 0, 10, 0.9, ConfidenceLevel.VERY_HIGH),
            PIIEntity(EntityType.EMAIL, "jane@email.com", 18, 32, 0.95, ConfidenceLevel.VERY_HIGH),
            PIIEntity(EntityType.PHONE, "555-1234", 36, 44, 0.85, ConfidenceLevel.HIGH)
        ]
        
        redacted_text, redacted_entities = redactor.apply_redaction(text, entities)
        
        expected = "[PERSON] called ************* at ********."
        assert redacted_text == expected
        assert len(redacted_entities) == 3
    
    @patch.object(TextRedactor, 'detect_entities_spacy')
    @patch.object(TextRedactor, 'detect_entities_transformers') 
    @patch.object(TextRedactor, 'detect_entities_patterns')
    def test_redact_full_pipeline(self, mock_patterns, mock_transformers, mock_spacy, sample_config, sample_text):
        """Test full redaction pipeline"""
        redactor = TextRedactor(sample_config)
        
        # Mock entity detection methods
        mock_spacy.return_value = [
            PIIEntity(EntityType.PERSON, "John Smith", 0, 10, 0.8, ConfidenceLevel.HIGH)
        ]
        mock_transformers.return_value = []
        mock_patterns.return_value = [
            PIIEntity(EntityType.EMAIL, "jane.doe@email.com", 21, 40, 0.9, ConfidenceLevel.VERY_HIGH)
        ]
        
        result = redactor.redact(sample_text)
        
        assert result.content_type == "text"
        assert result.total_entities == 2
        assert result.redacted_count == 2
        assert result.processing_time > 0
        assert len(result.confidence_scores) > 0
        
        # Verify detection methods were called
        mock_spacy.assert_called_once_with(sample_text)
        mock_transformers.assert_called_once_with(sample_text)
        mock_patterns.assert_called_once_with(sample_text)
    
    def test_redact_with_output_path(self, sample_config, sample_text):
        """Test redaction with file output"""
        redactor = TextRedactor(sample_config)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            output_path = Path(f.name)
        
        try:
            with patch.object(redactor, 'detect_entities_spacy', return_value=[]):
                with patch.object(redactor, 'detect_entities_transformers', return_value=[]):
                    with patch.object(redactor, 'detect_entities_patterns', return_value=[]):
                        result = redactor.redact(sample_text, output_path)
            
            assert result.redacted_content_path == str(output_path)
            assert output_path.exists()
            
            # Check file content
            content = output_path.read_text(encoding='utf-8')
            assert len(content) > 0
            
        finally:
            output_path.unlink(missing_ok=True)
    
    def test_map_spacy_label(self, sample_config):
        """Test spaCy label mapping"""
        redactor = TextRedactor(sample_config)
        
        assert redactor._map_spacy_label("PERSON") == EntityType.PERSON
        assert redactor._map_spacy_label("PER") == EntityType.PERSON
        assert redactor._map_spacy_label("ORG") == EntityType.ORGANIZATION
        assert redactor._map_spacy_label("GPE") == EntityType.LOCATION
        assert redactor._map_spacy_label("LOC") == EntityType.LOCATION
        assert redactor._map_spacy_label("DATE") == EntityType.DATE
        assert redactor._map_spacy_label("TIME") == EntityType.DATE
        assert redactor._map_spacy_label("UNKNOWN") is None
    
    def test_map_transformers_label(self, sample_config):
        """Test Transformers label mapping"""
        redactor = TextRedactor(sample_config)
        
        assert redactor._map_transformers_label("PER") == EntityType.PERSON
        assert redactor._map_transformers_label("PERSON") == EntityType.PERSON
        assert redactor._map_transformers_label("ORG") == EntityType.ORGANIZATION
        assert redactor._map_transformers_label("LOC") == EntityType.LOCATION
        assert redactor._map_transformers_label("MISC") == EntityType.CUSTOM
        assert redactor._map_transformers_label("UNKNOWN") is None
    
    def test_get_confidence_level(self, sample_config):
        """Test confidence level mapping"""
        redactor = TextRedactor(sample_config)
        
        assert redactor._get_confidence_level(0.95) == ConfidenceLevel.VERY_HIGH
        assert redactor._get_confidence_level(0.85) == ConfidenceLevel.HIGH
        assert redactor._get_confidence_level(0.65) == ConfidenceLevel.MEDIUM
        assert redactor._get_confidence_level(0.45) == ConfidenceLevel.LOW
    
    def test_entities_overlap(self, sample_config):
        """Test entity overlap detection"""
        redactor = TextRedactor(sample_config)
        
        entity1 = PIIEntity(EntityType.PERSON, "John", 0, 4, 0.9, ConfidenceLevel.VERY_HIGH)
        entity2 = PIIEntity(EntityType.PERSON, "John Smith", 0, 10, 0.8, ConfidenceLevel.HIGH)
        entity3 = PIIEntity(EntityType.EMAIL, "test@email.com", 15, 29, 0.9, ConfidenceLevel.VERY_HIGH)
        
        # Overlapping entities
        assert redactor._entities_overlap(entity1, entity2) is True
        
        # Non-overlapping entities
        assert redactor._entities_overlap(entity1, entity3) is False
        assert redactor._entities_overlap(entity2, entity3) is False
    
    def test_generate_replacement_methods(self, sample_config):
        """Test different replacement generation methods"""
        redactor = TextRedactor(sample_config)
        
        person_entity = PIIEntity(EntityType.PERSON, "John Smith", 0, 10, 0.9, ConfidenceLevel.VERY_HIGH)
        
        # Test MASK
        mask_replacement = redactor._generate_replacement(person_entity, RedactionType.MASK)
        assert mask_replacement == "**********"
        
        # Test REPLACE
        replace_replacement = redactor._generate_replacement(person_entity, RedactionType.REPLACE)
        assert replace_replacement == "[PERSON]"
        
        # Test DELETE
        delete_replacement = redactor._generate_replacement(person_entity, RedactionType.DELETE)
        assert delete_replacement == ""
        
        # Test ANONYMIZE
        anon_replacement = redactor._generate_replacement(person_entity, RedactionType.ANONYMIZE)
        assert anon_replacement == "John Doe"
    
    def test_calculate_confidence_scores(self, sample_config, sample_entities):
        """Test confidence score calculation"""
        redactor = TextRedactor(sample_config)
        
        scores = redactor._calculate_confidence_scores(sample_entities)
        
        assert "PERSON" in scores
        assert "EMAIL" in scores 
        assert "PHONE" in scores
        
        assert scores["PERSON"] == 0.95
        assert scores["EMAIL"] == 0.99
        assert scores["PHONE"] == 0.88