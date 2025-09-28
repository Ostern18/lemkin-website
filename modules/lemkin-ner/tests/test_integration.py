"""
Integration tests for lemkin-ner package.
"""

import pytest
import tempfile
import json
from pathlib import Path
from datetime import datetime

from lemkin_ner import (
    LegalNERProcessor, 
    NERConfig, 
    Entity, 
    EntityType, 
    LanguageCode,
    create_default_config,
    EntityValidator
)


class TestLemkinNERIntegration:
    """Integration tests for the complete NER pipeline"""
    
    def setup_method(self):
        """Setup test configuration and sample data"""
        self.config = create_default_config()
        self.config.min_confidence = 0.3  # Lower threshold for testing
        self.config.entity_types = [
            EntityType.PERSON, EntityType.ORGANIZATION, 
            EntityType.LOCATION, EntityType.DATE, EntityType.LEGAL_ENTITY
        ]
        
        # Sample legal text
        self.sample_text = """
        In the case of Smith v. ABC Corporation, filed on January 15, 2023, 
        the plaintiff John Smith alleges that ABC Corporation, headquartered in New York,
        violated contract terms. The case was heard by Judge Patricia Williams
        in the Southern District Court of New York. Attorney Michael Johnson
        represented the plaintiff, while Sarah Davis from Davis & Associates
        represented the defendant. The contract in question was signed in Los Angeles
        and governed by California law.
        """
        
        self.multilingual_text = {
            "en": "Attorney John Smith filed a motion in New York Supreme Court.",
            "es": "El abogado Juan Pérez presentó una moción en el Tribunal Supremo de Madrid.",
            "fr": "L'avocat Pierre Dubois a déposé une requête au Tribunal de Paris."
        }
    
    def test_basic_entity_extraction(self):
        """Test basic entity extraction functionality"""
        processor = LegalNERProcessor(self.config)
        
        result = processor.process_text(self.sample_text, "test_doc_1")
        
        assert "entities" in result
        assert "metadata" in result
        assert len(result["entities"]) > 0
        
        # Check that we found some expected entities
        entity_texts = [e["text"] for e in result["entities"]]
        
        # Should find some person names
        person_entities = [e for e in result["entities"] if e["entity_type"] == "PERSON"]
        assert len(person_entities) > 0
        
        # Should find some organizations
        org_entities = [e for e in result["entities"] if e["entity_type"] == "ORGANIZATION"]
        assert len(org_entities) > 0
        
        # Should find some locations
        location_entities = [e for e in result["entities"] if e["entity_type"] == "LOCATION"]
        assert len(location_entities) > 0
    
    def test_multilingual_processing(self):
        """Test multilingual entity extraction"""
        processor = LegalNERProcessor(self.config)
        
        results = []
        for lang_code, text in self.multilingual_text.items():
            language = LanguageCode(lang_code)
            result = processor.process_text(text, f"doc_{lang_code}", language)
            results.append(result)
            
            assert result["metadata"]["language"] == lang_code
            assert len(result["entities"]) > 0
    
    def test_batch_processing(self):
        """Test batch processing of multiple texts"""
        processor = LegalNERProcessor(self.config)
        
        texts = [
            "Judge Williams presided over the case.",
            "The law firm of Smith & Associates handled the appeal.",
            "The hearing was scheduled for March 10, 2024."
        ]
        
        results = processor.process_batch(texts)
        
        assert len(results) == 3
        for result in results:
            assert "entities" in result
            assert "metadata" in result
    
    def test_entity_linking(self):
        """Test cross-document entity linking"""
        processor = LegalNERProcessor(self.config)
        
        # Create multiple document results
        texts = [
            "John Smith filed a lawsuit against ABC Corp.",
            "Mr. Smith's attorney argued the case effectively.",
            "ABC Corporation denied the allegations."
        ]
        
        document_results = []
        for i, text in enumerate(texts):
            result = processor.process_text(text, f"doc_{i}")
            document_results.append(result)
        
        # Link entities across documents
        entity_graph = processor.link_entities_across_documents(document_results)
        
        assert entity_graph.entities
        assert len(entity_graph.entities) > 0
        
        # Check that relationships were created
        if len(entity_graph.relationships) > 0:
            assert entity_graph.relationships[0]["source_id"] in entity_graph.entities
            assert entity_graph.relationships[0]["target_id"] in entity_graph.entities
    
    def test_entity_validation(self):
        """Test entity validation functionality"""
        processor = LegalNERProcessor(self.config)
        validator = EntityValidator(self.config)
        
        # Extract entities
        result = processor.process_text(self.sample_text, "validation_test")
        entities = [Entity.model_validate(e) for e in result["entities"]]
        
        # Validate entities
        validation_results = validator.validate_batch(entities)
        
        assert len(validation_results) == len(entities)
        
        for validation_result in validation_results:
            assert hasattr(validation_result, 'is_valid')
            assert hasattr(validation_result, 'validation_confidence')
            assert isinstance(validation_result.issues, list)
            assert isinstance(validation_result.suggestions, list)
    
    def test_quality_report_generation(self):
        """Test quality report generation"""
        processor = LegalNERProcessor(self.config)
        validator = EntityValidator(self.config)
        
        # Process text and validate
        result = processor.process_text(self.sample_text, "quality_test")
        entities = [Entity.model_validate(e) for e in result["entities"]]
        validation_results = validator.validate_batch(entities)
        
        # Generate quality report
        quality_report = validator.generate_quality_report(validation_results)
        
        assert "summary" in quality_report
        assert "by_entity_type" in quality_report
        assert "quality_distribution" in quality_report
        assert "recommendations" in quality_report
        
        summary = quality_report["summary"]
        assert "total_entities" in summary
        assert "valid_entities" in summary
        assert "validity_rate" in summary
        assert "average_confidence" in summary
    
    def test_human_review_task_creation(self):
        """Test creation of human review tasks"""
        processor = LegalNERProcessor(self.config)
        validator = EntityValidator(self.config)
        
        # Process and validate entities
        result = processor.process_text(self.sample_text, "review_test")
        entities = [Entity.model_validate(e) for e in result["entities"]]
        validation_results = validator.validate_batch(entities)
        
        # Create review tasks in temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            review_summary = validator.create_human_review_tasks(
                validation_results, temp_dir
            )
            
            assert "total_results" in review_summary
            assert "review_needed" in review_summary
            assert "output_directory" in review_summary
            
            # Check that files were created
            temp_path = Path(temp_dir)
            created_files = list(temp_path.glob("*.json")) + list(temp_path.glob("*.csv")) + list(temp_path.glob("*.md"))
            assert len(created_files) > 0
    
    def test_configuration_validation(self):
        """Test configuration validation"""
        from lemkin_ner import validate_config
        
        # Test valid configuration
        config = create_default_config()
        issues = validate_config(config)
        assert len(issues) == 0
        
        # Test invalid configuration
        config.min_confidence = 1.5  # Invalid value
        config.similarity_threshold = -0.1  # Invalid value
        
        issues = validate_config(config)
        assert len(issues) > 0
        assert any("min_confidence" in issue for issue in issues)
        assert any("similarity_threshold" in issue for issue in issues)
    
    def test_export_functionality(self):
        """Test different export formats"""
        processor = LegalNERProcessor(self.config)
        result = processor.process_text(self.sample_text, "export_test")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Test JSON export
            json_file = temp_path / "results.json"
            processor.export_results(result, json_file, "json")
            assert json_file.exists()
            
            # Verify JSON content
            with open(json_file, 'r') as f:
                exported_data = json.load(f)
            assert "entities" in exported_data
            
            # Test CSV export
            csv_file = temp_path / "results.csv"
            processor.export_results(result, csv_file, "csv")
            assert csv_file.exists()
            
            # Test XML export
            xml_file = temp_path / "results.xml"
            processor.export_results(result, xml_file, "xml")
            assert xml_file.exists()
    
    def test_legal_entity_recognition(self):
        """Test legal-specific entity recognition"""
        legal_text = """
        The Supreme Court of the United States ruled in Brown v. Board of Education.
        The case cited 14th Amendment and Title VII of the Civil Rights Act.
        Judge Marshall wrote the majority opinion. The plaintiff was represented
        by the NAACP Legal Defense Fund.
        """
        
        processor = LegalNERProcessor(self.config)
        result = processor.process_text(legal_text, "legal_test")
        
        entities = result["entities"]
        
        # Should find legal entities
        legal_entities = [e for e in entities if e.get("entity_type") in ["LEGAL_ENTITY", "COURT", "CASE_NAME"]]
        assert len(legal_entities) > 0
        
        # Check for specific legal patterns
        entity_texts = [e["text"].lower() for e in entities]
        
        # Should find court references
        court_found = any("supreme court" in text or "court" in text for text in entity_texts)
        assert court_found
    
    def test_error_handling(self):
        """Test error handling with invalid inputs"""
        processor = LegalNERProcessor(self.config)
        
        # Test empty text
        result = processor.process_text("", "empty_test")
        assert result["entities"] == []
        assert "error" not in result["metadata"] or result["metadata"]["status"] == "empty_text"
        
        # Test very long text (should not crash)
        long_text = "This is a test. " * 10000
        result = processor.process_text(long_text, "long_test")
        assert "entities" in result
        
        # Test non-existent file
        with pytest.raises(FileNotFoundError):
            processor.process_document("non_existent_file.txt")
    
    def test_confidence_filtering(self):
        """Test confidence-based entity filtering"""
        # High confidence threshold
        high_conf_config = create_default_config()
        high_conf_config.min_confidence = 0.9
        
        processor_high = LegalNERProcessor(high_conf_config)
        result_high = processor_high.process_text(self.sample_text, "high_conf_test")
        
        # Low confidence threshold
        low_conf_config = create_default_config()
        low_conf_config.min_confidence = 0.1
        
        processor_low = LegalNERProcessor(low_conf_config)
        result_low = processor_low.process_text(self.sample_text, "low_conf_test")
        
        # Should find more entities with lower threshold
        assert len(result_low["entities"]) >= len(result_high["entities"])
        
        # All entities should meet confidence threshold
        for entity in result_high["entities"]:
            assert entity["confidence"] >= 0.9
        
        for entity in result_low["entities"]:
            assert entity["confidence"] >= 0.1


class TestConfigurationManagement:
    """Tests for configuration management"""
    
    def test_default_config_creation(self):
        """Test default configuration creation"""
        config = create_default_config()
        
        assert isinstance(config, NERConfig)
        assert config.primary_language == LanguageCode.EN
        assert len(config.entity_types) > 0
        assert 0.0 <= config.min_confidence <= 1.0
        assert 0.0 <= config.similarity_threshold <= 1.0
    
    def test_config_serialization(self):
        """Test configuration serialization and deserialization"""
        config = create_default_config()
        config.min_confidence = 0.7
        config.entity_types = [EntityType.PERSON, EntityType.ORGANIZATION]
        
        # Test model_dump
        config_dict = config.model_dump()
        assert isinstance(config_dict, dict)
        assert config_dict["min_confidence"] == 0.7
        
        # Test model_validate
        reconstructed_config = NERConfig.model_validate(config_dict)
        assert reconstructed_config.min_confidence == 0.7
        assert len(reconstructed_config.entity_types) == 2
    
    def test_config_with_file_paths(self):
        """Test configuration with file paths"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            legal_terminology = {
                "en": {
                    "legal_entities": ["attorney", "judge", "court"],
                    "legal_documents": ["motion", "brief", "order"]
                }
            }
            json.dump(legal_terminology, f)
            temp_file = f.name
        
        try:
            config = create_default_config()
            config.legal_terminology_file = temp_file
            
            # Should not raise an error
            processor = LegalNERProcessor(config)
            assert processor.config.legal_terminology_file == temp_file
            
        finally:
            Path(temp_file).unlink()


class TestEntityModel:
    """Tests for Entity and related data models"""
    
    def test_entity_creation(self):
        """Test Entity model creation and validation"""
        entity = Entity(
            text="John Smith",
            entity_type=EntityType.PERSON,
            start_pos=0,
            end_pos=10,
            confidence=0.95,
            language=LanguageCode.EN,
            document_id="test_doc"
        )
        
        assert entity.text == "John Smith"
        assert entity.entity_type == EntityType.PERSON
        assert entity.confidence == 0.95
        assert len(entity.entity_id) > 0  # UUID should be generated
        assert isinstance(entity.timestamp, datetime)
    
    def test_entity_serialization(self):
        """Test Entity serialization"""
        entity = Entity(
            text="ABC Corp",
            entity_type=EntityType.ORGANIZATION,
            start_pos=10,
            end_pos=18,
            confidence=0.88,
            language=LanguageCode.EN,
            document_id="test_doc",
            aliases=["ABC Corporation", "ABC Inc"]
        )
        
        entity_dict = entity.to_dict()
        
        assert entity_dict["text"] == "ABC Corp"
        assert entity_dict["entity_type"] == "ORGANIZATION"
        assert entity_dict["confidence"] == 0.88
        assert "aliases" in entity_dict
        assert len(entity_dict["aliases"]) == 2
    
    def test_entity_validation_constraints(self):
        """Test Entity model validation constraints"""
        # Test invalid confidence (should be between 0 and 1)
        with pytest.raises(ValueError):
            Entity(
                text="Test",
                entity_type=EntityType.PERSON,
                start_pos=0,
                end_pos=4,
                confidence=1.5,  # Invalid
                language=LanguageCode.EN,
                document_id="test"
            )
        
        # Test negative positions
        with pytest.raises(ValueError):
            Entity(
                text="Test",
                entity_type=EntityType.PERSON,
                start_pos=-1,  # Invalid
                end_pos=4,
                confidence=0.8,
                language=LanguageCode.EN,
                document_id="test"
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])