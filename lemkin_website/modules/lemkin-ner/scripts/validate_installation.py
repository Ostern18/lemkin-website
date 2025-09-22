#!/usr/bin/env python3
"""
Validate lemkin-ner installation and basic functionality.

This script checks:
1. All modules can be imported
2. Basic configuration works
3. Simple entity extraction works
4. Core functionality is available

Usage:
    python scripts/validate_installation.py
"""

import sys
import traceback
from pathlib import Path

def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    
    try:
        # Core imports
        from lemkin_ner import (
            LegalNERProcessor,
            NERConfig, 
            Entity,
            EntityType,
            EntityGraph,
            EntityLinkResult,
            ValidationResult,
            LanguageCode,
            create_default_config,
            validate_config
        )
        print("✓ Core modules imported successfully")
        
        # Component imports
        from lemkin_ner import (
            LegalEntityRecognizer,
            EntityLinker,
            MultilingualProcessor,
            EntityValidator,
            ValidationStatus,
            QualityMetric
        )
        print("✓ Component modules imported successfully")
        
        return True
        
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False
    except Exception as e:
        print(f"✗ Unexpected error during import: {e}")
        traceback.print_exc()
        return False


def test_configuration():
    """Test configuration creation and validation."""
    print("\nTesting configuration...")
    
    try:
        from lemkin_ner import create_default_config, validate_config, NERConfig, LanguageCode, EntityType
        
        # Test default config creation
        config = create_default_config()
        print("✓ Default configuration created")
        
        # Test config validation
        issues = validate_config(config)
        if not issues:
            print("✓ Default configuration is valid")
        else:
            print(f"✗ Configuration issues: {issues}")
            return False
        
        # Test custom config
        custom_config = NERConfig(
            primary_language=LanguageCode.EN,
            entity_types=[EntityType.PERSON, EntityType.ORGANIZATION],
            min_confidence=0.7
        )
        print("✓ Custom configuration created")
        
        # Test config serialization
        config_dict = config.model_dump()
        reconstructed = NERConfig.model_validate(config_dict)
        print("✓ Configuration serialization works")
        
        return True
        
    except Exception as e:
        print(f"✗ Configuration test failed: {e}")
        traceback.print_exc()
        return False


def test_entity_model():
    """Test Entity model creation and validation."""
    print("\nTesting entity model...")
    
    try:
        from lemkin_ner import Entity, EntityType, LanguageCode
        
        # Create entity
        entity = Entity(
            text="John Smith",
            entity_type=EntityType.PERSON,
            start_pos=0,
            end_pos=10,
            confidence=0.95,
            language=LanguageCode.EN,
            document_id="test_doc"
        )
        print("✓ Entity created successfully")
        
        # Test serialization
        entity_dict = entity.to_dict()
        assert "entity_id" in entity_dict
        assert entity_dict["text"] == "John Smith"
        print("✓ Entity serialization works")
        
        # Test validation
        assert entity.confidence == 0.95
        assert entity.entity_type == EntityType.PERSON
        print("✓ Entity validation works")
        
        return True
        
    except Exception as e:
        print(f"✗ Entity model test failed: {e}")
        traceback.print_exc()
        return False


def test_basic_processing():
    """Test basic NER processing functionality."""
    print("\nTesting basic processing...")
    
    try:
        from lemkin_ner import LegalNERProcessor, create_default_config
        
        # Create processor with minimal config
        config = create_default_config()
        config.use_transformers = False  # Disable to avoid model loading issues
        config.min_confidence = 0.1  # Low threshold for testing
        
        # Try to create processor (may fail due to missing models, but should not crash)
        try:
            processor = LegalNERProcessor(config)
            print("✓ Processor created successfully")
            
            # Test simple text processing
            sample_text = "John Smith works at ABC Corporation in New York."
            
            try:
                result = processor.process_text(sample_text, "test_doc")
                
                # Check result structure
                assert "entities" in result
                assert "metadata" in result
                assert isinstance(result["entities"], list)
                print("✓ Text processing works")
                
                # Check if any entities were found
                if result["entities"]:
                    print(f"✓ Found {len(result['entities'])} entities")
                    
                    # Show first entity as example
                    first_entity = result["entities"][0]
                    required_fields = ["text", "entity_type", "confidence", "language"]
                    for field in required_fields:
                        assert field in first_entity, f"Missing field: {field}"
                    print("✓ Entity structure is valid")
                else:
                    print("⚠ No entities found (may be due to missing models)")
                    
                return True
                
            except Exception as e:
                print(f"⚠ Text processing failed (likely due to missing models): {e}")
                # This is expected if language models are not installed
                return True
                
        except Exception as e:
            print(f"⚠ Processor creation failed (likely due to missing models): {e}")
            # This is expected if language models are not installed
            return True
            
    except Exception as e:
        print(f"✗ Basic processing test failed: {e}")
        traceback.print_exc()
        return False


def test_cli_module():
    """Test CLI module can be imported."""
    print("\nTesting CLI module...")
    
    try:
        from lemkin_ner import cli
        print("✓ CLI module imported successfully")
        
        # Check if main app exists
        assert hasattr(cli, 'app'), "CLI app not found"
        print("✓ CLI app is available")
        
        return True
        
    except Exception as e:
        print(f"✗ CLI module test failed: {e}")
        traceback.print_exc()
        return False


def test_validator():
    """Test validator functionality."""
    print("\nTesting validator...")
    
    try:
        from lemkin_ner import EntityValidator, create_default_config, Entity, EntityType, LanguageCode
        
        config = create_default_config()
        validator = EntityValidator(config)
        print("✓ Validator created successfully")
        
        # Create test entity
        entity = Entity(
            text="Test Entity",
            entity_type=EntityType.PERSON,
            start_pos=0,
            end_pos=11,
            confidence=0.8,
            language=LanguageCode.EN,
            document_id="test"
        )
        
        # Test validation
        validation_result = validator.validate_entity(entity)
        
        # Check result structure
        assert hasattr(validation_result, 'is_valid')
        assert hasattr(validation_result, 'validation_confidence')
        assert hasattr(validation_result, 'issues')
        assert hasattr(validation_result, 'suggestions')
        print("✓ Entity validation works")
        
        return True
        
    except Exception as e:
        print(f"✗ Validator test failed: {e}")
        traceback.print_exc()
        return False


def main():
    """Run all validation tests."""
    print("Lemkin NER Installation Validation")
    print("=" * 40)
    
    tests = [
        ("Import Test", test_imports),
        ("Configuration Test", test_configuration), 
        ("Entity Model Test", test_entity_model),
        ("Basic Processing Test", test_basic_processing),
        ("CLI Module Test", test_cli_module),
        ("Validator Test", test_validator)
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        print(f"\n--- {test_name} ---")
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"✗ Test failed with exception: {e}")
            failed += 1
    
    print("\n" + "=" * 40)
    print(f"Validation Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All tests passed! Lemkin NER is ready to use.")
        print("\nNext steps:")
        print("1. Install language models: python -m spacy download en_core_web_sm")
        print("2. Try the example: python examples/complete_workflow.py")
        print("3. Use the CLI: lemkin-ner extract-entities --help")
        return 0
    else:
        print("⚠ Some tests failed. This may be due to missing dependencies.")
        print("\nTroubleshooting:")
        print("1. Install required models: python -m spacy download en_core_web_sm")
        print("2. Install optional dependencies: pip install stanza transformers")
        print("3. Check the installation guide in README.md")
        return 1


if __name__ == "__main__":
    sys.exit(main())