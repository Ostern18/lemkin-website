#!/usr/bin/env python3
"""
Test script for Lemkin Images implementation
"""

import sys
import traceback
from pathlib import Path

# Add the src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "lemkin-images" / "src"))

def test_imports():
    """Test that all modules can be imported successfully"""
    print("Testing imports...")
    
    try:
        # Test core imports
        from lemkin_images.core import (
            ImageAuthenticator, ImageAuthConfig, AuthenticityReport,
            ReverseSearchResults, ManipulationAnalysis, GeolocationResult,
            MetadataForensics, ImageMetadata, SearchEngine, ImageFormat
        )
        print("✓ Core imports successful")
        
        # Test individual module imports
        from lemkin_images.reverse_search import ReverseImageSearcher
        print("✓ Reverse search module import successful")
        
        from lemkin_images.manipulation_detector import ImageManipulationDetector
        print("✓ Manipulation detector module import successful")
        
        from lemkin_images.geolocation_helper import ImageGeolocator
        print("✓ Geolocation helper module import successful")
        
        from lemkin_images.metadata_forensics import MetadataForensicsAnalyzer
        print("✓ Metadata forensics module import successful")
        
        # Test package-level imports
        import lemkin_images
        print("✓ Package import successful")
        
        return True
        
    except Exception as e:
        print(f"✗ Import failed: {str(e)}")
        traceback.print_exc()
        return False

def test_basic_functionality():
    """Test basic functionality without requiring external dependencies"""
    print("\nTesting basic functionality...")
    
    try:
        from lemkin_images import ImageAuthenticator, ImageAuthConfig
        
        # Test configuration creation
        config = ImageAuthConfig()
        print(f"✓ Configuration created with {len(config.search_engines)} search engines")
        
        # Test authenticator creation
        authenticator = ImageAuthenticator(config)
        print("✓ ImageAuthenticator created successfully")
        
        # Test that component analyzers are None initially (lazy loading)
        assert authenticator._reverse_searcher is None
        assert authenticator._manipulation_detector is None
        assert authenticator._geolocation_helper is None
        assert authenticator._metadata_forensics is None
        print("✓ Lazy loading initialized correctly")
        
        return True
        
    except Exception as e:
        print(f"✗ Basic functionality test failed: {str(e)}")
        traceback.print_exc()
        return False

def test_data_models():
    """Test data model creation and validation"""
    print("\nTesting data models...")
    
    try:
        from lemkin_images.core import (
            ImageAuthConfig, SearchResult, SearchEngine, ManipulationIndicator,
            ManipulationType, GeolocationData, ImageMetadata, ImageFormat
        )
        
        # Test ImageAuthConfig
        config = ImageAuthConfig(
            max_search_results=100,
            manipulation_threshold=0.8,
            enable_visual_geolocation=True
        )
        assert config.max_search_results == 100
        assert config.manipulation_threshold == 0.8
        print("✓ ImageAuthConfig validation works")
        
        # Test SearchResult
        search_result = SearchResult(
            search_engine=SearchEngine.GOOGLE,
            url="https://example.com/image.jpg",
            title="Test Image",
            similarity_score=0.95
        )
        assert search_result.search_engine == SearchEngine.GOOGLE
        assert search_result.similarity_score == 0.95
        print("✓ SearchResult creation works")
        
        # Test ManipulationIndicator
        indicator = ManipulationIndicator(
            manipulation_type=ManipulationType.COPY_MOVE,
            confidence=0.85,
            severity="high",
            detection_method="block_matching",
            description="Detected copy-move regions"
        )
        assert indicator.manipulation_type == ManipulationType.COPY_MOVE
        assert indicator.confidence == 0.85
        print("✓ ManipulationIndicator creation works")
        
        # Test GeolocationData
        geo_data = GeolocationData(
            latitude=40.7589,
            longitude=-73.9851,
            source="exif",
            confidence=0.9,
            country="United States",
            city="New York"
        )
        assert abs(geo_data.latitude - 40.7589) < 0.0001
        assert geo_data.source == "exif"
        print("✓ GeolocationData creation works")
        
        # Test ImageMetadata
        metadata = ImageMetadata(
            file_name="test.jpg",
            file_path="/path/to/test.jpg",
            file_size_bytes=1024000,
            file_format=ImageFormat.JPEG,
            width=1920,
            height=1080,
            md5_hash="d41d8cd98f00b204e9800998ecf8427e",
            sha256_hash="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        )
        assert metadata.file_format == ImageFormat.JPEG
        assert metadata.width == 1920
        print("✓ ImageMetadata creation works")
        
        return True
        
    except Exception as e:
        print(f"✗ Data model test failed: {str(e)}")
        traceback.print_exc()
        return False

def test_cli_imports():
    """Test CLI module imports"""
    print("\nTesting CLI imports...")
    
    try:
        from lemkin_images.cli import cli
        print("✓ CLI module import successful")
        
        # Test that click commands are properly set up
        commands = cli.commands
        expected_commands = {'reverse-search', 'detect-manipulation', 'geolocate', 'analyze-metadata', 'authenticate', 'generate-config'}
        
        actual_commands = set(commands.keys())
        missing_commands = expected_commands - actual_commands
        
        if missing_commands:
            print(f"✗ Missing CLI commands: {missing_commands}")
            return False
        
        print(f"✓ All {len(expected_commands)} CLI commands available: {list(actual_commands)}")
        return True
        
    except Exception as e:
        print(f"✗ CLI import test failed: {str(e)}")
        traceback.print_exc()
        return False

def test_package_info():
    """Test package information and metadata"""
    print("\nTesting package information...")
    
    try:
        import lemkin_images
        
        # Test version
        version = lemkin_images.get_version()
        print(f"✓ Package version: {version}")
        
        # Test package info
        info = lemkin_images.get_package_info()
        assert 'name' in info
        assert 'version' in info
        assert 'description' in info
        print(f"✓ Package info available: {info['name']} v{info['version']}")
        
        # Test __all__ exports
        all_exports = lemkin_images.__all__
        print(f"✓ Package exports {len(all_exports)} items")
        
        # Test that key exports are available
        key_exports = ['ImageAuthenticator', 'ImageAuthConfig', 'AuthenticityReport']
        for export in key_exports:
            assert hasattr(lemkin_images, export), f"Missing export: {export}"
        print(f"✓ Key exports available: {key_exports}")
        
        return True
        
    except Exception as e:
        print(f"✗ Package info test failed: {str(e)}")
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("=" * 60)
    print("LEMKIN IMAGES IMPLEMENTATION TEST")
    print("=" * 60)
    
    tests = [
        test_imports,
        test_basic_functionality,
        test_data_models,
        test_cli_imports,
        test_package_info
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                print(f"Test {test.__name__} failed")
        except Exception as e:
            print(f"Test {test.__name__} crashed: {str(e)}")
    
    print("\n" + "=" * 60)
    print(f"TEST SUMMARY: {passed}/{total} tests passed")
    print("=" * 60)
    
    if passed == total:
        print("🎉 All tests passed! Lemkin Images implementation is working correctly.")
        return 0
    else:
        print("❌ Some tests failed. Please check the implementation.")
        return 1

if __name__ == "__main__":
    sys.exit(main())