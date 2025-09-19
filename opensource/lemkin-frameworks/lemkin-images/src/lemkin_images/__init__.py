"""
Lemkin Image Verification Suite

Comprehensive image authenticity verification and forensic analysis toolkit for legal investigations.

This package provides:
- Multi-engine reverse image search
- Advanced manipulation detection 
- Image geolocation extraction and verification
- EXIF metadata forensic analysis
- Comprehensive authenticity reporting

Legal Compliance: Meets standards for digital evidence handling in legal proceedings
"""

__version__ = "1.0.0"
__author__ = "Lemkin Digital Forensics"
__email__ = "forensics@lemkin.com"

# Core classes and functions
from .core import (
    # Main authenticator class
    ImageAuthenticator,
    
    # Configuration
    ImageAuthConfig,
    
    # Data models
    AuthenticityReport,
    ReverseSearchResults,
    ManipulationAnalysis,
    GeolocationResult,
    MetadataForensics,
    ImageMetadata,
    
    # Enums
    ImageFormat,
    ManipulationType,
    AnalysisStatus,
    SearchEngine,
    
    # Individual results
    SearchResult,
    ManipulationIndicator,
    GeolocationData,
)

# Module-specific functions
from .reverse_search import reverse_search_image, ReverseImageSearcher
from .manipulation_detector import detect_image_manipulation, ImageManipulationDetector
from .geolocation_helper import geolocate_image, ImageGeolocator
from .metadata_forensics import analyze_image_metadata, MetadataForensicsAnalyzer

__all__ = [
    # Main classes
    'ImageAuthenticator',
    'ImageAuthConfig',
    
    # Individual analyzers
    'ReverseImageSearcher',
    'ImageManipulationDetector', 
    'ImageGeolocator',
    'MetadataForensicsAnalyzer',
    
    # Data models
    'AuthenticityReport',
    'ReverseSearchResults',
    'ManipulationAnalysis',
    'GeolocationResult',
    'MetadataForensics',
    'ImageMetadata',
    'SearchResult',
    'ManipulationIndicator',
    'GeolocationData',
    
    # Enums
    'ImageFormat',
    'ManipulationType',
    'AnalysisStatus',
    'SearchEngine',
    
    # Convenience functions
    'reverse_search_image',
    'detect_image_manipulation',
    'geolocate_image', 
    'analyze_image_metadata',
]

# Package metadata
PACKAGE_INFO = {
    'name': 'lemkin-images',
    'version': __version__,
    'description': 'Comprehensive image authenticity verification toolkit',
    'author': __author__,
    'author_email': __email__,
    'license': 'MIT',
    'url': 'https://github.com/lemkin/lemkin-frameworks',
    'keywords': ['forensics', 'image-analysis', 'authenticity', 'manipulation-detection'],
    'classifiers': [
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Legal Industry',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Topic :: Scientific/Engineering :: Image Recognition',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
}

def get_version():
    """Get package version"""
    return __version__

def get_package_info():
    """Get complete package information"""
    return PACKAGE_INFO.copy()