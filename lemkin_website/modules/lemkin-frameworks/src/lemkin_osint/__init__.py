"""
Lemkin OSINT Collection Toolkit

Ethical open-source intelligence gathering toolkit for digital investigations.
Implements responsible collection practices while respecting platform terms of service
and complying with the Berkeley Protocol for Digital Investigations.

Key Features:
- Ethical social media data collection within ToS limits
- Web content preservation using Wayback Machine API
- EXIF/XMP metadata extraction from images and videos  
- Source credibility assessment and verification
- Chain of custody documentation
- Berkeley Protocol compliance

Usage:
    from lemkin_osint import OSINTCollector, OSINTConfig
    
    config = OSINTConfig()
    collector = OSINTCollector(config)
    
    # Collect social media evidence
    collection = await collector.collect_social_media_evidence(
        query="search term",
        platforms=[PlatformType.TWITTER, PlatformType.REDDIT]
    )
    
    # Archive web content
    archives = await collector.archive_web_content(["https://example.com"])
    
    # Extract metadata
    metadata = collector.extract_media_metadata("/path/to/image.jpg")
    
    # Verify source credibility
    assessment = collector.verify_source_credibility(source)

License: MIT
Author: Lemkin Framework Team
"""

__version__ = "1.0.0"
__author__ = "Lemkin Framework Team"
__license__ = "MIT"
__description__ = "Ethical OSINT Collection Toolkit for Digital Investigations"

# Core imports
from .core import (
    # Main classes
    OSINTCollector,
    OSINTConfig,
    
    # Data models
    OSINTCollection,
    ArchiveCollection,
    MediaMetadata,
    CredibilityAssessment,
    Source,
    SocialMediaPost,
    WebContent,
    ArchiveEntry,
    CollectionResult,
    
    # Enums
    PlatformType,
    ContentType,
    CredibilityLevel,
    CollectionStatus,
)

# Module imports
from .social_scraper import SocialMediaScraper, ToSViolationError, RateLimitExceededError
from .web_archiver import WebArchiver, ArchiveError
from .metadata_extractor import MetadataExtractor, MetadataExtractionError
from .source_verifier import SourceVerifier, VerificationError

# Convenience functions for common operations
def create_osint_collector(config: OSINTConfig = None) -> OSINTCollector:
    """
    Create an OSINT collector with default or provided configuration
    
    Args:
        config: Optional configuration. Uses defaults if None.
        
    Returns:
        Configured OSINTCollector instance
    """
    return OSINTCollector(config or OSINTConfig())


def create_source(name: str, url: str = None, platform: PlatformType = None) -> Source:
    """
    Create a Source object with basic information
    
    Args:
        name: Source name
        url: Optional source URL
        platform: Optional platform type
        
    Returns:
        Source object
    """
    return Source(name=name, url=url, platform=platform)


def get_supported_platforms() -> list[PlatformType]:
    """
    Get list of supported social media platforms
    
    Returns:
        List of supported PlatformType values
    """
    return list(PlatformType)


def get_supported_content_types() -> list[ContentType]:
    """
    Get list of supported content types for metadata extraction
    
    Returns:
        List of supported ContentType values
    """
    return list(ContentType)


# Package metadata
__all__ = [
    # Version info
    "__version__",
    "__author__", 
    "__license__",
    "__description__",
    
    # Core classes
    "OSINTCollector",
    "OSINTConfig",
    
    # Data models
    "OSINTCollection",
    "ArchiveCollection", 
    "MediaMetadata",
    "CredibilityAssessment",
    "Source",
    "SocialMediaPost",
    "WebContent",
    "ArchiveEntry",
    "CollectionResult",
    
    # Enums
    "PlatformType",
    "ContentType",
    "CredibilityLevel",
    "CollectionStatus",
    
    # Specialized classes
    "SocialMediaScraper",
    "WebArchiver",
    "MetadataExtractor", 
    "SourceVerifier",
    
    # Exceptions
    "ToSViolationError",
    "RateLimitExceededError",
    "ArchiveError",
    "MetadataExtractionError",
    "VerificationError",
    
    # Convenience functions
    "create_osint_collector",
    "create_source",
    "get_supported_platforms",
    "get_supported_content_types",
]

# Package-level configuration
import logging

# Set up basic logging configuration for the package
_logger = logging.getLogger(__name__)
_logger.addHandler(logging.NullHandler())  # Prevent "No handler" warnings

# Berkeley Protocol compliance notice
BERKELEY_PROTOCOL_NOTICE = """
This toolkit is designed to comply with the Berkeley Protocol on Digital Open Source
Investigations. Users are responsible for ensuring their investigations follow all
applicable laws, regulations, and ethical guidelines.

Key principles:
1. Respect platform Terms of Service
2. Collect only publicly available information
3. Implement proper rate limiting and ethical practices
4. Maintain chain of custody documentation
5. Verify source credibility and assess information quality
6. Protect privacy and avoid harm to individuals
"""

def show_berkeley_protocol_notice():
    """Display Berkeley Protocol compliance notice"""
    print(BERKELEY_PROTOCOL_NOTICE)


# Initialization warnings for missing dependencies
def check_dependencies():
    """Check for optional dependencies and warn if missing"""
    optional_deps = {
        'PIL': 'Pillow - Required for image metadata extraction',
        'mutagen': 'mutagen - Required for audio metadata extraction', 
        'PyPDF2': 'PyPDF2 - Required for PDF metadata extraction',
        'docx': 'python-docx - Required for Word document metadata',
        'whois': 'python-whois - Required for domain verification',
        'aiohttp': 'aiohttp - Required for async web operations',
    }
    
    missing_deps = []
    
    for module, description in optional_deps.items():
        try:
            __import__(module)
        except ImportError:
            missing_deps.append(f"  • {description}")
    
    if missing_deps:
        _logger.warning(
            "Some optional dependencies are missing. "
            "Install them for full functionality:\n" + "\n".join(missing_deps)
        )

# Check dependencies on import (but don't fail)
try:
    check_dependencies()
except Exception:
    pass  # Don't fail if dependency check fails