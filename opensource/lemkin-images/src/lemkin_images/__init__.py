"""
Lemkin Image Verification Suite

Image authenticity verification and manipulation detection for legal investigations.
"""

__version__ = "0.1.0"
__author__ = "Lemkin AI Contributors"
__email__ = "contributors@lemkin.org"

from .core import (
    ReverseImageSearcher,
    ManipulationDetector,
    GeolocationHelper,
    MetadataForensics,
    ReverseSearchResults,
    ManipulationAnalysis,
    GeolocationResult,
    ImageMetadata,
    ImageAnalysis,
)

__all__ = [
    "ReverseImageSearcher",
    "ManipulationDetector",
    "GeolocationHelper",
    "MetadataForensics",
    "ReverseSearchResults",
    "ManipulationAnalysis",
    "GeolocationResult",
    "ImageMetadata",
    "ImageAnalysis",
]