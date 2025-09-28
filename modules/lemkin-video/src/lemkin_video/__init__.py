"""
Lemkin Video Authentication Toolkit

Video authenticity verification and manipulation detection for legal investigations.
"""

__version__ = "0.1.0"
__author__ = "Lemkin AI Contributors"
__email__ = "contributors@lemkin.org"

from .core import (
    DeepfakeDetector,
    VideoFingerprinter,
    CompressionAnalyzer,
    FrameAnalyzer,
    DeepfakeAnalysis,
    VideoFingerprint,
    CompressionAnalysis,
    KeyFrame,
    VideoMetadata,
    AuthenticityReport,
)

__all__ = [
    "DeepfakeDetector",
    "VideoFingerprinter",
    "CompressionAnalyzer",
    "FrameAnalyzer",
    "DeepfakeAnalysis",
    "VideoFingerprint",
    "CompressionAnalysis",
    "KeyFrame",
    "VideoMetadata",
    "AuthenticityReport",
]