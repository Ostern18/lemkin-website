"""
Lemkin Video Authentication Toolkit

A comprehensive toolkit for video authenticity verification and manipulation detection,
designed specifically for legal professionals and digital forensics investigators.

Key Features:
- Deepfake detection using state-of-the-art ML models
- Video fingerprinting for duplicate detection
- Compression analysis for authenticity verification
- Frame-level tampering detection
- Metadata and EXIF analysis
- Timeline inconsistency detection
- Chain of custody preservation

Compliance: Berkeley Protocol for Digital Investigations
"""

from .core import (
    VideoAuthenticator,
    VideoAuthConfig,
    DeepfakeAnalysis,
    VideoFingerprint,
    CompressionAnalysis,
    KeyFrame,
    VideoMetadata,
    AuthenticityReport,
    TamperingIndicator,
    AnalysisResult,
    AuthenticityLevel,
    TamperingType,
    AnalysisStatus,
    CompressionLevel,
    FrameType,
)

from .deepfake_detector import DeepfakeDetector
from .video_fingerprinter import VideoFingerprinter
from .compression_analyzer import CompressionAnalyzer
from .frame_analyzer import FrameAnalyzer

__version__ = "0.1.0"
__author__ = "Lemkin AI Contributors"
__email__ = "contributors@lemkin.org"

__all__ = [
    # Core classes
    "VideoAuthenticator",
    "VideoAuthConfig",
    
    # Analysis components
    "DeepfakeDetector",
    "VideoFingerprinter", 
    "CompressionAnalyzer",
    "FrameAnalyzer",
    
    # Data models
    "DeepfakeAnalysis",
    "VideoFingerprint",
    "CompressionAnalysis",
    "KeyFrame",
    "VideoMetadata",
    "AuthenticityReport",
    "TamperingIndicator",
    "AnalysisResult",
    
    # Enums
    "AuthenticityLevel",
    "TamperingType",
    "AnalysisStatus",
    "CompressionLevel",
    "FrameType",
]