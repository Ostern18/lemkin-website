"""
Lemkin Audio Analysis Toolkit

Audio analysis, speech transcription, and authentication for legal investigations.
"""

__version__ = "0.1.0"
__author__ = "Lemkin AI Contributors"
__email__ = "contributors@lemkin.org"

from .core import (
    SpeechTranscriber,
    SpeakerIdentifier,
    AudioAuthenticator,
    AudioEnhancer,
    AudioAnalysis,
    TranscriptionResult,
    SpeakerProfile,
    AuthenticityReport,
    EnhancementResult,
)

__all__ = [
    "SpeechTranscriber",
    "SpeakerIdentifier",
    "AudioAuthenticator",
    "AudioEnhancer",
    "AudioAnalysis",
    "TranscriptionResult",
    "SpeakerProfile",
    "AuthenticityReport",
    "EnhancementResult",
]