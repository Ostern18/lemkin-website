"""
Lemkin PII Redaction Pipeline

This package provides automated redaction of personally identifiable information
(PII) from text, images, audio, and video content to protect witness and victim
privacy in legal investigations.
"""

__version__ = "0.1.0"
__author__ = "Lemkin AI Contributors"
__email__ = "contributors@lemkin.org"

from .core import (
    PIIRedactor,
    RedactionConfig,
    RedactionResult,
    RedactionType,
    PIIEntity,
    EntityType,
    ConfidenceLevel,
)

from .text_redactor import TextRedactor
from .image_redactor import ImageRedactor  
from .audio_redactor import AudioRedactor
from .video_redactor import VideoRedactor

__all__ = [
    "PIIRedactor",
    "RedactionConfig",
    "RedactionResult", 
    "RedactionType",
    "PIIEntity",
    "EntityType",
    "ConfidenceLevel",
    "TextRedactor",
    "ImageRedactor",
    "AudioRedactor", 
    "VideoRedactor",
]