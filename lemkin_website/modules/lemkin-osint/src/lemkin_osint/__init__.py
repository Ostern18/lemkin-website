"""
Lemkin OSINT Collection Toolkit

Systematic open-source intelligence gathering while respecting platform terms of service
for legal investigations and human rights work.
"""

__version__ = "0.1.0"
__author__ = "Lemkin AI Contributors"
__email__ = "contributors@lemkin.org"

from .core import (
    OSINTCollector,
    WebArchiver,
    MetadataExtractor,
    SourceVerifier,
    OSINTCollection,
    ArchiveCollection,
    MediaMetadata,
    CredibilityAssessment,
    Source,
)

__all__ = [
    "OSINTCollector",
    "WebArchiver",
    "MetadataExtractor",
    "SourceVerifier",
    "OSINTCollection",
    "ArchiveCollection",
    "MediaMetadata",
    "CredibilityAssessment",
    "Source",
]