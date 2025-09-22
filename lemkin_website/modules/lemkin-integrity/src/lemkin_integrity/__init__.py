"""
Lemkin Evidence Integrity Toolkit

This package provides cryptographic integrity verification and chain of custody
management for legal evidence to ensure admissibility in court proceedings.
"""

__version__ = "0.1.0"
__author__ = "Lemkin AI Contributors"
__email__ = "contributors@lemkin.org"

from .core import (
    EvidenceIntegrityManager,
    EvidenceMetadata,
    EvidenceHash,
    CustodyEntry,
    IntegrityReport,
    CourtManifest,
    ActionType,
    IntegrityStatus,
)

__all__ = [
    "EvidenceIntegrityManager",
    "EvidenceMetadata", 
    "EvidenceHash",
    "CustodyEntry",
    "IntegrityReport",
    "CourtManifest",
    "ActionType",
    "IntegrityStatus",
]