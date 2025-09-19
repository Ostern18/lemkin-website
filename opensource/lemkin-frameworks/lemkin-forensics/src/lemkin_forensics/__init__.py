"""
Lemkin Digital Forensics Helpers

This package provides digital forensics analysis tools for legal investigators,
making complex forensics procedures accessible to non-technical professionals.
"""

__version__ = "0.1.0"
__author__ = "Lemkin AI Contributors"
__email__ = "contributors@lemkin.org"

from .core import (
    DigitalForensicsAnalyzer,
    ForensicsConfig,
    DigitalEvidence,
    AnalysisResult,
    EvidenceType,
    AnalysisStatus,
)

from .file_analyzer import FileAnalyzer, FileSystemAnalysis, FileArtifact
from .network_processor import NetworkProcessor, NetworkAnalysis, NetworkFlow
from .mobile_analyzer import MobileAnalyzer, MobileDataExtraction, MobileArtifact
from .authenticity_verifier import AuthenticityVerifier, AuthenticityReport

__all__ = [
    "DigitalForensicsAnalyzer",
    "ForensicsConfig",
    "DigitalEvidence",
    "AnalysisResult",
    "EvidenceType",
    "AnalysisStatus",
    "FileAnalyzer",
    "FileSystemAnalysis",
    "FileArtifact",
    "NetworkProcessor",
    "NetworkAnalysis", 
    "NetworkFlow",
    "MobileAnalyzer",
    "MobileDataExtraction",
    "MobileArtifact",
    "AuthenticityVerifier",
    "AuthenticityReport",
]