"""
Lemkin Legal Research and Citation Analysis Toolkit

Comprehensive legal research capabilities for finding precedents, analyzing citations,
and conducting legal database searches for investigations.
"""

__version__ = "0.1.0"
__author__ = "Lemkin AI Contributors"
__email__ = "contributors@lemkin.org"

from .core import (
    LegalResearcher,
    CitationAnalyzer,
    CaseLawSearcher,
    RegulatorySearcher,
    LegalResearchResult,
    CitationAnalysis,
    CaseSearchResult,
    RegulatoryResult,
    LegalDocument,
    Citation,
)

__all__ = [
    "LegalResearcher",
    "CitationAnalyzer",
    "CaseLawSearcher",
    "RegulatorySearcher",
    "LegalResearchResult",
    "CitationAnalysis",
    "CaseSearchResult",
    "RegulatoryResult",
    "LegalDocument",
    "Citation",
]