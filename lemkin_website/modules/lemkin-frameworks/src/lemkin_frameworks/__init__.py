"""
Lemkin Legal Framework Mapper

This package provides mapping of evidence to specific legal framework elements
for international law violation assessment and legal analysis.
"""

__version__ = "0.1.0"
__author__ = "Lemkin AI Contributors"
__email__ = "contributors@lemkin.org"

from .core import (
    LegalFrameworkMapper,
    FrameworkConfig,
    Evidence,
    LegalElement,
    FrameworkAnalysis,
    LegalAssessment,
    ElementSatisfaction,
    GapAnalysis,
    LegalFramework,
    ViolationType,
    SatisfactionLevel,
    EvidenceType,
    ConfidenceLevel,
    ElementStatus,
)

from .rome_statute import RomeStatuteAnalyzer, RomeStatuteAnalysis
from .geneva_conventions import GenevaAnalyzer, GenevaAnalysis
from .human_rights_frameworks import HumanRightsAnalyzer, HumanRightsAnalysis
from .element_analyzer import ElementAnalyzer, ConfidenceScore, EvidenceRelevance

__all__ = [
    "LegalFrameworkMapper",
    "FrameworkConfig",
    "Evidence",
    "LegalElement",
    "FrameworkAnalysis",
    "LegalAssessment",
    "ElementSatisfaction",
    "GapAnalysis",
    "LegalFramework",
    "ViolationType",
    "SatisfactionLevel",
    "EvidenceType",
    "ConfidenceLevel",
    "ElementStatus",
    "RomeStatuteAnalyzer",
    "RomeStatuteAnalysis",
    "GenevaAnalyzer", 
    "GenevaAnalysis",
    "HumanRightsAnalyzer",
    "HumanRightsAnalysis",
    "ElementAnalyzer",
    "ConfidenceScore",
    "EvidenceRelevance",
]