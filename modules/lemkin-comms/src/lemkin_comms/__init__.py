"""
Lemkin Communication Analysis Toolkit

Communication pattern analysis, network mapping, and correlation for legal investigations.
"""

__version__ = "0.1.0"
__author__ = "Lemkin AI Contributors"
__email__ = "contributors@lemkin.org"

from .core import (
    CommunicationAnalyzer,
    NetworkMapper,
    PatternDetector,
    TimelineCorrelator,
    CommunicationAnalysis,
    NetworkAnalysis,
    PatternResult,
    TimelineCorrelation,
    Contact,
    Communication,
)

__all__ = [
    "CommunicationAnalyzer",
    "NetworkMapper",
    "PatternDetector",
    "TimelineCorrelator",
    "CommunicationAnalysis",
    "NetworkAnalysis",
    "PatternResult",
    "TimelineCorrelation",
    "Contact",
    "Communication",
]