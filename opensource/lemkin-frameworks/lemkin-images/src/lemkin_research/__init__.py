"""
Lemkin Legal Research Assistant

Comprehensive legal research and precedent analysis toolkit for legal professionals.

This package provides:
- Legal database search and retrieval (Westlaw, LexisNexis, Google Scholar)
- Case law similarity and relevance analysis
- Legal citation parsing and validation (Bluebook, ALWD formats)
- Multi-source legal research compilation
- Precedent ranking and relevance scoring
- Legal memo generation

Legal Compliance: Designed for professional legal research workflows
"""

__version__ = "1.0.0"
__author__ = "Lemkin Legal Research"
__email__ = "research@lemkin.com"

# Core classes and functions
from .core import (
    # Main research assistant class
    LegalResearchAssistant,
    
    # Configuration
    ResearchConfig,
    
    # Data models
    CaseLawResults,
    SimilarCase,
    LegalCitation,
    ResearchSummary,
    CaseOpinion,
    Precedent,
    CitationFormat,
    LegalDatabase,
    SearchQuery,
    LegalMemo,
    
    # Enums
    CitationType,
    JurisdictionType,
    DatabaseType,
    ResearchStatus,
    RelevanceScore,
    CitationStyle,
    
    # Individual results
    DatabaseResult,
    PrecedentMatch,
    CitationMatch,
    
    # Convenience functions
    create_research_assistant,
    create_default_config,
)

# Module-specific functions
from .case_law_searcher import search_case_law, CaseLawSearcher
from .precedent_analyzer import find_similar_precedents, PrecedentAnalyzer
from .citation_processor import parse_legal_citations, CitationProcessor
from .research_aggregator import aggregate_research, ResearchAggregator

__all__ = [
    # Main classes
    'LegalResearchAssistant',
    'ResearchConfig',
    
    # Individual analyzers
    'CaseLawSearcher',
    'PrecedentAnalyzer',
    'CitationProcessor',
    'ResearchAggregator',
    
    # Data models
    'CaseLawResults',
    'SimilarCase',
    'LegalCitation',
    'ResearchSummary',
    'CaseOpinion',
    'Precedent',
    'CitationFormat',
    'LegalDatabase',
    'SearchQuery',
    'LegalMemo',
    'DatabaseResult',
    'PrecedentMatch',
    'CitationMatch',
    
    # Enums
    'CitationType',
    'JurisdictionType',
    'DatabaseType',
    'ResearchStatus',
    'RelevanceScore',
    'CitationStyle',
    
    # Convenience functions
    'search_case_law',
    'find_similar_precedents',
    'parse_legal_citations',
    'aggregate_research',
    'create_research_assistant',
    'create_default_config',
]

# Package metadata
PACKAGE_INFO = {
    'name': 'lemkin-research',
    'version': __version__,
    'description': 'Comprehensive legal research and precedent analysis toolkit',
    'author': __author__,
    'author_email': __email__,
    'license': 'MIT',
    'url': 'https://github.com/lemkin/lemkin-frameworks',
    'keywords': ['legal-research', 'case-law', 'precedent-analysis', 'citation-parsing'],
    'classifiers': [
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Legal Industry',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Topic :: Text Processing :: Linguistic',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
}

def get_version():
    """Get package version"""
    return __version__

def get_package_info():
    """Get complete package information"""
    return PACKAGE_INFO.copy()