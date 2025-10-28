"""
Configuration for Historical Context & Background Researcher Agent
"""

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class HistoricalResearcherConfig:
    """Configuration for Historical Context & Background Researcher agent."""

    # Model settings
    model: str = "claude-sonnet-4-5-20250929"
    max_tokens: int = 16384  # Large for comprehensive research
    temperature: float = 0.2  # Balanced for analytical research

    # Research scope options
    include_historical_context: bool = True
    analyze_political_dynamics: bool = True
    profile_key_actors: bool = True
    assess_cultural_factors: bool = True
    examine_regional_context: bool = True
    research_legal_background: bool = True

    # Research depth settings
    max_historical_period_years: int = 100  # How far back to research
    max_actors_to_profile: int = 20
    max_analogous_cases: int = 10
    include_minor_actors: bool = False  # Focus on key actors only

    # Source types to prioritize
    prioritized_source_types: Optional[List[str]] = None

    # Analysis thresholds
    min_source_credibility: float = 0.5
    min_confidence_for_inclusion: float = 0.4
    require_source_corroboration: bool = True

    # Web search integration
    enable_web_search: bool = True
    max_web_sources: int = 50
    search_academic_sources: bool = True
    search_government_sources: bool = True
    search_news_archives: bool = True

    # Output options
    include_source_assessment: bool = True
    include_confidence_scores: bool = True
    generate_research_recommendations: bool = True
    identify_information_gaps: bool = True
    max_research_recommendations: int = 10

    # Legal focus options
    focus_on_legal_relevance: bool = True
    include_transitional_justice: bool = True
    research_previous_cases: bool = True
    assess_institutional_capacity: bool = True

    def __post_init__(self):
        """Set defaults for mutable fields."""
        if self.prioritized_source_types is None:
            self.prioritized_source_types = [
                "government_documents",
                "academic_research",
                "un_reports",
                "ngo_reports",
                "news_archives",
                "legal_proceedings",
                "witness_testimony",
                "historical_archives"
            ]


DEFAULT_CONFIG = HistoricalResearcherConfig()

# Comprehensive research configuration for major cases
COMPREHENSIVE_RESEARCH_CONFIG = HistoricalResearcherConfig(
    max_tokens=20000,  # Maximum for deep analysis
    temperature=0.15,  # Lower for maximum accuracy
    max_historical_period_years=150,
    max_actors_to_profile=30,
    max_analogous_cases=15,
    include_minor_actors=True,
    max_web_sources=100,
    require_source_corroboration=True,
    max_research_recommendations=15
)

# Focused research configuration for specific questions
TARGETED_RESEARCH_CONFIG = HistoricalResearcherConfig(
    max_tokens=12000,
    analyze_political_dynamics=True,
    profile_key_actors=True,
    assess_cultural_factors=False,  # Skip if not relevant
    examine_regional_context=False,  # Skip if not relevant
    max_actors_to_profile=10,
    max_analogous_cases=5,
    include_minor_actors=False,
    max_web_sources=30
)

# Legal-focused research configuration
LEGAL_CONTEXT_CONFIG = HistoricalResearcherConfig(
    research_legal_background=True,
    focus_on_legal_relevance=True,
    include_transitional_justice=True,
    research_previous_cases=True,
    assess_institutional_capacity=True,
    analyze_political_dynamics=True,  # Important for legal context
    examine_regional_context=True,  # For jurisdiction questions
    search_government_sources=True,
    search_academic_sources=True,
    min_source_credibility=0.7  # Higher standard for legal research
)

# Real-time research configuration for developing situations
CURRENT_EVENTS_CONFIG = HistoricalResearcherConfig(
    max_historical_period_years=20,  # Focus on recent history
    enable_web_search=True,
    search_news_archives=True,
    search_government_sources=True,
    max_web_sources=75,
    temperature=0.25,  # Slightly higher for emerging analysis
    min_confidence_for_inclusion=0.3,  # Lower to capture developing info
    require_source_corroboration=False  # May not be available for recent events
)