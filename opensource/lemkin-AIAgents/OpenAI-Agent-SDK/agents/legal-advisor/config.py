"""
Configuration for Legal Framework & Jurisdiction Advisor Agent
"""

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class LegalAdvisorConfig:
    """Configuration for Legal Framework & Jurisdiction Advisor agent."""

    # Model settings
    model: str = "gpt-4o"
    max_tokens: int = 16384  # Large for comprehensive legal analysis
    temperature: float = 0.1  # Very low for precise legal analysis

    # Analysis scope options
    analyze_international_law: bool = True
    analyze_domestic_law: bool = True
    assess_jurisdiction: bool = True
    map_legal_elements: bool = True
    analyze_precedents: bool = True
    develop_strategies: bool = True

    # Legal framework priorities
    legal_frameworks: Optional[List[str]] = None

    # Jurisdictional analysis options
    consider_icc_jurisdiction: bool = True
    analyze_universal_jurisdiction: bool = True
    assess_domestic_courts: bool = True
    examine_regional_courts: bool = True
    evaluate_hybrid_mechanisms: bool = False

    # Analysis depth settings
    max_precedents_analyzed: int = 15
    max_alternative_strategies: int = 8
    include_procedural_analysis: bool = True
    include_political_considerations: bool = True

    # Evidence and procedure focus
    assess_evidence_admissibility: bool = True
    analyze_victim_participation: bool = True
    consider_immunity_issues: bool = True
    evaluate_cooperation_requirements: bool = True

    # Strategic considerations
    prioritize_victim_interests: bool = True
    consider_practical_feasibility: bool = True
    assess_political_risks: bool = True
    evaluate_timing_factors: bool = True

    # Quality requirements
    min_legal_confidence: float = 0.6
    require_precedent_support: bool = True
    mandate_alternative_analysis: bool = True
    include_risk_assessment: bool = True

    # Output options
    include_implementation_steps: bool = True
    generate_research_recommendations: bool = True
    provide_confidence_scores: bool = True
    include_timeline_analysis: bool = True

    def __post_init__(self):
        """Set defaults for mutable fields."""
        if self.legal_frameworks is None:
            self.legal_frameworks = [
                "international_criminal_law",
                "international_humanitarian_law",
                "international_human_rights_law",
                "domestic_criminal_law",
                "constitutional_law",
                "regional_human_rights_law"
            ]


DEFAULT_CONFIG = LegalAdvisorConfig()

# Comprehensive legal analysis configuration
COMPREHENSIVE_ANALYSIS_CONFIG = LegalAdvisorConfig(
    max_tokens=20000,  # Maximum for detailed analysis
    temperature=0.05,  # Extremely precise
    max_precedents_analyzed=25,
    max_alternative_strategies=12,
    include_procedural_analysis=True,
    include_political_considerations=True,
    assess_evidence_admissibility=True,
    consider_immunity_issues=True,
    evaluate_cooperation_requirements=True,
    include_timeline_analysis=True,
    min_legal_confidence=0.7
)

# ICC-focused analysis configuration
ICC_FOCUSED_CONFIG = LegalAdvisorConfig(
    consider_icc_jurisdiction=True,
    analyze_universal_jurisdiction=False,  # Focus on ICC
    assess_domestic_courts=True,  # For complementarity
    examine_regional_courts=False,
    legal_frameworks=[
        "international_criminal_law",
        "rome_statute",
        "domestic_criminal_law"  # For complementarity analysis
    ],
    include_procedural_analysis=True,
    evaluate_cooperation_requirements=True,
    assess_political_risks=True
)

# Domestic prosecution configuration
DOMESTIC_PROSECUTION_CONFIG = LegalAdvisorConfig(
    consider_icc_jurisdiction=False,
    analyze_universal_jurisdiction=True,
    assess_domestic_courts=True,
    examine_regional_courts=True,
    analyze_domestic_law=True,
    legal_frameworks=[
        "domestic_criminal_law",
        "constitutional_law",
        "international_criminal_law",  # For universal jurisdiction
        "international_human_rights_law"
    ],
    consider_immunity_issues=True,
    include_political_considerations=True
)

# Strategic planning configuration
STRATEGIC_PLANNING_CONFIG = LegalAdvisorConfig(
    develop_strategies=True,
    include_political_considerations=True,
    assess_political_risks=True,
    evaluate_timing_factors=True,
    consider_practical_feasibility=True,
    prioritize_victim_interests=True,
    max_alternative_strategies=10,
    include_implementation_steps=True,
    evaluate_cooperation_requirements=True,
    include_timeline_analysis=True
)

# Quick legal assessment configuration
RAPID_ASSESSMENT_CONFIG = LegalAdvisorConfig(
    max_tokens=8192,  # Reduced for speed
    max_precedents_analyzed=8,
    max_alternative_strategies=5,
    include_procedural_analysis=False,  # Skip for speed
    include_political_considerations=False,  # Focus on law only
    evaluate_cooperation_requirements=False,
    include_timeline_analysis=False,
    mandate_alternative_analysis=False
)

# Human rights focus configuration
HUMAN_RIGHTS_FOCUS_CONFIG = LegalAdvisorConfig(
    legal_frameworks=[
        "international_human_rights_law",
        "regional_human_rights_law",
        "constitutional_law",
        "domestic_human_rights_law"
    ],
    examine_regional_courts=True,
    analyze_victim_participation=True,
    prioritize_victim_interests=True,
    consider_icc_jurisdiction=False,  # Focus on HR mechanisms
    analyze_universal_jurisdiction=False,
    assess_domestic_courts=True,
    include_procedural_analysis=True
)