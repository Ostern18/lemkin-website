"""
Configuration for Enforced Disappearance Investigator Agent
"""

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class DisappearanceInvestigatorConfig:
    """Configuration for Enforced Disappearance Investigator agent."""

    # Model settings
    model: str = "gpt-4o"
    max_tokens: int = 18000  # Large for comprehensive disappearance analysis
    temperature: float = 0.1  # Low for precise legal analysis

    # Analysis scope options
    analyze_legal_elements: bool = True
    assess_pattern_analysis: bool = True
    evaluate_state_obligations: bool = True
    analyze_family_rights: bool = True
    assess_institutional_involvement: bool = True
    evaluate_search_investigation: bool = True

    # Legal framework priorities
    legal_frameworks: Optional[List[str]] = None

    # Pattern analysis settings
    identify_systematic_patterns: bool = True
    analyze_temporal_patterns: bool = True
    assess_geographic_patterns: bool = True
    evaluate_demographic_patterns: bool = True
    map_modus_operandi: bool = True

    # State obligation analysis
    assess_prevention_obligations: bool = True
    evaluate_investigation_obligations: bool = True
    analyze_information_obligations: bool = True
    assess_remedy_obligations: bool = True

    # Family-centered analysis
    prioritize_family_rights: bool = True
    analyze_family_impact: bool = True
    assess_right_to_truth: bool = True
    evaluate_reparations_needs: bool = True
    consider_protection_needs: bool = True

    # Evidence quality requirements
    min_confidence_threshold: float = 0.6
    require_corroboration: bool = True
    assess_witness_credibility: bool = True
    evaluate_documentary_evidence: bool = True
    consider_chain_of_custody: bool = True

    # Investigation analysis
    evaluate_search_adequacy: bool = True
    assess_investigation_quality: bool = True
    identify_obstacles: bool = True
    document_good_practices: bool = True

    # Institutional analysis depth
    map_command_structures: bool = True
    analyze_coordination_mechanisms: bool = True
    assess_institutional_culture: bool = True
    evaluate_accountability_mechanisms: bool = True

    # Output options
    include_legal_implications: bool = True
    provide_recommendations: bool = True
    assess_evidence_gaps: bool = True
    generate_family_support_analysis: bool = True
    include_confidence_scores: bool = True

    def __post_init__(self):
        """Set defaults for mutable fields."""
        if self.legal_frameworks is None:
            self.legal_frameworks = [
                "icpped",  # International Convention on Enforced Disappearance
                "rome_statute",
                "inter_american_convention",
                "iccpr",
                "customary_international_law",
                "regional_human_rights_law"
            ]


DEFAULT_CONFIG = DisappearanceInvestigatorConfig()

# Family-centered analysis configuration
FAMILY_CENTERED_CONFIG = DisappearanceInvestigatorConfig(
    prioritize_family_rights=True,
    analyze_family_impact=True,
    assess_right_to_truth=True,
    evaluate_reparations_needs=True,
    consider_protection_needs=True,
    generate_family_support_analysis=True,
    analyze_information_obligations=True,
    assess_remedy_obligations=True,
    temperature=0.15,  # Slightly higher for sensitive analysis
    legal_frameworks=[
        "icpped",
        "inter_american_convention",
        "iccpr",
        "regional_human_rights_law"
    ]
)

# Pattern analysis configuration
PATTERN_ANALYSIS_CONFIG = DisappearanceInvestigatorConfig(
    assess_pattern_analysis=True,
    identify_systematic_patterns=True,
    analyze_temporal_patterns=True,
    assess_geographic_patterns=True,
    evaluate_demographic_patterns=True,
    map_modus_operandi=True,
    assess_institutional_involvement=True,
    map_command_structures=True,
    analyze_coordination_mechanisms=True,
    max_tokens=20000,  # Extended for pattern analysis
    min_confidence_threshold=0.7
)

# State obligations assessment configuration
STATE_OBLIGATIONS_CONFIG = DisappearanceInvestigatorConfig(
    evaluate_state_obligations=True,
    assess_prevention_obligations=True,
    evaluate_investigation_obligations=True,
    analyze_information_obligations=True,
    assess_remedy_obligations=True,
    evaluate_search_investigation=True,
    evaluate_search_adequacy=True,
    assess_investigation_quality=True,
    identify_obstacles=True,
    document_good_practices=True,
    temperature=0.08  # Very precise for state obligation analysis
)

# Legal proceedings configuration
LEGAL_PROCEEDINGS_CONFIG = DisappearanceInvestigatorConfig(
    temperature=0.05,  # Maximum precision for legal proceedings
    analyze_legal_elements=True,
    include_legal_implications=True,
    min_confidence_threshold=0.8,  # High threshold for legal proceedings
    require_corroboration=True,
    assess_witness_credibility=True,
    evaluate_documentary_evidence=True,
    consider_chain_of_custody=True,
    legal_frameworks=[
        "icpped",
        "rome_statute",
        "customary_international_law"
    ]
)

# Investigation quality assessment configuration
INVESTIGATION_ASSESSMENT_CONFIG = DisappearanceInvestigatorConfig(
    evaluate_search_investigation=True,
    evaluate_search_adequacy=True,
    assess_investigation_quality=True,
    identify_obstacles=True,
    document_good_practices=True,
    evaluate_investigation_obligations=True,
    assess_institutional_involvement=True,
    evaluate_accountability_mechanisms=True,
    assess_evidence_gaps=True,
    provide_recommendations=True
)

# Systematic disappearance analysis configuration
SYSTEMATIC_ANALYSIS_CONFIG = DisappearanceInvestigatorConfig(
    assess_pattern_analysis=True,
    identify_systematic_patterns=True,
    assess_institutional_involvement=True,
    map_command_structures=True,
    analyze_coordination_mechanisms=True,
    assess_institutional_culture=True,
    evaluate_state_obligations=True,
    legal_frameworks=[
        "icpped",
        "rome_statute",
        "crimes_against_humanity",
        "customary_international_law"
    ],
    max_tokens=22000,  # Extended for systematic analysis
    min_confidence_threshold=0.75
)

# Rapid assessment configuration
RAPID_ASSESSMENT_CONFIG = DisappearanceInvestigatorConfig(
    max_tokens=12000,  # Reduced for speed
    analyze_legal_elements=True,
    assess_pattern_analysis=False,  # Skip detailed pattern analysis
    evaluate_state_obligations=False,  # Skip detailed state analysis
    assess_institutional_involvement=False,  # Skip institutional analysis
    min_confidence_threshold=0.5,  # Lower threshold for rapid assessment
    require_corroboration=False,  # Skip for speed
    include_confidence_scores=False,  # Skip for speed
    temperature=0.15
)