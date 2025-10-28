"""
Configuration for Siege & Starvation Warfare Analyst Agent
"""

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class SiegeStarvationAnalystConfig:
    """Configuration for Siege & Starvation Warfare Analyst agent."""

    # Model settings
    model: str = "gpt-4o"
    max_tokens: int = 16384  # Large for comprehensive siege analysis
    temperature: float = 0.1  # Low for factual precision

    # Analysis scope options
    analyze_supply_access: bool = True
    assess_humanitarian_access: bool = True
    evaluate_population_impact: bool = True
    map_siege_infrastructure: bool = True
    analyze_legal_elements: bool = True
    assess_command_responsibility: bool = True

    # Data analysis settings
    calculate_nutrition_metrics: bool = True
    assess_health_impacts: bool = True
    analyze_supply_flow_data: bool = True
    evaluate_aid_delivery: bool = True
    track_checkpoint_restrictions: bool = True
    analyze_civilian_casualties: bool = True

    # Legal analysis settings
    legal_frameworks: Optional[List[str]] = None
    assess_war_crimes: bool = True
    evaluate_crimes_against_humanity: bool = True
    analyze_genocide_indicators: bool = True
    consider_ihl_violations: bool = True

    # Pattern analysis options
    detect_systematic_starvation: bool = True
    analyze_policy_indicators: bool = True
    assess_deliberate_denial: bool = True
    evaluate_alternative_causes: bool = True
    analyze_temporal_patterns: bool = True

    # Evidence quality requirements
    min_evidence_confidence: float = 0.5
    require_corroboration: bool = True
    assess_data_reliability: bool = True
    evaluate_witness_testimony: bool = True

    # Geographic analysis
    perform_spatial_analysis: bool = True
    map_siege_lines: bool = True
    track_checkpoint_locations: bool = True
    analyze_access_routes: bool = True
    assess_territorial_control: bool = True

    # Humanitarian impact assessment
    assess_population_needs: bool = True
    evaluate_medical_access: bool = True
    analyze_water_sanitation: bool = True
    assess_education_impact: bool = True
    evaluate_economic_destruction: bool = True

    # Output options
    include_legal_implications: bool = True
    provide_recommendations: bool = True
    assess_evidence_gaps: bool = True
    generate_policy_analysis: bool = True
    include_confidence_scores: bool = True

    def __post_init__(self):
        """Set defaults for mutable fields."""
        if self.legal_frameworks is None:
            self.legal_frameworks = [
                "rome_statute",
                "geneva_conventions",
                "additional_protocols",
                "customary_ihl",
                "siege_warfare_rules",
                "starvation_prohibitions"
            ]


DEFAULT_CONFIG = SiegeStarvationAnalystConfig()

# Humanitarian assessment configuration
HUMANITARIAN_CONFIG = SiegeStarvationAnalystConfig(
    assess_humanitarian_access=True,
    evaluate_population_impact=True,
    calculate_nutrition_metrics=True,
    assess_health_impacts=True,
    assess_population_needs=True,
    evaluate_medical_access=True,
    analyze_water_sanitation=True,
    max_tokens=20000
)

# Legal proceedings configuration
LEGAL_PROCEEDINGS_CONFIG = SiegeStarvationAnalystConfig(
    analyze_legal_elements=True,
    assess_war_crimes=True,
    evaluate_crimes_against_humanity=True,
    assess_command_responsibility=True,
    consider_ihl_violations=True,
    min_evidence_confidence=0.7,
    require_corroboration=True
)

# Policy analysis configuration
POLICY_ANALYSIS_CONFIG = SiegeStarvationAnalystConfig(
    analyze_policy_indicators=True,
    assess_deliberate_denial=True,
    detect_systematic_starvation=True,
    generate_policy_analysis=True,
    include_legal_implications=True,
    provide_recommendations=True
)
