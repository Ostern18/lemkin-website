"""
Configuration for Genocide Intent Analyzer Agent
"""

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class GenocideIntentAnalyzerConfig:
    """Configuration for Genocide Intent Analyzer agent."""

    # Model settings
    model: str = "gpt-4o"
    max_tokens: int = 20000  # Large for comprehensive genocide analysis
    temperature: float = 0.05  # Very low for legal precision

    # Analysis scope options
    analyze_direct_evidence: bool = True
    analyze_circumstantial_evidence: bool = True
    assess_targeting_patterns: bool = True
    evaluate_contextual_factors: bool = True
    analyze_perpetrator_intent: bool = True
    apply_jurisprudence: bool = True

    # Protected group analysis
    analyze_group_identity: bool = True
    assess_substantial_part: bool = True
    evaluate_group_targeting: bool = True
    consider_intersectional_identity: bool = True

    # Intent evidence standards
    min_intent_confidence: float = 0.7  # High threshold for genocide intent
    require_corroborating_evidence: bool = True
    assess_alternative_explanations: bool = True
    evaluate_evidence_consistency: bool = True

    # Legal framework priorities
    legal_frameworks: Optional[List[str]] = None

    # Jurisprudential analysis
    apply_icty_jurisprudence: bool = True
    apply_ictr_jurisprudence: bool = True
    apply_icj_jurisprudence: bool = True
    consider_domestic_precedents: bool = True
    max_precedent_cases: int = 15

    # Pattern analysis options
    analyze_temporal_patterns: bool = True
    assess_geographic_patterns: bool = True
    evaluate_escalation_dynamics: bool = True
    map_institutional_involvement: bool = True

    # Contextual analysis depth
    include_historical_context: bool = True
    analyze_political_context: bool = True
    assess_social_dynamics: bool = True
    evaluate_economic_factors: bool = True

    # Prevention and risk assessment
    assess_ongoing_risk: bool = True
    evaluate_prevention_indicators: bool = True
    identify_vulnerable_populations: bool = True
    recommend_prevention_measures: bool = True

    # Evidence quality requirements
    authenticate_evidence: bool = True
    assess_source_credibility: bool = True
    evaluate_chain_of_custody: bool = True
    consider_admissibility_standards: bool = True

    # Output options
    include_comparative_analysis: bool = True
    provide_legal_recommendations: bool = True
    generate_prevention_analysis: bool = True
    include_confidence_scores: bool = True
    assess_prosecution_viability: bool = True

    def __post_init__(self):
        """Set defaults for mutable fields."""
        if self.legal_frameworks is None:
            self.legal_frameworks = [
                "genocide_convention",
                "rome_statute",
                "icty_statute",
                "ictr_statute",
                "customary_international_law",
                "regional_conventions"
            ]


DEFAULT_CONFIG = GenocideIntentAnalyzerConfig()

# High-precision legal analysis configuration
LEGAL_ANALYSIS_CONFIG = GenocideIntentAnalyzerConfig(
    temperature=0.02,  # Maximum precision for legal analysis
    max_tokens=25000,  # Extended for detailed legal analysis
    min_intent_confidence=0.8,  # Very high threshold
    require_corroborating_evidence=True,
    assess_alternative_explanations=True,
    apply_jurisprudence=True,
    apply_icty_jurisprudence=True,
    apply_ictr_jurisprudence=True,
    apply_icj_jurisprudence=True,
    max_precedent_cases=20,
    assess_prosecution_viability=True,
    consider_admissibility_standards=True
)

# Prevention-focused configuration
PREVENTION_ANALYSIS_CONFIG = GenocideIntentAnalyzerConfig(
    assess_ongoing_risk=True,
    evaluate_prevention_indicators=True,
    identify_vulnerable_populations=True,
    recommend_prevention_measures=True,
    analyze_temporal_patterns=True,
    evaluate_escalation_dynamics=True,
    include_historical_context=True,
    analyze_political_context=True,
    generate_prevention_analysis=True,
    temperature=0.1  # Balanced for prevention analysis
)

# Pattern analysis configuration
PATTERN_ANALYSIS_CONFIG = GenocideIntentAnalyzerConfig(
    assess_targeting_patterns=True,
    analyze_temporal_patterns=True,
    assess_geographic_patterns=True,
    evaluate_escalation_dynamics=True,
    map_institutional_involvement=True,
    analyze_perpetrator_intent=True,
    include_historical_context=True,
    analyze_political_context=True,
    assess_social_dynamics=True,
    max_tokens=18000
)

# Evidence-focused configuration
EVIDENCE_ANALYSIS_CONFIG = GenocideIntentAnalyzerConfig(
    analyze_direct_evidence=True,
    analyze_circumstantial_evidence=True,
    min_intent_confidence=0.75,
    require_corroborating_evidence=True,
    assess_alternative_explanations=True,
    evaluate_evidence_consistency=True,
    authenticate_evidence=True,
    assess_source_credibility=True,
    evaluate_chain_of_custody=True,
    consider_admissibility_standards=True,
    temperature=0.05
)

# Comparative analysis configuration
COMPARATIVE_ANALYSIS_CONFIG = GenocideIntentAnalyzerConfig(
    include_comparative_analysis=True,
    apply_jurisprudence=True,
    apply_icty_jurisprudence=True,
    apply_ictr_jurisprudence=True,
    apply_icj_jurisprudence=True,
    consider_domestic_precedents=True,
    max_precedent_cases=25,  # Extended for comparative analysis
    include_historical_context=True,
    evaluate_contextual_factors=True,
    max_tokens=22000
)

# Rapid assessment configuration
RAPID_ASSESSMENT_CONFIG = GenocideIntentAnalyzerConfig(
    max_tokens=12000,  # Reduced for speed
    analyze_direct_evidence=True,
    analyze_circumstantial_evidence=True,
    assess_targeting_patterns=True,
    min_intent_confidence=0.6,  # Lower threshold for rapid assessment
    apply_jurisprudence=False,  # Skip detailed jurisprudence
    include_comparative_analysis=False,  # Skip for speed
    max_precedent_cases=5,  # Limited precedents
    include_historical_context=False,  # Focus on immediate evidence
    temperature=0.1
)

# Court proceedings configuration
COURT_PROCEEDINGS_CONFIG = GenocideIntentAnalyzerConfig(
    temperature=0.01,  # Maximum precision for court proceedings
    max_tokens=25000,
    min_intent_confidence=0.85,  # Highest threshold for court
    require_corroborating_evidence=True,
    assess_alternative_explanations=True,
    evaluate_evidence_consistency=True,
    authenticate_evidence=True,
    assess_source_credibility=True,
    evaluate_chain_of_custody=True,
    consider_admissibility_standards=True,
    apply_jurisprudence=True,
    assess_prosecution_viability=True,
    include_confidence_scores=True
)