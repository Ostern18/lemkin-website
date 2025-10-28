"""
Configuration for Torture & Ill-Treatment Analyst Agent
"""

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class TortureAnalystConfig:
    """Configuration for Torture & Ill-Treatment Analyst agent."""

    # Model settings
    model: str = "claude-sonnet-4-5-20250929"
    max_tokens: int = 16384  # Large for comprehensive torture analysis
    temperature: float = 0.1  # Very low for medical/legal precision

    # Analysis scope options
    apply_istanbul_protocol: bool = True
    analyze_legal_elements: bool = True
    assess_medical_evidence: bool = True
    evaluate_detention_conditions: bool = True
    analyze_perpetrator_responsibility: bool = True
    identify_systematic_patterns: bool = True

    # Medical analysis settings
    require_medical_consistency: bool = True
    assess_psychological_evidence: bool = True
    evaluate_expert_testimony: bool = True
    consider_cultural_factors: bool = True
    min_medical_confidence: float = 0.6

    # Legal analysis settings
    legal_frameworks: Optional[List[str]] = None
    assess_command_responsibility: bool = True
    evaluate_state_responsibility: bool = True
    analyze_victim_rights: bool = True
    consider_statute_limitations: bool = True

    # Pattern analysis options
    detect_systematic_torture: bool = True
    analyze_institutional_practices: bool = True
    map_torture_methods: bool = True
    assess_training_indicators: bool = True
    evaluate_policy_evidence: bool = True

    # Evidence quality requirements
    min_evidence_confidence: float = 0.5
    require_corroboration: bool = True
    assess_witness_credibility: bool = True
    evaluate_documentation_quality: bool = True

    # Victim-centered considerations
    prioritize_victim_welfare: bool = True
    consider_trauma_impact: bool = True
    assess_victim_support_needs: bool = True
    maintain_confidentiality: bool = True

    # Output options
    include_legal_implications: bool = True
    provide_recommendations: bool = True
    assess_evidence_gaps: bool = True
    generate_expert_opinion: bool = True
    include_confidence_scores: bool = True

    def __post_init__(self):
        """Set defaults for mutable fields."""
        if self.legal_frameworks is None:
            self.legal_frameworks = [
                "convention_against_torture",
                "istanbul_protocol",
                "rome_statute",
                "geneva_conventions",
                "iccpr",
                "regional_conventions"
            ]


DEFAULT_CONFIG = TortureAnalystConfig()

# High-precision medical analysis configuration
MEDICAL_ANALYSIS_CONFIG = TortureAnalystConfig(
    temperature=0.05,  # Maximum precision for medical analysis
    max_tokens=20000,  # Extended for detailed medical analysis
    apply_istanbul_protocol=True,
    assess_medical_evidence=True,
    assess_psychological_evidence=True,
    evaluate_expert_testimony=True,
    consider_cultural_factors=True,
    min_medical_confidence=0.7,  # Higher threshold for medical findings
    require_corroboration=True,
    evaluate_documentation_quality=True
)

# Legal proceedings configuration
LEGAL_PROCEEDINGS_CONFIG = TortureAnalystConfig(
    analyze_legal_elements=True,
    assess_command_responsibility=True,
    evaluate_state_responsibility=True,
    analyze_victim_rights=True,
    consider_statute_limitations=True,
    include_legal_implications=True,
    min_evidence_confidence=0.7,  # Higher threshold for legal proceedings
    require_corroboration=True,
    assess_witness_credibility=True,
    generate_expert_opinion=True
)

# Systematic torture investigation configuration
SYSTEMATIC_ANALYSIS_CONFIG = TortureAnalystConfig(
    identify_systematic_patterns=True,
    analyze_institutional_practices=True,
    map_torture_methods=True,
    assess_training_indicators=True,
    evaluate_policy_evidence=True,
    detect_systematic_torture=True,
    analyze_perpetrator_responsibility=True,
    max_tokens=18000,  # Large for pattern analysis
    legal_frameworks=[
        "convention_against_torture",
        "rome_statute",
        "crimes_against_humanity",
        "geneva_conventions"
    ]
)

# Victim-centered analysis configuration
VICTIM_CENTERED_CONFIG = TortureAnalystConfig(
    prioritize_victim_welfare=True,
    consider_trauma_impact=True,
    assess_victim_support_needs=True,
    maintain_confidentiality=True,
    assess_psychological_evidence=True,
    consider_cultural_factors=True,
    analyze_victim_rights=True,
    temperature=0.15,  # Slightly higher for sensitive analysis
    provide_recommendations=True,
    include_legal_implications=True
)

# Rapid assessment configuration
RAPID_ASSESSMENT_CONFIG = TortureAnalystConfig(
    max_tokens=8192,  # Reduced for speed
    analyze_legal_elements=True,
    assess_medical_evidence=True,
    apply_istanbul_protocol=False,  # Skip detailed protocol application
    evaluate_detention_conditions=False,  # Skip for speed
    identify_systematic_patterns=False,  # Focus on individual case
    assess_evidence_gaps=False,  # Skip detailed gap analysis
    min_evidence_confidence=0.4,  # Lower threshold for rapid assessment
    require_corroboration=False
)

# Expert testimony configuration
EXPERT_TESTIMONY_CONFIG = TortureAnalystConfig(
    temperature=0.05,  # Maximum precision for expert testimony
    apply_istanbul_protocol=True,
    assess_medical_evidence=True,
    assess_psychological_evidence=True,
    evaluate_expert_testimony=True,
    analyze_legal_elements=True,
    generate_expert_opinion=True,
    include_confidence_scores=True,
    min_medical_confidence=0.8,  # Very high threshold for expert testimony
    min_evidence_confidence=0.7,
    require_corroboration=True,
    evaluate_documentation_quality=True
)