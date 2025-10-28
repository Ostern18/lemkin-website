"""
Configuration for Evidence Gap & Next Steps Identifier Agent
"""

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class GapIdentifierConfig:
    """Configuration for gap identifier agent."""

    # Model settings
    model: str = "claude-sonnet-4-5-20250929"
    max_tokens: int = 12288  # Large for comprehensive analysis
    temperature: float = 0.2  # Slightly higher for creative problem-solving

    # Analysis options
    assess_legal_elements: bool = True
    generate_interview_questions: bool = True
    suggest_document_requests: bool = True
    recommend_expert_consultations: bool = True
    identify_alternative_strategies: bool = True
    assess_risks: bool = True

    # Prioritization
    max_priority_actions: int = 10  # Limit recommendations to avoid overwhelm
    focus_on_critical_gaps: bool = True

    # Supported charge types for legal element mapping
    supported_charge_types: Optional[List[str]] = None

    # Output options
    include_timeline_recommendations: bool = True
    include_resource_estimates: bool = True

    def __post_init__(self):
        """Set defaults for mutable fields."""
        if self.supported_charge_types is None:
            self.supported_charge_types = [
                "torture",
                "war_crimes",
                "crimes_against_humanity",
                "genocide",
                "murder",
                "assault",
                "sexual_violence",
                "enforced_disappearance",
                "unlawful_detention",
                "destruction_of_property",
                "pillage",
                "attacks_on_civilians",
                "use_of_prohibited_weapons",
                "child_recruitment",
                "starvation_as_weapon"
            ]


DEFAULT_CONFIG = GapIdentifierConfig()

FOCUSED_CONFIG = GapIdentifierConfig(
    max_priority_actions=5,
    focus_on_critical_gaps=True,
    assess_risks=True
)
