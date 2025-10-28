"""
Configuration for Satellite Imagery Analyst Agent
"""

from dataclasses import dataclass


@dataclass
class ImageryAnalystConfig:
    """Configuration for satellite imagery analyst."""

    # Model settings
    model: str = "gpt-4o"
    max_tokens: int = 8192
    temperature: float = 0.1  # Low for precision

    # Analysis options
    perform_change_detection: bool = True
    estimate_measurements: bool = True
    identify_military_indicators: bool = True
    assess_damage: bool = True
    provide_geolocation_assistance: bool = True

    # Confidence thresholds
    min_confidence_for_definitive: float = 0.85
    expert_review_threshold: float = 0.6

    # Output options
    include_annotations: bool = True
    include_alternative_interpretations: bool = True


DEFAULT_CONFIG = ImageryAnalystConfig()

MASS_GRAVE_ASSESSMENT_CONFIG = ImageryAnalystConfig(
    temperature=0.0,
    min_confidence_for_definitive=0.9,
    expert_review_threshold=0.7,
    include_alternative_interpretations=True
)
