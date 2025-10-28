"""
Configuration for Ballistics & Weapons Identifier Agent
"""

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class BallisticsWeaponsIdentifierConfig:
    """Configuration for Ballistics & Weapons Identifier agent."""

    # Model settings
    model: str = "gpt-4o"
    max_tokens: int = 10000
    temperature: float = 0.1

    # Analysis scope
    identify_weapons: bool = True
    analyze_ammunition: bool = True
    assess_ballistics: bool = True
    evaluate_wound_patterns: bool = True
    trace_weapon_origins: bool = True
    analyze_weapon_markings: bool = True

    # Visual analysis
    process_weapon_images: bool = True
    analyze_ammunition_markings: bool = True
    identify_modifications: bool = True
    assess_weapon_condition: bool = True

    # Legal integration
    link_weapons_to_incidents: bool = True
    link_weapons_to_actors: bool = True
    assess_weapon_legality: bool = True
    evaluate_targeting: bool = True

    # Output options
    generate_visual_guides: bool = True
    provide_technical_specifications: bool = True
    include_confidence_scores: bool = True
    generate_identification_reports: bool = True

    # Evidence thresholds
    min_identification_confidence: float = 0.6


DEFAULT_CONFIG = BallisticsWeaponsIdentifierConfig()

# Investigation configuration
INVESTIGATION_CONFIG = BallisticsWeaponsIdentifierConfig(
    identify_weapons=True,
    trace_weapon_origins=True,
    link_weapons_to_incidents=True,
    link_weapons_to_actors=True,
    max_tokens=12000
)

# Technical analysis configuration
TECHNICAL_ANALYSIS_CONFIG = BallisticsWeaponsIdentifierConfig(
    analyze_ammunition=True,
    analyze_weapon_markings=True,
    identify_modifications=True,
    provide_technical_specifications=True,
    temperature=0.05
)
