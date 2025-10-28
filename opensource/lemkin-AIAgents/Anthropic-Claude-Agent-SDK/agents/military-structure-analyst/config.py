"""
Configuration for Military Structure & Tactics Analyst Agent
"""

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class MilitaryStructureAnalystConfig:
    """Configuration for Military Structure & Tactics Analyst agent."""

    # Model settings
    model: str = "claude-sonnet-4-5-20250929"
    max_tokens: int = 14000
    temperature: float = 0.1

    # Analysis scope
    analyze_unit_structures: bool = True
    map_command_hierarchies: bool = True
    explain_tactical_operations: bool = True
    assess_military_doctrine: bool = True
    analyze_attack_patterns: bool = True
    identify_command_control: bool = True

    # Command responsibility
    map_authority_relationships: bool = True
    assess_commander_knowledge: bool = True
    identify_decision_makers: bool = True
    analyze_orders_policies: bool = True

    # Tactical analysis
    evaluate_military_necessity: bool = True
    assess_proportionality: bool = True
    analyze_precautions: bool = True
    identify_ihl_violations: bool = True

    # Output options
    generate_structure_diagrams: bool = True
    provide_expert_consultation: bool = True
    include_doctrine_analysis: bool = True
    assess_training_indicators: bool = True
    include_confidence_scores: bool = True

    # Evidence thresholds
    min_analysis_confidence: float = 0.5
    require_corroboration: bool = True


DEFAULT_CONFIG = MilitaryStructureAnalystConfig()

# Command responsibility configuration
COMMAND_RESPONSIBILITY_CONFIG = MilitaryStructureAnalystConfig(
    map_command_hierarchies=True,
    map_authority_relationships=True,
    assess_commander_knowledge=True,
    identify_decision_makers=True,
    analyze_orders_policies=True,
    min_analysis_confidence=0.7,
    max_tokens=16000
)

# Tactical analysis configuration
TACTICAL_ANALYSIS_CONFIG = MilitaryStructureAnalystConfig(
    explain_tactical_operations=True,
    assess_military_doctrine=True,
    analyze_attack_patterns=True,
    evaluate_military_necessity=True,
    assess_proportionality=True,
    analyze_precautions=True,
    temperature=0.05
)
