"""
Configuration for OSINT Synthesis Agent
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class OSINTConfig:
    """Configuration for OSINT Synthesis Agent."""

    # Model configuration
    model: str = "gpt-4o"
    temperature: float = 0.2  # Low temperature for consistent analysis
    max_tokens: int = 4096

    # Credibility thresholds
    high_credibility_threshold: float = 0.8
    medium_credibility_threshold: float = 0.5

    # Analysis settings
    require_human_review_below_credibility: float = 0.6
    minimum_sources_for_verification: int = 2

    # Output settings
    include_source_links: bool = True
    generate_timeline: bool = True
    generate_heatmap_data: bool = True


DEFAULT_CONFIG = OSINTConfig()
