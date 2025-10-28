"""
Configuration for OSINT Synthesis Agent
"""

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class OSINTConfig:
    """Configuration for OSINT synthesis agent."""

    # Model settings
    model: str = "claude-sonnet-4-5-20250929"
    max_tokens: int = 12288  # Large for comprehensive analysis
    temperature: float = 0.2  # Slightly higher for analysis

    # Analysis options
    perform_credibility_assessment: bool = True
    detect_coordination: bool = True
    analyze_temporal_patterns: bool = True
    generate_geographic_heat_map: bool = True
    identify_narratives: bool = True

    # Source types to monitor
    monitored_source_types: Optional[List[str]] = None

    # Verification thresholds
    min_credibility_score: float = 0.4
    min_verification_confidence: float = 0.6
    coordination_detection_threshold: float = 0.7

    # Red flag triggers
    flag_bot_activity: bool = True
    flag_disinformation: bool = True
    flag_coordinated_campaigns: bool = True

    # Output options
    include_source_urls: bool = True
    include_actor_profiles: bool = True
    max_claims_in_output: int = 50

    def __post_init__(self):
        """Set defaults for mutable fields."""
        if self.monitored_source_types is None:
            self.monitored_source_types = [
                "social_media",
                "news_outlets",
                "official_sources",
                "forums",
                "blogs",
                "video_platforms",
                "messaging_apps"
            ]


DEFAULT_CONFIG = OSINTConfig()

# Focused configuration for specific monitoring
TARGETED_MONITORING_CONFIG = OSINTConfig(
    temperature=0.1,
    min_credibility_score=0.6,
    min_verification_confidence=0.75,
    flag_bot_activity=True,
    flag_coordinated_campaigns=True
)

# Real-time monitoring configuration
REAL_TIME_CONFIG = OSINTConfig(
    max_tokens=8192,
    analyze_temporal_patterns=True,
    generate_geographic_heat_map=True,
    min_credibility_score=0.3  # Lower to catch emerging info
)
