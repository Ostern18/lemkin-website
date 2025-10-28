"""
Configuration for Social Media Evidence Harvester Agent
"""

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class SocialMediaHarvesterConfig:
    """Configuration for Social Media Evidence Harvester agent."""

    # Model settings
    model: str = "gpt-4o"
    max_tokens: int = 8192  # Large for detailed evidence documentation
    temperature: float = 0.1  # Very low for precise evidence analysis

    # Analysis options
    perform_authenticity_assessment: bool = True
    analyze_network_patterns: bool = True
    extract_conversation_context: bool = True
    assess_legal_admissibility: bool = True
    preserve_chain_of_custody: bool = True

    # Platform types to handle
    supported_platforms: Optional[List[str]] = None

    # Authentication thresholds
    min_authenticity_confidence: float = 0.6
    bot_detection_threshold: float = 0.7
    coordination_detection_threshold: float = 0.8

    # Evidence quality requirements
    min_screenshot_quality: str = "medium"  # low|medium|high
    require_timestamp_extraction: bool = True
    require_engagement_metrics: bool = False  # Optional for evidence
    preserve_metadata: bool = True

    # Legal considerations
    flag_privacy_issues: bool = True
    assess_authentication_needs: bool = True
    document_admissibility_factors: bool = True
    track_chain_of_custody: bool = True

    # Network analysis options
    map_interaction_patterns: bool = True
    detect_coordinated_behavior: bool = True
    analyze_amplification: bool = True
    max_network_depth: int = 2  # Degrees of separation to analyze

    # Output options
    include_technical_analysis: bool = True
    include_legal_assessment: bool = True
    max_conversation_depth: int = 10  # Max thread depth to preserve
    generate_evidence_summary: bool = True

    def __post_init__(self):
        """Set defaults for mutable fields."""
        if self.supported_platforms is None:
            self.supported_platforms = [
                "Twitter",
                "Facebook",
                "Instagram",
                "TikTok",
                "LinkedIn",
                "YouTube",
                "Telegram",
                "WhatsApp",
                "Reddit",
                "Discord"
            ]


DEFAULT_CONFIG = SocialMediaHarvesterConfig()

# High-precision configuration for critical evidence
LEGAL_EVIDENCE_CONFIG = SocialMediaHarvesterConfig(
    temperature=0.05,  # Maximum precision
    min_authenticity_confidence=0.8,
    require_timestamp_extraction=True,
    preserve_metadata=True,
    assess_authentication_needs=True,
    document_admissibility_factors=True,
    track_chain_of_custody=True,
    include_legal_assessment=True,
    min_screenshot_quality="high"
)

# Network analysis configuration for coordinated behavior investigation
NETWORK_ANALYSIS_CONFIG = SocialMediaHarvesterConfig(
    analyze_network_patterns=True,
    detect_coordinated_behavior=True,
    analyze_amplification=True,
    coordination_detection_threshold=0.6,  # Lower to catch subtle coordination
    max_network_depth=3,
    map_interaction_patterns=True
)

# Rapid processing configuration for large volumes
BULK_PROCESSING_CONFIG = SocialMediaHarvesterConfig(
    max_tokens=4096,  # Smaller for faster processing
    perform_authenticity_assessment=True,
    analyze_network_patterns=False,  # Skip for speed
    extract_conversation_context=False,  # Skip for speed
    max_conversation_depth=3,  # Reduced depth
    include_technical_analysis=False  # Skip detailed technical analysis
)