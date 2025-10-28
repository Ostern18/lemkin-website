"""
Configuration for Digital Forensics & Metadata Analyst Agent
"""

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class DigitalForensicsAnalystConfig:
    """Configuration for Digital Forensics & Metadata Analyst agent."""

    # Model settings
    model: str = "claude-sonnet-4-5-20250929"
    max_tokens: int = 16384  # Large for comprehensive forensics analysis
    temperature: float = 0.05  # Very low for technical precision

    # Analysis scope options
    extract_metadata: bool = True
    verify_authenticity: bool = True
    detect_manipulation: bool = True
    reconstruct_timeline: bool = True
    analyze_communications: bool = True
    perform_comparison_analysis: bool = True

    # Metadata extraction settings
    extract_exif_data: bool = True
    extract_file_system_metadata: bool = True
    extract_application_metadata: bool = True
    extract_device_metadata: bool = True
    extract_gps_data: bool = True
    extract_user_metadata: bool = True

    # Authentication and integrity
    calculate_file_hashes: bool = True
    verify_digital_signatures: bool = True
    detect_steganography: bool = True
    analyze_compression_artifacts: bool = True
    assess_provenance: bool = True

    # Hash algorithms to use
    hash_algorithms: Optional[List[str]] = None

    # Timeline analysis
    correlate_timestamps: bool = True
    detect_timestamp_anomalies: bool = True
    analyze_timezone_consistency: bool = True
    reconstruct_event_sequence: bool = True

    # Communication analysis depth
    analyze_email_headers: bool = True
    verify_sender_authentication: bool = True
    analyze_routing_information: bool = True
    extract_communication_metadata: bool = True

    # Pattern analysis options
    identify_device_patterns: bool = True
    detect_manipulation_patterns: bool = True
    analyze_temporal_patterns: bool = True
    identify_geographical_patterns: bool = True

    # Evidence quality requirements
    min_confidence_threshold: float = 0.7
    require_hash_verification: bool = True
    maintain_chain_of_custody: bool = True
    document_all_procedures: bool = True

    # Legal admissibility focus
    assess_legal_admissibility: bool = True
    identify_expert_testimony_needs: bool = True
    document_authentication_factors: bool = True
    anticipate_legal_challenges: bool = True

    # Technical analysis depth
    analyze_file_structure: bool = True
    detect_encryption: bool = True
    analyze_compression: bool = True
    verify_format_compliance: bool = True

    # Output options
    include_technical_details: bool = True
    provide_expert_conclusions: bool = True
    generate_recommendations: bool = True
    include_chain_of_custody: bool = True
    include_confidence_scores: bool = True

    def __post_init__(self):
        """Set defaults for mutable fields."""
        if self.hash_algorithms is None:
            self.hash_algorithms = [
                "md5",
                "sha1",
                "sha256",
                "sha512"
            ]


DEFAULT_CONFIG = DigitalForensicsAnalystConfig()

# High-precision forensics configuration for court proceedings
COURT_PROCEEDINGS_CONFIG = DigitalForensicsAnalystConfig(
    temperature=0.02,  # Maximum precision for court evidence
    max_tokens=20000,  # Extended for detailed analysis
    min_confidence_threshold=0.85,  # Very high threshold for court
    require_hash_verification=True,
    maintain_chain_of_custody=True,
    document_all_procedures=True,
    assess_legal_admissibility=True,
    identify_expert_testimony_needs=True,
    document_authentication_factors=True,
    anticipate_legal_challenges=True,
    include_technical_details=True,
    provide_expert_conclusions=True
)

# Authentication-focused configuration
AUTHENTICATION_ANALYSIS_CONFIG = DigitalForensicsAnalystConfig(
    verify_authenticity=True,
    detect_manipulation=True,
    calculate_file_hashes=True,
    verify_digital_signatures=True,
    detect_steganography=True,
    analyze_compression_artifacts=True,
    assess_provenance=True,
    analyze_file_structure=True,
    verify_format_compliance=True,
    min_confidence_threshold=0.8,
    temperature=0.03  # Very precise for authentication
)

# Timeline reconstruction configuration
TIMELINE_ANALYSIS_CONFIG = DigitalForensicsAnalystConfig(
    reconstruct_timeline=True,
    correlate_timestamps=True,
    detect_timestamp_anomalies=True,
    analyze_timezone_consistency=True,
    reconstruct_event_sequence=True,
    analyze_temporal_patterns=True,
    extract_file_system_metadata=True,
    extract_application_metadata=True,
    extract_device_metadata=True,
    max_tokens=18000  # Extended for timeline analysis
)

# Communication forensics configuration
COMMUNICATION_ANALYSIS_CONFIG = DigitalForensicsAnalystConfig(
    analyze_communications=True,
    analyze_email_headers=True,
    verify_sender_authentication=True,
    analyze_routing_information=True,
    extract_communication_metadata=True,
    correlate_timestamps=True,
    identify_temporal_patterns=True,
    extract_metadata=True,
    verify_authenticity=True
)

# Metadata extraction configuration
METADATA_EXTRACTION_CONFIG = DigitalForensicsAnalystConfig(
    extract_metadata=True,
    extract_exif_data=True,
    extract_file_system_metadata=True,
    extract_application_metadata=True,
    extract_device_metadata=True,
    extract_gps_data=True,
    extract_user_metadata=True,
    identify_device_patterns=True,
    analyze_temporal_patterns=True,
    identify_geographical_patterns=True,
    temperature=0.08  # Precise for metadata extraction
)

# Pattern analysis configuration
PATTERN_ANALYSIS_CONFIG = DigitalForensicsAnalystConfig(
    perform_comparison_analysis=True,
    identify_device_patterns=True,
    detect_manipulation_patterns=True,
    analyze_temporal_patterns=True,
    identify_geographical_patterns=True,
    correlate_timestamps=True,
    extract_metadata=True,
    verify_authenticity=True,
    max_tokens=18000  # Extended for pattern analysis
)

# Rapid forensics configuration
RAPID_ANALYSIS_CONFIG = DigitalForensicsAnalystConfig(
    max_tokens=10000,  # Reduced for speed
    extract_metadata=True,
    calculate_file_hashes=True,
    verify_authenticity=True,
    detect_manipulation=False,  # Skip detailed manipulation analysis
    reconstruct_timeline=False,  # Skip timeline reconstruction
    analyze_communications=False,  # Skip communication analysis
    perform_comparison_analysis=False,  # Skip comparison analysis
    min_confidence_threshold=0.6,  # Lower threshold for rapid analysis
    include_technical_details=False,  # Simplified output
    temperature=0.1
)

# Mobile device forensics configuration
MOBILE_FORENSICS_CONFIG = DigitalForensicsAnalystConfig(
    extract_exif_data=True,
    extract_gps_data=True,
    extract_device_metadata=True,
    identify_device_patterns=True,
    identify_geographical_patterns=True,
    analyze_temporal_patterns=True,
    correlate_timestamps=True,
    detect_timestamp_anomalies=True,
    analyze_timezone_consistency=True,
    verify_authenticity=True,
    detect_manipulation=True
)