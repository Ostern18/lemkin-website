"""
Configuration for Comparative Document Analyzer Agent
"""

from dataclasses import dataclass
from typing import List


@dataclass
class ComparativeAnalyzerConfig:
    """Configuration for comparative analyzer agent."""

    # Model settings
    model: str = "claude-sonnet-4-5-20250929"
    max_tokens: int = 16384  # Large for multi-document analysis
    temperature: float = 0.1

    # Analysis options
    detect_patterns: bool = True
    analyze_metadata: bool = True
    generate_similarity_matrix: bool = True
    check_timeline_consistency: bool = True

    # Thresholds
    min_similarity_threshold: float = 0.3  # Below this, documents considered dissimilar
    high_similarity_threshold: float = 0.85  # Above this, flag for copy-paste
    red_flag_confidence_threshold: float = 0.7

    # Comparison types
    supported_comparison_types: List[str] = None

    # Output options
    include_full_diff: bool = True
    max_diff_entries: int = 100

    # Human review
    human_review_high_severity_flags: bool = True

    def __post_init__(self):
        """Set defaults for mutable fields."""
        if self.supported_comparison_types is None:
            self.supported_comparison_types = [
                "version_comparison",
                "multi_document_similarity",
                "pattern_analysis",
                "forgery_detection",
                "redaction_analysis"
            ]


DEFAULT_CONFIG = ComparativeAnalyzerConfig()
