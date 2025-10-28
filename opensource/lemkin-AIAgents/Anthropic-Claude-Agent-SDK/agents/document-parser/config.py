"""
Configuration for Multi-Format Document Parser Agent
"""

from typing import List, Dict, Any
from dataclasses import dataclass


@dataclass
class ParserConfig:
    """Configuration for document parser agent."""

    # Model settings
    model: str = "claude-sonnet-4-5-20250929"
    max_tokens: int = 8192  # Larger for detailed extraction
    temperature: float = 0.1  # Very low for accuracy

    # Quality thresholds
    min_confidence_threshold: float = 0.7
    min_ocr_quality_threshold: float = 0.6

    # Document type classification
    supported_document_types: List[str] = None

    # Processing options
    enable_table_extraction: bool = True
    enable_handwriting_detection: bool = True
    preserve_formatting: bool = True
    extract_metadata: bool = True

    # Language support
    supported_languages: List[str] = None

    # Output options
    output_format: str = "json"  # json, markdown, or both
    include_confidence_scores: bool = True
    include_quality_flags: bool = True

    # Human review triggers
    human_review_low_confidence: bool = True
    human_review_threshold: float = 0.5

    def __post_init__(self):
        """Set defaults for mutable fields."""
        if self.supported_document_types is None:
            self.supported_document_types = [
                "contract",
                "legal_brief",
                "court_filing",
                "witness_statement",
                "police_report",
                "medical_record",
                "forensic_report",
                "correspondence",
                "memorandum",
                "official_document",
                "identification_document",
                "financial_record",
                "military_order",
                "incident_report",
                "interview_transcript",
                "affidavit",
                "declaration",
                "expert_report",
                "technical_document",
                "other"
            ]

        if self.supported_languages is None:
            # Claude supports many languages; list major ones for investigations
            self.supported_languages = [
                "en",  # English
                "es",  # Spanish
                "fr",  # French
                "ar",  # Arabic
                "ru",  # Russian
                "zh",  # Chinese
                "pt",  # Portuguese
                "de",  # German
                "uk",  # Ukrainian
                "fa",  # Farsi/Persian
                "tr",  # Turkish
                "sw",  # Swahili
                "am",  # Amharic
                "my",  # Burmese
                "so",  # Somali
                "other"
            ]

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "min_confidence_threshold": self.min_confidence_threshold,
            "min_ocr_quality_threshold": self.min_ocr_quality_threshold,
            "supported_document_types": self.supported_document_types,
            "enable_table_extraction": self.enable_table_extraction,
            "enable_handwriting_detection": self.enable_handwriting_detection,
            "preserve_formatting": self.preserve_formatting,
            "extract_metadata": self.extract_metadata,
            "supported_languages": self.supported_languages,
            "output_format": self.output_format,
            "include_confidence_scores": self.include_confidence_scores,
            "include_quality_flags": self.include_quality_flags,
            "human_review_low_confidence": self.human_review_low_confidence,
            "human_review_threshold": self.human_review_threshold
        }


# Default configuration
DEFAULT_CONFIG = ParserConfig()


# High-accuracy configuration (for critical evidence)
HIGH_ACCURACY_CONFIG = ParserConfig(
    temperature=0.0,
    min_confidence_threshold=0.85,
    min_ocr_quality_threshold=0.75,
    human_review_threshold=0.7,
    human_review_low_confidence=True
)


# Fast processing configuration (for initial triage)
FAST_CONFIG = ParserConfig(
    temperature=0.2,
    max_tokens=4096,
    enable_table_extraction=False,
    min_confidence_threshold=0.6,
    human_review_low_confidence=False
)
