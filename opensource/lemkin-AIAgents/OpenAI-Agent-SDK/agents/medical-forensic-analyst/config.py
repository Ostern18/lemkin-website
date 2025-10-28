"""
Configuration for Medical & Forensic Record Analyst Agent
"""

from dataclasses import dataclass


@dataclass
class MedicalForensicConfig:
    """Configuration for medical/forensic analyst agent."""

    # Model settings
    model: str = "gpt-4o"
    max_tokens: int = 8192
    temperature: float = 0.1  # Low for medical accuracy

    # Analysis options
    apply_istanbul_protocol: bool = True
    identify_torture_indicators: bool = True
    check_temporal_consistency: bool = True
    generate_layperson_summary: bool = True

    # Thresholds
    min_confidence_for_torture_finding: float = 0.7
    high_severity_threshold: str = "high"

    # Privacy
    redact_patient_identifiers: bool = True

    # Human review
    require_review_for_torture_findings: bool = True
    require_review_for_death_cases: bool = True
    require_review_for_inconsistencies: bool = True


DEFAULT_CONFIG = MedicalForensicConfig()

TORTURE_ASSESSMENT_CONFIG = MedicalForensicConfig(
    apply_istanbul_protocol=True,
    identify_torture_indicators=True,
    min_confidence_for_torture_finding=0.85,
    require_review_for_torture_findings=True
)
