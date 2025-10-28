"""
Configuration for Forensic Analysis Reviewer Agent
"""

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class ForensicAnalysisReviewerConfig:
    """Configuration for Forensic Analysis Reviewer agent."""

    # Model settings
    model: str = "claude-sonnet-4-5-20250929"
    max_tokens: int = 12000
    temperature: float = 0.1  # Low for technical precision

    # Analysis scope
    interpret_dna_reports: bool = True
    analyze_ballistics: bool = True
    review_autopsy_reports: bool = True
    assess_pathology: bool = True
    evaluate_trace_evidence: bool = True
    analyze_toxicology: bool = True

    # Legal integration
    map_to_legal_elements: bool = True
    assess_expert_credibility: bool = True
    identify_technical_issues: bool = True
    evaluate_methodology: bool = True

    # Output options
    generate_non_expert_summaries: bool = True
    include_technical_details: bool = True
    provide_follow_up_questions: bool = True
    assess_evidence_strength: bool = True
    include_confidence_scores: bool = True

    # Evidence thresholds
    min_evidence_confidence: float = 0.5
    require_methodology_assessment: bool = True


DEFAULT_CONFIG = ForensicAnalysisReviewerConfig()

# Legal proceedings configuration
LEGAL_PROCEEDINGS_CONFIG = ForensicAnalysisReviewerConfig(
    map_to_legal_elements=True,
    generate_non_expert_summaries=True,
    assess_expert_credibility=True,
    min_evidence_confidence=0.7,
    max_tokens=16000
)

# Technical review configuration
TECHNICAL_REVIEW_CONFIG = ForensicAnalysisReviewerConfig(
    include_technical_details=True,
    evaluate_methodology=True,
    identify_technical_issues=True,
    provide_follow_up_questions=True,
    temperature=0.05
)
