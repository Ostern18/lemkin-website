"""
Configuration for NGO & UN Reporting Specialist Agent
"""

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class NGOUNReporterConfig:
    """Configuration for NGO & UN Reporting Specialist agent."""

    # Model settings
    model: str = "gpt-4o"
    max_tokens: int = 16384  # Large for comprehensive reports
    temperature: float = 0.1  # Low for professional precision

    # Report generation scope
    generate_executive_summary: bool = True
    create_detailed_analysis: bool = True
    provide_recommendations: bool = True
    include_legal_citations: bool = True
    format_for_submission: bool = True

    # UN mechanism specialization
    upr_submissions: bool = True
    treaty_body_reports: bool = True
    special_procedures_communications: bool = True
    human_rights_council_submissions: bool = True
    security_council_briefings: bool = True
    icj_memorial_support: bool = True

    # NGO report types
    shadow_reports: bool = True
    advocacy_materials: bool = True
    press_releases: bool = True
    policy_briefs: bool = True
    fact_finding_reports: bool = True
    campaign_materials: bool = True

    # Content analysis depth
    analyze_legal_frameworks: bool = True
    assess_state_obligations: bool = True
    identify_violations: bool = True
    evaluate_remedial_measures: bool = True
    review_implementation_gaps: bool = True

    # Documentation standards
    verify_source_reliability: bool = True
    maintain_chain_of_custody: bool = True
    protect_witness_identity: bool = True
    ensure_factual_accuracy: bool = True
    follow_diplomatic_protocols: bool = True

    # Report formatting options
    comply_with_word_limits: bool = True
    use_official_citation_format: bool = True
    include_executive_summary: bool = True
    create_annexes: bool = True
    generate_visual_aids: bool = False

    # Target audiences
    un_mechanisms: bool = True
    ngo_networks: bool = True
    media_outlets: bool = True
    academic_institutions: bool = True
    government_officials: bool = True
    civil_society: bool = True

    # Quality assurance
    fact_check_requirements: bool = True
    legal_review_needed: bool = True
    sensitivity_review: bool = True
    translation_requirements: bool = False
    accessibility_compliance: bool = True

    # Languages supported
    supported_languages: Optional[List[str]] = None

    # Confidentiality and security
    protect_sensitive_information: bool = True
    anonymize_sources: bool = True
    redact_identifying_details: bool = True
    implement_security_protocols: bool = True

    # Advocacy strategy
    strategic_messaging: bool = True
    timing_considerations: bool = True
    coalition_building: bool = True
    media_engagement: bool = True
    follow_up_planning: bool = True

    def __post_init__(self):
        """Set defaults for mutable fields."""
        if self.supported_languages is None:
            self.supported_languages = [
                "English",
                "French",
                "Spanish",
                "Arabic",
                "Russian",
                "Chinese"
            ]


DEFAULT_CONFIG = NGOUNReporterConfig()

# UPR submission configuration
UPR_SUBMISSION_CONFIG = NGOUNReporterConfig(
    temperature=0.05,  # Maximum precision for UPR
    max_tokens=12000,  # Optimized for 2,815 word limit
    comply_with_word_limits=True,
    upr_submissions=True,
    treaty_body_reports=False,  # Focus on UPR only
    special_procedures_communications=False,
    use_official_citation_format=True,
    follow_diplomatic_protocols=True,
    fact_check_requirements=True,
    legal_review_needed=True
)

# Treaty body shadow report configuration
TREATY_BODY_CONFIG = NGOUNReporterConfig(
    treaty_body_reports=True,
    shadow_reports=True,
    analyze_legal_frameworks=True,
    assess_state_obligations=True,
    identify_violations=True,
    evaluate_remedial_measures=True,
    review_implementation_gaps=True,
    comply_with_word_limits=True,
    use_official_citation_format=True,
    max_tokens=18000  # Extended for detailed treaty analysis
)

# Special procedures communication configuration
SPECIAL_PROCEDURES_CONFIG = NGOUNReporterConfig(
    special_procedures_communications=True,
    urgent_action_format=True,
    individual_complaint_format=True,
    protect_witness_identity=True,
    anonymize_sources=True,
    implement_security_protocols=True,
    fact_check_requirements=True,
    temperature=0.07,  # Precise for legal communications
    max_tokens=14000
)

# Advocacy campaign configuration
ADVOCACY_CAMPAIGN_CONFIG = NGOUNReporterConfig(
    advocacy_materials=True,
    press_releases=True,
    policy_briefs=True,
    campaign_materials=True,
    strategic_messaging=True,
    timing_considerations=True,
    coalition_building=True,
    media_engagement=True,
    follow_up_planning=True,
    generate_visual_aids=True,
    temperature=0.15,  # Slightly higher for creative messaging
    accessibility_compliance=True
)

# Fact-finding mission configuration
FACT_FINDING_CONFIG = NGOUNReporterConfig(
    fact_finding_reports=True,
    verify_source_reliability=True,
    maintain_chain_of_custody=True,
    protect_witness_identity=True,
    ensure_factual_accuracy=True,
    create_detailed_analysis=True,
    include_legal_citations=True,
    create_annexes=True,
    fact_check_requirements=True,
    legal_review_needed=True,
    sensitivity_review=True,
    max_tokens=20000  # Extended for comprehensive reporting
)

# Emergency/urgent response configuration
URGENT_RESPONSE_CONFIG = NGOUNReporterConfig(
    max_tokens=8000,  # Reduced for speed
    temperature=0.08,
    special_procedures_communications=True,
    press_releases=True,
    urgent_action_format=True,
    protect_witness_identity=True,
    implement_security_protocols=True,
    strategic_messaging=True,
    timing_considerations=True,
    comply_with_word_limits=False,  # Speed over strict limits
    generate_visual_aids=False,  # Skip for speed
    fact_check_requirements=True  # Still maintain accuracy
)

# High-security configuration
HIGH_SECURITY_CONFIG = NGOUNReporterConfig(
    temperature=0.05,  # Maximum precision
    protect_sensitive_information=True,
    anonymize_sources=True,
    redact_identifying_details=True,
    implement_security_protocols=True,
    protect_witness_identity=True,
    maintain_chain_of_custody=True,
    sensitivity_review=True,
    fact_check_requirements=True,
    legal_review_needed=True,
    follow_diplomatic_protocols=True
)

# Media engagement configuration
MEDIA_ENGAGEMENT_CONFIG = NGOUNReporterConfig(
    press_releases=True,
    media_outlets=True,
    strategic_messaging=True,
    timing_considerations=True,
    accessibility_compliance=True,
    generate_visual_aids=True,
    temperature=0.12,  # Balanced for engaging content
    max_tokens=10000,  # Optimized for media formats
    protect_sensitive_information=True,
    anonymize_sources=True
)

# Academic/research configuration
ACADEMIC_RESEARCH_CONFIG = NGOUNReporterConfig(
    academic_institutions=True,
    create_detailed_analysis=True,
    include_legal_citations=True,
    analyze_legal_frameworks=True,
    assess_state_obligations=True,
    use_official_citation_format=True,
    create_annexes=True,
    fact_check_requirements=True,
    legal_review_needed=True,
    max_tokens=20000,  # Extended for academic depth
    temperature=0.08  # Precise for academic standards
)

# Multilingual reporting configuration
MULTILINGUAL_CONFIG = NGOUNReporterConfig(
    translation_requirements=True,
    supported_languages=[
        "English", "French", "Spanish", "Arabic",
        "Russian", "Chinese", "Portuguese", "German"
    ],
    accessibility_compliance=True,
    follow_diplomatic_protocols=True,
    use_official_citation_format=True,
    sensitivity_review=True
)