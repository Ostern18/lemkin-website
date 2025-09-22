"""
Lemkin Export: Data Export & Compliance Suite

A comprehensive toolkit for exporting data and ensuring compliance with 
international court submission requirements, including ICC, ICTY, ICTR formats,
and privacy regulations like GDPR, CCPA, and PIPEDA.

This module provides:
- International court format compliance (ICC, ICTY, ICTR)
- Privacy regulation compliance (GDPR, CCPA, PIPEDA)
- Evidence package creation with digital signatures
- Chain of custody preservation
- Data anonymization and redaction
- Audit trail generation
- Format validation and compliance checking
"""

from typing import List, Optional

from .core import (
    ExportManager,
    CaseData,
    Evidence,
    PersonalData,
    ICCSubmission,
    CourtPackage,
    ComplianceReport,
    ValidationResult,
    SubmissionMetadata,
    EvidencePackage,
    PrivacyAssessment,
    ExportConfig,
    CourtRequirements,
    DataProtectionSettings,
    AuditTrail,
    ChainOfCustody,
    ExportError,
    ComplianceError,
    ValidationError,
    FormatError,
)

from .icc_formatter import (
    ICCFormatter,
    format_for_icc,
    ICCSubmissionSchema,
    ICCMetadata,
)

from .court_packager import (
    CourtPackager,
    create_court_package,
    EvidenceValidator,
    DigitalSignature,
    PackageManifest,
)

from .privacy_compliance import (
    PrivacyCompliance,
    ensure_privacy_compliance,
    GDPRCompliance,
    CCPACompliance,
    PIPEDACompliance,
    DataAnonymizer,
    RedactionEngine,
)

from .format_validator import (
    FormatValidator,
    validate_submission_format,
    CourtSpecifications,
    ValidationRule,
    ComplianceChecker,
)

# Package metadata
__version__ = "0.1.0"
__author__ = "Lemkin AI Contributors"
__email__ = "contributors@lemkin.org"
__license__ = "Apache-2.0"

# Main exports for easy access
__all__ = [
    # Core classes and functions
    "ExportManager",
    "format_for_icc",
    "create_court_package", 
    "ensure_privacy_compliance",
    "validate_submission_format",
    
    # Data models
    "CaseData",
    "Evidence", 
    "PersonalData",
    "ICCSubmission",
    "CourtPackage",
    "ComplianceReport",
    "ValidationResult",
    "SubmissionMetadata",
    "EvidencePackage",
    "PrivacyAssessment",
    "ExportConfig",
    "CourtRequirements",
    "DataProtectionSettings",
    "AuditTrail",
    "ChainOfCustody",
    
    # Specialized classes
    "ICCFormatter",
    "CourtPackager",
    "PrivacyCompliance",
    "FormatValidator",
    "ICCSubmissionSchema",
    "ICCMetadata",
    "EvidenceValidator",
    "DigitalSignature",
    "PackageManifest",
    "GDPRCompliance",
    "CCPACompliance",
    "PIPEDACompliance",
    "DataAnonymizer",
    "RedactionEngine",
    "CourtSpecifications",
    "ValidationRule",
    "ComplianceChecker",
    
    # Exceptions
    "ExportError",
    "ComplianceError", 
    "ValidationError",
    "FormatError",
    
    # Package metadata
    "__version__",
    "__author__",
    "__email__",
    "__license__",
]

# Default configuration
DEFAULT_CONFIG = ExportConfig(
    output_format="zip",
    include_metadata=True,
    preserve_chain_of_custody=True,
    apply_digital_signatures=True,
    privacy_level="strict",
    audit_trail_enabled=True,
    compliance_checks=["gdpr", "ccpa", "pipeda"],
    court_formats=["icc", "icty", "ictr"],
)

def get_version() -> str:
    """Get the package version."""
    return __version__

def get_default_config() -> ExportConfig:
    """Get the default export configuration."""
    return DEFAULT_CONFIG.copy()

def create_export_manager(
    config: Optional[ExportConfig] = None,
    enable_audit: bool = True,
    strict_compliance: bool = True,
) -> ExportManager:
    """
    Create a configured ExportManager instance.
    
    Args:
        config: Custom export configuration
        enable_audit: Enable audit trail logging
        strict_compliance: Use strict compliance checking
        
    Returns:
        Configured ExportManager instance
    """
    if config is None:
        config = get_default_config()
        
    return ExportManager(
        config=config,
        enable_audit=enable_audit,
        strict_compliance=strict_compliance,
    )