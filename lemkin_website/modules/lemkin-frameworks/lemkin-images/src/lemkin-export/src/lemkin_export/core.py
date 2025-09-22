"""
Core data models and main ExportManager class for Lemkin Export.

This module provides the foundational classes and data structures for
international court submission and compliance management.
"""

import json
import uuid
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Literal
from dataclasses import dataclass, field

from pydantic import BaseModel, Field, validator, root_validator
from loguru import logger


# Enums for standardized values
class CourtType(str, Enum):
    """International court types."""
    ICC = "icc"  # International Criminal Court
    ICTY = "icty"  # International Criminal Tribunal for the former Yugoslavia
    ICTR = "ictr"  # International Criminal Tribunal for Rwanda
    ICJ = "icj"  # International Court of Justice
    ECHR = "echr"  # European Court of Human Rights
    IACHR = "iachr"  # Inter-American Court of Human Rights


class PrivacyRegulation(str, Enum):
    """Privacy regulation standards."""
    GDPR = "gdpr"  # General Data Protection Regulation (EU)
    CCPA = "ccpa"  # California Consumer Privacy Act (US)
    PIPEDA = "pipeda"  # Personal Information Protection and Electronic Documents Act (Canada)
    LGPD = "lgpd"  # Lei Geral de Proteção de Dados (Brazil)
    PDPA = "pdpa"  # Personal Data Protection Act (Singapore)


class EvidenceType(str, Enum):
    """Types of evidence."""
    DOCUMENT = "document"
    IMAGE = "image"
    VIDEO = "video"
    AUDIO = "audio"
    TESTIMONY = "testimony"
    EXPERT_REPORT = "expert_report"
    PHYSICAL_EVIDENCE = "physical_evidence"
    DIGITAL_EVIDENCE = "digital_evidence"


class SubmissionStatus(str, Enum):
    """Submission processing status."""
    DRAFT = "draft"
    VALIDATING = "validating"
    READY = "ready"
    SUBMITTED = "submitted"
    ACCEPTED = "accepted"
    REJECTED = "rejected"


class ComplianceStatus(str, Enum):
    """Compliance check status."""
    COMPLIANT = "compliant"
    NON_COMPLIANT = "non_compliant"
    NEEDS_REVIEW = "needs_review"
    PARTIALLY_COMPLIANT = "partially_compliant"


# Custom exceptions
class ExportError(Exception):
    """Base exception for export operations."""
    pass


class ComplianceError(ExportError):
    """Exception for compliance violations."""
    pass


class ValidationError(ExportError):
    """Exception for validation failures."""
    pass


class FormatError(ExportError):
    """Exception for format-related errors."""
    pass


# Data models using Pydantic
class SubmissionMetadata(BaseModel):
    """Metadata for court submissions."""
    submission_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    title: str = Field(..., min_length=1, max_length=500)
    case_number: Optional[str] = None
    court: CourtType = Field(...)
    submitter_name: str = Field(..., min_length=1)
    submitter_organization: Optional[str] = None
    submission_date: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    language: str = Field(default="en", regex=r"^[a-z]{2}(-[A-Z]{2})?$")
    classification_level: str = Field(default="public", regex=r"^(public|confidential|restricted|secret)$")
    urgency: str = Field(default="normal", regex=r"^(low|normal|high|urgent)$")
    keywords: List[str] = Field(default_factory=list)
    description: Optional[str] = None


class ChainOfCustody(BaseModel):
    """Chain of custody tracking."""
    custody_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    evidence_id: str = Field(...)
    custodian_name: str = Field(...)
    custodian_role: str = Field(...)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    action: str = Field(...)  # collected, transferred, analyzed, etc.
    location: Optional[str] = None
    hash_value: Optional[str] = None  # Digital fingerprint
    signature: Optional[str] = None
    notes: Optional[str] = None


class Evidence(BaseModel):
    """Evidence item representation."""
    evidence_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    title: str = Field(..., min_length=1)
    evidence_type: EvidenceType = Field(...)
    file_path: Optional[str] = None
    file_hash: Optional[str] = None
    file_size: Optional[int] = None
    mime_type: Optional[str] = None
    collection_date: Optional[datetime] = None
    source: Optional[str] = None
    authenticity_verified: bool = Field(default=False)
    chain_of_custody: List[ChainOfCustody] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    redaction_applied: bool = Field(default=False)
    privacy_compliant: bool = Field(default=False)


class PersonalData(BaseModel):
    """Personal data subject to privacy regulations."""
    data_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    data_subject_id: Optional[str] = None
    data_type: str = Field(...)  # name, address, phone, etc.
    value: str = Field(...)
    source_evidence_id: Optional[str] = None
    sensitivity_level: str = Field(default="medium", regex=r"^(low|medium|high|critical)$")
    applicable_regulations: List[PrivacyRegulation] = Field(default_factory=list)
    anonymized: bool = Field(default=False)
    pseudonymized: bool = Field(default=False)
    redacted: bool = Field(default=False)
    consent_obtained: bool = Field(default=False)
    lawful_basis: Optional[str] = None
    retention_period: Optional[int] = None  # days
    processing_purpose: str = Field(...)


class CaseData(BaseModel):
    """Complete case data structure."""
    case_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    case_name: str = Field(..., min_length=1)
    case_number: Optional[str] = None
    court: CourtType = Field(...)
    legal_system: str = Field(default="international")
    parties: Dict[str, List[str]] = Field(default_factory=dict)  # plaintiff, defendant, etc.
    evidence: List[Evidence] = Field(default_factory=list)
    personal_data: List[PersonalData] = Field(default_factory=list)
    metadata: SubmissionMetadata = Field(...)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class DataProtectionSettings(BaseModel):
    """Privacy and data protection configuration."""
    enabled_regulations: List[PrivacyRegulation] = Field(default_factory=lambda: [PrivacyRegulation.GDPR])
    anonymization_threshold: float = Field(default=0.8, ge=0.0, le=1.0)
    pseudonymization_enabled: bool = Field(default=True)
    consent_required: bool = Field(default=True)
    retention_period_days: int = Field(default=365, ge=1)
    cross_border_transfer_allowed: bool = Field(default=False)
    encryption_required: bool = Field(default=True)
    audit_all_access: bool = Field(default=True)
    data_subject_rights: Dict[str, bool] = Field(default_factory=lambda: {
        "access": True,
        "rectification": True,
        "erasure": True,
        "portability": True,
        "restriction": True,
        "objection": True
    })


class CourtRequirements(BaseModel):
    """Court-specific submission requirements."""
    court: CourtType = Field(...)
    format_version: str = Field(default="1.0")
    required_sections: List[str] = Field(default_factory=list)
    max_file_size_mb: int = Field(default=100, ge=1)
    allowed_file_types: List[str] = Field(default_factory=lambda: [".pdf", ".xml", ".zip"])
    metadata_required: bool = Field(default=True)
    digital_signature_required: bool = Field(default=True)
    language_requirements: List[str] = Field(default_factory=lambda: ["en"])
    submission_deadline: Optional[datetime] = None
    classification_levels: List[str] = Field(default_factory=lambda: ["public", "confidential"])
    authentication_method: str = Field(default="certificate")


class ExportConfig(BaseModel):
    """Export configuration settings."""
    output_format: str = Field(default="zip", regex=r"^(zip|tar|tar\.gz|directory)$")
    include_metadata: bool = Field(default=True)
    preserve_chain_of_custody: bool = Field(default=True)
    apply_digital_signatures: bool = Field(default=True)
    privacy_level: str = Field(default="strict", regex=r"^(minimal|standard|strict|maximum)$")
    audit_trail_enabled: bool = Field(default=True)
    compliance_checks: List[str] = Field(default_factory=lambda: ["gdpr", "ccpa"])
    court_formats: List[str] = Field(default_factory=lambda: ["icc"])
    encryption_enabled: bool = Field(default=True)
    compression_level: int = Field(default=6, ge=0, le=9)
    output_directory: str = Field(default="./exports")
    temp_directory: str = Field(default="./temp")
    data_protection: DataProtectionSettings = Field(default_factory=DataProtectionSettings)
    court_requirements: Dict[CourtType, CourtRequirements] = Field(default_factory=dict)


class AuditTrail(BaseModel):
    """Audit trail entry."""
    audit_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    user_id: str = Field(...)
    action: str = Field(...)
    resource_type: str = Field(...)
    resource_id: str = Field(...)
    details: Dict[str, Any] = Field(default_factory=dict)
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    success: bool = Field(default=True)
    error_message: Optional[str] = None


class PrivacyAssessment(BaseModel):
    """Privacy impact assessment results."""
    assessment_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    data_subject_count: int = Field(default=0, ge=0)
    personal_data_types: List[str] = Field(default_factory=list)
    applicable_regulations: List[PrivacyRegulation] = Field(default_factory=list)
    compliance_status: ComplianceStatus = Field(...)
    risk_level: str = Field(default="medium", regex=r"^(low|medium|high|critical)$")
    mitigation_measures: List[str] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)
    assessment_date: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    assessor: str = Field(...)
    review_required: bool = Field(default=False)


class ValidationResult(BaseModel):
    """Format validation results."""
    validation_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    is_valid: bool = Field(...)
    court: CourtType = Field(...)
    format_version: str = Field(...)
    validation_timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    compliance_checks: Dict[str, bool] = Field(default_factory=dict)
    recommendations: List[str] = Field(default_factory=list)
    validator_version: str = Field(default="1.0")


class ComplianceReport(BaseModel):
    """Comprehensive compliance report."""
    report_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    case_id: str = Field(...)
    report_date: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    overall_status: ComplianceStatus = Field(...)
    privacy_assessment: PrivacyAssessment = Field(...)
    validation_results: List[ValidationResult] = Field(default_factory=list)
    audit_trail: List[AuditTrail] = Field(default_factory=list)
    compliance_scores: Dict[str, float] = Field(default_factory=dict)
    remediation_actions: List[str] = Field(default_factory=list)
    certification_status: Optional[str] = None
    next_review_date: Optional[datetime] = None
    report_generated_by: str = Field(...)


class DigitalSignature(BaseModel):
    """Digital signature information."""
    signature_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    algorithm: str = Field(default="RSA-SHA256")
    certificate_fingerprint: str = Field(...)
    signature_value: str = Field(...)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    signer_name: str = Field(...)
    signer_organization: Optional[str] = None
    is_valid: bool = Field(default=True)
    trust_chain_verified: bool = Field(default=False)


class PackageManifest(BaseModel):
    """Evidence package manifest."""
    manifest_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    package_name: str = Field(...)
    creation_date: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    creator: str = Field(...)
    evidence_count: int = Field(default=0, ge=0)
    total_size_bytes: int = Field(default=0, ge=0)
    files: List[Dict[str, Any]] = Field(default_factory=list)
    checksums: Dict[str, str] = Field(default_factory=dict)
    digital_signatures: List[DigitalSignature] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class EvidencePackage(BaseModel):
    """Court-ready evidence package."""
    package_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    case_id: str = Field(...)
    court: CourtType = Field(...)
    manifest: PackageManifest = Field(...)
    evidence: List[Evidence] = Field(default_factory=list)
    compliance_report: ComplianceReport = Field(...)
    package_path: Optional[str] = None
    encrypted: bool = Field(default=False)
    compression_used: bool = Field(default=True)
    integrity_verified: bool = Field(default=False)
    submission_ready: bool = Field(default=False)


class ICCSubmission(BaseModel):
    """ICC-specific submission format."""
    submission_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    icc_case_number: Optional[str] = None
    document_type: str = Field(...)
    classification: str = Field(default="public")
    filing_date: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    language: str = Field(default="en")
    pages: int = Field(default=0, ge=0)
    word_count: int = Field(default=0, ge=0)
    attachments: List[str] = Field(default_factory=list)
    confidentiality_level: str = Field(default="public")
    filing_party: str = Field(...)
    xml_content: str = Field(...)
    metadata: SubmissionMetadata = Field(...)
    validation_status: ValidationResult = Field(...)


class CourtPackage(BaseModel):
    """Generic court submission package."""
    package_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    court: CourtType = Field(...)
    case_data: CaseData = Field(...)
    evidence_package: EvidencePackage = Field(...)
    submission_format: Union[ICCSubmission, Dict[str, Any]] = Field(...)
    compliance_report: ComplianceReport = Field(...)
    package_path: Optional[str] = None
    ready_for_submission: bool = Field(default=False)
    submission_date: Optional[datetime] = None
    status: SubmissionStatus = Field(default=SubmissionStatus.DRAFT)


class ExportManager:
    """
    Main class for managing data exports and compliance operations.
    
    The ExportManager coordinates all aspects of preparing legal data for
    international court submissions while ensuring privacy compliance.
    """
    
    def __init__(
        self,
        config: Optional[ExportConfig] = None,
        enable_audit: bool = True,
        strict_compliance: bool = True
    ):
        """
        Initialize the ExportManager.
        
        Args:
            config: Export configuration settings
            enable_audit: Enable audit trail logging
            strict_compliance: Use strict compliance checking
        """
        self.config = config or ExportConfig()
        self.enable_audit = enable_audit
        self.strict_compliance = strict_compliance
        
        # Initialize components (will be set by importing modules)
        self._icc_formatter = None
        self._court_packager = None
        self._privacy_compliance = None
        self._format_validator = None
        
        # Audit trail storage
        self.audit_trail: List[AuditTrail] = []
        
        # Setup logging
        logger.info(f"ExportManager initialized with config: {config}")
        
    def set_icc_formatter(self, formatter):
        """Set the ICC formatter component."""
        self._icc_formatter = formatter
        
    def set_court_packager(self, packager):
        """Set the court packager component."""
        self._court_packager = packager
        
    def set_privacy_compliance(self, compliance):
        """Set the privacy compliance component."""
        self._privacy_compliance = compliance
        
    def set_format_validator(self, validator):
        """Set the format validator component."""
        self._format_validator = validator
    
    def export_case(
        self,
        case_data: CaseData,
        court: CourtType,
        output_path: Optional[Path] = None,
    ) -> CourtPackage:
        """
        Export a complete case for court submission.
        
        Args:
            case_data: The case data to export
            court: Target court type
            output_path: Optional output directory path
            
        Returns:
            Court package ready for submission
            
        Raises:
            ExportError: If export fails
            ComplianceError: If compliance checks fail
        """
        try:
            self._log_audit_event(
                "export_case_started",
                "case",
                case_data.case_id,
                {"court": court.value}
            )
            
            # Step 1: Privacy compliance check
            if self._privacy_compliance:
                compliance_report = self._privacy_compliance.assess_case_data(case_data)
                if compliance_report.overall_status == ComplianceStatus.NON_COMPLIANT and self.strict_compliance:
                    raise ComplianceError(f"Case data is not privacy compliant: {compliance_report}")
            else:
                # Create basic compliance report
                privacy_assessment = PrivacyAssessment(
                    compliance_status=ComplianceStatus.NEEDS_REVIEW,
                    assessor="system",
                    data_subject_count=len(case_data.personal_data)
                )
                compliance_report = ComplianceReport(
                    case_id=case_data.case_id,
                    overall_status=ComplianceStatus.NEEDS_REVIEW,
                    privacy_assessment=privacy_assessment,
                    report_generated_by="export_manager"
                )
            
            # Step 2: Create evidence package
            if self._court_packager:
                evidence_package = self._court_packager.create_package(case_data.evidence, case_data)
            else:
                # Create basic evidence package
                manifest = PackageManifest(
                    package_name=f"{case_data.case_name}_evidence",
                    creator="export_manager",
                    evidence_count=len(case_data.evidence)
                )
                evidence_package = EvidencePackage(
                    case_id=case_data.case_id,
                    court=court,
                    manifest=manifest,
                    evidence=case_data.evidence,
                    compliance_report=compliance_report
                )
            
            # Step 3: Format for specific court
            if court == CourtType.ICC and self._icc_formatter:
                submission_format = self._icc_formatter.format_case(case_data)
            else:
                # Create generic submission format
                submission_format = {
                    "court": court.value,
                    "case_data": case_data.dict(),
                    "formatted_by": "export_manager"
                }
            
            # Step 4: Validate format
            validation_results = []
            if self._format_validator:
                validation_result = self._format_validator.validate_for_court(submission_format, court)
                validation_results.append(validation_result)
                
                if not validation_result.is_valid and self.strict_compliance:
                    raise ValidationError(f"Submission format validation failed: {validation_result.errors}")
            
            # Step 5: Create final court package
            court_package = CourtPackage(
                court=court,
                case_data=case_data,
                evidence_package=evidence_package,
                submission_format=submission_format,
                compliance_report=compliance_report,
                status=SubmissionStatus.READY if not validation_results or all(v.is_valid for v in validation_results) else SubmissionStatus.DRAFT
            )
            
            # Step 6: Save package if output path provided
            if output_path:
                self._save_court_package(court_package, output_path)
            
            self._log_audit_event(
                "export_case_completed",
                "case",
                case_data.case_id,
                {"court": court.value, "package_id": court_package.package_id}
            )
            
            return court_package
            
        except Exception as e:
            self._log_audit_event(
                "export_case_failed",
                "case", 
                case_data.case_id,
                {"error": str(e)},
                success=False,
                error_message=str(e)
            )
            raise ExportError(f"Failed to export case: {e}") from e
    
    def _save_court_package(self, package: CourtPackage, output_path: Path) -> None:
        """Save court package to disk."""
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        package_file = output_path / f"{package.package_id}.json"
        with open(package_file, 'w', encoding='utf-8') as f:
            json.dump(package.dict(), f, indent=2, default=str)
        
        package.package_path = str(package_file)
    
    def _log_audit_event(
        self,
        action: str,
        resource_type: str,
        resource_id: str,
        details: Optional[Dict[str, Any]] = None,
        success: bool = True,
        error_message: Optional[str] = None
    ) -> None:
        """Log an audit trail event."""
        if not self.enable_audit:
            return
            
        audit_entry = AuditTrail(
            user_id="system",  # In a real system, this would be the authenticated user
            action=action,
            resource_type=resource_type,
            resource_id=resource_id,
            details=details or {},
            success=success,
            error_message=error_message
        )
        
        self.audit_trail.append(audit_entry)
        logger.info(f"Audit: {action} on {resource_type}:{resource_id} - Success: {success}")
    
    def get_audit_trail(self, resource_id: Optional[str] = None) -> List[AuditTrail]:
        """
        Retrieve audit trail entries.
        
        Args:
            resource_id: Filter by specific resource ID
            
        Returns:
            List of audit trail entries
        """
        if resource_id:
            return [entry for entry in self.audit_trail if entry.resource_id == resource_id]
        return self.audit_trail.copy()
    
    def generate_compliance_report(self, case_id: str) -> ComplianceReport:
        """
        Generate a comprehensive compliance report for a case.
        
        Args:
            case_id: The case ID to generate report for
            
        Returns:
            Comprehensive compliance report
        """
        audit_entries = self.get_audit_trail(case_id)
        
        # Create basic privacy assessment
        privacy_assessment = PrivacyAssessment(
            compliance_status=ComplianceStatus.NEEDS_REVIEW,
            assessor="export_manager"
        )
        
        # Create compliance report
        report = ComplianceReport(
            case_id=case_id,
            overall_status=ComplianceStatus.NEEDS_REVIEW,
            privacy_assessment=privacy_assessment,
            audit_trail=audit_entries,
            report_generated_by="export_manager"
        )
        
        return report