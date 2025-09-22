"""
Core classes and data models for the Lemkin Report Generator Suite.

This module provides the main ReportGenerator class and comprehensive
data models for legal report generation, evidence cataloging, and 
professional document formatting.
"""

from datetime import datetime, date
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Set, Tuple
from uuid import UUID, uuid4
from decimal import Decimal

from pydantic import BaseModel, Field, validator, ConfigDict
from loguru import logger


# Enums for type safety and standardization

class ReportType(str, Enum):
    """Types of legal reports that can be generated"""
    FACT_SHEET = "fact_sheet"
    EVIDENCE_CATALOG = "evidence_catalog"
    LEGAL_BRIEF = "legal_brief"
    CASE_SUMMARY = "case_summary"
    WITNESS_REPORT = "witness_report"
    EXPERT_REPORT = "expert_report"
    DISCOVERY_RESPONSE = "discovery_response"
    MOTION_BRIEF = "motion_brief"
    APPELLATE_BRIEF = "appellate_brief"
    SETTLEMENT_MEMO = "settlement_memo"


class ExportFormat(str, Enum):
    """Supported export formats for reports"""
    PDF = "pdf"
    DOCX = "docx"
    HTML = "html"
    LATEX = "latex"
    MARKDOWN = "markdown"
    RTF = "rtf"
    ODT = "odt"
    PLAIN_TEXT = "txt"


class EvidenceType(str, Enum):
    """Types of evidence that can be cataloged"""
    DOCUMENT = "document"
    PHOTOGRAPH = "photograph"
    VIDEO = "video"
    AUDIO = "audio"
    PHYSICAL = "physical"
    DIGITAL = "digital"
    TESTIMONY = "testimony"
    EXPERT_OPINION = "expert_opinion"
    FORENSIC = "forensic"
    FINANCIAL = "financial"
    MEDICAL = "medical"
    SCIENTIFIC = "scientific"


class TemplateType(str, Enum):
    """Types of legal document templates"""
    STANDARD = "standard"
    COURT_SPECIFIC = "court_specific"
    JURISDICTION_SPECIFIC = "jurisdiction_specific"
    PRACTICE_AREA_SPECIFIC = "practice_area_specific"
    CUSTOM = "custom"
    FIRM_BRANDED = "firm_branded"


class ReportStatus(str, Enum):
    """Status of report generation process"""
    DRAFT = "draft"
    IN_REVIEW = "in_review"
    APPROVED = "approved"
    FINALIZED = "finalized"
    FILED = "filed"
    ARCHIVED = "archived"
    REJECTED = "rejected"


class CitationStyle(str, Enum):
    """Legal citation formatting styles"""
    BLUEBOOK = "bluebook"
    ALWD = "alwd"
    APA = "apa"
    MLA = "mla"
    CHICAGO = "chicago"
    LOCAL_RULES = "local_rules"
    CUSTOM = "custom"


class DocumentStandard(str, Enum):
    """Document formatting standards"""
    FEDERAL_COURT = "federal_court"
    STATE_COURT = "state_court"
    APPELLATE_COURT = "appellate_court"
    TRIAL_COURT = "trial_court"
    ADMINISTRATIVE = "administrative"
    ARBITRATION = "arbitration"
    MEDIATION = "mediation"


class EvidenceAuthenticity(str, Enum):
    """Evidence authenticity status"""
    AUTHENTIC = "authentic"
    SUSPICIOUS = "suspicious"
    MANIPULATED = "manipulated"
    INCONCLUSIVE = "inconclusive"
    PENDING_VERIFICATION = "pending_verification"


class ConfidentialityLevel(str, Enum):
    """Document confidentiality levels"""
    PUBLIC = "public"
    CONFIDENTIAL = "confidential"
    HIGHLY_CONFIDENTIAL = "highly_confidential"
    ATTORNEYS_EYES_ONLY = "attorneys_eyes_only"
    RESTRICTED = "restricted"


# Base data models

class BaseReportModel(BaseModel):
    """Base model for all report entities"""
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        use_enum_values=True,
        arbitrary_types_allowed=True
    )
    
    id: str = Field(default_factory=lambda: str(uuid4()))
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    version: int = Field(default=1, ge=1)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class PersonInfo(BaseReportModel):
    """Information about a person involved in legal proceedings"""
    full_name: str = Field(..., min_length=1)
    title: Optional[str] = None
    role: str = Field(..., description="Role in case (attorney, client, witness, etc.)")
    organization: Optional[str] = None
    contact_email: Optional[str] = None
    contact_phone: Optional[str] = None
    address: Optional[str] = None
    bar_number: Optional[str] = None
    qualifications: List[str] = Field(default_factory=list)


class CaseInfo(BaseReportModel):
    """Basic case information"""
    case_number: str = Field(..., min_length=1)
    case_name: str = Field(..., min_length=1)
    court: str = Field(..., min_length=1)
    judge: Optional[str] = None
    jurisdiction: str = Field(..., min_length=1)
    practice_area: str = Field(..., min_length=1)
    case_type: str = Field(..., description="civil, criminal, administrative, etc.")
    filing_date: Optional[date] = None
    trial_date: Optional[date] = None
    statute_of_limitations: Optional[date] = None
    key_parties: List[PersonInfo] = Field(default_factory=list)


class LegalCitation(BaseReportModel):
    """Standardized legal citation"""
    full_citation: str = Field(..., min_length=1)
    short_citation: str = Field(..., min_length=1)
    citation_type: str = Field(..., description="case, statute, regulation, etc.")
    pin_cite: Optional[str] = None
    parenthetical: Optional[str] = None
    relevance_score: float = Field(default=0.0, ge=0.0, le=1.0)
    notes: Optional[str] = None


class Evidence(BaseReportModel):
    """Individual piece of evidence"""
    evidence_id: str = Field(..., min_length=1)
    title: str = Field(..., min_length=1)
    description: str = Field(..., min_length=1)
    evidence_type: EvidenceType
    source: str = Field(..., min_length=1)
    
    # Chain of custody
    custodian: str = Field(..., min_length=1)
    date_collected: Optional[date] = None
    collection_method: Optional[str] = None
    chain_of_custody: List[Dict[str, Any]] = Field(default_factory=list)
    
    # File information
    file_path: Optional[str] = None
    file_size: Optional[int] = None
    file_hash: Optional[str] = None
    file_format: Optional[str] = None
    
    # Authenticity
    authenticity_status: EvidenceAuthenticity = Field(EvidenceAuthenticity.PENDING_VERIFICATION)
    authentication_method: Optional[str] = None
    authentication_date: Optional[date] = None
    authenticating_party: Optional[str] = None
    
    # Legal considerations
    admissibility_status: str = Field(default="pending")
    privilege_claims: List[str] = Field(default_factory=list)
    confidentiality_level: ConfidentialityLevel = Field(ConfidentialityLevel.CONFIDENTIAL)
    
    # Analysis results
    forensic_analysis: Optional[Dict[str, Any]] = None
    expert_opinions: List[Dict[str, Any]] = Field(default_factory=list)
    
    # Relevance and weight
    relevance_to_case: str = Field(..., min_length=1)
    evidential_weight: float = Field(default=0.0, ge=0.0, le=1.0)
    supporting_evidence_ids: List[str] = Field(default_factory=list)
    contradictory_evidence_ids: List[str] = Field(default_factory=list)
    
    # Additional metadata
    tags: List[str] = Field(default_factory=list)
    categories: List[str] = Field(default_factory=list)
    exhibit_number: Optional[str] = None
    deposition_references: List[str] = Field(default_factory=list)


class ReportSection(BaseReportModel):
    """Individual section of a legal report"""
    title: str = Field(..., min_length=1)
    content: str = Field(..., min_length=1)
    section_type: str = Field(..., description="header, body, conclusion, etc.")
    order_index: int = Field(..., ge=0)
    subsections: List['ReportSection'] = Field(default_factory=list)
    citations: List[LegalCitation] = Field(default_factory=list)
    evidence_references: List[str] = Field(default_factory=list)
    formatting_notes: Optional[str] = None
    review_required: bool = Field(default=False)


class LegalTemplate(BaseReportModel):
    """Template for legal document generation"""
    name: str = Field(..., min_length=1)
    template_type: TemplateType
    jurisdiction: Optional[str] = None
    court: Optional[str] = None
    practice_area: Optional[str] = None
    document_standard: DocumentStandard
    
    # Template structure
    sections: List[ReportSection] = Field(default_factory=list)
    required_fields: List[str] = Field(default_factory=list)
    optional_fields: List[str] = Field(default_factory=list)
    
    # Formatting specifications
    page_layout: Dict[str, Any] = Field(default_factory=dict)
    font_specifications: Dict[str, Any] = Field(default_factory=dict)
    citation_style: CitationStyle = Field(CitationStyle.BLUEBOOK)
    
    # Template metadata
    version: str = Field(default="1.0")
    author: Optional[str] = None
    last_updated: datetime = Field(default_factory=datetime.utcnow)
    usage_count: int = Field(default=0, ge=0)
    
    # Validation rules
    validation_rules: Dict[str, Any] = Field(default_factory=dict)
    review_checklist: List[str] = Field(default_factory=list)


class TemplateMetadata(BaseReportModel):
    """Metadata for template management"""
    template_id: str = Field(..., min_length=1)
    name: str = Field(..., min_length=1)
    description: str = Field(..., min_length=1)
    category: str = Field(..., min_length=1)
    tags: List[str] = Field(default_factory=list)
    
    # Usage statistics
    times_used: int = Field(default=0, ge=0)
    last_used: Optional[datetime] = None
    average_generation_time: Optional[float] = None
    
    # Quality metrics
    user_rating: Optional[float] = Field(None, ge=0.0, le=5.0)
    review_scores: List[float] = Field(default_factory=list)
    error_rate: Optional[float] = Field(None, ge=0.0, le=1.0)
    
    # Maintenance info
    requires_update: bool = Field(default=False)
    deprecation_date: Optional[date] = None
    replacement_template_id: Optional[str] = None


class CaseData(BaseReportModel):
    """Comprehensive case data for report generation"""
    case_info: CaseInfo
    evidence_list: List[Evidence] = Field(default_factory=list)
    witnesses: List[PersonInfo] = Field(default_factory=list)
    attorneys: List[PersonInfo] = Field(default_factory=list)
    
    # Case timeline
    key_dates: Dict[str, date] = Field(default_factory=dict)
    chronology: List[Dict[str, Any]] = Field(default_factory=list)
    
    # Legal analysis
    legal_theories: List[str] = Field(default_factory=list)
    causes_of_action: List[str] = Field(default_factory=list)
    defenses: List[str] = Field(default_factory=list)
    precedent_cases: List[LegalCitation] = Field(default_factory=list)
    
    # Factual summary
    statement_of_facts: str = Field(default="")
    disputed_facts: List[str] = Field(default_factory=list)
    undisputed_facts: List[str] = Field(default_factory=list)
    
    # Discovery information
    document_requests: List[Dict[str, Any]] = Field(default_factory=list)
    interrogatories: List[Dict[str, Any]] = Field(default_factory=list)
    depositions: List[Dict[str, Any]] = Field(default_factory=list)
    
    # Financial information
    damages_claimed: Optional[Decimal] = None
    settlement_offers: List[Dict[str, Any]] = Field(default_factory=list)
    costs_incurred: Optional[Decimal] = None
    
    # Strategic considerations
    strengths: List[str] = Field(default_factory=list)
    weaknesses: List[str] = Field(default_factory=list)
    risks: List[str] = Field(default_factory=list)
    opportunities: List[str] = Field(default_factory=list)


class FactSheet(BaseReportModel):
    """Standardized fact sheet document"""
    case_data: CaseData
    report_type: ReportType = Field(ReportType.FACT_SHEET)
    
    # Document metadata
    title: str = Field(..., min_length=1)
    author: PersonInfo
    prepared_for: Optional[PersonInfo] = None
    preparation_date: date = Field(default_factory=date.today)
    
    # Content sections
    executive_summary: str = Field(..., min_length=1)
    factual_background: str = Field(..., min_length=1)
    legal_issues: List[str] = Field(default_factory=list)
    key_evidence: List[str] = Field(default_factory=list)  # Evidence IDs
    witness_summary: str = Field(default="")
    preliminary_analysis: str = Field(default="")
    recommendations: List[str] = Field(default_factory=list)
    
    # Formatting and status
    formatting_applied: bool = Field(default=False)
    review_status: ReportStatus = Field(ReportStatus.DRAFT)
    confidentiality_level: ConfidentialityLevel = Field(ConfidentialityLevel.CONFIDENTIAL)
    
    # Quality assurance
    review_checklist_completed: bool = Field(default=False)
    reviewer: Optional[PersonInfo] = None
    review_date: Optional[date] = None
    approval_required: bool = Field(default=True)


class EvidenceCatalog(BaseReportModel):
    """Comprehensive evidence inventory and catalog"""
    case_data: CaseData
    report_type: ReportType = Field(ReportType.EVIDENCE_CATALOG)
    
    # Catalog metadata
    title: str = Field(..., min_length=1)
    custodian: PersonInfo
    catalog_date: date = Field(default_factory=date.today)
    
    # Evidence organization
    evidence_by_category: Dict[str, List[str]] = Field(default_factory=dict)  # Evidence IDs
    evidence_by_relevance: Dict[str, List[str]] = Field(default_factory=dict)
    evidence_timeline: List[Dict[str, Any]] = Field(default_factory=list)
    
    # Chain of custody tracking
    custody_log: List[Dict[str, Any]] = Field(default_factory=list)
    authentication_summary: Dict[str, int] = Field(default_factory=dict)
    pending_authentications: List[str] = Field(default_factory=list)
    
    # Analysis summaries
    forensic_analysis_summary: str = Field(default="")
    expert_opinions_summary: str = Field(default="")
    admissibility_analysis: Dict[str, str] = Field(default_factory=dict)
    
    # Quality control
    inventory_complete: bool = Field(default=False)
    verification_date: Optional[date] = None
    discrepancies_noted: List[str] = Field(default_factory=list)


class LegalBrief(BaseReportModel):
    """Auto-populated legal brief document"""
    case_data: CaseData
    template: LegalTemplate
    report_type: ReportType = Field(ReportType.LEGAL_BRIEF)
    
    # Brief metadata
    brief_type: str = Field(..., min_length=1)  # motion, response, reply, etc.
    title: str = Field(..., min_length=1)
    author: PersonInfo
    opposing_counsel: Optional[PersonInfo] = None
    filing_deadline: Optional[date] = None
    
    # Legal arguments
    statement_of_issues: List[str] = Field(default_factory=list)
    argument_sections: List[ReportSection] = Field(default_factory=list)
    conclusion: str = Field(default="")
    prayer_for_relief: str = Field(default="")
    
    # Supporting materials
    authorities_cited: List[LegalCitation] = Field(default_factory=list)
    evidence_cited: List[str] = Field(default_factory=list)  # Evidence IDs
    exhibits: List[Dict[str, Any]] = Field(default_factory=list)
    
    # Document compliance
    word_count: int = Field(default=0, ge=0)
    page_limit_compliance: bool = Field(default=True)
    citation_format_verified: bool = Field(default=False)
    court_rules_compliance: Dict[str, bool] = Field(default_factory=dict)
    
    # Review and approval
    internal_review_complete: bool = Field(default=False)
    client_approval: bool = Field(default=False)
    ready_for_filing: bool = Field(default=False)


class ExportSettings(BaseReportModel):
    """Settings for document export and formatting"""
    export_format: ExportFormat
    output_directory: Optional[Path] = None
    filename_template: str = Field(default="{report_type}_{case_number}_{date}")
    
    # PDF settings
    pdf_template: Optional[str] = None
    pdf_metadata: Dict[str, str] = Field(default_factory=dict)
    pdf_security: Dict[str, Any] = Field(default_factory=dict)
    
    # Word document settings
    docx_template: Optional[str] = None
    track_changes: bool = Field(default=False)
    document_protection: Optional[str] = None
    
    # Formatting options
    page_margins: Tuple[float, float, float, float] = Field(default=(1.0, 1.0, 1.0, 1.0))
    font_family: str = Field(default="Times New Roman")
    font_size: int = Field(default=12, ge=8, le=72)
    line_spacing: float = Field(default=2.0, ge=1.0, le=3.0)
    
    # Header and footer
    include_header: bool = Field(default=True)
    include_footer: bool = Field(default=True)
    page_numbers: bool = Field(default=True)
    watermark: Optional[str] = None
    
    # Quality settings
    image_quality: str = Field(default="high")
    compression_level: str = Field(default="balanced")
    embed_fonts: bool = Field(default=True)
    
    # Export options
    separate_attachments: bool = Field(default=True)
    include_metadata: bool = Field(default=True)
    digital_signature: bool = Field(default=False)
    archive_copy: bool = Field(default=True)


class ExportedReport(BaseReportModel):
    """Exported report with metadata and file information"""
    source_report: Union[FactSheet, EvidenceCatalog, LegalBrief]
    export_settings: ExportSettings
    
    # Export metadata
    export_timestamp: datetime = Field(default_factory=datetime.utcnow)
    exported_by: PersonInfo
    export_format: ExportFormat
    
    # File information
    output_path: Path
    file_size_bytes: int = Field(..., ge=0)
    file_hash: str = Field(..., min_length=1)
    
    # Quality metrics
    export_successful: bool = Field(default=False)
    validation_passed: bool = Field(default=False)
    compliance_check_passed: bool = Field(default=False)
    
    # Error handling
    export_errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    
    # Tracking
    download_count: int = Field(default=0, ge=0)
    last_accessed: Optional[datetime] = None
    retention_date: Optional[date] = None


class ReportConfig(BaseReportModel):
    """Configuration for report generation operations"""
    # General settings
    firm_name: str = Field(default="Legal Firm")
    firm_address: str = Field(default="")
    firm_phone: str = Field(default="")
    firm_email: str = Field(default="")
    
    # Default formatting
    default_citation_style: CitationStyle = Field(CitationStyle.BLUEBOOK)
    default_document_standard: DocumentStandard = Field(DocumentStandard.FEDERAL_COURT)
    default_confidentiality: ConfidentialityLevel = Field(ConfidentialityLevel.CONFIDENTIAL)
    
    # Templates
    template_directory: Optional[Path] = None
    custom_templates: Dict[str, str] = Field(default_factory=dict)
    default_templates: Dict[ReportType, str] = Field(default_factory=dict)
    
    # Quality assurance
    require_review: bool = Field(default=True)
    require_approval: bool = Field(default=True)
    auto_spell_check: bool = Field(default=True)
    auto_citation_check: bool = Field(default=True)
    
    # Export settings
    default_export_format: ExportFormat = Field(ExportFormat.PDF)
    output_directory: Optional[Path] = None
    archive_exports: bool = Field(default=True)
    retention_days: int = Field(default=2555, ge=1)  # 7 years default
    
    # Security settings
    encrypt_confidential: bool = Field(default=True)
    digital_signatures: bool = Field(default=False)
    access_logging: bool = Field(default=True)
    
    # Performance settings
    max_file_size_mb: int = Field(default=100, ge=1)
    batch_processing_enabled: bool = Field(default=True)
    parallel_generation: bool = Field(default=True)
    max_concurrent_jobs: int = Field(default=4, ge=1)
    
    # Integration settings
    case_management_integration: bool = Field(default=False)
    document_management_integration: bool = Field(default=False)
    billing_integration: bool = Field(default=False)
    
    # Logging and monitoring
    log_level: str = Field(default="INFO")
    log_file: Optional[Path] = None
    enable_analytics: bool = Field(default=True)
    performance_monitoring: bool = Field(default=True)


# Update ReportSection to handle self-referencing
ReportSection.model_rebuild()


class ReportGenerator:
    """
    Main class for legal report generation operations.
    
    Provides comprehensive report generation capabilities including:
    - Standardized fact sheet creation
    - Comprehensive evidence cataloging
    - Auto-populated legal brief templates
    - Multi-format document export
    - Template management and customization
    - Batch report processing
    - Quality assurance and validation
    """
    
    def __init__(self, config: Optional[ReportConfig] = None):
        """Initialize the report generator"""
        self.config = config or ReportConfig()
        self.session_id = str(uuid4())
        
        # Initialize component generators (will be set by individual modules)
        self._fact_sheet_generator = None
        self._evidence_cataloger = None
        self._legal_brief_formatter = None
        self._export_manager = None
        
        # Report state
        self._active_reports: Dict[str, Any] = {}
        self._generation_history: List[Dict[str, Any]] = []
        
        logger.info(f"Report Generator initialized with session {self.session_id}")
    
    @property
    def fact_sheet_generator(self):
        """Lazy-loaded fact sheet generator"""
        if self._fact_sheet_generator is None:
            from .fact_sheet_generator import FactSheetGenerator
            self._fact_sheet_generator = FactSheetGenerator(self.config)
        return self._fact_sheet_generator
    
    @property
    def evidence_cataloger(self):
        """Lazy-loaded evidence cataloger"""
        if self._evidence_cataloger is None:
            from .evidence_cataloger import EvidenceCataloger
            self._evidence_cataloger = EvidenceCataloger(self.config)
        return self._evidence_cataloger
    
    @property
    def legal_brief_formatter(self):
        """Lazy-loaded legal brief formatter"""
        if self._legal_brief_formatter is None:
            from .legal_brief_formatter import LegalBriefFormatter
            self._legal_brief_formatter = LegalBriefFormatter(self.config)
        return self._legal_brief_formatter
    
    @property
    def export_manager(self):
        """Lazy-loaded export manager"""
        if self._export_manager is None:
            from .export_manager import ExportManager
            self._export_manager = ExportManager(self.config)
        return self._export_manager
    
    def generate_fact_sheet(
        self,
        case_data: CaseData,
        template: Optional[str] = None,
        author: Optional[PersonInfo] = None
    ) -> FactSheet:
        """
        Generate standardized fact sheet for a case
        
        Args:
            case_data: Complete case information
            template: Custom template name (optional)
            author: Report author information
            
        Returns:
            FactSheet with standardized case summary
        """
        return self.fact_sheet_generator.generate(case_data, template, author)
    
    def catalog_evidence(
        self,
        evidence_list: List[Evidence],
        case_data: CaseData,
        custodian: PersonInfo
    ) -> EvidenceCatalog:
        """
        Create comprehensive evidence inventory and catalog
        
        Args:
            evidence_list: List of evidence items to catalog
            case_data: Associated case information
            custodian: Person responsible for evidence custody
            
        Returns:
            EvidenceCatalog with organized evidence inventory
        """
        return self.evidence_cataloger.catalog(evidence_list, case_data, custodian)
    
    def format_legal_brief(
        self,
        case_data: CaseData,
        template: Union[str, LegalTemplate],
        brief_type: str,
        author: PersonInfo
    ) -> LegalBrief:
        """
        Generate auto-populated legal brief from template
        
        Args:
            case_data: Case information for brief population
            template: Brief template name or LegalTemplate object
            brief_type: Type of brief (motion, response, etc.)
            author: Brief author information
            
        Returns:
            LegalBrief with populated content
        """
        return self.legal_brief_formatter.format(case_data, template, brief_type, author)
    
    def export_report(
        self,
        report: Union[FactSheet, EvidenceCatalog, LegalBrief],
        format_type: ExportFormat,
        settings: Optional[ExportSettings] = None,
        exported_by: Optional[PersonInfo] = None
    ) -> ExportedReport:
        """
        Export report to specified format with quality validation
        
        Args:
            report: Report to export
            format_type: Target export format
            settings: Custom export settings
            exported_by: Person exporting the report
            
        Returns:
            ExportedReport with file information and metadata
        """
        return self.export_manager.export(report, format_type, settings, exported_by)
    
    def batch_generate_reports(
        self,
        cases: List[CaseData],
        report_types: List[ReportType],
        settings: Optional[Dict[str, Any]] = None
    ) -> List[Union[FactSheet, EvidenceCatalog, LegalBrief]]:
        """
        Generate multiple reports in batch process
        
        Args:
            cases: List of cases to generate reports for
            report_types: Types of reports to generate
            settings: Batch processing settings
            
        Returns:
            List of generated reports
        """
        results = []
        batch_settings = settings or {}
        
        logger.info(f"Starting batch generation for {len(cases)} cases")
        
        for case_data in cases:
            case_reports = []
            
            for report_type in report_types:
                try:
                    if report_type == ReportType.FACT_SHEET:
                        report = self.generate_fact_sheet(case_data)
                    elif report_type == ReportType.EVIDENCE_CATALOG:
                        # Need custodian info for evidence catalog
                        custodian = case_data.attorneys[0] if case_data.attorneys else PersonInfo(
                            full_name="Unknown Custodian",
                            role="attorney"
                        )
                        report = self.catalog_evidence(case_data.evidence_list, case_data, custodian)
                    elif report_type == ReportType.LEGAL_BRIEF:
                        # Need author and brief type for legal brief
                        author = case_data.attorneys[0] if case_data.attorneys else PersonInfo(
                            full_name="Unknown Author",
                            role="attorney"
                        )
                        report = self.format_legal_brief(
                            case_data, 
                            "standard", 
                            "motion", 
                            author
                        )
                    else:
                        logger.warning(f"Unsupported report type for batch generation: {report_type}")
                        continue
                    
                    case_reports.append(report)
                    logger.info(f"Generated {report_type} for case {case_data.case_info.case_number}")
                    
                except Exception as e:
                    logger.error(f"Failed to generate {report_type} for case {case_data.case_info.case_number}: {str(e)}")
                    continue
            
            results.extend(case_reports)
        
        logger.info(f"Batch generation completed: {len(results)} reports generated")
        return results
    
    def validate_report(
        self,
        report: Union[FactSheet, EvidenceCatalog, LegalBrief]
    ) -> Dict[str, Any]:
        """
        Perform quality assurance validation on generated report
        
        Args:
            report: Report to validate
            
        Returns:
            Dictionary with validation results and recommendations
        """
        validation_results = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "recommendations": [],
            "completeness_score": 0.0,
            "quality_score": 0.0
        }
        
        # Basic validation checks
        if not report.id:
            validation_results["errors"].append("Report missing unique identifier")
            validation_results["valid"] = False
        
        if isinstance(report, FactSheet):
            if not report.executive_summary:
                validation_results["warnings"].append("Executive summary is empty")
            if not report.factual_background:
                validation_results["errors"].append("Factual background is required")
                validation_results["valid"] = False
        
        elif isinstance(report, EvidenceCatalog):
            if not report.case_data.evidence_list:
                validation_results["warnings"].append("No evidence items cataloged")
            if not report.inventory_complete:
                validation_results["recommendations"].append("Complete evidence inventory verification")
        
        elif isinstance(report, LegalBrief):
            if not report.argument_sections:
                validation_results["errors"].append("Legal brief must contain argument sections")
                validation_results["valid"] = False
            if not report.authorities_cited:
                validation_results["warnings"].append("No legal authorities cited")
        
        # Calculate quality scores
        completeness_factors = []
        quality_factors = []
        
        # Add specific scoring logic here based on report type
        if validation_results["valid"]:
            validation_results["completeness_score"] = sum(completeness_factors) / len(completeness_factors) if completeness_factors else 0.8
            validation_results["quality_score"] = sum(quality_factors) / len(quality_factors) if quality_factors else 0.8
        
        return validation_results
    
    def get_available_templates(
        self,
        template_type: Optional[TemplateType] = None,
        jurisdiction: Optional[str] = None
    ) -> List[TemplateMetadata]:
        """
        Get list of available templates with filtering
        
        Args:
            template_type: Filter by template type
            jurisdiction: Filter by jurisdiction
            
        Returns:
            List of available template metadata
        """
        # This would typically query a template database
        # For now, return sample templates
        templates = [
            TemplateMetadata(
                template_id="fact_sheet_standard",
                name="Standard Fact Sheet",
                description="Standard fact sheet template for general cases",
                category="fact_sheet",
                tags=["standard", "general"],
                times_used=150,
                user_rating=4.2
            ),
            TemplateMetadata(
                template_id="brief_motion_standard",
                name="Standard Motion Brief",
                description="Standard template for motion briefs",
                category="legal_brief",
                tags=["motion", "standard"],
                times_used=89,
                user_rating=4.5
            ),
            TemplateMetadata(
                template_id="evidence_catalog_forensic",
                name="Forensic Evidence Catalog",
                description="Specialized template for forensic evidence cataloging",
                category="evidence_catalog",
                tags=["forensic", "technical"],
                times_used=45,
                user_rating=4.8
            )
        ]
        
        # Apply filters
        filtered_templates = templates
        if template_type:
            filtered_templates = [t for t in filtered_templates if template_type.value in t.tags]
        if jurisdiction:
            # Would filter by jurisdiction if stored in template metadata
            pass
        
        return filtered_templates
    
    def save_template(
        self,
        template: LegalTemplate,
        overwrite: bool = False
    ) -> TemplateMetadata:
        """
        Save custom template for future use
        
        Args:
            template: Template to save
            overwrite: Whether to overwrite existing template
            
        Returns:
            TemplateMetadata for saved template
        """
        # This would typically save to a template database
        metadata = TemplateMetadata(
            template_id=template.id,
            name=template.name,
            description=f"Custom {template.template_type} template",
            category=template.template_type.value,
            times_used=0
        )
        
        logger.info(f"Template saved: {template.name}")
        return metadata
    
    def get_generation_history(self) -> List[Dict[str, Any]]:
        """Get history of report generation for current session"""
        return self._generation_history.copy()
    
    def get_active_reports(self) -> Dict[str, Any]:
        """Get currently active/in-progress reports"""
        return self._active_reports.copy()


# Convenience functions for direct module usage

def create_report_generator(config: Optional[ReportConfig] = None) -> ReportGenerator:
    """Create a new ReportGenerator instance"""
    return ReportGenerator(config)


def create_default_config() -> ReportConfig:
    """Create a default report configuration"""
    return ReportConfig()


def create_case_data(
    case_number: str,
    case_name: str,
    court: str,
    jurisdiction: str,
    practice_area: str
) -> CaseData:
    """Create basic case data structure"""
    case_info = CaseInfo(
        case_number=case_number,
        case_name=case_name,
        court=court,
        jurisdiction=jurisdiction,
        practice_area=practice_area,
        case_type="civil"  # default
    )
    
    return CaseData(case_info=case_info)


# Export all models and classes
__all__ = [
    # Main classes
    'ReportGenerator',
    'ReportConfig',
    
    # Data models
    'FactSheet',
    'EvidenceCatalog',
    'LegalBrief', 
    'ExportedReport',
    'CaseData',
    'Evidence',
    'LegalTemplate',
    'ReportSection',
    'ExportSettings',
    'TemplateMetadata',
    'PersonInfo',
    'CaseInfo',
    'LegalCitation',
    
    # Enums
    'ReportType',
    'ExportFormat',
    'EvidenceType',
    'TemplateType', 
    'ReportStatus',
    'CitationStyle',
    'DocumentStandard',
    'EvidenceAuthenticity',
    'ConfidentialityLevel',
    
    # Base classes
    'BaseReportModel',
    
    # Convenience functions
    'create_report_generator',
    'create_default_config',
    'create_case_data',
]