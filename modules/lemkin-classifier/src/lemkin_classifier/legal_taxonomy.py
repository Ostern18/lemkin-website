"""
Legal document taxonomy and category definitions for standardized classification.

This module defines the hierarchical structure of legal document types and categories
used for automated classification in legal proceedings and investigations.
"""

from enum import Enum
from typing import Dict, List, Optional, Set
from pydantic import BaseModel, Field


class DocumentType(str, Enum):
    """Primary document types in legal proceedings"""
    # Evidence documents
    WITNESS_STATEMENT = "witness_statement"
    POLICE_REPORT = "police_report"
    MEDICAL_RECORD = "medical_record"
    COURT_FILING = "court_filing"
    
    # Government and institutional documents
    GOVERNMENT_DOCUMENT = "government_document"
    MILITARY_REPORT = "military_report"
    DIPLOMATIC_COMMUNICATION = "diplomatic_communication"
    OFFICIAL_CORRESPONDENCE = "official_correspondence"
    
    # Communication records
    EMAIL = "email"
    SMS_MESSAGE = "sms_message"
    PHONE_TRANSCRIPT = "phone_transcript"
    CHAT_MESSAGE = "chat_message"
    SOCIAL_MEDIA_POST = "social_media_post"
    
    # Expert and technical documents
    EXPERT_TESTIMONY = "expert_testimony"
    FORENSIC_REPORT = "forensic_report"
    TECHNICAL_ANALYSIS = "technical_analysis"
    SCIENTIFIC_REPORT = "scientific_report"
    
    # Media and multimedia evidence
    PHOTO_EVIDENCE = "photo_evidence"
    VIDEO_EVIDENCE = "video_evidence"
    AUDIO_RECORDING = "audio_recording"
    DOCUMENT_SCAN = "document_scan"
    
    # Financial and business documents
    FINANCIAL_RECORD = "financial_record"
    CONTRACT = "contract"
    BUSINESS_DOCUMENT = "business_document"
    INVOICE = "invoice"
    
    # Legal process documents
    SUBPOENA = "subpoena"
    WARRANT = "warrant"
    AFFIDAVIT = "affidavit"
    DEPOSITION = "deposition"
    MOTION = "motion"
    
    # Other/Unknown
    OTHER = "other"
    UNKNOWN = "unknown"


class LegalDomain(str, Enum):
    """Legal domains and areas of law"""
    CRIMINAL_LAW = "criminal_law"
    CIVIL_RIGHTS = "civil_rights"
    INTERNATIONAL_HUMANITARIAN_LAW = "international_humanitarian_law"
    HUMAN_RIGHTS_LAW = "human_rights_law"
    ADMINISTRATIVE_LAW = "administrative_law"
    CONSTITUTIONAL_LAW = "constitutional_law"
    CORPORATE_LAW = "corporate_law"
    FAMILY_LAW = "family_law"
    IMMIGRATION_LAW = "immigration_law"
    ENVIRONMENTAL_LAW = "environmental_law"
    GENERAL = "general"


class LegalDocumentCategory(BaseModel):
    """Structured legal document category with hierarchical classification"""
    
    document_type: DocumentType = Field(description="Primary document type")
    legal_domain: LegalDomain = Field(description="Legal domain or area of law")
    subcategory: Optional[str] = Field(default=None, description="Specific subcategory within type")
    urgency_level: str = Field(default="medium", description="Processing urgency: low, medium, high, critical")
    sensitivity_level: str = Field(default="standard", description="Sensitivity: public, internal, confidential, restricted")
    
    # Classification metadata
    keywords: List[str] = Field(default_factory=list, description="Key terms associated with category")
    required_fields: List[str] = Field(default_factory=list, description="Required metadata fields")
    typical_sources: List[str] = Field(default_factory=list, description="Typical document sources")
    
    # Processing requirements
    requires_human_review: bool = Field(default=False, description="Whether human review is required")
    redaction_required: bool = Field(default=False, description="Whether PII redaction is typically needed")
    chain_of_custody_critical: bool = Field(default=False, description="Whether chain of custody is critical")
    
    class Config:
        use_enum_values = True


class CategoryHierarchy(BaseModel):
    """Hierarchical structure of legal document categories"""
    
    primary_categories: Dict[DocumentType, LegalDocumentCategory]
    subcategories: Dict[str, List[LegalDocumentCategory]]
    domain_mapping: Dict[LegalDomain, List[DocumentType]]
    
    class Config:
        use_enum_values = True


# Predefined category definitions
CATEGORY_DEFINITIONS: Dict[DocumentType, LegalDocumentCategory] = {
    DocumentType.WITNESS_STATEMENT: LegalDocumentCategory(
        document_type=DocumentType.WITNESS_STATEMENT,
        legal_domain=LegalDomain.CRIMINAL_LAW,
        urgency_level="high",
        sensitivity_level="confidential",
        keywords=["witness", "statement", "testimony", "account", "observed", "saw"],
        required_fields=["witness_name", "date", "location", "incident_reference"],
        typical_sources=["police_station", "court", "legal_office", "investigator"],
        requires_human_review=True,
        redaction_required=True,
        chain_of_custody_critical=True
    ),
    
    DocumentType.POLICE_REPORT: LegalDocumentCategory(
        document_type=DocumentType.POLICE_REPORT,
        legal_domain=LegalDomain.CRIMINAL_LAW,
        urgency_level="high",
        sensitivity_level="restricted",
        keywords=["police", "report", "incident", "crime", "arrest", "investigation"],
        required_fields=["report_number", "officer_name", "date", "location"],
        typical_sources=["police_department", "law_enforcement", "sheriff_office"],
        requires_human_review=True,
        redaction_required=True,
        chain_of_custody_critical=True
    ),
    
    DocumentType.MEDICAL_RECORD: LegalDocumentCategory(
        document_type=DocumentType.MEDICAL_RECORD,
        legal_domain=LegalDomain.CIVIL_RIGHTS,
        urgency_level="medium",
        sensitivity_level="restricted",
        keywords=["medical", "health", "patient", "diagnosis", "treatment", "hospital"],
        required_fields=["patient_id", "provider", "date", "medical_record_number"],
        typical_sources=["hospital", "clinic", "medical_center", "doctor_office"],
        requires_human_review=True,
        redaction_required=True,
        chain_of_custody_critical=False
    ),
    
    DocumentType.COURT_FILING: LegalDocumentCategory(
        document_type=DocumentType.COURT_FILING,
        legal_domain=LegalDomain.GENERAL,
        urgency_level="high",
        sensitivity_level="internal",
        keywords=["court", "filing", "motion", "petition", "complaint", "answer"],
        required_fields=["case_number", "court", "filing_date", "document_type"],
        typical_sources=["courthouse", "legal_office", "attorney", "clerk"],
        requires_human_review=False,
        redaction_required=False,
        chain_of_custody_critical=True
    ),
    
    DocumentType.GOVERNMENT_DOCUMENT: LegalDocumentCategory(
        document_type=DocumentType.GOVERNMENT_DOCUMENT,
        legal_domain=LegalDomain.ADMINISTRATIVE_LAW,
        urgency_level="medium",
        sensitivity_level="confidential",
        keywords=["government", "official", "agency", "department", "ministry", "bureau"],
        required_fields=["agency", "document_id", "date", "classification"],
        typical_sources=["government_agency", "ministry", "department", "bureau"],
        requires_human_review=True,
        redaction_required=True,
        chain_of_custody_critical=True
    ),
    
    DocumentType.MILITARY_REPORT: LegalDocumentCategory(
        document_type=DocumentType.MILITARY_REPORT,
        legal_domain=LegalDomain.INTERNATIONAL_HUMANITARIAN_LAW,
        urgency_level="critical",
        sensitivity_level="restricted",
        keywords=["military", "armed forces", "combat", "operation", "mission", "deployment"],
        required_fields=["unit", "commander", "operation_name", "date", "classification"],
        typical_sources=["military_unit", "command", "defense_department"],
        requires_human_review=True,
        redaction_required=True,
        chain_of_custody_critical=True
    ),
    
    DocumentType.EMAIL: LegalDocumentCategory(
        document_type=DocumentType.EMAIL,
        legal_domain=LegalDomain.GENERAL,
        urgency_level="low",
        sensitivity_level="internal",
        keywords=["email", "message", "correspondence", "communication", "sent", "received"],
        required_fields=["sender", "recipient", "date", "subject"],
        typical_sources=["email_server", "email_client", "communication_platform"],
        requires_human_review=False,
        redaction_required=True,
        chain_of_custody_critical=False
    ),
    
    DocumentType.EXPERT_TESTIMONY: LegalDocumentCategory(
        document_type=DocumentType.EXPERT_TESTIMONY,
        legal_domain=LegalDomain.GENERAL,
        urgency_level="high",
        sensitivity_level="confidential",
        keywords=["expert", "testimony", "analysis", "opinion", "professional", "qualified"],
        required_fields=["expert_name", "qualifications", "date", "subject_matter"],
        typical_sources=["expert_witness", "consultant", "specialist"],
        requires_human_review=True,
        redaction_required=True,
        chain_of_custody_critical=True
    ),
    
    DocumentType.FORENSIC_REPORT: LegalDocumentCategory(
        document_type=DocumentType.FORENSIC_REPORT,
        legal_domain=LegalDomain.CRIMINAL_LAW,
        urgency_level="critical",
        sensitivity_level="restricted",
        keywords=["forensic", "analysis", "evidence", "laboratory", "scientific", "examination"],
        required_fields=["lab_id", "analyst", "evidence_id", "date", "methodology"],
        typical_sources=["forensic_lab", "crime_lab", "scientific_institution"],
        requires_human_review=True,
        redaction_required=False,
        chain_of_custody_critical=True
    ),
    
    DocumentType.FINANCIAL_RECORD: LegalDocumentCategory(
        document_type=DocumentType.FINANCIAL_RECORD,
        legal_domain=LegalDomain.CORPORATE_LAW,
        urgency_level="medium",
        sensitivity_level="confidential",
        keywords=["financial", "banking", "transaction", "account", "payment", "money"],
        required_fields=["account_number", "bank", "date", "transaction_type"],
        typical_sources=["bank", "financial_institution", "accounting_firm"],
        requires_human_review=False,
        redaction_required=True,
        chain_of_custody_critical=False
    ),
}

# Domain-specific document type mappings
DOMAIN_DOCUMENT_MAPPING: Dict[LegalDomain, List[DocumentType]] = {
    LegalDomain.CRIMINAL_LAW: [
        DocumentType.WITNESS_STATEMENT, DocumentType.POLICE_REPORT,
        DocumentType.FORENSIC_REPORT, DocumentType.EXPERT_TESTIMONY,
        DocumentType.COURT_FILING, DocumentType.WARRANT, DocumentType.AFFIDAVIT
    ],
    LegalDomain.CIVIL_RIGHTS: [
        DocumentType.MEDICAL_RECORD, DocumentType.GOVERNMENT_DOCUMENT,
        DocumentType.WITNESS_STATEMENT, DocumentType.EXPERT_TESTIMONY,
        DocumentType.COURT_FILING
    ],
    LegalDomain.INTERNATIONAL_HUMANITARIAN_LAW: [
        DocumentType.MILITARY_REPORT, DocumentType.GOVERNMENT_DOCUMENT,
        DocumentType.DIPLOMATIC_COMMUNICATION, DocumentType.WITNESS_STATEMENT,
        DocumentType.EXPERT_TESTIMONY
    ],
    LegalDomain.HUMAN_RIGHTS_LAW: [
        DocumentType.WITNESS_STATEMENT, DocumentType.MEDICAL_RECORD,
        DocumentType.GOVERNMENT_DOCUMENT, DocumentType.EXPERT_TESTIMONY,
        DocumentType.PHOTO_EVIDENCE, DocumentType.VIDEO_EVIDENCE
    ],
    LegalDomain.ADMINISTRATIVE_LAW: [
        DocumentType.GOVERNMENT_DOCUMENT, DocumentType.OFFICIAL_CORRESPONDENCE,
        DocumentType.COURT_FILING, DocumentType.EMAIL
    ],
    LegalDomain.CORPORATE_LAW: [
        DocumentType.FINANCIAL_RECORD, DocumentType.CONTRACT,
        DocumentType.BUSINESS_DOCUMENT, DocumentType.EMAIL,
        DocumentType.COURT_FILING
    ],
}


def get_category_hierarchy() -> CategoryHierarchy:
    """Get the complete legal document category hierarchy"""
    
    # Build subcategories based on legal domains
    subcategories = {}
    for domain, doc_types in DOMAIN_DOCUMENT_MAPPING.items():
        subcategories[domain.value] = [
            CATEGORY_DEFINITIONS[doc_type] for doc_type in doc_types
            if doc_type in CATEGORY_DEFINITIONS
        ]
    
    return CategoryHierarchy(
        primary_categories=CATEGORY_DEFINITIONS,
        subcategories=subcategories,
        domain_mapping=DOMAIN_DOCUMENT_MAPPING
    )


def get_supported_categories() -> List[DocumentType]:
    """Get list of all supported document categories"""
    return list(CATEGORY_DEFINITIONS.keys())


def validate_category(document_type: str, legal_domain: Optional[str] = None) -> bool:
    """
    Validate if a document type and legal domain combination is supported
    
    Args:
        document_type: Document type to validate
        legal_domain: Optional legal domain to validate
        
    Returns:
        True if valid combination, False otherwise
    """
    try:
        doc_type_enum = DocumentType(document_type)
    except ValueError:
        return False
    
    if legal_domain is None:
        return doc_type_enum in CATEGORY_DEFINITIONS
    
    try:
        domain_enum = LegalDomain(legal_domain)
    except ValueError:
        return False
    
    # Check if document type is valid for the legal domain
    if domain_enum in DOMAIN_DOCUMENT_MAPPING:
        return doc_type_enum in DOMAIN_DOCUMENT_MAPPING[domain_enum]
    
    return False


def get_category_keywords(document_type: DocumentType) -> List[str]:
    """Get keywords associated with a document category"""
    if document_type in CATEGORY_DEFINITIONS:
        return CATEGORY_DEFINITIONS[document_type].keywords
    return []


def get_urgency_level(document_type: DocumentType) -> str:
    """Get urgency level for a document type"""
    if document_type in CATEGORY_DEFINITIONS:
        return CATEGORY_DEFINITIONS[document_type].urgency_level
    return "medium"


def requires_human_review(document_type: DocumentType) -> bool:
    """Check if document type requires human review"""
    if document_type in CATEGORY_DEFINITIONS:
        return CATEGORY_DEFINITIONS[document_type].requires_human_review
    return True  # Default to requiring review for unknown types


def get_sensitivity_level(document_type: DocumentType) -> str:
    """Get sensitivity level for a document type"""
    if document_type in CATEGORY_DEFINITIONS:
        return CATEGORY_DEFINITIONS[document_type].sensitivity_level
    return "standard"


def get_categories_by_domain(legal_domain: LegalDomain) -> List[LegalDocumentCategory]:
    """Get all document categories for a specific legal domain"""
    if legal_domain not in DOMAIN_DOCUMENT_MAPPING:
        return []
    
    categories = []
    for doc_type in DOMAIN_DOCUMENT_MAPPING[legal_domain]:
        if doc_type in CATEGORY_DEFINITIONS:
            categories.append(CATEGORY_DEFINITIONS[doc_type])
    
    return categories


def get_high_priority_categories() -> List[DocumentType]:
    """Get document types that are high priority or critical"""
    high_priority = []
    for doc_type, category in CATEGORY_DEFINITIONS.items():
        if category.urgency_level in ["high", "critical"]:
            high_priority.append(doc_type)
    return high_priority


def get_categories_requiring_chain_of_custody() -> List[DocumentType]:
    """Get document types that require strict chain of custody"""
    custody_critical = []
    for doc_type, category in CATEGORY_DEFINITIONS.items():
        if category.chain_of_custody_critical:
            custody_critical.append(doc_type)
    return custody_critical