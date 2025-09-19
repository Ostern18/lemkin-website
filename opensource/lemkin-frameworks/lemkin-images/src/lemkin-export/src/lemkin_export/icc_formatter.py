"""
ICC (International Criminal Court) submission format compliance module.

This module provides functionality to format legal data according to ICC
submission requirements, including XML schema compliance and metadata standards.
"""

import uuid
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from lxml import etree
from pydantic import BaseModel, Field, validator
from loguru import logger
from defusedxml import ElementTree as DefusedET

from .core import (
    CaseData, Evidence, ICCSubmission, SubmissionMetadata, ValidationResult,
    CourtType, ExportError, ValidationError, FormatError
)


class ICCMetadata(BaseModel):
    """ICC-specific metadata structure."""
    document_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    icc_case_number: Optional[str] = None
    document_type: str = Field(..., regex=r"^(filing|evidence|motion|brief|order|judgment)$")
    classification: str = Field(default="public", regex=r"^(public|confidential|ex_parte|under_seal)$")
    filing_party: str = Field(..., regex=r"^(prosecution|defense|victims|registry|chambers)$")
    language: str = Field(default="en", regex=r"^[a-z]{2}(-[A-Z]{2})?$")
    urgency: str = Field(default="normal", regex=r"^(routine|normal|urgent|immediate)$")
    pages: int = Field(default=0, ge=0)
    word_count: int = Field(default=0, ge=0)
    filing_date: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    received_date: Optional[datetime] = None
    public_redacted_version: bool = Field(default=False)
    confidentiality_level: str = Field(default="public")
    related_documents: List[str] = Field(default_factory=list)
    keywords: List[str] = Field(default_factory=list)
    subject_matter: List[str] = Field(default_factory=list)
    
    @validator('icc_case_number')
    def validate_case_number(cls, v):
        if v and not v.startswith('ICC-'):
            raise ValueError('ICC case number must start with ICC-')
        return v


class ICCSubmissionSchema:
    """ICC XML submission schema definitions and validation."""
    
    NAMESPACE = "http://www.icc-cpi.int/submission/v1.0"
    SCHEMA_VERSION = "1.0"
    
    @classmethod
    def get_xsd_schema(cls) -> str:
        """Get the ICC submission XSD schema."""
        return """<?xml version="1.0" encoding="UTF-8"?>
<xs:schema xmlns:xs="http://www.w3.org/2001/XMLSchema"
           xmlns:icc="http://www.icc-cpi.int/submission/v1.0"
           targetNamespace="http://www.icc-cpi.int/submission/v1.0"
           elementFormDefault="qualified">
           
    <xs:element name="ICCSubmission">
        <xs:complexType>
            <xs:sequence>
                <xs:element name="Metadata" type="icc:MetadataType"/>
                <xs:element name="Document" type="icc:DocumentType"/>
                <xs:element name="Evidence" type="icc:EvidenceType" minOccurs="0" maxOccurs="unbounded"/>
                <xs:element name="Signature" type="icc:SignatureType" minOccurs="0"/>
            </xs:sequence>
            <xs:attribute name="version" type="xs:string" fixed="1.0"/>
            <xs:attribute name="submissionId" type="xs:string" use="required"/>
        </xs:complexType>
    </xs:element>
    
    <xs:complexType name="MetadataType">
        <xs:sequence>
            <xs:element name="DocumentId" type="xs:string"/>
            <xs:element name="CaseNumber" type="xs:string" minOccurs="0"/>
            <xs:element name="DocumentType" type="icc:DocumentTypeEnum"/>
            <xs:element name="Classification" type="icc:ClassificationEnum"/>
            <xs:element name="FilingParty" type="icc:FilingPartyEnum"/>
            <xs:element name="Language" type="xs:language"/>
            <xs:element name="FilingDate" type="xs:dateTime"/>
            <xs:element name="Pages" type="xs:nonNegativeInteger"/>
            <xs:element name="WordCount" type="xs:nonNegativeInteger"/>
            <xs:element name="Keywords" type="icc:KeywordListType" minOccurs="0"/>
            <xs:element name="SubjectMatter" type="icc:SubjectMatterListType" minOccurs="0"/>
        </xs:sequence>
    </xs:complexType>
    
    <xs:complexType name="DocumentType">
        <xs:sequence>
            <xs:element name="Title" type="xs:string"/>
            <xs:element name="Content" type="xs:string"/>
            <xs:element name="Attachments" type="icc:AttachmentListType" minOccurs="0"/>
        </xs:sequence>
    </xs:complexType>
    
    <xs:complexType name="EvidenceType">
        <xs:sequence>
            <xs:element name="EvidenceId" type="xs:string"/>
            <xs:element name="Title" type="xs:string"/>
            <xs:element name="Type" type="icc:EvidenceTypeEnum"/>
            <xs:element name="Source" type="xs:string" minOccurs="0"/>
            <xs:element name="CollectionDate" type="xs:dateTime" minOccurs="0"/>
            <xs:element name="ChainOfCustody" type="icc:ChainOfCustodyType" minOccurs="0"/>
            <xs:element name="Hash" type="xs:string" minOccurs="0"/>
        </xs:sequence>
    </xs:complexType>
    
    <xs:simpleType name="DocumentTypeEnum">
        <xs:restriction base="xs:string">
            <xs:enumeration value="filing"/>
            <xs:enumeration value="evidence"/>
            <xs:enumeration value="motion"/>
            <xs:enumeration value="brief"/>
            <xs:enumeration value="order"/>
            <xs:enumeration value="judgment"/>
        </xs:restriction>
    </xs:simpleType>
    
    <xs:simpleType name="ClassificationEnum">
        <xs:restriction base="xs:string">
            <xs:enumeration value="public"/>
            <xs:enumeration value="confidential"/>
            <xs:enumeration value="ex_parte"/>
            <xs:enumeration value="under_seal"/>
        </xs:restriction>
    </xs:simpleType>
    
    <xs:simpleType name="FilingPartyEnum">
        <xs:restriction base="xs:string">
            <xs:enumeration value="prosecution"/>
            <xs:enumeration value="defense"/>
            <xs:enumeration value="victims"/>
            <xs:enumeration value="registry"/>
            <xs:enumeration value="chambers"/>
        </xs:restriction>
    </xs:simpleType>
    
    <xs:simpleType name="EvidenceTypeEnum">
        <xs:restriction base="xs:string">
            <xs:enumeration value="document"/>
            <xs:enumeration value="image"/>
            <xs:enumeration value="video"/>
            <xs:enumeration value="audio"/>
            <xs:enumeration value="testimony"/>
            <xs:enumeration value="expert_report"/>
            <xs:enumeration value="physical_evidence"/>
            <xs:enumeration value="digital_evidence"/>
        </xs:restriction>
    </xs:simpleType>
    
    <!-- Additional complex types would be defined here -->
    
</xs:schema>"""

    @classmethod
    def validate_xml(cls, xml_content: str) -> ValidationResult:
        """
        Validate XML content against ICC schema.
        
        Args:
            xml_content: The XML content to validate
            
        Returns:
            Validation result with errors/warnings
        """
        errors = []
        warnings = []
        
        try:
            # Parse XML with defusedxml for security
            root = DefusedET.fromstring(xml_content.encode('utf-8'))
            
            # Basic structure validation
            if root.tag != f"{{{cls.NAMESPACE}}}ICCSubmission":
                errors.append("Root element must be ICCSubmission with correct namespace")
            
            # Check required attributes
            if 'submissionId' not in root.attrib:
                errors.append("Missing required attribute: submissionId")
            
            if 'version' not in root.attrib:
                errors.append("Missing required attribute: version")
            elif root.attrib['version'] != cls.SCHEMA_VERSION:
                warnings.append(f"Schema version mismatch: expected {cls.SCHEMA_VERSION}")
            
            # Validate required child elements
            required_elements = ['Metadata', 'Document']
            for element_name in required_elements:
                if root.find(f".//{{{cls.NAMESPACE}}}{element_name}") is None:
                    errors.append(f"Missing required element: {element_name}")
            
            # Additional validation would go here...
            
        except Exception as e:
            errors.append(f"XML parsing error: {str(e)}")
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            court=CourtType.ICC,
            format_version=cls.SCHEMA_VERSION,
            errors=errors,
            warnings=warnings,
            validator_version="1.0"
        )


class ICCFormatter:
    """
    Formats legal data according to ICC submission requirements.
    
    This class handles the conversion of case data into ICC-compliant
    XML format with proper metadata and structure.
    """
    
    def __init__(self, strict_validation: bool = True):
        """
        Initialize ICC formatter.
        
        Args:
            strict_validation: Whether to enforce strict validation
        """
        self.strict_validation = strict_validation
        self.schema = ICCSubmissionSchema()
    
    def format_case(self, case_data: CaseData) -> ICCSubmission:
        """
        Format complete case data for ICC submission.
        
        Args:
            case_data: The case data to format
            
        Returns:
            ICC submission object
            
        Raises:
            FormatError: If formatting fails
            ValidationError: If validation fails in strict mode
        """
        try:
            logger.info(f"Formatting case {case_data.case_id} for ICC submission")
            
            # Create ICC metadata
            icc_metadata = self._create_icc_metadata(case_data.metadata)
            
            # Generate XML content
            xml_content = self._generate_xml(case_data, icc_metadata)
            
            # Validate if strict mode
            validation_result = self.schema.validate_xml(xml_content)
            if self.strict_validation and not validation_result.is_valid:
                raise ValidationError(f"ICC format validation failed: {validation_result.errors}")
            
            # Create ICC submission
            submission = ICCSubmission(
                icc_case_number=case_data.case_number,
                document_type=icc_metadata.document_type,
                classification=icc_metadata.classification,
                filing_party=icc_metadata.filing_party,
                language=icc_metadata.language,
                pages=icc_metadata.pages,
                word_count=icc_metadata.word_count,
                xml_content=xml_content,
                metadata=case_data.metadata,
                validation_status=validation_result
            )
            
            logger.info(f"Successfully formatted case {case_data.case_id} for ICC")
            return submission
            
        except Exception as e:
            logger.error(f"Failed to format case {case_data.case_id} for ICC: {e}")
            raise FormatError(f"ICC formatting failed: {e}") from e
    
    def format_evidence(self, evidence: Evidence) -> str:
        """
        Format a single evidence item for ICC submission.
        
        Args:
            evidence: The evidence item to format
            
        Returns:
            XML string for the evidence item
        """
        evidence_elem = ET.Element("Evidence", xmlns=ICCSubmissionSchema.NAMESPACE)
        
        # Add evidence details
        ET.SubElement(evidence_elem, "EvidenceId").text = evidence.evidence_id
        ET.SubElement(evidence_elem, "Title").text = evidence.title
        ET.SubElement(evidence_elem, "Type").text = evidence.evidence_type.value
        
        if evidence.source:
            ET.SubElement(evidence_elem, "Source").text = evidence.source
        
        if evidence.collection_date:
            ET.SubElement(evidence_elem, "CollectionDate").text = evidence.collection_date.isoformat()
        
        if evidence.file_hash:
            ET.SubElement(evidence_elem, "Hash").text = evidence.file_hash
        
        # Add chain of custody if available
        if evidence.chain_of_custody:
            chain_elem = ET.SubElement(evidence_elem, "ChainOfCustody")
            for custody in evidence.chain_of_custody:
                custody_entry = ET.SubElement(chain_elem, "CustodyEntry")
                ET.SubElement(custody_entry, "Custodian").text = custody.custodian_name
                ET.SubElement(custody_entry, "Action").text = custody.action
                ET.SubElement(custody_entry, "Timestamp").text = custody.timestamp.isoformat()
                if custody.location:
                    ET.SubElement(custody_entry, "Location").text = custody.location
        
        return ET.tostring(evidence_elem, encoding='unicode')
    
    def _create_icc_metadata(self, metadata: SubmissionMetadata) -> ICCMetadata:
        """Create ICC-specific metadata from submission metadata."""
        return ICCMetadata(
            icc_case_number=metadata.case_number,
            document_type="filing",  # Default, could be configurable
            filing_party="prosecution",  # Default, should be configurable
            language=metadata.language,
            filing_date=metadata.submission_date,
            classification=metadata.classification_level,
            keywords=metadata.keywords,
            pages=0,  # Would be calculated from content
            word_count=0,  # Would be calculated from content
        )
    
    def _generate_xml(self, case_data: CaseData, icc_metadata: ICCMetadata) -> str:
        """Generate the complete ICC submission XML."""
        # Create root element
        root = ET.Element(
            "ICCSubmission",
            xmlns=ICCSubmissionSchema.NAMESPACE,
            version=ICCSubmissionSchema.SCHEMA_VERSION,
            submissionId=case_data.case_id
        )
        
        # Add metadata section
        metadata_elem = ET.SubElement(root, "Metadata")
        ET.SubElement(metadata_elem, "DocumentId").text = icc_metadata.document_id
        
        if icc_metadata.icc_case_number:
            ET.SubElement(metadata_elem, "CaseNumber").text = icc_metadata.icc_case_number
        
        ET.SubElement(metadata_elem, "DocumentType").text = icc_metadata.document_type
        ET.SubElement(metadata_elem, "Classification").text = icc_metadata.classification
        ET.SubElement(metadata_elem, "FilingParty").text = icc_metadata.filing_party
        ET.SubElement(metadata_elem, "Language").text = icc_metadata.language
        ET.SubElement(metadata_elem, "FilingDate").text = icc_metadata.filing_date.isoformat()
        ET.SubElement(metadata_elem, "Pages").text = str(icc_metadata.pages)
        ET.SubElement(metadata_elem, "WordCount").text = str(icc_metadata.word_count)
        
        # Add keywords if present
        if icc_metadata.keywords:
            keywords_elem = ET.SubElement(metadata_elem, "Keywords")
            for keyword in icc_metadata.keywords:
                ET.SubElement(keywords_elem, "Keyword").text = keyword
        
        # Add subject matter if present
        if icc_metadata.subject_matter:
            subject_elem = ET.SubElement(metadata_elem, "SubjectMatter")
            for subject in icc_metadata.subject_matter:
                ET.SubElement(subject_elem, "Subject").text = subject
        
        # Add document section
        doc_elem = ET.SubElement(root, "Document")
        ET.SubElement(doc_elem, "Title").text = case_data.case_name
        ET.SubElement(doc_elem, "Content").text = f"Case submission for {case_data.case_name}"
        
        # Add attachments if any
        if case_data.evidence:
            attachments_elem = ET.SubElement(doc_elem, "Attachments")
            for evidence in case_data.evidence:
                attachment_elem = ET.SubElement(attachments_elem, "Attachment")
                ET.SubElement(attachment_elem, "FileName").text = f"{evidence.title}.{evidence.mime_type or 'pdf'}"
                ET.SubElement(attachment_elem, "Type").text = evidence.evidence_type.value
                if evidence.file_hash:
                    ET.SubElement(attachment_elem, "Hash").text = evidence.file_hash
        
        # Add evidence sections
        for evidence in case_data.evidence:
            evidence_xml = self.format_evidence(evidence)
            # Parse and append the evidence XML
            evidence_elem = DefusedET.fromstring(evidence_xml)
            root.append(evidence_elem)
        
        # Add case parties information
        if case_data.parties:
            parties_elem = ET.SubElement(root, "Parties")
            for party_type, party_list in case_data.parties.items():
                party_type_elem = ET.SubElement(parties_elem, "PartyType", type=party_type)
                for party in party_list:
                    ET.SubElement(party_type_elem, "Party").text = party
        
        # Format and return XML
        ET.indent(root, space="  ")
        xml_str = ET.tostring(root, encoding='unicode', xml_declaration=True)
        
        return xml_str
    
    def validate_submission(self, submission: ICCSubmission) -> ValidationResult:
        """
        Validate an ICC submission.
        
        Args:
            submission: The ICC submission to validate
            
        Returns:
            Validation result
        """
        return self.schema.validate_xml(submission.xml_content)
    
    def export_to_file(
        self,
        submission: ICCSubmission,
        file_path: Union[str, Path],
        include_metadata: bool = True
    ) -> None:
        """
        Export ICC submission to file.
        
        Args:
            submission: The ICC submission to export
            file_path: Output file path
            include_metadata: Whether to include metadata file
        """
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write XML content
        xml_file = file_path.with_suffix('.xml')
        with open(xml_file, 'w', encoding='utf-8') as f:
            f.write(submission.xml_content)
        
        # Write metadata if requested
        if include_metadata:
            metadata_file = file_path.with_suffix('.metadata.json')
            with open(metadata_file, 'w', encoding='utf-8') as f:
                f.write(submission.json(indent=2))
        
        logger.info(f"ICC submission exported to {xml_file}")


def format_for_icc(case_data: CaseData) -> ICCSubmission:
    """
    Convenience function to format case data for ICC submission.
    
    Args:
        case_data: The case data to format
        
    Returns:
        ICC submission object
    """
    formatter = ICCFormatter(strict_validation=True)
    return formatter.format_case(case_data)