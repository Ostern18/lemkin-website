"""
Submission format validation and compliance checking module.

This module provides comprehensive validation of legal submission formats
for various international courts, ensuring compliance with specific
court requirements and technical standards.
"""

import json
import re
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass
from enum import Enum

from lxml import etree
from jsonschema import validate as json_validate, ValidationError as JsonValidationError
from pydantic import BaseModel, Field, validator
from loguru import logger
from defusedxml import ElementTree as DefusedET

from .core import (
    CourtType, ValidationResult, ICCSubmission, CourtPackage,
    SubmissionStatus, ValidationError, FormatError
)


class ValidationSeverity(str, Enum):
    """Severity levels for validation issues."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class ValidationCategory(str, Enum):
    """Categories of validation checks."""
    STRUCTURE = "structure"
    CONTENT = "content"
    FORMAT = "format"
    METADATA = "metadata"
    SIGNATURE = "signature"
    COMPLIANCE = "compliance"
    ACCESSIBILITY = "accessibility"


@dataclass
class ValidationRule:
    """Represents a validation rule."""
    rule_id: str
    description: str
    category: ValidationCategory
    severity: ValidationSeverity
    court_types: List[CourtType]
    check_function: Callable[[Any], Tuple[bool, str]]
    auto_fix: bool = False


class CourtSpecifications:
    """
    Court-specific format specifications and requirements.
    
    Contains detailed specifications for various international courts
    including file formats, metadata requirements, and submission rules.
    """
    
    ICC_SPECS = {
        "formats": ["xml", "pdf"],
        "max_file_size_mb": 50,
        "required_metadata": [
            "document_id", "case_number", "document_type", "classification",
            "filing_party", "language", "filing_date"
        ],
        "supported_languages": ["en", "fr", "es", "ar", "ru", "zh"],
        "document_types": [
            "filing", "evidence", "motion", "brief", "order", "judgment",
            "request", "response", "submission", "application"
        ],
        "classification_levels": ["public", "confidential", "ex_parte", "under_seal"],
        "filing_parties": ["prosecution", "defense", "victims", "registry", "chambers"],
        "xml_namespace": "http://www.icc-cpi.int/submission/v1.0",
        "schema_version": "1.0",
        "digital_signature_required": True,
        "max_pages": 500,
        "max_word_count": 100000
    }
    
    ICTY_SPECS = {
        "formats": ["pdf", "doc", "docx"],
        "max_file_size_mb": 25,
        "required_metadata": [
            "case_number", "document_type", "filing_party", "date_filed"
        ],
        "supported_languages": ["en", "fr", "bs", "hr", "sr"],
        "document_types": [
            "indictment", "motion", "brief", "evidence", "judgment", "order"
        ],
        "classification_levels": ["public", "confidential"],
        "digital_signature_required": False,
        "max_pages": 200
    }
    
    ICTR_SPECS = {
        "formats": ["pdf", "doc"],
        "max_file_size_mb": 30,
        "required_metadata": [
            "case_number", "document_type", "filing_party", "date_filed"
        ],
        "supported_languages": ["en", "fr", "rw"],
        "document_types": [
            "indictment", "motion", "brief", "evidence", "judgment", "order"
        ],
        "classification_levels": ["public", "confidential"],
        "digital_signature_required": False,
        "max_pages": 300
    }
    
    @classmethod
    def get_specs(cls, court: CourtType) -> Dict[str, Any]:
        """Get specifications for a specific court."""
        specs_map = {
            CourtType.ICC: cls.ICC_SPECS,
            CourtType.ICTY: cls.ICTY_SPECS,
            CourtType.ICTR: cls.ICTR_SPECS,
        }
        
        return specs_map.get(court, {})
    
    @classmethod
    def get_all_supported_courts(cls) -> List[CourtType]:
        """Get list of all supported court types."""
        return [CourtType.ICC, CourtType.ICTY, CourtType.ICTR]


class ComplianceChecker:
    """
    Checks compliance with various international standards and regulations.
    
    Implements checks for accessibility standards, international document
    formats, and court-specific compliance requirements.
    """
    
    def __init__(self):
        """Initialize compliance checker."""
        self.accessibility_standards = ["WCAG2.1", "PDF/UA", "Section508"]
        self.document_standards = ["ISO32000", "ISO14289", "ISO19005"]
    
    def check_accessibility_compliance(self, file_path: str) -> Tuple[bool, List[str], List[str]]:
        """
        Check document accessibility compliance.
        
        Args:
            file_path: Path to document file
            
        Returns:
            Tuple of (is_compliant, errors, recommendations)
        """
        errors = []
        recommendations = []
        
        file_path = Path(file_path)
        
        # Basic file checks
        if not file_path.exists():
            errors.append("File does not exist")
            return False, errors, recommendations
        
        # PDF-specific accessibility checks
        if file_path.suffix.lower() == '.pdf':
            # These would be actual PDF accessibility checks in a real implementation
            # For now, we'll do basic validation
            recommendations.append("Verify PDF has proper document structure and tags")
            recommendations.append("Ensure PDF has alternative text for images")
            recommendations.append("Check PDF for proper reading order")
            recommendations.append("Verify PDF has appropriate color contrast")
        
        # General document checks
        if file_path.stat().st_size > 100 * 1024 * 1024:  # 100MB
            recommendations.append("Large file size may impact accessibility")
        
        return len(errors) == 0, errors, recommendations
    
    def check_iso_compliance(self, file_path: str) -> Tuple[bool, List[str]]:
        """Check compliance with ISO standards."""
        errors = []
        file_path = Path(file_path)
        
        # Basic ISO compliance checks
        if file_path.suffix.lower() == '.pdf':
            # Would implement actual PDF/A validation
            pass
        
        return len(errors) == 0, errors
    
    def check_court_specific_compliance(
        self,
        submission: Dict[str, Any],
        court: CourtType
    ) -> Tuple[bool, List[str], List[str]]:
        """
        Check court-specific compliance requirements.
        
        Args:
            submission: Submission data to check
            court: Target court
            
        Returns:
            Tuple of (is_compliant, errors, warnings)
        """
        specs = CourtSpecifications.get_specs(court)
        errors = []
        warnings = []
        
        if not specs:
            warnings.append(f"No specifications available for court: {court.value}")
            return True, errors, warnings
        
        # Check required metadata
        for required_field in specs.get("required_metadata", []):
            if required_field not in submission:
                errors.append(f"Missing required metadata field: {required_field}")
        
        # Check document type
        document_type = submission.get("document_type")
        if document_type and document_type not in specs.get("document_types", []):
            errors.append(f"Invalid document type: {document_type}")
        
        # Check classification level
        classification = submission.get("classification")
        if classification and classification not in specs.get("classification_levels", []):
            errors.append(f"Invalid classification level: {classification}")
        
        # Check language
        language = submission.get("language")
        if language and language not in specs.get("supported_languages", []):
            warnings.append(f"Language may not be supported by court: {language}")
        
        # Check digital signature requirement
        if specs.get("digital_signature_required") and not submission.get("digital_signature"):
            errors.append("Digital signature required but not provided")
        
        return len(errors) == 0, errors, warnings


class XMLValidator:
    """Validates XML documents against schemas and court-specific requirements."""
    
    def __init__(self):
        """Initialize XML validator."""
        self.schemas = {}  # Cache for loaded schemas
    
    def validate_xml_structure(self, xml_content: str) -> Tuple[bool, List[str]]:
        """
        Validate basic XML structure and well-formedness.
        
        Args:
            xml_content: XML content to validate
            
        Returns:
            Tuple of (is_valid, errors)
        """
        errors = []
        
        try:
            # Parse XML to check well-formedness
            DefusedET.fromstring(xml_content.encode('utf-8'))
        except ET.ParseError as e:
            errors.append(f"XML parsing error: {str(e)}")
        except Exception as e:
            errors.append(f"XML validation error: {str(e)}")
        
        return len(errors) == 0, errors
    
    def validate_against_schema(
        self,
        xml_content: str,
        schema_path: Optional[str] = None,
        court: Optional[CourtType] = None
    ) -> Tuple[bool, List[str]]:
        """
        Validate XML against XSD schema.
        
        Args:
            xml_content: XML content to validate
            schema_path: Path to XSD schema file
            court: Court type for built-in schemas
            
        Returns:
            Tuple of (is_valid, errors)
        """
        errors = []
        
        try:
            xml_doc = etree.fromstring(xml_content.encode('utf-8'))
            
            # Load schema
            if schema_path:
                with open(schema_path, 'r', encoding='utf-8') as f:
                    schema_content = f.read()
            elif court == CourtType.ICC:
                # Use built-in ICC schema
                from .icc_formatter import ICCSubmissionSchema
                schema_content = ICCSubmissionSchema.get_xsd_schema()
            else:
                errors.append("No schema provided for validation")
                return False, errors
            
            schema_doc = etree.fromstring(schema_content.encode('utf-8'))
            schema = etree.XMLSchema(schema_doc)
            
            # Validate
            if not schema.validate(xml_doc):
                for error in schema.error_log:
                    errors.append(f"Schema validation error: {error}")
            
        except Exception as e:
            errors.append(f"Schema validation failed: {str(e)}")
        
        return len(errors) == 0, errors
    
    def validate_icc_xml(self, xml_content: str) -> ValidationResult:
        """Validate ICC-specific XML submission."""
        from .icc_formatter import ICCSubmissionSchema
        return ICCSubmissionSchema.validate_xml(xml_content)


class JSONValidator:
    """Validates JSON documents and structured data."""
    
    def __init__(self):
        """Initialize JSON validator."""
        self.schemas = {}
    
    def validate_json_structure(self, json_content: str) -> Tuple[bool, List[str]]:
        """Validate JSON structure and syntax."""
        errors = []
        
        try:
            json.loads(json_content)
        except json.JSONDecodeError as e:
            errors.append(f"JSON parsing error: {str(e)}")
        except Exception as e:
            errors.append(f"JSON validation error: {str(e)}")
        
        return len(errors) == 0, errors
    
    def validate_against_schema(
        self,
        json_content: str,
        schema: Dict[str, Any]
    ) -> Tuple[bool, List[str]]:
        """Validate JSON against JSON Schema."""
        errors = []
        
        try:
            data = json.loads(json_content)
            json_validate(instance=data, schema=schema)
        except JsonValidationError as e:
            errors.append(f"JSON schema validation error: {str(e)}")
        except Exception as e:
            errors.append(f"JSON validation failed: {str(e)}")
        
        return len(errors) == 0, errors


class FormatValidator:
    """
    Main format validation coordinator.
    
    Provides comprehensive validation of legal submission formats
    including structure, content, metadata, and compliance checking.
    """
    
    def __init__(self, strict_mode: bool = True):
        """
        Initialize format validator.
        
        Args:
            strict_mode: Whether to enforce strict validation
        """
        self.strict_mode = strict_mode
        self.compliance_checker = ComplianceChecker()
        self.xml_validator = XMLValidator()
        self.json_validator = JSONValidator()
        
        # Initialize validation rules
        self.validation_rules = self._init_validation_rules()
    
    def validate_for_court(
        self,
        submission: Union[Dict[str, Any], ICCSubmission, str],
        court: CourtType,
        format_type: Optional[str] = None
    ) -> ValidationResult:
        """
        Validate submission format for specific court.
        
        Args:
            submission: Submission data to validate
            court: Target court type
            format_type: Format type (xml, json, etc.)
            
        Returns:
            Validation result with details
        """
        try:
            logger.info(f"Validating submission for {court.value}")
            
            errors = []
            warnings = []
            compliance_checks = {}
            recommendations = []
            
            # Determine format type if not provided
            if format_type is None:
                if isinstance(submission, str):
                    if submission.strip().startswith('<?xml') or submission.strip().startswith('<'):
                        format_type = "xml"
                    elif submission.strip().startswith('{') or submission.strip().startswith('['):
                        format_type = "json"
                elif isinstance(submission, ICCSubmission):
                    format_type = "icc_submission"
                else:
                    format_type = "structured_data"
            
            # Format-specific validation
            if format_type == "xml":
                is_valid, xml_errors = self.xml_validator.validate_xml_structure(submission)
                errors.extend(xml_errors)
                
                if court == CourtType.ICC and is_valid:
                    icc_result = self.xml_validator.validate_icc_xml(submission)
                    errors.extend(icc_result.errors)
                    warnings.extend(icc_result.warnings)
            
            elif format_type == "json":
                is_valid, json_errors = self.json_validator.validate_json_structure(submission)
                errors.extend(json_errors)
            
            elif format_type == "icc_submission":
                # Validate ICC submission object
                icc_result = self.xml_validator.validate_icc_xml(submission.xml_content)
                errors.extend(icc_result.errors)
                warnings.extend(icc_result.warnings)
                submission_dict = submission.dict()
            else:
                submission_dict = submission if isinstance(submission, dict) else {}
            
            # Court-specific compliance checks
            if isinstance(submission, dict):
                submission_dict = submission
            elif hasattr(submission, 'dict'):
                submission_dict = submission.dict()
            else:
                submission_dict = {}
            
            compliance_ok, comp_errors, comp_warnings = self.compliance_checker.check_court_specific_compliance(
                submission_dict, court
            )
            compliance_checks["court_specific"] = compliance_ok
            errors.extend(comp_errors)
            warnings.extend(comp_warnings)
            
            # Run validation rules
            for rule in self.validation_rules:
                if court in rule.court_types or not rule.court_types:
                    try:
                        rule_passed, rule_message = rule.check_function(submission_dict)
                        compliance_checks[rule.rule_id] = rule_passed
                        
                        if not rule_passed:
                            if rule.severity in [ValidationSeverity.ERROR, ValidationSeverity.CRITICAL]:
                                errors.append(f"{rule.rule_id}: {rule_message}")
                            else:
                                warnings.append(f"{rule.rule_id}: {rule_message}")
                    except Exception as e:
                        warnings.append(f"Rule {rule.rule_id} check failed: {str(e)}")
            
            # Generate recommendations
            if warnings:
                recommendations.append("Review and address validation warnings")
            if not compliance_checks.get("court_specific", True):
                recommendations.append("Ensure all court-specific requirements are met")
            
            # Determine overall validity
            is_overall_valid = len(errors) == 0
            if self.strict_mode:
                is_overall_valid = is_overall_valid and len([w for w in warnings if "required" in w.lower()]) == 0
            
            # Get court specs for format version
            specs = CourtSpecifications.get_specs(court)
            format_version = specs.get("schema_version", "1.0")
            
            result = ValidationResult(
                is_valid=is_overall_valid,
                court=court,
                format_version=format_version,
                errors=errors,
                warnings=warnings,
                compliance_checks=compliance_checks,
                recommendations=recommendations,
                validator_version="1.0"
            )
            
            logger.info(f"Validation completed for {court.value}: {'PASS' if is_overall_valid else 'FAIL'}")
            return result
            
        except Exception as e:
            logger.error(f"Validation failed for {court.value}: {e}")
            return ValidationResult(
                is_valid=False,
                court=court,
                format_version="1.0",
                errors=[f"Validation process failed: {str(e)}"],
                warnings=[],
                compliance_checks={},
                recommendations=["Fix validation process errors"],
                validator_version="1.0"
            )
    
    def validate_court_package(self, package: CourtPackage) -> List[ValidationResult]:
        """
        Validate a complete court package.
        
        Args:
            package: Court package to validate
            
        Returns:
            List of validation results for different aspects
        """
        results = []
        
        # Validate main submission format
        main_result = self.validate_for_court(
            package.submission_format,
            package.court
        )
        results.append(main_result)
        
        # Validate evidence package
        evidence_validation = self._validate_evidence_package(package.evidence_package)
        results.append(evidence_validation)
        
        # Validate case data
        case_validation = self._validate_case_data(package.case_data, package.court)
        results.append(case_validation)
        
        return results
    
    def batch_validate(
        self,
        submissions: List[Tuple[Any, CourtType]],
        parallel: bool = False
    ) -> List[ValidationResult]:
        """
        Validate multiple submissions.
        
        Args:
            submissions: List of (submission, court) tuples
            parallel: Whether to validate in parallel
            
        Returns:
            List of validation results
        """
        results = []
        
        for submission, court in submissions:
            try:
                result = self.validate_for_court(submission, court)
                results.append(result)
            except Exception as e:
                error_result = ValidationResult(
                    is_valid=False,
                    court=court,
                    format_version="1.0",
                    errors=[f"Validation failed: {str(e)}"],
                    warnings=[],
                    compliance_checks={},
                    recommendations=[],
                    validator_version="1.0"
                )
                results.append(error_result)
        
        return results
    
    def _validate_evidence_package(self, evidence_package) -> ValidationResult:
        """Validate evidence package structure and content."""
        errors = []
        warnings = []
        compliance_checks = {}
        
        # Check manifest
        if not evidence_package.manifest:
            errors.append("Evidence package missing manifest")
        else:
            if not evidence_package.manifest.package_name:
                warnings.append("Evidence package missing name")
            
            if evidence_package.manifest.evidence_count != len(evidence_package.evidence):
                errors.append("Evidence count mismatch between manifest and actual evidence")
        
        # Check evidence items
        for evidence in evidence_package.evidence:
            if not evidence.evidence_id:
                errors.append("Evidence item missing ID")
            if not evidence.title:
                errors.append(f"Evidence {evidence.evidence_id} missing title")
            if not evidence.chain_of_custody:
                warnings.append(f"Evidence {evidence.evidence_id} missing chain of custody")
        
        compliance_checks["manifest_complete"] = len(errors) == 0
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            court=evidence_package.court,
            format_version="1.0",
            errors=errors,
            warnings=warnings,
            compliance_checks=compliance_checks,
            recommendations=["Ensure all evidence items have complete metadata"],
            validator_version="1.0"
        )
    
    def _validate_case_data(self, case_data, court: CourtType) -> ValidationResult:
        """Validate case data structure."""
        errors = []
        warnings = []
        compliance_checks = {}
        
        # Basic case data validation
        if not case_data.case_id:
            errors.append("Case missing ID")
        if not case_data.case_name:
            errors.append("Case missing name")
        if case_data.court != court:
            warnings.append("Case court type doesn't match submission court")
        
        # Metadata validation
        if not case_data.metadata:
            errors.append("Case missing metadata")
        elif not case_data.metadata.title:
            errors.append("Case metadata missing title")
        
        compliance_checks["case_data_complete"] = len(errors) == 0
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            court=court,
            format_version="1.0",
            errors=errors,
            warnings=warnings,
            compliance_checks=compliance_checks,
            recommendations=["Ensure case data is complete and consistent"],
            validator_version="1.0"
        )
    
    def _init_validation_rules(self) -> List[ValidationRule]:
        """Initialize validation rules."""
        rules = []
        
        # General structure rules
        rules.append(ValidationRule(
            rule_id="STRUCT_001",
            description="Submission must have required metadata",
            category=ValidationCategory.STRUCTURE,
            severity=ValidationSeverity.ERROR,
            court_types=[],  # Apply to all courts
            check_function=lambda x: (bool(x.get("title") and x.get("date")), "Missing title or date")
        ))
        
        # ICC-specific rules
        rules.append(ValidationRule(
            rule_id="ICC_001",
            description="ICC submissions must have case number",
            category=ValidationCategory.CONTENT,
            severity=ValidationSeverity.WARNING,
            court_types=[CourtType.ICC],
            check_function=lambda x: (bool(x.get("icc_case_number")), "Missing ICC case number")
        ))
        
        # Content rules
        rules.append(ValidationRule(
            rule_id="CONTENT_001",
            description="Document must not exceed maximum page limit",
            category=ValidationCategory.CONTENT,
            severity=ValidationSeverity.ERROR,
            court_types=[CourtType.ICC],
            check_function=lambda x: (x.get("pages", 0) <= 500, "Document exceeds maximum page limit")
        ))
        
        return rules


def validate_submission_format(submission: Any, court: str) -> ValidationResult:
    """
    Convenience function to validate submission format.
    
    Args:
        submission: Submission data to validate
        court: Court identifier string
        
    Returns:
        Validation result
    """
    try:
        court_type = CourtType(court.lower())
    except ValueError:
        return ValidationResult(
            is_valid=False,
            court=CourtType.ICC,  # Default
            format_version="1.0",
            errors=[f"Unsupported court type: {court}"],
            warnings=[],
            compliance_checks={},
            recommendations=[],
            validator_version="1.0"
        )
    
    validator = FormatValidator(strict_mode=True)
    return validator.validate_for_court(submission, court_type)