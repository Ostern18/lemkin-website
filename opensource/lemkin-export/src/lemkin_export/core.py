"""
Lemkin Multi-format Export Core

Comprehensive export system for legal investigations supporting multiple formats,
platforms, and compliance standards with automated data validation and packaging.
"""

from datetime import datetime, date
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, IO
import json
import zipfile
import tarfile
import shutil
import hashlib
import uuid
from dataclasses import dataclass, asdict

from pydantic import BaseModel, Field, validator
from loguru import logger
import pandas as pd
import xlsxwriter
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment
import xml.etree.ElementTree as ET
from xml.dom import minidom
import csv
import yaml


class ExportFormat(str, Enum):
    """Supported export formats."""
    # Document formats
    PDF = "pdf"
    HTML = "html"
    DOCX = "docx"
    MARKDOWN = "markdown"
    TXT = "txt"

    # Data formats
    CSV = "csv"
    XLSX = "xlsx"
    JSON = "json"
    XML = "xml"
    YAML = "yaml"
    PARQUET = "parquet"

    # Archive formats
    ZIP = "zip"
    TAR_GZ = "tar.gz"
    TAR_XZ = "tar.xz"

    # Legal formats
    EDRM_XML = "edrm_xml"
    CONCORDANCE = "concordance"
    RELATIVITY = "relativity"
    LEGAL_XML = "legal_xml"

    # Platform formats
    SALESFORCE = "salesforce"
    SHAREPOINT = "sharepoint"
    TEAMS = "teams"


class CompressionType(str, Enum):
    """Compression algorithms."""
    NONE = "none"
    GZIP = "gzip"
    BZIP2 = "bzip2"
    XZ = "xz"
    ZIP = "zip"


class ExportScope(str, Enum):
    """Export data scope."""
    CASE_COMPLETE = "case_complete"
    EVIDENCE_ONLY = "evidence_only"
    ANALYSIS_ONLY = "analysis_only"
    REPORTS_ONLY = "reports_only"
    METADATA_ONLY = "metadata_only"
    CUSTOM = "custom"


class ValidationLevel(str, Enum):
    """Export validation levels."""
    NONE = "none"
    BASIC = "basic"
    STANDARD = "standard"
    STRICT = "strict"
    FORENSIC = "forensic"


class ExportItem(BaseModel):
    """Individual item to be exported."""
    item_id: str
    item_type: str
    source_path: Optional[Path] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    content: Optional[Any] = None
    file_hash: Optional[str] = None
    file_size: Optional[int] = None
    created_at: datetime = Field(default_factory=datetime.now)
    last_modified: Optional[datetime] = None
    permissions: List[str] = Field(default_factory=list)
    tags: List[str] = Field(default_factory=list)


class ExportManifest(BaseModel):
    """Export package manifest."""
    manifest_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    case_id: str
    export_type: ExportScope
    export_format: ExportFormat
    created_at: datetime = Field(default_factory=datetime.now)
    created_by: str
    total_items: int
    total_size: int
    compression: CompressionType = CompressionType.NONE
    validation_level: ValidationLevel
    checksum: Optional[str] = None
    items: List[ExportItem] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    compliance_info: Dict[str, Any] = Field(default_factory=dict)


class ExportPackage(BaseModel):
    """Complete export package."""
    package_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    manifest: ExportManifest
    output_path: Path
    package_size: int
    encrypted: bool = False
    password_protected: bool = False
    digital_signature: Optional[str] = None
    export_log: List[str] = Field(default_factory=list)


class DataValidator:
    """Validates export data integrity and compliance."""

    def __init__(self, validation_level: ValidationLevel = ValidationLevel.STANDARD):
        self.validation_level = validation_level
        self.validation_errors = []
        self.validation_warnings = []

    def validate_export_items(self, items: List[ExportItem]) -> Dict[str, Any]:
        """Validate export items for integrity and compliance."""
        self.validation_errors.clear()
        self.validation_warnings.clear()

        for item in items:
            self._validate_item(item)

        return {
            "valid": len(self.validation_errors) == 0,
            "errors": self.validation_errors,
            "warnings": self.validation_warnings,
            "items_validated": len(items)
        }

    def _validate_item(self, item: ExportItem) -> None:
        """Validate individual export item."""
        # Basic validation
        if not item.item_id:
            self.validation_errors.append(f"Item missing ID: {item}")

        # File validation
        if item.source_path:
            if not item.source_path.exists():
                self.validation_errors.append(f"Source file not found: {item.source_path}")
            else:
                # Validate file hash if provided
                if item.file_hash and self.validation_level in [ValidationLevel.STRICT, ValidationLevel.FORENSIC]:
                    calculated_hash = self._calculate_file_hash(item.source_path)
                    if calculated_hash != item.file_hash:
                        self.validation_errors.append(f"File hash mismatch for {item.source_path}")

        # Metadata validation
        if self.validation_level in [ValidationLevel.STANDARD, ValidationLevel.STRICT, ValidationLevel.FORENSIC]:
            required_metadata = ["created_date", "file_type"]
            for field in required_metadata:
                if field not in item.metadata:
                    self.validation_warnings.append(f"Missing metadata field '{field}' for item {item.item_id}")

        # Forensic validation
        if self.validation_level == ValidationLevel.FORENSIC:
            if not item.file_hash:
                self.validation_errors.append(f"File hash required for forensic validation: {item.item_id}")
            if not item.metadata.get("chain_of_custody"):
                self.validation_warnings.append(f"Chain of custody not documented for {item.item_id}")

    def _calculate_file_hash(self, file_path: Path, algorithm: str = "sha256") -> str:
        """Calculate file hash for integrity verification."""
        hash_algo = hashlib.new(algorithm)

        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_algo.update(chunk)

        return hash_algo.hexdigest()

    def validate_compliance(self, manifest: ExportManifest,
                          compliance_standards: List[str]) -> Dict[str, Any]:
        """Validate export against compliance standards."""
        compliance_results = {}

        for standard in compliance_standards:
            if standard == "GDPR":
                compliance_results[standard] = self._validate_gdpr_compliance(manifest)
            elif standard == "HIPAA":
                compliance_results[standard] = self._validate_hipaa_compliance(manifest)
            elif standard == "SOX":
                compliance_results[standard] = self._validate_sox_compliance(manifest)
            elif standard == "ISO27001":
                compliance_results[standard] = self._validate_iso27001_compliance(manifest)

        return compliance_results

    def _validate_gdpr_compliance(self, manifest: ExportManifest) -> Dict[str, Any]:
        """Validate GDPR compliance requirements."""
        warnings = []
        errors = []

        # Check for PII handling documentation
        has_pii_docs = any("pii_handling" in item.metadata for item in manifest.items)
        if not has_pii_docs:
            warnings.append("No PII handling documentation found")

        # Check for consent records
        has_consent = any("consent_record" in item.metadata for item in manifest.items)
        if not has_consent:
            warnings.append("No consent records found for personal data")

        return {"valid": len(errors) == 0, "warnings": warnings, "errors": errors}

    def _validate_hipaa_compliance(self, manifest: ExportManifest) -> Dict[str, Any]:
        """Validate HIPAA compliance requirements."""
        warnings = []
        errors = []

        # Check for PHI protection measures
        has_phi_protection = manifest.encrypted or manifest.password_protected
        if not has_phi_protection:
            errors.append("PHI data must be encrypted for HIPAA compliance")

        return {"valid": len(errors) == 0, "warnings": warnings, "errors": errors}

    def _validate_sox_compliance(self, manifest: ExportManifest) -> Dict[str, Any]:
        """Validate SOX compliance requirements."""
        warnings = []
        errors = []

        # Check for audit trail
        has_audit_trail = "audit_trail" in manifest.metadata
        if not has_audit_trail:
            warnings.append("Audit trail documentation recommended for SOX compliance")

        return {"valid": len(errors) == 0, "warnings": warnings, "errors": errors}

    def _validate_iso27001_compliance(self, manifest: ExportManifest) -> Dict[str, Any]:
        """Validate ISO 27001 compliance requirements."""
        warnings = []
        errors = []

        # Check for information classification
        classified_items = [item for item in manifest.items if "classification" in item.metadata]
        if len(classified_items) != len(manifest.items):
            warnings.append("Not all items have information classification")

        return {"valid": len(errors) == 0, "warnings": warnings, "errors": errors}


class FormatConverter:
    """Converts data between different export formats."""

    def __init__(self):
        self.conversion_map = {
            ExportFormat.CSV: self._to_csv,
            ExportFormat.XLSX: self._to_xlsx,
            ExportFormat.JSON: self._to_json,
            ExportFormat.XML: self._to_xml,
            ExportFormat.YAML: self._to_yaml,
            ExportFormat.HTML: self._to_html,
            ExportFormat.MARKDOWN: self._to_markdown,
            ExportFormat.PARQUET: self._to_parquet,
            ExportFormat.EDRM_XML: self._to_edrm_xml,
            ExportFormat.CONCORDANCE: self._to_concordance,
            ExportFormat.LEGAL_XML: self._to_legal_xml,
        }

    def convert_data(self, data: Any, target_format: ExportFormat,
                    output_path: Path, **kwargs) -> Path:
        """Convert data to target format."""
        if target_format not in self.conversion_map:
            raise ValueError(f"Unsupported format: {target_format}")

        converter = self.conversion_map[target_format]
        return converter(data, output_path, **kwargs)

    def _to_csv(self, data: Any, output_path: Path, **kwargs) -> Path:
        """Convert to CSV format."""
        if isinstance(data, pd.DataFrame):
            data.to_csv(output_path, index=False, **kwargs)
        elif isinstance(data, list) and data and isinstance(data[0], dict):
            df = pd.DataFrame(data)
            df.to_csv(output_path, index=False, **kwargs)
        else:
            # Simple CSV writer
            with open(output_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                if isinstance(data, list):
                    for row in data:
                        writer.writerow(row if isinstance(row, (list, tuple)) else [row])
                else:
                    writer.writerow([data])

        return output_path

    def _to_xlsx(self, data: Any, output_path: Path, **kwargs) -> Path:
        """Convert to Excel format."""
        workbook = xlsxwriter.Workbook(str(output_path))
        worksheet = workbook.add_worksheet()

        # Header format
        header_format = workbook.add_format({
            'bold': True,
            'bg_color': '#D7E4BC',
            'border': 1
        })

        if isinstance(data, pd.DataFrame):
            # Write headers
            for col, header in enumerate(data.columns):
                worksheet.write(0, col, header, header_format)

            # Write data
            for row, (_, record) in enumerate(data.iterrows(), 1):
                for col, value in enumerate(record):
                    worksheet.write(row, col, value)

        elif isinstance(data, list) and data and isinstance(data[0], dict):
            # Write headers
            headers = list(data[0].keys())
            for col, header in enumerate(headers):
                worksheet.write(0, col, header, header_format)

            # Write data
            for row, record in enumerate(data, 1):
                for col, header in enumerate(headers):
                    worksheet.write(row, col, record.get(header, ''))

        workbook.close()
        return output_path

    def _to_json(self, data: Any, output_path: Path, **kwargs) -> Path:
        """Convert to JSON format."""
        with open(output_path, 'w', encoding='utf-8') as f:
            if hasattr(data, 'to_dict'):
                json.dump(data.to_dict(), f, indent=2, default=str, **kwargs)
            elif hasattr(data, '__dict__'):
                json.dump(data.__dict__, f, indent=2, default=str, **kwargs)
            else:
                json.dump(data, f, indent=2, default=str, **kwargs)

        return output_path

    def _to_xml(self, data: Any, output_path: Path, **kwargs) -> Path:
        """Convert to XML format."""
        root = ET.Element("export_data")

        if isinstance(data, list):
            for i, item in enumerate(data):
                item_element = ET.SubElement(root, "item", id=str(i))
                self._dict_to_xml(item, item_element)
        elif isinstance(data, dict):
            self._dict_to_xml(data, root)
        else:
            root.text = str(data)

        # Pretty print XML
        rough_string = ET.tostring(root, 'unicode')
        reparsed = minidom.parseString(rough_string)
        pretty_xml = reparsed.toprettyxml(indent="  ")

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(pretty_xml)

        return output_path

    def _dict_to_xml(self, data: dict, parent: ET.Element) -> None:
        """Convert dictionary to XML elements."""
        for key, value in data.items():
            if isinstance(value, dict):
                child = ET.SubElement(parent, str(key))
                self._dict_to_xml(value, child)
            elif isinstance(value, list):
                for item in value:
                    child = ET.SubElement(parent, str(key))
                    if isinstance(item, dict):
                        self._dict_to_xml(item, child)
                    else:
                        child.text = str(item)
            else:
                child = ET.SubElement(parent, str(key))
                child.text = str(value)

    def _to_yaml(self, data: Any, output_path: Path, **kwargs) -> Path:
        """Convert to YAML format."""
        with open(output_path, 'w', encoding='utf-8') as f:
            if hasattr(data, 'to_dict'):
                yaml.dump(data.to_dict(), f, default_flow_style=False, **kwargs)
            elif hasattr(data, '__dict__'):
                yaml.dump(data.__dict__, f, default_flow_style=False, **kwargs)
            else:
                yaml.dump(data, f, default_flow_style=False, **kwargs)

        return output_path

    def _to_html(self, data: Any, output_path: Path, **kwargs) -> Path:
        """Convert to HTML format."""
        html_content = """
<!DOCTYPE html>
<html>
<head>
    <title>Export Data</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; }
        table { border-collapse: collapse; width: 100%; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #f2f2f2; font-weight: bold; }
        .metadata { background-color: #f9f9f9; padding: 15px; margin: 20px 0; }
    </style>
</head>
<body>
        """

        if isinstance(data, pd.DataFrame):
            html_content += data.to_html(escape=False, table_id="export-data")
        elif isinstance(data, list) and data and isinstance(data[0], dict):
            df = pd.DataFrame(data)
            html_content += df.to_html(escape=False, table_id="export-data")
        else:
            html_content += f"<div class='metadata'><pre>{json.dumps(data, indent=2, default=str)}</pre></div>"

        html_content += "</body></html>"

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

        return output_path

    def _to_markdown(self, data: Any, output_path: Path, **kwargs) -> Path:
        """Convert to Markdown format."""
        markdown_content = ["# Export Data\n"]

        if isinstance(data, pd.DataFrame):
            markdown_content.append(data.to_markdown())
        elif isinstance(data, list) and data and isinstance(data[0], dict):
            df = pd.DataFrame(data)
            markdown_content.append(df.to_markdown())
        else:
            markdown_content.append("```json")
            markdown_content.append(json.dumps(data, indent=2, default=str))
            markdown_content.append("```")

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(markdown_content))

        return output_path

    def _to_parquet(self, data: Any, output_path: Path, **kwargs) -> Path:
        """Convert to Parquet format."""
        if isinstance(data, pd.DataFrame):
            data.to_parquet(output_path, **kwargs)
        elif isinstance(data, list) and data and isinstance(data[0], dict):
            df = pd.DataFrame(data)
            df.to_parquet(output_path, **kwargs)
        else:
            raise ValueError("Parquet format requires tabular data")

        return output_path

    def _to_edrm_xml(self, data: Any, output_path: Path, **kwargs) -> Path:
        """Convert to EDRM XML format."""
        root = ET.Element("Root")
        root.set("xmlns", "http://www.edrm.net/schemas/edrm")

        batch = ET.SubElement(root, "Batch")
        batch.set("batchId", str(kwargs.get("batch_id", "1")))

        if isinstance(data, list):
            for item in data:
                document = ET.SubElement(batch, "Document")
                if isinstance(item, dict):
                    self._add_edrm_fields(document, item)

        # Pretty print XML
        rough_string = ET.tostring(root, 'unicode')
        reparsed = minidom.parseString(rough_string)
        pretty_xml = reparsed.toprettyxml(indent="  ")

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(pretty_xml)

        return output_path

    def _add_edrm_fields(self, parent: ET.Element, data: dict) -> None:
        """Add EDRM-specific fields to document."""
        # Map common fields to EDRM format
        field_mapping = {
            "document_id": "DocumentId",
            "title": "Title",
            "author": "Author",
            "created_date": "CreatedDate",
            "file_path": "FilePath",
            "file_size": "FileSize",
            "hash": "Hash"
        }

        for key, value in data.items():
            edrm_field = field_mapping.get(key, key)
            field_element = ET.SubElement(parent, "Field")
            field_element.set("name", edrm_field)
            field_element.text = str(value)

    def _to_concordance(self, data: Any, output_path: Path, **kwargs) -> Path:
        """Convert to Concordance format."""
        # Concordance uses pipe-delimited format
        if isinstance(data, pd.DataFrame):
            data.to_csv(output_path, sep='þ', index=False, encoding='utf-8')
        elif isinstance(data, list) and data and isinstance(data[0], dict):
            df = pd.DataFrame(data)
            df.to_csv(output_path, sep='þ', index=False, encoding='utf-8')
        else:
            raise ValueError("Concordance format requires tabular data")

        return output_path

    def _to_legal_xml(self, data: Any, output_path: Path, **kwargs) -> Path:
        """Convert to Legal XML format."""
        root = ET.Element("LegalExport")
        root.set("version", "1.0")
        root.set("created", datetime.now().isoformat())

        case_info = ET.SubElement(root, "CaseInformation")
        case_info.set("caseId", str(kwargs.get("case_id", "unknown")))

        documents = ET.SubElement(root, "Documents")

        if isinstance(data, list):
            for item in data:
                doc_element = ET.SubElement(documents, "Document")
                if isinstance(item, dict):
                    self._dict_to_xml(item, doc_element)

        # Pretty print XML
        rough_string = ET.tostring(root, 'unicode')
        reparsed = minidom.parseString(rough_string)
        pretty_xml = reparsed.toprettyxml(indent="  ")

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(pretty_xml)

        return output_path


class CompressionManager:
    """Handles compression and archiving of export packages."""

    def __init__(self):
        self.compression_handlers = {
            CompressionType.ZIP: self._create_zip,
            CompressionType.GZIP: self._create_tar_gz,
            CompressionType.XZ: self._create_tar_xz,
        }

    def compress_package(self, source_path: Path, output_path: Path,
                        compression_type: CompressionType) -> Path:
        """Compress export package."""
        if compression_type == CompressionType.NONE:
            return source_path

        if compression_type not in self.compression_handlers:
            raise ValueError(f"Unsupported compression type: {compression_type}")

        handler = self.compression_handlers[compression_type]
        return handler(source_path, output_path)

    def _create_zip(self, source_path: Path, output_path: Path) -> Path:
        """Create ZIP archive."""
        with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            if source_path.is_file():
                zipf.write(source_path, source_path.name)
            else:
                for file_path in source_path.rglob('*'):
                    if file_path.is_file():
                        arcname = file_path.relative_to(source_path)
                        zipf.write(file_path, arcname)

        return output_path

    def _create_tar_gz(self, source_path: Path, output_path: Path) -> Path:
        """Create TAR.GZ archive."""
        with tarfile.open(output_path, 'w:gz') as tar:
            tar.add(source_path, arcname=source_path.name)

        return output_path

    def _create_tar_xz(self, source_path: Path, output_path: Path) -> Path:
        """Create TAR.XZ archive."""
        with tarfile.open(output_path, 'w:xz') as tar:
            tar.add(source_path, arcname=source_path.name)

        return output_path

    def calculate_compression_ratio(self, original_size: int, compressed_size: int) -> float:
        """Calculate compression ratio."""
        if original_size == 0:
            return 0.0
        return (1 - compressed_size / original_size) * 100


class ExportEngine:
    """Main export processing engine."""

    def __init__(self, output_base_path: Optional[Path] = None):
        self.output_base_path = output_base_path or Path("exports")
        self.output_base_path.mkdir(exist_ok=True)

        self.validator = DataValidator()
        self.converter = FormatConverter()
        self.compressor = CompressionManager()

    def create_export_package(self, case_id: str, items: List[ExportItem],
                             export_format: ExportFormat,
                             export_scope: ExportScope,
                             created_by: str,
                             validation_level: ValidationLevel = ValidationLevel.STANDARD,
                             compression: CompressionType = CompressionType.NONE,
                             **kwargs) -> ExportPackage:
        """Create complete export package."""
        # Validate items
        self.validator.validation_level = validation_level
        validation_result = self.validator.validate_export_items(items)

        if not validation_result["valid"]:
            raise ValueError(f"Export validation failed: {validation_result['errors']}")

        # Create manifest
        total_size = sum(item.file_size or 0 for item in items)
        manifest = ExportManifest(
            case_id=case_id,
            export_type=export_scope,
            export_format=export_format,
            created_by=created_by,
            total_items=len(items),
            total_size=total_size,
            compression=compression,
            validation_level=validation_level,
            items=items
        )

        # Create export directory
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        export_dir = self.output_base_path / f"{case_id}_{export_scope}_{timestamp}"
        export_dir.mkdir(parents=True, exist_ok=True)

        export_log = []

        try:
            # Export items
            exported_files = []
            for item in items:
                exported_file = self._export_item(item, export_dir, export_format, **kwargs)
                if exported_file:
                    exported_files.append(exported_file)
                    export_log.append(f"Exported: {item.item_id} -> {exported_file.name}")

            # Create manifest file
            manifest_file = export_dir / "manifest.json"
            with open(manifest_file, 'w') as f:
                json.dump(manifest.dict(), f, indent=2, default=str)

            # Create readme file
            readme_file = export_dir / "README.txt"
            self._create_readme(readme_file, manifest, validation_result)

            # Calculate package checksum
            package_checksum = self._calculate_directory_checksum(export_dir)
            manifest.checksum = package_checksum

            # Update manifest with checksum
            with open(manifest_file, 'w') as f:
                json.dump(manifest.dict(), f, indent=2, default=str)

            # Apply compression if specified
            final_output_path = export_dir
            if compression != CompressionType.NONE:
                compressed_file = export_dir.parent / f"{export_dir.name}.{compression}"
                final_output_path = self.compressor.compress_package(
                    export_dir, compressed_file, compression
                )
                export_log.append(f"Compressed package: {compressed_file}")

                # Clean up uncompressed directory
                shutil.rmtree(export_dir)

            # Create export package
            package = ExportPackage(
                manifest=manifest,
                output_path=final_output_path,
                package_size=self._calculate_path_size(final_output_path),
                export_log=export_log
            )

            logger.info(f"Created export package: {final_output_path}")
            return package

        except Exception as e:
            # Clean up on failure
            if export_dir.exists():
                shutil.rmtree(export_dir)
            logger.error(f"Export failed: {e}")
            raise

    def _export_item(self, item: ExportItem, export_dir: Path,
                    export_format: ExportFormat, **kwargs) -> Optional[Path]:
        """Export individual item."""
        if item.source_path and item.source_path.exists():
            # Copy file to export directory
            output_file = export_dir / item.source_path.name
            shutil.copy2(item.source_path, output_file)
            return output_file

        elif item.content:
            # Convert content to specified format
            output_file = export_dir / f"{item.item_id}.{export_format}"
            return self.converter.convert_data(item.content, export_format, output_file, **kwargs)

        return None

    def _create_readme(self, readme_file: Path, manifest: ExportManifest,
                      validation_result: Dict[str, Any]) -> None:
        """Create README file for export package."""
        content = f"""
LEMKIN EXPORT PACKAGE README
=============================

Package Information:
- Case ID: {manifest.case_id}
- Export Type: {manifest.export_type}
- Export Format: {manifest.export_format}
- Created: {manifest.created_at.strftime('%Y-%m-%d %H:%M:%S')}
- Created By: {manifest.created_by}
- Total Items: {manifest.total_items}
- Total Size: {manifest.total_size:,} bytes
- Validation Level: {manifest.validation_level}
- Compression: {manifest.compression}

Package Integrity:
- Checksum (SHA-256): {manifest.checksum}
- Validation Status: {'PASSED' if validation_result['valid'] else 'FAILED'}
- Items Validated: {validation_result['items_validated']}

Files in Package:
{chr(10).join([f"- {item.item_id} ({item.item_type})" for item in manifest.items])}

Validation Warnings:
{chr(10).join([f"- {warning}" for warning in validation_result.get('warnings', [])]) or 'None'}

Validation Errors:
{chr(10).join([f"- {error}" for error in validation_result.get('errors', [])]) or 'None'}

Usage Instructions:
1. Verify package integrity using the provided checksum
2. Extract compressed packages to access individual files
3. Refer to manifest.json for detailed metadata about each item
4. Contact the investigation team for questions about specific items

CONFIDENTIAL - This package contains sensitive investigation data.
Handle according to your organization's security policies.
        """.strip()

        readme_file.write_text(content)

    def _calculate_directory_checksum(self, directory: Path) -> str:
        """Calculate checksum for entire directory."""
        hasher = hashlib.sha256()

        for file_path in sorted(directory.rglob('*')):
            if file_path.is_file():
                with open(file_path, 'rb') as f:
                    for chunk in iter(lambda: f.read(4096), b""):
                        hasher.update(chunk)

        return hasher.hexdigest()

    def _calculate_path_size(self, path: Path) -> int:
        """Calculate total size of path (file or directory)."""
        if path.is_file():
            return path.stat().st_size
        else:
            return sum(f.stat().st_size for f in path.rglob('*') if f.is_file())

    def export_to_platform(self, package: ExportPackage,
                          platform_config: Dict[str, Any]) -> Dict[str, Any]:
        """Export package to external platform."""
        platform = platform_config.get("platform")

        if platform == "sharepoint":
            return self._export_to_sharepoint(package, platform_config)
        elif platform == "salesforce":
            return self._export_to_salesforce(package, platform_config)
        elif platform == "teams":
            return self._export_to_teams(package, platform_config)
        else:
            raise ValueError(f"Unsupported platform: {platform}")

    def _export_to_sharepoint(self, package: ExportPackage,
                             config: Dict[str, Any]) -> Dict[str, Any]:
        """Export to SharePoint (placeholder implementation)."""
        # In a real implementation, this would use SharePoint APIs
        return {
            "status": "success",
            "platform": "sharepoint",
            "location": config.get("site_url", ""),
            "package_id": package.package_id,
            "exported_at": datetime.now().isoformat()
        }

    def _export_to_salesforce(self, package: ExportPackage,
                             config: Dict[str, Any]) -> Dict[str, Any]:
        """Export to Salesforce (placeholder implementation)."""
        # In a real implementation, this would use Salesforce APIs
        return {
            "status": "success",
            "platform": "salesforce",
            "org_id": config.get("org_id", ""),
            "package_id": package.package_id,
            "exported_at": datetime.now().isoformat()
        }

    def _export_to_teams(self, package: ExportPackage,
                        config: Dict[str, Any]) -> Dict[str, Any]:
        """Export to Microsoft Teams (placeholder implementation)."""
        # In a real implementation, this would use Teams Graph APIs
        return {
            "status": "success",
            "platform": "teams",
            "team_id": config.get("team_id", ""),
            "package_id": package.package_id,
            "exported_at": datetime.now().isoformat()
        }


# Export all classes and functions
__all__ = [
    "ExportFormat",
    "CompressionType",
    "ExportScope",
    "ValidationLevel",
    "ExportItem",
    "ExportManifest",
    "ExportPackage",
    "DataValidator",
    "FormatConverter",
    "CompressionManager",
    "ExportEngine"
]