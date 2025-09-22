"""
Court-ready evidence package creation and validation module.

This module provides functionality to create comprehensive evidence packages
that meet international court requirements, including digital signatures,
chain of custody preservation, and integrity verification.
"""

import hashlib
import json
import shutil
import tarfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple, BinaryIO

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.serialization import pkcs12
import pandas as pd
from loguru import logger
from tqdm import tqdm

from .core import (
    CaseData, Evidence, EvidencePackage, CourtPackage, PackageManifest,
    DigitalSignature, ChainOfCustody, CourtType, ExportError, ValidationError,
    FormatError, AuditTrail
)


class EvidenceValidator:
    """
    Validates evidence items for court submission requirements.
    
    This class performs comprehensive validation of evidence items
    to ensure they meet court standards for admissibility.
    """
    
    ALLOWED_EXTENSIONS = {
        '.pdf', '.doc', '.docx', '.txt', '.rtf',  # Documents
        '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff',  # Images
        '.mp4', '.avi', '.mov', '.wmv', '.flv',  # Videos
        '.mp3', '.wav', '.flac', '.aac', '.ogg',  # Audio
        '.xml', '.json', '.csv', '.xlsx'  # Data files
    }
    
    MAX_FILE_SIZE_MB = 500
    MIN_CHAIN_OF_CUSTODY_ENTRIES = 1
    
    def __init__(self, strict_mode: bool = True):
        """
        Initialize evidence validator.
        
        Args:
            strict_mode: Whether to enforce strict validation rules
        """
        self.strict_mode = strict_mode
    
    def validate_evidence(self, evidence: Evidence) -> Tuple[bool, List[str], List[str]]:
        """
        Validate a single evidence item.
        
        Args:
            evidence: The evidence item to validate
            
        Returns:
            Tuple of (is_valid, errors, warnings)
        """
        errors = []
        warnings = []
        
        # Basic metadata validation
        if not evidence.title or len(evidence.title.strip()) == 0:
            errors.append("Evidence title is required and cannot be empty")
        
        if len(evidence.title) > 500:
            warnings.append("Evidence title is very long (>500 characters)")
        
        # File validation if file path provided
        if evidence.file_path:
            file_path = Path(evidence.file_path)
            
            # Check file exists
            if not file_path.exists():
                errors.append(f"Evidence file does not exist: {evidence.file_path}")
            else:
                # Check file extension
                if file_path.suffix.lower() not in self.ALLOWED_EXTENSIONS:
                    if self.strict_mode:
                        errors.append(f"File extension not allowed: {file_path.suffix}")
                    else:
                        warnings.append(f"Unusual file extension: {file_path.suffix}")
                
                # Check file size
                try:
                    file_size = file_path.stat().st_size
                    size_mb = file_size / (1024 * 1024)
                    
                    if size_mb > self.MAX_FILE_SIZE_MB:
                        if self.strict_mode:
                            errors.append(f"File too large: {size_mb:.1f}MB (max: {self.MAX_FILE_SIZE_MB}MB)")
                        else:
                            warnings.append(f"Large file: {size_mb:.1f}MB")
                    
                    # Update evidence metadata if missing
                    if not evidence.file_size:
                        evidence.file_size = file_size
                        
                except OSError as e:
                    errors.append(f"Cannot access file: {e}")
        
        # Chain of custody validation
        if self.strict_mode and len(evidence.chain_of_custody) < self.MIN_CHAIN_OF_CUSTODY_ENTRIES:
            errors.append(f"Insufficient chain of custody entries (min: {self.MIN_CHAIN_OF_CUSTODY_ENTRIES})")
        
        # Hash validation
        if evidence.file_path and evidence.file_hash:
            try:
                calculated_hash = self._calculate_file_hash(Path(evidence.file_path))
                if calculated_hash != evidence.file_hash:
                    errors.append("File hash mismatch - file may have been tampered with")
            except Exception as e:
                warnings.append(f"Could not verify file hash: {e}")
        elif evidence.file_path and not evidence.file_hash:
            warnings.append("File hash not provided - integrity cannot be verified")
        
        # Authenticity check
        if self.strict_mode and not evidence.authenticity_verified:
            warnings.append("Evidence authenticity not verified")
        
        # Privacy compliance check
        if self.strict_mode and not evidence.privacy_compliant:
            warnings.append("Evidence privacy compliance not verified")
        
        return len(errors) == 0, errors, warnings
    
    def validate_evidence_batch(
        self, 
        evidence_list: List[Evidence]
    ) -> Tuple[List[Evidence], List[Evidence], Dict[str, List[str]]]:
        """
        Validate a batch of evidence items.
        
        Args:
            evidence_list: List of evidence items to validate
            
        Returns:
            Tuple of (valid_evidence, invalid_evidence, validation_details)
        """
        valid_evidence = []
        invalid_evidence = []
        validation_details = {}
        
        for evidence in evidence_list:
            is_valid, errors, warnings = self.validate_evidence(evidence)
            
            validation_details[evidence.evidence_id] = {
                'errors': errors,
                'warnings': warnings,
                'is_valid': is_valid
            }
            
            if is_valid:
                valid_evidence.append(evidence)
            else:
                invalid_evidence.append(evidence)
        
        return valid_evidence, invalid_evidence, validation_details
    
    def _calculate_file_hash(self, file_path: Path, algorithm: str = 'sha256') -> str:
        """Calculate hash of a file."""
        hash_func = getattr(hashlib, algorithm)()
        
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b""):
                hash_func.update(chunk)
        
        return hash_func.hexdigest()


class DigitalSignatureManager:
    """
    Manages digital signatures for evidence packages.
    
    Provides functionality to create and verify digital signatures
    using various cryptographic algorithms and certificate formats.
    """
    
    def __init__(self, key_size: int = 2048):
        """
        Initialize digital signature manager.
        
        Args:
            key_size: RSA key size for generated keys
        """
        self.key_size = key_size
        self._private_key = None
        self._public_key = None
    
    def generate_key_pair(self) -> Tuple[bytes, bytes]:
        """
        Generate a new RSA key pair.
        
        Returns:
            Tuple of (private_key_pem, public_key_pem)
        """
        self._private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=self.key_size,
        )
        self._public_key = self._private_key.public_key()
        
        private_pem = self._private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()
        )
        
        public_pem = self._public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )
        
        return private_pem, public_pem
    
    def load_key_pair(self, private_key_pem: bytes, password: Optional[bytes] = None):
        """Load an existing key pair from PEM format."""
        self._private_key = serialization.load_pem_private_key(
            private_key_pem, password=password
        )
        self._public_key = self._private_key.public_key()
    
    def sign_data(self, data: bytes, signer_name: str) -> DigitalSignature:
        """
        Create a digital signature for data.
        
        Args:
            data: The data to sign
            signer_name: Name of the signer
            
        Returns:
            Digital signature object
        """
        if not self._private_key:
            raise ValueError("No private key loaded")
        
        signature = self._private_key.sign(
            data,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH
            ),
            hashes.SHA256()
        )
        
        # Calculate certificate fingerprint (simplified)
        cert_fingerprint = hashlib.sha256(
            self._public_key.public_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PublicFormat.SubjectPublicKeyInfo
            )
        ).hexdigest()[:16]
        
        return DigitalSignature(
            algorithm="RSA-SHA256",
            certificate_fingerprint=cert_fingerprint,
            signature_value=signature.hex(),
            signer_name=signer_name,
            is_valid=True,
            trust_chain_verified=False  # Would require actual certificate chain
        )
    
    def verify_signature(self, data: bytes, signature: DigitalSignature) -> bool:
        """
        Verify a digital signature.
        
        Args:
            data: The original data
            signature: The digital signature to verify
            
        Returns:
            True if signature is valid
        """
        if not self._public_key:
            return False
        
        try:
            signature_bytes = bytes.fromhex(signature.signature_value)
            self._public_key.verify(
                signature_bytes,
                data,
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                hashes.SHA256()
            )
            return True
        except Exception:
            return False


class CourtPackager:
    """
    Creates court-ready evidence packages with proper validation and signatures.
    
    This class handles the complete process of creating evidence packages
    that meet international court submission requirements.
    """
    
    def __init__(
        self,
        signature_manager: Optional[DigitalSignatureManager] = None,
        validator: Optional[EvidenceValidator] = None,
        enable_compression: bool = True
    ):
        """
        Initialize court packager.
        
        Args:
            signature_manager: Digital signature manager instance
            validator: Evidence validator instance
            enable_compression: Whether to compress packages
        """
        self.signature_manager = signature_manager or DigitalSignatureManager()
        self.validator = validator or EvidenceValidator()
        self.enable_compression = enable_compression
    
    def create_package(
        self,
        evidence_list: List[Evidence],
        case_data: CaseData,
        output_path: Optional[Union[str, Path]] = None,
        package_format: str = "zip"
    ) -> EvidencePackage:
        """
        Create a complete evidence package for court submission.
        
        Args:
            evidence_list: List of evidence items to package
            case_data: Associated case data
            output_path: Output directory path
            package_format: Package format (zip, tar, tar.gz, directory)
            
        Returns:
            Evidence package object
            
        Raises:
            ExportError: If package creation fails
            ValidationError: If validation fails
        """
        try:
            logger.info(f"Creating evidence package for case {case_data.case_id}")
            
            # Step 1: Validate all evidence
            valid_evidence, invalid_evidence, validation_details = \
                self.validator.validate_evidence_batch(evidence_list)
            
            if invalid_evidence and self.validator.strict_mode:
                error_msg = f"Found {len(invalid_evidence)} invalid evidence items"
                logger.error(error_msg)
                raise ValidationError(error_msg)
            
            # Step 2: Create package manifest
            manifest = self._create_manifest(valid_evidence, case_data)
            
            # Step 3: Create temporary working directory
            if output_path:
                work_dir = Path(output_path) / f"package_{case_data.case_id}"
            else:
                work_dir = Path.cwd() / "temp" / f"package_{case_data.case_id}"
            
            work_dir.mkdir(parents=True, exist_ok=True)
            
            # Step 4: Copy evidence files and create directory structure
            evidence_dir = work_dir / "evidence"
            evidence_dir.mkdir(exist_ok=True)
            
            processed_files = []
            total_size = 0
            
            for evidence in tqdm(valid_evidence, desc="Processing evidence"):
                if evidence.file_path and Path(evidence.file_path).exists():
                    source_path = Path(evidence.file_path)
                    target_path = evidence_dir / f"{evidence.evidence_id}_{source_path.name}"
                    
                    # Copy file
                    shutil.copy2(source_path, target_path)
                    
                    # Update evidence metadata
                    evidence.file_size = target_path.stat().st_size
                    evidence.file_hash = self._calculate_file_hash(target_path)
                    
                    processed_files.append({
                        "evidence_id": evidence.evidence_id,
                        "original_name": source_path.name,
                        "package_name": target_path.name,
                        "size": evidence.file_size,
                        "hash": evidence.file_hash,
                        "type": evidence.evidence_type.value
                    })
                    
                    total_size += evidence.file_size
            
            # Step 5: Create metadata files
            self._create_metadata_files(work_dir, case_data, valid_evidence, manifest, validation_details)
            
            # Step 6: Update manifest with file information
            manifest.evidence_count = len(valid_evidence)
            manifest.total_size_bytes = total_size
            manifest.files = processed_files
            manifest.checksums = {f["package_name"]: f["hash"] for f in processed_files}
            
            # Step 7: Create digital signatures
            signatures = []
            if self.signature_manager:
                manifest_data = manifest.json().encode('utf-8')
                signature = self.signature_manager.sign_data(manifest_data, "court_packager")
                signatures.append(signature)
                manifest.digital_signatures = signatures
            
            # Step 8: Create final package
            if package_format == "directory":
                package_path = work_dir
            else:
                package_path = self._create_compressed_package(work_dir, package_format)
            
            # Step 9: Create compliance report (simplified)
            from .core import ComplianceReport, PrivacyAssessment, ComplianceStatus
            
            privacy_assessment = PrivacyAssessment(
                data_subject_count=len([e for e in valid_evidence if e.redaction_applied]),
                compliance_status=ComplianceStatus.NEEDS_REVIEW,
                assessor="court_packager",
                personal_data_types=["redacted_content"]
            )
            
            compliance_report = ComplianceReport(
                case_id=case_data.case_id,
                overall_status=ComplianceStatus.NEEDS_REVIEW,
                privacy_assessment=privacy_assessment,
                report_generated_by="court_packager"
            )
            
            # Step 10: Create final evidence package
            evidence_package = EvidencePackage(
                case_id=case_data.case_id,
                court=case_data.court,
                manifest=manifest,
                evidence=valid_evidence,
                compliance_report=compliance_report,
                package_path=str(package_path),
                encrypted=False,  # Would implement encryption separately
                compression_used=package_format != "directory",
                integrity_verified=len(signatures) > 0,
                submission_ready=len(invalid_evidence) == 0
            )
            
            logger.info(f"Successfully created evidence package: {package_path}")
            return evidence_package
            
        except Exception as e:
            logger.error(f"Failed to create evidence package: {e}")
            raise ExportError(f"Package creation failed: {e}") from e
    
    def verify_package(self, package_path: Union[str, Path]) -> Tuple[bool, List[str]]:
        """
        Verify the integrity of an evidence package.
        
        Args:
            package_path: Path to the package file or directory
            
        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []
        package_path = Path(package_path)
        
        try:
            if package_path.is_file():
                # Extract compressed package to temporary directory
                temp_dir = package_path.parent / f"verify_{package_path.stem}"
                temp_dir.mkdir(exist_ok=True)
                
                if package_path.suffix == '.zip':
                    with zipfile.ZipFile(package_path, 'r') as zf:
                        zf.extractall(temp_dir)
                elif package_path.suffix in ['.tar', '.tgz']:
                    with tarfile.open(package_path, 'r:*') as tf:
                        tf.extractall(temp_dir)
                else:
                    errors.append(f"Unsupported package format: {package_path.suffix}")
                    return False, errors
                
                work_dir = temp_dir
            else:
                work_dir = package_path
            
            # Check required files
            required_files = ['manifest.json', 'case_metadata.json', 'evidence']
            for required_file in required_files:
                if not (work_dir / required_file).exists():
                    errors.append(f"Missing required file/directory: {required_file}")
            
            # Load and verify manifest
            manifest_path = work_dir / 'manifest.json'
            if manifest_path.exists():
                with open(manifest_path, 'r') as f:
                    manifest_data = json.load(f)
                
                # Verify file checksums
                evidence_dir = work_dir / 'evidence'
                if evidence_dir.exists():
                    for file_info in manifest_data.get('files', []):
                        file_path = evidence_dir / file_info['package_name']
                        if not file_path.exists():
                            errors.append(f"Missing evidence file: {file_info['package_name']}")
                        else:
                            # Verify checksum
                            actual_hash = self._calculate_file_hash(file_path)
                            expected_hash = file_info['hash']
                            if actual_hash != expected_hash:
                                errors.append(f"Hash mismatch for {file_info['package_name']}")
            
            # Clean up temporary directory if created
            if package_path.is_file() and 'temp_dir' in locals():
                shutil.rmtree(temp_dir, ignore_errors=True)
            
            return len(errors) == 0, errors
            
        except Exception as e:
            errors.append(f"Verification error: {str(e)}")
            return False, errors
    
    def _create_manifest(self, evidence_list: List[Evidence], case_data: CaseData) -> PackageManifest:
        """Create package manifest."""
        return PackageManifest(
            package_name=f"{case_data.case_name}_evidence_package",
            creator="lemkin-export",
            evidence_count=len(evidence_list),
            metadata={
                "case_id": case_data.case_id,
                "case_name": case_data.case_name,
                "court": case_data.court.value,
                "creation_tool": "lemkin-export court_packager",
                "format_version": "1.0"
            }
        )
    
    def _create_metadata_files(
        self,
        work_dir: Path,
        case_data: CaseData,
        evidence_list: List[Evidence],
        manifest: PackageManifest,
        validation_details: Dict[str, Any]
    ) -> None:
        """Create metadata files in the package."""
        # Save manifest
        with open(work_dir / 'manifest.json', 'w', encoding='utf-8') as f:
            json.dump(manifest.dict(), f, indent=2, default=str)
        
        # Save case metadata
        with open(work_dir / 'case_metadata.json', 'w', encoding='utf-8') as f:
            json.dump(case_data.dict(), f, indent=2, default=str)
        
        # Save evidence metadata
        evidence_metadata = [e.dict() for e in evidence_list]
        with open(work_dir / 'evidence_metadata.json', 'w', encoding='utf-8') as f:
            json.dump(evidence_metadata, f, indent=2, default=str)
        
        # Save validation report
        with open(work_dir / 'validation_report.json', 'w', encoding='utf-8') as f:
            json.dump(validation_details, f, indent=2, default=str)
        
        # Create CSV index for easy reference
        evidence_df = pd.DataFrame([
            {
                'evidence_id': e.evidence_id,
                'title': e.title,
                'type': e.evidence_type.value,
                'collection_date': e.collection_date,
                'source': e.source,
                'file_size': e.file_size,
                'privacy_compliant': e.privacy_compliant,
                'authenticity_verified': e.authenticity_verified
            }
            for e in evidence_list
        ])
        evidence_df.to_csv(work_dir / 'evidence_index.csv', index=False)
    
    def _create_compressed_package(self, work_dir: Path, package_format: str) -> Path:
        """Create compressed package from working directory."""
        package_name = f"{work_dir.name}"
        
        if package_format == "zip":
            package_path = work_dir.parent / f"{package_name}.zip"
            with zipfile.ZipFile(package_path, 'w', zipfile.ZIP_DEFLATED) as zf:
                for file_path in work_dir.rglob('*'):
                    if file_path.is_file():
                        arcname = file_path.relative_to(work_dir)
                        zf.write(file_path, arcname)
        
        elif package_format in ["tar", "tar.gz"]:
            suffix = ".tar.gz" if package_format == "tar.gz" else ".tar"
            package_path = work_dir.parent / f"{package_name}{suffix}"
            mode = "w:gz" if package_format == "tar.gz" else "w"
            
            with tarfile.open(package_path, mode) as tf:
                tf.add(work_dir, arcname=package_name)
        
        else:
            raise ValueError(f"Unsupported package format: {package_format}")
        
        # Clean up working directory
        shutil.rmtree(work_dir, ignore_errors=True)
        
        return package_path
    
    def _calculate_file_hash(self, file_path: Path, algorithm: str = 'sha256') -> str:
        """Calculate hash of a file."""
        hash_func = getattr(hashlib, algorithm)()
        
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b""):
                hash_func.update(chunk)
        
        return hash_func.hexdigest()


def create_court_package(evidence: List[Evidence], case_data: CaseData) -> CourtPackage:
    """
    Convenience function to create a court-ready evidence package.
    
    Args:
        evidence: List of evidence items
        case_data: Associated case data
        
    Returns:
        Court package ready for submission
    """
    packager = CourtPackager()
    evidence_package = packager.create_package(evidence, case_data)
    
    # Create a basic court package
    court_package = CourtPackage(
        court=case_data.court,
        case_data=case_data,
        evidence_package=evidence_package,
        submission_format={"format": "standard", "version": "1.0"},
        compliance_report=evidence_package.compliance_report,
        ready_for_submission=evidence_package.submission_ready
    )
    
    return court_package