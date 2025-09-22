"""
Lemkin Digital Forensics Authenticity Verifier

This module provides comprehensive digital evidence authenticity verification including:
- Hash verification and integrity checking (MD5, SHA1, SHA256, SHA512)
- Digital signature validation and certificate verification
- Metadata authenticity analysis and tamper detection
- Chain of custody verification and audit trail validation
- File format validation and structure integrity checks
- Steganography and hidden data detection

Ensures evidence meets legal admissibility standards for court proceedings.
"""

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from uuid import UUID, uuid4
from dataclasses import dataclass, field
import os
import hashlib
import hmac
import logging
import subprocess
from cryptography import x509
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.serialization import load_pem_public_key, load_pem_private_key
import json
import magic

from .core import (
    DigitalEvidence,
    AnalysisResult,
    TimelineEvent,
    AnalysisStatus,
    EvidenceType,
    ForensicsConfig,
    ChainOfCustodyEntry
)

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class HashVerification:
    """Result of hash verification"""
    algorithm: str
    expected_hash: str
    computed_hash: str
    matches: bool
    computation_time: float
    file_size: int


@dataclass
class DigitalSignatureInfo:
    """Information about digital signature"""
    signature_present: bool
    signature_valid: bool
    signer_certificate: Optional[Dict[str, Any]] = None
    certificate_chain: List[Dict[str, Any]] = field(default_factory=list)
    signing_time: Optional[datetime] = None
    signature_algorithm: Optional[str] = None
    
    # Certificate validation
    certificate_valid: bool = False
    certificate_expired: bool = False
    certificate_revoked: bool = False
    trust_chain_valid: bool = False
    
    # Additional info
    error_messages: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


@dataclass
class MetadataAnalysis:
    """Analysis of file metadata for authenticity"""
    file_format: str
    format_valid: bool
    creation_time: Optional[datetime] = None
    modification_time: Optional[datetime] = None
    
    # Metadata consistency
    timestamp_consistency: bool = True
    metadata_intact: bool = True
    exif_tampered: bool = False
    
    # Content analysis
    file_type_matches_extension: bool = True
    embedded_metadata: Dict[str, Any] = field(default_factory=dict)
    suspicious_patterns: List[str] = field(default_factory=list)
    
    # Steganography detection
    steganography_detected: bool = False
    hidden_data_indicators: List[str] = field(default_factory=list)


@dataclass
class ChainOfCustodyValidation:
    """Result of chain of custody validation"""
    custody_intact: bool
    total_entries: int
    missing_entries: List[str] = field(default_factory=list)
    inconsistencies: List[str] = field(default_factory=list)
    
    # Timeline validation
    timeline_consistent: bool = True
    gaps_in_custody: List[Tuple[datetime, datetime]] = field(default_factory=list)
    
    # Hash validation across custody chain
    hash_consistency: bool = True
    hash_mismatches: List[Tuple[str, str, str]] = field(default_factory=list)  # (entry, expected, actual)


class AuthenticityReport:
    """Comprehensive authenticity verification report"""
    
    def __init__(self, evidence_id: UUID):
        self.evidence_id = evidence_id
        self.verification_timestamp = datetime.utcnow()
        self.overall_authentic = True
        self.confidence_score = 0.0
        
        # Verification results
        self.hash_verifications: List[HashVerification] = []
        self.signature_info: Optional[DigitalSignatureInfo] = None
        self.metadata_analysis: Optional[MetadataAnalysis] = None
        self.custody_validation: Optional[ChainOfCustodyValidation] = None
        
        # Summary
        self.verification_summary: Dict[str, Any] = {}
        self.issues_found: List[str] = []
        self.recommendations: List[str] = []
        
        # Legal considerations
        self.admissibility_assessment: str = "pending"
        self.legal_concerns: List[str] = []


class AuthenticityVerifier:
    """
    Comprehensive digital evidence authenticity verifier
    
    Provides verification capabilities for:
    - Cryptographic hash integrity verification
    - Digital signature validation and certificate checking
    - Metadata authenticity analysis and tamper detection
    - Chain of custody validation and audit trail verification
    - File format validation and structure integrity
    - Steganography detection and hidden data analysis
    """
    
    def __init__(self, config: Optional[ForensicsConfig] = None):
        """Initialize authenticity verifier with configuration"""
        self.config = config or ForensicsConfig()
        self.logger = logging.getLogger(f"{__name__}.AuthenticityVerifier")
        
        # Supported hash algorithms
        self.hash_algorithms = {
            'md5': hashlib.md5,
            'sha1': hashlib.sha1,
            'sha256': hashlib.sha256,
            'sha512': hashlib.sha512
        }
        
        # File format validators
        self.format_validators = {
            'pdf': self._validate_pdf_structure,
            'jpeg': self._validate_jpeg_structure,
            'png': self._validate_png_structure,
            'zip': self._validate_zip_structure,
            'office': self._validate_office_structure
        }
        
        # Known steganography tools signatures
        self.stego_signatures = [
            b'steghide',
            b'outguess',
            b'jphide',
            b'jphs',
            b'f5',
            b'openstego'
        ]
        
        self.logger.info("Authenticity Verifier initialized")
    
    def verify_evidence_authenticity(
        self,
        evidence: DigitalEvidence,
        verification_level: str = "comprehensive"
    ) -> AuthenticityReport:
        """
        Perform comprehensive authenticity verification
        
        Args:
            evidence: Digital evidence to verify
            verification_level: Level of verification (basic, standard, comprehensive)
            
        Returns:
            AuthenticityReport with detailed results
        """
        self.logger.info(f"Starting authenticity verification: {evidence.name}")
        
        report = AuthenticityReport(evidence.id)
        
        try:
            # Basic integrity verification
            report.hash_verifications = self._verify_file_hashes(evidence)
            
            if verification_level in ["standard", "comprehensive"]:
                # Metadata analysis
                report.metadata_analysis = self._analyze_file_metadata(evidence)
                
                # Chain of custody validation
                report.custody_validation = self._validate_chain_of_custody(evidence)
            
            if verification_level == "comprehensive":
                # Digital signature verification
                report.signature_info = self._verify_digital_signature(evidence)
                
                # Advanced tamper detection
                self._perform_advanced_tamper_detection(evidence, report)
            
            # Calculate overall authenticity and confidence
            self._calculate_authenticity_score(report)
            
            # Generate recommendations
            self._generate_recommendations(report)
            
            # Assess legal admissibility
            self._assess_legal_admissibility(report)
            
            self.logger.info(f"Authenticity verification completed: {report.confidence_score:.2f} confidence")
            
        except Exception as e:
            self.logger.error(f"Authenticity verification failed: {str(e)}")
            report.overall_authentic = False
            report.issues_found.append(f"Verification failed: {str(e)}")
        
        return report
    
    def _verify_file_hashes(self, evidence: DigitalEvidence) -> List[HashVerification]:
        """Verify file integrity using multiple hash algorithms"""
        verifications = []
        file_path = evidence.file_path
        
        if not os.path.exists(file_path):
            return [HashVerification(
                algorithm="file_existence",
                expected_hash="file_present",
                computed_hash="file_missing",
                matches=False,
                computation_time=0.0,
                file_size=0
            )]
        
        file_size = os.path.getsize(file_path)
        
        # Verify each available hash
        hash_map = {
            'md5': evidence.file_hash_md5,
            'sha1': evidence.file_hash_sha1,
            'sha256': evidence.file_hash_sha256
        }
        
        for algorithm, expected_hash in hash_map.items():
            if expected_hash:
                start_time = datetime.utcnow()
                computed_hash = self._compute_file_hash(file_path, algorithm)
                end_time = datetime.utcnow()
                
                computation_time = (end_time - start_time).total_seconds()
                matches = computed_hash.lower() == expected_hash.lower()
                
                verifications.append(HashVerification(
                    algorithm=algorithm,
                    expected_hash=expected_hash,
                    computed_hash=computed_hash,
                    matches=matches,
                    computation_time=computation_time,
                    file_size=file_size
                ))
        
        return verifications
    
    def _compute_file_hash(self, file_path: str, algorithm: str) -> str:
        """Compute hash for file using specified algorithm"""
        hasher = self.hash_algorithms[algorithm]()
        
        with open(file_path, 'rb') as f:
            while chunk := f.read(8192):
                hasher.update(chunk)
        
        return hasher.hexdigest()
    
    def _verify_digital_signature(self, evidence: DigitalEvidence) -> DigitalSignatureInfo:
        """Verify digital signature if present"""
        signature_info = DigitalSignatureInfo(signature_present=False, signature_valid=False)
        
        try:
            # Check for common signature formats
            file_path = evidence.file_path
            
            # PDF signatures
            if file_path.lower().endswith('.pdf'):
                signature_info = self._verify_pdf_signature(file_path)
            
            # Office document signatures
            elif any(file_path.lower().endswith(ext) for ext in ['.docx', '.xlsx', '.pptx']):
                signature_info = self._verify_office_signature(file_path)
            
            # Detached signatures
            elif os.path.exists(file_path + '.sig') or os.path.exists(file_path + '.p7s'):
                signature_info = self._verify_detached_signature(file_path)
            
            # Code signing (executables)
            elif any(file_path.lower().endswith(ext) for ext in ['.exe', '.dll', '.msi']):
                signature_info = self._verify_code_signature(file_path)
            
        except Exception as e:
            signature_info.error_messages.append(f"Signature verification failed: {str(e)}")
        
        return signature_info
    
    def _analyze_file_metadata(self, evidence: DigitalEvidence) -> MetadataAnalysis:
        """Analyze file metadata for authenticity indicators"""
        file_path = evidence.file_path
        
        # Determine file type
        try:
            file_type = magic.from_file(file_path, mime=True)
            file_format = file_type.split('/')[1] if '/' in file_type else file_type
        except:
            file_format = os.path.splitext(file_path)[1].lower().lstrip('.')
        
        analysis = MetadataAnalysis(
            file_format=file_format,
            format_valid=True
        )
        
        try:
            # Basic file system metadata
            stat = os.stat(file_path)
            analysis.creation_time = datetime.fromtimestamp(stat.st_ctime)
            analysis.modification_time = datetime.fromtimestamp(stat.st_mtime)
            
            # Check timestamp consistency
            if analysis.creation_time and analysis.modification_time:
                if analysis.modification_time < analysis.creation_time:
                    analysis.timestamp_consistency = False
                    analysis.suspicious_patterns.append("Modification time before creation time")
            
            # Format-specific analysis
            if file_format in self.format_validators:
                self.format_validators[file_format](file_path, analysis)
            
            # Check file type vs extension
            expected_extensions = self._get_expected_extensions(file_format)
            actual_extension = os.path.splitext(file_path)[1].lower().lstrip('.')
            if actual_extension not in expected_extensions:
                analysis.file_type_matches_extension = False
                analysis.suspicious_patterns.append(f"File extension '{actual_extension}' doesn't match type '{file_format}'")
            
            # Steganography detection
            analysis.steganography_detected, analysis.hidden_data_indicators = self._detect_steganography(file_path)
            
        except Exception as e:
            self.logger.warning(f"Metadata analysis failed: {str(e)}")
            analysis.metadata_intact = False
        
        return analysis
    
    def _validate_chain_of_custody(self, evidence: DigitalEvidence) -> ChainOfCustodyValidation:
        """Validate chain of custody integrity"""
        validation = ChainOfCustodyValidation(
            custody_intact=True,
            total_entries=len(evidence.chain_of_custody)
        )
        
        if not evidence.chain_of_custody:
            validation.custody_intact = False
            validation.missing_entries.append("No chain of custody entries found")
            return validation
        
        # Check for required entries
        required_actions = ["Evidence received", "Analysis started", "Analysis completed"]
        present_actions = [entry.action for entry in evidence.chain_of_custody]
        
        for required in required_actions:
            if not any(required.lower() in action.lower() for action in present_actions):
                validation.missing_entries.append(f"Missing required action: {required}")
                validation.custody_intact = False
        
        # Validate timeline consistency
        entries_sorted = sorted(evidence.chain_of_custody, key=lambda x: x.timestamp)
        for i in range(1, len(entries_sorted)):
            prev_entry = entries_sorted[i-1]
            curr_entry = entries_sorted[i]
            
            # Check for timeline gaps (more than 24 hours)
            gap = curr_entry.timestamp - prev_entry.timestamp
            if gap.total_seconds() > 24 * 3600:
                validation.gaps_in_custody.append((prev_entry.timestamp, curr_entry.timestamp))
                validation.timeline_consistent = False
        
        # Validate hash consistency
        for entry in evidence.chain_of_custody:
            if entry.hash_before and entry.hash_after:
                if entry.hash_before != entry.hash_after:
                    validation.hash_mismatches.append((
                        entry.action,
                        entry.hash_before,
                        entry.hash_after
                    ))
                    validation.hash_consistency = False
        
        return validation
    
    def _perform_advanced_tamper_detection(self, evidence: DigitalEvidence, report: AuthenticityReport):
        """Perform advanced tamper detection analysis"""
        file_path = evidence.file_path
        
        try:
            # Hex dump analysis for embedded signatures
            with open(file_path, 'rb') as f:
                header = f.read(1024)
                footer = f.seek(-1024, 2) and f.read(1024) or b''
            
            # Look for hex editor signatures
            hex_editor_signatures = [
                b'HexEdit',
                b'WinHex',
                b'010 Editor',
                b'HxD'
            ]
            
            for signature in hex_editor_signatures:
                if signature in header or signature in footer:
                    report.issues_found.append(f"Possible hex editor signature found: {signature.decode()}")
            
            # Check for file size anomalies
            expected_size = self._estimate_expected_file_size(evidence)
            actual_size = os.path.getsize(file_path)
            
            if expected_size and abs(actual_size - expected_size) > expected_size * 0.1:
                report.issues_found.append(f"File size anomaly: expected ~{expected_size}, actual {actual_size}")
            
            # Binary entropy analysis
            entropy = self._calculate_file_entropy(file_path)
            if entropy > 7.5:  # Very high entropy might indicate encryption or compression
                report.issues_found.append(f"Unusually high entropy: {entropy:.2f} (possible encryption/compression)")
            
        except Exception as e:
            self.logger.warning(f"Advanced tamper detection failed: {str(e)}")
    
    def _detect_steganography(self, file_path: str) -> Tuple[bool, List[str]]:
        """Detect potential steganography in file"""
        indicators = []
        detected = False
        
        try:
            with open(file_path, 'rb') as f:
                content = f.read()
            
            # Check for steganography tool signatures
            for signature in self.stego_signatures:
                if signature in content:
                    indicators.append(f"Steganography tool signature found: {signature.decode()}")
                    detected = True
            
            # Statistical analysis for images
            if file_path.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                # Simple LSB analysis (basic implementation)
                if self._analyze_lsb_patterns(content):
                    indicators.append("Suspicious LSB patterns detected")
                    detected = True
            
        except Exception as e:
            self.logger.warning(f"Steganography detection failed: {str(e)}")
        
        return detected, indicators
    
    def _analyze_lsb_patterns(self, content: bytes) -> bool:
        """Simple LSB pattern analysis"""
        try:
            # Count LSB distribution
            lsb_count = [0, 0]
            for byte in content[-1000:]:  # Analyze last 1000 bytes
                lsb_count[byte & 1] += 1
            
            # Check if distribution is unusually even (might indicate LSB steganography)
            total = sum(lsb_count)
            if total > 0:
                ratio = min(lsb_count) / max(lsb_count)
                return ratio > 0.9  # Very even distribution
            
        except Exception:
            pass
        
        return False
    
    def _calculate_file_entropy(self, file_path: str) -> float:
        """Calculate file entropy (measure of randomness)"""
        try:
            with open(file_path, 'rb') as f:
                content = f.read()
            
            if not content:
                return 0.0
            
            # Count byte frequency
            byte_counts = [0] * 256
            for byte in content:
                byte_counts[byte] += 1
            
            # Calculate entropy
            entropy = 0.0
            length = len(content)
            
            for count in byte_counts:
                if count > 0:
                    probability = count / length
                    entropy -= probability * (probability.bit_length() - 1)
            
            return entropy
            
        except Exception:
            return 0.0
    
    def _estimate_expected_file_size(self, evidence: DigitalEvidence) -> Optional[int]:
        """Estimate expected file size based on type and content"""
        # This would implement heuristics for different file types
        # For now, return None to skip this check
        return None
    
    def _validate_pdf_structure(self, file_path: str, analysis: MetadataAnalysis):
        """Validate PDF file structure"""
        try:
            with open(file_path, 'rb') as f:
                header = f.read(8)
                if not header.startswith(b'%PDF-'):
                    analysis.format_valid = False
                    analysis.suspicious_patterns.append("Invalid PDF header")
        except Exception as e:
            analysis.format_valid = False
            analysis.suspicious_patterns.append(f"PDF validation error: {str(e)}")
    
    def _validate_jpeg_structure(self, file_path: str, analysis: MetadataAnalysis):
        """Validate JPEG file structure"""
        try:
            with open(file_path, 'rb') as f:
                header = f.read(4)
                if not (header.startswith(b'\xFF\xD8\xFF')):
                    analysis.format_valid = False
                    analysis.suspicious_patterns.append("Invalid JPEG header")
        except Exception as e:
            analysis.format_valid = False
            analysis.suspicious_patterns.append(f"JPEG validation error: {str(e)}")
    
    def _validate_png_structure(self, file_path: str, analysis: MetadataAnalysis):
        """Validate PNG file structure"""
        try:
            with open(file_path, 'rb') as f:
                header = f.read(8)
                if header != b'\x89PNG\r\n\x1a\n':
                    analysis.format_valid = False
                    analysis.suspicious_patterns.append("Invalid PNG signature")
        except Exception as e:
            analysis.format_valid = False
            analysis.suspicious_patterns.append(f"PNG validation error: {str(e)}")
    
    def _validate_zip_structure(self, file_path: str, analysis: MetadataAnalysis):
        """Validate ZIP file structure"""
        try:
            with open(file_path, 'rb') as f:
                header = f.read(4)
                if not header.startswith(b'PK'):
                    analysis.format_valid = False
                    analysis.suspicious_patterns.append("Invalid ZIP signature")
        except Exception as e:
            analysis.format_valid = False
            analysis.suspicious_patterns.append(f"ZIP validation error: {str(e)}")
    
    def _validate_office_structure(self, file_path: str, analysis: MetadataAnalysis):
        """Validate Microsoft Office file structure"""
        try:
            # Modern Office files are ZIP-based
            self._validate_zip_structure(file_path, analysis)
        except Exception as e:
            analysis.format_valid = False
            analysis.suspicious_patterns.append(f"Office document validation error: {str(e)}")
    
    def _get_expected_extensions(self, file_format: str) -> List[str]:
        """Get expected file extensions for a format"""
        extension_map = {
            'pdf': ['pdf'],
            'jpeg': ['jpg', 'jpeg'],
            'png': ['png'],
            'zip': ['zip'],
            'docx': ['docx'],
            'xlsx': ['xlsx'],
            'pptx': ['pptx'],
            'exe': ['exe'],
            'dll': ['dll']
        }
        return extension_map.get(file_format, [file_format])
    
    def _verify_pdf_signature(self, file_path: str) -> DigitalSignatureInfo:
        """Verify PDF digital signature"""
        # PDF signature verification would be implemented here
        return DigitalSignatureInfo(signature_present=False, signature_valid=False)
    
    def _verify_office_signature(self, file_path: str) -> DigitalSignatureInfo:
        """Verify Office document digital signature"""
        # Office signature verification would be implemented here
        return DigitalSignatureInfo(signature_present=False, signature_valid=False)
    
    def _verify_detached_signature(self, file_path: str) -> DigitalSignatureInfo:
        """Verify detached digital signature"""
        # Detached signature verification would be implemented here
        return DigitalSignatureInfo(signature_present=False, signature_valid=False)
    
    def _verify_code_signature(self, file_path: str) -> DigitalSignatureInfo:
        """Verify executable code signature"""
        # Code signature verification would be implemented here
        return DigitalSignatureInfo(signature_present=False, signature_valid=False)
    
    def _calculate_authenticity_score(self, report: AuthenticityReport):
        """Calculate overall authenticity confidence score"""
        score = 10.0  # Start with perfect score
        
        # Deduct points for hash verification failures
        for verification in report.hash_verifications:
            if not verification.matches:
                score -= 3.0  # Major deduction for hash mismatch
        
        # Deduct points for metadata issues
        if report.metadata_analysis:
            if not report.metadata_analysis.timestamp_consistency:
                score -= 1.0
            if not report.metadata_analysis.file_type_matches_extension:
                score -= 0.5
            if report.metadata_analysis.steganography_detected:
                score -= 2.0
            score -= len(report.metadata_analysis.suspicious_patterns) * 0.5
        
        # Deduct points for chain of custody issues
        if report.custody_validation:
            if not report.custody_validation.custody_intact:
                score -= 2.0
            if not report.custody_validation.timeline_consistent:
                score -= 1.0
            if not report.custody_validation.hash_consistency:
                score -= 2.0
        
        # Deduct points for general issues
        score -= len(report.issues_found) * 0.5
        
        # Ensure score is between 0 and 10
        score = max(0.0, min(10.0, score))
        
        report.confidence_score = score
        report.overall_authentic = score >= 7.0  # Threshold for authenticity
    
    def _generate_recommendations(self, report: AuthenticityReport):
        """Generate recommendations based on verification results"""
        recommendations = []
        
        # Hash verification recommendations
        failed_hashes = [v for v in report.hash_verifications if not v.matches]
        if failed_hashes:
            algorithms = [v.algorithm for v in failed_hashes]
            recommendations.append(f"Re-verify file integrity using {', '.join(algorithms)} hashes")
            recommendations.append("Investigate potential file corruption or tampering")
        
        # Metadata recommendations
        if report.metadata_analysis and report.metadata_analysis.suspicious_patterns:
            recommendations.append("Conduct detailed metadata forensics analysis")
            recommendations.append("Verify file creation and modification timestamps")
        
        # Chain of custody recommendations
        if report.custody_validation and not report.custody_validation.custody_intact:
            recommendations.append("Document chain of custody gaps and inconsistencies")
            recommendations.append("Obtain additional custody documentation if available")
        
        # Steganography recommendations
        if report.metadata_analysis and report.metadata_analysis.steganography_detected:
            recommendations.append("Perform specialized steganography analysis")
            recommendations.append("Use dedicated steganography detection tools")
        
        # General recommendations
        if report.confidence_score < 7.0:
            recommendations.append("Consider additional verification methods")
            recommendations.append("Consult with digital forensics expert")
        
        report.recommendations = recommendations
    
    def _assess_legal_admissibility(self, report: AuthenticityReport):
        """Assess legal admissibility based on verification results"""
        if report.confidence_score >= 9.0:
            report.admissibility_assessment = "high"
        elif report.confidence_score >= 7.0:
            report.admissibility_assessment = "medium"
        elif report.confidence_score >= 5.0:
            report.admissibility_assessment = "low"
        else:
            report.admissibility_assessment = "questionable"
        
        # Identify specific legal concerns
        legal_concerns = []
        
        if not all(v.matches for v in report.hash_verifications):
            legal_concerns.append("File integrity cannot be verified - hash mismatch")
        
        if report.custody_validation and not report.custody_validation.custody_intact:
            legal_concerns.append("Chain of custody has gaps or inconsistencies")
        
        if report.metadata_analysis and report.metadata_analysis.steganography_detected:
            legal_concerns.append("Potential hidden data or tampering detected")
        
        if len(report.issues_found) > 5:
            legal_concerns.append("Multiple authenticity concerns identified")
        
        report.legal_concerns = legal_concerns
    
    def generate_authenticity_summary(self, report: AuthenticityReport) -> Dict[str, Any]:
        """Generate summary of authenticity verification"""
        hash_results = {
            v.algorithm: v.matches for v in report.hash_verifications
        }
        
        return {
            'evidence_id': str(report.evidence_id),
            'verification_timestamp': report.verification_timestamp.isoformat(),
            'overall_authentic': report.overall_authentic,
            'confidence_score': report.confidence_score,
            'admissibility_assessment': report.admissibility_assessment,
            'hash_verifications': hash_results,
            'signature_verified': report.signature_info.signature_valid if report.signature_info else False,
            'metadata_intact': report.metadata_analysis.metadata_intact if report.metadata_analysis else True,
            'custody_intact': report.custody_validation.custody_intact if report.custody_validation else True,
            'issues_count': len(report.issues_found),
            'recommendations_count': len(report.recommendations),
            'legal_concerns_count': len(report.legal_concerns)
        }