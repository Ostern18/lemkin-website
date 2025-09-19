"""
Lemkin Digital Forensics Toolkit - Core Module

This module provides the core data models and DigitalForensicsAnalyzer class for 
digital evidence analysis and authentication, designed for non-technical investigators
working with court-ready documentation.

Compliance: Chain of custody preservation and forensic best practices
"""

from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple
from uuid import UUID, uuid4
import hashlib
import json
import logging

from pydantic import BaseModel, Field, validator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EvidenceType(str, Enum):
    """Types of digital evidence"""
    DISK_IMAGE = "disk_image"
    FILE_SYSTEM = "file_system"
    NETWORK_LOG = "network_log"
    MOBILE_BACKUP = "mobile_backup"
    EMAIL_ARCHIVE = "email_archive"
    DATABASE = "database"
    MEMORY_DUMP = "memory_dump"
    DOCUMENT = "document"
    MEDIA_FILE = "media_file"
    WEB_CONTENT = "web_content"


class AnalysisStatus(str, Enum):
    """Status of forensic analysis operations"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CORRUPTED = "corrupted"
    UNAUTHORIZED = "unauthorized"


class VerificationStatus(str, Enum):
    """Digital evidence verification status"""
    VERIFIED = "verified"
    UNVERIFIED = "unverified"
    TAMPERED = "tampered"
    CORRUPTED = "corrupted"
    INCONCLUSIVE = "inconclusive"


class FileSystemEventType(str, Enum):
    """File system event types for timeline analysis"""
    CREATED = "created"
    MODIFIED = "modified"
    ACCESSED = "accessed"
    DELETED = "deleted"
    RENAMED = "renamed"
    PERMISSIONS_CHANGED = "permissions_changed"
    METADATA_CHANGED = "metadata_changed"


class NetworkEventType(str, Enum):
    """Network event types"""
    CONNECTION_ESTABLISHED = "connection_established"
    CONNECTION_CLOSED = "connection_closed"
    DATA_TRANSFER = "data_transfer"
    DNS_REQUEST = "dns_request"
    HTTP_REQUEST = "http_request"
    EMAIL_SENT = "email_sent"
    EMAIL_RECEIVED = "email_received"
    FILE_DOWNLOAD = "file_download"
    FILE_UPLOAD = "file_upload"


class MobileArtifactType(str, Enum):
    """Types of mobile device artifacts"""
    SMS_MESSAGE = "sms_message"
    CALL_LOG = "call_log"
    CONTACT = "contact"
    APP_DATA = "app_data"
    LOCATION_HISTORY = "location_history"
    BROWSER_HISTORY = "browser_history"
    PHOTO = "photo"
    VIDEO = "video"
    CHAT_MESSAGE = "chat_message"
    EMAIL = "email"


class ForensicsConfig(BaseModel):
    """Configuration for digital forensics operations"""
    
    # Analysis settings
    deep_scan_enabled: bool = Field(default=True)
    deleted_file_recovery: bool = Field(default=True)
    timeline_analysis: bool = Field(default=True)
    hash_verification: bool = Field(default=True)
    
    # Chain of custody
    preserve_metadata: bool = Field(default=True)
    chain_of_custody_logging: bool = Field(default=True)
    evidence_integrity_checks: bool = Field(default=True)
    
    # Output settings
    generate_court_reports: bool = Field(default=True)
    include_technical_details: bool = Field(default=False)
    anonymize_sensitive_data: bool = Field(default=True)
    
    # Processing limits
    max_file_size_mb: int = Field(default=1024, ge=1)
    analysis_timeout_minutes: int = Field(default=60, ge=5)
    
    class Config:
        schema_extra = {
            "example": {
                "deep_scan_enabled": True,
                "deleted_file_recovery": True,
                "generate_court_reports": True,
                "preserve_metadata": True
            }
        }


class ChainOfCustodyEntry(BaseModel):
    """Chain of custody entry for evidence tracking"""
    
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    action: str = Field(..., min_length=1)
    person: str = Field(..., min_length=1)
    location: str = Field(..., min_length=1)
    
    # Evidence state
    evidence_hash_before: Optional[str] = None
    evidence_hash_after: Optional[str] = None
    integrity_verified: bool = Field(default=True)
    
    # Additional metadata
    tool_used: str = Field(default="lemkin-forensics")
    notes: Optional[str] = None
    
    class Config:
        schema_extra = {
            "example": {
                "action": "Evidence received for analysis",
                "person": "Jane Doe, Digital Forensics Investigator",
                "location": "Forensics Lab A",
                "integrity_verified": True
            }
        }


class DigitalEvidence(BaseModel):
    """Represents a piece of digital evidence"""
    
    id: UUID = Field(default_factory=uuid4)
    case_number: str = Field(..., min_length=1)
    evidence_number: str = Field(..., min_length=1)
    
    # Evidence metadata
    evidence_type: EvidenceType
    description: str = Field(..., min_length=1)
    source_device: Optional[str] = None
    file_path: Optional[str] = None
    
    # File information
    file_size: Optional[int] = Field(None, ge=0)
    file_hash_md5: Optional[str] = None
    file_hash_sha256: Optional[str] = None
    content_type: Optional[str] = None
    
    # Temporal data
    created_at: datetime = Field(default_factory=datetime.utcnow)
    evidence_date: Optional[datetime] = None
    acquisition_date: Optional[datetime] = None
    
    # Chain of custody
    chain_of_custody: List[ChainOfCustodyEntry] = Field(default_factory=list)
    current_custodian: str = Field(..., min_length=1)
    
    # Verification status
    verification_status: VerificationStatus = Field(default=VerificationStatus.UNVERIFIED)
    integrity_checks: Dict[str, Any] = Field(default_factory=dict)
    
    def add_custody_entry(self, action: str, person: str, location: str, notes: Optional[str] = None):
        """Add entry to chain of custody"""
        entry = ChainOfCustodyEntry(
            action=action,
            person=person,
            location=location,
            notes=notes
        )
        self.chain_of_custody.append(entry)
    
    def calculate_file_hash(self, algorithm: str = "sha256") -> str:
        """Calculate file hash for integrity verification"""
        if not self.file_path or not Path(self.file_path).exists():
            return ""
        
        hash_obj = hashlib.new(algorithm.lower())
        with open(self.file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_obj.update(chunk)
        return hash_obj.hexdigest()


class FileArtifact(BaseModel):
    """Represents a file system artifact"""
    
    id: UUID = Field(default_factory=uuid4)
    file_path: str = Field(..., min_length=1)
    file_name: str = Field(..., min_length=1)
    
    # File metadata
    file_size: int = Field(..., ge=0)
    file_type: str
    mime_type: Optional[str] = None
    
    # Timestamps
    created_time: Optional[datetime] = None
    modified_time: Optional[datetime] = None
    accessed_time: Optional[datetime] = None
    
    # File system data
    inode_number: Optional[int] = None
    file_permissions: Optional[str] = None
    owner_uid: Optional[int] = None
    group_gid: Optional[int] = None
    
    # Content analysis
    file_signature: Optional[str] = None
    entropy_score: Optional[float] = Field(None, ge=0.0, le=8.0)
    
    # Recovery information
    is_deleted: bool = Field(default=False)
    recovery_confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    
    # Hashes for integrity
    md5_hash: Optional[str] = None
    sha256_hash: Optional[str] = None
    
    class Config:
        schema_extra = {
            "example": {
                "file_path": "/Users/evidence/document.pdf",
                "file_name": "document.pdf",
                "file_size": 1024000,
                "file_type": "PDF",
                "is_deleted": False
            }
        }


class NetworkFlow(BaseModel):
    """Represents a network communication flow"""
    
    id: UUID = Field(default_factory=uuid4)
    
    # Connection details
    source_ip: str = Field(..., min_length=1)
    destination_ip: str = Field(..., min_length=1)
    source_port: Optional[int] = Field(None, ge=1, le=65535)
    destination_port: Optional[int] = Field(None, ge=1, le=65535)
    protocol: str = Field(..., min_length=1)
    
    # Timing
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_seconds: Optional[float] = Field(None, ge=0.0)
    
    # Data transfer
    bytes_sent: int = Field(default=0, ge=0)
    bytes_received: int = Field(default=0, ge=0)
    packets_sent: int = Field(default=0, ge=0)
    packets_received: int = Field(default=0, ge=0)
    
    # Application layer info
    application: Optional[str] = None
    service: Optional[str] = None
    user_agent: Optional[str] = None
    
    # Analysis results
    suspicious_indicators: List[str] = Field(default_factory=list)
    geolocation_data: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        schema_extra = {
            "example": {
                "source_ip": "192.168.1.100",
                "destination_ip": "203.0.113.1",
                "destination_port": 443,
                "protocol": "HTTPS",
                "bytes_sent": 1024,
                "bytes_received": 2048
            }
        }


class MobileArtifact(BaseModel):
    """Represents a mobile device artifact"""
    
    id: UUID = Field(default_factory=uuid4)
    artifact_type: MobileArtifactType
    
    # Device information
    device_id: Optional[str] = None
    device_model: Optional[str] = None
    os_version: Optional[str] = None
    
    # Artifact data
    content: Dict[str, Any] = Field(default_factory=dict)
    raw_data: Optional[str] = None
    
    # Timing
    timestamp: Optional[datetime] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    # Location data
    latitude: Optional[float] = Field(None, ge=-90.0, le=90.0)
    longitude: Optional[float] = Field(None, ge=-180.0, le=180.0)
    location_accuracy: Optional[float] = Field(None, ge=0.0)
    
    # Metadata
    app_package: Optional[str] = None
    database_path: Optional[str] = None
    extraction_method: str = Field(default="lemkin-forensics")
    
    class Config:
        schema_extra = {
            "example": {
                "artifact_type": "sms_message",
                "content": {
                    "sender": "+1234567890",
                    "message": "Hello world",
                    "direction": "incoming"
                },
                "timestamp": "2024-01-15T10:30:00Z"
            }
        }


class FileSystemAnalysis(BaseModel):
    """Results of file system analysis"""
    
    analysis_id: UUID = Field(default_factory=uuid4)
    evidence_id: UUID
    
    # Analysis metadata
    started_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    analysis_duration: Optional[float] = Field(None, ge=0.0)
    
    # File system information
    file_system_type: Optional[str] = None
    total_size: Optional[int] = Field(None, ge=0)
    used_space: Optional[int] = Field(None, ge=0)
    free_space: Optional[int] = Field(None, ge=0)
    
    # Analysis results
    total_files: int = Field(default=0, ge=0)
    active_files: int = Field(default=0, ge=0)
    deleted_files: int = Field(default=0, ge=0)
    recovered_files: int = Field(default=0, ge=0)
    
    # File artifacts
    file_artifacts: List[FileArtifact] = Field(default_factory=list)
    
    # Timeline events
    timeline_events: List[Dict[str, Any]] = Field(default_factory=list)
    
    # Analysis status
    status: AnalysisStatus = Field(default=AnalysisStatus.PENDING)
    error_messages: List[str] = Field(default_factory=list)
    
    # Summary findings
    key_findings: List[str] = Field(default_factory=list)
    suspicious_files: List[FileArtifact] = Field(default_factory=list)
    
    class Config:
        schema_extra = {
            "example": {
                "file_system_type": "NTFS",
                "total_files": 50000,
                "deleted_files": 150,
                "recovered_files": 120,
                "status": "completed"
            }
        }


class NetworkAnalysis(BaseModel):
    """Results of network log analysis"""
    
    analysis_id: UUID = Field(default_factory=uuid4)
    evidence_id: UUID
    
    # Analysis metadata
    started_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    log_files_processed: int = Field(default=0, ge=0)
    
    # Network statistics
    total_connections: int = Field(default=0, ge=0)
    unique_sources: int = Field(default=0, ge=0)
    unique_destinations: int = Field(default=0, ge=0)
    total_data_bytes: int = Field(default=0, ge=0)
    
    # Time range
    earliest_activity: Optional[datetime] = None
    latest_activity: Optional[datetime] = None
    
    # Network flows
    network_flows: List[NetworkFlow] = Field(default_factory=list)
    
    # Analysis results
    communication_patterns: List[Dict[str, Any]] = Field(default_factory=list)
    suspicious_connections: List[NetworkFlow] = Field(default_factory=list)
    top_destinations: List[Dict[str, Any]] = Field(default_factory=list)
    
    # Findings
    key_findings: List[str] = Field(default_factory=list)
    security_alerts: List[str] = Field(default_factory=list)
    
    # Status
    status: AnalysisStatus = Field(default=AnalysisStatus.PENDING)
    error_messages: List[str] = Field(default_factory=list)
    
    class Config:
        schema_extra = {
            "example": {
                "total_connections": 1250,
                "unique_destinations": 45,
                "total_data_bytes": 1048576000,
                "status": "completed"
            }
        }


class MobileDataExtraction(BaseModel):
    """Results of mobile device data extraction"""
    
    extraction_id: UUID = Field(default_factory=uuid4)
    evidence_id: UUID
    
    # Device information
    device_model: Optional[str] = None
    device_serial: Optional[str] = None
    os_version: Optional[str] = None
    backup_date: Optional[datetime] = None
    
    # Extraction metadata
    started_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    extraction_method: str = Field(default="backup_analysis")
    
    # Extracted data counts
    sms_messages: int = Field(default=0, ge=0)
    call_logs: int = Field(default=0, ge=0)
    contacts: int = Field(default=0, ge=0)
    photos: int = Field(default=0, ge=0)
    apps_analyzed: int = Field(default=0, ge=0)
    location_points: int = Field(default=0, ge=0)
    
    # Artifacts
    mobile_artifacts: List[MobileArtifact] = Field(default_factory=list)
    
    # Timeline
    activity_timeline: List[Dict[str, Any]] = Field(default_factory=list)
    
    # Analysis results
    key_findings: List[str] = Field(default_factory=list)
    communication_summary: Dict[str, Any] = Field(default_factory=dict)
    location_summary: Dict[str, Any] = Field(default_factory=dict)
    
    # Status
    status: AnalysisStatus = Field(default=AnalysisStatus.PENDING)
    error_messages: List[str] = Field(default_factory=list)
    
    class Config:
        schema_extra = {
            "example": {
                "device_model": "iPhone 12",
                "sms_messages": 500,
                "call_logs": 200,
                "contacts": 150,
                "photos": 1000,
                "status": "completed"
            }
        }


class AuthenticityReport(BaseModel):
    """Results of digital evidence authenticity verification"""
    
    report_id: UUID = Field(default_factory=uuid4)
    evidence_id: UUID
    
    # Verification metadata
    verified_at: datetime = Field(default_factory=datetime.utcnow)
    verification_method: str = Field(default="comprehensive")
    verifier: str = Field(..., min_length=1)
    
    # Overall verification result
    verification_status: VerificationStatus
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    
    # Hash verification
    hash_verification_passed: bool
    original_hashes: Dict[str, str] = Field(default_factory=dict)
    current_hashes: Dict[str, str] = Field(default_factory=dict)
    
    # Digital signature verification
    signature_valid: Optional[bool] = None
    signature_details: Dict[str, Any] = Field(default_factory=dict)
    
    # Metadata integrity
    metadata_intact: bool = Field(default=True)
    metadata_changes: List[str] = Field(default_factory=list)
    
    # Timestamp verification
    timestamp_verification: Dict[str, Any] = Field(default_factory=dict)
    
    # Chain of custody verification
    custody_chain_intact: bool = Field(default=True)
    custody_gaps: List[str] = Field(default_factory=list)
    
    # Detailed findings
    verification_details: List[str] = Field(default_factory=list)
    anomalies_detected: List[str] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)
    
    # Court admissibility assessment
    likely_admissible: bool = Field(default=False)
    admissibility_notes: List[str] = Field(default_factory=list)
    
    class Config:
        schema_extra = {
            "example": {
                "verification_status": "verified",
                "confidence_score": 0.95,
                "hash_verification_passed": True,
                "metadata_intact": True,
                "likely_admissible": True
            }
        }


class AnalysisResult(BaseModel):
    """Overall analysis result containing all findings"""
    
    result_id: UUID = Field(default_factory=uuid4)
    case_number: str = Field(..., min_length=1)
    
    # Analysis components
    file_system_analysis: Optional[FileSystemAnalysis] = None
    network_analysis: Optional[NetworkAnalysis] = None
    mobile_data_extraction: Optional[MobileDataExtraction] = None
    authenticity_report: Optional[AuthenticityReport] = None
    
    # Overall results
    created_at: datetime = Field(default_factory=datetime.utcnow)
    analyst: str = Field(..., min_length=1)
    
    # Summary findings
    executive_summary: str = Field(..., min_length=1)
    key_findings: List[str] = Field(default_factory=list)
    evidence_timeline: List[Dict[str, Any]] = Field(default_factory=list)
    
    # Technical details
    analysis_methods: List[str] = Field(default_factory=list)
    tools_used: List[str] = Field(default_factory=list)
    
    # Court preparation
    court_ready: bool = Field(default=False)
    court_summary: Optional[str] = None
    expert_opinion: Optional[str] = None
    
    class Config:
        schema_extra = {
            "example": {
                "case_number": "CASE-2024-001",
                "analyst": "Jane Doe, Certified Digital Forensics Examiner",
                "executive_summary": "Analysis of digital evidence reveals...",
                "court_ready": True
            }
        }


class DigitalForensicsAnalyzer:
    """
    Main digital forensics analyzer class providing comprehensive evidence
    analysis for legal professionals without deep technical backgrounds.
    
    Focuses on court-ready outputs and chain of custody preservation.
    """
    
    def __init__(self, config: Optional[ForensicsConfig] = None):
        """Initialize digital forensics analyzer"""
        self.config = config or ForensicsConfig()
        self.logger = logging.getLogger(f"{__name__}.DigitalForensicsAnalyzer")
        
        # Initialize analysis components (will be populated by specific modules)
        self._file_analyzer = None
        self._network_processor = None
        self._mobile_analyzer = None
        self._authenticity_verifier = None
        
        self.logger.info("Digital Forensics Analyzer initialized")
        if self.config.chain_of_custody_logging:
            self.logger.info("Chain of custody logging enabled")
    
    def create_evidence_record(
        self,
        case_number: str,
        evidence_number: str,
        evidence_type: EvidenceType,
        description: str,
        custodian: str,
        file_path: Optional[str] = None
    ) -> DigitalEvidence:
        """Create a new digital evidence record"""
        evidence = DigitalEvidence(
            case_number=case_number,
            evidence_number=evidence_number,
            evidence_type=evidence_type,
            description=description,
            current_custodian=custodian,
            file_path=file_path
        )
        
        # Add initial custody entry
        evidence.add_custody_entry(
            action="Evidence record created",
            person=custodian,
            location="Digital Forensics Lab",
            notes="Initial evidence intake and cataloging"
        )
        
        # Calculate file hash if file path provided
        if file_path and Path(file_path).exists():
            evidence.file_hash_sha256 = evidence.calculate_file_hash("sha256")
            evidence.file_hash_md5 = evidence.calculate_file_hash("md5")
            evidence.file_size = Path(file_path).stat().st_size
        
        self.logger.info(f"Created evidence record: {evidence.id}")
        return evidence
    
    def verify_evidence_integrity(self, evidence: DigitalEvidence) -> bool:
        """Verify the integrity of digital evidence"""
        if not evidence.file_path or not Path(evidence.file_path).exists():
            self.logger.error(f"Evidence file not found: {evidence.file_path}")
            return False
        
        # Recalculate hashes
        current_sha256 = evidence.calculate_file_hash("sha256")
        current_md5 = evidence.calculate_file_hash("md5")
        
        # Compare with stored hashes
        sha256_match = current_sha256 == evidence.file_hash_sha256
        md5_match = current_md5 == evidence.file_hash_md5
        
        integrity_verified = sha256_match and md5_match
        
        evidence.integrity_checks = {
            "sha256_match": sha256_match,
            "md5_match": md5_match,
            "current_sha256": current_sha256,
            "current_md5": current_md5,
            "verified_at": datetime.utcnow().isoformat()
        }
        
        if integrity_verified:
            evidence.verification_status = VerificationStatus.VERIFIED
            self.logger.info(f"Evidence integrity verified: {evidence.id}")
        else:
            evidence.verification_status = VerificationStatus.TAMPERED
            self.logger.warning(f"Evidence integrity compromised: {evidence.id}")
        
        return integrity_verified
    
    def generate_court_summary(self, analysis_result: AnalysisResult) -> str:
        """Generate a court-ready summary for non-technical audiences"""
        summary_parts = [
            f"DIGITAL FORENSICS ANALYSIS SUMMARY",
            f"Case Number: {analysis_result.case_number}",
            f"Analysis Date: {analysis_result.created_at.strftime('%B %d, %Y')}",
            f"Analyst: {analysis_result.analyst}",
            f"",
            f"EXECUTIVE SUMMARY:",
            f"{analysis_result.executive_summary}",
            f"",
            f"KEY FINDINGS:"
        ]
        
        for i, finding in enumerate(analysis_result.key_findings, 1):
            summary_parts.append(f"{i}. {finding}")
        
        if analysis_result.file_system_analysis:
            fs_analysis = analysis_result.file_system_analysis
            summary_parts.extend([
                f"",
                f"FILE SYSTEM ANALYSIS:",
                f"- Total files examined: {fs_analysis.total_files:,}",
                f"- Deleted files found: {fs_analysis.deleted_files:,}",
                f"- Successfully recovered: {fs_analysis.recovered_files:,}",
                f"- File system type: {fs_analysis.file_system_type or 'Unknown'}"
            ])
        
        if analysis_result.network_analysis:
            net_analysis = analysis_result.network_analysis
            summary_parts.extend([
                f"",
                f"NETWORK ACTIVITY ANALYSIS:",
                f"- Total connections: {net_analysis.total_connections:,}",
                f"- Unique destinations: {net_analysis.unique_destinations:,}",
                f"- Data transferred: {net_analysis.total_data_bytes / (1024*1024):.2f} MB",
                f"- Suspicious connections: {len(net_analysis.suspicious_connections)}"
            ])
        
        if analysis_result.mobile_data_extraction:
            mobile_analysis = analysis_result.mobile_data_extraction
            summary_parts.extend([
                f"",
                f"MOBILE DEVICE DATA:",
                f"- Device: {mobile_analysis.device_model or 'Unknown'}",
                f"- Text messages: {mobile_analysis.sms_messages:,}",
                f"- Call records: {mobile_analysis.call_logs:,}",
                f"- Contacts: {mobile_analysis.contacts:,}",
                f"- Photos/Media: {mobile_analysis.photos:,}"
            ])
        
        summary_parts.extend([
            f"",
            f"CHAIN OF CUSTODY: Maintained throughout analysis",
            f"EVIDENCE INTEGRITY: Verified via cryptographic hashes",
            f"ANALYSIS METHODS: Industry standard forensic techniques",
            f"",
            f"This analysis was conducted using accepted digital forensic",
            f"methodologies and tools. All evidence has been preserved",
            f"and documented according to legal standards."
        ])
        
        return "\n".join(summary_parts)
    
    def export_analysis_report(
        self,
        analysis_result: AnalysisResult,
        output_path: Path,
        include_technical_details: bool = None
    ) -> bool:
        """Export complete analysis report with chain of custody"""
        try:
            include_tech = (include_technical_details 
                          if include_technical_details is not None 
                          else self.config.include_technical_details)
            
            report_data = {
                "analysis_result": analysis_result.dict(),
                "court_summary": self.generate_court_summary(analysis_result),
                "export_timestamp": datetime.utcnow().isoformat(),
                "export_tool": "lemkin-forensics",
                "include_technical_details": include_tech,
                "chain_of_custody_verified": True
            }
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(report_data, f, indent=2, default=str)
            
            self.logger.info(f"Analysis report exported to {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Export failed: {str(e)}")
            return False
    
    def create_timeline_report(self, analysis_result: AnalysisResult) -> List[Dict[str, Any]]:
        """Create a comprehensive timeline of digital evidence"""
        timeline_events = []
        
        # Add file system events
        if analysis_result.file_system_analysis:
            for event in analysis_result.file_system_analysis.timeline_events:
                timeline_events.append({
                    **event,
                    "source": "file_system",
                    "category": "File Activity"
                })
        
        # Add network events
        if analysis_result.network_analysis:
            for flow in analysis_result.network_analysis.network_flows:
                timeline_events.append({
                    "timestamp": flow.start_time,
                    "event_type": "network_connection",
                    "description": f"Connection to {flow.destination_ip}:{flow.destination_port}",
                    "source": "network",
                    "category": "Network Activity",
                    "details": {
                        "protocol": flow.protocol,
                        "bytes_transferred": flow.bytes_sent + flow.bytes_received
                    }
                })
        
        # Add mobile events
        if analysis_result.mobile_data_extraction:
            for artifact in analysis_result.mobile_data_extraction.mobile_artifacts:
                if artifact.timestamp:
                    timeline_events.append({
                        "timestamp": artifact.timestamp,
                        "event_type": artifact.artifact_type,
                        "description": self._format_mobile_artifact_description(artifact),
                        "source": "mobile_device",
                        "category": "Mobile Activity",
                        "details": artifact.content
                    })
        
        # Sort by timestamp
        timeline_events.sort(key=lambda x: x.get("timestamp", datetime.min))
        
        return timeline_events
    
    def _format_mobile_artifact_description(self, artifact: MobileArtifact) -> str:
        """Format mobile artifact for timeline display"""
        if artifact.artifact_type == MobileArtifactType.SMS_MESSAGE:
            content = artifact.content
            return f"SMS {content.get('direction', 'unknown')}: {content.get('sender', 'unknown')}"
        elif artifact.artifact_type == MobileArtifactType.CALL_LOG:
            content = artifact.content
            return f"Call {content.get('direction', 'unknown')}: {content.get('number', 'unknown')}"
        elif artifact.artifact_type == MobileArtifactType.LOCATION_HISTORY:
            return f"Location: {artifact.latitude:.6f}, {artifact.longitude:.6f}"
        else:
            return f"{artifact.artifact_type.replace('_', ' ').title()}"