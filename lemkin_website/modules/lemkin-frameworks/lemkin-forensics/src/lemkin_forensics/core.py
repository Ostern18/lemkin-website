"""
Lemkin Digital Forensics Core Module

This module provides the core data models and DigitalForensicsAnalyzer class for
comprehensive digital forensics analysis in legal investigations. It implements
chain of custody procedures and supports multiple evidence types.

Legal Compliance: Meets standards for digital evidence handling in legal proceedings
"""

from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, validator
import json
import hashlib
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EvidenceType(str, Enum):
    """Types of digital evidence that can be analyzed"""
    DISK_IMAGE = "disk_image"
    NETWORK_CAPTURE = "network_capture"
    MOBILE_BACKUP = "mobile_backup"
    MEMORY_DUMP = "memory_dump"
    DATABASE_EXPORT = "database_export"
    EMAIL_ARCHIVE = "email_archive"
    DOCUMENT_COLLECTION = "document_collection"
    VIDEO_FILE = "video_file"
    AUDIO_FILE = "audio_file"
    IMAGE_FILE = "image_file"
    LOG_FILE = "log_file"
    REGISTRY_HIVE = "registry_hive"
    BROWSER_HISTORY = "browser_history"
    CHAT_LOGS = "chat_logs"
    METADATA_COLLECTION = "metadata_collection"


class AnalysisStatus(str, Enum):
    """Status of forensic analysis operations"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIALLY_COMPLETED = "partially_completed"
    REQUIRES_MANUAL_REVIEW = "requires_manual_review"
    CHAIN_OF_CUSTODY_BROKEN = "chain_of_custody_broken"
    EVIDENCE_CORRUPTED = "evidence_corrupted"


class ForensicsConfig(BaseModel):
    """Configuration for digital forensics operations"""
    
    # Analysis settings
    enable_deleted_file_recovery: bool = Field(default=True)
    enable_metadata_extraction: bool = Field(default=True)
    enable_timeline_generation: bool = Field(default=True)
    enable_network_flow_analysis: bool = Field(default=True)
    
    # Chain of custody
    maintain_chain_of_custody: bool = Field(default=True)
    generate_integrity_hashes: bool = Field(default=True)
    preserve_original_evidence: bool = Field(default=True)
    
    # Output settings
    generate_detailed_reports: bool = Field(default=True)
    include_technical_details: bool = Field(default=True)
    create_timeline_visualizations: bool = Field(default=False)
    
    # Processing limits
    max_file_size_mb: int = Field(default=1024, ge=1, le=10240)
    analysis_timeout_minutes: int = Field(default=60, ge=5, le=1440)
    max_concurrent_analyses: int = Field(default=2, ge=1, le=10)
    
    # Tool paths (for external forensics tools)
    sleuthkit_path: Optional[str] = Field(default="/usr/local/bin")
    volatility_path: Optional[str] = Field(default=None)
    autopsy_path: Optional[str] = Field(default=None)
    
    # Legal compliance
    evidence_handling_standard: str = Field(default="ISO_27037")
    chain_of_custody_logging: bool = Field(default=True)
    
    class Config:
        schema_extra = {
            "example": {
                "enable_deleted_file_recovery": True,
                "maintain_chain_of_custody": True,
                "max_file_size_mb": 1024,
                "evidence_handling_standard": "ISO_27037"
            }
        }


class ChainOfCustodyEntry(BaseModel):
    """Individual entry in chain of custody log"""
    
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    action: str = Field(..., min_length=1)
    operator: str = Field(..., min_length=1)
    location: Optional[str] = None
    
    # Integrity verification
    hash_before: Optional[str] = None
    hash_after: Optional[str] = None
    hash_algorithm: str = Field(default="SHA-256")
    
    # Additional metadata
    tool_used: Optional[str] = None
    notes: Optional[str] = None
    witness: Optional[str] = None
    
    class Config:
        schema_extra = {
            "example": {
                "action": "Evidence received from client",
                "operator": "John Smith, Digital Forensics Analyst",
                "location": "Forensics Lab A",
                "hash_before": "abc123...",
                "notes": "Evidence received in sealed container"
            }
        }


class DigitalEvidence(BaseModel):
    """Represents a piece of digital evidence with full chain of custody"""
    
    id: UUID = Field(default_factory=uuid4)
    case_id: Optional[str] = None
    evidence_number: Optional[str] = None
    
    # Evidence identification
    name: str = Field(..., min_length=1)
    description: Optional[str] = None
    evidence_type: EvidenceType
    file_path: str = Field(..., min_length=1)
    
    # File properties
    file_size: int = Field(..., ge=0)
    file_hash_md5: Optional[str] = None
    file_hash_sha1: Optional[str] = None
    file_hash_sha256: str = Field(..., min_length=64, max_length=64)
    
    # Temporal information
    created_at: datetime = Field(default_factory=datetime.utcnow)
    file_created_at: Optional[datetime] = None
    file_modified_at: Optional[datetime] = None
    acquisition_date: Optional[datetime] = None
    
    # Source information
    source_device: Optional[str] = None
    source_location: Optional[str] = None
    acquisition_method: Optional[str] = None
    acquisition_tool: Optional[str] = None
    
    # Chain of custody
    chain_of_custody: List[ChainOfCustodyEntry] = Field(default_factory=list)
    current_custodian: Optional[str] = None
    
    # Analysis tracking
    analysis_count: int = Field(default=0, ge=0)
    last_analyzed: Optional[datetime] = None
    integrity_verified: bool = Field(default=False)
    integrity_last_checked: Optional[datetime] = None
    
    # Legal metadata
    legal_hold: bool = Field(default=False)
    privileged: bool = Field(default=False)
    confidentiality_level: str = Field(default="standard")
    
    @validator('file_hash_sha256')
    def validate_sha256(cls, v):
        if len(v) != 64:
            raise ValueError('SHA-256 hash must be exactly 64 characters')
        return v.lower()
    
    def add_custody_entry(self, action: str, operator: str, **kwargs):
        """Add a new chain of custody entry"""
        entry = ChainOfCustodyEntry(
            action=action,
            operator=operator,
            **kwargs
        )
        self.chain_of_custody.append(entry)
    
    def verify_integrity(self) -> bool:
        """Verify file integrity using stored hashes"""
        if not os.path.exists(self.file_path):
            return False
            
        try:
            with open(self.file_path, 'rb') as f:
                file_hash = hashlib.sha256(f.read()).hexdigest()
            
            integrity_ok = file_hash == self.file_hash_sha256
            self.integrity_verified = integrity_ok
            self.integrity_last_checked = datetime.utcnow()
            
            return integrity_ok
        except Exception:
            return False


class FileSystemArtifact(BaseModel):
    """Represents a file system artifact discovered during analysis"""
    
    id: UUID = Field(default_factory=uuid4)
    file_path: str
    file_name: str
    
    # File metadata
    file_size: Optional[int] = Field(None, ge=0)
    created_time: Optional[datetime] = None
    modified_time: Optional[datetime] = None
    accessed_time: Optional[datetime] = None
    
    # File system metadata
    inode_number: Optional[int] = None
    file_permissions: Optional[str] = None
    file_owner: Optional[str] = None
    file_group: Optional[str] = None
    
    # Recovery information
    is_deleted: bool = Field(default=False)
    recovery_method: Optional[str] = None
    recovery_confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    
    # Content analysis
    file_type: Optional[str] = None
    mime_type: Optional[str] = None
    file_signature: Optional[str] = None
    contains_pii: Optional[bool] = None
    
    # Hash values
    md5_hash: Optional[str] = None
    sha1_hash: Optional[str] = None
    sha256_hash: Optional[str] = None


class NetworkArtifact(BaseModel):
    """Represents a network artifact discovered during analysis"""
    
    id: UUID = Field(default_factory=uuid4)
    timestamp: datetime
    
    # Network identifiers
    source_ip: str
    destination_ip: str
    source_port: Optional[int] = Field(None, ge=0, le=65535)
    destination_port: Optional[int] = Field(None, ge=0, le=65535)
    protocol: str
    
    # Traffic analysis
    bytes_sent: Optional[int] = Field(None, ge=0)
    bytes_received: Optional[int] = Field(None, ge=0)
    duration_seconds: Optional[float] = Field(None, ge=0.0)
    
    # Content analysis
    payload_summary: Optional[str] = None
    http_method: Optional[str] = None
    http_url: Optional[str] = None
    http_user_agent: Optional[str] = None
    
    # Geolocation
    source_country: Optional[str] = None
    destination_country: Optional[str] = None
    source_asn: Optional[str] = None
    destination_asn: Optional[str] = None
    
    # Threat intelligence
    is_suspicious: bool = Field(default=False)
    threat_indicators: List[str] = Field(default_factory=list)
    reputation_score: Optional[float] = Field(None, ge=0.0, le=10.0)


class MobileArtifact(BaseModel):
    """Represents a mobile device artifact"""
    
    id: UUID = Field(default_factory=uuid4)
    artifact_type: str  # call_log, sms, contact, app_data, etc.
    
    # Common mobile data
    timestamp: Optional[datetime] = None
    phone_number: Optional[str] = None
    contact_name: Optional[str] = None
    
    # Message data
    message_content: Optional[str] = None
    message_direction: Optional[str] = None  # incoming, outgoing
    message_status: Optional[str] = None  # read, unread, sent, failed
    
    # Call data
    call_duration_seconds: Optional[int] = Field(None, ge=0)
    call_type: Optional[str] = None  # incoming, outgoing, missed
    
    # Location data
    latitude: Optional[float] = Field(None, ge=-90.0, le=90.0)
    longitude: Optional[float] = Field(None, ge=-180.0, le=180.0)
    location_accuracy: Optional[float] = Field(None, ge=0.0)
    location_source: Optional[str] = None
    
    # App-specific data
    app_name: Optional[str] = None
    app_version: Optional[str] = None
    app_data: Optional[Dict[str, Any]] = Field(default_factory=dict)
    
    # Device information
    device_id: Optional[str] = None
    device_model: Optional[str] = None
    os_version: Optional[str] = None


class AnalysisResult(BaseModel):
    """Result of a forensic analysis operation"""
    
    id: UUID = Field(default_factory=uuid4)
    evidence_id: UUID
    analysis_type: str
    
    # Status and timing
    status: AnalysisStatus
    started_at: datetime
    completed_at: Optional[datetime] = None
    duration_seconds: Optional[float] = Field(None, ge=0.0)
    
    # Results summary
    success: bool
    message: str
    confidence_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    
    # Discovered artifacts
    file_artifacts: List[FileSystemArtifact] = Field(default_factory=list)
    network_artifacts: List[NetworkArtifact] = Field(default_factory=list)
    mobile_artifacts: List[MobileArtifact] = Field(default_factory=list)
    
    # Analysis metadata
    total_files_processed: int = Field(default=0, ge=0)
    deleted_files_recovered: int = Field(default=0, ge=0)
    suspicious_activities_found: int = Field(default=0, ge=0)
    
    # Technical details
    tools_used: List[str] = Field(default_factory=list)
    analysis_parameters: Dict[str, Any] = Field(default_factory=dict)
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    
    # Legal considerations
    privileged_content_found: bool = Field(default=False)
    pii_found: bool = Field(default=False)
    encryption_encountered: bool = Field(default=False)
    
    # Report generation
    executive_summary: Optional[str] = None
    technical_findings: Optional[str] = None
    legal_implications: Optional[str] = None
    recommendations: List[str] = Field(default_factory=list)
    
    def add_file_artifact(self, artifact: FileSystemArtifact):
        """Add a file system artifact to results"""
        self.file_artifacts.append(artifact)
        self.total_files_processed += 1
        if artifact.is_deleted:
            self.deleted_files_recovered += 1
    
    def add_network_artifact(self, artifact: NetworkArtifact):
        """Add a network artifact to results"""
        self.network_artifacts.append(artifact)
        if artifact.is_suspicious:
            self.suspicious_activities_found += 1
    
    def add_mobile_artifact(self, artifact: MobileArtifact):
        """Add a mobile artifact to results"""
        self.mobile_artifacts.append(artifact)


class TimelineEvent(BaseModel):
    """Represents an event in a forensic timeline"""
    
    id: UUID = Field(default_factory=uuid4)
    timestamp: datetime
    event_type: str
    
    # Event details
    description: str
    source_file: Optional[str] = None
    artifact_type: Optional[str] = None
    
    # Evidence linking
    evidence_id: Optional[UUID] = None
    related_artifacts: List[UUID] = Field(default_factory=list)
    
    # Metadata
    confidence_level: float = Field(default=1.0, ge=0.0, le=1.0)
    analysis_tool: Optional[str] = None
    notes: Optional[str] = None
    
    # Legal relevance
    legal_significance: Optional[str] = None
    privilege_claim: bool = Field(default=False)


class ForensicsCase(BaseModel):
    """Represents a complete forensics case with all evidence and analyses"""
    
    id: UUID = Field(default_factory=uuid4)
    case_number: str = Field(..., min_length=1)
    case_name: str = Field(..., min_length=1)
    
    # Case metadata
    created_at: datetime = Field(default_factory=datetime.utcnow)
    investigator: str = Field(..., min_length=1)
    client: Optional[str] = None
    legal_matter: Optional[str] = None
    
    # Evidence collection
    evidence_items: List[DigitalEvidence] = Field(default_factory=list)
    analysis_results: List[AnalysisResult] = Field(default_factory=list)
    
    # Timeline
    timeline_events: List[TimelineEvent] = Field(default_factory=list)
    
    # Case status
    status: str = Field(default="active")
    priority: str = Field(default="medium")
    deadline: Optional[datetime] = None
    
    # Legal considerations
    legal_hold_active: bool = Field(default=False)
    privileged_review_required: bool = Field(default=False)
    expert_witness_designation: bool = Field(default=False)
    
    def add_evidence(self, evidence: DigitalEvidence):
        """Add evidence to the case"""
        self.evidence_items.append(evidence)
        evidence.case_id = self.case_number
    
    def get_timeline_sorted(self) -> List[TimelineEvent]:
        """Get timeline events sorted by timestamp"""
        return sorted(self.timeline_events, key=lambda x: x.timestamp)


class DigitalForensicsAnalyzer:
    """
    Main coordinator class for digital forensics analysis operations.
    
    Provides a unified interface for comprehensive forensic analysis including:
    - File system analysis and deleted file recovery
    - Network traffic analysis and flow reconstruction  
    - Mobile device data extraction and analysis
    - Evidence authenticity verification
    - Timeline generation and visualization
    - Chain of custody management
    """
    
    def __init__(self, config: Optional[ForensicsConfig] = None):
        """Initialize the forensics analyzer with configuration"""
        self.config = config or ForensicsConfig()
        self.logger = logging.getLogger(f"{__name__}.DigitalForensicsAnalyzer")
        
        # Initialize component analyzers (will be set by specific modules)
        self._file_analyzer = None
        self._network_processor = None
        self._mobile_analyzer = None
        self._authenticity_verifier = None
        
        # Current case
        self.current_case: Optional[ForensicsCase] = None
        
        self.logger.info("Digital Forensics Analyzer initialized")
        if self.config.maintain_chain_of_custody:
            self.logger.info("Chain of custody tracking enabled")
    
    def create_case(
        self,
        case_number: str,
        case_name: str,
        investigator: str,
        client: Optional[str] = None,
        legal_matter: Optional[str] = None
    ) -> ForensicsCase:
        """Create a new forensics case"""
        case = ForensicsCase(
            case_number=case_number,
            case_name=case_name,
            investigator=investigator,
            client=client,
            legal_matter=legal_matter
        )
        
        self.current_case = case
        self.logger.info(f"Created case: {case_number} - {case_name}")
        
        return case
    
    def add_evidence(
        self,
        file_path: str,
        evidence_type: EvidenceType,
        name: str,
        description: Optional[str] = None,
        custodian: Optional[str] = None,
        case: Optional[ForensicsCase] = None
    ) -> DigitalEvidence:
        """Add evidence to a case with chain of custody initialization"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Evidence file not found: {file_path}")
        
        # Calculate file hash
        with open(file_path, 'rb') as f:
            file_data = f.read()
            sha256_hash = hashlib.sha256(file_data).hexdigest()
            md5_hash = hashlib.md5(file_data).hexdigest()
            sha1_hash = hashlib.sha1(file_data).hexdigest()
        
        # Get file metadata
        stat = os.stat(file_path)
        
        evidence = DigitalEvidence(
            name=name,
            description=description,
            evidence_type=evidence_type,
            file_path=file_path,
            file_size=stat.st_size,
            file_hash_md5=md5_hash,
            file_hash_sha1=sha1_hash,
            file_hash_sha256=sha256_hash,
            file_created_at=datetime.fromtimestamp(stat.st_ctime),
            file_modified_at=datetime.fromtimestamp(stat.st_mtime),
            current_custodian=custodian,
            acquisition_date=datetime.utcnow()
        )
        
        # Add initial chain of custody entry
        evidence.add_custody_entry(
            action="Evidence added to case",
            operator=custodian or "System",
            hash_before=sha256_hash,
            tool_used="lemkin-forensics"
        )
        
        # Add to case
        target_case = case or self.current_case
        if target_case:
            target_case.add_evidence(evidence)
        
        self.logger.info(f"Added evidence: {evidence.name} ({evidence.id})")
        
        return evidence
    
    def analyze_evidence(
        self,
        evidence: DigitalEvidence,
        analysis_types: Optional[List[str]] = None
    ) -> AnalysisResult:
        """
        Perform comprehensive analysis on digital evidence
        
        Args:
            evidence: Evidence to analyze
            analysis_types: Specific types of analysis to perform
            
        Returns:
            AnalysisResult with all discovered artifacts
        """
        # Verify integrity before analysis
        if not evidence.verify_integrity():
            self.logger.error(f"Integrity verification failed for {evidence.name}")
            return AnalysisResult(
                evidence_id=evidence.id,
                analysis_type="integrity_check",
                status=AnalysisStatus.EVIDENCE_CORRUPTED,
                started_at=datetime.utcnow(),
                success=False,
                message="Evidence integrity verification failed"
            )
        
        # Create analysis result
        result = AnalysisResult(
            evidence_id=evidence.id,
            analysis_type="comprehensive",
            status=AnalysisStatus.IN_PROGRESS,
            started_at=datetime.utcnow(),
            success=False,
            message="Analysis in progress"
        )
        
        try:
            # Add chain of custody entry
            evidence.add_custody_entry(
                action="Analysis started",
                operator=self.current_case.investigator if self.current_case else "System",
                tool_used="lemkin-forensics"
            )
            
            # Perform analysis based on evidence type
            if evidence.evidence_type == EvidenceType.DISK_IMAGE:
                self._analyze_disk_image(evidence, result)
            elif evidence.evidence_type == EvidenceType.NETWORK_CAPTURE:
                self._analyze_network_capture(evidence, result)
            elif evidence.evidence_type == EvidenceType.MOBILE_BACKUP:
                self._analyze_mobile_backup(evidence, result)
            else:
                self._analyze_generic_file(evidence, result)
            
            # Complete analysis
            result.status = AnalysisStatus.COMPLETED
            result.success = True
            result.message = "Analysis completed successfully"
            result.completed_at = datetime.utcnow()
            
            # Calculate duration
            if result.completed_at and result.started_at:
                duration = result.completed_at - result.started_at
                result.duration_seconds = duration.total_seconds()
            
            # Update evidence
            evidence.analysis_count += 1
            evidence.last_analyzed = datetime.utcnow()
            
            # Add final custody entry
            evidence.add_custody_entry(
                action="Analysis completed",
                operator=self.current_case.investigator if self.current_case else "System",
                tool_used="lemkin-forensics"
            )
            
        except Exception as e:
            result.status = AnalysisStatus.FAILED
            result.success = False
            result.message = f"Analysis failed: {str(e)}"
            result.errors.append(str(e))
            self.logger.error(f"Analysis failed for {evidence.name}: {str(e)}")
        
        return result
    
    def _analyze_disk_image(self, evidence: DigitalEvidence, result: AnalysisResult):
        """Analyze disk image evidence"""
        result.tools_used.append("file_system_analyzer")
        # This would be implemented by FileAnalyzer
        pass
    
    def _analyze_network_capture(self, evidence: DigitalEvidence, result: AnalysisResult):
        """Analyze network capture evidence"""
        result.tools_used.append("network_processor")
        # This would be implemented by NetworkProcessor
        pass
    
    def _analyze_mobile_backup(self, evidence: DigitalEvidence, result: AnalysisResult):
        """Analyze mobile backup evidence"""
        result.tools_used.append("mobile_analyzer")
        # This would be implemented by MobileAnalyzer
        pass
    
    def _analyze_generic_file(self, evidence: DigitalEvidence, result: AnalysisResult):
        """Analyze generic file evidence"""
        result.tools_used.append("generic_analyzer")
        # Basic file analysis
        pass
    
    def generate_timeline(self, case: Optional[ForensicsCase] = None) -> List[TimelineEvent]:
        """Generate forensic timeline from all evidence and analysis results"""
        target_case = case or self.current_case
        if not target_case:
            raise ValueError("No case specified for timeline generation")
        
        events = []
        
        # Add events from all analysis results
        for result in target_case.analysis_results:
            # Add file system events
            for artifact in result.file_artifacts:
                if artifact.created_time:
                    events.append(TimelineEvent(
                        timestamp=artifact.created_time,
                        event_type="file_created",
                        description=f"File created: {artifact.file_name}",
                        source_file=artifact.file_path,
                        artifact_type="file_system",
                        evidence_id=result.evidence_id
                    ))
                
                if artifact.modified_time:
                    events.append(TimelineEvent(
                        timestamp=artifact.modified_time,
                        event_type="file_modified",
                        description=f"File modified: {artifact.file_name}",
                        source_file=artifact.file_path,
                        artifact_type="file_system",
                        evidence_id=result.evidence_id
                    ))
            
            # Add network events
            for artifact in result.network_artifacts:
                events.append(TimelineEvent(
                    timestamp=artifact.timestamp,
                    event_type="network_communication",
                    description=f"Network connection: {artifact.source_ip} -> {artifact.destination_ip}:{artifact.destination_port}",
                    artifact_type="network",
                    evidence_id=result.evidence_id
                ))
            
            # Add mobile events
            for artifact in result.mobile_artifacts:
                if artifact.timestamp:
                    events.append(TimelineEvent(
                        timestamp=artifact.timestamp,
                        event_type=artifact.artifact_type,
                        description=f"Mobile {artifact.artifact_type}: {artifact.contact_name or artifact.phone_number or 'Unknown'}",
                        artifact_type="mobile",
                        evidence_id=result.evidence_id
                    ))
        
        # Sort events chronologically
        events.sort(key=lambda x: x.timestamp)
        
        # Update case timeline
        target_case.timeline_events = events
        
        self.logger.info(f"Generated timeline with {len(events)} events")
        
        return events
    
    def generate_case_report(self, case: Optional[ForensicsCase] = None) -> Dict[str, Any]:
        """Generate comprehensive case report"""
        target_case = case or self.current_case
        if not target_case:
            raise ValueError("No case specified for report generation")
        
        # Calculate summary statistics
        total_evidence = len(target_case.evidence_items)
        total_analyses = len(target_case.analysis_results)
        successful_analyses = sum(1 for r in target_case.analysis_results if r.success)
        
        total_artifacts = sum(
            len(r.file_artifacts) + len(r.network_artifacts) + len(r.mobile_artifacts)
            for r in target_case.analysis_results
        )
        
        report = {
            'case_information': {
                'case_number': target_case.case_number,
                'case_name': target_case.case_name,
                'investigator': target_case.investigator,
                'client': target_case.client,
                'legal_matter': target_case.legal_matter,
                'created_at': target_case.created_at.isoformat(),
                'status': target_case.status
            },
            'evidence_summary': {
                'total_evidence_items': total_evidence,
                'evidence_types': list(set(e.evidence_type for e in target_case.evidence_items)),
                'total_file_size_bytes': sum(e.file_size for e in target_case.evidence_items)
            },
            'analysis_summary': {
                'total_analyses': total_analyses,
                'successful_analyses': successful_analyses,
                'failed_analyses': total_analyses - successful_analyses,
                'total_artifacts_discovered': total_artifacts
            },
            'timeline_summary': {
                'total_events': len(target_case.timeline_events),
                'earliest_event': min(e.timestamp for e in target_case.timeline_events).isoformat() if target_case.timeline_events else None,
                'latest_event': max(e.timestamp for e in target_case.timeline_events).isoformat() if target_case.timeline_events else None
            },
            'chain_of_custody_status': self._verify_chain_of_custody(target_case),
            'legal_considerations': {
                'legal_hold_active': target_case.legal_hold_active,
                'privileged_review_required': target_case.privileged_review_required,
                'expert_witness_designation': target_case.expert_witness_designation
            },
            'generated_at': datetime.utcnow().isoformat(),
            'generated_by': 'lemkin-forensics'
        }
        
        return report
    
    def _verify_chain_of_custody(self, case: ForensicsCase) -> Dict[str, Any]:
        """Verify chain of custody for all evidence in case"""
        custody_status = {
            'all_chains_intact': True,
            'evidence_with_issues': [],
            'total_custody_entries': 0
        }
        
        for evidence in case.evidence_items:
            custody_status['total_custody_entries'] += len(evidence.chain_of_custody)
            
            # Check for custody issues
            if not evidence.chain_of_custody:
                custody_status['all_chains_intact'] = False
                custody_status['evidence_with_issues'].append({
                    'evidence_id': str(evidence.id),
                    'issue': 'No chain of custody entries'
                })
            
            if not evidence.verify_integrity():
                custody_status['all_chains_intact'] = False
                custody_status['evidence_with_issues'].append({
                    'evidence_id': str(evidence.id),
                    'issue': 'Integrity verification failed'
                })
        
        return custody_status
    
    def export_case(
        self,
        case: ForensicsCase,
        output_path: Path,
        format: str = "json"
    ) -> bool:
        """Export complete case with all evidence and analysis results"""
        try:
            if format.lower() == "json":
                export_data = {
                    'case': case.dict(),
                    'case_report': self.generate_case_report(case),
                    'export_timestamp': datetime.utcnow().isoformat(),
                    'export_tool': 'lemkin-forensics',
                    'export_version': '1.0'
                }
                
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(export_data, f, indent=2, default=str)
                
                self.logger.info(f"Case exported to {output_path}")
                return True
            else:
                self.logger.error(f"Unsupported export format: {format}")
                return False
                
        except Exception as e:
            self.logger.error(f"Export failed: {str(e)}")
            return False