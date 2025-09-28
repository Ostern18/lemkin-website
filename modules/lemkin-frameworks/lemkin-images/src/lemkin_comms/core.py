"""
Lemkin Communication Analysis Suite Core Module

This module provides the core data models and CommunicationAnalyzer class for
comprehensive communication analysis and pattern detection. It implements
multi-platform communication analysis, network mapping, and forensic-grade
evidence handling for legal investigations.

Legal Compliance: Meets standards for digital evidence handling in legal proceedings
"""

from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple, Set
from uuid import UUID, uuid4
import json
import hashlib
import logging
import os

from pydantic import BaseModel, Field, validator
import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CommunicationType(str, Enum):
    """Types of communication that can be analyzed"""
    CHAT = "chat"
    EMAIL = "email"
    SMS = "sms"
    CALL = "call"
    VIDEO_CALL = "video_call"
    VOICE_MESSAGE = "voice_message"
    FILE_TRANSFER = "file_transfer"
    LOCATION_SHARE = "location_share"


class PlatformType(str, Enum):
    """Communication platforms supported for analysis"""
    WHATSAPP = "whatsapp"
    TELEGRAM = "telegram"
    SIGNAL = "signal"
    DISCORD = "discord"
    SLACK = "slack"
    EMAIL = "email"
    SMS = "sms"
    FACEBOOK_MESSENGER = "facebook_messenger"
    INSTAGRAM = "instagram"
    TWITTER = "twitter"
    LINKEDIN = "linkedin"
    SKYPE = "skype"
    TEAMS = "teams"
    ZOOM = "zoom"


class AnalysisStatus(str, Enum):
    """Status of communication analysis operations"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIALLY_COMPLETED = "partially_completed"
    REQUIRES_MANUAL_REVIEW = "requires_manual_review"


class PatternType(str, Enum):
    """Types of communication patterns that can be detected"""
    TEMPORAL = "temporal"
    FREQUENCY = "frequency"
    BURST = "burst"
    REGULAR = "regular"
    ANOMALOUS = "anomalous"
    COORDINATED = "coordinated"
    SUSPICIOUS = "suspicious"
    HIERARCHY = "hierarchy"
    CLUSTER = "cluster"


class AnomalyType(str, Enum):
    """Types of anomalies that can be detected in communications"""
    UNUSUAL_TIME = "unusual_time"
    UNUSUAL_FREQUENCY = "unusual_frequency"
    SUDDEN_SILENCE = "sudden_silence"
    BURST_ACTIVITY = "burst_activity"
    NEW_CONTACT = "new_contact"
    UNUSUAL_CONTENT = "unusual_content"
    PLATFORM_SWITCH = "platform_switch"
    ENCRYPTED_INCREASE = "encrypted_increase"
    DELETION_PATTERN = "deletion_pattern"


class AnalysisLevel(str, Enum):
    """Depth of analysis to perform"""
    BASIC = "basic"
    STANDARD = "standard"
    COMPREHENSIVE = "comprehensive"
    FORENSIC = "forensic"


class Contact(BaseModel):
    """Represents a communication contact/participant"""
    contact_id: str = Field(description="Unique identifier for the contact")
    name: Optional[str] = Field(default=None, description="Display name of contact")
    phone_number: Optional[str] = Field(default=None, description="Phone number")
    email_address: Optional[str] = Field(default=None, description="Email address")
    username: Optional[str] = Field(default=None, description="Platform username")
    platform_id: Optional[str] = Field(default=None, description="Platform-specific ID")
    platforms: List[PlatformType] = Field(default_factory=list, description="Platforms used")
    aliases: List[str] = Field(default_factory=list, description="Known aliases")
    first_seen: Optional[datetime] = Field(default=None, description="First communication")
    last_seen: Optional[datetime] = Field(default=None, description="Last communication")
    total_messages: int = Field(default=0, description="Total messages exchanged")
    
    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


class Communication(BaseModel):
    """Base model for all communication types"""
    message_id: str = Field(description="Unique message identifier")
    platform: PlatformType = Field(description="Communication platform")
    comm_type: CommunicationType = Field(description="Type of communication")
    timestamp: datetime = Field(description="When communication occurred")
    sender_id: str = Field(description="Sender contact ID")
    recipient_ids: List[str] = Field(description="Recipient contact IDs")
    content: Optional[str] = Field(default=None, description="Message content")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Platform metadata")
    attachments: List[str] = Field(default_factory=list, description="File attachments")
    thread_id: Optional[str] = Field(default=None, description="Conversation thread ID")
    replied_to: Optional[str] = Field(default=None, description="Message being replied to")
    forwarded_from: Optional[str] = Field(default=None, description="Original message ID")
    deleted: bool = Field(default=False, description="Message was deleted")
    edited: bool = Field(default=False, description="Message was edited")
    encrypted: bool = Field(default=False, description="Message was encrypted")
    
    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


class ChatMessage(Communication):
    """Specific model for chat messages"""
    group_id: Optional[str] = Field(default=None, description="Group chat identifier")
    is_group: bool = Field(default=False, description="Is group message")
    read_receipts: List[Dict[str, datetime]] = Field(default_factory=list)
    reaction_count: int = Field(default=0, description="Number of reactions")
    mentions: List[str] = Field(default_factory=list, description="Mentioned users")
    quoted_message: Optional[str] = Field(default=None, description="Quoted message ID")


class EmailMessage(Communication):
    """Specific model for email messages"""
    subject: str = Field(description="Email subject line")
    cc_recipients: List[str] = Field(default_factory=list, description="CC recipients")
    bcc_recipients: List[str] = Field(default_factory=list, description="BCC recipients")
    html_content: Optional[str] = Field(default=None, description="HTML email content")
    headers: Dict[str, str] = Field(default_factory=dict, description="Email headers")
    importance: Optional[str] = Field(default=None, description="Message importance")
    read: bool = Field(default=False, description="Message was read")
    spam_score: Optional[float] = Field(default=None, description="Spam probability")


class NetworkNode(BaseModel):
    """Node in communication network graph"""
    node_id: str = Field(description="Unique node identifier")
    contact: Contact = Field(description="Contact information")
    centrality_score: float = Field(default=0.0, description="Network centrality")
    message_count: int = Field(default=0, description="Total messages")
    connection_count: int = Field(default=0, description="Number of connections")
    activity_level: str = Field(default="low", description="Activity level classification")
    suspicious_score: float = Field(default=0.0, description="Suspicious activity score")


class NetworkEdge(BaseModel):
    """Edge in communication network graph"""
    source_id: str = Field(description="Source node ID")
    target_id: str = Field(description="Target node ID")
    message_count: int = Field(description="Number of messages exchanged")
    first_contact: datetime = Field(description="First communication")
    last_contact: datetime = Field(description="Last communication")
    platforms: List[PlatformType] = Field(description="Platforms used")
    relationship_strength: float = Field(description="Calculated relationship strength")
    is_bidirectional: bool = Field(default=True, description="Two-way communication")
    
    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


class CommunicationNetwork(BaseModel):
    """Represents the communication network structure"""
    network_id: str = Field(default_factory=lambda: str(uuid4()))
    nodes: List[NetworkNode] = Field(description="Network nodes")
    edges: List[NetworkEdge] = Field(description="Network edges")
    total_messages: int = Field(description="Total messages in network")
    date_range: Tuple[datetime, datetime] = Field(description="Network time span")
    platforms: List[PlatformType] = Field(description="Platforms analyzed")
    density: float = Field(description="Network density score")
    clustering_coefficient: float = Field(description="Network clustering")
    
    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


class PatternMatch(BaseModel):
    """Represents a detected communication pattern"""
    pattern_id: str = Field(default_factory=lambda: str(uuid4()))
    pattern_type: PatternType = Field(description="Type of pattern detected")
    description: str = Field(description="Pattern description")
    confidence: float = Field(description="Detection confidence score")
    participants: List[str] = Field(description="Involved contact IDs")
    time_range: Tuple[datetime, datetime] = Field(description="Pattern time span")
    frequency: Optional[float] = Field(default=None, description="Pattern frequency")
    platforms: List[PlatformType] = Field(description="Platforms involved")
    evidence: List[str] = Field(description="Supporting message IDs")
    significance: str = Field(description="Pattern significance level")
    
    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


class AnomalyIndicator(BaseModel):
    """Represents detected communication anomaly"""
    anomaly_id: str = Field(default_factory=lambda: str(uuid4()))
    anomaly_type: AnomalyType = Field(description="Type of anomaly")
    description: str = Field(description="Anomaly description")
    severity: str = Field(description="Anomaly severity level")
    confidence: float = Field(description="Detection confidence")
    affected_contacts: List[str] = Field(description="Involved contacts")
    detection_time: datetime = Field(default_factory=datetime.now)
    time_range: Tuple[datetime, datetime] = Field(description="Anomaly time span")
    baseline_value: Optional[float] = Field(default=None, description="Expected value")
    actual_value: Optional[float] = Field(default=None, description="Observed value")
    deviation_score: float = Field(description="Deviation from normal")
    
    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


class TemporalPattern(BaseModel):
    """Temporal communication pattern analysis"""
    pattern_id: str = Field(default_factory=lambda: str(uuid4()))
    time_unit: str = Field(description="Time unit for analysis")
    frequency_distribution: Dict[str, int] = Field(description="Message frequency")
    peak_times: List[str] = Field(description="Peak activity periods")
    quiet_periods: List[str] = Field(description="Low activity periods")
    regularity_score: float = Field(description="Pattern regularity")
    seasonal_patterns: Dict[str, float] = Field(default_factory=dict)
    trend_direction: str = Field(description="Communication trend")


class Anomaly(BaseModel):
    """Comprehensive anomaly representation"""
    anomaly: AnomalyIndicator = Field(description="Anomaly details")
    context: Dict[str, Any] = Field(description="Contextual information")
    related_messages: List[str] = Field(description="Related message IDs")
    investigation_notes: List[str] = Field(default_factory=list)
    status: str = Field(default="open", description="Investigation status")


class ChatAnalysis(BaseModel):
    """Results of chat export analysis"""
    analysis_id: str = Field(default_factory=lambda: str(uuid4()))
    platform: PlatformType = Field(description="Platform analyzed")
    total_messages: int = Field(description="Total messages processed")
    total_participants: int = Field(description="Number of participants")
    date_range: Tuple[datetime, datetime] = Field(description="Analysis time span")
    message_statistics: Dict[str, Any] = Field(description="Message statistics")
    participant_analysis: Dict[str, Any] = Field(description="Participant analysis")
    temporal_patterns: List[TemporalPattern] = Field(description="Time-based patterns")
    group_dynamics: Dict[str, Any] = Field(default_factory=dict)
    content_analysis: Dict[str, Any] = Field(default_factory=dict)
    media_analysis: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


class EmailAnalysis(BaseModel):
    """Results of email analysis"""
    analysis_id: str = Field(default_factory=lambda: str(uuid4()))
    total_emails: int = Field(description="Total emails processed")
    thread_count: int = Field(description="Number of email threads")
    participants: List[Contact] = Field(description="Email participants")
    date_range: Tuple[datetime, datetime] = Field(description="Analysis time span")
    thread_analysis: Dict[str, Any] = Field(description="Thread statistics")
    relationship_mapping: Dict[str, Any] = Field(description="Contact relationships")
    subject_analysis: Dict[str, Any] = Field(description="Subject line analysis")
    attachment_analysis: Dict[str, Any] = Field(description="Attachment statistics")
    spam_analysis: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


class NetworkGraph(BaseModel):
    """Communication network visualization data"""
    graph_id: str = Field(default_factory=lambda: str(uuid4()))
    network: CommunicationNetwork = Field(description="Network structure")
    layout_data: Dict[str, Any] = Field(description="Graph layout information")
    visualization_config: Dict[str, Any] = Field(description="Visualization settings")
    metrics: Dict[str, float] = Field(description="Network metrics")
    communities: List[List[str]] = Field(description="Detected communities")
    central_figures: List[str] = Field(description="Most central contacts")
    isolated_nodes: List[str] = Field(description="Isolated contacts")
    suspicious_clusters: List[List[str]] = Field(default_factory=list)


class PatternAnalysis(BaseModel):
    """Results of pattern detection analysis"""
    analysis_id: str = Field(default_factory=lambda: str(uuid4()))
    detected_patterns: List[PatternMatch] = Field(description="Detected patterns")
    anomalies: List[Anomaly] = Field(description="Detected anomalies")
    temporal_analysis: Dict[str, Any] = Field(description="Time-based analysis")
    behavioral_profiles: Dict[str, Any] = Field(description="User behavior profiles")
    risk_assessment: Dict[str, Any] = Field(description="Risk assessment results")
    recommendations: List[str] = Field(description="Investigation recommendations")
    confidence_summary: Dict[str, float] = Field(description="Confidence metrics")


class CommsConfig(BaseModel):
    """Configuration for communication analysis operations"""
    
    # Analysis settings
    analysis_level: AnalysisLevel = Field(default=AnalysisLevel.STANDARD)
    enable_content_analysis: bool = Field(default=True)
    enable_network_analysis: bool = Field(default=True)
    enable_pattern_detection: bool = Field(default=True)
    enable_anomaly_detection: bool = Field(default=True)
    
    # Processing settings
    max_file_size_mb: int = Field(default=500, ge=1, le=5000)
    chunk_size: int = Field(default=1000, ge=100, le=10000)
    parallel_processing: bool = Field(default=True)
    max_workers: int = Field(default=4, ge=1, le=16)
    
    # Pattern detection settings
    pattern_confidence_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    anomaly_sensitivity: float = Field(default=0.8, ge=0.0, le=1.0)
    temporal_window_days: int = Field(default=7, ge=1, le=365)
    
    # Network analysis settings
    min_connection_strength: float = Field(default=0.1, ge=0.0, le=1.0)
    max_network_size: int = Field(default=10000, ge=100, le=100000)
    community_detection_algorithm: str = Field(default="louvain")
    
    # Privacy and security settings
    anonymize_contacts: bool = Field(default=False)
    hash_sensitive_data: bool = Field(default=True)
    encrypt_results: bool = Field(default=False)
    audit_trail: bool = Field(default=True)
    
    # Output settings
    output_format: str = Field(default="json")
    include_visualizations: bool = Field(default=True)
    generate_reports: bool = Field(default=True)
    export_raw_data: bool = Field(default=False)
    
    # Forensic settings
    chain_of_custody: bool = Field(default=True)
    digital_signatures: bool = Field(default=True)
    evidence_tagging: bool = Field(default=True)
    timestamp_verification: bool = Field(default=True)


class AnalysisResult(BaseModel):
    """Complete analysis results container"""
    result_id: str = Field(default_factory=lambda: str(uuid4()))
    analysis_timestamp: datetime = Field(default_factory=datetime.now)
    config: CommsConfig = Field(description="Analysis configuration used")
    
    # Core analysis results
    chat_analysis: Optional[ChatAnalysis] = Field(default=None)
    email_analysis: Optional[EmailAnalysis] = Field(default=None)
    network_graph: Optional[NetworkGraph] = Field(default=None)
    pattern_analysis: Optional[PatternAnalysis] = Field(default=None)
    
    # Summary statistics
    total_communications: int = Field(description="Total communications analyzed")
    total_contacts: int = Field(description="Total unique contacts")
    platforms_analyzed: List[PlatformType] = Field(description="Platforms processed")
    analysis_duration: float = Field(description="Analysis time in seconds")
    
    # Quality metrics
    data_quality_score: float = Field(description="Data quality assessment")
    completeness_score: float = Field(description="Analysis completeness")
    confidence_score: float = Field(description="Overall confidence")
    
    # Forensic metadata
    evidence_hash: str = Field(description="Evidence integrity hash")
    chain_of_custody: List[Dict[str, Any]] = Field(default_factory=list)
    digital_signature: Optional[str] = Field(default=None)
    
    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


class CommunicationAnalyzer:
    """Main class for comprehensive communication analysis"""
    
    def __init__(self, config: CommsConfig = None):
        self.config = config or CommsConfig()
        self.logger = logging.getLogger(__name__)
        self._processors = {}
        self._initialize_processors()
    
    def _initialize_processors(self):
        """Initialize specialized processors"""
        from .chat_processor import ChatProcessor
        from .email_analyzer import EmailAnalyzer
        from .network_mapper import NetworkMapper
        from .pattern_detector import PatternDetector
        
        self._processors = {
            'chat': ChatProcessor(self.config),
            'email': EmailAnalyzer(self.config),
            'network': NetworkMapper(self.config),
            'patterns': PatternDetector(self.config)
        }
    
    def analyze_communications(
        self,
        communications: List[Communication],
        analysis_types: Optional[List[str]] = None
    ) -> AnalysisResult:
        """Perform comprehensive communication analysis"""
        start_time = datetime.now()
        
        if analysis_types is None:
            analysis_types = ['chat', 'email', 'network', 'patterns']
        
        result = AnalysisResult(
            config=self.config,
            total_communications=len(communications),
            platforms_analyzed=list(set(c.platform for c in communications)),
            evidence_hash=self._calculate_evidence_hash(communications)
        )
        
        try:
            # Separate communications by type
            chat_messages = [c for c in communications if isinstance(c, ChatMessage)]
            email_messages = [c for c in communications if isinstance(c, EmailMessage)]
            
            # Perform analysis
            if 'chat' in analysis_types and chat_messages:
                result.chat_analysis = self._processors['chat'].analyze(chat_messages)
            
            if 'email' in analysis_types and email_messages:
                result.email_analysis = self._processors['email'].analyze(email_messages)
            
            if 'network' in analysis_types:
                result.network_graph = self._processors['network'].analyze(communications)
            
            if 'patterns' in analysis_types:
                result.pattern_analysis = self._processors['patterns'].analyze(communications)
            
            # Calculate metrics
            result.total_contacts = len(set(
                [c.sender_id for c in communications] +
                [r for c in communications for r in c.recipient_ids]
            ))
            
            result.analysis_duration = (datetime.now() - start_time).total_seconds()
            result.data_quality_score = self._assess_data_quality(communications)
            result.completeness_score = self._assess_completeness(result)
            result.confidence_score = self._calculate_confidence(result)
            
            if self.config.chain_of_custody:
                result.chain_of_custody.append({
                    'action': 'analysis_completed',
                    'timestamp': datetime.now().isoformat(),
                    'operator': os.getenv('USER', 'unknown'),
                    'details': f'Analyzed {len(communications)} communications'
                })
            
        except Exception as e:
            self.logger.error(f"Analysis failed: {str(e)}")
            raise
        
        return result
    
    def _calculate_evidence_hash(self, communications: List[Communication]) -> str:
        """Calculate hash for evidence integrity"""
        content = json.dumps([c.dict() for c in communications], sort_keys=True, default=str)
        return hashlib.sha256(content.encode()).hexdigest()
    
    def _assess_data_quality(self, communications: List[Communication]) -> float:
        """Assess quality of input data"""
        if not communications:
            return 0.0
        
        quality_factors = []
        
        # Completeness of timestamps
        valid_timestamps = sum(1 for c in communications if c.timestamp)
        quality_factors.append(valid_timestamps / len(communications))
        
        # Completeness of content
        with_content = sum(1 for c in communications if c.content)
        quality_factors.append(with_content / len(communications))
        
        # Sender information completeness
        valid_senders = sum(1 for c in communications if c.sender_id)
        quality_factors.append(valid_senders / len(communications))
        
        return sum(quality_factors) / len(quality_factors)
    
    def _assess_completeness(self, result: AnalysisResult) -> float:
        """Assess completeness of analysis"""
        components = [
            result.chat_analysis is not None,
            result.email_analysis is not None,
            result.network_graph is not None,
            result.pattern_analysis is not None
        ]
        return sum(components) / len(components)
    
    def _calculate_confidence(self, result: AnalysisResult) -> float:
        """Calculate overall confidence score"""
        confidence_scores = []
        
        if result.pattern_analysis:
            pattern_confidences = [p.confidence for p in result.pattern_analysis.detected_patterns]
            if pattern_confidences:
                confidence_scores.append(np.mean(pattern_confidences))
        
        confidence_scores.extend([
            result.data_quality_score,
            result.completeness_score
        ])
        
        return np.mean(confidence_scores) if confidence_scores else 0.0


# Utility functions
def load_communications(file_path: Path) -> List[Communication]:
    """Load communications from file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    communications = []
    for item in data:
        if item.get('comm_type') == CommunicationType.CHAT:
            communications.append(ChatMessage(**item))
        elif item.get('comm_type') == CommunicationType.EMAIL:
            communications.append(EmailMessage(**item))
        else:
            communications.append(Communication(**item))
    
    return communications


def export_analysis_results(result: AnalysisResult, output_path: Path):
    """Export analysis results to file"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result.dict(), f, indent=2, default=str)


def generate_forensic_report(
    result: AnalysisResult, 
    template_path: Optional[Path] = None
) -> str:
    """Generate forensic investigation report"""
    report = f"""
DIGITAL FORENSIC COMMUNICATION ANALYSIS REPORT
==============================================

Analysis ID: {result.result_id}
Analysis Date: {result.analysis_timestamp.isoformat()}
Evidence Hash: {result.evidence_hash}

SUMMARY
-------
Total Communications: {result.total_communications}
Total Contacts: {result.total_contacts}
Platforms Analyzed: {', '.join(result.platforms_analyzed)}
Analysis Duration: {result.analysis_duration:.2f} seconds

QUALITY METRICS
--------------
Data Quality Score: {result.data_quality_score:.2f}
Completeness Score: {result.completeness_score:.2f}
Overall Confidence: {result.confidence_score:.2f}

"""
    
    if result.pattern_analysis:
        report += f"""
PATTERN ANALYSIS
---------------
Patterns Detected: {len(result.pattern_analysis.detected_patterns)}
Anomalies Found: {len(result.pattern_analysis.anomalies)}

"""
        
        for pattern in result.pattern_analysis.detected_patterns[:5]:  # Top 5
            report += f"- {pattern.pattern_type}: {pattern.description} (Confidence: {pattern.confidence:.2f})\n"
    
    if result.network_graph:
        report += f"""
NETWORK ANALYSIS
---------------
Network Density: {result.network_graph.metrics.get('density', 'N/A')}
Communities Detected: {len(result.network_graph.communities)}
Central Figures: {len(result.network_graph.central_figures)}
Suspicious Clusters: {len(result.network_graph.suspicious_clusters)}

"""
    
    report += """
CHAIN OF CUSTODY
---------------
"""
    for entry in result.chain_of_custody:
        report += f"- {entry['timestamp']}: {entry['action']} by {entry['operator']}\n"
    
    return report