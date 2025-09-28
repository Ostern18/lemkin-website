"""
Lemkin Evidence Dashboard Generator Core Module

This module provides the core data models and DashboardGenerator class for
creating professional evidence dashboards, interactive visualizations, and
comprehensive case presentation materials for legal proceedings.
"""

from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, validator
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DashboardType(str, Enum):
    """Types of dashboards that can be generated"""
    CASE_OVERVIEW = "case_overview"
    TIMELINE = "timeline"
    NETWORK_GRAPH = "network_graph"
    METRICS = "metrics"
    EXECUTIVE_SUMMARY = "executive_summary"
    EVIDENCE_BOARD = "evidence_board"
    INVESTIGATION_TRACKER = "investigation_tracker"


class VisualizationType(str, Enum):
    """Types of visualizations supported"""
    TIMELINE = "timeline"
    NETWORK_GRAPH = "network_graph"
    HEAT_MAP = "heat_map"
    BAR_CHART = "bar_chart"
    LINE_CHART = "line_chart"
    PIE_CHART = "pie_chart"
    SCATTER_PLOT = "scatter_plot"
    GEOGRAPHICAL_MAP = "geographical_map"
    RELATIONSHIP_DIAGRAM = "relationship_diagram"
    EVIDENCE_FLOW = "evidence_flow"


class ExportFormat(str, Enum):
    """Supported export formats"""
    HTML = "html"
    PDF = "pdf"
    PNG = "png"
    SVG = "svg"
    JSON = "json"
    CSV = "csv"
    EXCEL = "excel"
    POWERPOINT = "powerpoint"


class EventType(str, Enum):
    """Types of events that can be tracked"""
    INCIDENT = "incident"
    EVIDENCE_COLLECTED = "evidence_collected"
    WITNESS_INTERVIEW = "witness_interview"
    LEGAL_ACTION = "legal_action"
    ANALYSIS_COMPLETED = "analysis_completed"
    COURT_PROCEEDING = "court_proceeding"
    SETTLEMENT = "settlement"
    DEADLINE = "deadline"
    MILESTONE = "milestone"
    COMMUNICATION = "communication"


class EntityType(str, Enum):
    """Types of entities in investigations"""
    PERSON = "person"
    ORGANIZATION = "organization"
    LOCATION = "location"
    EVIDENCE = "evidence"
    DOCUMENT = "document"
    DEVICE = "device"
    ACCOUNT = "account"
    VEHICLE = "vehicle"
    EVENT = "event"
    CONCEPT = "concept"


class RelationshipType(str, Enum):
    """Types of relationships between entities"""
    ASSOCIATED_WITH = "associated_with"
    EMPLOYED_BY = "employed_by"
    OWNED_BY = "owned_by"
    LOCATED_AT = "located_at"
    CREATED_BY = "created_by"
    COMMUNICATED_WITH = "communicated_with"
    PRESENT_AT = "present_at"
    RESPONSIBLE_FOR = "responsible_for"
    RELATED_TO = "related_to"
    DEPENDS_ON = "depends_on"


class InvestigationStatus(str, Enum):
    """Investigation status values"""
    INITIATED = "initiated"
    ACTIVE = "active"
    ON_HOLD = "on_hold"
    UNDER_REVIEW = "under_review"
    COMPLETED = "completed"
    CLOSED = "closed"
    ARCHIVED = "archived"


class EvidenceStatus(str, Enum):
    """Evidence status values"""
    COLLECTED = "collected"
    ANALYZED = "analyzed"
    VERIFIED = "verified"
    DISPUTED = "disputed"
    EXCLUDED = "excluded"
    PENDING = "pending"


class DashboardConfig(BaseModel):
    """Configuration for dashboard generation and display"""
    
    # Display settings
    theme: str = Field(default="professional", description="Visual theme for dashboards")
    color_scheme: str = Field(default="default", description="Color scheme to use")
    layout_style: str = Field(default="responsive", description="Layout style")
    
    # Interactive features
    enable_filtering: bool = Field(default=True)
    enable_search: bool = Field(default=True)
    enable_export: bool = Field(default=True)
    enable_real_time_updates: bool = Field(default=False)
    
    # Performance settings
    max_timeline_events: int = Field(default=1000, ge=1, le=10000)
    max_network_nodes: int = Field(default=500, ge=1, le=2000)
    lazy_loading: bool = Field(default=True)
    
    # Security settings
    require_authentication: bool = Field(default=False)
    enable_audit_logging: bool = Field(default=True)
    data_encryption: bool = Field(default=True)
    
    # Collaboration settings
    enable_comments: bool = Field(default=True)
    enable_sharing: bool = Field(default=False)
    enable_version_control: bool = Field(default=True)


class VisualizationSettings(BaseModel):
    """Settings for specific visualization types"""
    
    # Timeline settings
    timeline_granularity: str = Field(default="day", description="day, hour, minute")
    show_event_details: bool = Field(default=True)
    group_by_category: bool = Field(default=True)
    
    # Network graph settings
    layout_algorithm: str = Field(default="force_directed", description="Graph layout algorithm")
    node_size_attribute: Optional[str] = Field(None, description="Attribute for node sizing")
    edge_weight_attribute: Optional[str] = Field(None, description="Attribute for edge weighting")
    show_labels: bool = Field(default=True)
    
    # Geographic map settings
    map_provider: str = Field(default="openstreetmap", description="Map tile provider")
    clustering_enabled: bool = Field(default=True)
    heat_map_overlay: bool = Field(default=False)
    
    # Chart settings
    show_data_labels: bool = Field(default=True)
    enable_zoom: bool = Field(default=True)
    enable_pan: bool = Field(default=True)
    show_legend: bool = Field(default=True)


class ExportOptions(BaseModel):
    """Options for exporting dashboards and visualizations"""
    
    format: ExportFormat
    quality: str = Field(default="high", description="Export quality: low, medium, high")
    resolution: Optional[Tuple[int, int]] = Field(None, description="Resolution for image exports")
    include_interactive_features: bool = Field(default=True)
    
    # PDF specific options
    page_size: str = Field(default="A4", description="Page size for PDF export")
    orientation: str = Field(default="portrait", description="portrait or landscape")
    
    # Security options
    password_protect: bool = Field(default=False)
    watermark: Optional[str] = Field(None, description="Watermark text")
    
    # Metadata
    include_metadata: bool = Field(default=True)
    author: Optional[str] = None
    title: Optional[str] = None
    description: Optional[str] = None


class Evidence(BaseModel):
    """Represents a piece of evidence in an investigation"""
    
    id: UUID = Field(default_factory=uuid4)
    case_id: str = Field(..., min_length=1)
    
    # Basic information
    title: str = Field(..., min_length=1)
    description: str = Field(..., min_length=1)
    evidence_type: str = Field(..., min_length=1)
    status: EvidenceStatus = Field(default=EvidenceStatus.COLLECTED)
    
    # File information
    file_path: Optional[str] = None
    file_size_bytes: Optional[int] = Field(None, ge=0)
    file_hash: Optional[str] = None
    mime_type: Optional[str] = None
    
    # Chain of custody
    collected_by: str = Field(..., min_length=1)
    collected_at: datetime = Field(default_factory=datetime.utcnow)
    custody_chain: List[Dict[str, Any]] = Field(default_factory=list)
    
    # Analysis results
    analysis_completed: bool = Field(default=False)
    authenticity_verified: bool = Field(default=False)
    analysis_results: Dict[str, Any] = Field(default_factory=dict)
    
    # Legal considerations
    admissible: Optional[bool] = None
    legal_notes: List[str] = Field(default_factory=list)
    
    # Metadata
    tags: List[str] = Field(default_factory=list)
    priority: str = Field(default="medium", description="low, medium, high, critical")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class Event(BaseModel):
    """Represents an event in an investigation timeline"""
    
    id: UUID = Field(default_factory=uuid4)
    case_id: str = Field(..., min_length=1)
    
    # Basic information
    title: str = Field(..., min_length=1)
    description: str = Field(..., min_length=1)
    event_type: EventType
    
    # Timing
    timestamp: datetime = Field(..., description="When the event occurred")
    end_timestamp: Optional[datetime] = Field(None, description="For duration events")
    timezone: Optional[str] = Field(None, description="Timezone for the event")
    
    # Location
    location: Optional[str] = None
    latitude: Optional[float] = Field(None, ge=-90.0, le=90.0)
    longitude: Optional[float] = Field(None, ge=-180.0, le=180.0)
    
    # Participants and evidence
    participants: List[str] = Field(default_factory=list)
    evidence_ids: List[UUID] = Field(default_factory=list)
    
    # Significance
    importance: str = Field(default="medium", description="low, medium, high, critical")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    
    # Metadata
    tags: List[str] = Field(default_factory=list)
    source: Optional[str] = None
    verified: bool = Field(default=False)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class Entity(BaseModel):
    """Represents an entity (person, organization, etc.) in an investigation"""
    
    id: UUID = Field(default_factory=uuid4)
    case_id: str = Field(..., min_length=1)
    
    # Basic information
    name: str = Field(..., min_length=1)
    entity_type: EntityType
    aliases: List[str] = Field(default_factory=list)
    
    # Details
    description: Optional[str] = None
    attributes: Dict[str, Any] = Field(default_factory=dict)
    
    # Contact information (for persons/organizations)
    contact_info: Dict[str, str] = Field(default_factory=dict)
    
    # Geographic information
    primary_location: Optional[str] = None
    associated_locations: List[str] = Field(default_factory=list)
    
    # Investigation relevance
    role_in_case: Optional[str] = None
    importance: str = Field(default="medium", description="low, medium, high, critical")
    
    # Legal status
    subject_of_investigation: bool = Field(default=False)
    witness: bool = Field(default=False)
    
    # Metadata
    tags: List[str] = Field(default_factory=list)
    first_mentioned: Optional[datetime] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class Relationship(BaseModel):
    """Represents a relationship between two entities"""
    
    id: UUID = Field(default_factory=uuid4)
    case_id: str = Field(..., min_length=1)
    
    # Relationship definition
    source_entity_id: UUID = Field(..., description="Source entity ID")
    target_entity_id: UUID = Field(..., description="Target entity ID")
    relationship_type: RelationshipType
    
    # Details
    description: Optional[str] = None
    strength: float = Field(default=0.5, ge=0.0, le=1.0, description="Relationship strength")
    direction: str = Field(default="bidirectional", description="unidirectional, bidirectional")
    
    # Temporal aspects
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    duration: Optional[str] = None
    
    # Evidence
    supporting_evidence: List[UUID] = Field(default_factory=list)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    
    # Context
    context: Optional[str] = None
    relevance: str = Field(default="medium", description="low, medium, high, critical")
    
    # Metadata
    tags: List[str] = Field(default_factory=list)
    verified: bool = Field(default=False)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class Investigation(BaseModel):
    """Represents an investigation case"""
    
    id: UUID = Field(default_factory=uuid4)
    case_id: str = Field(..., min_length=1, description="Human-readable case identifier")
    
    # Basic information
    title: str = Field(..., min_length=1)
    description: str = Field(..., min_length=1)
    status: InvestigationStatus = Field(default=InvestigationStatus.INITIATED)
    
    # Timing
    start_date: datetime = Field(default_factory=datetime.utcnow)
    end_date: Optional[datetime] = None
    deadline: Optional[datetime] = None
    
    # Personnel
    lead_investigator: str = Field(..., min_length=1)
    team_members: List[str] = Field(default_factory=list)
    legal_counsel: List[str] = Field(default_factory=list)
    
    # Case details
    jurisdiction: Optional[str] = None
    legal_basis: Optional[str] = None
    case_priority: str = Field(default="medium", description="low, medium, high, critical")
    
    # Progress tracking
    milestones: List[Dict[str, Any]] = Field(default_factory=list)
    current_phase: Optional[str] = None
    completion_percentage: float = Field(default=0.0, ge=0.0, le=100.0)
    
    # Associated data
    evidence_count: int = Field(default=0, ge=0)
    entity_count: int = Field(default=0, ge=0)
    event_count: int = Field(default=0, ge=0)
    
    # Metadata
    tags: List[str] = Field(default_factory=list)
    confidentiality_level: str = Field(default="standard", description="public, standard, confidential, restricted")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class CaseOverview(BaseModel):
    """High-level overview of a case for dashboard display"""
    
    investigation: Investigation
    
    # Summary statistics
    total_evidence: int = Field(default=0, ge=0)
    verified_evidence: int = Field(default=0, ge=0)
    total_entities: int = Field(default=0, ge=0)
    total_events: int = Field(default=0, ge=0)
    total_relationships: int = Field(default=0, ge=0)
    
    # Key dates
    case_start_date: datetime
    last_activity_date: Optional[datetime] = None
    next_deadline: Optional[datetime] = None
    
    # Key participants
    key_entities: List[Entity] = Field(default_factory=list)
    witnesses: List[str] = Field(default_factory=list)
    
    # Critical information
    critical_events: List[Event] = Field(default_factory=list)
    high_priority_evidence: List[Evidence] = Field(default_factory=list)
    
    # Progress indicators
    progress_by_category: Dict[str, float] = Field(default_factory=dict)
    completion_status: Dict[str, str] = Field(default_factory=dict)
    
    # Geographic distribution
    locations_involved: List[str] = Field(default_factory=list)
    primary_jurisdiction: Optional[str] = None


class TimelineVisualization(BaseModel):
    """Represents a timeline visualization of case events"""
    
    id: UUID = Field(default_factory=uuid4)
    case_id: str = Field(..., min_length=1)
    title: str = Field(..., min_length=1)
    
    # Timeline data
    events: List[Event] = Field(default_factory=list)
    time_range_start: datetime
    time_range_end: datetime
    granularity: str = Field(default="day", description="minute, hour, day, week, month")
    
    # Visualization settings
    layout: str = Field(default="horizontal", description="horizontal, vertical")
    group_by: Optional[str] = Field(None, description="Group events by category")
    color_coding: Dict[str, str] = Field(default_factory=dict)
    
    # Interactive features
    filterable_categories: List[str] = Field(default_factory=list)
    searchable_fields: List[str] = Field(default_factory=list)
    
    # Export settings
    export_options: Optional[ExportOptions] = None
    
    # Metadata
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class NetworkGraph(BaseModel):
    """Represents a network graph visualization of entity relationships"""
    
    id: UUID = Field(default_factory=uuid4)
    case_id: str = Field(..., min_length=1)
    title: str = Field(..., min_length=1)
    
    # Graph data
    entities: List[Entity] = Field(default_factory=list)
    relationships: List[Relationship] = Field(default_factory=list)
    
    # Layout settings
    layout_algorithm: str = Field(default="force_directed")
    node_sizing: str = Field(default="importance", description="uniform, importance, connections")
    edge_weighting: str = Field(default="strength", description="uniform, strength, confidence")
    
    # Visual settings
    show_labels: bool = Field(default=True)
    label_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    color_scheme: str = Field(default="entity_type")
    
    # Interactive features
    clustering_enabled: bool = Field(default=True)
    filter_by_relationship_type: bool = Field(default=True)
    highlight_paths: bool = Field(default=True)
    
    # Export settings
    export_options: Optional[ExportOptions] = None
    
    # Metadata
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class MetricsDashboard(BaseModel):
    """Represents investigation metrics and progress tracking"""
    
    id: UUID = Field(default_factory=uuid4)
    case_id: str = Field(..., min_length=1)
    title: str = Field(..., min_length=1)
    
    # Progress metrics
    overall_progress: float = Field(..., ge=0.0, le=100.0)
    evidence_collection_progress: float = Field(..., ge=0.0, le=100.0)
    analysis_progress: float = Field(..., ge=0.0, le=100.0)
    verification_progress: float = Field(..., ge=0.0, le=100.0)
    
    # Quality metrics
    evidence_quality_score: float = Field(..., ge=0.0, le=100.0)
    data_completeness: float = Field(..., ge=0.0, le=100.0)
    verification_rate: float = Field(..., ge=0.0, le=100.0)
    
    # Productivity metrics
    items_processed_today: int = Field(default=0, ge=0)
    items_processed_week: int = Field(default=0, ge=0)
    items_processed_month: int = Field(default=0, ge=0)
    
    # Timeline metrics
    days_since_start: int = Field(..., ge=0)
    days_until_deadline: Optional[int] = Field(None, ge=0)
    milestone_completion: Dict[str, bool] = Field(default_factory=dict)
    
    # Team metrics
    team_size: int = Field(..., ge=1)
    average_workload: float = Field(..., ge=0.0)
    productivity_trend: str = Field(default="stable", description="increasing, stable, decreasing")
    
    # Key performance indicators
    critical_path_items: int = Field(default=0, ge=0)
    blockers: int = Field(default=0, ge=0)
    risks: int = Field(default=0, ge=0)
    
    # Export settings
    export_options: Optional[ExportOptions] = None
    
    # Update tracking
    last_updated: datetime = Field(default_factory=datetime.utcnow)
    update_frequency: str = Field(default="daily", description="hourly, daily, weekly")


class Dashboard(BaseModel):
    """Main dashboard containing multiple visualizations and data"""
    
    id: UUID = Field(default_factory=uuid4)
    case_id: str = Field(..., min_length=1)
    title: str = Field(..., min_length=1)
    dashboard_type: DashboardType
    
    # Configuration
    config: DashboardConfig
    
    # Content components
    case_overview: Optional[CaseOverview] = None
    timeline_visualization: Optional[TimelineVisualization] = None
    network_graph: Optional[NetworkGraph] = None
    metrics_dashboard: Optional[MetricsDashboard] = None
    
    # Additional visualizations
    custom_charts: List[Dict[str, Any]] = Field(default_factory=list)
    geographic_maps: List[Dict[str, Any]] = Field(default_factory=list)
    
    # Layout and presentation
    layout_config: Dict[str, Any] = Field(default_factory=dict)
    responsive_design: bool = Field(default=True)
    
    # Access and sharing
    access_level: str = Field(default="private", description="public, shared, private, restricted")
    shared_with: List[str] = Field(default_factory=list)
    
    # Version control
    version: int = Field(default=1, ge=1)
    change_log: List[Dict[str, Any]] = Field(default_factory=list)
    
    # Metadata
    created_by: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    def add_visualization(self, visualization_type: VisualizationType, data: Dict[str, Any]):
        """Add a custom visualization to the dashboard"""
        visualization = {
            "id": str(uuid4()),
            "type": visualization_type,
            "data": data,
            "created_at": datetime.utcnow().isoformat()
        }
        self.custom_charts.append(visualization)
        self.updated_at = datetime.utcnow()
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics for the dashboard"""
        stats = {
            "total_visualizations": len(self.custom_charts) + len(self.geographic_maps),
            "has_timeline": self.timeline_visualization is not None,
            "has_network_graph": self.network_graph is not None,
            "has_metrics": self.metrics_dashboard is not None,
            "last_updated": self.updated_at,
            "version": self.version
        }
        
        if self.case_overview:
            stats.update({
                "total_evidence": self.case_overview.total_evidence,
                "total_entities": self.case_overview.total_entities,
                "total_events": self.case_overview.total_events,
                "total_relationships": self.case_overview.total_relationships
            })
        
        return stats


class DashboardGenerator:
    """
    Main class for generating professional evidence dashboards and visualizations
    for legal case presentation and investigation tracking.
    """
    
    def __init__(self, config: Optional[DashboardConfig] = None):
        """Initialize the dashboard generator with configuration"""
        self.config = config or DashboardConfig()
        self.logger = logging.getLogger(f"{__name__}.DashboardGenerator")
        
        # Initialize component generators
        self._case_dashboard_generator = None
        self._timeline_visualizer = None
        self._network_grapher = None
        self._metrics_tracker = None
        
        self.logger.info("Dashboard Generator initialized")
    
    def generate_case_dashboard(self, case_id: str, investigation: Investigation, 
                              evidence_list: List[Evidence], entities: List[Entity],
                              events: List[Event], relationships: List[Relationship]) -> Dashboard:
        """Generate a comprehensive case overview dashboard"""
        
        # Create case overview
        case_overview = CaseOverview(
            investigation=investigation,
            total_evidence=len(evidence_list),
            verified_evidence=len([e for e in evidence_list if e.authenticity_verified]),
            total_entities=len(entities),
            total_events=len(events),
            total_relationships=len(relationships),
            case_start_date=investigation.start_date,
            key_entities=[e for e in entities if e.importance in ["high", "critical"]],
            critical_events=[e for e in events if e.importance in ["high", "critical"]],
            high_priority_evidence=[e for e in evidence_list if e.priority in ["high", "critical"]]
        )
        
        # Create dashboard
        dashboard = Dashboard(
            case_id=case_id,
            title=f"Case Overview: {investigation.title}",
            dashboard_type=DashboardType.CASE_OVERVIEW,
            config=self.config,
            case_overview=case_overview,
            created_by="Dashboard Generator"
        )
        
        self.logger.info(f"Generated case dashboard for case: {case_id}")
        return dashboard
    
    def create_interactive_timeline(self, case_id: str, events: List[Event], 
                                   title: Optional[str] = None) -> TimelineVisualization:
        """Create an interactive timeline visualization of case events"""
        
        if not events:
            raise ValueError("Cannot create timeline with empty events list")
        
        # Calculate time range
        timestamps = [e.timestamp for e in events]
        time_range_start = min(timestamps)
        time_range_end = max(timestamps)
        
        # Create timeline visualization
        timeline = TimelineVisualization(
            case_id=case_id,
            title=title or f"Case Timeline: {case_id}",
            events=events,
            time_range_start=time_range_start,
            time_range_end=time_range_end,
            filterable_categories=list(set([e.event_type.value for e in events])),
            searchable_fields=["title", "description", "participants"]
        )
        
        self.logger.info(f"Created timeline with {len(events)} events for case: {case_id}")
        return timeline
    
    def visualize_entity_network(self, case_id: str, entities: List[Entity], 
                                relationships: List[Relationship],
                                title: Optional[str] = None) -> NetworkGraph:
        """Create a network graph visualization of entity relationships"""
        
        if not entities:
            raise ValueError("Cannot create network graph with empty entities list")
        
        # Create network graph
        network_graph = NetworkGraph(
            case_id=case_id,
            title=title or f"Entity Network: {case_id}",
            entities=entities,
            relationships=relationships,
            show_labels=True,
            clustering_enabled=len(entities) > 20,
            filter_by_relationship_type=len(relationships) > 10
        )
        
        self.logger.info(f"Created network graph with {len(entities)} entities and {len(relationships)} relationships for case: {case_id}")
        return network_graph
    
    def track_investigation_metrics(self, case_id: str, investigation: Investigation,
                                   evidence_list: List[Evidence], events: List[Event]) -> MetricsDashboard:
        """Generate investigation progress and metrics dashboard"""
        
        # Calculate progress metrics
        total_evidence = len(evidence_list)
        verified_evidence = len([e for e in evidence_list if e.authenticity_verified])
        analyzed_evidence = len([e for e in evidence_list if e.analysis_completed])
        
        evidence_collection_progress = 100.0  # Assume collection is complete if evidence exists
        analysis_progress = (analyzed_evidence / total_evidence * 100) if total_evidence > 0 else 0.0
        verification_progress = (verified_evidence / total_evidence * 100) if total_evidence > 0 else 0.0
        overall_progress = investigation.completion_percentage
        
        # Calculate quality metrics
        evidence_quality_score = verification_progress  # Simplified calculation
        data_completeness = 85.0  # Placeholder - would be calculated based on required fields
        verification_rate = verification_progress
        
        # Calculate timeline metrics
        days_since_start = (datetime.utcnow() - investigation.start_date).days
        days_until_deadline = None
        if investigation.deadline:
            days_until_deadline = (investigation.deadline - datetime.utcnow()).days
        
        # Create metrics dashboard
        metrics = MetricsDashboard(
            case_id=case_id,
            title=f"Investigation Metrics: {investigation.title}",
            overall_progress=overall_progress,
            evidence_collection_progress=evidence_collection_progress,
            analysis_progress=analysis_progress,
            verification_progress=verification_progress,
            evidence_quality_score=evidence_quality_score,
            data_completeness=data_completeness,
            verification_rate=verification_rate,
            days_since_start=days_since_start,
            days_until_deadline=days_until_deadline,
            team_size=len(investigation.team_members) + 1,  # +1 for lead investigator
            average_workload=50.0,  # Placeholder
            items_processed_today=5,  # Placeholder
            items_processed_week=25,  # Placeholder
            items_processed_month=100  # Placeholder
        )
        
        self.logger.info(f"Generated metrics dashboard for case: {case_id}")
        return metrics
    
    def export_dashboard(self, dashboard: Dashboard, output_path: Path, 
                        export_format: ExportFormat = ExportFormat.HTML) -> bool:
        """Export dashboard to specified format"""
        try:
            if export_format == ExportFormat.JSON:
                with open(output_path, 'w') as f:
                    json.dump(dashboard.dict(), f, indent=2, default=str)
            elif export_format == ExportFormat.HTML:
                # Generate HTML representation
                html_content = self._generate_html_dashboard(dashboard)
                with open(output_path, 'w') as f:
                    f.write(html_content)
            else:
                self.logger.warning(f"Export format {export_format} not yet implemented")
                return False
            
            self.logger.info(f"Dashboard exported to: {output_path}")
            return True
        
        except Exception as e:
            self.logger.error(f"Failed to export dashboard: {str(e)}")
            return False
    
    def _generate_html_dashboard(self, dashboard: Dashboard) -> str:
        """Generate HTML representation of dashboard"""
        html_template = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>{dashboard.title}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 0; padding: 20px; background: #f5f5f5; }}
                .dashboard {{ max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; }}
                .header {{ border-bottom: 2px solid #333; padding-bottom: 20px; margin-bottom: 20px; }}
                .section {{ margin-bottom: 30px; }}
                .metrics {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; }}
                .metric-card {{ background: #f8f9fa; padding: 15px; border-radius: 5px; text-align: center; }}
                .metric-value {{ font-size: 2em; font-weight: bold; color: #007bff; }}
                .metric-label {{ color: #666; margin-top: 5px; }}
            </style>
        </head>
        <body>
            <div class="dashboard">
                <div class="header">
                    <h1>{dashboard.title}</h1>
                    <p>Case ID: {dashboard.case_id}</p>
                    <p>Generated: {dashboard.created_at.strftime('%Y-%m-%d %H:%M')}</p>
                </div>
        """
        
        # Add case overview if available
        if dashboard.case_overview:
            overview = dashboard.case_overview
            html_template += f"""
                <div class="section">
                    <h2>Case Overview</h2>
                    <div class="metrics">
                        <div class="metric-card">
                            <div class="metric-value">{overview.total_evidence}</div>
                            <div class="metric-label">Total Evidence</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-value">{overview.total_entities}</div>
                            <div class="metric-label">Entities</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-value">{overview.total_events}</div>
                            <div class="metric-label">Events</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-value">{overview.total_relationships}</div>
                            <div class="metric-label">Relationships</div>
                        </div>
                    </div>
                </div>
            """
        
        # Add metrics if available
        if dashboard.metrics_dashboard:
            metrics = dashboard.metrics_dashboard
            html_template += f"""
                <div class="section">
                    <h2>Investigation Progress</h2>
                    <div class="metrics">
                        <div class="metric-card">
                            <div class="metric-value">{metrics.overall_progress:.1f}%</div>
                            <div class="metric-label">Overall Progress</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-value">{metrics.evidence_quality_score:.1f}%</div>
                            <div class="metric-label">Evidence Quality</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-value">{metrics.days_since_start}</div>
                            <div class="metric-label">Days Since Start</div>
                        </div>
                    </div>
                </div>
            """
        
        html_template += """
            </div>
        </body>
        </html>
        """
        
        return html_template