"""
Core functionality for temporal information extraction and timeline construction.
"""

import json
import uuid
from datetime import datetime, timezone, date, time
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any, Union
from dataclasses import dataclass, field
from pydantic import BaseModel, Field, ConfigDict
import logging
from loguru import logger

# Configure logging
logging.basicConfig(level=logging.INFO)


class TemporalEntityType(Enum):
    """Types of temporal entities that can be recognized"""
    DATE = "DATE"
    TIME = "TIME"
    DATETIME = "DATETIME"
    DURATION = "DURATION"
    PERIOD = "PERIOD"
    FREQUENCY = "FREQUENCY"
    RELATIVE_TIME = "RELATIVE_TIME"
    FUZZY_TIME = "FUZZY_TIME"
    DATE_RANGE = "DATE_RANGE"
    TIME_RANGE = "TIME_RANGE"


class TimelineEventType(Enum):
    """Types of timeline events"""
    INSTANT = "INSTANT"
    DURATION = "DURATION"
    START = "START"
    END = "END"
    MILESTONE = "MILESTONE"
    SEQUENCE = "SEQUENCE"
    CONCURRENT = "CONCURRENT"
    RECURRING = "RECURRING"


class InconsistencyType(Enum):
    """Types of temporal inconsistencies"""
    CHRONOLOGICAL_VIOLATION = "CHRONOLOGICAL_VIOLATION"
    DATE_CONFLICT = "DATE_CONFLICT"
    DURATION_MISMATCH = "DURATION_MISMATCH"
    CAUSALITY_VIOLATION = "CAUSALITY_VIOLATION"
    IMPOSSIBLE_SEQUENCE = "IMPOSSIBLE_SEQUENCE"
    OVERLAPPING_EXCLUSIVE_EVENTS = "OVERLAPPING_EXCLUSIVE_EVENTS"
    MISSING_TEMPORAL_CONTEXT = "MISSING_TEMPORAL_CONTEXT"
    AMBIGUOUS_REFERENCE = "AMBIGUOUS_REFERENCE"


class LanguageCode(Enum):
    """Supported language codes for temporal extraction"""
    EN = "en"  # English
    ES = "es"  # Spanish  
    FR = "fr"  # French
    AR = "ar"  # Arabic
    RU = "ru"  # Russian
    ZH = "zh"  # Chinese
    DE = "de"  # German
    IT = "it"  # Italian
    PT = "pt"  # Portuguese
    JA = "ja"  # Japanese
    KO = "ko"  # Korean
    NL = "nl"  # Dutch


class TemporalEntity(BaseModel):
    """Represents a temporal entity extracted from text"""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    entity_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    text: str = Field(..., description="The temporal expression as it appears in text")
    entity_type: TemporalEntityType = Field(..., description="Type of temporal entity")
    start_pos: int = Field(..., description="Start position in original text")
    end_pos: int = Field(..., description="End position in original text")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Extraction confidence score")
    language: LanguageCode = Field(..., description="Language of the temporal expression")
    document_id: str = Field(..., description="Source document identifier")
    
    # Parsed temporal information
    parsed_date: Optional[datetime] = Field(default=None, description="Parsed datetime value")
    start_date: Optional[datetime] = Field(default=None, description="Start of date range")
    end_date: Optional[datetime] = Field(default=None, description="End of date range")
    duration_seconds: Optional[int] = Field(default=None, description="Duration in seconds")
    is_fuzzy: bool = Field(default=False, description="Whether temporal reference is fuzzy/approximate")
    uncertainty_range: Optional[Tuple[datetime, datetime]] = Field(default=None, description="Uncertainty range")
    
    # Context information
    context: Optional[str] = Field(default=None, description="Surrounding text context")
    normalized_form: Optional[str] = Field(default=None, description="Normalized temporal expression")
    relative_to: Optional[str] = Field(default=None, description="Reference point for relative times")
    
    # Metadata
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    extraction_timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "entity_id": self.entity_id,
            "text": self.text,
            "entity_type": self.entity_type.value,
            "start_pos": self.start_pos,
            "end_pos": self.end_pos,
            "confidence": self.confidence,
            "language": self.language.value,
            "document_id": self.document_id,
            "parsed_date": self.parsed_date.isoformat() if self.parsed_date else None,
            "start_date": self.start_date.isoformat() if self.start_date else None,
            "end_date": self.end_date.isoformat() if self.end_date else None,
            "duration_seconds": self.duration_seconds,
            "is_fuzzy": self.is_fuzzy,
            "uncertainty_range": [
                self.uncertainty_range[0].isoformat(),
                self.uncertainty_range[1].isoformat()
            ] if self.uncertainty_range else None,
            "context": self.context,
            "normalized_form": self.normalized_form,
            "relative_to": self.relative_to,
            "metadata": self.metadata,
            "extraction_timestamp": self.extraction_timestamp.isoformat()
        }


class Event(BaseModel):
    """Represents a timeline event"""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    title: str = Field(..., description="Event title or description")
    event_type: TimelineEventType = Field(..., description="Type of timeline event")
    start_time: datetime = Field(..., description="Event start time")
    end_time: Optional[datetime] = Field(default=None, description="Event end time")
    document_id: str = Field(..., description="Source document identifier")
    
    # Event content
    description: Optional[str] = Field(default=None, description="Detailed event description")
    participants: List[str] = Field(default_factory=list, description="Event participants")
    locations: List[str] = Field(default_factory=list, description="Event locations")
    
    # Temporal information
    temporal_entities: List[TemporalEntity] = Field(default_factory=list, description="Associated temporal entities")
    is_fuzzy: bool = Field(default=False, description="Whether event timing is approximate")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Event confidence score")
    uncertainty_range: Optional[Tuple[datetime, datetime]] = Field(default=None, description="Time uncertainty range")
    
    # Relationships
    before_events: List[str] = Field(default_factory=list, description="Events that occur before this one")
    after_events: List[str] = Field(default_factory=list, description="Events that occur after this one")
    concurrent_events: List[str] = Field(default_factory=list, description="Events that occur simultaneously")
    parent_event: Optional[str] = Field(default=None, description="Parent event ID for sub-events")
    child_events: List[str] = Field(default_factory=list, description="Sub-event IDs")
    
    # Context and source
    source_text: Optional[str] = Field(default=None, description="Source text from which event was extracted")
    context_sentences: List[str] = Field(default_factory=list, description="Context sentences")
    extraction_method: str = Field(default="automatic", description="How event was extracted")
    
    # Metadata
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional event metadata")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    
    def duration(self) -> Optional[int]:
        """Calculate event duration in seconds"""
        if self.end_time:
            return int((self.end_time - self.start_time).total_seconds())
        return None
    
    def overlaps_with(self, other: 'Event') -> bool:
        """Check if this event overlaps with another event"""
        if not self.end_time or not other.end_time:
            return False
        return (self.start_time < other.end_time and 
                other.start_time < self.end_time)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "event_id": self.event_id,
            "title": self.title,
            "event_type": self.event_type.value,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "document_id": self.document_id,
            "description": self.description,
            "participants": self.participants,
            "locations": self.locations,
            "temporal_entities": [te.to_dict() for te in self.temporal_entities],
            "is_fuzzy": self.is_fuzzy,
            "confidence": self.confidence,
            "uncertainty_range": [
                self.uncertainty_range[0].isoformat(),
                self.uncertainty_range[1].isoformat()
            ] if self.uncertainty_range else None,
            "before_events": self.before_events,
            "after_events": self.after_events,
            "concurrent_events": self.concurrent_events,
            "parent_event": self.parent_event,
            "child_events": self.child_events,
            "source_text": self.source_text,
            "context_sentences": self.context_sentences,
            "extraction_method": self.extraction_method,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "duration_seconds": self.duration()
        }


class Inconsistency(BaseModel):
    """Represents a temporal inconsistency"""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    inconsistency_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    inconsistency_type: InconsistencyType = Field(..., description="Type of inconsistency")
    severity: str = Field(..., description="Inconsistency severity (low, medium, high, critical)")
    description: str = Field(..., description="Human-readable description of inconsistency")
    affected_events: List[str] = Field(..., description="IDs of affected events")
    affected_entities: List[str] = Field(default_factory=list, description="IDs of affected temporal entities")
    
    # Detection information
    detection_method: str = Field(..., description="Method used to detect inconsistency")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Detection confidence")
    
    # Resolution suggestions
    suggested_resolution: Optional[str] = Field(default=None, description="Suggested resolution")
    resolution_confidence: float = Field(default=0.0, ge=0.0, le=1.0, description="Resolution suggestion confidence")
    requires_human_review: bool = Field(default=True, description="Whether human review is required")
    
    # Context
    context_information: Dict[str, Any] = Field(default_factory=dict, description="Contextual information")
    source_documents: List[str] = Field(default_factory=list, description="Source document IDs")
    
    # Metadata
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    detected_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "inconsistency_id": self.inconsistency_id,
            "inconsistency_type": self.inconsistency_type.value,
            "severity": self.severity,
            "description": self.description,
            "affected_events": self.affected_events,
            "affected_entities": self.affected_entities,
            "detection_method": self.detection_method,
            "confidence": self.confidence,
            "suggested_resolution": self.suggested_resolution,
            "resolution_confidence": self.resolution_confidence,
            "requires_human_review": self.requires_human_review,
            "context_information": self.context_information,
            "source_documents": self.source_documents,
            "metadata": self.metadata,
            "detected_at": self.detected_at.isoformat()
        }


class Timeline(BaseModel):
    """Represents a chronological timeline"""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    timeline_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    title: str = Field(..., description="Timeline title")
    events: List[Event] = Field(default_factory=list, description="Chronologically ordered events")
    temporal_entities: List[TemporalEntity] = Field(default_factory=list, description="All temporal entities")
    
    # Timeline metadata
    start_date: Optional[datetime] = Field(default=None, description="Timeline start date")
    end_date: Optional[datetime] = Field(default=None, description="Timeline end date")
    source_documents: List[str] = Field(default_factory=list, description="Source document IDs")
    
    # Quality metrics
    completeness_score: float = Field(default=0.0, ge=0.0, le=1.0, description="Timeline completeness score")
    confidence_score: float = Field(default=0.0, ge=0.0, le=1.0, description="Overall confidence score")
    consistency_score: float = Field(default=0.0, ge=0.0, le=1.0, description="Internal consistency score")
    
    # Processing information
    construction_method: str = Field(default="automatic", description="Timeline construction method")
    processing_metadata: Dict[str, Any] = Field(default_factory=dict, description="Processing metadata")
    
    # Metadata
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional timeline metadata")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    
    def add_event(self, event: Event) -> None:
        """Add event to timeline maintaining chronological order"""
        self.events.append(event)
        self.events.sort(key=lambda x: x.start_time)
        self.updated_at = datetime.now(timezone.utc)
        
        # Update timeline date bounds
        self._update_date_bounds()
    
    def remove_event(self, event_id: str) -> bool:
        """Remove event from timeline"""
        original_length = len(self.events)
        self.events = [e for e in self.events if e.event_id != event_id]
        
        if len(self.events) < original_length:
            self.updated_at = datetime.now(timezone.utc)
            self._update_date_bounds()
            return True
        return False
    
    def get_events_in_range(self, start: datetime, end: datetime) -> List[Event]:
        """Get events within specified date range"""
        return [
            event for event in self.events
            if (event.start_time >= start and event.start_time <= end) or
               (event.end_time and event.end_time >= start and event.start_time <= end)
        ]
    
    def get_concurrent_events(self, target_event: Event) -> List[Event]:
        """Get events that occur concurrently with target event"""
        if not target_event.end_time:
            return []
        
        return [
            event for event in self.events
            if event.event_id != target_event.event_id and 
               target_event.overlaps_with(event)
        ]
    
    def _update_date_bounds(self) -> None:
        """Update timeline start and end dates based on events"""
        if not self.events:
            self.start_date = None
            self.end_date = None
            return
        
        self.start_date = min(event.start_time for event in self.events)
        
        end_times = [
            event.end_time for event in self.events 
            if event.end_time is not None
        ]
        if end_times:
            self.end_date = max(end_times)
        else:
            self.end_date = max(event.start_time for event in self.events)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "timeline_id": self.timeline_id,
            "title": self.title,
            "events": [event.to_dict() for event in self.events],
            "temporal_entities": [entity.to_dict() for entity in self.temporal_entities],
            "start_date": self.start_date.isoformat() if self.start_date else None,
            "end_date": self.end_date.isoformat() if self.end_date else None,
            "source_documents": self.source_documents,
            "completeness_score": self.completeness_score,
            "confidence_score": self.confidence_score,
            "consistency_score": self.consistency_score,
            "construction_method": self.construction_method,
            "processing_metadata": self.processing_metadata,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }


class ValidationResult(BaseModel):
    """Result of timeline validation"""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    validation_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timeline_id: str = Field(..., description="ID of validated timeline")
    is_consistent: bool = Field(..., description="Whether timeline is internally consistent")
    consistency_score: float = Field(..., ge=0.0, le=1.0, description="Overall consistency score")
    
    # Detected inconsistencies
    inconsistencies: List[Inconsistency] = Field(default_factory=list, description="Detected inconsistencies")
    critical_issues: int = Field(default=0, description="Number of critical inconsistencies")
    high_priority_issues: int = Field(default=0, description="Number of high priority issues")
    medium_priority_issues: int = Field(default=0, description="Number of medium priority issues")
    low_priority_issues: int = Field(default=0, description="Number of low priority issues")
    
    # Validation metadata
    validation_method: str = Field(..., description="Validation method used")
    validation_coverage: float = Field(default=1.0, ge=0.0, le=1.0, description="Validation coverage percentage")
    
    # Suggestions and recommendations
    improvement_suggestions: List[str] = Field(default_factory=list, description="Timeline improvement suggestions")
    confidence_assessment: str = Field(..., description="Overall confidence assessment")
    
    # Quality metrics
    temporal_accuracy: float = Field(default=0.0, ge=0.0, le=1.0, description="Temporal accuracy score")
    completeness: float = Field(default=0.0, ge=0.0, le=1.0, description="Timeline completeness score")
    logical_consistency: float = Field(default=0.0, ge=0.0, le=1.0, description="Logical consistency score")
    
    # Metadata
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional validation metadata")
    validated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "validation_id": self.validation_id,
            "timeline_id": self.timeline_id,
            "is_consistent": self.is_consistent,
            "consistency_score": self.consistency_score,
            "inconsistencies": [inc.to_dict() for inc in self.inconsistencies],
            "critical_issues": self.critical_issues,
            "high_priority_issues": self.high_priority_issues,
            "medium_priority_issues": self.medium_priority_issues,
            "low_priority_issues": self.low_priority_issues,
            "validation_method": self.validation_method,
            "validation_coverage": self.validation_coverage,
            "improvement_suggestions": self.improvement_suggestions,
            "confidence_assessment": self.confidence_assessment,
            "temporal_accuracy": self.temporal_accuracy,
            "completeness": self.completeness,
            "logical_consistency": self.logical_consistency,
            "metadata": self.metadata,
            "validated_at": self.validated_at.isoformat()
        }


class TimelineVisualization(BaseModel):
    """Represents a timeline visualization configuration"""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    visualization_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timeline_id: str = Field(..., description="Associated timeline ID")
    visualization_type: str = Field(..., description="Type of visualization (plotly, bokeh, pyvis)")
    
    # Visualization settings
    title: str = Field(..., description="Visualization title")
    width: int = Field(default=1200, description="Visualization width in pixels")
    height: int = Field(default=600, description="Visualization height in pixels")
    theme: str = Field(default="light", description="Visualization theme")
    
    # Display options
    show_uncertainty: bool = Field(default=True, description="Whether to show uncertainty ranges")
    show_connections: bool = Field(default=True, description="Whether to show event connections")
    show_tooltips: bool = Field(default=True, description="Whether to show interactive tooltips")
    group_by_category: bool = Field(default=False, description="Whether to group events by category")
    
    # Filtering and highlighting
    event_filters: Dict[str, Any] = Field(default_factory=dict, description="Event filtering criteria")
    highlighted_events: List[str] = Field(default_factory=list, description="Event IDs to highlight")
    color_scheme: Dict[str, str] = Field(default_factory=dict, description="Color scheme for events")
    
    # Export options
    export_formats: List[str] = Field(default_factory=lambda: ["html"], description="Available export formats")
    interactive: bool = Field(default=True, description="Whether visualization is interactive")
    
    # Generated content
    html_content: Optional[str] = Field(default=None, description="Generated HTML content")
    json_data: Optional[Dict[str, Any]] = Field(default=None, description="Visualization data in JSON format")
    
    # Metadata
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional visualization metadata")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "visualization_id": self.visualization_id,
            "timeline_id": self.timeline_id,
            "visualization_type": self.visualization_type,
            "title": self.title,
            "width": self.width,
            "height": self.height,
            "theme": self.theme,
            "show_uncertainty": self.show_uncertainty,
            "show_connections": self.show_connections,
            "show_tooltips": self.show_tooltips,
            "group_by_category": self.group_by_category,
            "event_filters": self.event_filters,
            "highlighted_events": self.highlighted_events,
            "color_scheme": self.color_scheme,
            "export_formats": self.export_formats,
            "interactive": self.interactive,
            "html_content": self.html_content,
            "json_data": self.json_data,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat()
        }


class TimelineConfig(BaseModel):
    """Configuration for timeline processing"""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    # Language settings
    primary_language: LanguageCode = Field(default=LanguageCode.EN)
    supported_languages: List[LanguageCode] = Field(default_factory=lambda: [LanguageCode.EN])
    auto_detect_language: bool = Field(default=True)
    
    # Temporal extraction settings
    extract_dates: bool = Field(default=True)
    extract_times: bool = Field(default=True)
    extract_durations: bool = Field(default=True)
    extract_relative_times: bool = Field(default=True)
    handle_fuzzy_dates: bool = Field(default=True)
    min_confidence: float = Field(default=0.6, ge=0.0, le=1.0)
    
    # Event sequencing settings
    enable_uncertainty_handling: bool = Field(default=True)
    uncertainty_window_hours: int = Field(default=24, ge=0)
    allow_overlapping_events: bool = Field(default=True)
    infer_event_relationships: bool = Field(default=True)
    max_events_per_timeline: int = Field(default=10000, gt=0)
    
    # Validation settings
    enable_validation: bool = Field(default=True)
    strict_chronology: bool = Field(default=False)
    validate_causality: bool = Field(default=True)
    consistency_threshold: float = Field(default=0.8, ge=0.0, le=1.0)
    require_human_review: bool = Field(default=False)
    
    # Visualization settings
    default_visualization: str = Field(default="plotly")
    enable_interactive_features: bool = Field(default=True)
    default_theme: str = Field(default="light")
    
    # Processing settings
    batch_size: int = Field(default=100, gt=0)
    parallel_processing: bool = Field(default=True)
    max_workers: int = Field(default=4, gt=0)
    
    # Output settings
    output_format: str = Field(default="json")
    include_source_text: bool = Field(default=True)
    include_metadata: bool = Field(default=True)
    
    # Advanced settings
    custom_temporal_patterns: Dict[str, str] = Field(default_factory=dict)
    domain_specific_rules: List[str] = Field(default_factory=list)
    
    def validate_configuration(self) -> List[str]:
        """Validate configuration and return list of issues"""
        issues = []
        
        if self.min_confidence < 0.0 or self.min_confidence > 1.0:
            issues.append("min_confidence must be between 0.0 and 1.0")
        
        if self.consistency_threshold < 0.0 or self.consistency_threshold > 1.0:
            issues.append("consistency_threshold must be between 0.0 and 1.0")
        
        if self.uncertainty_window_hours < 0:
            issues.append("uncertainty_window_hours must be non-negative")
        
        if self.max_events_per_timeline <= 0:
            issues.append("max_events_per_timeline must be positive")
        
        if self.batch_size <= 0:
            issues.append("batch_size must be positive")
        
        if self.max_workers <= 0:
            issues.append("max_workers must be positive")
        
        if not self.supported_languages:
            issues.append("At least one supported language must be specified")
        
        return issues


class TimelineProcessor:
    """
    Main processor for temporal information extraction and timeline construction
    """
    
    def __init__(self, config: TimelineConfig):
        """
        Initialize the Timeline processor
        
        Args:
            config: Timeline configuration settings
        """
        self.config = config
        self.temporal_extractor = None
        self.event_sequencer = None
        self.timeline_visualizer = None
        self.temporal_validator = None
        
        # Validate configuration
        config_issues = config.validate_configuration()
        if config_issues:
            raise ValueError(f"Configuration issues: {', '.join(config_issues)}")
        
        # Initialize components
        self._initialize_components()
        
        logger.info("TimelineProcessor initialized with config: {}", config.model_dump())
    
    def _initialize_components(self) -> None:
        """Initialize timeline processing components"""
        try:
            # Import and initialize components
            from .temporal_extractor import TemporalExtractor
            from .event_sequencer import EventSequencer
            from .timeline_visualizer import TimelineVisualizer
            from .temporal_validator import TemporalValidator
            
            self.temporal_extractor = TemporalExtractor(self.config)
            self.event_sequencer = EventSequencer(self.config)
            self.timeline_visualizer = TimelineVisualizer(self.config)
            self.temporal_validator = TemporalValidator(self.config)
            
            logger.info("All timeline components initialized successfully")
            
        except ImportError as e:
            logger.error("Failed to import timeline components: {}", e)
            raise
        except Exception as e:
            logger.error("Failed to initialize timeline components: {}", e)
            raise
    
    def extract_temporal_references(self, text: str, document_id: Optional[str] = None,
                                   language: Optional[LanguageCode] = None) -> List[TemporalEntity]:
        """
        Extract temporal references from text
        
        Args:
            text: Input text to process
            document_id: Optional document identifier
            language: Optional language code
            
        Returns:
            List of extracted temporal entities
        """
        if not text or not text.strip():
            logger.warning("Empty or whitespace-only text provided")
            return []
        
        if not document_id:
            document_id = str(uuid.uuid4())
        
        logger.info("Extracting temporal references from document: {} (length: {} chars)", 
                   document_id, len(text))
        
        try:
            # Use temporal extractor to extract entities
            temporal_entities = self.temporal_extractor.extract_temporal_references(
                text, document_id, language
            )
            
            # Filter by confidence threshold
            filtered_entities = [
                entity for entity in temporal_entities 
                if entity.confidence >= self.config.min_confidence
            ]
            
            logger.info("Extracted {} temporal entities (filtered from {})",
                       len(filtered_entities), len(temporal_entities))
            
            return filtered_entities
            
        except Exception as e:
            logger.error("Error extracting temporal references: {}", e)
            raise
    
    def sequence_events(self, events: List[Event]) -> Timeline:
        """
        Create chronologically sequenced timeline from events
        
        Args:
            events: List of events to sequence
            
        Returns:
            Chronologically ordered timeline
        """
        if not events:
            logger.warning("No events provided for sequencing")
            return Timeline(title="Empty Timeline")
        
        logger.info("Sequencing {} events into timeline", len(events))
        
        try:
            # Use event sequencer to create timeline
            timeline = self.event_sequencer.sequence_events(events)
            
            # Calculate quality metrics
            timeline.completeness_score = self._calculate_completeness_score(timeline)
            timeline.confidence_score = self._calculate_confidence_score(timeline)
            
            logger.info("Timeline created with {} events", len(timeline.events))
            return timeline
            
        except Exception as e:
            logger.error("Error sequencing events: {}", e)
            raise
    
    def detect_temporal_inconsistencies(self, timeline: Timeline) -> List[Inconsistency]:
        """
        Detect temporal inconsistencies in timeline
        
        Args:
            timeline: Timeline to validate
            
        Returns:
            List of detected inconsistencies
        """
        if not self.config.enable_validation:
            logger.info("Timeline validation is disabled")
            return []
        
        logger.info("Detecting temporal inconsistencies in timeline: {}", timeline.timeline_id)
        
        try:
            # Use temporal validator to detect inconsistencies
            inconsistencies = self.temporal_validator.detect_temporal_inconsistencies(timeline)
            
            # Update timeline consistency score
            timeline.consistency_score = self._calculate_consistency_score(timeline, inconsistencies)
            timeline.updated_at = datetime.now(timezone.utc)
            
            logger.info("Detected {} inconsistencies", len(inconsistencies))
            return inconsistencies
            
        except Exception as e:
            logger.error("Error detecting temporal inconsistencies: {}", e)
            raise
    
    def generate_interactive_timeline(self, timeline: Timeline,
                                     visualization_type: str = "plotly") -> TimelineVisualization:
        """
        Generate interactive timeline visualization
        
        Args:
            timeline: Timeline to visualize
            visualization_type: Type of visualization to generate
            
        Returns:
            Timeline visualization object
        """
        logger.info("Generating interactive timeline visualization: {}", visualization_type)
        
        try:
            # Use timeline visualizer to generate visualization
            visualization = self.timeline_visualizer.generate_interactive_timeline(
                timeline, visualization_type
            )
            
            logger.info("Interactive timeline visualization created")
            return visualization
            
        except Exception as e:
            logger.error("Error generating timeline visualization: {}", e)
            raise
    
    def process_document(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Process document file for timeline construction
        
        Args:
            file_path: Path to document file
            
        Returns:
            Dictionary containing timeline processing results
        """
        file_path = Path(file_path)
        if not file_path.exists():
            logger.error("Document file not found: {}", file_path)
            raise FileNotFoundError(f"Document file not found: {file_path}")
        
        logger.info("Processing document file: {}", file_path)
        
        try:
            # Extract text from document
            text = self._extract_text_from_document(file_path)
            document_id = file_path.stem
            
            # Extract temporal entities
            temporal_entities = self.extract_temporal_references(text, document_id)
            
            # Convert temporal entities to events
            events = self._entities_to_events(temporal_entities, text, document_id)
            
            # Create timeline
            timeline = self.sequence_events(events)
            timeline.title = f"Timeline: {file_path.name}"
            timeline.source_documents = [document_id]
            timeline.temporal_entities = temporal_entities
            
            # Validate timeline
            inconsistencies = []
            validation_result = None
            if self.config.enable_validation:
                inconsistencies = self.detect_temporal_inconsistencies(timeline)
                validation_result = ValidationResult(
                    timeline_id=timeline.timeline_id,
                    is_consistent=len([i for i in inconsistencies if i.severity in ["high", "critical"]]) == 0,
                    consistency_score=timeline.consistency_score,
                    inconsistencies=inconsistencies,
                    validation_method="automatic",
                    confidence_assessment="automated_analysis"
                )
            
            result = {
                "timeline": timeline.to_dict(),
                "temporal_entities": [entity.to_dict() for entity in temporal_entities],
                "inconsistencies": [inc.to_dict() for inc in inconsistencies],
                "validation_result": validation_result.to_dict() if validation_result else None,
                "metadata": {
                    "document_id": document_id,
                    "file_path": str(file_path),
                    "file_size": file_path.stat().st_size,
                    "file_type": file_path.suffix.lower(),
                    "text_length": len(text),
                    "processing_timestamp": datetime.now(timezone.utc).isoformat(),
                    "config_hash": hash(str(self.config.model_dump()))
                }
            }
            
            logger.info("Document processing completed: {} events, {} temporal entities",
                       len(timeline.events), len(temporal_entities))
            
            return result
            
        except Exception as e:
            logger.error("Error processing document {}: {}", file_path, e)
            raise
    
    def _calculate_completeness_score(self, timeline: Timeline) -> float:
        """Calculate timeline completeness score"""
        if not timeline.events:
            return 0.0
        
        # Basic completeness based on event density and coverage
        total_duration = 0
        if timeline.start_date and timeline.end_date:
            total_duration = (timeline.end_date - timeline.start_date).total_seconds()
        
        if total_duration == 0:
            return 1.0
        
        # Calculate based on event density and temporal coverage
        event_coverage = len(timeline.events) / max(1, total_duration / (24 * 3600))  # events per day
        
        return min(1.0, event_coverage / 10)  # Normalize to 0-1 range
    
    def _calculate_confidence_score(self, timeline: Timeline) -> float:
        """Calculate overall timeline confidence score"""
        if not timeline.events:
            return 0.0
        
        # Average confidence of all events
        total_confidence = sum(event.confidence for event in timeline.events)
        return total_confidence / len(timeline.events)
    
    def _calculate_consistency_score(self, timeline: Timeline, 
                                   inconsistencies: List[Inconsistency]) -> float:
        """Calculate timeline consistency score"""
        if not timeline.events:
            return 1.0
        
        # Weight inconsistencies by severity
        severity_weights = {"low": 0.1, "medium": 0.3, "high": 0.7, "critical": 1.0}
        
        penalty = 0.0
        for inconsistency in inconsistencies:
            weight = severity_weights.get(inconsistency.severity, 0.5)
            penalty += weight * (1.0 - inconsistency.confidence)
        
        # Normalize penalty by number of events
        if timeline.events:
            penalty = penalty / len(timeline.events)
        
        return max(0.0, 1.0 - penalty)
    
    def _entities_to_events(self, temporal_entities: List[TemporalEntity], 
                           text: str, document_id: str) -> List[Event]:
        """Convert temporal entities to timeline events"""
        events = []
        
        for entity in temporal_entities:
            if entity.parsed_date:
                # Create event from temporal entity
                event = Event(
                    title=f"Event: {entity.text}",
                    event_type=TimelineEventType.INSTANT,
                    start_time=entity.parsed_date,
                    end_time=entity.end_date,
                    document_id=document_id,
                    description=entity.context or f"Temporal reference: {entity.text}",
                    temporal_entities=[entity],
                    is_fuzzy=entity.is_fuzzy,
                    confidence=entity.confidence,
                    uncertainty_range=entity.uncertainty_range,
                    source_text=entity.text,
                    extraction_method="temporal_entity_conversion"
                )
                events.append(event)
        
        return events
    
    def _dict_to_temporal_entity(self, entity_dict: Dict[str, Any]) -> TemporalEntity:
        """Convert dictionary to TemporalEntity object"""
        # Parse datetime strings back to datetime objects
        parsed_date = None
        if entity_dict.get('parsed_date'):
            try:
                parsed_date = datetime.fromisoformat(entity_dict['parsed_date'].replace('Z', '+00:00'))
            except (ValueError, AttributeError):
                pass
        
        start_date = None
        if entity_dict.get('start_date'):
            try:
                start_date = datetime.fromisoformat(entity_dict['start_date'].replace('Z', '+00:00'))
            except (ValueError, AttributeError):
                pass
        
        end_date = None
        if entity_dict.get('end_date'):
            try:
                end_date = datetime.fromisoformat(entity_dict['end_date'].replace('Z', '+00:00'))
            except (ValueError, AttributeError):
                pass
        
        uncertainty_range = None
        if entity_dict.get('uncertainty_range'):
            try:
                ur = entity_dict['uncertainty_range']
                if isinstance(ur, list) and len(ur) == 2:
                    uncertainty_range = (
                        datetime.fromisoformat(ur[0].replace('Z', '+00:00')),
                        datetime.fromisoformat(ur[1].replace('Z', '+00:00'))
                    )
            except (ValueError, AttributeError):
                pass
        
        return TemporalEntity(
            entity_id=entity_dict.get('entity_id', str(uuid.uuid4())),
            text=entity_dict.get('text', ''),
            entity_type=TemporalEntityType(entity_dict.get('entity_type', 'DATE')),
            start_pos=entity_dict.get('start_pos', 0),
            end_pos=entity_dict.get('end_pos', 0),
            confidence=entity_dict.get('confidence', 0.0),
            language=LanguageCode(entity_dict.get('language', 'en')),
            document_id=entity_dict.get('document_id', ''),
            parsed_date=parsed_date,
            start_date=start_date,
            end_date=end_date,
            duration_seconds=entity_dict.get('duration_seconds'),
            is_fuzzy=entity_dict.get('is_fuzzy', False),
            uncertainty_range=uncertainty_range,
            context=entity_dict.get('context'),
            normalized_form=entity_dict.get('normalized_form'),
            relative_to=entity_dict.get('relative_to'),
            metadata=entity_dict.get('metadata', {})
        )
    
    def _dict_to_timeline(self, timeline_dict: Dict[str, Any]) -> Timeline:
        """Convert dictionary to Timeline object"""
        # Parse datetime strings
        start_date = None
        if timeline_dict.get('start_date'):
            try:
                start_date = datetime.fromisoformat(timeline_dict['start_date'].replace('Z', '+00:00'))
            except (ValueError, AttributeError):
                pass
        
        end_date = None
        if timeline_dict.get('end_date'):
            try:
                end_date = datetime.fromisoformat(timeline_dict['end_date'].replace('Z', '+00:00'))
            except (ValueError, AttributeError):
                pass
        
        # Convert event dictionaries to Event objects
        events = []
        for event_dict in timeline_dict.get('events', []):
            events.append(self._dict_to_event(event_dict))
        
        # Convert temporal entity dictionaries
        temporal_entities = []
        for te_dict in timeline_dict.get('temporal_entities', []):
            temporal_entities.append(self._dict_to_temporal_entity(te_dict))
        
        return Timeline(
            timeline_id=timeline_dict.get('timeline_id', str(uuid.uuid4())),
            title=timeline_dict.get('title', ''),
            events=events,
            temporal_entities=temporal_entities,
            start_date=start_date,
            end_date=end_date,
            source_documents=timeline_dict.get('source_documents', []),
            completeness_score=timeline_dict.get('completeness_score', 0.0),
            confidence_score=timeline_dict.get('confidence_score', 0.0),
            consistency_score=timeline_dict.get('consistency_score', 0.0),
            construction_method=timeline_dict.get('construction_method', 'manual'),
            processing_metadata=timeline_dict.get('processing_metadata', {}),
            metadata=timeline_dict.get('metadata', {})
        )
    
    def _dict_to_event(self, event_dict: Dict[str, Any]) -> Event:
        """Convert dictionary to Event object"""
        # Parse datetime strings
        start_time = datetime.fromisoformat(event_dict['start_time'].replace('Z', '+00:00'))
        
        end_time = None
        if event_dict.get('end_time'):
            try:
                end_time = datetime.fromisoformat(event_dict['end_time'].replace('Z', '+00:00'))
            except (ValueError, AttributeError):
                pass
        
        uncertainty_range = None
        if event_dict.get('uncertainty_range'):
            try:
                ur = event_dict['uncertainty_range']
                if isinstance(ur, list) and len(ur) == 2:
                    uncertainty_range = (
                        datetime.fromisoformat(ur[0].replace('Z', '+00:00')),
                        datetime.fromisoformat(ur[1].replace('Z', '+00:00'))
                    )
            except (ValueError, AttributeError):
                pass
        
        # Convert temporal entities
        temporal_entities = []
        for te_dict in event_dict.get('temporal_entities', []):
            temporal_entities.append(self._dict_to_temporal_entity(te_dict))
        
        return Event(
            event_id=event_dict.get('event_id', str(uuid.uuid4())),
            title=event_dict.get('title', ''),
            event_type=TimelineEventType(event_dict.get('event_type', 'INSTANT')),
            start_time=start_time,
            end_time=end_time,
            document_id=event_dict.get('document_id', ''),
            description=event_dict.get('description'),
            participants=event_dict.get('participants', []),
            locations=event_dict.get('locations', []),
            temporal_entities=temporal_entities,
            is_fuzzy=event_dict.get('is_fuzzy', False),
            confidence=event_dict.get('confidence', 1.0),
            uncertainty_range=uncertainty_range,
            before_events=event_dict.get('before_events', []),
            after_events=event_dict.get('after_events', []),
            concurrent_events=event_dict.get('concurrent_events', []),
            parent_event=event_dict.get('parent_event'),
            child_events=event_dict.get('child_events', []),
            source_text=event_dict.get('source_text'),
            context_sentences=event_dict.get('context_sentences', []),
            extraction_method=event_dict.get('extraction_method', 'manual'),
            metadata=event_dict.get('metadata', {})
        )
    
    def _extract_text_from_document(self, file_path: Path) -> str:
        """Extract text from various document formats"""
        suffix = file_path.suffix.lower()
        
        try:
            if suffix == '.txt':
                return file_path.read_text(encoding='utf-8')
            
            elif suffix == '.pdf':
                try:
                    import PyPDF2
                    with open(file_path, 'rb') as f:
                        reader = PyPDF2.PdfReader(f)
                        text = ""
                        for page in reader.pages:
                            text += page.extract_text() + "\n"
                        return text
                except ImportError:
                    logger.error("PyPDF2 not installed, cannot process PDF files")
                    raise
            
            elif suffix in ['.doc', '.docx']:
                try:
                    from docx import Document
                    doc = Document(file_path)
                    text = ""
                    for paragraph in doc.paragraphs:
                        text += paragraph.text + "\n"
                    return text
                except ImportError:
                    logger.error("python-docx not installed, cannot process Word files")
                    raise
            
            elif suffix in ['.html', '.htm']:
                try:
                    from bs4 import BeautifulSoup
                    with open(file_path, 'r', encoding='utf-8') as f:
                        soup = BeautifulSoup(f.read(), 'html.parser')
                        return soup.get_text()
                except ImportError:
                    logger.error("beautifulsoup4 not installed, cannot process HTML files")
                    raise
            
            else:
                logger.warning("Unsupported file format: {}, treating as plain text", suffix)
                return file_path.read_text(encoding='utf-8', errors='ignore')
                
        except Exception as e:
            logger.error("Error extracting text from {}: {}", file_path, e)
            raise


def create_default_timeline_config() -> TimelineConfig:
    """Create default timeline configuration"""
    return TimelineConfig()


def validate_timeline_config(config: TimelineConfig) -> List[str]:
    """Validate timeline configuration and return list of issues"""
    return config.validate_configuration()