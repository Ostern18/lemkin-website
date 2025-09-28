"""
Lemkin Timeline: Temporal information extraction and chronological narrative construction.

This package provides tools for extracting temporal information from documents
and constructing chronological narratives for legal investigations.
"""

from .core import (
    TimelineProcessor,
    TemporalEntity,
    Event,
    Timeline,
    Inconsistency,
    TimelineConfig,
    ValidationResult,
    TimelineVisualization,
    TemporalEntityType,
    TimelineEventType,
    InconsistencyType,
    create_default_timeline_config,
    validate_timeline_config
)

from .temporal_extractor import (
    TemporalExtractor,
    DateExtractor,
    TimeExtractor,
    DurationExtractor,
    extract_temporal_references
)

from .event_sequencer import (
    EventSequencer,
    ChronologicalSorter,
    UncertaintyHandler,
    sequence_events
)

from .timeline_visualizer import (
    TimelineVisualizer,
    PlotlyVisualizer,
    BokehVisualizer,
    generate_interactive_timeline
)

from .temporal_validator import (
    TemporalValidator,
    ConsistencyChecker,
    TemporalLogicEngine,
    detect_temporal_inconsistencies
)

__version__ = "0.1.0"
__author__ = "Lemkin AI Contributors"
__email__ = "contributors@lemkin.org"

__all__ = [
    # Core classes and functions
    "TimelineProcessor",
    "TemporalEntity",
    "Event", 
    "Timeline",
    "Inconsistency",
    "TimelineConfig",
    "ValidationResult",
    "TimelineVisualization",
    "TemporalEntityType",
    "TimelineEventType", 
    "InconsistencyType",
    "create_default_timeline_config",
    "validate_timeline_config",
    
    # Temporal extraction
    "TemporalExtractor",
    "DateExtractor",
    "TimeExtractor", 
    "DurationExtractor",
    "extract_temporal_references",
    
    # Event sequencing
    "EventSequencer",
    "ChronologicalSorter",
    "UncertaintyHandler",
    "sequence_events",
    
    # Timeline visualization
    "TimelineVisualizer",
    "PlotlyVisualizer",
    "BokehVisualizer",
    "generate_interactive_timeline",
    
    # Temporal validation
    "TemporalValidator",
    "ConsistencyChecker",
    "TemporalLogicEngine",
    "detect_temporal_inconsistencies"
]