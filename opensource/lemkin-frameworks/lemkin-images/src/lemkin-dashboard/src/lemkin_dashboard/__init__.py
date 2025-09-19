"""
Lemkin Evidence Dashboard Generator

Professional dashboard creation for case presentation and legal proceedings.
Provides interactive visualizations, timeline analysis, network mapping, and
comprehensive investigation tracking for legal professionals.
"""

from .core import (
    DashboardGenerator,
    Dashboard,
    TimelineVisualization,
    NetworkGraph,
    MetricsDashboard,
    CaseOverview,
    Investigation,
    Evidence,
    Event,
    Entity,
    Relationship,
    DashboardConfig,
    VisualizationSettings,
    ExportOptions,
)

from .case_dashboard import generate_case_dashboard
from .timeline_visualizer import create_interactive_timeline
from .network_grapher import visualize_entity_network
from .metrics_tracker import track_investigation_metrics

__version__ = "0.1.0"
__author__ = "Lemkin AI Contributors"

__all__ = [
    # Core classes
    "DashboardGenerator",
    "Dashboard",
    "TimelineVisualization", 
    "NetworkGraph",
    "MetricsDashboard",
    # Data models
    "CaseOverview",
    "Investigation",
    "Evidence",
    "Event", 
    "Entity",
    "Relationship",
    # Configuration
    "DashboardConfig",
    "VisualizationSettings",
    "ExportOptions",
    # Main functions
    "generate_case_dashboard",
    "create_interactive_timeline",
    "visualize_entity_network", 
    "track_investigation_metrics",
]