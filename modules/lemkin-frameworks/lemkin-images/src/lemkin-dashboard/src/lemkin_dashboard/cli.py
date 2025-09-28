"""
Lemkin Evidence Dashboard Generator CLI Interface

Command-line interface for comprehensive evidence dashboard generation and visualization:
- generate-dashboard: Create case overview dashboard
- create-timeline: Generate interactive timeline visualization  
- graph-network: Create network relationship visualizations
- track-metrics: Generate investigation progress metrics
- serve: Launch dashboard server
- export: Export dashboards to various formats

Provides user-friendly interface for professional legal case presentation tools.
"""

import click
import json
import os
import sys
import subprocess
import webbrowser
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List, Dict, Any
import logging
import traceback
import threading
import time

from .core import (
    DashboardGenerator,
    DashboardConfig,
    Dashboard,
    Investigation,
    Evidence,
    Event,
    Entity,
    Relationship,
    InvestigationStatus,
    EvidenceStatus,
    EventType,
    EntityType,
    RelationshipType,
    ExportFormat,
    VisualizationSettings,
    ExportOptions
)
from .case_dashboard import create_streamlit_case_dashboard, generate_case_dashboard
from .timeline_visualizer import TimelineVisualizer, create_interactive_timeline
from .network_grapher import NetworkAnalyzer, visualize_entity_network
from .metrics_tracker import InvestigationMetricsTracker, track_investigation_metrics

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def setup_output_directory(output_dir: Optional[str], operation: str) -> Path:
    """Setup and create output directory"""
    if not output_dir:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"lemkin_dashboard_{operation}_{timestamp}"
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path


def save_results_to_json(results: dict, output_path: Path, filename: str):
    """Save results to JSON file"""
    output_file = output_path / filename
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, default=str)
    click.echo(f"Results saved to: {output_file}")


def load_case_data(case_data_file: Optional[str]) -> Dict[str, Any]:
    """Load case data from JSON file or generate sample data"""
    if case_data_file and Path(case_data_file).exists():
        with open(case_data_file, 'r') as f:
            return json.load(f)
    else:
        # Return sample case data
        return _generate_sample_case_data()


def _generate_sample_case_data() -> Dict[str, Any]:
    """Generate sample case data for demonstration"""
    from uuid import uuid4
    
    base_date = datetime.now() - timedelta(days=60)
    
    return {
        "investigation": {
            "case_id": "CASE-2024-001",
            "title": "Corporate Fraud Investigation",
            "description": "Investigation into alleged financial irregularities",
            "status": "active",
            "lead_investigator": "Detective Sarah Johnson",
            "team_members": ["Officer Mike Brown", "Analyst Lisa Chen"],
            "case_priority": "high",
            "start_date": base_date.isoformat(),
            "completion_percentage": 75.0
        },
        "evidence": [
            {
                "title": "Financial Records - Q3 2024",
                "description": "Quarterly financial statements and transaction records",
                "evidence_type": "Financial Document",
                "status": "verified",
                "collected_by": "Detective Johnson",
                "collected_at": (base_date + timedelta(days=5)).isoformat(),
                "authenticity_verified": True,
                "analysis_completed": True,
                "priority": "high"
            },
            {
                "title": "Email Communications",
                "description": "Email exchanges between executives",
                "evidence_type": "Digital Evidence",
                "status": "analyzed",
                "collected_by": "Officer Brown",
                "collected_at": (base_date + timedelta(days=10)).isoformat(),
                "authenticity_verified": True,
                "analysis_completed": True,
                "priority": "critical"
            },
            {
                "title": "Bank Transfer Records",
                "description": "Suspicious transaction patterns identified",
                "evidence_type": "Financial Document",
                "status": "pending",
                "collected_by": "Analyst Chen",
                "collected_at": (base_date + timedelta(days=15)).isoformat(),
                "authenticity_verified": False,
                "analysis_completed": False,
                "priority": "high"
            }
        ],
        "entities": [
            {
                "name": "John Smith",
                "entity_type": "person",
                "importance": "critical",
                "role_in_case": "Primary suspect",
                "subject_of_investigation": True,
                "primary_location": "New York, NY"
            },
            {
                "name": "TechCorp Inc.",
                "entity_type": "organization", 
                "importance": "high",
                "role_in_case": "Affected company",
                "subject_of_investigation": False,
                "primary_location": "San Francisco, CA"
            },
            {
                "name": "Jane Doe",
                "entity_type": "person",
                "importance": "medium",
                "role_in_case": "Key witness",
                "witness": True,
                "primary_location": "Boston, MA"
            }
        ],
        "events": [
            {
                "title": "Initial Complaint Filed",
                "description": "Company reported suspicious financial activity",
                "event_type": "incident",
                "timestamp": base_date.isoformat(),
                "importance": "critical",
                "location": "San Francisco, CA"
            },
            {
                "title": "Financial Records Seized",
                "description": "Court-ordered seizure of financial documents",
                "event_type": "evidence_collected",
                "timestamp": (base_date + timedelta(days=7)).isoformat(),
                "importance": "high",
                "location": "TechCorp headquarters"
            },
            {
                "title": "Suspect Interview",
                "description": "Initial interview with primary suspect",
                "event_type": "witness_interview",
                "timestamp": (base_date + timedelta(days=20)).isoformat(),
                "importance": "high",
                "location": "Police Station"
            }
        ],
        "relationships": [
            {
                "source_entity": "John Smith",
                "target_entity": "TechCorp Inc.",
                "relationship_type": "employed_by",
                "strength": 0.9,
                "confidence": 0.95,
                "description": "Former CFO of TechCorp"
            },
            {
                "source_entity": "Jane Doe",
                "target_entity": "TechCorp Inc.",
                "relationship_type": "employed_by",
                "strength": 0.7,
                "confidence": 0.8,
                "description": "Current accounting manager"
            }
        ]
    }


@click.group()
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
@click.option('--config', '-c', type=click.Path(exists=True), help='Configuration file path')
@click.pass_context
def cli(ctx, verbose, config):
    """
    Lemkin Evidence Dashboard Generator
    
    Professional evidence dashboard creation and visualization toolkit for legal professionals.
    Generate interactive dashboards, timelines, network graphs, and investigation metrics.
    """
    ctx.ensure_object(dict)
    
    # Set logging level
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        click.echo("Verbose logging enabled")
    
    # Load configuration
    if config:
        try:
            with open(config, 'r') as f:
                config_data = json.load(f)
            ctx.obj['config'] = DashboardConfig(**config_data)
            click.echo(f"Configuration loaded from: {config}")
        except Exception as e:
            click.echo(f"Error loading configuration: {e}", err=True)
            sys.exit(1)
    else:
        ctx.obj['config'] = DashboardConfig()


@cli.command()
@click.argument('case_id', type=str)
@click.option('--output-dir', '-o', help='Output directory for dashboard files')
@click.option('--case-data', '-d', help='JSON file containing case data')
@click.option('--format', '-f', type=click.Choice(['html', 'json']), default='html', 
              help='Output format (default: html)')
@click.option('--theme', '-t', default='professional', help='Dashboard theme')
@click.option('--include-analytics', is_flag=True, help='Include advanced analytics')
@click.option('--interactive', is_flag=True, help='Generate interactive features')
@click.pass_context
def generate_dashboard(ctx, case_id, output_dir, case_data, format, theme, include_analytics, interactive):
    """
    Generate comprehensive case overview dashboard.
    
    Creates professional dashboard with case summary, evidence tracking,
    entity relationships, and progress metrics suitable for legal presentations.
    
    Example:
        lemkin-dashboard generate-dashboard CASE-2024-001 --theme professional --interactive
    """
    try:
        config = ctx.obj['config']
        config.theme = theme
        config.enable_filtering = interactive
        config.enable_search = interactive
        
        # Setup output directory
        output_path = setup_output_directory(output_dir, "dashboard")
        
        # Load case data
        case_data_dict = load_case_data(case_data)
        
        click.echo(f"Generating case dashboard for: {case_id}")
        click.echo(f"Theme: {theme}")
        click.echo(f"Output format: {format}")
        click.echo(f"Interactive features: {'enabled' if interactive else 'disabled'}")
        
        # Parse case data into objects
        investigation_data = case_data_dict['investigation']
        investigation = Investigation(
            case_id=case_id,
            title=investigation_data['title'],
            description=investigation_data['description'],
            status=InvestigationStatus(investigation_data['status']),
            lead_investigator=investigation_data['lead_investigator'],
            team_members=investigation_data.get('team_members', []),
            case_priority=investigation_data.get('case_priority', 'medium'),
            completion_percentage=investigation_data.get('completion_percentage', 0.0),
            start_date=datetime.fromisoformat(investigation_data['start_date'])
        )
        
        # Parse evidence
        evidence_list = []
        for evidence_data in case_data_dict.get('evidence', []):
            evidence = Evidence(
                case_id=case_id,
                title=evidence_data['title'],
                description=evidence_data['description'],
                evidence_type=evidence_data['evidence_type'],
                status=EvidenceStatus(evidence_data['status']),
                collected_by=evidence_data['collected_by'],
                collected_at=datetime.fromisoformat(evidence_data['collected_at']),
                authenticity_verified=evidence_data.get('authenticity_verified', False),
                analysis_completed=evidence_data.get('analysis_completed', False),
                priority=evidence_data.get('priority', 'medium')
            )
            evidence_list.append(evidence)
        
        # Parse entities
        entities = []
        for entity_data in case_data_dict.get('entities', []):
            entity = Entity(
                case_id=case_id,
                name=entity_data['name'],
                entity_type=EntityType(entity_data['entity_type']),
                importance=entity_data.get('importance', 'medium'),
                role_in_case=entity_data.get('role_in_case'),
                subject_of_investigation=entity_data.get('subject_of_investigation', False),
                witness=entity_data.get('witness', False),
                primary_location=entity_data.get('primary_location')
            )
            entities.append(entity)
        
        # Parse events
        events = []
        for event_data in case_data_dict.get('events', []):
            event = Event(
                case_id=case_id,
                title=event_data['title'],
                description=event_data['description'],
                event_type=EventType(event_data['event_type']),
                timestamp=datetime.fromisoformat(event_data['timestamp']),
                importance=event_data.get('importance', 'medium'),
                location=event_data.get('location')
            )
            events.append(event)
        
        # Parse relationships
        relationships = []
        entity_lookup = {e.name: e.id for e in entities}
        for rel_data in case_data_dict.get('relationships', []):
            if (rel_data['source_entity'] in entity_lookup and 
                rel_data['target_entity'] in entity_lookup):
                relationship = Relationship(
                    case_id=case_id,
                    source_entity_id=entity_lookup[rel_data['source_entity']],
                    target_entity_id=entity_lookup[rel_data['target_entity']],
                    relationship_type=RelationshipType(rel_data['relationship_type']),
                    strength=rel_data.get('strength', 0.5),
                    confidence=rel_data.get('confidence', 1.0),
                    description=rel_data.get('description')
                )
                relationships.append(relationship)
        
        # Generate dashboard
        with click.progressbar(length=100, label='Generating dashboard') as bar:
            generator = DashboardGenerator(config)
            dashboard = generator.generate_case_dashboard(
                case_id=case_id,
                investigation=investigation,
                evidence_list=evidence_list,
                entities=entities,
                events=events,
                relationships=relationships
            )
            bar.update(50)
            
            # Export dashboard
            if format == 'html':
                output_file = output_path / f"case_dashboard_{case_id}.html"
                success = generator.export_dashboard(dashboard, output_file, ExportFormat.HTML)
            else:
                output_file = output_path / f"case_dashboard_{case_id}.json"
                success = generator.export_dashboard(dashboard, output_file, ExportFormat.JSON)
            
            bar.update(100)
        
        # Include analytics if requested
        if include_analytics:
            click.echo("Generating advanced analytics...")
            
            # Generate timeline visualization
            timeline_visualizer = TimelineVisualizer(config)
            timeline = timeline_visualizer.create_interactive_timeline(case_id, events)
            timeline_fig = timeline_visualizer.generate_plotly_timeline(timeline)
            timeline_path = output_path / f"timeline_{case_id}.html"
            timeline_fig.write_html(str(timeline_path))
            
            # Generate network analysis
            network_analyzer = NetworkAnalyzer(config)
            network_graph = network_analyzer.create_network_graph(entities, relationships, case_id)
            network_fig = network_analyzer.generate_plotly_network(network_graph)
            network_path = output_path / f"network_{case_id}.html"
            network_fig.write_html(str(network_path))
            
            # Generate metrics tracking
            metrics_tracker = InvestigationMetricsTracker(config)
            metrics_dashboard = metrics_tracker.track_investigation_metrics(investigation, evidence_list, events)
            metrics_fig = metrics_tracker.generate_progress_visualization(metrics_dashboard)
            metrics_path = output_path / f"metrics_{case_id}.html"
            metrics_fig.write_html(str(metrics_path))
            
            click.echo(f"Analytics generated:")
            click.echo(f"  Timeline: {timeline_path}")
            click.echo(f"  Network: {network_path}")
            click.echo(f"  Metrics: {metrics_path}")
        
        if success:
            click.echo(f"\nDashboard generated successfully!")
            click.echo(f"Output file: {output_file}")
            click.echo(f"Dashboard stats:")
            click.echo(f"  Evidence items: {len(evidence_list)}")
            click.echo(f"  Entities: {len(entities)}")
            click.echo(f"  Events: {len(events)}")
            click.echo(f"  Relationships: {len(relationships)}")
        else:
            click.echo("Dashboard generation failed", err=True)
            sys.exit(1)
        
    except Exception as e:
        logger.error(f"Dashboard generation failed: {str(e)}")
        if ctx.obj.get('config') and logging.getLogger().level <= logging.DEBUG:
            click.echo(traceback.format_exc(), err=True)
        click.echo(f"Error: {str(e)}", err=True)
        sys.exit(1)


@cli.command()
@click.argument('case_id', type=str)
@click.option('--output-dir', '-o', help='Output directory for timeline files')
@click.option('--case-data', '-d', help='JSON file containing case data')
@click.option('--layout', '-l', type=click.Choice(['scatter', 'gantt', 'multi_track']), 
              default='scatter', help='Timeline layout type')
@click.option('--granularity', '-g', type=click.Choice(['minute', 'hour', 'day', 'week', 'month']),
              help='Time granularity (auto-detected if not specified)')
@click.option('--interactive', is_flag=True, help='Generate interactive timeline')
@click.option('--include-analysis', is_flag=True, help='Include temporal pattern analysis')
@click.pass_context
def create_timeline(ctx, case_id, output_dir, case_data, layout, granularity, interactive, include_analysis):
    """
    Generate interactive timeline visualization of case events.
    
    Creates comprehensive timeline with event correlation, pattern analysis,
    and temporal clustering for investigation workflow visualization.
    
    Example:
        lemkin-dashboard create-timeline CASE-2024-001 --layout multi_track --interactive
    """
    try:
        config = ctx.obj['config']
        
        # Setup output directory
        output_path = setup_output_directory(output_dir, "timeline")
        
        # Load case data
        case_data_dict = load_case_data(case_data)
        
        click.echo(f"Creating timeline visualization for: {case_id}")
        click.echo(f"Layout: {layout}")
        click.echo(f"Interactive features: {'enabled' if interactive else 'disabled'}")
        
        # Parse events
        events = []
        for event_data in case_data_dict.get('events', []):
            event = Event(
                case_id=case_id,
                title=event_data['title'],
                description=event_data['description'],
                event_type=EventType(event_data['event_type']),
                timestamp=datetime.fromisoformat(event_data['timestamp']),
                importance=event_data.get('importance', 'medium'),
                location=event_data.get('location')
            )
            events.append(event)
        
        if not events:
            click.echo("No events found in case data", err=True)
            sys.exit(1)
        
        # Generate timeline
        with click.progressbar(length=100, label='Creating timeline') as bar:
            visualizer = TimelineVisualizer(config)
            timeline = visualizer.create_interactive_timeline(case_id, events)
            bar.update(50)
            
            # Generate visualization
            fig = visualizer.generate_plotly_timeline(timeline, layout_type=layout)
            timeline_path = output_path / f"timeline_{case_id}_{layout}.html"
            fig.write_html(str(timeline_path))
            bar.update(75)
            
            # Export timeline data
            timeline_data_path = output_path / f"timeline_data_{case_id}.json"
            visualizer.export_timeline(timeline, timeline_data_path, ExportFormat.JSON)
            bar.update(100)
        
        # Include analysis if requested
        if include_analysis:
            click.echo("Generating temporal pattern analysis...")
            
            # Generate pattern analysis
            patterns = visualizer.analyze_temporal_patterns(timeline)
            patterns_path = output_path / f"timeline_patterns_{case_id}.json"
            with open(patterns_path, 'w') as f:
                json.dump(patterns, f, indent=2, default=str)
            
            # Generate correlation matrix
            correlation_fig = visualizer.generate_correlation_matrix(timeline)
            correlation_path = output_path / f"timeline_correlations_{case_id}.html"
            correlation_fig.write_html(str(correlation_path))
            
            click.echo(f"Pattern analysis saved to: {patterns_path}")
            click.echo(f"Correlations saved to: {correlation_path}")
        
        click.echo(f"\nTimeline created successfully!")
        click.echo(f"Timeline visualization: {timeline_path}")
        click.echo(f"Events processed: {len(events)}")
        click.echo(f"Time range: {timeline.time_range_start.date()} to {timeline.time_range_end.date()}")
        
    except Exception as e:
        logger.error(f"Timeline creation failed: {str(e)}")
        if ctx.obj.get('config') and logging.getLogger().level <= logging.DEBUG:
            click.echo(traceback.format_exc(), err=True)
        click.echo(f"Error: {str(e)}", err=True)
        sys.exit(1)


@cli.command()
@click.argument('case_id', type=str)
@click.option('--output-dir', '-o', help='Output directory for network files')
@click.option('--case-data', '-d', help='JSON file containing case data')
@click.option('--layout', '-l', default='spring', help='Graph layout algorithm')
@click.option('--include-analysis', is_flag=True, help='Include network analysis metrics')
@click.option('--find-communities', is_flag=True, help='Detect communities in network')
@click.option('--key-players', is_flag=True, help='Identify key players analysis')
@click.pass_context
def graph_network(ctx, case_id, output_dir, case_data, layout, include_analysis, find_communities, key_players):
    """
    Create network graph visualization of entity relationships.
    
    Generates interactive network graphs with community detection, centrality analysis,
    and relationship mapping for investigation network analysis.
    
    Example:
        lemkin-dashboard graph-network CASE-2024-001 --layout spring --find-communities
    """
    try:
        config = ctx.obj['config']
        
        # Setup output directory
        output_path = setup_output_directory(output_dir, "network")
        
        # Load case data
        case_data_dict = load_case_data(case_data)
        
        click.echo(f"Creating network graph for: {case_id}")
        click.echo(f"Layout algorithm: {layout}")
        
        # Parse entities
        entities = []
        for entity_data in case_data_dict.get('entities', []):
            entity = Entity(
                case_id=case_id,
                name=entity_data['name'],
                entity_type=EntityType(entity_data['entity_type']),
                importance=entity_data.get('importance', 'medium'),
                role_in_case=entity_data.get('role_in_case'),
                subject_of_investigation=entity_data.get('subject_of_investigation', False),
                witness=entity_data.get('witness', False),
                primary_location=entity_data.get('primary_location')
            )
            entities.append(entity)
        
        # Parse relationships
        relationships = []
        entity_lookup = {e.name: e.id for e in entities}
        for rel_data in case_data_dict.get('relationships', []):
            if (rel_data['source_entity'] in entity_lookup and 
                rel_data['target_entity'] in entity_lookup):
                relationship = Relationship(
                    case_id=case_id,
                    source_entity_id=entity_lookup[rel_data['source_entity']],
                    target_entity_id=entity_lookup[rel_data['target_entity']],
                    relationship_type=RelationshipType(rel_data['relationship_type']),
                    strength=rel_data.get('strength', 0.5),
                    confidence=rel_data.get('confidence', 1.0),
                    description=rel_data.get('description')
                )
                relationships.append(relationship)
        
        if not entities:
            click.echo("No entities found in case data", err=True)
            sys.exit(1)
        
        # Generate network graph
        with click.progressbar(length=100, label='Creating network') as bar:
            analyzer = NetworkAnalyzer(config)
            network_graph = analyzer.create_network_graph(entities, relationships, case_id)
            bar.update(25)
            
            # Generate visualization
            fig = analyzer.generate_plotly_network(network_graph, layout_algorithm=layout)
            network_path = output_path / f"network_{case_id}_{layout}.html"
            fig.write_html(str(network_path))
            bar.update(50)
            
            # Build NetworkX graph for analysis
            G = analyzer.build_networkx_graph(network_graph)
            bar.update(75)
            
            # Export network data
            network_data = network_graph.dict()
            network_data_path = output_path / f"network_data_{case_id}.json"
            with open(network_data_path, 'w') as f:
                json.dump(network_data, f, indent=2, default=str)
            bar.update(100)
        
        # Include analysis if requested
        analysis_results = {}
        
        if include_analysis:
            click.echo("Generating network analysis...")
            metrics = analyzer.analyze_network_metrics(G)
            metrics_path = output_path / f"network_metrics_{case_id}.json"
            with open(metrics_path, 'w') as f:
                json.dump(metrics, f, indent=2, default=str)
            analysis_results['metrics'] = metrics_path
            
        if find_communities:
            click.echo("Detecting communities...")
            communities = analyzer.detect_communities(G)
            communities_path = output_path / f"network_communities_{case_id}.json"
            with open(communities_path, 'w') as f:
                json.dump(communities, f, indent=2, default=str)
            analysis_results['communities'] = communities_path
            
        if key_players:
            click.echo("Identifying key players...")
            key_players_analysis = analyzer.analyze_key_players(G)
            key_players_path = output_path / f"network_key_players_{case_id}.json"
            with open(key_players_path, 'w') as f:
                json.dump(key_players_analysis, f, indent=2, default=str)
            analysis_results['key_players'] = key_players_path
        
        click.echo(f"\nNetwork graph created successfully!")
        click.echo(f"Network visualization: {network_path}")
        click.echo(f"Entities: {len(entities)}")
        click.echo(f"Relationships: {len(relationships)}")
        click.echo(f"Network density: {G.number_of_edges() / (G.number_of_nodes() * (G.number_of_nodes() - 1) / 2) * 100:.1f}%")
        
        if analysis_results:
            click.echo(f"Analysis files:")
            for analysis_type, path in analysis_results.items():
                click.echo(f"  {analysis_type.title()}: {path}")
        
    except Exception as e:
        logger.error(f"Network graph creation failed: {str(e)}")
        if ctx.obj.get('config') and logging.getLogger().level <= logging.DEBUG:
            click.echo(traceback.format_exc(), err=True)
        click.echo(f"Error: {str(e)}", err=True)
        sys.exit(1)


@cli.command()
@click.argument('case_id', type=str)
@click.option('--output-dir', '-o', help='Output directory for metrics files')
@click.option('--case-data', '-d', help='JSON file containing case data')
@click.option('--include-trends', is_flag=True, help='Include trend analysis')
@click.option('--include-forecasting', is_flag=True, help='Include completion forecasting')
@click.option('--include-risk-analysis', is_flag=True, help='Include risk assessment')
@click.pass_context
def track_metrics(ctx, case_id, output_dir, case_data, include_trends, include_forecasting, include_risk_analysis):
    """
    Generate investigation progress and metrics tracking dashboard.
    
    Creates comprehensive metrics analysis including progress tracking, quality assessment,
    productivity analysis, and risk evaluation for investigation management.
    
    Example:
        lemkin-dashboard track-metrics CASE-2024-001 --include-trends --include-risk-analysis
    """
    try:
        config = ctx.obj['config']
        
        # Setup output directory
        output_path = setup_output_directory(output_dir, "metrics")
        
        # Load case data
        case_data_dict = load_case_data(case_data)
        
        click.echo(f"Tracking metrics for: {case_id}")
        
        # Parse investigation data (similar to generate_dashboard)
        investigation_data = case_data_dict['investigation']
        investigation = Investigation(
            case_id=case_id,
            title=investigation_data['title'],
            description=investigation_data['description'],
            status=InvestigationStatus(investigation_data['status']),
            lead_investigator=investigation_data['lead_investigator'],
            team_members=investigation_data.get('team_members', []),
            case_priority=investigation_data.get('case_priority', 'medium'),
            completion_percentage=investigation_data.get('completion_percentage', 0.0),
            start_date=datetime.fromisoformat(investigation_data['start_date'])
        )
        
        # Parse evidence
        evidence_list = []
        for evidence_data in case_data_dict.get('evidence', []):
            evidence = Evidence(
                case_id=case_id,
                title=evidence_data['title'],
                description=evidence_data['description'],
                evidence_type=evidence_data['evidence_type'],
                status=EvidenceStatus(evidence_data['status']),
                collected_by=evidence_data['collected_by'],
                collected_at=datetime.fromisoformat(evidence_data['collected_at']),
                authenticity_verified=evidence_data.get('authenticity_verified', False),
                analysis_completed=evidence_data.get('analysis_completed', False),
                priority=evidence_data.get('priority', 'medium')
            )
            evidence_list.append(evidence)
        
        # Parse events
        events = []
        for event_data in case_data_dict.get('events', []):
            event = Event(
                case_id=case_id,
                title=event_data['title'],
                description=event_data['description'],
                event_type=EventType(event_data['event_type']),
                timestamp=datetime.fromisoformat(event_data['timestamp']),
                importance=event_data.get('importance', 'medium'),
                location=event_data.get('location')
            )
            events.append(event)
        
        # Generate metrics
        with click.progressbar(length=100, label='Tracking metrics') as bar:
            tracker = InvestigationMetricsTracker(config)
            metrics_dashboard = tracker.track_investigation_metrics(investigation, evidence_list, events)
            bar.update(25)
            
            # Generate progress visualization
            progress_fig = tracker.generate_progress_visualization(metrics_dashboard)
            progress_path = output_path / f"metrics_progress_{case_id}.html"
            progress_fig.write_html(str(progress_path))
            bar.update(50)
            
            # Generate productivity analysis
            productivity_fig = tracker.generate_productivity_analysis(metrics_dashboard)
            productivity_path = output_path / f"metrics_productivity_{case_id}.html"
            productivity_fig.write_html(str(productivity_path))
            bar.update(75)
            
            # Export metrics data
            metrics_data = metrics_dashboard.dict()
            metrics_data_path = output_path / f"metrics_data_{case_id}.json"
            with open(metrics_data_path, 'w') as f:
                json.dump(metrics_data, f, indent=2, default=str)
            bar.update(100)
        
        # Additional analyses
        analysis_results = {
            'progress': progress_path,
            'productivity': productivity_path,
            'data': metrics_data_path
        }
        
        if include_trends:
            click.echo("Generating trend analysis...")
            # Generate quality assessment
            quality_fig = tracker.generate_quality_assessment(evidence_list)
            quality_path = output_path / f"metrics_quality_{case_id}.html"
            quality_fig.write_html(str(quality_path))
            analysis_results['quality'] = quality_path
            
        if include_risk_analysis:
            click.echo("Generating risk analysis...")
            risk_fig = tracker.generate_risk_analysis(investigation, evidence_list, events)
            risk_path = output_path / f"metrics_risk_{case_id}.html"
            risk_fig.write_html(str(risk_path))
            analysis_results['risk'] = risk_path
        
        click.echo(f"\nMetrics tracking completed successfully!")
        click.echo(f"Overall progress: {metrics_dashboard.overall_progress:.1f}%")
        click.echo(f"Evidence quality: {metrics_dashboard.evidence_quality_score:.1f}%")
        click.echo(f"Team productivity: {metrics_dashboard.productivity_trend}")
        click.echo(f"Days since start: {metrics_dashboard.days_since_start}")
        
        if metrics_dashboard.days_until_deadline:
            click.echo(f"Days until deadline: {metrics_dashboard.days_until_deadline}")
        
        click.echo(f"\nGenerated files:")
        for analysis_type, path in analysis_results.items():
            click.echo(f"  {analysis_type.title()}: {path}")
        
    except Exception as e:
        logger.error(f"Metrics tracking failed: {str(e)}")
        if ctx.obj.get('config') and logging.getLogger().level <= logging.DEBUG:
            click.echo(traceback.format_exc(), err=True)
        click.echo(f"Error: {str(e)}", err=True)
        sys.exit(1)


@cli.command()
@click.argument('case_id', type=str)
@click.option('--case-data', '-d', help='JSON file containing case data')
@click.option('--port', '-p', type=int, default=8501, help='Server port (default: 8501)')
@click.option('--host', '-h', default='localhost', help='Server host (default: localhost)')
@click.option('--open-browser', is_flag=True, help='Open browser automatically')
@click.pass_context
def serve(ctx, case_id, case_data, port, host, open_browser):
    """
    Launch interactive dashboard server using Streamlit.
    
    Starts a web server with interactive dashboard interface for real-time
    case analysis, collaboration, and presentation capabilities.
    
    Example:
        lemkin-dashboard serve CASE-2024-001 --port 8501 --open-browser
    """
    try:
        click.echo(f"Starting dashboard server for case: {case_id}")
        click.echo(f"Server: http://{host}:{port}")
        
        # Check if Streamlit is available
        try:
            import streamlit
        except ImportError:
            click.echo("Streamlit is required for dashboard server. Install with: pip install streamlit", err=True)
            sys.exit(1)
        
        # Create temporary app file
        app_content = f"""
import sys
sys.path.append('{Path(__file__).parent}')

from lemkin_dashboard.case_dashboard import create_streamlit_case_dashboard

# Set page config
import streamlit as st
st.set_page_config(
    page_title="Lemkin Dashboard - {case_id}",
    page_icon="⚖️",
    layout="wide"
)

# Create dashboard
create_streamlit_case_dashboard("{case_id}")
"""
        
        app_file = Path("temp_dashboard_app.py")
        with open(app_file, 'w') as f:
            f.write(app_content)
        
        # Open browser if requested
        if open_browser:
            def open_browser_delayed():
                time.sleep(3)  # Wait for server to start
                webbrowser.open(f"http://{host}:{port}")
            
            threading.Thread(target=open_browser_delayed, daemon=True).start()
        
        # Start Streamlit server
        try:
            subprocess.run([
                sys.executable, "-m", "streamlit", "run", str(app_file),
                "--server.port", str(port),
                "--server.address", host,
                "--server.headless", "true" if not open_browser else "false"
            ], check=True)
        finally:
            # Clean up temporary file
            if app_file.exists():
                app_file.unlink()
        
    except KeyboardInterrupt:
        click.echo("\nDashboard server stopped")
    except Exception as e:
        logger.error(f"Dashboard server failed: {str(e)}")
        click.echo(f"Error: {str(e)}", err=True)
        sys.exit(1)


@cli.command()
@click.argument('dashboard_file', type=click.Path(exists=True))
@click.option('--output-dir', '-o', help='Output directory for exported files')
@click.option('--format', '-f', type=click.Choice(['html', 'pdf', 'png', 'json']), 
              multiple=True, default=['html'], help='Export formats')
@click.option('--quality', '-q', type=click.Choice(['low', 'medium', 'high']), 
              default='high', help='Export quality')
@click.option('--include-metadata', is_flag=True, help='Include metadata in export')
@click.pass_context
def export(ctx, dashboard_file, output_dir, format, quality, include_metadata):
    """
    Export dashboards and visualizations to various formats.
    
    Exports generated dashboards to HTML, PDF, PNG, or JSON formats
    with configurable quality settings and metadata inclusion.
    
    Example:
        lemkin-dashboard export dashboard.json -f html -f pdf --quality high
    """
    try:
        # Setup output directory
        output_path = setup_output_directory(output_dir, "export")
        
        # Load dashboard file
        with open(dashboard_file, 'r') as f:
            dashboard_data = json.load(f)
        
        click.echo(f"Exporting dashboard from: {dashboard_file}")
        click.echo(f"Export formats: {', '.join(format)}")
        click.echo(f"Quality: {quality}")
        
        # Create dashboard object
        config = ctx.obj['config']
        generator = DashboardGenerator(config)
        
        export_results = {}
        
        # Export to each requested format
        for export_format in format:
            click.echo(f"Exporting to {export_format.upper()}...")
            
            output_file = output_path / f"exported_dashboard.{export_format}"
            
            # Create export options
            export_options = ExportOptions(
                format=ExportFormat(export_format),
                quality=quality,
                include_metadata=include_metadata
            )
            
            if export_format == 'html':
                # Generate HTML export
                success = generator.export_dashboard(
                    Dashboard(**dashboard_data), 
                    output_file, 
                    ExportFormat.HTML
                )
            elif export_format == 'json':
                # JSON export
                with open(output_file, 'w') as f:
                    json.dump(dashboard_data, f, indent=2, default=str)
                success = True
            else:
                click.echo(f"Export format {export_format} not fully implemented yet")
                success = False
            
            if success:
                export_results[export_format] = output_file
                click.echo(f"✅ {export_format.upper()} export completed: {output_file}")
            else:
                click.echo(f"❌ {export_format.upper()} export failed", err=True)
        
        click.echo(f"\nExport completed!")
        click.echo(f"Output directory: {output_path}")
        click.echo(f"Successful exports: {len(export_results)}/{len(format)}")
        
        if export_results:
            click.echo("Exported files:")
            for fmt, path in export_results.items():
                click.echo(f"  {fmt.upper()}: {path}")
        
    except Exception as e:
        logger.error(f"Export failed: {str(e)}")
        if ctx.obj.get('config') and logging.getLogger().level <= logging.DEBUG:
            click.echo(traceback.format_exc(), err=True)
        click.echo(f"Error: {str(e)}", err=True)
        sys.exit(1)


@cli.command()
@click.option('--output', '-o', default='lemkin_dashboard_config.json', help='Output configuration file')
def generate_config(output):
    """
    Generate a sample configuration file with all available options.
    
    Creates a JSON configuration file with default values and documentation
    that can be customized and used with the --config option.
    """
    config = DashboardConfig()
    config_dict = config.dict()
    
    # Add documentation
    documented_config = {
        "_description": "Lemkin Dashboard Configuration File",
        "_version": "1.0.0",
        "_documentation": {
            "display_settings": {
                "theme": "Visual theme: professional, modern, minimal, dark",
                "color_scheme": "Color scheme: default, blue, green, purple",
                "layout_style": "Layout style: responsive, fixed, fluid"
            },
            "interactive_features": {
                "enable_filtering": "Enable filtering capabilities in dashboards",
                "enable_search": "Enable search functionality",
                "enable_export": "Enable export capabilities",
                "enable_real_time_updates": "Enable real-time data updates"
            },
            "performance_settings": {
                "max_timeline_events": "Maximum events to display in timeline (1-10000)",
                "max_network_nodes": "Maximum nodes to display in network graph (1-2000)",
                "lazy_loading": "Enable lazy loading for large datasets"
            },
            "security_settings": {
                "require_authentication": "Require user authentication",
                "enable_audit_logging": "Enable audit logging",
                "data_encryption": "Enable data encryption"
            },
            "collaboration_settings": {
                "enable_comments": "Enable commenting on dashboard elements",
                "enable_sharing": "Enable dashboard sharing",
                "enable_version_control": "Enable dashboard version control"
            }
        },
        **config_dict
    }
    
    output_path = Path(output)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(documented_config, f, indent=2)
    
    click.echo(f"Configuration file generated: {output_path}")
    click.echo(f"Edit this file and use with: --config {output_path}")


if __name__ == '__main__':
    cli()