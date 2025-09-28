"""
Lemkin Timeline Visualizer Module

Interactive timeline visualization using Plotly and Bokeh for legal case presentation.
Provides comprehensive timeline analysis, event correlation, and temporal pattern
recognition for investigation workflows.
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
from bokeh.plotting import figure, save, output_file
from bokeh.models import HoverTool, DatetimeTickFormatter, ColumnDataSource, LegendItem, Legend
from bokeh.palettes import Category20
from bokeh.layouts import column, row
from bokeh.io import curdoc
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
import logging
from pathlib import Path
import json

from .core import (
    TimelineVisualization, Event, EventType, DashboardConfig,
    ExportOptions, ExportFormat, VisualizationSettings
)

# Configure logging
logger = logging.getLogger(__name__)


class TimelineVisualizer:
    """
    Advanced timeline visualization system for legal case events and evidence.
    
    Provides multiple visualization modes:
    - Interactive timeline with zoom and pan
    - Multi-track timeline for different event categories
    - Timeline correlation analysis
    - Event pattern detection
    - Temporal clustering
    """
    
    def __init__(self, config: Optional[DashboardConfig] = None):
        """Initialize timeline visualizer with configuration"""
        self.config = config or DashboardConfig()
        self.logger = logging.getLogger(f"{__name__}.TimelineVisualizer")
        
        # Color schemes for different event types
        self.event_colors = {
            EventType.INCIDENT: "#ef4444",
            EventType.EVIDENCE_COLLECTED: "#3b82f6", 
            EventType.WITNESS_INTERVIEW: "#10b981",
            EventType.LEGAL_ACTION: "#8b5cf6",
            EventType.ANALYSIS_COMPLETED: "#f59e0b",
            EventType.COURT_PROCEEDING: "#ec4899",
            EventType.SETTLEMENT: "#06b6d4",
            EventType.DEADLINE: "#dc2626",
            EventType.MILESTONE: "#059669",
            EventType.COMMUNICATION: "#6366f1"
        }
        
        self.logger.info("Timeline Visualizer initialized")
    
    def create_interactive_timeline(self, case_id: str, events: List[Event], 
                                   title: Optional[str] = None,
                                   visualization_settings: Optional[VisualizationSettings] = None) -> TimelineVisualization:
        """
        Create comprehensive interactive timeline visualization
        
        Args:
            case_id: Case identifier
            events: List of events to visualize
            title: Timeline title
            visualization_settings: Specific visualization settings
            
        Returns:
            TimelineVisualization: Complete timeline visualization object
        """
        if not events:
            raise ValueError("Cannot create timeline with empty events list")
        
        # Sort events by timestamp
        sorted_events = sorted(events, key=lambda e: e.timestamp)
        
        # Calculate time range
        time_range_start = min(e.timestamp for e in events)
        time_range_end = max(e.timestamp for e in events)
        
        # Determine appropriate granularity
        time_span = time_range_end - time_range_start
        if time_span.days < 1:
            granularity = "hour"
        elif time_span.days < 30:
            granularity = "day"
        elif time_span.days < 365:
            granularity = "week"
        else:
            granularity = "month"
        
        # Create timeline visualization object
        timeline = TimelineVisualization(
            case_id=case_id,
            title=title or f"Interactive Timeline: {case_id}",
            events=sorted_events,
            time_range_start=time_range_start,
            time_range_end=time_range_end,
            granularity=granularity,
            filterable_categories=[event_type.value for event_type in EventType],
            searchable_fields=["title", "description", "participants", "location"]
        )
        
        self.logger.info(f"Created interactive timeline with {len(events)} events for case: {case_id}")
        return timeline
    
    def generate_plotly_timeline(self, timeline: TimelineVisualization, 
                                layout_type: str = "scatter") -> go.Figure:
        """
        Generate Plotly timeline visualization
        
        Args:
            timeline: Timeline visualization object
            layout_type: Type of layout ("scatter", "gantt", "multi_track")
            
        Returns:
            go.Figure: Plotly figure object
        """
        if layout_type == "scatter":
            return self._create_scatter_timeline(timeline)
        elif layout_type == "gantt":
            return self._create_gantt_timeline(timeline)
        elif layout_type == "multi_track":
            return self._create_multi_track_timeline(timeline)
        else:
            raise ValueError(f"Unsupported layout type: {layout_type}")
    
    def generate_bokeh_timeline(self, timeline: TimelineVisualization,
                               output_path: Optional[Path] = None) -> Any:
        """
        Generate Bokeh timeline visualization
        
        Args:
            timeline: Timeline visualization object
            output_path: Optional path to save HTML output
            
        Returns:
            Bokeh figure object
        """
        events = timeline.events
        
        # Prepare data for Bokeh
        event_data = {
            'timestamp': [e.timestamp for e in events],
            'title': [e.title for e in events],
            'description': [e.description for e in events],
            'event_type': [e.event_type.value for e in events],
            'importance': [e.importance for e in events],
            'location': [e.location or "Not specified" for e in events],
            'participants': [", ".join(e.participants) for e in events],
            'color': [self.event_colors.get(e.event_type, "#6b7280") for e in events]
        }
        
        source = ColumnDataSource(data=event_data)
        
        # Create figure
        p = figure(
            title=timeline.title,
            x_axis_type='datetime',
            width=1200,
            height=600,
            tools="pan,wheel_zoom,box_zoom,reset,save"
        )
        
        # Add scatter plot
        scatter = p.scatter(
            'timestamp', 'title', 
            source=source,
            size=12,
            color='color',
            alpha=0.7
        )
        
        # Add hover tool
        hover = HoverTool(
            tooltips=[
                ('Event', '@title'),
                ('Type', '@event_type'),
                ('Date', '@timestamp{%Y-%m-%d %H:%M}'),
                ('Description', '@description'),
                ('Location', '@location'),
                ('Participants', '@participants'),
                ('Importance', '@importance')
            ],
            formatters={'@timestamp': 'datetime'},
            renderers=[scatter]
        )
        p.add_tools(hover)
        
        # Format axes
        p.xaxis.formatter = DatetimeTickFormatter(
            hours="%H:%M",
            days="%Y-%m-%d",
            months="%Y-%m",
            years="%Y"
        )
        p.xaxis.axis_label = "Date"
        p.yaxis.axis_label = "Events"
        
        # Add styling
        p.title.text_font_size = "16pt"
        p.axis.axis_label_text_font_size = "12pt"
        p.axis.major_label_text_font_size = "10pt"
        
        # Save if output path provided
        if output_path:
            output_file(str(output_path))
            save(p)
            self.logger.info(f"Bokeh timeline saved to: {output_path}")
        
        return p
    
    def _create_scatter_timeline(self, timeline: TimelineVisualization) -> go.Figure:
        """Create scatter plot timeline"""
        events = timeline.events
        
        # Prepare data
        timestamps = [e.timestamp for e in events]
        titles = [e.title for e in events]
        descriptions = [e.description for e in events]
        event_types = [e.event_type.value for e in events]
        importance_sizes = [self._get_importance_size(e.importance) for e in events]
        colors = [self.event_colors.get(e.event_type, "#6b7280") for e in events]
        
        # Create figure
        fig = go.Figure()
        
        # Group events by type for legend
        event_type_groups = {}
        for i, event in enumerate(events):
            event_type = event.event_type.value
            if event_type not in event_type_groups:
                event_type_groups[event_type] = {
                    'timestamps': [],
                    'titles': [],
                    'descriptions': [],
                    'sizes': [],
                    'color': self.event_colors.get(event.event_type, "#6b7280")
                }
            
            event_type_groups[event_type]['timestamps'].append(timestamps[i])
            event_type_groups[event_type]['titles'].append(titles[i])
            event_type_groups[event_type]['descriptions'].append(descriptions[i])
            event_type_groups[event_type]['sizes'].append(importance_sizes[i])
        
        # Add traces for each event type
        for event_type, data in event_type_groups.items():
            fig.add_trace(go.Scatter(
                x=data['timestamps'],
                y=data['titles'],
                mode='markers',
                name=event_type.replace('_', ' ').title(),
                marker=dict(
                    size=data['sizes'],
                    color=data['color'],
                    opacity=0.7,
                    line=dict(width=1, color='white')
                ),
                text=data['descriptions'],
                hovertemplate="<b>%{y}</b><br>" +
                            "Date: %{x}<br>" +
                            "Type: " + event_type + "<br>" +
                            "Description: %{text}<br>" +
                            "<extra></extra>"
            ))
        
        # Update layout
        fig.update_layout(
            title={
                'text': timeline.title,
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 20}
            },
            xaxis_title="Date",
            yaxis_title="Events",
            hovermode='closest',
            height=600,
            showlegend=True,
            legend=dict(
                orientation="v",
                yanchor="top",
                y=1,
                xanchor="left",
                x=1.02
            )
        )
        
        return fig
    
    def _create_gantt_timeline(self, timeline: TimelineVisualization) -> go.Figure:
        """Create Gantt chart timeline for duration events"""
        events = timeline.events
        
        # Filter events with duration
        duration_events = [e for e in events if e.end_timestamp is not None]
        
        if not duration_events:
            # Fallback to scatter if no duration events
            return self._create_scatter_timeline(timeline)
        
        # Prepare Gantt data
        df_data = []
        for event in duration_events:
            df_data.append({
                'Task': event.title,
                'Start': event.timestamp,
                'Finish': event.end_timestamp,
                'Type': event.event_type.value,
                'Description': event.description
            })
        
        df = pd.DataFrame(df_data)
        
        # Create Gantt chart
        fig = px.timeline(
            df, 
            x_start="Start", 
            x_end="Finish", 
            y="Task",
            color="Type",
            title=timeline.title,
            hover_data=["Description"]
        )
        
        fig.update_yaxes(autorange="reversed")
        fig.update_layout(height=600)
        
        return fig
    
    def _create_multi_track_timeline(self, timeline: TimelineVisualization) -> go.Figure:
        """Create multi-track timeline with separate rows for different categories"""
        events = timeline.events
        
        # Group events by type
        event_tracks = {}
        for event in events:
            event_type = event.event_type.value
            if event_type not in event_tracks:
                event_tracks[event_type] = []
            event_tracks[event_type].append(event)
        
        # Create subplots
        fig = make_subplots(
            rows=len(event_tracks),
            cols=1,
            subplot_titles=list(event_tracks.keys()),
            vertical_spacing=0.05,
            shared_xaxes=True
        )
        
        # Add traces for each track
        for i, (event_type, track_events) in enumerate(event_tracks.items(), 1):
            timestamps = [e.timestamp for e in track_events]
            titles = [e.title for e in track_events]
            descriptions = [e.description for e in track_events]
            
            fig.add_trace(
                go.Scatter(
                    x=timestamps,
                    y=[1] * len(timestamps),  # Fixed y position for each track
                    mode='markers+text',
                    name=event_type.replace('_', ' ').title(),
                    text=titles,
                    textposition="top center",
                    marker=dict(
                        size=10,
                        color=self.event_colors.get(EventType(event_type), "#6b7280")
                    ),
                    hovertemplate="<b>%{text}</b><br>" +
                                "Date: %{x}<br>" +
                                "<extra></extra>",
                    showlegend=False
                ),
                row=i, col=1
            )
        
        # Update layout
        fig.update_layout(
            title=timeline.title,
            height=150 * len(event_tracks),
            showlegend=False
        )
        
        # Update y-axes to hide ticks
        for i in range(1, len(event_tracks) + 1):
            fig.update_yaxes(showticklabels=False, row=i, col=1)
        
        return fig
    
    def analyze_temporal_patterns(self, timeline: TimelineVisualization) -> Dict[str, Any]:
        """
        Analyze temporal patterns in the timeline
        
        Args:
            timeline: Timeline visualization object
            
        Returns:
            Dict containing pattern analysis results
        """
        events = timeline.events
        
        if not events:
            return {"error": "No events to analyze"}
        
        # Convert to DataFrame for analysis
        df_data = []
        for event in events:
            df_data.append({
                'timestamp': event.timestamp,
                'event_type': event.event_type.value,
                'importance': event.importance,
                'hour': event.timestamp.hour,
                'day_of_week': event.timestamp.weekday(),
                'month': event.timestamp.month
            })
        
        df = pd.DataFrame(df_data)
        
        # Analyze patterns
        patterns = {
            'total_events': len(events),
            'time_span_days': (timeline.time_range_end - timeline.time_range_start).days,
            'event_frequency': len(events) / max(1, (timeline.time_range_end - timeline.time_range_start).days),
            'event_type_distribution': df['event_type'].value_counts().to_dict(),
            'importance_distribution': df['importance'].value_counts().to_dict(),
            'hourly_distribution': df['hour'].value_counts().sort_index().to_dict(),
            'daily_distribution': df['day_of_week'].value_counts().sort_index().to_dict(),
            'monthly_distribution': df['month'].value_counts().sort_index().to_dict(),
            'peak_activity_periods': self._identify_peak_periods(df),
            'temporal_clusters': self._identify_temporal_clusters(events)
        }
        
        return patterns
    
    def generate_correlation_matrix(self, timeline: TimelineVisualization) -> go.Figure:
        """Generate correlation matrix for event types and temporal patterns"""
        events = timeline.events
        
        # Create correlation data
        df_data = []
        for event in events:
            df_data.append({
                'hour': event.timestamp.hour,
                'day_of_week': event.timestamp.weekday(),
                'month': event.timestamp.month,
                'event_type_code': list(EventType).index(event.event_type),
                'importance_code': ['low', 'medium', 'high', 'critical'].index(event.importance)
            })
        
        df = pd.DataFrame(df_data)
        
        if df.empty:
            return go.Figure().add_annotation(text="No data available for correlation analysis")
        
        # Calculate correlation matrix
        correlation_matrix = df.corr()
        
        # Create heatmap
        fig = px.imshow(
            correlation_matrix,
            title="Temporal Pattern Correlations",
            color_continuous_scale="RdBu",
            aspect="auto"
        )
        
        fig.update_layout(
            title="Event Pattern Correlation Matrix",
            height=500
        )
        
        return fig
    
    def export_timeline(self, timeline: TimelineVisualization, output_path: Path, 
                       format: ExportFormat = ExportFormat.HTML) -> bool:
        """
        Export timeline visualization to specified format
        
        Args:
            timeline: Timeline visualization object
            output_path: Path to save the export
            format: Export format
            
        Returns:
            bool: Success status
        """
        try:
            if format == ExportFormat.HTML:
                fig = self.generate_plotly_timeline(timeline)
                fig.write_html(str(output_path))
                
            elif format == ExportFormat.PNG:
                fig = self.generate_plotly_timeline(timeline)
                fig.write_image(str(output_path), format="png", width=1200, height=600)
                
            elif format == ExportFormat.SVG:
                fig = self.generate_plotly_timeline(timeline)
                fig.write_image(str(output_path), format="svg", width=1200, height=600)
                
            elif format == ExportFormat.JSON:
                timeline_data = timeline.dict()
                with open(output_path, 'w') as f:
                    json.dump(timeline_data, f, indent=2, default=str)
                    
            else:
                self.logger.warning(f"Export format {format} not supported")
                return False
            
            self.logger.info(f"Timeline exported to: {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to export timeline: {str(e)}")
            return False
    
    def _get_importance_size(self, importance: str) -> int:
        """Map importance level to marker size"""
        size_map = {
            "low": 8,
            "medium": 12,
            "high": 16,
            "critical": 20
        }
        return size_map.get(importance, 12)
    
    def _identify_peak_periods(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Identify peak activity periods"""
        peaks = []
        
        # Hourly peaks
        hourly_counts = df.groupby('hour').size()
        if not hourly_counts.empty:
            peak_hour = hourly_counts.idxmax()
            peaks.append({
                'type': 'hourly',
                'period': f"{peak_hour}:00",
                'event_count': hourly_counts[peak_hour]
            })
        
        # Daily peaks
        daily_counts = df.groupby('day_of_week').size()
        if not daily_counts.empty:
            peak_day = daily_counts.idxmax()
            day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            peaks.append({
                'type': 'daily',
                'period': day_names[peak_day],
                'event_count': daily_counts[peak_day]
            })
        
        return peaks
    
    def _identify_temporal_clusters(self, events: List[Event]) -> List[Dict[str, Any]]:
        """Identify temporal clusters of events"""
        if len(events) < 2:
            return []
        
        # Sort events by timestamp
        sorted_events = sorted(events, key=lambda e: e.timestamp)
        clusters = []
        current_cluster = [sorted_events[0]]
        
        # Group events within 24 hours of each other
        for i in range(1, len(sorted_events)):
            time_diff = sorted_events[i].timestamp - sorted_events[i-1].timestamp
            if time_diff.total_seconds() <= 24 * 3600:  # Within 24 hours
                current_cluster.append(sorted_events[i])
            else:
                if len(current_cluster) > 1:
                    clusters.append({
                        'start_time': current_cluster[0].timestamp,
                        'end_time': current_cluster[-1].timestamp,
                        'event_count': len(current_cluster),
                        'event_types': list(set([e.event_type.value for e in current_cluster]))
                    })
                current_cluster = [sorted_events[i]]
        
        # Add final cluster if it has multiple events
        if len(current_cluster) > 1:
            clusters.append({
                'start_time': current_cluster[0].timestamp,
                'end_time': current_cluster[-1].timestamp,
                'event_count': len(current_cluster),
                'event_types': list(set([e.event_type.value for e in current_cluster]))
            })
        
        return clusters


def create_interactive_timeline(case_id: str, events: List[Event], 
                               title: Optional[str] = None,
                               config: Optional[DashboardConfig] = None) -> TimelineVisualization:
    """
    Convenience function to create interactive timeline visualization
    
    Args:
        case_id: Case identifier
        events: List of events to visualize
        title: Optional timeline title
        config: Optional dashboard configuration
        
    Returns:
        TimelineVisualization: Complete timeline visualization
    """
    visualizer = TimelineVisualizer(config)
    return visualizer.create_interactive_timeline(case_id, events, title)


def generate_timeline_report(timeline: TimelineVisualization, 
                           output_dir: Path) -> Dict[str, Path]:
    """
    Generate comprehensive timeline analysis report
    
    Args:
        timeline: Timeline visualization object
        output_dir: Directory to save report files
        
    Returns:
        Dict mapping report types to file paths
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    visualizer = TimelineVisualizer()
    
    report_paths = {}
    
    try:
        # Generate main timeline visualization
        html_path = output_dir / f"timeline_{timeline.case_id}.html"
        visualizer.export_timeline(timeline, html_path, ExportFormat.HTML)
        report_paths['timeline_html'] = html_path
        
        # Generate pattern analysis
        patterns = visualizer.analyze_temporal_patterns(timeline)
        patterns_path = output_dir / f"timeline_patterns_{timeline.case_id}.json"
        with open(patterns_path, 'w') as f:
            json.dump(patterns, f, indent=2, default=str)
        report_paths['patterns'] = patterns_path
        
        # Generate correlation matrix
        correlation_fig = visualizer.generate_correlation_matrix(timeline)
        correlation_path = output_dir / f"timeline_correlations_{timeline.case_id}.html"
        correlation_fig.write_html(str(correlation_path))
        report_paths['correlations'] = correlation_path
        
        logger.info(f"Timeline report generated in: {output_dir}")
        
    except Exception as e:
        logger.error(f"Failed to generate timeline report: {str(e)}")
    
    return report_paths