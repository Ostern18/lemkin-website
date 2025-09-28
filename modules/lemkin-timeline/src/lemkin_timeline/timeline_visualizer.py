"""
Timeline visualization module supporting multiple interactive visualization libraries.
"""

import uuid
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path
import json
import logging
from loguru import logger

from .core import (
    Timeline, Event, TimelineVisualization, TimelineConfig,
    TimelineEventType, LanguageCode
)


class PlotlyVisualizer:
    """Creates interactive timelines using Plotly"""
    
    def __init__(self, config: TimelineConfig):
        self.config = config
        
        # Default color scheme
        self.default_colors = {
            TimelineEventType.INSTANT: '#1f77b4',
            TimelineEventType.DURATION: '#ff7f0e',
            TimelineEventType.START: '#2ca02c',
            TimelineEventType.END: '#d62728',
            TimelineEventType.MILESTONE: '#9467bd',
            TimelineEventType.SEQUENCE: '#8c564b',
            TimelineEventType.CONCURRENT: '#e377c2',
            TimelineEventType.RECURRING: '#7f7f7f'
        }
    
    def create_timeline_visualization(self, timeline: Timeline,
                                    visualization_config: Optional[Dict[str, Any]] = None) -> TimelineVisualization:
        """
        Create Plotly timeline visualization
        
        Args:
            timeline: Timeline to visualize
            visualization_config: Optional visualization configuration
            
        Returns:
            TimelineVisualization object with generated content
        """
        try:
            import plotly.graph_objects as go
            import plotly.express as px
            from plotly.subplots import make_subplots
            import plotly.offline as pyo
        except ImportError:
            logger.error("Plotly not available for timeline visualization")
            raise ImportError("Plotly is required for timeline visualization. Install with: pip install plotly")
        
        logger.info("Creating Plotly timeline visualization for {} events", len(timeline.events))
        
        config = visualization_config or {}
        
        # Create figure
        fig = go.Figure()
        
        # Process events for visualization
        events_data = self._prepare_events_data(timeline.events, config)
        
        # Add timeline traces
        self._add_timeline_traces(fig, events_data, config)
        
        # Customize layout
        self._customize_plotly_layout(fig, timeline, config)
        
        # Generate HTML
        html_content = self._generate_plotly_html(fig, timeline, config)
        
        # Create visualization object
        visualization = TimelineVisualization(
            timeline_id=timeline.timeline_id,
            visualization_type="plotly",
            title=config.get('title', f"Timeline: {timeline.title}"),
            width=config.get('width', 1200),
            height=config.get('height', 600),
            theme=config.get('theme', 'light'),
            show_uncertainty=config.get('show_uncertainty', True),
            show_connections=config.get('show_connections', True),
            show_tooltips=config.get('show_tooltips', True),
            html_content=html_content,
            json_data={'figure': fig.to_dict()},
            metadata={'events_count': len(timeline.events), 'library': 'plotly'}
        )
        
        logger.info("Plotly timeline visualization created successfully")
        return visualization
    
    def _prepare_events_data(self, events: List[Event], config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Prepare event data for Plotly visualization"""
        events_data = []
        
        for i, event in enumerate(events):
            # Calculate event position and duration
            start_time = event.start_time
            end_time = event.end_time or start_time
            duration = (end_time - start_time).total_seconds() / 3600  # hours
            
            # Determine color
            color = self._get_event_color(event, config)
            
            # Create hover text
            hover_text = self._create_hover_text(event)
            
            event_data = {
                'event_id': event.event_id,
                'title': event.title,
                'start_time': start_time,
                'end_time': end_time,
                'duration_hours': max(duration, 0.1),  # Minimum visual duration
                'color': color,
                'hover_text': hover_text,
                'y_position': i,
                'confidence': event.confidence,
                'is_fuzzy': event.is_fuzzy,
                'event_type': event.event_type.value,
                'participants': event.participants,
                'locations': event.locations
            }
            
            # Add uncertainty information
            if event.uncertainty_range and config.get('show_uncertainty', True):
                event_data['uncertainty_start'] = event.uncertainty_range[0]
                event_data['uncertainty_end'] = event.uncertainty_range[1]
            
            events_data.append(event_data)
        
        return events_data
    
    def _add_timeline_traces(self, fig: 'go.Figure', events_data: List[Dict[str, Any]], 
                           config: Dict[str, Any]) -> None:
        """Add timeline traces to Plotly figure"""
        import plotly.graph_objects as go
        
        # Add main event bars
        for event_data in events_data:
            # Main event bar
            fig.add_trace(go.Bar(
                x=[event_data['duration_hours']],
                y=[event_data['y_position']],
                base=[event_data['start_time']],
                orientation='h',
                name=event_data['title'],
                marker=dict(
                    color=event_data['color'],
                    opacity=0.8 if not event_data['is_fuzzy'] else 0.6,
                    line=dict(width=1, color='rgba(0,0,0,0.3)')
                ),
                text=event_data['title'],
                textposition='middle center',
                hovertext=event_data['hover_text'],
                hoverinfo='text',
                showlegend=False
            ))
            
            # Add uncertainty ranges if enabled
            if ('uncertainty_start' in event_data and 
                config.get('show_uncertainty', True)):
                
                uncertainty_duration = (
                    event_data['uncertainty_end'] - event_data['uncertainty_start']
                ).total_seconds() / 3600
                
                fig.add_trace(go.Bar(
                    x=[uncertainty_duration],
                    y=[event_data['y_position']],
                    base=[event_data['uncertainty_start']],
                    orientation='h',
                    marker=dict(
                        color=event_data['color'],
                        opacity=0.2,
                        line=dict(width=1, color='rgba(0,0,0,0.1)')
                    ),
                    hovertext=f"Uncertainty range for: {event_data['title']}",
                    hoverinfo='text',
                    showlegend=False
                ))
        
        # Add event connections if enabled
        if config.get('show_connections', True):
            self._add_event_connections(fig, events_data)
    
    def _add_event_connections(self, fig: 'go.Figure', events_data: List[Dict[str, Any]]) -> None:
        """Add connections between related events"""
        import plotly.graph_objects as go
        
        # Create lookup for events by ID
        event_lookup = {ed['event_id']: ed for ed in events_data}
        
        # Add lines for relationships
        for event_data in events_data:
            event_id = event_data['event_id']
            
            # Find the original event to get relationships
            # This would require passing the original events or storing relationships in events_data
            # For now, we'll add simple sequential connections
            if event_data['y_position'] > 0:
                prev_event = events_data[event_data['y_position'] - 1]
                
                # Add connection line
                fig.add_trace(go.Scatter(
                    x=[prev_event['end_time'], event_data['start_time']],
                    y=[prev_event['y_position'], event_data['y_position']],
                    mode='lines',
                    line=dict(color='rgba(128,128,128,0.3)', width=1, dash='dot'),
                    hoverinfo='skip',
                    showlegend=False
                ))
    
    def _customize_plotly_layout(self, fig: 'go.Figure', timeline: Timeline, 
                                config: Dict[str, Any]) -> None:
        """Customize Plotly figure layout"""
        
        # Determine theme colors
        theme = config.get('theme', 'light')
        if theme == 'dark':
            bg_color = '#2f2f2f'
            text_color = 'white'
            grid_color = 'rgba(255,255,255,0.2)'
        else:
            bg_color = 'white'
            text_color = 'black'
            grid_color = 'rgba(0,0,0,0.2)'
        
        fig.update_layout(
            title={
                'text': config.get('title', f"Timeline: {timeline.title}"),
                'x': 0.5,
                'font': {'size': 20, 'color': text_color}
            },
            xaxis={
                'title': 'Time',
                'type': 'date',
                'showgrid': True,
                'gridcolor': grid_color,
                'color': text_color
            },
            yaxis={
                'title': 'Events',
                'tickvals': list(range(len(timeline.events))),
                'ticktext': [event.title[:30] + '...' if len(event.title) > 30 
                           else event.title for event in timeline.events],
                'showgrid': True,
                'gridcolor': grid_color,
                'color': text_color
            },
            plot_bgcolor=bg_color,
            paper_bgcolor=bg_color,
            font={'color': text_color},
            height=config.get('height', max(400, len(timeline.events) * 40 + 200)),
            width=config.get('width', 1200),
            margin=dict(l=200, r=50, t=80, b=80),
            hovermode='closest'
        )
        
        # Add timeline metadata as annotation
        if timeline.start_date and timeline.end_date:
            duration = timeline.end_date - timeline.start_date
            metadata_text = (
                f"Timeline Span: {duration.days} days<br>"
                f"Events: {len(timeline.events)}<br>"
                f"Confidence: {timeline.confidence_score:.2f}"
            )
            
            fig.add_annotation(
                x=1, y=1,
                xref="paper", yref="paper",
                text=metadata_text,
                showarrow=False,
                xanchor="right",
                yanchor="top",
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor="black",
                borderwidth=1,
                font=dict(size=10)
            )
    
    def _generate_plotly_html(self, fig: 'go.Figure', timeline: Timeline, 
                             config: Dict[str, Any]) -> str:
        """Generate HTML content for Plotly visualization"""
        import plotly.offline as pyo
        
        # Generate HTML
        html_content = pyo.plot(
            fig, 
            output_type='div', 
            include_plotlyjs=True,
            div_id=f"timeline-{timeline.timeline_id}",
            config={
                'displayModeBar': True,
                'displaylogo': False,
                'modeBarButtonsToRemove': ['pan2d', 'lasso2d', 'select2d'],
                'responsive': True
            }
        )
        
        # Wrap in complete HTML document
        full_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <title>{config.get('title', 'Timeline Visualization')}</title>
            <style>
                body {{ margin: 20px; font-family: Arial, sans-serif; }}
                .timeline-header {{ margin-bottom: 20px; }}
                .timeline-info {{ font-size: 14px; color: #666; }}
            </style>
        </head>
        <body>
            <div class="timeline-header">
                <h1>{config.get('title', 'Timeline Visualization')}</h1>
                <div class="timeline-info">
                    Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} |
                    {len(timeline.events)} events |
                    Confidence: {timeline.confidence_score:.2%}
                </div>
            </div>
            {html_content}
        </body>
        </html>
        """
        
        return full_html
    
    def _get_event_color(self, event: Event, config: Dict[str, Any]) -> str:
        """Get color for event based on type and configuration"""
        color_scheme = config.get('color_scheme', {})
        
        # Check for custom color scheme
        if event.event_type.value in color_scheme:
            return color_scheme[event.event_type.value]
        
        # Use default colors
        return self.default_colors.get(event.event_type, '#1f77b4')
    
    def _create_hover_text(self, event: Event) -> str:
        """Create hover text for event"""
        hover_parts = [
            f"<b>{event.title}</b>",
            f"Start: {event.start_time.strftime('%Y-%m-%d %H:%M:%S')}"
        ]
        
        if event.end_time and event.end_time != event.start_time:
            hover_parts.append(f"End: {event.end_time.strftime('%Y-%m-%d %H:%M:%S')}")
            duration = event.end_time - event.start_time
            hover_parts.append(f"Duration: {self._format_duration(duration)}")
        
        if event.description:
            hover_parts.append(f"Description: {event.description[:100]}...")
        
        if event.participants:
            hover_parts.append(f"Participants: {', '.join(event.participants[:3])}")
        
        if event.locations:
            hover_parts.append(f"Locations: {', '.join(event.locations[:3])}")
        
        hover_parts.append(f"Confidence: {event.confidence:.2%}")
        
        if event.is_fuzzy:
            hover_parts.append("⚠️ Uncertain timing")
        
        return "<br>".join(hover_parts)
    
    def _format_duration(self, duration: timedelta) -> str:
        """Format duration for display"""
        total_seconds = int(duration.total_seconds())
        
        if total_seconds < 60:
            return f"{total_seconds} seconds"
        elif total_seconds < 3600:
            minutes = total_seconds // 60
            return f"{minutes} minutes"
        elif total_seconds < 86400:
            hours = total_seconds // 3600
            return f"{hours} hours"
        else:
            days = total_seconds // 86400
            return f"{days} days"


class BokehVisualizer:
    """Creates interactive timelines using Bokeh"""
    
    def __init__(self, config: TimelineConfig):
        self.config = config
    
    def create_timeline_visualization(self, timeline: Timeline,
                                    visualization_config: Optional[Dict[str, Any]] = None) -> TimelineVisualization:
        """
        Create Bokeh timeline visualization
        
        Args:
            timeline: Timeline to visualize
            visualization_config: Optional visualization configuration
            
        Returns:
            TimelineVisualization object with generated content
        """
        try:
            from bokeh.plotting import figure, save, output_file
            from bokeh.models import HoverTool, ColumnDataSource, DatetimeTickFormatter
            from bokeh.layouts import column
            from bokeh.embed import file_html
            from bokeh.resources import CDN
            import pandas as pd
        except ImportError:
            logger.error("Bokeh not available for timeline visualization")
            raise ImportError("Bokeh is required for timeline visualization. Install with: pip install bokeh")
        
        logger.info("Creating Bokeh timeline visualization for {} events", len(timeline.events))
        
        config = visualization_config or {}
        
        # Prepare data
        events_data = self._prepare_bokeh_data(timeline.events)
        source = ColumnDataSource(events_data)
        
        # Create figure
        p = figure(
            title=config.get('title', f"Timeline: {timeline.title}"),
            x_axis_type='datetime',
            width=config.get('width', 1200),
            height=config.get('height', 600),
            tools="pan,wheel_zoom,box_zoom,reset,save"
        )
        
        # Add timeline elements
        self._add_bokeh_timeline_elements(p, source, config)
        
        # Customize appearance
        self._customize_bokeh_appearance(p, timeline, config)
        
        # Generate HTML
        html_content = self._generate_bokeh_html(p, timeline, config)
        
        # Create visualization object
        visualization = TimelineVisualization(
            timeline_id=timeline.timeline_id,
            visualization_type="bokeh",
            title=config.get('title', f"Timeline: {timeline.title}"),
            width=config.get('width', 1200),
            height=config.get('height', 600),
            theme=config.get('theme', 'light'),
            html_content=html_content,
            metadata={'events_count': len(timeline.events), 'library': 'bokeh'}
        )
        
        logger.info("Bokeh timeline visualization created successfully")
        return visualization
    
    def _prepare_bokeh_data(self, events: List[Event]) -> Dict[str, List[Any]]:
        """Prepare event data for Bokeh visualization"""
        data = {
            'start_times': [],
            'end_times': [],
            'titles': [],
            'descriptions': [],
            'y_positions': [],
            'colors': [],
            'confidences': [],
            'is_fuzzy': []
        }
        
        for i, event in enumerate(events):
            data['start_times'].append(event.start_time)
            data['end_times'].append(event.end_time or event.start_time)
            data['titles'].append(event.title)
            data['descriptions'].append(event.description or '')
            data['y_positions'].append(i)
            data['colors'].append(self._get_bokeh_color(event))
            data['confidences'].append(event.confidence)
            data['is_fuzzy'].append(event.is_fuzzy)
        
        return data
    
    def _add_bokeh_timeline_elements(self, p: 'figure', source: 'ColumnDataSource',
                                   config: Dict[str, Any]) -> None:
        """Add timeline elements to Bokeh figure"""
        from bokeh.models import HoverTool
        
        # Add event rectangles
        p.quad(
            top='y_positions', bottom=[y-0.4 for y in source.data['y_positions']],
            left='start_times', right='end_times',
            color='colors', alpha=0.7,
            source=source
        )
        
        # Add hover tool
        hover = HoverTool(
            tooltips=[
                ('Event', '@titles'),
                ('Start', '@start_times{%F %T}'),
                ('End', '@end_times{%F %T}'),
                ('Description', '@descriptions'),
                ('Confidence', '@confidences{0.0%}'),
            ],
            formatters={
                '@start_times': 'datetime',
                '@end_times': 'datetime',
            }
        )
        p.add_tools(hover)
    
    def _customize_bokeh_appearance(self, p: 'figure', timeline: Timeline,
                                  config: Dict[str, Any]) -> None:
        """Customize Bokeh figure appearance"""
        from bokeh.models import DatetimeTickFormatter
        
        # Customize axes
        p.xaxis.formatter = DatetimeTickFormatter(
            hours=["%d %B %Y %H:%M"],
            days=["%d %B %Y"],
            months=["%B %Y"],
            years=["%Y"]
        )
        p.xaxis.major_label_orientation = 45
        
        # Y-axis labels
        p.yaxis.ticker = list(range(len(timeline.events)))
        p.yaxis.major_label_overrides = {
            i: event.title[:30] + ('...' if len(event.title) > 30 else '')
            for i, event in enumerate(timeline.events)
        }
        
        # Grid and styling
        p.grid.grid_line_alpha = 0.3
        p.title.text_font_size = "16pt"
        p.title.align = "center"
    
    def _generate_bokeh_html(self, p: 'figure', timeline: Timeline,
                           config: Dict[str, Any]) -> str:
        """Generate HTML content for Bokeh visualization"""
        from bokeh.embed import file_html
        from bokeh.resources import CDN
        
        html_content = file_html(p, CDN, config.get('title', 'Timeline Visualization'))
        return html_content
    
    def _get_bokeh_color(self, event: Event) -> str:
        """Get color for event in Bokeh visualization"""
        color_map = {
            TimelineEventType.INSTANT: '#1f77b4',
            TimelineEventType.DURATION: '#ff7f0e',
            TimelineEventType.START: '#2ca02c',
            TimelineEventType.END: '#d62728',
            TimelineEventType.MILESTONE: '#9467bd',
            TimelineEventType.SEQUENCE: '#8c564b',
            TimelineEventType.CONCURRENT: '#e377c2',
            TimelineEventType.RECURRING: '#7f7f7f'
        }
        return color_map.get(event.event_type, '#1f77b4')


class TimelineVisualizer:
    """Main timeline visualization coordinator"""
    
    def __init__(self, config: TimelineConfig):
        self.config = config
        self.plotly_visualizer = PlotlyVisualizer(config)
        self.bokeh_visualizer = BokehVisualizer(config)
        
        # Available export formats by visualization type
        self.export_formats = {
            'plotly': ['html', 'png', 'svg', 'pdf', 'json'],
            'bokeh': ['html', 'png', 'svg']
        }
    
    def generate_interactive_timeline(self, timeline: Timeline,
                                    visualization_type: str = "plotly",
                                    **kwargs) -> TimelineVisualization:
        """
        Generate interactive timeline visualization
        
        Args:
            timeline: Timeline to visualize
            visualization_type: Type of visualization ('plotly', 'bokeh')
            **kwargs: Additional visualization configuration
            
        Returns:
            TimelineVisualization object
        """
        if not timeline.events:
            logger.warning("Cannot visualize timeline with no events")
            raise ValueError("Timeline must contain at least one event for visualization")
        
        logger.info("Generating {} timeline visualization for timeline: {}",
                   visualization_type, timeline.timeline_id)
        
        # Merge configuration
        viz_config = {
            'width': kwargs.get('width', 1200),
            'height': kwargs.get('height', 600),
            'title': kwargs.get('title', timeline.title),
            'theme': kwargs.get('theme', self.config.default_theme),
            'show_uncertainty': kwargs.get('show_uncertainty', True),
            'show_connections': kwargs.get('show_connections', True),
            'show_tooltips': kwargs.get('show_tooltips', True),
            'color_scheme': kwargs.get('color_scheme', {}),
        }
        
        try:
            if visualization_type.lower() == 'plotly':
                return self.plotly_visualizer.create_timeline_visualization(timeline, viz_config)
            elif visualization_type.lower() == 'bokeh':
                return self.bokeh_visualizer.create_timeline_visualization(timeline, viz_config)
            else:
                raise ValueError(f"Unsupported visualization type: {visualization_type}")
                
        except Exception as e:
            logger.error("Error generating timeline visualization: {}", e)
            raise
    
    def export_visualization(self, visualization: TimelineVisualization,
                           output_path: Union[str, Path],
                           format: str = 'html') -> None:
        """
        Export timeline visualization to file
        
        Args:
            visualization: Visualization to export
            output_path: Output file path
            format: Export format
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info("Exporting {} visualization to {} as {}",
                   visualization.visualization_type, output_path, format)
        
        try:
            if format.lower() == 'html':
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(visualization.html_content)
            
            elif format.lower() == 'json':
                export_data = {
                    'visualization': visualization.to_dict(),
                    'metadata': {
                        'exported_at': datetime.now(timezone.utc).isoformat(),
                        'format': format,
                        'library': visualization.visualization_type
                    }
                }
                
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(export_data, f, indent=2, default=str)
            
            elif format.lower() in ['png', 'svg', 'pdf']:
                self._export_image_format(visualization, output_path, format)
            
            else:
                raise ValueError(f"Unsupported export format: {format}")
            
            logger.info("Visualization exported successfully to: {}", output_path)
            
        except Exception as e:
            logger.error("Error exporting visualization: {}", e)
            raise
    
    def _export_image_format(self, visualization: TimelineVisualization,
                           output_path: Path, format: str) -> None:
        """Export visualization as image format"""
        if visualization.visualization_type == 'plotly':
            try:
                import plotly.io as pio
                import kaleido  # Required for image export
                
                # Reconstruct figure from JSON data
                fig_dict = visualization.json_data.get('figure', {})
                if fig_dict:
                    import plotly.graph_objects as go
                    fig = go.Figure(fig_dict)
                    
                    if format.lower() == 'png':
                        fig.write_image(str(output_path), format='png')
                    elif format.lower() == 'svg':
                        fig.write_image(str(output_path), format='svg')
                    elif format.lower() == 'pdf':
                        fig.write_image(str(output_path), format='pdf')
                else:
                    raise ValueError("No figure data available for image export")
                    
            except ImportError:
                logger.error("kaleido package required for Plotly image export")
                raise ImportError("Install kaleido for image export: pip install kaleido")
        
        elif visualization.visualization_type == 'bokeh':
            try:
                from bokeh.io import export_png, export_svgs
                
                # This would require reconstructing the Bokeh figure
                # For now, raise an informative error
                raise NotImplementedError(
                    "Image export for Bokeh visualizations not yet implemented. "
                    "Use HTML export instead."
                )
                
            except ImportError:
                logger.error("Additional packages required for Bokeh image export")
                raise
        
        else:
            raise ValueError(f"Image export not supported for {visualization.visualization_type}")
    
    def create_comparison_visualization(self, timelines: List[Timeline],
                                      visualization_type: str = "plotly",
                                      **kwargs) -> TimelineVisualization:
        """
        Create visualization comparing multiple timelines
        
        Args:
            timelines: List of timelines to compare
            visualization_type: Type of visualization
            **kwargs: Additional configuration
            
        Returns:
            Comparative timeline visualization
        """
        if not timelines:
            raise ValueError("At least one timeline required for comparison")
        
        logger.info("Creating comparison visualization for {} timelines", len(timelines))
        
        # Merge all events with timeline identifiers
        all_events = []
        for i, timeline in enumerate(timelines):
            for event in timeline.events:
                # Create copy with timeline info
                comparison_event = Event(
                    title=f"[T{i+1}] {event.title}",
                    event_type=event.event_type,
                    start_time=event.start_time,
                    end_time=event.end_time,
                    document_id=event.document_id,
                    description=event.description,
                    participants=event.participants,
                    locations=event.locations,
                    confidence=event.confidence,
                    is_fuzzy=event.is_fuzzy,
                    metadata={**event.metadata, 'timeline_index': i, 'timeline_id': timeline.timeline_id}
                )
                all_events.append(comparison_event)
        
        # Create combined timeline
        combined_timeline = Timeline(
            title=kwargs.get('title', f"Comparison of {len(timelines)} Timelines"),
            events=sorted(all_events, key=lambda x: x.start_time),
            construction_method="comparison_merge"
        )
        
        # Generate visualization
        return self.generate_interactive_timeline(
            combined_timeline,
            visualization_type,
            **kwargs
        )
    
    def get_supported_formats(self, visualization_type: str) -> List[str]:
        """Get supported export formats for visualization type"""
        return self.export_formats.get(visualization_type.lower(), ['html'])


# Convenience function for direct usage
def generate_interactive_timeline(timeline: Timeline, 
                                 config: Optional[TimelineConfig] = None,
                                 visualization_type: str = "plotly",
                                 **kwargs) -> TimelineVisualization:
    """
    Convenience function to generate interactive timeline visualization
    
    Args:
        timeline: Timeline to visualize
        config: Timeline configuration (uses default if not provided)
        visualization_type: Type of visualization
        **kwargs: Additional visualization parameters
        
    Returns:
        Timeline visualization object
    """
    if config is None:
        from .core import create_default_timeline_config
        config = create_default_timeline_config()
    
    visualizer = TimelineVisualizer(config)
    return visualizer.generate_interactive_timeline(timeline, visualization_type, **kwargs)