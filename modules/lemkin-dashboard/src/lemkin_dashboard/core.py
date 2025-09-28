"""
Lemkin Investigation Dashboard Core

Comprehensive dashboard and visualization toolkit for legal investigations.
Provides interactive dashboards, case management interfaces, and data visualization
components for organizing and presenting investigation data.
"""

from datetime import datetime, date
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple
import json

from pydantic import BaseModel, Field, validator
from loguru import logger
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


class DashboardType(str, Enum):
    """Types of investigation dashboards."""
    CASE_OVERVIEW = "case_overview"
    EVIDENCE_TIMELINE = "evidence_timeline"
    ENTITY_NETWORK = "entity_network"
    DOCUMENT_ANALYSIS = "document_analysis"
    COMMUNICATION_FLOW = "communication_flow"
    GEOGRAPHIC_MAPPING = "geographic_mapping"
    COMPLIANCE_STATUS = "compliance_status"
    INVESTIGATION_PROGRESS = "investigation_progress"


class ChartType(str, Enum):
    """Types of charts and visualizations."""
    LINE_CHART = "line_chart"
    BAR_CHART = "bar_chart"
    PIE_CHART = "pie_chart"
    SCATTER_PLOT = "scatter_plot"
    HEATMAP = "heatmap"
    NETWORK_GRAPH = "network_graph"
    TIMELINE = "timeline"
    MAP_VISUALIZATION = "map_visualization"
    SANKEY_DIAGRAM = "sankey_diagram"
    TREEMAP = "treemap"


class DataSource(BaseModel):
    """Data source configuration for dashboard components."""
    source_id: str
    source_type: str
    connection_params: Dict[str, Any]
    refresh_interval: Optional[int] = None
    last_updated: Optional[datetime] = None


class ChartConfig(BaseModel):
    """Configuration for individual charts."""
    chart_id: str
    chart_type: ChartType
    title: str
    data_source: DataSource
    x_axis: Optional[str] = None
    y_axis: Optional[str] = None
    color_field: Optional[str] = None
    size_field: Optional[str] = None
    filters: Dict[str, Any] = Field(default_factory=dict)
    styling: Dict[str, Any] = Field(default_factory=dict)


class DashboardLayout(BaseModel):
    """Dashboard layout configuration."""
    layout_id: str
    grid_columns: int = 12
    components: List[Dict[str, Any]]
    responsive_breakpoints: Dict[str, int] = Field(default_factory=dict)


class Dashboard(BaseModel):
    """Investigation dashboard model."""
    dashboard_id: str
    title: str
    description: Optional[str] = None
    dashboard_type: DashboardType
    layout: DashboardLayout
    charts: List[ChartConfig]
    data_sources: List[DataSource]
    permissions: Dict[str, List[str]] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    tags: List[str] = Field(default_factory=list)


class CaseMetrics(BaseModel):
    """Case-level metrics and KPIs."""
    case_id: str
    total_documents: int
    processed_documents: int
    entities_identified: int
    relationships_mapped: int
    timeline_events: int
    evidence_items: int
    outstanding_tasks: int
    completion_percentage: float
    last_activity: datetime


class InvestigationProgress(BaseModel):
    """Investigation progress tracking."""
    investigation_id: str
    milestones: List[Dict[str, Any]]
    current_phase: str
    completion_status: Dict[str, float]
    resource_allocation: Dict[str, Any]
    timeline_adherence: float
    quality_metrics: Dict[str, float]


class DashboardBuilder:
    """Builder for creating investigation dashboards."""

    def __init__(self):
        self.dashboard = None
        self.chart_configs = []
        self.data_sources = []

    def create_dashboard(self, dashboard_id: str, title: str,
                        dashboard_type: DashboardType) -> 'DashboardBuilder':
        """Initialize a new dashboard."""
        layout = DashboardLayout(
            layout_id=f"{dashboard_id}_layout",
            grid_columns=12,
            components=[]
        )

        self.dashboard = Dashboard(
            dashboard_id=dashboard_id,
            title=title,
            dashboard_type=dashboard_type,
            layout=layout,
            charts=[],
            data_sources=[]
        )
        return self

    def add_data_source(self, source_id: str, source_type: str,
                       connection_params: Dict[str, Any]) -> 'DashboardBuilder':
        """Add a data source to the dashboard."""
        data_source = DataSource(
            source_id=source_id,
            source_type=source_type,
            connection_params=connection_params
        )
        self.data_sources.append(data_source)
        return self

    def add_chart(self, chart_id: str, chart_type: ChartType, title: str,
                 data_source_id: str, **kwargs) -> 'DashboardBuilder':
        """Add a chart to the dashboard."""
        data_source = next((ds for ds in self.data_sources if ds.source_id == data_source_id), None)
        if not data_source:
            raise ValueError(f"Data source {data_source_id} not found")

        chart_config = ChartConfig(
            chart_id=chart_id,
            chart_type=chart_type,
            title=title,
            data_source=data_source,
            **kwargs
        )
        self.chart_configs.append(chart_config)
        return self

    def set_layout(self, grid_columns: int = 12,
                  responsive_breakpoints: Optional[Dict[str, int]] = None) -> 'DashboardBuilder':
        """Configure dashboard layout."""
        if self.dashboard:
            self.dashboard.layout.grid_columns = grid_columns
            if responsive_breakpoints:
                self.dashboard.layout.responsive_breakpoints = responsive_breakpoints
        return self

    def build(self) -> Dashboard:
        """Build and return the dashboard."""
        if not self.dashboard:
            raise ValueError("Dashboard not initialized")

        self.dashboard.charts = self.chart_configs
        self.dashboard.data_sources = self.data_sources
        return self.dashboard


class VisualizationEngine:
    """Engine for creating investigation visualizations."""

    def __init__(self):
        self.color_schemes = {
            "default": ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"],
            "legal": ["#2E86AB", "#A23B72", "#F18F01", "#C73E1D", "#592E83"],
            "evidence": ["#264653", "#2A9D8F", "#E9C46A", "#F4A261", "#E76F51"]
        }

    def create_timeline_visualization(self, events: List[Dict[str, Any]],
                                    title: str = "Investigation Timeline") -> go.Figure:
        """Create timeline visualization for investigation events."""
        df = pd.DataFrame(events)

        if 'date' not in df.columns or 'event' not in df.columns:
            raise ValueError("Timeline data must have 'date' and 'event' columns")

        fig = go.Figure()

        # Add timeline points
        fig.add_trace(go.Scatter(
            x=df['date'],
            y=df['event'],
            mode='markers+lines',
            marker=dict(size=10, color=self.color_schemes["legal"][0]),
            line=dict(color=self.color_schemes["legal"][1]),
            text=df.get('description', ''),
            hovertemplate='<b>%{y}</b><br>Date: %{x}<br>%{text}<extra></extra>'
        ))

        fig.update_layout(
            title=title,
            xaxis_title="Date",
            yaxis_title="Events",
            hovermode='closest',
            height=600
        )

        return fig

    def create_network_visualization(self, nodes: List[Dict[str, Any]],
                                   edges: List[Dict[str, Any]],
                                   title: str = "Entity Network") -> go.Figure:
        """Create network visualization for entity relationships."""
        import networkx as nx

        # Create NetworkX graph
        G = nx.Graph()

        # Add nodes
        for node in nodes:
            G.add_node(node['id'], **node)

        # Add edges
        for edge in edges:
            G.add_edge(edge['source'], edge['target'], **edge)

        # Calculate layout
        pos = nx.spring_layout(G, k=1, iterations=50)

        # Extract node positions
        node_x = [pos[node]['x'] for node in G.nodes()]
        node_y = [pos[node]['y'] for node in G.nodes()]
        node_text = [G.nodes[node].get('label', node) for node in G.nodes()]

        # Extract edge positions
        edge_x = []
        edge_y = []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

        # Create edge trace
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            mode='lines'
        )

        # Create node trace
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            hoverinfo='text',
            text=node_text,
            textposition="middle center",
            marker=dict(
                size=20,
                color=self.color_schemes["legal"][0],
                line=dict(width=2, color='white')
            )
        )

        fig = go.Figure(data=[edge_trace, node_trace],
                       layout=go.Layout(
                           title=title,
                           titlefont_size=16,
                           showlegend=False,
                           hovermode='closest',
                           margin=dict(b=20,l=5,r=5,t=40),
                           annotations=[ dict(
                               text="Investigation network analysis",
                               showarrow=False,
                               xref="paper", yref="paper",
                               x=0.005, y=-0.002,
                               xanchor='left', yanchor='bottom',
                               font=dict(color="#888", size=10)
                           )],
                           xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                           yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                       ))

        return fig

    def create_evidence_heatmap(self, data: pd.DataFrame,
                              x_col: str, y_col: str, value_col: str,
                              title: str = "Evidence Distribution") -> go.Figure:
        """Create heatmap for evidence distribution analysis."""
        pivot_data = data.pivot(index=y_col, columns=x_col, values=value_col)

        fig = go.Figure(data=go.Heatmap(
            z=pivot_data.values,
            x=pivot_data.columns,
            y=pivot_data.index,
            colorscale='Viridis',
            hoverongaps=False
        ))

        fig.update_layout(
            title=title,
            xaxis_title=x_col,
            yaxis_title=y_col
        )

        return fig

    def create_progress_dashboard(self, metrics: CaseMetrics) -> go.Figure:
        """Create progress dashboard with key metrics."""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Document Processing', 'Entity Analysis',
                          'Evidence Timeline', 'Completion Status'),
            specs=[[{'type': 'indicator'}, {'type': 'bar'}],
                   [{'type': 'pie'}, {'type': 'indicator'}]]
        )

        # Document processing gauge
        fig.add_trace(go.Indicator(
            mode = "gauge+number+delta",
            value = (metrics.processed_documents / metrics.total_documents) * 100,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Documents Processed (%)"},
            gauge = {'axis': {'range': [None, 100]},
                    'bar': {'color': self.color_schemes["legal"][0]},
                    'steps': [
                        {'range': [0, 50], 'color': "lightgray"},
                        {'range': [50, 100], 'color': "gray"}],
                    'threshold': {'line': {'color': "red", 'width': 4},
                                'thickness': 0.75, 'value': 90}}),
            row=1, col=1)

        # Entity analysis bar chart
        fig.add_trace(go.Bar(
            x=['Entities', 'Relationships', 'Timeline Events'],
            y=[metrics.entities_identified, metrics.relationships_mapped, metrics.timeline_events],
            marker_color=self.color_schemes["legal"][:3]),
            row=1, col=2)

        # Evidence distribution pie chart
        evidence_data = {
            'Processed': metrics.evidence_items,
            'Pending': metrics.outstanding_tasks
        }
        fig.add_trace(go.Pie(
            labels=list(evidence_data.keys()),
            values=list(evidence_data.values()),
            marker_colors=self.color_schemes["evidence"][:2]),
            row=2, col=1)

        # Overall completion indicator
        fig.add_trace(go.Indicator(
            mode = "gauge+number",
            value = metrics.completion_percentage,
            title = {'text': "Overall Completion (%)"},
            gauge = {'axis': {'range': [None, 100]},
                    'bar': {'color': self.color_schemes["legal"][2]}}),
            row=2, col=2)

        fig.update_layout(
            title=f"Investigation Dashboard - Case {metrics.case_id}",
            height=800
        )

        return fig


class DashboardManager:
    """Manager for investigation dashboards."""

    def __init__(self, storage_path: Optional[Path] = None):
        self.storage_path = storage_path or Path("dashboards")
        self.storage_path.mkdir(exist_ok=True)
        self.dashboards: Dict[str, Dashboard] = {}
        self.visualization_engine = VisualizationEngine()

    def create_dashboard(self, dashboard_id: str, title: str,
                        dashboard_type: DashboardType) -> DashboardBuilder:
        """Create a new dashboard."""
        return DashboardBuilder().create_dashboard(dashboard_id, title, dashboard_type)

    def save_dashboard(self, dashboard: Dashboard) -> None:
        """Save dashboard configuration."""
        dashboard_file = self.storage_path / f"{dashboard.dashboard_id}.json"

        try:
            with open(dashboard_file, 'w') as f:
                json.dump(dashboard.dict(), f, indent=2, default=str)

            self.dashboards[dashboard.dashboard_id] = dashboard
            logger.info(f"Saved dashboard {dashboard.dashboard_id}")

        except Exception as e:
            logger.error(f"Failed to save dashboard {dashboard.dashboard_id}: {e}")
            raise

    def load_dashboard(self, dashboard_id: str) -> Optional[Dashboard]:
        """Load dashboard configuration."""
        if dashboard_id in self.dashboards:
            return self.dashboards[dashboard_id]

        dashboard_file = self.storage_path / f"{dashboard_id}.json"

        if not dashboard_file.exists():
            logger.warning(f"Dashboard {dashboard_id} not found")
            return None

        try:
            with open(dashboard_file, 'r') as f:
                dashboard_data = json.load(f)

            dashboard = Dashboard(**dashboard_data)
            self.dashboards[dashboard_id] = dashboard
            return dashboard

        except Exception as e:
            logger.error(f"Failed to load dashboard {dashboard_id}: {e}")
            return None

    def list_dashboards(self) -> List[Dashboard]:
        """List all available dashboards."""
        dashboards = []

        for dashboard_file in self.storage_path.glob("*.json"):
            dashboard_id = dashboard_file.stem
            dashboard = self.load_dashboard(dashboard_id)
            if dashboard:
                dashboards.append(dashboard)

        return dashboards

    def delete_dashboard(self, dashboard_id: str) -> bool:
        """Delete a dashboard."""
        dashboard_file = self.storage_path / f"{dashboard_id}.json"

        try:
            if dashboard_file.exists():
                dashboard_file.unlink()

            if dashboard_id in self.dashboards:
                del self.dashboards[dashboard_id]

            logger.info(f"Deleted dashboard {dashboard_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to delete dashboard {dashboard_id}: {e}")
            return False

    def generate_case_overview_dashboard(self, case_metrics: CaseMetrics) -> Dashboard:
        """Generate a comprehensive case overview dashboard."""
        builder = self.create_dashboard(
            f"case_overview_{case_metrics.case_id}",
            f"Case Overview - {case_metrics.case_id}",
            DashboardType.CASE_OVERVIEW
        )

        # Add data sources
        builder.add_data_source(
            "case_metrics",
            "case_data",
            {"case_id": case_metrics.case_id}
        )

        # Add charts
        builder.add_chart(
            "progress_overview",
            ChartType.PIE_CHART,
            "Progress Overview",
            "case_metrics"
        )

        builder.add_chart(
            "document_timeline",
            ChartType.TIMELINE,
            "Document Processing Timeline",
            "case_metrics"
        )

        builder.add_chart(
            "entity_network",
            ChartType.NETWORK_GRAPH,
            "Entity Relationships",
            "case_metrics"
        )

        dashboard = builder.build()
        self.save_dashboard(dashboard)
        return dashboard

    def create_custom_visualization(self, chart_config: ChartConfig,
                                  data: pd.DataFrame) -> go.Figure:
        """Create custom visualization based on configuration."""
        if chart_config.chart_type == ChartType.LINE_CHART:
            fig = px.line(data, x=chart_config.x_axis, y=chart_config.y_axis,
                         title=chart_config.title)
        elif chart_config.chart_type == ChartType.BAR_CHART:
            fig = px.bar(data, x=chart_config.x_axis, y=chart_config.y_axis,
                        title=chart_config.title)
        elif chart_config.chart_type == ChartType.PIE_CHART:
            fig = px.pie(data, names=chart_config.x_axis, values=chart_config.y_axis,
                        title=chart_config.title)
        elif chart_config.chart_type == ChartType.SCATTER_PLOT:
            fig = px.scatter(data, x=chart_config.x_axis, y=chart_config.y_axis,
                           size=chart_config.size_field, color=chart_config.color_field,
                           title=chart_config.title)
        else:
            # Default to bar chart
            fig = px.bar(data, x=chart_config.x_axis, y=chart_config.y_axis,
                        title=chart_config.title)

        # Apply custom styling
        if chart_config.styling:
            fig.update_layout(**chart_config.styling)

        return fig

    def export_dashboard(self, dashboard_id: str,
                        export_format: str = "html") -> Optional[Path]:
        """Export dashboard to specified format."""
        dashboard = self.load_dashboard(dashboard_id)
        if not dashboard:
            return None

        export_file = self.storage_path / f"{dashboard_id}_export.{export_format}"

        try:
            if export_format == "html":
                # Generate HTML dashboard
                html_content = self._generate_html_dashboard(dashboard)
                with open(export_file, 'w') as f:
                    f.write(html_content)
            elif export_format == "json":
                # Export as JSON
                with open(export_file, 'w') as f:
                    json.dump(dashboard.dict(), f, indent=2, default=str)
            else:
                logger.warning(f"Unsupported export format: {export_format}")
                return None

            logger.info(f"Exported dashboard {dashboard_id} to {export_file}")
            return export_file

        except Exception as e:
            logger.error(f"Failed to export dashboard {dashboard_id}: {e}")
            return None

    def _generate_html_dashboard(self, dashboard: Dashboard) -> str:
        """Generate HTML content for dashboard."""
        html_template = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{dashboard.title}</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .dashboard-header {{ text-align: center; margin-bottom: 30px; }}
                .chart-container {{ margin: 20px 0; }}
            </style>
        </head>
        <body>
            <div class="dashboard-header">
                <h1>{dashboard.title}</h1>
                <p>{dashboard.description or 'Investigation Dashboard'}</p>
            </div>
            <div id="dashboard-content">
                <!-- Charts will be inserted here -->
            </div>
        </body>
        </html>
        """
        return html_template


class ReportGenerator:
    """Generator for investigation reports and summaries."""

    def __init__(self):
        self.templates_path = Path("templates")
        self.templates_path.mkdir(exist_ok=True)

    def generate_case_summary(self, case_metrics: CaseMetrics,
                            progress: InvestigationProgress) -> Dict[str, Any]:
        """Generate comprehensive case summary."""
        summary = {
            "case_id": case_metrics.case_id,
            "generated_at": datetime.now(),
            "overview": {
                "total_documents": case_metrics.total_documents,
                "processed_documents": case_metrics.processed_documents,
                "processing_rate": case_metrics.processed_documents / case_metrics.total_documents,
                "entities_identified": case_metrics.entities_identified,
                "relationships_mapped": case_metrics.relationships_mapped,
                "timeline_events": case_metrics.timeline_events,
                "evidence_items": case_metrics.evidence_items
            },
            "progress": {
                "current_phase": progress.current_phase,
                "overall_completion": case_metrics.completion_percentage,
                "phase_completion": progress.completion_status,
                "timeline_adherence": progress.timeline_adherence
            },
            "quality_metrics": progress.quality_metrics,
            "outstanding_items": case_metrics.outstanding_tasks,
            "last_activity": case_metrics.last_activity
        }

        return summary

    def generate_executive_summary(self, case_metrics: CaseMetrics) -> str:
        """Generate executive summary text."""
        completion = case_metrics.completion_percentage

        summary = f"""
        EXECUTIVE SUMMARY - CASE {case_metrics.case_id}

        Investigation Progress: {completion:.1f}% Complete

        DOCUMENT ANALYSIS:
        - Total Documents: {case_metrics.total_documents:,}
        - Processed: {case_metrics.processed_documents:,} ({(case_metrics.processed_documents/case_metrics.total_documents)*100:.1f}%)
        - Remaining: {case_metrics.total_documents - case_metrics.processed_documents:,}

        ENTITY IDENTIFICATION:
        - Entities Identified: {case_metrics.entities_identified:,}
        - Relationships Mapped: {case_metrics.relationships_mapped:,}
        - Timeline Events: {case_metrics.timeline_events:,}

        EVIDENCE MANAGEMENT:
        - Evidence Items: {case_metrics.evidence_items:,}
        - Outstanding Tasks: {case_metrics.outstanding_tasks:,}

        RECOMMENDATIONS:
        {"- Continue current processing pace" if completion > 50 else "- Increase resource allocation to maintain timeline"}
        {"- Review entity relationships for patterns" if case_metrics.relationships_mapped > 100 else "- Focus on entity relationship mapping"}
        {"- Prepare for analysis phase" if completion > 80 else "- Maintain focus on evidence collection"}

        Last Activity: {case_metrics.last_activity.strftime('%Y-%m-%d %H:%M')}
        """

        return summary.strip()


# Export all classes and functions
__all__ = [
    "DashboardType",
    "ChartType",
    "DataSource",
    "ChartConfig",
    "DashboardLayout",
    "Dashboard",
    "CaseMetrics",
    "InvestigationProgress",
    "DashboardBuilder",
    "VisualizationEngine",
    "DashboardManager",
    "ReportGenerator"
]