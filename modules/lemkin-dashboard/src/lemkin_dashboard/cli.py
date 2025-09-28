"""
Lemkin Dashboard CLI

Command-line interface for managing investigation dashboards, creating visualizations,
and generating reports for legal investigation workflows.
"""

from datetime import datetime
from pathlib import Path
from typing import Optional, List
import json

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Prompt, Confirm
from loguru import logger

from .core import (
    DashboardManager,
    DashboardType,
    ChartType,
    CaseMetrics,
    InvestigationProgress,
    VisualizationEngine,
    ReportGenerator
)

app = typer.Typer(help="Lemkin Investigation Dashboard Management")
console = Console()


@app.command()
def create(
    dashboard_id: str = typer.Argument(..., help="Unique dashboard identifier"),
    title: str = typer.Argument(..., help="Dashboard title"),
    dashboard_type: DashboardType = typer.Option(
        DashboardType.CASE_OVERVIEW,
        help="Type of dashboard to create"
    ),
    storage_path: Optional[Path] = typer.Option(None, help="Dashboard storage path")
):
    """Create a new investigation dashboard."""
    console.print(f"[bold blue]Creating dashboard: {title}[/bold blue]")

    try:
        manager = DashboardManager(storage_path)

        # Create dashboard using builder
        builder = manager.create_dashboard(dashboard_id, title, dashboard_type)

        # Interactive dashboard configuration
        if Confirm.ask("Would you like to configure data sources interactively?"):
            configure_data_sources_interactive(builder)

        if Confirm.ask("Would you like to add charts interactively?"):
            configure_charts_interactive(builder)

        # Build and save dashboard
        dashboard = builder.build()
        manager.save_dashboard(dashboard)

        console.print(f"[bold green]✓[/bold green] Dashboard '{title}' created successfully!")
        console.print(f"Dashboard ID: {dashboard_id}")
        console.print(f"Type: {dashboard_type}")

    except Exception as e:
        console.print(f"[bold red]✗[/bold red] Failed to create dashboard: {e}")
        raise typer.Exit(1)


@app.command()
def list(
    storage_path: Optional[Path] = typer.Option(None, help="Dashboard storage path"),
    dashboard_type: Optional[DashboardType] = typer.Option(None, help="Filter by type")
):
    """List all available dashboards."""
    try:
        manager = DashboardManager(storage_path)
        dashboards = manager.list_dashboards()

        if dashboard_type:
            dashboards = [d for d in dashboards if d.dashboard_type == dashboard_type]

        if not dashboards:
            console.print("[yellow]No dashboards found.[/yellow]")
            return

        # Create table
        table = Table(title="Investigation Dashboards", show_header=True)
        table.add_column("ID", style="cyan", no_wrap=True)
        table.add_column("Title", style="magenta")
        table.add_column("Type", style="green")
        table.add_column("Charts", justify="right", style="blue")
        table.add_column("Created", style="dim")

        for dashboard in dashboards:
            table.add_row(
                dashboard.dashboard_id,
                dashboard.title,
                dashboard.dashboard_type,
                str(len(dashboard.charts)),
                dashboard.created_at.strftime('%Y-%m-%d')
            )

        console.print(table)

    except Exception as e:
        console.print(f"[bold red]✗[/bold red] Failed to list dashboards: {e}")
        raise typer.Exit(1)


@app.command()
def show(
    dashboard_id: str = typer.Argument(..., help="Dashboard ID to display"),
    storage_path: Optional[Path] = typer.Option(None, help="Dashboard storage path")
):
    """Show dashboard details and configuration."""
    try:
        manager = DashboardManager(storage_path)
        dashboard = manager.load_dashboard(dashboard_id)

        if not dashboard:
            console.print(f"[bold red]✗[/bold red] Dashboard '{dashboard_id}' not found")
            raise typer.Exit(1)

        # Dashboard overview panel
        overview_content = f"""
[bold]Title:[/bold] {dashboard.title}
[bold]Type:[/bold] {dashboard.dashboard_type}
[bold]Description:[/bold] {dashboard.description or 'No description'}
[bold]Charts:[/bold] {len(dashboard.charts)}
[bold]Data Sources:[/bold] {len(dashboard.data_sources)}
[bold]Created:[/bold] {dashboard.created_at.strftime('%Y-%m-%d %H:%M')}
[bold]Updated:[/bold] {dashboard.updated_at.strftime('%Y-%m-%d %H:%M')}
        """

        console.print(Panel(overview_content, title=f"Dashboard: {dashboard_id}", border_style="blue"))

        # Charts table
        if dashboard.charts:
            chart_table = Table(title="Charts", show_header=True)
            chart_table.add_column("Chart ID", style="cyan")
            chart_table.add_column("Title", style="magenta")
            chart_table.add_column("Type", style="green")
            chart_table.add_column("Data Source", style="blue")

            for chart in dashboard.charts:
                chart_table.add_row(
                    chart.chart_id,
                    chart.title,
                    chart.chart_type,
                    chart.data_source.source_id
                )

            console.print(chart_table)

        # Data sources table
        if dashboard.data_sources:
            source_table = Table(title="Data Sources", show_header=True)
            source_table.add_column("Source ID", style="cyan")
            source_table.add_column("Type", style="green")
            source_table.add_column("Last Updated", style="dim")

            for source in dashboard.data_sources:
                last_updated = source.last_updated.strftime('%Y-%m-%d %H:%M') if source.last_updated else 'Never'
                source_table.add_row(
                    source.source_id,
                    source.source_type,
                    last_updated
                )

            console.print(source_table)

    except Exception as e:
        console.print(f"[bold red]✗[/bold red] Failed to show dashboard: {e}")
        raise typer.Exit(1)


@app.command()
def export(
    dashboard_id: str = typer.Argument(..., help="Dashboard ID to export"),
    format: str = typer.Option("html", help="Export format (html, json)"),
    output: Optional[Path] = typer.Option(None, help="Output file path"),
    storage_path: Optional[Path] = typer.Option(None, help="Dashboard storage path")
):
    """Export dashboard to specified format."""
    console.print(f"[bold blue]Exporting dashboard '{dashboard_id}' to {format.upper()}...[/bold blue]")

    try:
        manager = DashboardManager(storage_path)

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task(f"Exporting to {format}...", total=None)

            export_path = manager.export_dashboard(dashboard_id, format)

            if not export_path:
                console.print(f"[bold red]✗[/bold red] Export failed")
                raise typer.Exit(1)

            # Move to specified output path if provided
            if output:
                output.parent.mkdir(parents=True, exist_ok=True)
                export_path.rename(output)
                export_path = output

            progress.update(task, completed=100)

        console.print(f"[bold green]✓[/bold green] Dashboard exported successfully!")
        console.print(f"Output file: {export_path}")

    except Exception as e:
        console.print(f"[bold red]✗[/bold red] Export failed: {e}")
        raise typer.Exit(1)


@app.command()
def delete(
    dashboard_id: str = typer.Argument(..., help="Dashboard ID to delete"),
    storage_path: Optional[Path] = typer.Option(None, help="Dashboard storage path"),
    force: bool = typer.Option(False, "--force", "-f", help="Force deletion without confirmation")
):
    """Delete a dashboard."""
    if not force:
        if not Confirm.ask(f"Are you sure you want to delete dashboard '{dashboard_id}'?"):
            console.print("Deletion cancelled.")
            return

    try:
        manager = DashboardManager(storage_path)

        if manager.delete_dashboard(dashboard_id):
            console.print(f"[bold green]✓[/bold green] Dashboard '{dashboard_id}' deleted successfully!")
        else:
            console.print(f"[bold red]✗[/bold red] Failed to delete dashboard '{dashboard_id}'")
            raise typer.Exit(1)

    except Exception as e:
        console.print(f"[bold red]✗[/bold red] Deletion failed: {e}")
        raise typer.Exit(1)


@app.command()
def generate(
    case_id: str = typer.Argument(..., help="Case ID for dashboard generation"),
    dashboard_type: DashboardType = typer.Option(
        DashboardType.CASE_OVERVIEW,
        help="Type of dashboard to generate"
    ),
    storage_path: Optional[Path] = typer.Option(None, help="Dashboard storage path"),
    metrics_file: Optional[Path] = typer.Option(None, help="Case metrics JSON file")
):
    """Generate dashboard from case metrics."""
    console.print(f"[bold blue]Generating {dashboard_type} dashboard for case {case_id}...[/bold blue]")

    try:
        manager = DashboardManager(storage_path)

        # Load or create sample metrics
        if metrics_file and metrics_file.exists():
            with open(metrics_file) as f:
                metrics_data = json.load(f)
            case_metrics = CaseMetrics(**metrics_data)
        else:
            # Create sample metrics for demonstration
            case_metrics = CaseMetrics(
                case_id=case_id,
                total_documents=1000,
                processed_documents=750,
                entities_identified=250,
                relationships_mapped=180,
                timeline_events=95,
                evidence_items=320,
                outstanding_tasks=50,
                completion_percentage=75.0,
                last_activity=datetime.now()
            )

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Generating dashboard...", total=None)

            dashboard = manager.generate_case_overview_dashboard(case_metrics)

            progress.update(task, completed=100)

        console.print(f"[bold green]✓[/bold green] Dashboard generated successfully!")
        console.print(f"Dashboard ID: {dashboard.dashboard_id}")
        console.print(f"Title: {dashboard.title}")

        # Show key metrics
        metrics_panel = f"""
[bold]Case Metrics Summary:[/bold]
• Documents: {case_metrics.processed_documents:,} / {case_metrics.total_documents:,} ({(case_metrics.processed_documents/case_metrics.total_documents)*100:.1f}%)
• Entities: {case_metrics.entities_identified:,}
• Relationships: {case_metrics.relationships_mapped:,}
• Timeline Events: {case_metrics.timeline_events:,}
• Evidence Items: {case_metrics.evidence_items:,}
• Overall Progress: {case_metrics.completion_percentage:.1f}%
        """
        console.print(Panel(metrics_panel, title="Case Overview", border_style="green"))

    except Exception as e:
        console.print(f"[bold red]✗[/bold red] Generation failed: {e}")
        raise typer.Exit(1)


@app.command()
def visualize(
    data_file: Path = typer.Argument(..., help="Data file for visualization"),
    chart_type: ChartType = typer.Argument(..., help="Type of chart to create"),
    title: str = typer.Option("Visualization", help="Chart title"),
    x_axis: Optional[str] = typer.Option(None, help="X-axis column"),
    y_axis: Optional[str] = typer.Option(None, help="Y-axis column"),
    output: Optional[Path] = typer.Option(None, help="Output HTML file")
):
    """Create standalone visualizations from data files."""
    console.print(f"[bold blue]Creating {chart_type} visualization...[/bold blue]")

    try:
        import pandas as pd

        # Load data
        if data_file.suffix.lower() == '.csv':
            data = pd.read_csv(data_file)
        elif data_file.suffix.lower() == '.json':
            data = pd.read_json(data_file)
        else:
            console.print(f"[bold red]✗[/bold red] Unsupported file format: {data_file.suffix}")
            raise typer.Exit(1)

        console.print(f"Loaded {len(data)} rows from {data_file}")

        # Interactive column selection if not provided
        if not x_axis or not y_axis:
            console.print("\nAvailable columns:")
            for i, col in enumerate(data.columns, 1):
                console.print(f"  {i}. {col}")

            if not x_axis:
                x_axis = Prompt.ask("Select X-axis column", choices=list(data.columns))
            if not y_axis and chart_type not in [ChartType.PIE_CHART]:
                y_axis = Prompt.ask("Select Y-axis column", choices=list(data.columns))

        engine = VisualizationEngine()

        # Create visualization based on type
        if chart_type == ChartType.TIMELINE:
            if 'date' not in data.columns or 'event' not in data.columns:
                console.print("[bold red]✗[/bold red] Timeline requires 'date' and 'event' columns")
                raise typer.Exit(1)

            events = data.to_dict('records')
            fig = engine.create_timeline_visualization(events, title)

        elif chart_type == ChartType.NETWORK_GRAPH:
            if 'nodes' not in data.columns or 'edges' not in data.columns:
                console.print("[bold red]✗[/bold red] Network graph requires 'nodes' and 'edges' data")
                raise typer.Exit(1)

            # Simplified network creation - would need proper node/edge data structure
            console.print("[yellow]Network visualization requires structured node/edge data[/yellow]")
            return

        elif chart_type == ChartType.HEATMAP:
            if len(data.columns) < 3:
                console.print("[bold red]✗[/bold red] Heatmap requires at least 3 columns")
                raise typer.Exit(1)

            z_col = Prompt.ask("Select value column for heatmap", choices=list(data.columns))
            fig = engine.create_evidence_heatmap(data, x_axis, y_axis, z_col, title)

        else:
            # Use chart config approach
            from .core import ChartConfig, DataSource

            data_source = DataSource(
                source_id="file_data",
                source_type="file",
                connection_params={"file_path": str(data_file)}
            )

            chart_config = ChartConfig(
                chart_id="standalone_chart",
                chart_type=chart_type,
                title=title,
                data_source=data_source,
                x_axis=x_axis,
                y_axis=y_axis
            )

            manager = DashboardManager()
            fig = manager.create_custom_visualization(chart_config, data)

        # Save visualization
        output_file = output or Path(f"visualization_{chart_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html")
        fig.write_html(str(output_file))

        console.print(f"[bold green]✓[/bold green] Visualization created successfully!")
        console.print(f"Output file: {output_file}")

        if Confirm.ask("Open visualization in browser?"):
            import webbrowser
            webbrowser.open(f"file://{output_file.absolute()}")

    except Exception as e:
        console.print(f"[bold red]✗[/bold red] Visualization failed: {e}")
        raise typer.Exit(1)


@app.command()
def report(
    case_id: str = typer.Argument(..., help="Case ID for report generation"),
    report_type: str = typer.Option("summary", help="Report type (summary, executive)"),
    output: Optional[Path] = typer.Option(None, help="Output file path"),
    metrics_file: Optional[Path] = typer.Option(None, help="Case metrics JSON file")
):
    """Generate investigation reports."""
    console.print(f"[bold blue]Generating {report_type} report for case {case_id}...[/bold blue]")

    try:
        # Load or create sample metrics
        if metrics_file and metrics_file.exists():
            with open(metrics_file) as f:
                metrics_data = json.load(f)
            case_metrics = CaseMetrics(**metrics_data)
        else:
            case_metrics = CaseMetrics(
                case_id=case_id,
                total_documents=1000,
                processed_documents=750,
                entities_identified=250,
                relationships_mapped=180,
                timeline_events=95,
                evidence_items=320,
                outstanding_tasks=50,
                completion_percentage=75.0,
                last_activity=datetime.now()
            )

        generator = ReportGenerator()

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Generating report...", total=None)

            if report_type == "executive":
                report_content = generator.generate_executive_summary(case_metrics)
            else:
                # Create progress data
                progress_data = InvestigationProgress(
                    investigation_id=case_id,
                    milestones=[],
                    current_phase="Evidence Analysis",
                    completion_status={"collection": 0.9, "analysis": 0.6, "reporting": 0.2},
                    resource_allocation={},
                    timeline_adherence=0.85,
                    quality_metrics={"accuracy": 0.95, "completeness": 0.88}
                )

                summary_data = generator.generate_case_summary(case_metrics, progress_data)
                report_content = json.dumps(summary_data, indent=2, default=str)

            progress.update(task, completed=100)

        # Save report
        output_file = output or Path(f"report_{case_id}_{report_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{'txt' if report_type == 'executive' else 'json'}")

        with open(output_file, 'w') as f:
            f.write(report_content)

        console.print(f"[bold green]✓[/bold green] Report generated successfully!")
        console.print(f"Output file: {output_file}")

        # Show preview for executive summary
        if report_type == "executive":
            console.print(Panel(report_content[:500] + "..." if len(report_content) > 500 else report_content,
                              title="Report Preview", border_style="green"))

    except Exception as e:
        console.print(f"[bold red]✗[/bold red] Report generation failed: {e}")
        raise typer.Exit(1)


def configure_data_sources_interactive(builder):
    """Interactive data source configuration."""
    while Confirm.ask("Add a data source?"):
        source_id = Prompt.ask("Data source ID")
        source_type = Prompt.ask("Data source type",
                                choices=["database", "file", "api", "case_data"])

        # Simple connection params
        connection_params = {}
        if source_type == "file":
            file_path = Prompt.ask("File path")
            connection_params["file_path"] = file_path
        elif source_type == "database":
            connection_params["host"] = Prompt.ask("Database host")
            connection_params["database"] = Prompt.ask("Database name")
        elif source_type == "api":
            connection_params["url"] = Prompt.ask("API URL")

        builder.add_data_source(source_id, source_type, connection_params)
        console.print(f"[green]✓[/green] Added data source: {source_id}")


def configure_charts_interactive(builder):
    """Interactive chart configuration."""
    if not builder.data_sources:
        console.print("[yellow]No data sources configured. Add data sources first.[/yellow]")
        return

    source_ids = [ds.source_id for ds in builder.data_sources]

    while Confirm.ask("Add a chart?"):
        chart_id = Prompt.ask("Chart ID")
        title = Prompt.ask("Chart title")

        chart_type = Prompt.ask("Chart type",
                               choices=[ct.value for ct in ChartType])
        chart_type_enum = ChartType(chart_type)

        data_source_id = Prompt.ask("Data source", choices=source_ids)

        kwargs = {}
        if chart_type_enum not in [ChartType.PIE_CHART, ChartType.NETWORK_GRAPH]:
            kwargs["x_axis"] = Prompt.ask("X-axis column (optional)", default="")
            kwargs["y_axis"] = Prompt.ask("Y-axis column (optional)", default="")

        builder.add_chart(chart_id, chart_type_enum, title, data_source_id, **kwargs)
        console.print(f"[green]✓[/green] Added chart: {title}")


if __name__ == "__main__":
    app()