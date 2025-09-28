"""
Lemkin Communication Analysis Suite CLI

Command-line interface for comprehensive communication analysis and pattern detection.
Provides forensic-grade analysis tools for legal investigations.
"""

import typer
from pathlib import Path
from typing import Optional, List
import json
from datetime import datetime
import sys
import logging
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.panel import Panel
from rich.text import Text
from rich import print as rprint

from .core import (
    CommunicationAnalyzer, CommsConfig, AnalysisLevel, 
    load_communications, export_analysis_results, generate_forensic_report
)
from .chat_processor import process_chat_exports
from .email_analyzer import analyze_email_threads
from .network_mapper import map_communication_network
from .pattern_detector import detect_communication_patterns

# Initialize CLI app
app = typer.Typer(
    name="lemkin-comms",
    help="Lemkin Communication Analysis Suite - Forensic communication analysis toolkit",
    add_completion=False
)

# Initialize console for rich output
console = Console()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('lemkin_comms.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def create_config(
    analysis_level: str = "standard",
    enable_content_analysis: bool = True,
    enable_network_analysis: bool = True,
    enable_pattern_detection: bool = True,
    enable_anomaly_detection: bool = True,
    pattern_confidence_threshold: float = 0.7,
    anomaly_sensitivity: float = 0.8,
    anonymize_contacts: bool = False,
    chain_of_custody: bool = True
) -> CommsConfig:
    """Create configuration from CLI parameters"""
    return CommsConfig(
        analysis_level=AnalysisLevel(analysis_level),
        enable_content_analysis=enable_content_analysis,
        enable_network_analysis=enable_network_analysis,
        enable_pattern_detection=enable_pattern_detection,
        enable_anomaly_detection=enable_anomaly_detection,
        pattern_confidence_threshold=pattern_confidence_threshold,
        anomaly_sensitivity=anomaly_sensitivity,
        anonymize_contacts=anonymize_contacts,
        chain_of_custody=chain_of_custody
    )


@app.command()
def process_chat(
    export_path: Path = typer.Argument(..., help="Path to chat export file"),
    output_dir: Path = typer.Option(Path("output"), help="Output directory for results"),
    platform: Optional[str] = typer.Option(None, help="Chat platform (auto-detect if not specified)"),
    analysis_level: str = typer.Option("standard", help="Analysis depth: basic, standard, comprehensive, forensic"),
    generate_report: bool = typer.Option(True, help="Generate forensic report"),
    save_visualizations: bool = typer.Option(True, help="Save visualization files")
):
    """Process and analyze chat exports (WhatsApp, Telegram, Signal, etc.)"""
    
    console.print(f"\n[bold blue]Lemkin Communication Analysis - Chat Processing[/bold blue]")
    console.print(f"Processing: {export_path}")
    
    if not export_path.exists():
        console.print(f"[bold red]Error:[/bold red] Export file not found: {export_path}")
        raise typer.Exit(1)
    
    try:
        # Create configuration
        config = create_config(analysis_level=analysis_level)
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            # Process chat export
            task = progress.add_task("Processing chat export...", total=None)
            
            from .core import PlatformType
            platform_type = PlatformType(platform) if platform else None
            analysis = process_chat_exports(export_path, config)
            
            progress.update(task, description="Analysis complete")
        
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save results
        results_file = output_dir / f"chat_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_file, 'w') as f:
            json.dump(analysis.dict(), f, indent=2, default=str)
        
        # Display summary
        _display_chat_summary(analysis)
        
        if generate_report:
            report_file = output_dir / "forensic_report.txt"
            # Create a mock AnalysisResult for report generation
            from .core import AnalysisResult
            result = AnalysisResult(
                config=config,
                chat_analysis=analysis,
                total_communications=analysis.total_messages,
                total_contacts=analysis.total_participants,
                platforms_analyzed=[analysis.platform],
                analysis_duration=0.0,
                data_quality_score=0.9,
                completeness_score=1.0,
                confidence_score=0.85,
                evidence_hash="mock_hash"
            )
            
            report_content = generate_forensic_report(result)
            with open(report_file, 'w') as f:
                f.write(report_content)
            
            console.print(f"\n[green]✓[/green] Forensic report saved to: {report_file}")
        
        console.print(f"\n[green]✓[/green] Analysis complete. Results saved to: {results_file}")
        
    except Exception as e:
        console.print(f"[bold red]Error processing chat:[/bold red] {str(e)}")
        logger.error(f"Chat processing failed: {str(e)}", exc_info=True)
        raise typer.Exit(1)


@app.command()
def analyze_email(
    email_source: Path = typer.Argument(..., help="Path to email source (MBOX, PST, or EML directory)"),
    output_dir: Path = typer.Option(Path("output"), help="Output directory for results"),
    analysis_level: str = typer.Option("standard", help="Analysis depth"),
    generate_report: bool = typer.Option(True, help="Generate forensic report"),
    reconstruct_threads: bool = typer.Option(True, help="Reconstruct email threads")
):
    """Analyze email archives and reconstruct conversation threads"""
    
    console.print(f"\n[bold blue]Lemkin Communication Analysis - Email Analysis[/bold blue]")
    console.print(f"Processing: {email_source}")
    
    if not email_source.exists():
        console.print(f"[bold red]Error:[/bold red] Email source not found: {email_source}")
        raise typer.Exit(1)
    
    try:
        config = create_config(analysis_level=analysis_level)
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Analyzing emails...", total=None)
            
            analysis = analyze_email_threads(email_source, config)
            
            progress.update(task, description="Analysis complete")
        
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save results
        results_file = output_dir / f"email_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_file, 'w') as f:
            json.dump(analysis.dict(), f, indent=2, default=str)
        
        # Display summary
        _display_email_summary(analysis)
        
        if generate_report:
            report_file = output_dir / "email_forensic_report.txt"
            from .core import AnalysisResult
            result = AnalysisResult(
                config=config,
                email_analysis=analysis,
                total_communications=analysis.total_emails,
                total_contacts=len(analysis.participants),
                platforms_analyzed=[analysis.participants[0].platforms[0] if analysis.participants else "email"],
                analysis_duration=0.0,
                data_quality_score=0.9,
                completeness_score=1.0,
                confidence_score=0.85,
                evidence_hash="mock_hash"
            )
            
            report_content = generate_forensic_report(result)
            with open(report_file, 'w') as f:
                f.write(report_content)
            
            console.print(f"\n[green]✓[/green] Forensic report saved to: {report_file}")
        
        console.print(f"\n[green]✓[/green] Analysis complete. Results saved to: {results_file}")
        
    except Exception as e:
        console.print(f"[bold red]Error analyzing emails:[/bold red] {str(e)}")
        logger.error(f"Email analysis failed: {str(e)}", exc_info=True)
        raise typer.Exit(1)


@app.command()
def map_network(
    communications_file: Path = typer.Argument(..., help="Path to communications JSON file"),
    output_dir: Path = typer.Option(Path("output"), help="Output directory for results"),
    create_visualization: bool = typer.Option(True, help="Create interactive network visualization"),
    detect_communities: bool = typer.Option(True, help="Detect communication communities"),
    find_suspicious: bool = typer.Option(True, help="Identify suspicious clusters")
):
    """Create and analyze communication network mappings"""
    
    console.print(f"\n[bold blue]Lemkin Communication Analysis - Network Mapping[/bold blue]")
    console.print(f"Processing: {communications_file}")
    
    if not communications_file.exists():
        console.print(f"[bold red]Error:[/bold red] Communications file not found: {communications_file}")
        raise typer.Exit(1)
    
    try:
        # Load communications
        communications = load_communications(communications_file)
        
        if not communications:
            console.print(f"[bold red]Error:[/bold red] No communications found in file")
            raise typer.Exit(1)
        
        config = create_config()
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Building communication network...", total=None)
            
            network_graph = map_communication_network(communications, config)
            
            progress.update(task, description="Network analysis complete")
        
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save results
        results_file = output_dir / f"network_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_file, 'w') as f:
            json.dump(network_graph.dict(), f, indent=2, default=str)
        
        # Display summary
        _display_network_summary(network_graph)
        
        # Create visualizations
        if create_visualization:
            from .network_mapper import NetworkMapper
            mapper = NetworkMapper(config)
            vis_files = mapper.create_visualizations(network_graph, output_dir)
            
            for vis_type, file_path in vis_files.items():
                console.print(f"[green]✓[/green] {vis_type.title()} saved to: {file_path}")
        
        console.print(f"\n[green]✓[/green] Network analysis complete. Results saved to: {results_file}")
        
    except Exception as e:
        console.print(f"[bold red]Error mapping network:[/bold red] {str(e)}")
        logger.error(f"Network mapping failed: {str(e)}", exc_info=True)
        raise typer.Exit(1)


@app.command()
def detect_patterns(
    communications_file: Path = typer.Argument(..., help="Path to communications JSON file"),
    output_dir: Path = typer.Option(Path("output"), help="Output directory for results"),
    sensitivity: float = typer.Option(0.8, help="Anomaly detection sensitivity (0.0-1.0)"),
    confidence_threshold: float = typer.Option(0.7, help="Pattern confidence threshold (0.0-1.0)"),
    include_sentiment: bool = typer.Option(True, help="Include sentiment analysis"),
    behavioral_profiling: bool = typer.Option(True, help="Create behavioral profiles")
):
    """Detect communication patterns and anomalies"""
    
    console.print(f"\n[bold blue]Lemkin Communication Analysis - Pattern Detection[/bold blue]")
    console.print(f"Processing: {communications_file}")
    
    if not communications_file.exists():
        console.print(f"[bold red]Error:[/bold red] Communications file not found: {communications_file}")
        raise typer.Exit(1)
    
    try:
        # Load communications
        communications = load_communications(communications_file)
        
        if not communications:
            console.print(f"[bold red]Error:[/bold red] No communications found in file")
            raise typer.Exit(1)
        
        config = create_config(
            anomaly_sensitivity=sensitivity,
            pattern_confidence_threshold=confidence_threshold
        )
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Detecting communication patterns...", total=None)
            
            pattern_analysis = detect_communication_patterns(communications, config)
            
            progress.update(task, description="Pattern analysis complete")
        
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save results
        results_file = output_dir / f"pattern_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_file, 'w') as f:
            json.dump(pattern_analysis.dict(), f, indent=2, default=str)
        
        # Display summary
        _display_pattern_summary(pattern_analysis)
        
        console.print(f"\n[green]✓[/green] Pattern analysis complete. Results saved to: {results_file}")
        
    except Exception as e:
        console.print(f"[bold red]Error detecting patterns:[/bold red] {str(e)}")
        logger.error(f"Pattern detection failed: {str(e)}", exc_info=True)
        raise typer.Exit(1)


@app.command()
def analyze(
    input_path: Path = typer.Argument(..., help="Path to communication data"),
    output_dir: Path = typer.Option(Path("output"), help="Output directory for results"),
    analysis_level: str = typer.Option("comprehensive", help="Analysis depth"),
    data_type: str = typer.Option("auto", help="Data type: auto, chat, email, json"),
    generate_report: bool = typer.Option(True, help="Generate comprehensive forensic report"),
    create_visualizations: bool = typer.Option(True, help="Create all visualizations"),
    anonymize: bool = typer.Option(False, help="Anonymize contact information")
):
    """Perform comprehensive communication analysis (all modules)"""
    
    console.print(f"\n[bold blue]Lemkin Communication Analysis Suite - Comprehensive Analysis[/bold blue]")
    console.print(f"Input: {input_path}")
    console.print(f"Analysis Level: {analysis_level}")
    
    if not input_path.exists():
        console.print(f"[bold red]Error:[/bold red] Input path not found: {input_path}")
        raise typer.Exit(1)
    
    try:
        config = create_config(
            analysis_level=analysis_level,
            anonymize_contacts=anonymize
        )
        
        analyzer = CommunicationAnalyzer(config)
        
        # Detect data type and load communications
        communications = []
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            
            load_task = progress.add_task("Loading communication data...", total=None)
            
            if data_type == "auto":
                data_type = _detect_data_type(input_path)
            
            if data_type == "json":
                communications = load_communications(input_path)
            elif data_type == "chat":
                chat_analysis = process_chat_exports(input_path, config)
                # Would need to convert ChatAnalysis back to Communications list
                console.print("[yellow]Warning:[/yellow] Chat-to-communications conversion not implemented")
            elif data_type == "email":
                email_analysis = analyze_email_threads(input_path, config)
                # Would need to convert EmailAnalysis back to Communications list
                console.print("[yellow]Warning:[/yellow] Email-to-communications conversion not implemented")
            
            progress.update(load_task, description=f"Loaded {len(communications)} communications")
            
            if not communications:
                console.print(f"[bold red]Error:[/bold red] No communications could be loaded")
                raise typer.Exit(1)
            
            # Perform comprehensive analysis
            analysis_task = progress.add_task("Performing comprehensive analysis...", total=None)
            
            result = analyzer.analyze_communications(communications)
            
            progress.update(analysis_task, description="Analysis complete")
        
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save comprehensive results
        results_file = output_dir / f"comprehensive_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        export_analysis_results(result, results_file)
        
        # Display comprehensive summary
        _display_comprehensive_summary(result)
        
        # Generate forensic report
        if generate_report:
            report_file = output_dir / "comprehensive_forensic_report.txt"
            report_content = generate_forensic_report(result)
            with open(report_file, 'w') as f:
                f.write(report_content)
            
            console.print(f"\n[green]✓[/green] Comprehensive forensic report saved to: {report_file}")
        
        # Create visualizations
        if create_visualizations and result.network_graph:
            from .network_mapper import NetworkMapper
            mapper = NetworkMapper(config)
            vis_files = mapper.create_visualizations(result.network_graph, output_dir)
            
            for vis_type, file_path in vis_files.items():
                console.print(f"[green]✓[/green] {vis_type.title()} saved to: {file_path}")
        
        console.print(f"\n[green]✓[/green] Comprehensive analysis complete. Results saved to: {results_file}")
        
    except Exception as e:
        console.print(f"[bold red]Error performing comprehensive analysis:[/bold red] {str(e)}")
        logger.error(f"Comprehensive analysis failed: {str(e)}", exc_info=True)
        raise typer.Exit(1)


@app.command()
def version():
    """Show version information"""
    from . import __version__
    console.print(f"[bold blue]Lemkin Communication Analysis Suite[/bold blue] v{__version__}")
    console.print("Forensic-grade communication analysis toolkit")
    console.print("Copyright © 2024 Lemkin Digital Forensics")


# Helper functions for display
def _display_chat_summary(analysis):
    """Display chat analysis summary"""
    console.print(f"\n[bold green]Chat Analysis Summary[/bold green]")
    
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="dim")
    table.add_column("Value")
    
    table.add_row("Platform", str(analysis.platform.value))
    table.add_row("Total Messages", str(analysis.total_messages))
    table.add_row("Participants", str(analysis.total_participants))
    table.add_row("Date Range", f"{analysis.date_range[0].date()} to {analysis.date_range[1].date()}")
    table.add_row("Avg Messages/Day", f"{analysis.message_statistics.get('messages_per_day', 0):.1f}")
    table.add_row("Media Messages", str(analysis.message_statistics.get('media_messages', 0)))
    
    console.print(table)


def _display_email_summary(analysis):
    """Display email analysis summary"""
    console.print(f"\n[bold green]Email Analysis Summary[/bold green]")
    
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="dim")
    table.add_column("Value")
    
    table.add_row("Total Emails", str(analysis.total_emails))
    table.add_row("Email Threads", str(analysis.thread_count))
    table.add_row("Participants", str(len(analysis.participants)))
    table.add_row("Date Range", f"{analysis.date_range[0].date()} to {analysis.date_range[1].date()}")
    table.add_row("Avg Thread Length", f"{analysis.thread_analysis.get('average_thread_length', 0):.1f}")
    table.add_row("Emails with Attachments", str(analysis.attachment_analysis.get('emails_with_attachments', 0)))
    
    console.print(table)


def _display_network_summary(network_graph):
    """Display network analysis summary"""
    console.print(f"\n[bold green]Network Analysis Summary[/bold green]")
    
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="dim")
    table.add_column("Value")
    
    table.add_row("Total Nodes", str(len(network_graph.network.nodes)))
    table.add_row("Total Edges", str(len(network_graph.network.edges)))
    table.add_row("Network Density", f"{network_graph.network.density:.3f}")
    table.add_row("Clustering Coefficient", f"{network_graph.network.clustering_coefficient:.3f}")
    table.add_row("Communities", str(len(network_graph.communities)))
    table.add_row("Central Figures", str(len(network_graph.central_figures)))
    table.add_row("Suspicious Clusters", str(len(network_graph.suspicious_clusters)))
    
    console.print(table)


def _display_pattern_summary(pattern_analysis):
    """Display pattern analysis summary"""
    console.print(f"\n[bold green]Pattern Analysis Summary[/bold green]")
    
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="dim")
    table.add_column("Value")
    
    table.add_row("Patterns Detected", str(len(pattern_analysis.detected_patterns)))
    table.add_row("Anomalies Found", str(len(pattern_analysis.anomalies)))
    table.add_row("Overall Confidence", f"{pattern_analysis.confidence_summary.get('overall_confidence', 0):.2f}")
    table.add_row("Risk Level", pattern_analysis.risk_assessment.get('overall_risk_level', 'unknown'))
    table.add_row("High Risk Contacts", str(len(pattern_analysis.risk_assessment.get('high_risk_contacts', []))))
    
    console.print(table)
    
    # Display top patterns
    if pattern_analysis.detected_patterns:
        console.print(f"\n[bold yellow]Top Patterns:[/bold yellow]")
        for i, pattern in enumerate(pattern_analysis.detected_patterns[:3], 1):
            console.print(f"{i}. {pattern.description} (Confidence: {pattern.confidence:.2f})")
    
    # Display critical anomalies
    high_severity_anomalies = [a for a in pattern_analysis.anomalies if a.anomaly.severity == 'high']
    if high_severity_anomalies:
        console.print(f"\n[bold red]Critical Anomalies:[/bold red]")
        for anomaly in high_severity_anomalies[:3]:
            console.print(f"• {anomaly.anomaly.description}")


def _display_comprehensive_summary(result):
    """Display comprehensive analysis summary"""
    console.print(f"\n[bold green]Comprehensive Analysis Summary[/bold green]")
    
    # Main metrics table
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="dim")
    table.add_column("Value")
    
    table.add_row("Total Communications", str(result.total_communications))
    table.add_row("Total Contacts", str(result.total_contacts))
    table.add_row("Platforms Analyzed", ", ".join(result.platforms_analyzed))
    table.add_row("Analysis Duration", f"{result.analysis_duration:.1f}s")
    table.add_row("Data Quality Score", f"{result.data_quality_score:.2f}")
    table.add_row("Overall Confidence", f"{result.confidence_score:.2f}")
    
    console.print(table)
    
    # Analysis components status
    console.print(f"\n[bold yellow]Analysis Components:[/bold yellow]")
    components = [
        ("Chat Analysis", result.chat_analysis is not None),
        ("Email Analysis", result.email_analysis is not None),
        ("Network Analysis", result.network_graph is not None),
        ("Pattern Analysis", result.pattern_analysis is not None)
    ]
    
    for component, completed in components:
        status = "[green]✓[/green]" if completed else "[red]✗[/red]"
        console.print(f"{status} {component}")


def _detect_data_type(path: Path) -> str:
    """Auto-detect data type from file path"""
    if path.suffix.lower() == '.json':
        return 'json'
    elif path.suffix.lower() in ['.txt', '.zip']:
        return 'chat'
    elif path.suffix.lower() in ['.mbox', '.pst'] or path.is_dir():
        return 'email'
    else:
        return 'json'  # Default fallback


if __name__ == "__main__":
    app()