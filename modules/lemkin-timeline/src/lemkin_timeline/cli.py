"""
Command-line interface for lemkin-timeline.
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, List, Dict, Any
import typer
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, TaskID
from rich import print as rprint
from rich.panel import Panel
from rich.syntax import Syntax
from rich.tree import Tree
import yaml

from .core import (
    TimelineProcessor, TimelineConfig, create_default_timeline_config,
    LanguageCode, TemporalEntityType, TimelineEventType, InconsistencyType
)
from .temporal_extractor import extract_temporal_references
from .event_sequencer import sequence_events
from .timeline_visualizer import generate_interactive_timeline
from .temporal_validator import detect_temporal_inconsistencies


# Create CLI app
app = typer.Typer(
    name="lemkin-timeline",
    help="Temporal information extraction and chronological narrative construction for legal investigations",
    add_completion=False
)

# Rich console for formatted output
console = Console()


@app.command()
def extract_temporal(
    input_path: str = typer.Argument(..., help="Path to input text file or directory"),
    output_path: Optional[str] = typer.Option(None, "--output", "-o", help="Output file path"),
    config_file: Optional[str] = typer.Option(None, "--config", "-c", help="Configuration file path"),
    language: Optional[str] = typer.Option(None, "--language", "-l", help="Document language (auto-detect if not specified)"),
    entity_types: Optional[List[str]] = typer.Option(None, "--types", "-t", help="Temporal entity types to extract"),
    min_confidence: Optional[float] = typer.Option(None, "--min-confidence", help="Minimum confidence threshold"),
    format: str = typer.Option("json", "--format", "-f", help="Output format (json, csv, xml)"),
    batch: bool = typer.Option(False, "--batch", help="Process multiple files in directory"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output")
):
    """
    Extract temporal references (dates, times, durations) from documents.
    """
    try:
        # Load configuration
        config = _load_config(config_file)
        
        # Override config with CLI arguments
        if language:
            try:
                config.primary_language = LanguageCode(language)
                config.auto_detect_language = False
            except ValueError:
                console.print(f"[red]Error: Invalid language code '{language}'[/red]")
                raise typer.Exit(1)
        
        if entity_types:
            try:
                config_types = []
                for et in entity_types:
                    config_types.append(TemporalEntityType(et.strip().upper()))
                # Update extraction flags based on selected types
                config.extract_dates = any(t in [TemporalEntityType.DATE, TemporalEntityType.DATETIME] for t in config_types)
                config.extract_times = any(t in [TemporalEntityType.TIME, TemporalEntityType.DATETIME] for t in config_types)
                config.extract_durations = TemporalEntityType.DURATION in config_types
            except ValueError as e:
                console.print(f"[red]Error: Invalid entity type - {e}[/red]")
                raise typer.Exit(1)
        
        if min_confidence is not None:
            if not 0.0 <= min_confidence <= 1.0:
                console.print("[red]Error: min_confidence must be between 0.0 and 1.0[/red]")
                raise typer.Exit(1)
            config.min_confidence = min_confidence
        
        # Initialize processor
        if verbose:
            console.print("[blue]Initializing temporal extractor...[/blue]")
        
        processor = TimelineProcessor(config)
        
        # Process input
        input_path_obj = Path(input_path)
        
        if not input_path_obj.exists():
            console.print(f"[red]Error: Input path '{input_path}' does not exist[/red]")
            raise typer.Exit(1)
        
        results = []
        
        if input_path_obj.is_file() and not batch:
            # Process single file
            if verbose:
                console.print(f"[blue]Processing file: {input_path}[/blue]")
            
            result = processor.process_document(input_path_obj)
            results.append(result)
            
        elif input_path_obj.is_dir() or batch:
            # Process directory
            if input_path_obj.is_file():
                console.print("[red]Error: --batch flag specified but input is a file[/red]")
                raise typer.Exit(1)
            
            # Find text files
            text_extensions = {'.txt', '.md', '.pdf', '.doc', '.docx', '.html', '.htm'}
            text_files = []
            
            for ext in text_extensions:
                text_files.extend(list(input_path_obj.glob(f"*{ext}")))
                text_files.extend(list(input_path_obj.glob(f"**/*{ext}")))
            
            if not text_files:
                console.print(f"[yellow]Warning: No text files found in {input_path}[/yellow]")
                raise typer.Exit(0)
            
            if verbose:
                console.print(f"[blue]Processing {len(text_files)} files...[/blue]")
            
            # Process files with progress bar
            with Progress() as progress:
                task = progress.add_task("Extracting temporal references...", total=len(text_files))
                
                for file_path in text_files:
                    try:
                        result = processor.process_document(file_path)
                        results.append(result)
                        progress.update(task, advance=1)
                    except Exception as e:
                        if verbose:
                            console.print(f"[red]Error processing {file_path}: {e}[/red]")
                        continue
        else:
            # Process text directly
            if verbose:
                console.print("[blue]Processing input as text...[/blue]")
            
            with open(input_path_obj, 'r', encoding='utf-8') as f:
                text = f.read()
            
            temporal_entities = processor.extract_temporal_references(text, input_path_obj.stem)
            result = {
                'temporal_entities': [entity.to_dict() for entity in temporal_entities],
                'metadata': {
                    'document_id': input_path_obj.stem,
                    'text_length': len(text),
                    'entities_count': len(temporal_entities)
                }
            }
            results.append(result)
        
        # Generate output
        if output_path:
            output_path_obj = Path(output_path)
            _export_results(results if len(results) > 1 else results[0], output_path_obj, format)
            console.print(f"[green]Results saved to: {output_path}[/green]")
        else:
            # Print to stdout
            _display_temporal_extraction_results(results, format, verbose)
        
        # Print summary
        total_entities = 0
        for result in results:
            total_entities += len(result.get('temporal_entities', []))
        console.print(f"\n[green]Temporal extraction complete:[/green] {total_entities} entities found from {len(results)} document(s)")
        
    except KeyboardInterrupt:
        console.print("\n[yellow]Operation cancelled by user[/yellow]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        if verbose:
            import traceback
            console.print(traceback.format_exc())
        raise typer.Exit(1)


@app.command()
def sequence_events_cmd(
    input_path: str = typer.Argument(..., help="Path to temporal entities JSON file or directory"),
    output_path: Optional[str] = typer.Option(None, "--output", "-o", help="Output file path for timeline"),
    config_file: Optional[str] = typer.Option(None, "--config", "-c", help="Configuration file path"),
    title: Optional[str] = typer.Option(None, "--title", help="Timeline title"),
    handle_uncertainty: Optional[bool] = typer.Option(None, "--uncertainty/--no-uncertainty", help="Handle temporal uncertainty"),
    max_events: Optional[int] = typer.Option(None, "--max-events", help="Maximum number of events in timeline"),
    format: str = typer.Option("json", "--format", "-f", help="Output format (json, csv)"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output")
):
    """
    Create chronological timeline from extracted temporal entities.
    """
    try:
        # Load configuration
        config = _load_config(config_file)
        
        # Override config with CLI arguments
        if handle_uncertainty is not None:
            config.enable_uncertainty_handling = handle_uncertainty
        
        if max_events is not None:
            if max_events <= 0:
                console.print("[red]Error: max_events must be positive[/red]")
                raise typer.Exit(1)
            config.max_events_per_timeline = max_events
        
        # Initialize processor
        if verbose:
            console.print("[blue]Initializing event sequencer...[/blue]")
        
        processor = TimelineProcessor(config)
        
        # Load input data
        input_path_obj = Path(input_path)
        
        if not input_path_obj.exists():
            console.print(f"[red]Error: Input path '{input_path}' does not exist[/red]")
            raise typer.Exit(1)
        
        temporal_entities = []
        
        if input_path_obj.is_file():
            # Load single file
            with open(input_path_obj, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Extract temporal entities from different data structures
            if isinstance(data, list):
                for item in data:
                    temporal_entities.extend(item.get('temporal_entities', []))
            else:
                temporal_entities = data.get('temporal_entities', [])
                
        else:
            # Load multiple JSON files from directory
            json_files = list(input_path_obj.glob("*.json"))
            
            if not json_files:
                console.print(f"[red]Error: No JSON files found in {input_path}[/red]")
                raise typer.Exit(1)
            
            if verbose:
                console.print(f"[blue]Loading {len(json_files)} JSON files...[/blue]")
            
            for json_file in json_files:
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        if isinstance(data, list):
                            for item in data:
                                temporal_entities.extend(item.get('temporal_entities', []))
                        else:
                            temporal_entities.extend(data.get('temporal_entities', []))
                except Exception as e:
                    if verbose:
                        console.print(f"[red]Error loading {json_file}: {e}[/red]")
                    continue
        
        if not temporal_entities:
            console.print("[red]Error: No temporal entities found in input data[/red]")
            raise typer.Exit(1)
        
        # Convert to Event objects and create timeline
        if verbose:
            console.print(f"[blue]Converting {len(temporal_entities)} temporal entities to events...[/blue]")
        
        events = processor._entities_to_events(
            [processor._dict_to_temporal_entity(te) for te in temporal_entities],
            "", "sequence_input"
        )
        
        # Create timeline
        if verbose:
            console.print(f"[blue]Sequencing {len(events)} events into timeline...[/blue]")
        
        timeline = processor.sequence_events(events)
        timeline.title = title or f"Generated Timeline ({len(events)} events)"
        
        # Generate output
        timeline_data = timeline.to_dict()
        
        if output_path:
            output_path_obj = Path(output_path)
            _export_results(timeline_data, output_path_obj, format)
            console.print(f"[green]Timeline saved to: {output_path}[/green]")
        else:
            # Print timeline summary
            _display_timeline_summary(timeline, verbose)
        
        # Print summary
        console.print(f"\n[green]Event sequencing complete:[/green] {len(timeline.events)} events in chronological order")
        
    except KeyboardInterrupt:
        console.print("\n[yellow]Operation cancelled by user[/yellow]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        if verbose:
            import traceback
            console.print(traceback.format_exc())
        raise typer.Exit(1)


@app.command()
def validate_timeline(
    input_path: str = typer.Argument(..., help="Path to timeline JSON file"),
    output_dir: Optional[str] = typer.Option(None, "--output-dir", "-o", help="Output directory for validation results"),
    config_file: Optional[str] = typer.Option(None, "--config", "-c", help="Configuration file path"),
    strict_chronology: Optional[bool] = typer.Option(None, "--strict/--relaxed", help="Use strict chronological validation"),
    consistency_threshold: Optional[float] = typer.Option(None, "--threshold", help="Consistency threshold (0.0-1.0)"),
    create_report: bool = typer.Option(True, "--report/--no-report", help="Create detailed validation report"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output")
):
    """
    Validate timeline for temporal inconsistencies and logical errors.
    """
    try:
        # Load configuration
        config = _load_config(config_file)
        
        # Override config with CLI arguments
        if strict_chronology is not None:
            config.strict_chronology = strict_chronology
        
        if consistency_threshold is not None:
            if not 0.0 <= consistency_threshold <= 1.0:
                console.print("[red]Error: consistency_threshold must be between 0.0 and 1.0[/red]")
                raise typer.Exit(1)
            config.consistency_threshold = consistency_threshold
        
        # Initialize processor
        if verbose:
            console.print("[blue]Initializing timeline validator...[/blue]")
        
        processor = TimelineProcessor(config)
        
        # Load timeline data
        input_path_obj = Path(input_path)
        
        if not input_path_obj.exists():
            console.print(f"[red]Error: Input file '{input_path}' does not exist[/red]")
            raise typer.Exit(1)
        
        with open(input_path_obj, 'r', encoding='utf-8') as f:
            timeline_data = json.load(f)
        
        # Reconstruct timeline object
        timeline = processor._dict_to_timeline(timeline_data)
        
        if not timeline.events:
            console.print("[red]Error: Timeline contains no events to validate[/red]")
            raise typer.Exit(1)
        
        # Perform validation
        if verbose:
            console.print(f"[blue]Validating timeline with {len(timeline.events)} events...[/blue]")
        
        validation_result = processor.temporal_validator.validate_timeline(timeline)
        inconsistencies = validation_result.inconsistencies
        
        # Display validation results
        _display_validation_results(validation_result, verbose)
        
        # Save detailed report if requested
        if create_report and output_dir:
            output_dir_path = Path(output_dir)
            output_dir_path.mkdir(parents=True, exist_ok=True)
            
            # Save validation results
            validation_file = output_dir_path / "validation_results.json"
            with open(validation_file, 'w', encoding='utf-8') as f:
                json.dump(validation_result.to_dict(), f, indent=2, default=str)
            
            # Save inconsistencies report
            inconsistencies_file = output_dir_path / "inconsistencies.json"
            with open(inconsistencies_file, 'w', encoding='utf-8') as f:
                json.dump([inc.to_dict() for inc in inconsistencies], f, indent=2, default=str)
            
            # Create HTML report
            html_report = _generate_validation_html_report(validation_result, timeline)
            report_file = output_dir_path / "validation_report.html"
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write(html_report)
            
            console.print(f"[green]Validation report saved to: {output_dir_path}[/green]")
        
        # Set exit code based on validation result
        if validation_result.critical_issues > 0:
            console.print(f"\n[red]VALIDATION FAILED:[/red] {validation_result.critical_issues} critical issues found")
            raise typer.Exit(2)  # Critical validation failure
        elif not validation_result.is_consistent:
            console.print(f"\n[yellow]VALIDATION WARNING:[/yellow] Timeline has consistency issues")
            raise typer.Exit(1)  # Non-critical validation issues
        else:
            console.print(f"\n[green]VALIDATION PASSED:[/green] Timeline is consistent")
            raise typer.Exit(0)  # Success
        
    except typer.Exit:
        raise
    except KeyboardInterrupt:
        console.print("\n[yellow]Operation cancelled by user[/yellow]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        if verbose:
            import traceback
            console.print(traceback.format_exc())
        raise typer.Exit(1)


@app.command()
def visualize(
    input_path: str = typer.Argument(..., help="Path to timeline JSON file"),
    output_path: Optional[str] = typer.Option(None, "--output", "-o", help="Output file path for visualization"),
    visualization_type: str = typer.Option("plotly", "--type", "-t", help="Visualization type (plotly, bokeh)"),
    config_file: Optional[str] = typer.Option(None, "--config", "-c", help="Configuration file path"),
    title: Optional[str] = typer.Option(None, "--title", help="Visualization title"),
    width: int = typer.Option(1200, "--width", help="Visualization width in pixels"),
    height: int = typer.Option(600, "--height", help="Visualization height in pixels"),
    theme: str = typer.Option("light", "--theme", help="Visualization theme (light, dark)"),
    show_uncertainty: bool = typer.Option(True, "--uncertainty/--no-uncertainty", help="Show uncertainty ranges"),
    show_connections: bool = typer.Option(True, "--connections/--no-connections", help="Show event connections"),
    format: str = typer.Option("html", "--format", "-f", help="Export format (html, png, svg, json)"),
    open_browser: bool = typer.Option(False, "--open", help="Open visualization in browser"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output")
):
    """
    Generate interactive timeline visualization.
    """
    try:
        # Load configuration
        config = _load_config(config_file)
        
        # Validate visualization type
        if visualization_type.lower() not in ['plotly', 'bokeh']:
            console.print(f"[red]Error: Unsupported visualization type '{visualization_type}'[/red]")
            console.print("[blue]Supported types: plotly, bokeh[/blue]")
            raise typer.Exit(1)
        
        # Initialize processor
        if verbose:
            console.print(f"[blue]Initializing {visualization_type} visualizer...[/blue]")
        
        processor = TimelineProcessor(config)
        
        # Load timeline data
        input_path_obj = Path(input_path)
        
        if not input_path_obj.exists():
            console.print(f"[red]Error: Input file '{input_path}' does not exist[/red]")
            raise typer.Exit(1)
        
        with open(input_path_obj, 'r', encoding='utf-8') as f:
            timeline_data = json.load(f)
        
        # Reconstruct timeline object
        timeline = processor._dict_to_timeline(timeline_data)
        
        if not timeline.events:
            console.print("[red]Error: Timeline contains no events to visualize[/red]")
            raise typer.Exit(1)
        
        # Generate visualization
        if verbose:
            console.print(f"[blue]Generating {visualization_type} visualization for {len(timeline.events)} events...[/blue]")
        
        visualization = processor.generate_interactive_timeline(
            timeline,
            visualization_type=visualization_type,
            title=title or timeline.title,
            width=width,
            height=height,
            theme=theme,
            show_uncertainty=show_uncertainty,
            show_connections=show_connections
        )
        
        # Export visualization
        if output_path:
            output_path_obj = Path(output_path)
        else:
            # Generate default filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path_obj = Path(f"timeline_visualization_{timestamp}.{format}")
        
        processor.timeline_visualizer.export_visualization(visualization, output_path_obj, format)
        
        console.print(f"[green]Visualization saved to: {output_path_obj}[/green]")
        
        # Open in browser if requested
        if open_browser and format.lower() == 'html':
            import webbrowser
            webbrowser.open(f"file://{output_path_obj.absolute()}")
            console.print("[blue]Opened visualization in browser[/blue]")
        
        # Print summary
        console.print(f"\n[green]Visualization complete:[/green] {visualization_type} timeline with {len(timeline.events)} events")
        
    except KeyboardInterrupt:
        console.print("\n[yellow]Operation cancelled by user[/yellow]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        if verbose:
            import traceback
            console.print(traceback.format_exc())
        raise typer.Exit(1)


@app.command()
def create_config(
    output_path: str = typer.Option("timeline_config.yaml", "--output", "-o", help="Output configuration file path"),
    format: str = typer.Option("yaml", "--format", "-f", help="Configuration format (yaml, json)"),
    interactive: bool = typer.Option(False, "--interactive", "-i", help="Interactive configuration creation"),
    template: str = typer.Option("default", "--template", help="Configuration template (default, legal, historical)")
):
    """
    Create a timeline configuration file template.
    """
    try:
        # Create configuration based on template
        if template == "legal":
            config = _create_legal_config()
        elif template == "historical":
            config = _create_historical_config()
        else:
            config = create_default_timeline_config()
        
        if interactive:
            config = _interactive_timeline_config_creation(config)
        
        output_path_obj = Path(output_path)
        
        if format.lower() == 'json' or output_path_obj.suffix.lower() == '.json':
            with open(output_path_obj, 'w', encoding='utf-8') as f:
                json.dump(config.model_dump(), f, indent=2, default=str)
        else:
            # Default to YAML
            with open(output_path_obj, 'w', encoding='utf-8') as f:
                yaml.safe_dump(config.model_dump(), f, default_flow_style=False)
        
        console.print(f"[green]Configuration file created: {output_path}[/green]")
        
        # Display configuration
        if format.lower() == 'json':
            syntax = Syntax(json.dumps(config.model_dump(), indent=2, default=str), "json")
        else:
            syntax = Syntax(yaml.safe_dump(config.model_dump(), default_flow_style=False), "yaml")
        
        console.print(Panel(syntax, title="Timeline Configuration"))
        
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def analyze(
    input_path: str = typer.Argument(..., help="Path to timeline JSON file"),
    output_path: Optional[str] = typer.Option(None, "--output", "-o", help="Output file for analysis results"),
    config_file: Optional[str] = typer.Option(None, "--config", "-c", help="Configuration file path"),
    analysis_types: List[str] = typer.Option(["summary"], "--analysis", help="Analysis types (summary, patterns, quality, gaps)"),
    detailed: bool = typer.Option(False, "--detailed", help="Generate detailed analysis"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output")
):
    """
    Analyze timeline patterns, quality, and completeness.
    """
    try:
        # Load configuration
        config = _load_config(config_file)
        
        # Initialize processor
        if verbose:
            console.print("[blue]Initializing timeline analyzer...[/blue]")
        
        processor = TimelineProcessor(config)
        
        # Load timeline data
        input_path_obj = Path(input_path)
        
        if not input_path_obj.exists():
            console.print(f"[red]Error: Input file '{input_path}' does not exist[/red]")
            raise typer.Exit(1)
        
        with open(input_path_obj, 'r', encoding='utf-8') as f:
            timeline_data = json.load(f)
        
        # Reconstruct timeline object
        timeline = processor._dict_to_timeline(timeline_data)
        
        if not timeline.events:
            console.print("[red]Error: Timeline contains no events to analyze[/red]")
            raise typer.Exit(1)
        
        # Perform analyses
        analysis_results = {}
        
        if "summary" in analysis_types:
            if verbose:
                console.print("[blue]Generating timeline summary...[/blue]")
            analysis_results['summary'] = _analyze_timeline_summary(timeline)
        
        if "patterns" in analysis_types:
            if verbose:
                console.print("[blue]Analyzing temporal patterns...[/blue]")
            analysis_results['patterns'] = processor.event_sequencer.analyze_temporal_patterns(timeline)
        
        if "quality" in analysis_types:
            if verbose:
                console.print("[blue]Assessing timeline quality...[/blue]")
            analysis_results['quality'] = _analyze_timeline_quality(timeline)
        
        if "gaps" in analysis_types:
            if verbose:
                console.print("[blue]Identifying temporal gaps...[/blue]")
            analysis_results['gaps'] = _analyze_temporal_gaps(timeline)
        
        # Display results
        _display_analysis_results(analysis_results, detailed)
        
        # Save results if requested
        if output_path:
            output_path_obj = Path(output_path)
            
            if output_path_obj.suffix.lower() == '.json':
                with open(output_path_obj, 'w', encoding='utf-8') as f:
                    json.dump(analysis_results, f, indent=2, default=str)
            elif output_path_obj.suffix.lower() in ['.yaml', '.yml']:
                with open(output_path_obj, 'w', encoding='utf-8') as f:
                    yaml.safe_dump(analysis_results, f, default_flow_style=False)
            else:
                # Default to JSON
                with open(output_path_obj, 'w', encoding='utf-8') as f:
                    json.dump(analysis_results, f, indent=2, default=str)
            
            console.print(f"[green]Analysis results saved to: {output_path}[/green]")
        
        console.print(f"\n[green]Timeline analysis complete[/green]")
        
    except KeyboardInterrupt:
        console.print("\n[yellow]Operation cancelled by user[/yellow]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        if verbose:
            import traceback
            console.print(traceback.format_exc())
        raise typer.Exit(1)


@app.command()
def version():
    """Show version information."""
    from . import __version__
    console.print(f"[green]lemkin-timeline version {__version__}[/green]")
    
    # Show component versions
    try:
        import plotly
        plotly_version = plotly.__version__
    except ImportError:
        plotly_version = "not installed"
    
    try:
        import bokeh
        bokeh_version = bokeh.__version__
    except ImportError:
        bokeh_version = "not installed"
    
    try:
        import spacy
        spacy_version = spacy.__version__
    except ImportError:
        spacy_version = "not installed"
    
    console.print(f"[blue]Dependencies:[/blue]")
    console.print(f"  Plotly: {plotly_version}")
    console.print(f"  Bokeh: {bokeh_version}")
    console.print(f"  spaCy: {spacy_version}")


# Helper functions

def _load_config(config_file: Optional[str]) -> TimelineConfig:
    """Load configuration from file or create default"""
    if config_file:
        config_path = Path(config_file)
        if not config_path.exists():
            console.print(f"[red]Error: Configuration file '{config_file}' does not exist[/red]")
            raise typer.Exit(1)
        
        try:
            if config_path.suffix.lower() in ['.yaml', '.yml']:
                with open(config_path, 'r', encoding='utf-8') as f:
                    config_dict = yaml.safe_load(f)
            else:
                with open(config_path, 'r', encoding='utf-8') as f:
                    config_dict = json.load(f)
            
            return TimelineConfig.model_validate(config_dict)
        except Exception as e:
            console.print(f"[red]Error loading configuration: {e}[/red]")
            raise typer.Exit(1)
    else:
        return create_default_timeline_config()


def _export_results(results: Any, output_path: Path, format: str) -> None:
    """Export results to file"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        if format.lower() == "json":
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        
        elif format.lower() == "csv":
            import pandas as pd
            
            # Flatten data for CSV export
            if isinstance(results, list):
                # Multiple documents
                all_entities = []
                for result in results:
                    entities = result.get('temporal_entities', [])
                    for entity in entities:
                        entity['document_id'] = result.get('metadata', {}).get('document_id', 'unknown')
                    all_entities.extend(entities)
                
                df = pd.DataFrame(all_entities)
            else:
                # Single document
                entities = results.get('temporal_entities', [])
                df = pd.DataFrame(entities)
            
            df.to_csv(output_path, index=False)
        
        elif format.lower() == "xml":
            # Simple XML export
            import xml.etree.ElementTree as ET
            
            root = ET.Element("timeline_results")
            
            if isinstance(results, list):
                doc_results = results
            else:
                doc_results = [results]
            
            for doc_result in doc_results:
                doc_elem = ET.SubElement(root, "document")
                doc_elem.set("id", doc_result.get("metadata", {}).get("document_id", "unknown"))
                
                # Add temporal entities
                entities_elem = ET.SubElement(doc_elem, "temporal_entities")
                for entity in doc_result.get("temporal_entities", []):
                    entity_elem = ET.SubElement(entities_elem, "temporal_entity")
                    for key, value in entity.items():
                        if isinstance(value, (str, int, float, bool)):
                            entity_elem.set(key, str(value))
            
            tree = ET.ElementTree(root)
            tree.write(output_path, encoding='utf-8', xml_declaration=True)
        
        else:
            raise ValueError(f"Unsupported export format: {format}")
            
    except Exception as e:
        console.print(f"[red]Error exporting results: {e}[/red]")
        raise


def _display_temporal_extraction_results(results: List[Dict[str, Any]], format: str, verbose: bool) -> None:
    """Display temporal extraction results"""
    if format.lower() == 'json':
        if len(results) == 1:
            print(json.dumps(results[0], indent=2, ensure_ascii=False, default=str))
        else:
            print(json.dumps(results, indent=2, ensure_ascii=False, default=str))
    else:
        # Table format for console
        for i, result in enumerate(results):
            if len(results) > 1:
                console.print(f"\n[bold]Document {i+1}:[/bold] {result.get('metadata', {}).get('document_id', 'Unknown')}")
            
            entities = result.get('temporal_entities', [])
            if not entities:
                console.print("[yellow]No temporal entities found[/yellow]")
                continue
            
            table = Table(show_header=True, header_style="bold magenta")
            table.add_column("Temporal Entity")
            table.add_column("Type")
            table.add_column("Confidence")
            table.add_column("Parsed Value")
            if verbose:
                table.add_column("Language")
                table.add_column("Context")
            
            for entity in entities[:20]:  # Limit display
                confidence_str = f"{entity.get('confidence', 0.0):.3f}"
                parsed_value = entity.get('normalized_form') or entity.get('parsed_date', '')
                if isinstance(parsed_value, str) and len(parsed_value) > 20:
                    parsed_value = parsed_value[:17] + "..."
                
                if verbose:
                    context = entity.get('context', '')[:40] + "..." if len(entity.get('context', '')) > 40 else entity.get('context', '')
                    table.add_row(
                        entity.get('text', ''),
                        entity.get('entity_type', ''),
                        confidence_str,
                        str(parsed_value),
                        entity.get('language', ''),
                        context
                    )
                else:
                    table.add_row(
                        entity.get('text', ''),
                        entity.get('entity_type', ''),
                        confidence_str,
                        str(parsed_value)
                    )
            
            console.print(table)
            
            if len(entities) > 20:
                console.print(f"[yellow]... and {len(entities) - 20} more entities[/yellow]")


def _display_timeline_summary(timeline: 'Timeline', verbose: bool) -> None:
    """Display timeline summary"""
    console.print("\n[bold]Timeline Summary[/bold]")
    
    # Basic info
    info_table = Table(show_header=False)
    info_table.add_column("Metric", style="bold")
    info_table.add_column("Value")
    
    info_table.add_row("Title", timeline.title)
    info_table.add_row("Total Events", str(len(timeline.events)))
    
    if timeline.start_date and timeline.end_date:
        duration = timeline.end_date - timeline.start_date
        info_table.add_row("Time Span", f"{duration.days} days")
        info_table.add_row("Start Date", timeline.start_date.strftime("%Y-%m-%d %H:%M"))
        info_table.add_row("End Date", timeline.end_date.strftime("%Y-%m-%d %H:%M"))
    
    info_table.add_row("Confidence Score", f"{timeline.confidence_score:.3f}")
    info_table.add_row("Consistency Score", f"{timeline.consistency_score:.3f}")
    
    console.print(info_table)
    
    if verbose and timeline.events:
        console.print("\n[bold]Recent Events:[/bold]")
        event_table = Table(show_header=True, header_style="bold cyan")
        event_table.add_column("Time")
        event_table.add_column("Event")
        event_table.add_column("Type")
        event_table.add_column("Confidence")
        
        for event in timeline.events[:10]:  # Show first 10 events
            event_table.add_row(
                event.start_time.strftime("%Y-%m-%d %H:%M"),
                event.title[:50] + ("..." if len(event.title) > 50 else ""),
                event.event_type.value,
                f"{event.confidence:.3f}"
            )
        
        console.print(event_table)
        
        if len(timeline.events) > 10:
            console.print(f"[yellow]... and {len(timeline.events) - 10} more events[/yellow]")


def _display_validation_results(validation_result: 'ValidationResult', verbose: bool) -> None:
    """Display timeline validation results"""
    console.print("\n[bold]Timeline Validation Results[/bold]")
    
    # Overall result
    if validation_result.is_consistent:
        console.print(f"[green]✓ VALIDATION PASSED[/green] (Score: {validation_result.consistency_score:.3f})")
    else:
        console.print(f"[red]✗ VALIDATION FAILED[/red] (Score: {validation_result.consistency_score:.3f})")
    
    # Issue summary
    summary_table = Table(show_header=True, header_style="bold red")
    summary_table.add_column("Severity")
    summary_table.add_column("Count")
    summary_table.add_column("Description")
    
    summary_table.add_row("Critical", str(validation_result.critical_issues), "Issues that must be resolved")
    summary_table.add_row("High", str(validation_result.high_priority_issues), "Important consistency problems")
    summary_table.add_row("Medium", str(validation_result.medium_priority_issues), "Moderate concerns")
    summary_table.add_row("Low", str(validation_result.low_priority_issues), "Minor issues")
    
    console.print("\n[bold]Issue Summary:[/bold]")
    console.print(summary_table)
    
    # Quality metrics
    console.print("\n[bold]Quality Metrics:[/bold]")
    quality_table = Table(show_header=False)
    quality_table.add_column("Metric", style="bold")
    quality_table.add_column("Score")
    
    quality_table.add_row("Temporal Accuracy", f"{validation_result.temporal_accuracy:.3f}")
    quality_table.add_row("Completeness", f"{validation_result.completeness:.3f}")
    quality_table.add_row("Logical Consistency", f"{validation_result.logical_consistency:.3f}")
    
    console.print(quality_table)
    
    # Detailed inconsistencies
    if verbose and validation_result.inconsistencies:
        console.print("\n[bold]Detailed Issues:[/bold]")
        
        for i, inconsistency in enumerate(validation_result.inconsistencies[:10], 1):
            severity_color = {
                'critical': 'red',
                'high': 'yellow',
                'medium': 'blue',
                'low': 'green'
            }.get(inconsistency.severity, 'white')
            
            console.print(f"\n[bold {severity_color}]{i}. {inconsistency.severity.upper()}:[/bold {severity_color}] {inconsistency.description}")
            console.print(f"   Type: {inconsistency.inconsistency_type.value}")
            console.print(f"   Confidence: {inconsistency.confidence:.3f}")
            
            if inconsistency.suggested_resolution:
                console.print(f"   Suggested fix: {inconsistency.suggested_resolution}")
        
        if len(validation_result.inconsistencies) > 10:
            remaining = len(validation_result.inconsistencies) - 10
            console.print(f"\n[yellow]... and {remaining} more issues[/yellow]")
    
    # Improvement suggestions
    if validation_result.improvement_suggestions:
        console.print("\n[bold]Improvement Suggestions:[/bold]")
        for suggestion in validation_result.improvement_suggestions:
            console.print(f"• {suggestion}")


def _display_analysis_results(results: Dict[str, Any], detailed: bool) -> None:
    """Display timeline analysis results"""
    console.print("\n[bold]Timeline Analysis Results[/bold]")
    
    for analysis_type, analysis_data in results.items():
        console.print(f"\n[bold cyan]{analysis_type.title()} Analysis:[/bold cyan]")
        
        if analysis_type == "summary":
            _display_summary_analysis(analysis_data)
        elif analysis_type == "patterns":
            _display_patterns_analysis(analysis_data, detailed)
        elif analysis_type == "quality":
            _display_quality_analysis(analysis_data, detailed)
        elif analysis_type == "gaps":
            _display_gaps_analysis(analysis_data, detailed)


def _display_summary_analysis(summary: Dict[str, Any]) -> None:
    """Display summary analysis"""
    table = Table(show_header=False)
    table.add_column("Metric", style="bold")
    table.add_column("Value")
    
    for key, value in summary.items():
        if isinstance(value, float):
            table.add_row(key.replace('_', ' ').title(), f"{value:.3f}")
        else:
            table.add_row(key.replace('_', ' ').title(), str(value))
    
    console.print(table)


def _display_patterns_analysis(patterns: Dict[str, Any], detailed: bool) -> None:
    """Display temporal patterns analysis"""
    for pattern_type, pattern_data in patterns.items():
        console.print(f"\n[yellow]{pattern_type.replace('_', ' ').title()}:[/yellow]")
        
        if isinstance(pattern_data, dict):
            for key, value in pattern_data.items():
                if isinstance(value, (int, float)):
                    console.print(f"  {key.replace('_', ' ').title()}: {value}")
                elif detailed:
                    console.print(f"  {key.replace('_', ' ').title()}: {value}")


def _display_quality_analysis(quality: Dict[str, Any], detailed: bool) -> None:
    """Display quality analysis"""
    # This would be implemented based on the quality analysis structure
    console.print("Quality analysis display not yet implemented")


def _display_gaps_analysis(gaps: Dict[str, Any], detailed: bool) -> None:
    """Display temporal gaps analysis"""
    # This would be implemented based on the gaps analysis structure
    console.print("Gaps analysis display not yet implemented")


def _create_legal_config() -> TimelineConfig:
    """Create configuration optimized for legal investigations"""
    config = create_default_timeline_config()
    config.strict_chronology = True
    config.validate_causality = True
    config.require_human_review = True
    config.consistency_threshold = 0.9
    config.enable_validation = True
    return config


def _create_historical_config() -> TimelineConfig:
    """Create configuration optimized for historical analysis"""
    config = create_default_timeline_config()
    config.handle_fuzzy_dates = True
    config.uncertainty_window_hours = 24 * 30  # 30 days uncertainty
    config.extract_relative_times = True
    config.min_confidence = 0.4  # Lower threshold for historical texts
    return config


def _interactive_timeline_config_creation(config: TimelineConfig) -> TimelineConfig:
    """Interactive configuration creation for timeline processing"""
    console.print("[bold]Interactive Timeline Configuration[/bold]\n")
    
    # Language settings
    available_languages = [lang.value for lang in LanguageCode]
    primary_lang = typer.prompt(
        f"Primary language ({'/'.join(available_languages)})",
        default=config.primary_language.value
    )
    
    try:
        config.primary_language = LanguageCode(primary_lang)
    except ValueError:
        console.print(f"[yellow]Invalid language '{primary_lang}', using default[/yellow]")
    
    # Temporal extraction settings
    config.extract_dates = typer.confirm("Extract dates?", default=config.extract_dates)
    config.extract_times = typer.confirm("Extract times?", default=config.extract_times)
    config.extract_durations = typer.confirm("Extract durations?", default=config.extract_durations)
    config.handle_fuzzy_dates = typer.confirm("Handle fuzzy/approximate dates?", default=config.handle_fuzzy_dates)
    
    # Confidence threshold
    min_confidence = typer.prompt(
        "Minimum confidence threshold (0.0-1.0)",
        default=config.min_confidence,
        type=float
    )
    
    if 0.0 <= min_confidence <= 1.0:
        config.min_confidence = min_confidence
    else:
        console.print("[yellow]Invalid confidence value, using default[/yellow]")
    
    # Validation settings
    config.enable_validation = typer.confirm("Enable timeline validation?", default=config.enable_validation)
    if config.enable_validation:
        config.strict_chronology = typer.confirm("Use strict chronological validation?", default=config.strict_chronology)
        config.validate_causality = typer.confirm("Validate causal relationships?", default=config.validate_causality)
    
    # Uncertainty handling
    config.enable_uncertainty_handling = typer.confirm("Handle temporal uncertainty?", default=config.enable_uncertainty_handling)
    if config.enable_uncertainty_handling:
        uncertainty_hours = typer.prompt(
            "Uncertainty window in hours",
            default=config.uncertainty_window_hours,
            type=int
        )
        if uncertainty_hours >= 0:
            config.uncertainty_window_hours = uncertainty_hours
    
    return config


def _generate_validation_html_report(validation_result: 'ValidationResult', timeline: 'Timeline') -> str:
    """Generate HTML validation report"""
    html_template = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Timeline Validation Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
            .section {{ margin: 20px 0; }}
            .metric {{ display: inline-block; margin: 10px; padding: 10px; background-color: #f9f9f9; border-radius: 3px; }}
            .issue {{ margin: 10px 0; padding: 10px; border-left: 4px solid #ccc; }}
            .critical {{ border-left-color: #d32f2f; }}
            .high {{ border-left-color: #f57f17; }}
            .medium {{ border-left-color: #1976d2; }}
            .low {{ border-left-color: #388e3c; }}
            table {{ width: 100%; border-collapse: collapse; }}
            th, td {{ padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }}
            th {{ background-color: #f2f2f2; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>Timeline Validation Report</h1>
            <p>Timeline: {timeline.title}</p>
            <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p>Status: {'PASSED' if validation_result.is_consistent else 'FAILED'}</p>
        </div>
        
        <div class="section">
            <h2>Summary</h2>
            <div class="metric">Consistency Score: {validation_result.consistency_score:.3f}</div>
            <div class="metric">Critical Issues: {validation_result.critical_issues}</div>
            <div class="metric">High Priority: {validation_result.high_priority_issues}</div>
            <div class="metric">Medium Priority: {validation_result.medium_priority_issues}</div>
            <div class="metric">Low Priority: {validation_result.low_priority_issues}</div>
        </div>
        
        <div class="section">
            <h2>Quality Metrics</h2>
            <div class="metric">Temporal Accuracy: {validation_result.temporal_accuracy:.3f}</div>
            <div class="metric">Completeness: {validation_result.completeness:.3f}</div>
            <div class="metric">Logical Consistency: {validation_result.logical_consistency:.3f}</div>
        </div>
        
        <div class="section">
            <h2>Issues Found</h2>
            {''.join([
                f'<div class="issue {inc.severity}"><strong>{inc.severity.upper()}</strong>: {inc.description}</div>'
                for inc in validation_result.inconsistencies[:20]
            ])}
        </div>
        
        <div class="section">
            <h2>Improvement Suggestions</h2>
            <ul>
                {''.join([f'<li>{suggestion}</li>' for suggestion in validation_result.improvement_suggestions])}
            </ul>
        </div>
    </body>
    </html>
    """
    return html_template


# Additional helper functions for timeline analysis

def _analyze_timeline_summary(timeline: 'Timeline') -> Dict[str, Any]:
    """Generate timeline summary analysis"""
    return {
        'total_events': len(timeline.events),
        'time_span_days': (timeline.end_date - timeline.start_date).days if timeline.start_date and timeline.end_date else 0,
        'average_confidence': sum(e.confidence for e in timeline.events) / len(timeline.events) if timeline.events else 0,
        'events_with_end_times': len([e for e in timeline.events if e.end_time]),
        'events_with_participants': len([e for e in timeline.events if e.participants]),
        'events_with_locations': len([e for e in timeline.events if e.locations]),
        'fuzzy_events': len([e for e in timeline.events if e.is_fuzzy])
    }


def _analyze_timeline_quality(timeline: 'Timeline') -> Dict[str, Any]:
    """Analyze timeline quality metrics"""
    # This would implement comprehensive quality analysis
    return {
        'data_completeness': 0.8,  # Placeholder
        'temporal_precision': 0.7,
        'source_reliability': 0.9,
        'cross_validation_score': 0.6
    }


def _analyze_temporal_gaps(timeline: 'Timeline') -> Dict[str, Any]:
    """Identify significant temporal gaps in timeline"""
    if len(timeline.events) < 2:
        return {'gaps': [], 'largest_gap_days': 0}
    
    gaps = []
    sorted_events = sorted(timeline.events, key=lambda x: x.start_time)
    
    for i in range(len(sorted_events) - 1):
        current_event = sorted_events[i]
        next_event = sorted_events[i + 1]
        
        # Calculate gap
        gap_start = current_event.end_time or current_event.start_time
        gap_duration = next_event.start_time - gap_start
        
        if gap_duration.days > 1:  # Gaps longer than 1 day
            gaps.append({
                'start_event': current_event.title,
                'end_event': next_event.title,
                'gap_duration_days': gap_duration.days,
                'gap_start': gap_start.isoformat(),
                'gap_end': next_event.start_time.isoformat()
            })
    
    return {
        'gaps': gaps,
        'total_gaps': len(gaps),
        'largest_gap_days': max([g['gap_duration_days'] for g in gaps], default=0),
        'average_gap_days': sum([g['gap_duration_days'] for g in gaps]) / len(gaps) if gaps else 0
    }


if __name__ == "__main__":
    app()