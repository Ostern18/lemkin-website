"""
Command-line interface for lemkin-ner.
"""

import json
import sys
from pathlib import Path
from typing import Optional, List, Dict, Any
import typer
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, TaskID
from rich import print as rprint
from rich.panel import Panel
from rich.syntax import Syntax
import yaml

from .core import LegalNERProcessor, NERConfig, LanguageCode, EntityType, create_default_config
from .entity_validator import EntityValidator


# Create CLI app
app = typer.Typer(
    name="lemkin-ner",
    help="Multilingual named entity recognition and linking for legal investigations",
    add_completion=False
)

# Rich console for formatted output
console = Console()


@app.command()
def extract_entities(
    input_path: str = typer.Argument(..., help="Path to input text file or directory"),
    output_path: Optional[str] = typer.Option(None, "--output", "-o", help="Output file path"),
    config_file: Optional[str] = typer.Option(None, "--config", "-c", help="Configuration file path"),
    language: Optional[str] = typer.Option(None, "--language", "-l", help="Document language (auto-detect if not specified)"),
    entity_types: Optional[List[str]] = typer.Option(None, "--types", "-t", help="Entity types to extract (comma-separated)"),
    min_confidence: Optional[float] = typer.Option(None, "--min-confidence", help="Minimum confidence threshold"),
    format: str = typer.Option("json", "--format", "-f", help="Output format (json, csv, xml)"),
    batch: bool = typer.Option(False, "--batch", help="Process multiple files in directory"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output")
):
    """
    Extract named entities from text or documents.
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
                config.entity_types = [EntityType(t.strip().upper()) for t in entity_types]
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
            console.print("[blue]Initializing NER processor...[/blue]")
        
        processor = LegalNERProcessor(config)
        
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
                task = progress.add_task("Processing files...", total=len(text_files))
                
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
            
            result = processor.process_text(text, input_path_obj.stem)
            results.append(result)
        
        # Generate output
        if output_path:
            output_path_obj = Path(output_path)
            processor.export_results(results if len(results) > 1 else results[0], output_path_obj, format)
            console.print(f"[green]Results saved to: {output_path}[/green]")
        else:
            # Print to stdout
            _display_results(results, format, verbose)
        
        # Print summary
        total_entities = sum(len(r.get('entities', [])) for r in results)
        console.print(f"\n[green]Extraction complete:[/green] {total_entities} entities found from {len(results)} document(s)")
        
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
def link_entities(
    input_path: str = typer.Argument(..., help="Path to extracted entities JSON file or directory"),
    output_path: Optional[str] = typer.Option(None, "--output", "-o", help="Output file path for entity graph"),
    config_file: Optional[str] = typer.Option(None, "--config", "-c", help="Configuration file path"),
    similarity_threshold: Optional[float] = typer.Option(None, "--threshold", help="Similarity threshold for linking"),
    format: str = typer.Option("json", "--format", "-f", help="Output format (json, gexf)"),
    deduplicate: bool = typer.Option(True, "--deduplicate/--no-deduplicate", help="Remove duplicate entities"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output")
):
    """
    Link entities across documents and create entity graph.
    """
    try:
        # Load configuration
        config = _load_config(config_file)
        
        # Override config with CLI arguments
        if similarity_threshold is not None:
            if not 0.0 <= similarity_threshold <= 1.0:
                console.print("[red]Error: similarity_threshold must be between 0.0 and 1.0[/red]")
                raise typer.Exit(1)
            config.similarity_threshold = similarity_threshold
        
        # Initialize processor
        if verbose:
            console.print("[blue]Initializing entity linker...[/blue]")
        
        processor = LegalNERProcessor(config)
        
        # Load input data
        input_path_obj = Path(input_path)
        
        if not input_path_obj.exists():
            console.print(f"[red]Error: Input path '{input_path}' does not exist[/red]")
            raise typer.Exit(1)
        
        document_results = []
        
        if input_path_obj.is_file():
            # Load single file
            with open(input_path_obj, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            if isinstance(data, list):
                document_results = data
            else:
                document_results = [data]
                
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
                            document_results.extend(data)
                        else:
                            document_results.append(data)
                except Exception as e:
                    if verbose:
                        console.print(f"[red]Error loading {json_file}: {e}[/red]")
                    continue
        
        if not document_results:
            console.print("[red]Error: No valid document results found[/red]")
            raise typer.Exit(1)
        
        # Link entities
        if verbose:
            console.print(f"[blue]Linking entities across {len(document_results)} documents...[/blue]")
        
        entity_graph = processor.link_entities_across_documents(document_results)
        
        # Deduplicate entities if requested
        if deduplicate:
            if verbose:
                console.print("[blue]Deduplicating entities...[/blue]")
            
            original_count = len(entity_graph.entities)
            all_entities = list(entity_graph.entities.values())
            deduplicated_entities = processor.entity_linker.deduplicate_entities(all_entities)
            
            # Rebuild graph with deduplicated entities
            entity_graph.entities.clear()
            for entity in deduplicated_entities:
                entity_graph.entities[entity.entity_id] = entity
            
            if verbose:
                console.print(f"[green]Deduplicated: {original_count} -> {len(deduplicated_entities)} entities[/green]")
        
        # Generate output
        if output_path:
            output_path_obj = Path(output_path)
            
            if format.lower() == "gexf":
                _export_entity_graph_gexf(entity_graph, output_path_obj)
            else:
                # JSON format
                graph_data = {
                    "graph_id": entity_graph.graph_id,
                    "entities": [entity.to_dict() for entity in entity_graph.entities.values()],
                    "relationships": entity_graph.relationships,
                    "metadata": entity_graph.metadata,
                    "created_at": entity_graph.created_at.isoformat()
                }
                
                with open(output_path_obj, 'w', encoding='utf-8') as f:
                    json.dump(graph_data, f, indent=2, ensure_ascii=False, default=str)
            
            console.print(f"[green]Entity graph saved to: {output_path}[/green]")
        else:
            # Display summary
            _display_entity_graph_summary(entity_graph, verbose)
        
        # Print summary
        console.print(f"\n[green]Entity linking complete:[/green] {len(entity_graph.entities)} entities, {len(entity_graph.relationships)} relationships")
        
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
def validate_entities(
    input_path: str = typer.Argument(..., help="Path to extracted entities JSON file"),
    output_dir: Optional[str] = typer.Option(None, "--output-dir", "-o", help="Output directory for validation results"),
    config_file: Optional[str] = typer.Option(None, "--config", "-c", help="Configuration file path"),
    create_review_tasks: bool = typer.Option(True, "--create-review-tasks/--no-review-tasks", help="Create human review tasks"),
    validation_threshold: Optional[float] = typer.Option(None, "--threshold", help="Validation confidence threshold"),
    require_human_review: Optional[bool] = typer.Option(None, "--require-human-review", help="Require human review for all entities"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output")
):
    """
    Validate extracted entities and create human review tasks.
    """
    try:
        # Load configuration
        config = _load_config(config_file)
        
        # Override config with CLI arguments
        if validation_threshold is not None:
            if not 0.0 <= validation_threshold <= 1.0:
                console.print("[red]Error: validation_threshold must be between 0.0 and 1.0[/red]")
                raise typer.Exit(1)
            config.validation_threshold = validation_threshold
        
        if require_human_review is not None:
            config.require_human_review = require_human_review
        
        # Initialize validator
        if verbose:
            console.print("[blue]Initializing entity validator...[/blue]")
        
        validator = EntityValidator(config)
        
        # Load input data
        input_path_obj = Path(input_path)
        
        if not input_path_obj.exists():
            console.print(f"[red]Error: Input file '{input_path}' does not exist[/red]")
            raise typer.Exit(1)
        
        with open(input_path_obj, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Extract entities from data
        entities = []
        if isinstance(data, list):
            # Multiple documents
            for doc_result in data:
                for entity_dict in doc_result.get('entities', []):
                    from .core import Entity
                    entity = Entity.model_validate(entity_dict)
                    entities.append(entity)
        else:
            # Single document
            for entity_dict in data.get('entities', []):
                from .core import Entity
                entity = Entity.model_validate(entity_dict)
                entities.append(entity)
        
        if not entities:
            console.print("[red]Error: No entities found in input data[/red]")
            raise typer.Exit(1)
        
        # Validate entities
        if verbose:
            console.print(f"[blue]Validating {len(entities)} entities...[/blue]")
        
        validation_results = validator.validate_batch(entities)
        
        # Generate quality report
        quality_report = validator.generate_quality_report(validation_results)
        
        # Create review tasks if requested
        review_summary = None
        if create_review_tasks:
            output_dir_path = Path(output_dir) if output_dir else Path("validation_output")
            review_summary = validator.create_human_review_tasks(validation_results, output_dir_path)
        
        # Save validation results
        if output_dir:
            output_dir_path = Path(output_dir)
            output_dir_path.mkdir(parents=True, exist_ok=True)
            
            # Save validation results
            validation_file = output_dir_path / "validation_results.json"
            with open(validation_file, 'w', encoding='utf-8') as f:
                json.dump([result.model_dump() for result in validation_results], f, indent=2, default=str)
            
            # Save quality report
            quality_file = output_dir_path / "quality_report.json"
            with open(quality_file, 'w', encoding='utf-8') as f:
                json.dump(quality_report, f, indent=2, default=str)
            
            console.print(f"[green]Validation results saved to: {output_dir_path}[/green]")
        
        # Display summary
        _display_validation_summary(validation_results, quality_report, review_summary, verbose)
        
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
def analyze_graph(
    input_path: str = typer.Argument(..., help="Path to entity graph JSON file"),
    output_path: Optional[str] = typer.Option(None, "--output", "-o", help="Output file for analysis results"),
    metrics: List[str] = typer.Option(["centrality", "clusters", "components"], "--metrics", help="Analysis metrics to compute"),
    min_cluster_size: int = typer.Option(3, "--min-cluster-size", help="Minimum cluster size for community detection"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output")
):
    """
    Analyze entity relationship graphs.
    """
    try:
        # Load entity graph
        input_path_obj = Path(input_path)
        
        if not input_path_obj.exists():
            console.print(f"[red]Error: Input file '{input_path}' does not exist[/red]")
            raise typer.Exit(1)
        
        if verbose:
            console.print(f"[blue]Loading entity graph from: {input_path}[/blue]")
        
        with open(input_path_obj, 'r', encoding='utf-8') as f:
            graph_data = json.load(f)
        
        # Analyze graph
        analysis_results = _analyze_entity_graph(graph_data, metrics, min_cluster_size, verbose)
        
        # Save or display results
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
        else:
            _display_graph_analysis(analysis_results, verbose)
        
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
    output_path: str = typer.Option("ner_config.yaml", "--output", "-o", help="Output configuration file path"),
    format: str = typer.Option("yaml", "--format", "-f", help="Configuration format (yaml, json)"),
    interactive: bool = typer.Option(False, "--interactive", "-i", help="Interactive configuration creation")
):
    """
    Create a configuration file template.
    """
    try:
        config = create_default_config()
        
        if interactive:
            config = _interactive_config_creation(config)
        
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
        
        console.print(Panel(syntax, title="Configuration File"))
        
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def process_feedback(
    feedback_path: str = typer.Argument(..., help="Path to human feedback file (JSON or CSV)"),
    output_path: Optional[str] = typer.Option(None, "--output", "-o", help="Output path for updated models/rules"),
    config_file: Optional[str] = typer.Option(None, "--config", "-c", help="Configuration file path"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output")
):
    """
    Process human feedback to improve entity extraction.
    """
    try:
        # Load configuration
        config = _load_config(config_file)
        
        # Initialize validator
        validator = EntityValidator(config)
        
        # Process feedback
        if verbose:
            console.print(f"[blue]Processing feedback from: {feedback_path}[/blue]")
        
        feedback_summary = validator.process_human_feedback(feedback_path)
        
        # Display summary
        _display_feedback_summary(feedback_summary, verbose)
        
        # Save updated configuration or models if output path provided
        if output_path:
            # This would involve retraining or updating models based on feedback
            # For now, just save the feedback summary
            output_path_obj = Path(output_path)
            with open(output_path_obj, 'w', encoding='utf-8') as f:
                json.dump(feedback_summary, f, indent=2, default=str)
            
            console.print(f"[green]Feedback processing results saved to: {output_path}[/green]")
        
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
    console.print(f"[green]lemkin-ner version {__version__}[/green]")


# Helper functions

def _load_config(config_file: Optional[str]) -> NERConfig:
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
            
            return NERConfig.model_validate(config_dict)
        except Exception as e:
            console.print(f"[red]Error loading configuration: {e}[/red]")
            raise typer.Exit(1)
    else:
        return create_default_config()


def _display_results(results: List[Dict[str, Any]], format: str, verbose: bool) -> None:
    """Display extraction results"""
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
            
            entities = result.get('entities', [])
            if not entities:
                console.print("[yellow]No entities found[/yellow]")
                continue
            
            table = Table(show_header=True, header_style="bold magenta")
            table.add_column("Entity")
            table.add_column("Type")
            table.add_column("Confidence")
            if verbose:
                table.add_column("Language")
                table.add_column("Context")
            
            for entity in entities[:20]:  # Limit display
                confidence_str = f"{entity.get('confidence', 0.0):.3f}"
                if verbose:
                    context = entity.get('context', '')[:50] + "..." if len(entity.get('context', '')) > 50 else entity.get('context', '')
                    table.add_row(
                        entity.get('text', ''),
                        entity.get('entity_type', ''),
                        confidence_str,
                        entity.get('language', ''),
                        context
                    )
                else:
                    table.add_row(
                        entity.get('text', ''),
                        entity.get('entity_type', ''),
                        confidence_str
                    )
            
            console.print(table)
            
            if len(entities) > 20:
                console.print(f"[yellow]... and {len(entities) - 20} more entities[/yellow]")


def _display_entity_graph_summary(entity_graph, verbose: bool) -> None:
    """Display entity graph summary"""
    console.print("\n[bold]Entity Graph Summary[/bold]")
    
    # Basic statistics
    stats_table = Table(show_header=False)
    stats_table.add_column("Metric", style="bold")
    stats_table.add_column("Value")
    
    stats_table.add_row("Total Entities", str(len(entity_graph.entities)))
    stats_table.add_row("Total Relationships", str(len(entity_graph.relationships)))
    
    # Entity type distribution
    type_counts = {}
    for entity in entity_graph.entities.values():
        entity_type = entity.entity_type.value
        type_counts[entity_type] = type_counts.get(entity_type, 0) + 1
    
    for entity_type, count in sorted(type_counts.items()):
        stats_table.add_row(f"  {entity_type}", str(count))
    
    console.print(stats_table)
    
    if verbose and entity_graph.relationships:
        # Relationship types
        rel_types = {}
        for rel in entity_graph.relationships:
            rel_type = rel.get('relationship_type', 'Unknown')
            rel_types[rel_type] = rel_types.get(rel_type, 0) + 1
        
        console.print("\n[bold]Relationship Types:[/bold]")
        rel_table = Table(show_header=True, header_style="bold cyan")
        rel_table.add_column("Type")
        rel_table.add_column("Count")
        
        for rel_type, count in sorted(rel_types.items()):
            rel_table.add_row(rel_type, str(count))
        
        console.print(rel_table)


def _display_validation_summary(validation_results, quality_report, review_summary, verbose: bool) -> None:
    """Display validation summary"""
    console.print("\n[bold]Validation Summary[/bold]")
    
    # Overall statistics
    summary_table = Table(show_header=False)
    summary_table.add_column("Metric", style="bold")
    summary_table.add_column("Value")
    
    summary = quality_report.get('summary', {})
    summary_table.add_row("Total Entities", str(summary.get('total_entities', 0)))
    summary_table.add_row("Valid Entities", str(summary.get('valid_entities', 0)))
    summary_table.add_row("Validity Rate", f"{summary.get('validity_rate', 0.0) * 100:.1f}%")
    summary_table.add_row("Average Confidence", f"{summary.get('average_confidence', 0.0):.3f}")
    
    console.print(summary_table)
    
    # Quality distribution
    if 'quality_distribution' in quality_report:
        console.print("\n[bold]Quality Distribution:[/bold]")
        qual_table = Table(show_header=True, header_style="bold green")
        qual_table.add_column("Quality Level")
        qual_table.add_column("Count")
        qual_table.add_column("Percentage")
        
        total = summary.get('total_entities', 1)
        for level, count in quality_report['quality_distribution'].items():
            percentage = (count / total) * 100 if total > 0 else 0
            qual_table.add_row(level.title(), str(count), f"{percentage:.1f}%")
        
        console.print(qual_table)
    
    # Review task summary
    if review_summary:
        console.print("\n[bold]Human Review Tasks:[/bold]")
        review_table = Table(show_header=False)
        review_table.add_column("Priority", style="bold")
        review_table.add_column("Count")
        
        review_table.add_row("High Priority", str(review_summary.get('high_priority', 0)))
        review_table.add_row("Medium Priority", str(review_summary.get('medium_priority', 0)))
        review_table.add_row("Low Priority", str(review_summary.get('low_priority', 0)))
        review_table.add_row("Total Review Needed", str(review_summary.get('review_needed', 0)))
        
        console.print(review_table)
    
    # Top issues
    if verbose and 'top_issues' in quality_report:
        console.print("\n[bold]Top Issues:[/bold]")
        for issue_info in quality_report['top_issues'][:5]:
            console.print(f"• {issue_info['issue']} ({issue_info['count']} times)")


def _display_feedback_summary(feedback_summary, verbose: bool) -> None:
    """Display feedback processing summary"""
    console.print("\n[bold]Feedback Processing Summary[/bold]")
    
    summary_table = Table(show_header=False)
    summary_table.add_column("Metric", style="bold")
    summary_table.add_column("Value")
    
    summary_table.add_row("Total Entries", str(feedback_summary.get('total_entries', 0)))
    summary_table.add_row("Processed", str(feedback_summary.get('processed', 0)))
    summary_table.add_row("Approved", str(feedback_summary.get('approvals', 0)))
    summary_table.add_row("Rejected", str(feedback_summary.get('rejections', 0)))
    summary_table.add_row("Corrected", str(feedback_summary.get('corrections', 0)))
    
    console.print(summary_table)


def _display_graph_analysis(analysis_results, verbose: bool) -> None:
    """Display graph analysis results"""
    console.print("\n[bold]Entity Graph Analysis[/bold]")
    
    # Display analysis results in formatted tables
    for metric, results in analysis_results.items():
        if metric == 'basic_stats':
            console.print(f"\n[bold cyan]{metric.replace('_', ' ').title()}[/bold cyan]")
            stats_table = Table(show_header=False)
            stats_table.add_column("Metric", style="bold")
            stats_table.add_column("Value")
            
            for key, value in results.items():
                stats_table.add_row(key.replace('_', ' ').title(), str(value))
            
            console.print(stats_table)
        
        elif isinstance(results, dict) and verbose:
            console.print(f"\n[bold cyan]{metric.replace('_', ' ').title()}[/bold cyan]")
            console.print(json.dumps(results, indent=2, default=str))


def _interactive_config_creation(config: NERConfig) -> NERConfig:
    """Interactive configuration creation"""
    console.print("[bold]Interactive Configuration Creation[/bold]\n")
    
    # Language selection
    available_languages = [lang.value for lang in LanguageCode]
    primary_lang = typer.prompt(
        f"Primary language ({'/'.join(available_languages)})",
        default=config.primary_language.value
    )
    
    try:
        config.primary_language = LanguageCode(primary_lang)
    except ValueError:
        console.print(f"[yellow]Invalid language '{primary_lang}', using default[/yellow]")
    
    # Entity types
    available_types = [t.value for t in EntityType]
    console.print(f"\nAvailable entity types: {', '.join(available_types)}")
    entity_types_str = typer.prompt(
        "Entity types to extract (comma-separated)",
        default=','.join([t.value for t in config.entity_types])
    )
    
    try:
        selected_types = [EntityType(t.strip().upper()) for t in entity_types_str.split(',')]
        config.entity_types = selected_types
    except ValueError as e:
        console.print(f"[yellow]Invalid entity type, using defaults: {e}[/yellow]")
    
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
    
    # Enable features
    config.enable_entity_linking = typer.confirm(
        "Enable entity linking?",
        default=config.enable_entity_linking
    )
    
    config.enable_validation = typer.confirm(
        "Enable validation?",
        default=config.enable_validation
    )
    
    config.require_human_review = typer.confirm(
        "Require human review?",
        default=config.require_human_review
    )
    
    return config


def _export_entity_graph_gexf(entity_graph, output_path: Path) -> None:
    """Export entity graph to GEXF format"""
    try:
        import networkx as nx
        
        # Create NetworkX graph
        G = nx.Graph()
        
        # Add nodes (entities)
        for entity in entity_graph.entities.values():
            G.add_node(
                entity.entity_id,
                label=entity.text,
                entity_type=entity.entity_type.value,
                confidence=entity.confidence,
                language=entity.language.value,
                document_id=entity.document_id
            )
        
        # Add edges (relationships)
        for rel in entity_graph.relationships:
            G.add_edge(
                rel['source_id'],
                rel['target_id'],
                relationship_type=rel['relationship_type'],
                confidence=rel.get('confidence', 1.0)
            )
        
        # Export to GEXF
        nx.write_gexf(G, output_path)
        
    except ImportError:
        console.print("[red]Error: NetworkX not available for GEXF export[/red]")
        raise typer.Exit(1)


def _analyze_entity_graph(graph_data: Dict[str, Any], metrics: List[str], 
                         min_cluster_size: int, verbose: bool) -> Dict[str, Any]:
    """Analyze entity graph"""
    try:
        import networkx as nx
        from collections import defaultdict, Counter
        
        # Create NetworkX graph from data
        G = nx.Graph()
        
        # Add nodes
        entities = graph_data.get('entities', [])
        for entity in entities:
            G.add_node(
                entity['entity_id'],
                **{k: v for k, v in entity.items() if k != 'entity_id'}
            )
        
        # Add edges
        relationships = graph_data.get('relationships', [])
        for rel in relationships:
            G.add_edge(
                rel['source_id'],
                rel['target_id'],
                **{k: v for k, v in rel.items() if k not in ['source_id', 'target_id']}
            )
        
        analysis_results = {}
        
        # Basic statistics
        analysis_results['basic_stats'] = {
            'num_nodes': G.number_of_nodes(),
            'num_edges': G.number_of_edges(),
            'density': nx.density(G),
            'is_connected': nx.is_connected(G),
            'num_connected_components': nx.number_connected_components(G)
        }
        
        # Centrality measures
        if 'centrality' in metrics and G.number_of_nodes() > 0:
            if verbose:
                console.print("[blue]Computing centrality measures...[/blue]")
            
            try:
                degree_cent = nx.degree_centrality(G)
                betweenness_cent = nx.betweenness_centrality(G)
                closeness_cent = nx.closeness_centrality(G)
                
                # Get top entities by centrality
                analysis_results['centrality'] = {
                    'top_degree': sorted(degree_cent.items(), key=lambda x: x[1], reverse=True)[:10],
                    'top_betweenness': sorted(betweenness_cent.items(), key=lambda x: x[1], reverse=True)[:10],
                    'top_closeness': sorted(closeness_cent.items(), key=lambda x: x[1], reverse=True)[:10]
                }
            except Exception as e:
                if verbose:
                    console.print(f"[yellow]Warning: Could not compute centrality: {e}[/yellow]")
        
        # Community detection
        if 'clusters' in metrics and G.number_of_nodes() > 0:
            if verbose:
                console.print("[blue]Detecting communities...[/blue]")
            
            try:
                communities = list(nx.connected_components(G))
                large_communities = [c for c in communities if len(c) >= min_cluster_size]
                
                analysis_results['clusters'] = {
                    'num_communities': len(communities),
                    'large_communities': len(large_communities),
                    'largest_community_size': max(len(c) for c in communities) if communities else 0,
                    'community_sizes': [len(c) for c in communities]
                }
            except Exception as e:
                if verbose:
                    console.print(f"[yellow]Warning: Could not detect communities: {e}[/yellow]")
        
        # Connected components analysis
        if 'components' in metrics and G.number_of_nodes() > 0:
            if verbose:
                console.print("[blue]Analyzing connected components...[/blue]")
            
            try:
                components = list(nx.connected_components(G))
                component_sizes = [len(c) for c in components]
                
                analysis_results['components'] = {
                    'num_components': len(components),
                    'component_sizes': component_sizes,
                    'largest_component_size': max(component_sizes) if component_sizes else 0,
                    'isolated_nodes': sum(1 for size in component_sizes if size == 1)
                }
            except Exception as e:
                if verbose:
                    console.print(f"[yellow]Warning: Could not analyze components: {e}[/yellow]")
        
        return analysis_results
        
    except ImportError:
        console.print("[red]Error: NetworkX required for graph analysis[/red]")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()