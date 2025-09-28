import typer
import json
from pathlib import Path
from typing import Optional, List
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.json import JSON
from datetime import datetime, timezone

from .core import PIIRedactor, RedactionConfig, EntityType, RedactionType

app = typer.Typer(
    name="lemkin-redaction",
    help="PII redaction toolkit for text, image, audio, and video content",
    no_args_is_help=True
)
console = Console()

@app.command()
def redact_text(
    text: str = typer.Argument(..., help="Text content to redact"),
    output_path: Optional[Path] = typer.Option(None, help="Output file path for redacted text"),
    min_confidence: float = typer.Option(0.7, help="Minimum confidence threshold (0.0-1.0)"),
    entity_types: Optional[str] = typer.Option(None, help="Comma-separated entity types to redact"),
    config_file: Optional[Path] = typer.Option(None, help="Path to JSON configuration file"),
    generate_report: bool = typer.Option(True, help="Generate redaction report")
):
    """Redact PII from text content"""
    
    try:
        # Load configuration
        config = _load_config(config_file, min_confidence, entity_types)
        config.generate_report = generate_report
        
        # Initialize redactor
        redactor = PIIRedactor(config)
        
        # Perform redaction
        with console.status("Redacting text content..."):
            result = redactor.redact_text(text, output_path)
        
        # Display results
        _display_redaction_result(result, "Text Redaction")
        
        if output_path and result.redacted_content_path:
            console.print(f"[green]✓ Redacted text saved to: {result.redacted_content_path}[/green]")
        
        if result.report_path:
            console.print(f"[blue]📊 Report saved to: {result.report_path}[/blue]")
            
    except Exception as e:
        console.print(f"[red]Error redacting text: {e}[/red]")
        raise typer.Exit(1)

@app.command()
def redact_image(
    image_path: Path = typer.Argument(..., help="Path to image file"),
    output_path: Optional[Path] = typer.Option(None, help="Output path for redacted image"),
    min_confidence: float = typer.Option(0.7, help="Minimum confidence threshold (0.0-1.0)"),
    entity_types: Optional[str] = typer.Option(None, help="Comma-separated entity types to redact"),
    config_file: Optional[Path] = typer.Option(None, help="Path to JSON configuration file"),
    generate_report: bool = typer.Option(True, help="Generate redaction report")
):
    """Redact PII from image files"""
    
    if not image_path.exists():
        console.print(f"[red]Error: Image file not found: {image_path}[/red]")
        raise typer.Exit(1)
    
    try:
        # Load configuration
        config = _load_config(config_file, min_confidence, entity_types)
        config.generate_report = generate_report
        
        # Initialize redactor
        redactor = PIIRedactor(config)
        
        # Perform redaction
        with console.status(f"Redacting image: {image_path.name}..."):
            result = redactor.redact_image(image_path, output_path)
        
        # Display results
        _display_redaction_result(result, "Image Redaction")
        
        if result.redacted_content_path:
            console.print(f"[green]✓ Redacted image saved to: {result.redacted_content_path}[/green]")
        
        if result.report_path:
            console.print(f"[blue]📊 Report saved to: {result.report_path}[/blue]")
            
    except Exception as e:
        console.print(f"[red]Error redacting image: {e}[/red]")
        raise typer.Exit(1)

@app.command()
def redact_audio(
    audio_path: Path = typer.Argument(..., help="Path to audio file"),
    output_path: Optional[Path] = typer.Option(None, help="Output path for redacted audio"),
    min_confidence: float = typer.Option(0.7, help="Minimum confidence threshold (0.0-1.0)"),
    entity_types: Optional[str] = typer.Option(None, help="Comma-separated entity types to redact"),
    config_file: Optional[Path] = typer.Option(None, help="Path to JSON configuration file"),
    generate_report: bool = typer.Option(True, help="Generate redaction report")
):
    """Redact PII from audio files"""
    
    if not audio_path.exists():
        console.print(f"[red]Error: Audio file not found: {audio_path}[/red]")
        raise typer.Exit(1)
    
    try:
        # Load configuration
        config = _load_config(config_file, min_confidence, entity_types)
        config.generate_report = generate_report
        
        # Initialize redactor
        redactor = PIIRedactor(config)
        
        # Perform redaction
        with console.status(f"Redacting audio: {audio_path.name}..."):
            result = redactor.redact_audio(audio_path, output_path)
        
        # Display results
        _display_redaction_result(result, "Audio Redaction")
        
        if result.redacted_content_path:
            console.print(f"[green]✓ Redacted audio saved to: {result.redacted_content_path}[/green]")
        
        if result.report_path:
            console.print(f"[blue]📊 Report saved to: {result.report_path}[/blue]")
            
    except Exception as e:
        console.print(f"[red]Error redacting audio: {e}[/red]")
        raise typer.Exit(1)

@app.command()
def redact_video(
    video_path: Path = typer.Argument(..., help="Path to video file"),
    output_path: Optional[Path] = typer.Option(None, help="Output path for redacted video"),
    min_confidence: float = typer.Option(0.7, help="Minimum confidence threshold (0.0-1.0)"),
    entity_types: Optional[str] = typer.Option(None, help="Comma-separated entity types to redact"),
    config_file: Optional[Path] = typer.Option(None, help="Path to JSON configuration file"),
    generate_report: bool = typer.Option(True, help="Generate redaction report")
):
    """Redact PII from video files"""
    
    if not video_path.exists():
        console.print(f"[red]Error: Video file not found: {video_path}[/red]")
        raise typer.Exit(1)
    
    try:
        # Load configuration
        config = _load_config(config_file, min_confidence, entity_types)
        config.generate_report = generate_report
        
        # Initialize redactor
        redactor = PIIRedactor(config)
        
        # Perform redaction
        with console.status(f"Redacting video: {video_path.name}..."):
            result = redactor.redact_video(video_path, output_path)
        
        # Display results
        _display_redaction_result(result, "Video Redaction")
        
        if result.redacted_content_path:
            console.print(f"[green]✓ Redacted video saved to: {result.redacted_content_path}[/green]")
        
        if result.report_path:
            console.print(f"[blue]📊 Report saved to: {result.report_path}[/blue]")
            
    except Exception as e:
        console.print(f"[red]Error redacting video: {e}[/red]")
        raise typer.Exit(1)

@app.command()
def redact_file(
    file_path: Path = typer.Argument(..., help="Path to file (auto-detects type)"),
    output_path: Optional[Path] = typer.Option(None, help="Output path for redacted file"),
    min_confidence: float = typer.Option(0.7, help="Minimum confidence threshold (0.0-1.0)"),
    entity_types: Optional[str] = typer.Option(None, help="Comma-separated entity types to redact"),
    config_file: Optional[Path] = typer.Option(None, help="Path to JSON configuration file"),
    generate_report: bool = typer.Option(True, help="Generate redaction report")
):
    """Auto-detect file type and apply appropriate redaction"""
    
    if not file_path.exists():
        console.print(f"[red]Error: File not found: {file_path}[/red]")
        raise typer.Exit(1)
    
    try:
        # Load configuration
        config = _load_config(config_file, min_confidence, entity_types)
        config.generate_report = generate_report
        
        # Initialize redactor
        redactor = PIIRedactor(config)
        
        # Detect and display file type
        suffix = file_path.suffix.lower()
        supported_formats = redactor.get_supported_formats()
        
        file_type = None
        for format_type, extensions in supported_formats.items():
            if suffix in extensions:
                file_type = format_type
                break
        
        if not file_type:
            console.print(f"[red]Unsupported file type: {suffix}[/red]")
            console.print(f"Supported formats: {json.dumps(supported_formats, indent=2)}")
            raise typer.Exit(1)
        
        console.print(f"[cyan]Detected file type: {file_type}[/cyan]")
        
        # Perform redaction
        with console.status(f"Redacting {file_type} file: {file_path.name}..."):
            result = redactor.redact_file(file_path, output_path)
        
        # Display results
        _display_redaction_result(result, f"File Redaction ({file_type.title()})")
        
        if result.redacted_content_path:
            console.print(f"[green]✓ Redacted file saved to: {result.redacted_content_path}[/green]")
        
        if result.report_path:
            console.print(f"[blue]📊 Report saved to: {result.report_path}[/blue]")
            
    except Exception as e:
        console.print(f"[red]Error redacting file: {e}[/red]")
        raise typer.Exit(1)

@app.command()
def batch_redact(
    input_dir: Path = typer.Argument(..., help="Directory containing files to redact"),
    output_dir: Path = typer.Argument(..., help="Output directory for redacted files"),
    pattern: str = typer.Option("*", help="File pattern to match (e.g., '*.txt', '*.jpg')"),
    min_confidence: float = typer.Option(0.7, help="Minimum confidence threshold (0.0-1.0)"),
    entity_types: Optional[str] = typer.Option(None, help="Comma-separated entity types to redact"),
    config_file: Optional[Path] = typer.Option(None, help="Path to JSON configuration file"),
    generate_report: bool = typer.Option(True, help="Generate redaction reports"),
    recursive: bool = typer.Option(False, help="Process subdirectories recursively")
):
    """Process multiple files in batch"""
    
    if not input_dir.exists() or not input_dir.is_dir():
        console.print(f"[red]Error: Input directory not found: {input_dir}[/red]")
        raise typer.Exit(1)
    
    try:
        # Load configuration
        config = _load_config(config_file, min_confidence, entity_types)
        config.generate_report = generate_report
        
        # Initialize redactor
        redactor = PIIRedactor(config)
        
        # Find files to process
        if recursive:
            file_paths = list(input_dir.rglob(pattern))
        else:
            file_paths = list(input_dir.glob(pattern))
        
        file_paths = [p for p in file_paths if p.is_file()]
        
        if not file_paths:
            console.print(f"[yellow]No files found matching pattern: {pattern}[/yellow]")
            return
        
        console.print(f"[cyan]Found {len(file_paths)} files to process[/cyan]")
        
        # Process files
        with console.status("Processing batch redaction..."):
            results = redactor.batch_redact(file_paths, output_dir)
        
        # Display summary
        successful = sum(1 for r in results if not r.errors)
        failed = len(results) - successful
        total_entities = sum(r.redacted_count for r in results)
        
        table = Table(title="Batch Redaction Summary")
        table.add_column("Metric", style="cyan")
        table.add_column("Count", style="green")
        
        table.add_row("Files Processed", str(len(results)))
        table.add_row("Successful", str(successful))
        table.add_row("Failed", str(failed))
        table.add_row("Total Entities Redacted", str(total_entities))
        
        console.print(table)
        console.print(f"[green]✓ Batch redaction completed. Output saved to: {output_dir}[/green]")
        
        # Show failed files if any
        if failed > 0:
            console.print("\n[red]Failed Files:[/red]")
            for result in results:
                if result.errors:
                    console.print(f"  • {result.errors[0]}")
                    
    except Exception as e:
        console.print(f"[red]Error in batch redaction: {e}[/red]")
        raise typer.Exit(1)

@app.command()
def configure(
    config_path: Path = typer.Option("redaction_config.json", help="Configuration file path"),
    entity_types: Optional[str] = typer.Option(None, help="Comma-separated entity types"),
    min_confidence: Optional[float] = typer.Option(None, help="Minimum confidence threshold"),
    language: Optional[str] = typer.Option(None, help="Language code (e.g., 'en', 'es')"),
    preserve_formatting: Optional[bool] = typer.Option(None, help="Preserve text formatting"),
    generate_reports: Optional[bool] = typer.Option(None, help="Generate redaction reports"),
    show_config: bool = typer.Option(False, help="Show current configuration")
):
    """Manage redaction configuration settings"""
    
    try:
        if show_config:
            # Display current configuration
            if config_path.exists():
                with open(config_path, 'r') as f:
                    config_data = json.load(f)
                console.print(JSON.from_data(config_data))
            else:
                default_config = RedactionConfig()
                console.print("No configuration file found. Default configuration:")
                console.print(JSON.from_data(default_config.dict()))
            return
        
        # Create or update configuration
        config_data = {}
        if config_path.exists():
            with open(config_path, 'r') as f:
                config_data = json.load(f)
        
        # Update with provided values
        if entity_types:
            entity_list = [EntityType(et.strip().upper()) for et in entity_types.split(",")]
            config_data['entity_types'] = [et.value for et in entity_list]
        
        if min_confidence is not None:
            config_data['min_confidence'] = min_confidence
        
        if language:
            config_data['language'] = language
        
        if preserve_formatting is not None:
            config_data['preserve_formatting'] = preserve_formatting
        
        if generate_reports is not None:
            config_data['generate_report'] = generate_reports
        
        # Save configuration
        with open(config_path, 'w') as f:
            json.dump(config_data, f, indent=2)
        
        console.print(f"[green]✓ Configuration saved to: {config_path}[/green]")
        console.print("Current configuration:")
        console.print(JSON.from_data(config_data))
        
    except Exception as e:
        console.print(f"[red]Error managing configuration: {e}[/red]")
        raise typer.Exit(1)

@app.command()
def list_formats():
    """List supported file formats"""
    
    redactor = PIIRedactor()
    supported_formats = redactor.get_supported_formats()
    
    table = Table(title="Supported File Formats")
    table.add_column("Category", style="cyan")
    table.add_column("Extensions", style="green")
    
    for category, extensions in supported_formats.items():
        table.add_row(category.title(), ", ".join(extensions))
    
    console.print(table)

@app.command()
def list_entities():
    """List available entity types for redaction"""
    
    table = Table(title="Available Entity Types")
    table.add_column("Entity Type", style="cyan")
    table.add_column("Description", style="white")
    
    entity_descriptions = {
        "PERSON": "Personal names and identifiers",
        "ORGANIZATION": "Company and organization names",
        "LOCATION": "Geographic locations and addresses",
        "EMAIL": "Email addresses",
        "PHONE": "Phone numbers",
        "SSN": "Social Security Numbers",
        "CREDIT_CARD": "Credit card numbers",
        "IP_ADDRESS": "IP addresses",
        "DATE": "Dates and timestamps",
        "ADDRESS": "Physical addresses",
        "MEDICAL": "Medical information and identifiers",
        "FINANCIAL": "Financial account numbers",
        "CUSTOM": "Custom patterns defined by user"
    }
    
    for entity_type in EntityType:
        description = entity_descriptions.get(entity_type.value, "Custom entity type")
        table.add_row(entity_type.value, description)
    
    console.print(table)

def _load_config(config_file: Optional[Path], min_confidence: float, entity_types: Optional[str]) -> RedactionConfig:
    """Load configuration from file or create from parameters"""
    config_data = {}
    
    # Load from file if provided
    if config_file and config_file.exists():
        with open(config_file, 'r') as f:
            config_data = json.load(f)
    
    # Override with command line parameters
    if min_confidence != 0.7:  # Default value
        config_data['min_confidence'] = min_confidence
    
    if entity_types:
        entity_list = [EntityType(et.strip().upper()) for et in entity_types.split(",")]
        config_data['entity_types'] = entity_list
    
    return RedactionConfig(**config_data)

def _display_redaction_result(result, title: str):
    """Display redaction results in a formatted table"""
    
    status_color = "green" if not result.errors else "red"
    
    panel_content = f"""
[bold]Operation ID:[/bold] {result.operation_id}
[bold]Content Type:[/bold] {result.content_type}
[bold]Status:[/bold] [{status_color}]{"SUCCESS" if not result.errors else "ERROR"}[/{status_color}]
[bold]Entities Detected:[/bold] {result.total_entities}
[bold]Entities Redacted:[/bold] {result.redacted_count}
[bold]Processing Time:[/bold] {result.processing_time:.2f}s
[bold]Timestamp:[/bold] {result.timestamp.isoformat()}
    """
    
    console.print(Panel(panel_content, title=title))
    
    if result.entities_redacted:
        entity_table = Table(title="Redacted Entities")
        entity_table.add_column("Type", style="cyan")
        entity_table.add_column("Text", style="yellow")
        entity_table.add_column("Confidence", style="green")
        entity_table.add_column("Position", style="blue")
        
        for entity in result.entities_redacted[:10]:  # Show first 10
            entity_table.add_row(
                entity.entity_type.value,
                entity.text[:30] + "..." if len(entity.text) > 30 else entity.text,
                f"{entity.confidence:.2f}",
                f"{entity.start_pos}-{entity.end_pos}"
            )
        
        console.print(entity_table)
        
        if len(result.entities_redacted) > 10:
            console.print(f"[dim]... and {len(result.entities_redacted) - 10} more entities[/dim]")
    
    if result.warnings:
        console.print("\n[yellow]Warnings:[/yellow]")
        for warning in result.warnings:
            console.print(f"  • {warning}")
    
    if result.errors:
        console.print("\n[red]Errors:[/red]")
        for error in result.errors:
            console.print(f"  • {error}")

if __name__ == "__main__":
    app()