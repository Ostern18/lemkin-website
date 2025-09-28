"""
Command-line interface for lemkin-audio toolkit.
Provides comprehensive audio analysis capabilities through CLI commands.
"""

import json
import sys
from pathlib import Path
from typing import List, Optional

import typer
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.panel import Panel
from rich.syntax import Syntax
from rich import print as rprint
from loguru import logger

from . import (
    transcribe_audio,
    identify_speakers,
    enhance_audio_quality,
    detect_audio_manipulation,
    AudioAnalyzer,
    AudioAnalysisConfig,
    get_supported_languages,
    validate_language_code,
    __version__
)

# Initialize CLI app
app = typer.Typer(
    name="lemkin-audio",
    help="Professional audio analysis toolkit for forensic applications",
    no_args_is_help=True
)

# Initialize console
console = Console()

# Configure logging
logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level="INFO"
)


@app.command()
def version():
    """Display version information."""
    console.print(f"[bold green]lemkin-audio[/bold green] version {__version__}")


@app.command()
def transcribe(
    audio_file: Path = typer.Argument(..., help="Path to audio file to transcribe"),
    language: Optional[str] = typer.Option(None, "--language", "-l", help="Target language code (e.g., 'en', 'es', 'fr')"),
    model: str = typer.Option("large-v3", "--model", "-m", help="Whisper model to use"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output file path (JSON format)"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose output"),
    word_timestamps: bool = typer.Option(True, "--word-timestamps/--no-word-timestamps", help="Include word-level timestamps")
):
    """
    Transcribe audio file to text with multi-language support.
    
    Supports automatic language detection and provides word-level timestamps
    with confidence scores for forensic applications.
    """
    if verbose:
        logger.configure(handlers=[{"sink": sys.stderr, "level": "DEBUG"}])
    
    # Validate inputs
    if not audio_file.exists():
        console.print(f"[bold red]Error:[/bold red] Audio file not found: {audio_file}")
        raise typer.Exit(1)
    
    if language and not validate_language_code(language):
        console.print(f"[bold red]Error:[/bold red] Unsupported language code: {language}")
        console.print("Use 'lemkin-audio list-languages' to see supported languages")
        raise typer.Exit(1)
    
    # Show progress
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task("Transcribing audio...", total=None)
        
        try:
            # Perform transcription
            result = transcribe_audio(
                audio_path=audio_file,
                language=language,
                model_name=model
            )
            
            progress.update(task, description="Transcription completed!")
            
        except Exception as e:
            console.print(f"[bold red]Error during transcription:[/bold red] {e}")
            raise typer.Exit(1)
    
    # Display results
    console.print("\n[bold green]Transcription Results[/bold green]")
    console.print(f"[bold]File:[/bold] {audio_file}")
    console.print(f"[bold]Language:[/bold] {result.language}")
    console.print(f"[bold]Confidence:[/bold] {result.confidence:.3f}")
    console.print(f"[bold]Duration:[/bold] {result.audio_metadata.duration:.2f} seconds")
    console.print(f"[bold]Model:[/bold] {result.model_used}")
    
    # Show transcription text
    console.print("\n[bold]Transcription:[/bold]")
    console.print(Panel(result.full_text, title="Text", border_style="blue"))
    
    # Show segments if verbose
    if verbose and result.segments:
        console.print("\n[bold]Segments:[/bold]")
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Start", style="dim", width=8)
        table.add_column("End", style="dim", width=8)
        table.add_column("Confidence", justify="center", width=10)
        table.add_column("Text")
        
        for segment in result.segments[:10]:  # Show first 10 segments
            table.add_row(
                f"{segment.start_time:.2f}s",
                f"{segment.end_time:.2f}s", 
                f"{segment.confidence:.3f}",
                segment.text[:50] + "..." if len(segment.text) > 50 else segment.text
            )
        
        console.print(table)
    
    # Save output if requested
    if output:
        save_results(result, output, "transcription")
        console.print(f"\n[green]Results saved to:[/green] {output}")


@app.command()
def identify_speaker(
    audio_file: Path = typer.Argument(..., help="Path to audio file for speaker analysis"),
    n_speakers: Optional[int] = typer.Option(None, "--speakers", "-s", help="Expected number of speakers (auto-detect if not specified)"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output file path (JSON format)"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose output")
):
    """
    Identify and analyze speakers in audio file.
    
    Performs speaker diarization (who spoke when) and creates voice biometric
    profiles for forensic speaker identification.
    """
    if verbose:
        logger.configure(handlers=[{"sink": sys.stderr, "level": "DEBUG"}])
    
    # Validate inputs
    if not audio_file.exists():
        console.print(f"[bold red]Error:[/bold red] Audio file not found: {audio_file}")
        raise typer.Exit(1)
    
    # Show progress
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task("Analyzing speakers...", total=None)
        
        try:
            # Create config with speaker count if provided
            config = AudioAnalysisConfig()
            
            # Perform speaker analysis
            result = identify_speakers(audio_path=audio_file, n_speakers=n_speakers, config=config)
            
            progress.update(task, description="Speaker analysis completed!")
            
        except Exception as e:
            console.print(f"[bold red]Error during speaker analysis:[/bold red] {e}")
            raise typer.Exit(1)
    
    # Display results
    console.print("\n[bold green]Speaker Analysis Results[/bold green]")
    console.print(f"[bold]File:[/bold] {audio_file}")
    console.print(f"[bold]Total Speakers:[/bold] {result.total_speakers}")
    console.print(f"[bold]Duration:[/bold] {result.audio_metadata.duration:.2f} seconds")
    console.print(f"[bold]Speaker Changes:[/bold] {len(result.speaker_changes)}")
    
    # Show speaker profiles
    if result.speakers:
        console.print("\n[bold]Speaker Profiles:[/bold]")
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Speaker ID", style="cyan")
        table.add_column("Samples", justify="center")
        table.add_column("Quality", justify="center")
        table.add_column("Avg Pitch", justify="center")
        table.add_column("Pitch Std", justify="center")
        
        for speaker in result.speakers:
            avg_pitch = speaker.average_features.get('pitch_mean', 0)
            pitch_std = speaker.average_features.get('pitch_std', 0)
            
            table.add_row(
                speaker.speaker_id,
                str(speaker.sample_count),
                f"{speaker.quality_score:.3f}",
                f"{avg_pitch:.1f} Hz" if avg_pitch > 0 else "N/A",
                f"{pitch_std:.1f}" if pitch_std > 0 else "N/A"
            )
        
        console.print(table)
    
    # Show diarization if verbose
    if verbose and result.diarization_segments:
        console.print("\n[bold]Diarization Segments:[/bold]")
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Start", style="dim", width=8)
        table.add_column("End", style="dim", width=8)
        table.add_column("Speaker", style="cyan")
        table.add_column("Confidence", justify="center")
        
        for segment in result.diarization_segments[:15]:  # Show first 15
            table.add_row(
                f"{segment['start_time']:.2f}s",
                f"{segment['end_time']:.2f}s",
                segment['speaker_id'],
                f"{segment['confidence']:.3f}"
            )
        
        console.print(table)
    
    # Save output if requested
    if output:
        save_results(result, output, "speaker_analysis")
        console.print(f"\n[green]Results saved to:[/green] {output}")


@app.command()
def enhance(
    audio_file: Path = typer.Argument(..., help="Path to audio file to enhance"),
    output_dir: Optional[Path] = typer.Option(None, "--output-dir", "-d", help="Output directory for enhanced audio"),
    noise_reduction: bool = typer.Option(True, "--noise-reduction/--no-noise-reduction", help="Enable noise reduction"),
    normalize: bool = typer.Option(True, "--normalize/--no-normalize", help="Normalize audio levels"),
    target_sr: int = typer.Option(16000, "--sample-rate", "-sr", help="Target sample rate"),
    output_format: Optional[Path] = typer.Option(None, "--output", "-o", help="Output file path for results (JSON format)"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose output")
):
    """
    Enhance audio quality with noise reduction and optimization.
    
    Applies professional audio enhancement techniques while preserving
    evidential integrity for forensic applications.
    """
    if verbose:
        logger.configure(handlers=[{"sink": sys.stderr, "level": "DEBUG"}])
    
    # Validate inputs
    if not audio_file.exists():
        console.print(f"[bold red]Error:[/bold red] Audio file not found: {audio_file}")
        raise typer.Exit(1)
    
    # Show progress
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task("Enhancing audio...", total=None)
        
        try:
            # Create configuration
            config = AudioAnalysisConfig(
                noise_reduction_enabled=noise_reduction,
                normalize_audio=normalize,
                target_sample_rate=target_sr,
                output_directory=output_dir
            )
            
            # Perform enhancement
            result = enhance_audio_quality(audio_path=audio_file, config=config)
            
            progress.update(task, description="Audio enhancement completed!")
            
        except Exception as e:
            console.print(f"[bold red]Error during enhancement:[/bold red] {e}")
            raise typer.Exit(1)
    
    # Display results
    console.print("\n[bold green]Audio Enhancement Results[/bold green]")
    console.print(f"[bold]Original File:[/bold] {audio_file}")
    console.print(f"[bold]Enhanced File:[/bold] {result.enhanced_file_path}")
    console.print(f"[bold]Processing Time:[/bold] {result.processing_time:.2f} seconds")
    
    # Show applied enhancements
    console.print(f"\n[bold]Applied Enhancements:[/bold]")
    for enhancement in result.enhancement_applied:
        console.print(f"  • {enhancement.replace('_', ' ').title()}")
    
    # Show quality improvements
    console.print(f"\n[bold]Quality Improvements:[/bold]")
    improvements_table = Table(show_header=True, header_style="bold magenta")
    improvements_table.add_column("Metric", style="cyan")
    improvements_table.add_column("Before", justify="center")
    improvements_table.add_column("After", justify="center")
    improvements_table.add_column("Change", justify="center")
    
    # Compare key metrics
    metrics_comparison = [
        ("Dynamic Range", f"{result.before_metrics.dynamic_range_db:.2f} dB", 
         f"{result.after_metrics.dynamic_range_db:.2f} dB",
         f"{result.quality_improvement['dynamic_range_change']:+.2f} dB"),
        ("Peak Level", f"{result.before_metrics.peak_level_db:.2f} dB",
         f"{result.after_metrics.peak_level_db:.2f} dB", 
         f"{result.quality_improvement['peak_level_change']:+.2f} dB"),
        ("RMS Level", f"{result.before_metrics.rms_level_db:.2f} dB",
         f"{result.after_metrics.rms_level_db:.2f} dB",
         f"{result.quality_improvement['rms_level_change']:+.2f} dB")
    ]
    
    for metric, before, after, change in metrics_comparison:
        improvements_table.add_row(metric, before, after, change)
    
    console.print(improvements_table)
    
    # Save output if requested
    if output_format:
        save_results(result, output_format, "enhancement")
        console.print(f"\n[green]Results saved to:[/green] {output_format}")


@app.command()
def detect_manipulation(
    audio_file: Path = typer.Argument(..., help="Path to audio file to analyze for manipulation"),
    threshold: float = typer.Option(0.7, "--threshold", "-t", help="Authenticity threshold (0.0-1.0)"),
    enable_deepfake: bool = typer.Option(True, "--deepfake/--no-deepfake", help="Enable deepfake detection"),
    enable_compression: bool = typer.Option(True, "--compression/--no-compression", help="Enable compression analysis"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output file path (JSON format)"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose output")
):
    """
    Detect audio manipulation and assess authenticity.
    
    Analyzes audio for signs of editing, splicing, deepfakes, and other
    manipulations using forensic-grade detection algorithms.
    """
    if verbose:
        logger.configure(handlers=[{"sink": sys.stderr, "level": "DEBUG"}])
    
    # Validate inputs
    if not audio_file.exists():
        console.print(f"[bold red]Error:[/bold red] Audio file not found: {audio_file}")
        raise typer.Exit(1)
    
    # Show progress
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task("Analyzing authenticity...", total=None)
        
        try:
            # Create configuration
            config = AudioAnalysisConfig(
                authenticity_threshold=threshold,
                enable_deepfake_detection=enable_deepfake,
                enable_compression_analysis=enable_compression
            )
            
            # Perform authenticity analysis
            result = detect_audio_manipulation(audio_path=audio_file, config=config)
            
            progress.update(task, description="Authenticity analysis completed!")
            
        except Exception as e:
            console.print(f"[bold red]Error during authenticity analysis:[/bold red] {e}")
            raise typer.Exit(1)
    
    # Display results
    authenticity_color = "green" if result.is_authentic else "red"
    authenticity_text = "AUTHENTIC" if result.is_authentic else "MANIPULATED"
    
    console.print(f"\n[bold {authenticity_color}]Audio Authenticity Analysis[/bold {authenticity_color}]")
    console.print(f"[bold]File:[/bold] {audio_file}")
    console.print(f"[bold]Status:[/bold] [{authenticity_color}]{authenticity_text}[/{authenticity_color}]")
    console.print(f"[bold]Confidence:[/bold] {result.confidence.value.upper()}")
    console.print(f"[bold]Authenticity Score:[/bold] {result.authenticity_score:.3f}")
    console.print(f"[bold]Processing Time:[/bold] {result.processing_time:.2f} seconds")
    
    # Show detected manipulations
    if result.manipulations_detected and result.manipulations_detected[0].value != "none":
        console.print(f"\n[bold red]Detected Manipulations:[/bold red]")
        for manipulation in result.manipulations_detected:
            console.print(f"  • {manipulation.value.replace('_', ' ').title()}")
    else:
        console.print(f"\n[bold green]No manipulations detected[/bold green]")
    
    # Show manipulation locations if any
    if result.manipulation_locations and verbose:
        console.print(f"\n[bold]Manipulation Locations:[/bold]")
        locations_table = Table(show_header=True, header_style="bold magenta")
        locations_table.add_column("Time", style="dim")
        locations_table.add_column("Type", style="yellow")
        locations_table.add_column("Confidence", justify="center")
        
        for location in result.manipulation_locations[:10]:  # Show first 10
            locations_table.add_row(
                f"{location['time']:.2f}s",
                location['type'],
                f"{location['confidence']:.3f}"
            )
        
        console.print(locations_table)
    
    # Show technical analysis if verbose
    if verbose:
        console.print(f"\n[bold]Technical Analysis Summary:[/bold]")
        
        # Temporal analysis
        if 'temporal_analysis' in result.technical_analysis:
            temporal = result.temporal_analysis
            console.print(f"[cyan]Temporal Consistency:[/cyan] {temporal['overall_consistency']:.3f}")
            console.print(f"[cyan]Potential Splices:[/cyan] {temporal['splice_count']}")
        
        # Spectral analysis  
        if 'spectral_analysis' in result.technical_analysis:
            spectral = result.spectral_analysis
            console.print(f"[cyan]Spectral Consistency:[/cyan] {spectral.get('spectral_flatness_consistency', 0):.3f}")
        
        # Deepfake analysis
        if 'deepfake_analysis' in result.technical_analysis:
            deepfake = result.technical_analysis['deepfake_analysis']
            console.print(f"[cyan]Deepfake Likelihood:[/cyan] {deepfake.get('deepfake_likelihood', 0):.3f}")
    
    # Save output if requested
    if output:
        save_results(result, output, "authenticity")
        console.print(f"\n[green]Results saved to:[/green] {output}")


@app.command()
def analyze(
    audio_file: Path = typer.Argument(..., help="Path to audio file for comprehensive analysis"),
    language: Optional[str] = typer.Option(None, "--language", "-l", help="Target language for transcription"),
    output_dir: Optional[Path] = typer.Option(None, "--output-dir", "-d", help="Output directory for all results"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose output"),
    save_enhanced: bool = typer.Option(False, "--save-enhanced", help="Save enhanced audio file")
):
    """
    Perform comprehensive audio analysis including all available features.
    
    Combines transcription, speaker analysis, enhancement, and authenticity 
    detection in a single comprehensive forensic analysis workflow.
    """
    if verbose:
        logger.configure(handlers=[{"sink": sys.stderr, "level": "DEBUG"}])
    
    # Validate inputs
    if not audio_file.exists():
        console.print(f"[bold red]Error:[/bold red] Audio file not found: {audio_file}")
        raise typer.Exit(1)
    
    # Create output directory if specified
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Show progress
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task("Running comprehensive analysis...", total=None)
        
        try:
            # Create configuration
            config = AudioAnalysisConfig(
                target_language=language,
                output_directory=output_dir,
                preserve_originals=True
            )
            
            # Perform comprehensive analysis
            analyzer = AudioAnalyzer(config)
            results = analyzer.analyze_comprehensive(audio_file)
            
            progress.update(task, description="Comprehensive analysis completed!")
            
        except Exception as e:
            console.print(f"[bold red]Error during analysis:[/bold red] {e}")
            raise typer.Exit(1)
    
    # Display summary
    console.print(f"\n[bold green]Comprehensive Audio Analysis Report[/bold green]")
    console.print(f"[bold]File:[/bold] {audio_file}")
    console.print(f"[bold]Duration:[/bold] {results['metadata'].duration:.2f} seconds")
    console.print(f"[bold]Sample Rate:[/bold] {results['metadata'].sample_rate} Hz")
    console.print(f"[bold]Format:[/bold] {results['metadata'].format.value}")
    
    # Transcription summary
    if 'transcription' in results:
        transcription = results['transcription']
        console.print(f"\n[bold cyan]Transcription:[/bold cyan]")
        console.print(f"  Language: {transcription.language}")
        console.print(f"  Confidence: {transcription.confidence:.3f}")
        console.print(f"  Text: {transcription.full_text[:100]}{'...' if len(transcription.full_text) > 100 else ''}")
    
    # Speaker analysis summary
    if 'speaker_analysis' in results:
        speaker_analysis = results['speaker_analysis']
        console.print(f"\n[bold cyan]Speaker Analysis:[/bold cyan]")
        console.print(f"  Total Speakers: {speaker_analysis.total_speakers}")
        console.print(f"  Speaker Changes: {len(speaker_analysis.speaker_changes)}")
    
    # Enhancement summary
    if 'enhanced_audio' in results:
        enhancement = results['enhanced_audio']
        console.print(f"\n[bold cyan]Audio Enhancement:[/bold cyan]")
        console.print(f"  Techniques Applied: {len(enhancement.enhancement_applied)}")
        console.print(f"  Quality Improvement: {enhancement.quality_improvement.get('dynamic_range_change', 0):+.2f} dB dynamic range")
    
    # Authenticity summary
    if 'authenticity' in results:
        authenticity = results['authenticity']
        status_color = "green" if authenticity.is_authentic else "red"
        status_text = "AUTHENTIC" if authenticity.is_authentic else "MANIPULATED"
        console.print(f"\n[bold cyan]Authenticity Analysis:[/bold cyan]")
        console.print(f"  Status: [{status_color}]{status_text}[/{status_color}]")
        console.print(f"  Confidence: {authenticity.confidence.value.upper()}")
        console.print(f"  Score: {authenticity.authenticity_score:.3f}")
    
    # Save comprehensive report
    if output_dir:
        report_path = output_dir / f"{audio_file.stem}_comprehensive_report.json"
        save_results(results, report_path, "comprehensive")
        console.print(f"\n[green]Comprehensive report saved to:[/green] {report_path}")


@app.command()
def list_languages():
    """List all supported languages for transcription."""
    languages = get_supported_languages()
    
    console.print("\n[bold green]Supported Languages[/bold green]")
    
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Code", style="cyan", width=6)
    table.add_column("Language", style="white")
    
    for code, name in sorted(languages.items()):
        table.add_row(code, name.title())
    
    console.print(table)
    console.print(f"\n[dim]Total: {len(languages)} languages supported[/dim]")


def save_results(results, output_path: Path, analysis_type: str):
    """Save analysis results to JSON file."""
    try:
        # Convert results to JSON-serializable format
        if hasattr(results, 'dict'):
            data = results.dict()
        elif isinstance(results, dict):
            data = results
        else:
            data = {"error": "Unable to serialize results"}
        
        # Add metadata
        data['_metadata'] = {
            'analysis_type': analysis_type,
            'lemkin_audio_version': __version__,
            'export_timestamp': str(Path.cwd())  # Using a simple timestamp alternative
        }
        
        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save to file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)
            
    except Exception as e:
        logger.error(f"Failed to save results: {e}")
        console.print(f"[bold red]Warning:[/bold red] Failed to save results to {output_path}")


if __name__ == "__main__":
    app()