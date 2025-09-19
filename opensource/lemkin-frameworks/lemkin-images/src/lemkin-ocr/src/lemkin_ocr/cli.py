"""
Command Line Interface for Lemkin OCR Suite

This module provides a comprehensive CLI for document digitization operations
including OCR, layout analysis, handwriting recognition, and quality assessment.
Designed for legal document processing workflows.
"""

import click
import json
import sys
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime
import logging

# Rich for beautiful output
try:
    from rich.console import Console
    from rich.table import Table
    from rich.progress import Progress, TaskID
    from rich.panel import Panel
    from rich.text import Text
    from rich.tree import Tree
    from rich import print as rich_print
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    print("Rich not available - using basic output")

from .core import (
    DocumentDigitizer, OCRConfig, OCREngine, ProcessingResult,
    ProcessingStatus, DocumentType
)
from .multilingual_ocr import MultilingualOCR, ocr_document
from .layout_analyzer import LayoutAnalyzer, analyze_document_layout
from .handwriting_processor import HandwritingProcessor, process_handwriting
from .quality_assessor import QualityAssessor, assess_ocr_quality

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Rich console if available
console = Console() if RICH_AVAILABLE else None


def print_output(message: str, style: str = None):
    """Print output with Rich formatting if available"""
    if RICH_AVAILABLE and console:
        console.print(message, style=style)
    else:
        print(message)


def print_json(data: Dict[str, Any], indent: int = 2):
    """Print JSON data with formatting"""
    json_str = json.dumps(data, indent=indent, default=str)
    if RICH_AVAILABLE and console:
        console.print_json(json_str)
    else:
        print(json_str)


def create_progress_bar(description: str = "Processing"):
    """Create a progress bar if Rich is available"""
    if RICH_AVAILABLE:
        return Progress()
    return None


@click.group()
@click.version_option(version="0.1.0", prog_name="lemkin-ocr")
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose output')
@click.option('--config-file', '-c', type=click.Path(exists=True), help='Configuration file path')
@click.pass_context
def cli(ctx, verbose: bool, config_file: Optional[str]):
    """
    Lemkin OCR & Document Digitization Suite
    
    High-accuracy OCR and document processing for legal applications.
    """
    # Ensure context object exists
    ctx.ensure_object(dict)
    
    # Set logging level
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        ctx.obj['verbose'] = True
    else:
        ctx.obj['verbose'] = False
    
    # Load configuration
    if config_file:
        ctx.obj['config'] = load_config_file(config_file)
    else:
        ctx.obj['config'] = OCRConfig()
    
    if verbose:
        print_output("Lemkin OCR Suite initialized", style="bold green")
        if config_file:
            print_output(f"Configuration loaded from: {config_file}", style="dim")


def load_config_file(config_path: str) -> OCRConfig:
    """Load configuration from JSON file"""
    try:
        with open(config_path, 'r') as f:
            config_data = json.load(f)
        return OCRConfig(**config_data)
    except Exception as e:
        print_output(f"Error loading config file: {e}", style="bold red")
        sys.exit(1)


@cli.command()
@click.argument('image_path', type=click.Path(exists=True))
@click.option('--language', '-l', default='en', help='Primary language code (ISO 639-1)')
@click.option('--output', '-o', type=click.Path(), help='Output directory for results')
@click.option('--format', 'output_format', multiple=True, 
              type=click.Choice(['txt', 'json', 'pdf']), default=['txt'],
              help='Output formats (can specify multiple)')
@click.option('--engines', multiple=True, 
              type=click.Choice(['tesseract', 'easyocr', 'paddleocr', 'all']),
              default=['tesseract'], help='OCR engines to use')
@click.option('--quality-threshold', type=float, default=0.8,
              help='Minimum quality threshold (0.0-1.0)')
@click.pass_context
def ocr_document_cmd(ctx, image_path: str, language: str, output: Optional[str],
                     output_format: List[str], engines: List[str], quality_threshold: float):
    """Extract text from images or PDFs using OCR"""
    
    config = ctx.obj['config']
    verbose = ctx.obj.get('verbose', False)
    
    # Update config with CLI options
    config.primary_language = language
    config.ocr_engines = [OCREngine(engine) for engine in engines]
    config.quality_threshold = quality_threshold
    
    # Set output formats
    config.output_txt = 'txt' in output_format
    config.output_json = 'json' in output_format
    config.generate_searchable_pdf = 'pdf' in output_format
    
    path = Path(image_path)
    
    print_output(f"Processing document: {path.name}", style="bold blue")
    if verbose:
        print_output(f"Language: {language}", style="dim")
        print_output(f"Engines: {', '.join(engines)}", style="dim")
    
    try:
        # Create progress bar
        progress = create_progress_bar("OCR Processing")
        
        if progress:
            with progress:
                task = progress.add_task("Extracting text...", total=100)
                progress.update(task, advance=25)
                
                # Perform OCR
                ocr = MultilingualOCR(config)
                progress.update(task, advance=50)
                
                result = ocr.perform_ocr(path, language)
                progress.update(task, advance=100)
        else:
            print_output("Extracting text...")
            ocr = MultilingualOCR(config)
            result = ocr.perform_ocr(path, language)
        
        # Display results
        display_ocr_results(result, verbose)
        
        # Save output files if specified
        if output:
            save_ocr_output(result, Path(output), output_format, path.stem)
            print_output(f"Results saved to: {output}", style="green")
        
    except Exception as e:
        print_output(f"Error processing document: {e}", style="bold red")
        if verbose:
            import traceback
            print_output(traceback.format_exc(), style="dim red")
        sys.exit(1)


@cli.command()
@click.argument('image_path', type=click.Path(exists=True))
@click.option('--output', '-o', type=click.Path(), help='Output directory for results')
@click.option('--detailed', is_flag=True, help='Show detailed layout information')
@click.pass_context
def analyze_layout(ctx, image_path: str, output: Optional[str], detailed: bool):
    """Analyze document layout and structure"""
    
    config = ctx.obj['config']
    verbose = ctx.obj.get('verbose', False)
    
    path = Path(image_path)
    
    print_output(f"Analyzing layout: {path.name}", style="bold blue")
    
    try:
        # Create progress bar
        progress = create_progress_bar("Layout Analysis")
        
        if progress:
            with progress:
                task = progress.add_task("Analyzing structure...", total=100)
                progress.update(task, advance=25)
                
                analyzer = LayoutAnalyzer(config)
                progress.update(task, advance=50)
                
                result = analyzer.analyze_layout(path)
                progress.update(task, advance=100)
        else:
            print_output("Analyzing document structure...")
            analyzer = LayoutAnalyzer(config)
            result = analyzer.analyze_layout(path)
        
        # Display results
        display_layout_results(result, detailed, verbose)
        
        # Save output if specified
        if output:
            save_layout_output(result, Path(output), path.stem)
            print_output(f"Results saved to: {output}", style="green")
        
    except Exception as e:
        print_output(f"Error analyzing layout: {e}", style="bold red")
        if verbose:
            import traceback
            print_output(traceback.format_exc(), style="dim red")
        sys.exit(1)


@cli.command()
@click.argument('image_path', type=click.Path(exists=True))
@click.option('--output', '-o', type=click.Path(), help='Output directory for results')
@click.option('--model', default='trocr', help='Handwriting recognition model')
@click.option('--confidence-threshold', type=float, default=0.7,
              help='Minimum confidence threshold for handwriting')
@click.pass_context
def process_handwriting(ctx, image_path: str, output: Optional[str], 
                       model: str, confidence_threshold: float):
    """Recognize handwritten text in documents"""
    
    config = ctx.obj['config']
    verbose = ctx.obj.get('verbose', False)
    
    # Update config
    config.handwriting_model = model
    config.handwriting_confidence_threshold = confidence_threshold
    
    path = Path(image_path)
    
    print_output(f"Processing handwriting: {path.name}", style="bold blue")
    if verbose:
        print_output(f"Model: {model}", style="dim")
        print_output(f"Confidence threshold: {confidence_threshold}", style="dim")
    
    try:
        # Create progress bar
        progress = create_progress_bar("Handwriting Recognition")
        
        if progress:
            with progress:
                task = progress.add_task("Recognizing handwriting...", total=100)
                progress.update(task, advance=25)
                
                processor = HandwritingProcessor(config)
                progress.update(task, advance=50)
                
                result = processor.process_handwriting(path)
                progress.update(task, advance=100)
        else:
            print_output("Recognizing handwritten text...")
            processor = HandwritingProcessor(config)
            result = processor.process_handwriting(path)
        
        # Display results
        display_handwriting_results(result, verbose)
        
        # Save output if specified
        if output:
            save_handwriting_output(result, Path(output), path.stem)
            print_output(f"Results saved to: {output}", style="green")
        
    except Exception as e:
        print_output(f"Error processing handwriting: {e}", style="bold red")
        if verbose:
            import traceback
            print_output(traceback.format_exc(), style="dim red")
        sys.exit(1)


@cli.command()
@click.argument('result_path', type=click.Path(exists=True))
@click.option('--detailed', is_flag=True, help='Show detailed quality metrics')
@click.pass_context
def assess_quality(ctx, result_path: str, detailed: bool):
    """Assess OCR quality and provide improvement recommendations"""
    
    config = ctx.obj['config']
    verbose = ctx.obj.get('verbose', False)
    
    path = Path(result_path)
    
    print_output(f"Assessing quality: {path.name}", style="bold blue")
    
    try:
        # Load processing result
        with open(path, 'r') as f:
            result_data = json.load(f)
        
        # Convert to ProcessingResult object
        processing_result = ProcessingResult(**result_data)
        
        # Assess quality
        progress = create_progress_bar("Quality Assessment")
        
        if progress:
            with progress:
                task = progress.add_task("Analyzing quality...", total=100)
                progress.update(task, advance=25)
                
                assessor = QualityAssessor(config)
                progress.update(task, advance=50)
                
                assessment = assessor.assess_quality(processing_result)
                progress.update(task, advance=100)
        else:
            print_output("Analyzing OCR quality...")
            assessor = QualityAssessor(config)
            assessment = assessor.assess_quality(processing_result)
        
        # Display results
        display_quality_assessment(assessment, detailed, verbose)
        
    except Exception as e:
        print_output(f"Error assessing quality: {e}", style="bold red")
        if verbose:
            import traceback
            print_output(traceback.format_exc(), style="dim red")
        sys.exit(1)


@cli.command()
@click.argument('input_dir', type=click.Path(exists=True, file_okay=False))
@click.option('--output', '-o', type=click.Path(), required=True,
              help='Output directory for batch results')
@click.option('--language', '-l', default='en', help='Primary language code')
@click.option('--pattern', default='*.{jpg,jpeg,png,tiff,pdf}',
              help='File pattern to match')
@click.option('--workers', type=int, default=4, help='Number of parallel workers')
@click.pass_context
def batch_process(ctx, input_dir: str, output: str, language: str, 
                 pattern: str, workers: int):
    """Process multiple documents in batch"""
    
    config = ctx.obj['config']
    verbose = ctx.obj.get('verbose', False)
    
    # Update config
    config.primary_language = language
    config.max_workers = workers
    config.parallel_processing = True
    
    input_path = Path(input_dir)
    output_path = Path(output)
    
    # Find matching files
    import glob
    file_patterns = pattern.split(',')
    all_files = []
    for file_pattern in file_patterns:
        matches = list(input_path.glob(file_pattern.strip()))
        all_files.extend(matches)
    
    if not all_files:
        print_output(f"No files found matching pattern: {pattern}", style="yellow")
        return
    
    print_output(f"Found {len(all_files)} files to process", style="bold blue")
    if verbose:
        for file_path in all_files[:10]:  # Show first 10
            print_output(f"  - {file_path.name}", style="dim")
        if len(all_files) > 10:
            print_output(f"  ... and {len(all_files) - 10} more", style="dim")
    
    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)
    
    try:
        # Initialize digitizer
        digitizer = DocumentDigitizer(config)
        
        # Process files with progress bar
        progress = create_progress_bar("Batch Processing")
        
        if progress:
            with progress:
                task = progress.add_task("Processing documents...", total=len(all_files))
                
                results = []
                for file_path in all_files:
                    try:
                        result = digitizer.process_document(
                            image_path=file_path,
                            language=language,
                            output_dir=output_path / file_path.stem
                        )
                        results.append(result)
                        
                        # Save individual result
                        result_file = output_path / f"{file_path.stem}_result.json"
                        with open(result_file, 'w') as f:
                            f.write(result.json(indent=2))
                        
                        progress.update(task, advance=1)
                        
                    except Exception as e:
                        print_output(f"Failed to process {file_path.name}: {e}", style="red")
                        progress.update(task, advance=1)
                        continue
        else:
            print_output("Processing documents...")
            results = digitizer.batch_process_documents(
                input_paths=all_files,
                output_dir=output_path,
                language=language
            )
        
        # Generate batch summary
        generate_batch_summary(results, output_path, verbose)
        
        print_output(f"Batch processing completed. Results saved to: {output_path}", style="green")
        
    except Exception as e:
        print_output(f"Error in batch processing: {e}", style="bold red")
        if verbose:
            import traceback
            print_output(traceback.format_exc(), style="dim red")
        sys.exit(1)


def display_ocr_results(result, verbose: bool = False):
    """Display OCR results in a formatted way"""
    if RICH_AVAILABLE and console:
        # Create a panel with results
        panel_content = f"""
[bold]Text Extracted:[/bold]
{result.text[:500] + '...' if len(result.text) > 500 else result.text}

[bold]Statistics:[/bold]
• Words: {result.word_count}
• Characters: {result.character_count}
• Confidence: {result.overall_confidence:.1%}
• Processing time: {result.processing_duration_seconds:.2f}s
• Primary language: {result.primary_language or 'Unknown'}
"""
        
        if verbose and result.engines_used:
            panel_content += f"\n[bold]Engines used:[/bold] {', '.join(result.engines_used)}"
        
        console.print(Panel(panel_content, title="OCR Results", border_style="blue"))
        
        if verbose and result.engine_results:
            # Show engine comparison
            table = Table(title="Engine Comparison", show_header=True)
            table.add_column("Engine", style="cyan")
            table.add_column("Confidence", style="green")
            table.add_column("Word Count", style="yellow")
            
            for engine, data in result.engine_results.items():
                table.add_row(
                    engine,
                    f"{data.get('confidence', 0):.1%}",
                    str(data.get('word_count', 0))
                )
            
            console.print(table)
    else:
        print(f"\n=== OCR Results ===")
        print(f"Text: {result.text[:200] + '...' if len(result.text) > 200 else result.text}")
        print(f"Words: {result.word_count}")
        print(f"Confidence: {result.overall_confidence:.1%}")
        print(f"Processing time: {result.processing_duration_seconds:.2f}s")


def display_layout_results(result, detailed: bool = False, verbose: bool = False):
    """Display layout analysis results"""
    structure = result.structure
    
    if RICH_AVAILABLE and console:
        panel_content = f"""
[bold]Document Structure:[/bold]
• Document type: {structure.document_type}
• Text regions: {result.total_text_regions}
• Tables: {result.total_table_regions}
• Images: {result.total_image_regions}
• Handwritten regions: {result.total_handwritten_regions}

[bold]Quality Metrics:[/bold]
• Layout confidence: {result.layout_confidence:.1%}
• Text line quality: {result.text_line_quality:.1%}
• Multi-column: {'Yes' if result.is_multi_column else 'No'}
• Complex layout: {'Yes' if result.has_complex_layout else 'No'}
"""
        
        console.print(Panel(panel_content, title="Layout Analysis", border_style="green"))
        
        if detailed and structure.text_regions:
            # Show text regions
            table = Table(title="Text Regions", show_header=True)
            table.add_column("ID", style="cyan")
            table.add_column("Type", style="yellow")
            table.add_column("Position", style="blue")
            table.add_column("Size", style="green")
            
            for i, region in enumerate(structure.text_regions[:10]):  # First 10
                bbox = region.bounding_box
                table.add_row(
                    str(i + 1),
                    region.region_type,
                    f"({bbox.x}, {bbox.y})",
                    f"{bbox.width}×{bbox.height}"
                )
            
            console.print(table)
    else:
        print(f"\n=== Layout Analysis ===")
        print(f"Document type: {structure.document_type}")
        print(f"Text regions: {result.total_text_regions}")
        print(f"Tables: {result.total_table_regions}")
        print(f"Images: {result.total_image_regions}")
        print(f"Layout confidence: {result.layout_confidence:.1%}")


def display_handwriting_results(result, verbose: bool = False):
    """Display handwriting recognition results"""
    if RICH_AVAILABLE and console:
        panel_content = f"""
[bold]Handwritten Text Found:[/bold]
{result.total_handwritten_text[:300] + '...' if len(result.total_handwritten_text) > 300 else result.total_handwritten_text}

[bold]Statistics:[/bold]
• Regions detected: {result.total_regions}
• Average confidence: {result.average_confidence:.1%}
• Legibility score: {result.legibility_score:.1%}
• Manual review needed: {'Yes' if result.requires_manual_review else 'No'}
• Dominant style: {result.dominant_writing_style or 'Unknown'}
"""
        
        console.print(Panel(panel_content, title="Handwriting Recognition", border_style="purple"))
        
        if verbose and result.handwritten_regions:
            table = Table(title="Handwritten Regions", show_header=True)
            table.add_column("Region", style="cyan")
            table.add_column("Text", style="white")
            table.add_column("Confidence", style="green")
            table.add_column("Style", style="yellow")
            
            for i, region in enumerate(result.handwritten_regions[:5]):  # First 5
                table.add_row(
                    str(i + 1),
                    region.text[:50] + '...' if len(region.text) > 50 else region.text,
                    f"{region.confidence:.1%}",
                    region.writing_style or 'Unknown'
                )
            
            console.print(table)
    else:
        print(f"\n=== Handwriting Recognition ===")
        print(f"Text: {result.total_handwritten_text[:200] + '...' if len(result.total_handwritten_text) > 200 else result.total_handwritten_text}")
        print(f"Regions: {result.total_regions}")
        print(f"Confidence: {result.average_confidence:.1%}")
        print(f"Manual review: {'Yes' if result.requires_manual_review else 'No'}")


def display_quality_assessment(assessment, detailed: bool = False, verbose: bool = False):
    """Display quality assessment results"""
    if RICH_AVAILABLE and console:
        # Overall quality panel
        panel_content = f"""
[bold]Overall Quality Score: {assessment.overall_quality_score:.1%}[/bold]

[bold]Component Scores:[/bold]
• Text accuracy: {assessment.text_accuracy_score:.1%}
• Layout accuracy: {assessment.layout_accuracy_score:.1%}
• Confidence reliability: {assessment.confidence_reliability_score:.1%}

[bold]Image Quality:[/bold]
• Resolution: {'✓' if assessment.image_resolution_adequate else '✗'}
• Contrast: {'✓' if assessment.image_contrast_adequate else '✗'}
• Skew: {'✓' if assessment.image_skew_acceptable else '✗'}
• Noise level: {assessment.image_noise_level}

[bold]Legal Compliance:[/bold]
• Meets standards: {'✓' if assessment.meets_legal_standards else '✗'}
• Admissible quality: {'✓' if assessment.admissible_quality else '✗'}
• Expert review needed: {'Yes' if assessment.requires_expert_validation else 'No'}
"""
        
        # Color based on quality
        color = "green" if assessment.overall_quality_score > 0.8 else "yellow" if assessment.overall_quality_score > 0.6 else "red"
        
        console.print(Panel(panel_content, title="Quality Assessment", border_style=color))
        
        # Issues and suggestions
        if assessment.quality_issues:
            issues_text = "\n".join(f"• {issue}" for issue in assessment.quality_issues)
            console.print(Panel(issues_text, title="Quality Issues", border_style="red"))
        
        if assessment.improvement_suggestions:
            suggestions_text = "\n".join(f"• {suggestion}" for suggestion in assessment.improvement_suggestions)
            console.print(Panel(suggestions_text, title="Improvement Suggestions", border_style="blue"))
    else:
        print(f"\n=== Quality Assessment ===")
        print(f"Overall quality: {assessment.overall_quality_score:.1%}")
        print(f"Text accuracy: {assessment.text_accuracy_score:.1%}")
        print(f"Legal standards: {'Met' if assessment.meets_legal_standards else 'Not met'}")
        
        if assessment.quality_issues:
            print(f"\nIssues: {', '.join(assessment.quality_issues)}")
        
        if assessment.improvement_suggestions:
            print(f"\nSuggestions: {', '.join(assessment.improvement_suggestions)}")


def save_ocr_output(result, output_dir: Path, formats: List[str], base_name: str):
    """Save OCR output in specified formats"""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if 'txt' in formats:
        txt_file = output_dir / f"{base_name}.txt"
        with open(txt_file, 'w', encoding='utf-8') as f:
            f.write(result.text)
    
    if 'json' in formats:
        json_file = output_dir / f"{base_name}.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            f.write(result.json(indent=2))


def save_layout_output(result, output_dir: Path, base_name: str):
    """Save layout analysis output"""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    json_file = output_dir / f"{base_name}_layout.json"
    with open(json_file, 'w', encoding='utf-8') as f:
        f.write(result.json(indent=2))


def save_handwriting_output(result, output_dir: Path, base_name: str):
    """Save handwriting recognition output"""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save text
    txt_file = output_dir / f"{base_name}_handwriting.txt"
    with open(txt_file, 'w', encoding='utf-8') as f:
        f.write(result.total_handwritten_text)
    
    # Save detailed results
    json_file = output_dir / f"{base_name}_handwriting.json"
    with open(json_file, 'w', encoding='utf-8') as f:
        f.write(result.json(indent=2))


def generate_batch_summary(results: List[ProcessingResult], output_dir: Path, verbose: bool = False):
    """Generate summary report for batch processing"""
    summary = {
        'total_documents': len(results),
        'successful': sum(1 for r in results if r.success),
        'failed': sum(1 for r in results if not r.success),
        'average_processing_time': sum(r.total_processing_time_seconds for r in results) / len(results) if results else 0,
        'total_processing_time': sum(r.total_processing_time_seconds for r in results),
        'average_confidence': 0.0,
        'document_types': {},
        'quality_distribution': {'excellent': 0, 'good': 0, 'fair': 0, 'poor': 0}
    }
    
    # Calculate averages and distributions
    successful_results = [r for r in results if r.success and r.ocr_result]
    
    if successful_results:
        summary['average_confidence'] = sum(r.ocr_result.overall_confidence for r in successful_results) / len(successful_results)
        
        # Document types
        for result in successful_results:
            if result.layout_analysis:
                doc_type = result.layout_analysis.structure.document_type
                summary['document_types'][doc_type] = summary['document_types'].get(doc_type, 0) + 1
        
        # Quality distribution
        for result in successful_results:
            confidence = result.ocr_result.overall_confidence
            if confidence >= 0.95:
                summary['quality_distribution']['excellent'] += 1
            elif confidence >= 0.85:
                summary['quality_distribution']['good'] += 1
            elif confidence >= 0.70:
                summary['quality_distribution']['fair'] += 1
            else:
                summary['quality_distribution']['poor'] += 1
    
    # Save summary
    summary_file = output_dir / 'batch_summary.json'
    with open(summary_file, 'w') as f:
        json.dumps(summary, f, indent=2, default=str)
    
    # Display summary
    if RICH_AVAILABLE and console:
        panel_content = f"""
[bold]Batch Processing Summary[/bold]

[bold]Documents Processed:[/bold] {summary['total_documents']}
• Successful: {summary['successful']} ({summary['successful']/summary['total_documents']*100:.1f}%)
• Failed: {summary['failed']} ({summary['failed']/summary['total_documents']*100:.1f}%)

[bold]Performance:[/bold]
• Average confidence: {summary['average_confidence']:.1%}
• Average processing time: {summary['average_processing_time']:.2f}s per document
• Total processing time: {summary['total_processing_time']:.2f}s

[bold]Quality Distribution:[/bold]
• Excellent (95%+): {summary['quality_distribution']['excellent']}
• Good (85-94%): {summary['quality_distribution']['good']}
• Fair (70-84%): {summary['quality_distribution']['fair']}
• Poor (<70%): {summary['quality_distribution']['poor']}
"""
        
        console.print(Panel(panel_content, title="Batch Summary", border_style="blue"))
    else:
        print(f"\n=== Batch Processing Summary ===")
        print(f"Total documents: {summary['total_documents']}")
        print(f"Successful: {summary['successful']}")
        print(f"Failed: {summary['failed']}")
        print(f"Average confidence: {summary['average_confidence']:.1%}")


# Additional utility commands
@cli.command()
@click.option('--engines', is_flag=True, help='List available OCR engines')
@click.option('--languages', is_flag=True, help='List supported languages')
@click.option('--models', is_flag=True, help='List available handwriting models')
def list_capabilities(engines: bool, languages: bool, models: bool):
    """List available capabilities and configurations"""
    
    if engines:
        print_output("Available OCR Engines:", style="bold blue")
        available_engines = []
        
        try:
            import pytesseract
            available_engines.append("✓ Tesseract")
        except ImportError:
            available_engines.append("✗ Tesseract (not installed)")
        
        try:
            import easyocr
            available_engines.append("✓ EasyOCR")
        except ImportError:
            available_engines.append("✗ EasyOCR (not installed)")
        
        try:
            import paddleocr
            available_engines.append("✓ PaddleOCR")
        except ImportError:
            available_engines.append("✗ PaddleOCR (not installed)")
        
        for engine in available_engines:
            print_output(f"  {engine}")
    
    if languages:
        print_output("Supported Languages (selection):", style="bold blue")
        common_languages = [
            "en (English)", "es (Spanish)", "fr (French)", "de (German)",
            "it (Italian)", "pt (Portuguese)", "ru (Russian)", "ja (Japanese)",
            "ko (Korean)", "zh (Chinese)", "ar (Arabic)", "hi (Hindi)"
        ]
        for lang in common_languages:
            print_output(f"  {lang}")
        print_output("  ... and 100+ more languages supported")
    
    if models:
        print_output("Available Handwriting Models:", style="bold blue")
        models_list = [
            "trocr (default) - Microsoft TrOCR",
            "microsoft/trocr-base-handwritten",
            "microsoft/trocr-large-handwritten"
        ]
        for model in models_list:
            print_output(f"  {model}")


@cli.command()
@click.option('--output', '-o', type=click.Path(), default='lemkin_ocr_config.json',
              help='Output configuration file path')
def generate_config(output: str):
    """Generate a sample configuration file"""
    
    config = OCRConfig()
    
    config_dict = config.dict()
    config_dict['_description'] = "Lemkin OCR Configuration File"
    config_dict['_version'] = "0.1.0"
    
    with open(output, 'w') as f:
        json.dump(config_dict, f, indent=2, default=str)
    
    print_output(f"Configuration file generated: {output}", style="green")
    print_output("Edit this file to customize OCR settings", style="dim")


if __name__ == '__main__':
    cli()