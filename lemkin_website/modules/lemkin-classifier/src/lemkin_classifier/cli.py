"""
Command Line Interface for the Lemkin Legal Document Classifier.

This module provides a comprehensive CLI with commands for document classification,
batch processing, model training, evaluation, and taxonomy management.
"""

import json
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any
import sys

import typer
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.panel import Panel
from rich.json import JSON
import pandas as pd

from .core import (
    DocumentClassifier,
    DocumentContent,
    ClassificationConfig,
    ClassificationResult
)
from .legal_taxonomy import (
    get_supported_categories,
    get_category_hierarchy,
    validate_category,
    DocumentType,
    LegalDomain
)
from .confidence_scorer import ConfidenceScorer, ScoreThresholds
from .batch_processor import (
    BatchProcessor,
    DocumentBatch,
    ProcessingConfig,
    ProcessingMode
)

# Initialize Typer app and Rich console
app = typer.Typer(
    name="lemkin-classifier",
    help="Legal Document Classifier for evidence triage and case organization",
    add_completion=False
)

console = Console()

# Global configuration
DEFAULT_MODEL = "distilbert-base-uncased"
DEFAULT_OUTPUT_DIR = "./output"


def setup_logging(verbose: bool = False, quiet: bool = False) -> None:
    """Setup logging configuration"""
    if quiet:
        logging.basicConfig(level=logging.ERROR)
    elif verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)


@app.command()
def classify_document(
    input_path: Path = typer.Argument(..., help="Path to document file"),
    model_path: Optional[str] = typer.Option(None, "--model", "-m", help="Path to fine-tuned model"),
    confidence_threshold: float = typer.Option(0.7, "--threshold", "-t", help="Confidence threshold"),
    output_file: Optional[Path] = typer.Option(None, "--output", "-o", help="Output file for results"),
    format_output: str = typer.Option("json", "--format", "-f", help="Output format: json, table, csv"),
    include_confidence: bool = typer.Option(True, "--confidence/--no-confidence", help="Include confidence assessment"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Quiet mode")
) -> None:
    """
    Classify a single legal document.
    
    This command processes a single document file and returns classification results
    including document type, legal domain, and confidence assessment.
    """
    setup_logging(verbose, quiet)
    
    try:
        # Validate input file
        if not input_path.exists():
            console.print(f"[red]Error: File not found: {input_path}[/red]")
            raise typer.Exit(1)
        
        # Initialize classifier
        config = ClassificationConfig(
            model_name=DEFAULT_MODEL,
            model_path=model_path,
            confidence_threshold=confidence_threshold
        )
        
        with console.status("[bold blue]Initializing classifier..."):
            classifier = DocumentClassifier(config)
        
        # Initialize confidence scorer if requested
        confidence_scorer = None
        if include_confidence:
            confidence_scorer = ConfidenceScorer()
        
        # Classify document
        console.print(f"[blue]Classifying document:[/blue] {input_path}")
        
        with console.status("[bold blue]Processing document..."):
            result = classifier.classify_file(input_path)
        
        # Display results
        if format_output == "table":
            _display_classification_table(result)
        elif format_output == "csv":
            _display_classification_csv(result, output_file)
        else:  # json
            _display_classification_json(result, output_file)
        
        # Save to file if requested
        if output_file and format_output == "json":
            with open(output_file, 'w') as f:
                json.dump(result.dict(), f, indent=2, default=str)
            console.print(f"[green]Results saved to:[/green] {output_file}")
        
        console.print("[green]Classification completed successfully![/green]")
        
    except Exception as e:
        console.print(f"[red]Error during classification: {e}[/red]")
        if verbose:
            console.print_exception()
        raise typer.Exit(1)


@app.command()
def batch_classify(
    input_dir: Path = typer.Argument(..., help="Directory containing documents to classify"),
    output_dir: Path = typer.Option(Path(DEFAULT_OUTPUT_DIR), "--output", "-o", help="Output directory"),
    model_path: Optional[str] = typer.Option(None, "--model", "-m", help="Path to fine-tuned model"),
    file_patterns: Optional[List[str]] = typer.Option(None, "--patterns", "-p", help="File patterns to match"),
    max_workers: int = typer.Option(4, "--workers", "-w", help="Maximum number of worker threads"),
    batch_size: int = typer.Option(32, "--batch-size", "-b", help="Processing batch size"),
    processing_mode: ProcessingMode = typer.Option(ProcessingMode.THREADED, "--mode", help="Processing mode"),
    confidence_threshold: float = typer.Option(0.7, "--threshold", "-t", help="Confidence threshold"),
    output_format: str = typer.Option("json", "--format", "-f", help="Output format: json, csv"),
    recursive: bool = typer.Option(True, "--recursive/--no-recursive", help="Search recursively"),
    continue_on_error: bool = typer.Option(True, "--continue/--fail-fast", help="Continue on individual errors"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Quiet mode")
) -> None:
    """
    Process multiple legal documents in batch mode.
    
    This command efficiently processes large numbers of documents with
    progress tracking, error handling, and parallel processing capabilities.
    """
    setup_logging(verbose, quiet)
    
    try:
        # Validate input directory
        if not input_dir.exists():
            console.print(f"[red]Error: Directory not found: {input_dir}[/red]")
            raise typer.Exit(1)
        
        # Initialize classifier
        config = ClassificationConfig(
            model_name=DEFAULT_MODEL,
            model_path=model_path,
            confidence_threshold=confidence_threshold
        )
        
        processing_config = ProcessingConfig(
            max_workers=max_workers,
            batch_size=batch_size,
            processing_mode=processing_mode,
            continue_on_error=continue_on_error,
            output_format=output_format,
            enable_progress_bar=not quiet
        )
        
        console.print("[blue]Initializing batch processor...[/blue]")
        classifier = DocumentClassifier(config)
        confidence_scorer = ConfidenceScorer()
        batch_processor = BatchProcessor(classifier, confidence_scorer, processing_config)
        
        # Create document batch
        console.print(f"[blue]Scanning directory:[/blue] {input_dir}")
        patterns = file_patterns or ['*.pdf', '*.docx', '*.txt']
        batch = batch_processor.create_batch_from_directory(
            input_dir,
            batch_name=f"Batch: {input_dir.name}",
            file_patterns=patterns,
            recursive=recursive
        )
        
        console.print(f"[green]Found {len(batch.documents)} documents to process[/green]")
        
        # Process batch
        def progress_callback(progress: float, message: str):
            if not quiet:
                console.print(f"\r{message} ({progress:.1%})", end="")
        
        console.print("[blue]Starting batch processing...[/blue]")
        result = batch_processor.process_batch(
            batch,
            output_dir=output_dir,
            progress_callback=progress_callback if not quiet else None
        )
        
        # Display summary
        _display_batch_summary(result)
        
        console.print(f"[green]Batch processing completed![/green]")
        console.print(f"[blue]Results saved to:[/blue] {output_dir}")
        
    except Exception as e:
        console.print(f"[red]Error during batch processing: {e}[/red]")
        if verbose:
            console.print_exception()
        raise typer.Exit(1)


@app.command()
def train_model(
    training_data: Path = typer.Argument(..., help="Path to training data file (CSV/JSON)"),
    output_dir: Path = typer.Argument(..., help="Directory to save trained model"),
    base_model: str = typer.Option(DEFAULT_MODEL, "--base-model", help="Base model for fine-tuning"),
    validation_split: float = typer.Option(0.2, "--validation-split", help="Validation data split ratio"),
    batch_size: int = typer.Option(16, "--batch-size", help="Training batch size"),
    max_length: int = typer.Option(512, "--max-length", help="Maximum sequence length"),
    epochs: int = typer.Option(3, "--epochs", help="Number of training epochs"),
    learning_rate: float = typer.Option(5e-5, "--learning-rate", help="Learning rate"),
    text_column: str = typer.Option("text", "--text-col", help="Text column name in training data"),
    label_column: str = typer.Option("label", "--label-col", help="Label column name in training data"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Quiet mode")
) -> None:
    """
    Fine-tune a BERT model on legal document training data.
    
    This command trains a specialized model for legal document classification
    using your organization's specific document types and categories.
    """
    setup_logging(verbose, quiet)
    
    try:
        # Validate input file
        if not training_data.exists():
            console.print(f"[red]Error: Training data file not found: {training_data}[/red]")
            raise typer.Exit(1)
        
        # Load training data
        console.print(f"[blue]Loading training data:[/blue] {training_data}")
        
        if training_data.suffix == '.csv':
            df = pd.read_csv(training_data)
        elif training_data.suffix == '.json':
            df = pd.read_json(training_data)
        else:
            console.print("[red]Error: Training data must be CSV or JSON format[/red]")
            raise typer.Exit(1)
        
        # Validate columns
        if text_column not in df.columns:
            console.print(f"[red]Error: Text column '{text_column}' not found in data[/red]")
            raise typer.Exit(1)
        
        if label_column not in df.columns:
            console.print(f"[red]Error: Label column '{label_column}' not found in data[/red]")
            raise typer.Exit(1)
        
        # Prepare training data
        training_pairs = list(zip(df[text_column].astype(str), df[label_column].astype(str)))
        
        console.print(f"[green]Loaded {len(training_pairs)} training examples[/green]")
        console.print(f"[blue]Unique labels:[/blue] {sorted(set(df[label_column]))}")
        
        # Initialize classifier
        config = ClassificationConfig(
            model_name=base_model,
            max_length=max_length,
            batch_size=batch_size
        )
        
        console.print("[blue]Initializing classifier for training...[/blue]")
        classifier = DocumentClassifier(config)
        
        # Train model
        console.print("[blue]Starting model training...[/blue]")
        with console.status("[bold blue]Training model..."):
            metrics = classifier.train_model(
                training_data=training_pairs,
                validation_split=validation_split,
                output_dir=str(output_dir)
            )
        
        # Display training results
        _display_training_metrics(metrics)
        
        console.print(f"[green]Model training completed![/green]")
        console.print(f"[blue]Model saved to:[/blue] {output_dir}")
        
    except Exception as e:
        console.print(f"[red]Error during model training: {e}[/red]")
        if verbose:
            console.print_exception()
        raise typer.Exit(1)


@app.command()
def evaluate_model(
    test_data: Path = typer.Argument(..., help="Path to test data file (CSV/JSON)"),
    model_path: Optional[str] = typer.Option(None, "--model", "-m", help="Path to model to evaluate"),
    text_column: str = typer.Option("text", "--text-col", help="Text column name in test data"),
    label_column: str = typer.Option("label", "--label-col", help="Label column name in test data"),
    output_file: Optional[Path] = typer.Option(None, "--output", "-o", help="Save evaluation results"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Quiet mode")
) -> None:
    """
    Evaluate model performance on test data.
    
    This command assesses classification accuracy, precision, recall,
    and other performance metrics on a held-out test dataset.
    """
    setup_logging(verbose, quiet)
    
    try:
        # Validate input file
        if not test_data.exists():
            console.print(f"[red]Error: Test data file not found: {test_data}[/red]")
            raise typer.Exit(1)
        
        # Load test data
        console.print(f"[blue]Loading test data:[/blue] {test_data}")
        
        if test_data.suffix == '.csv':
            df = pd.read_csv(test_data)
        elif test_data.suffix == '.json':
            df = pd.read_json(test_data)
        else:
            console.print("[red]Error: Test data must be CSV or JSON format[/red]")
            raise typer.Exit(1)
        
        # Validate columns
        if text_column not in df.columns or label_column not in df.columns:
            console.print("[red]Error: Required columns not found in test data[/red]")
            raise typer.Exit(1)
        
        # Prepare test data
        test_pairs = list(zip(df[text_column].astype(str), df[label_column].astype(str)))
        
        console.print(f"[green]Loaded {len(test_pairs)} test examples[/green]")
        
        # Initialize classifier
        config = ClassificationConfig(
            model_name=DEFAULT_MODEL,
            model_path=model_path
        )
        
        console.print("[blue]Initializing classifier...[/blue]")
        classifier = DocumentClassifier(config)
        
        # Evaluate model
        console.print("[blue]Evaluating model performance...[/blue]")
        with console.status("[bold blue]Running evaluation..."):
            metrics = classifier.evaluate_model(test_pairs)
        
        # Display results
        _display_evaluation_metrics(metrics)
        
        # Save results if requested
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(metrics.dict(), f, indent=2, default=str)
            console.print(f"[green]Evaluation results saved to:[/green] {output_file}")
        
        console.print("[green]Model evaluation completed![/green]")
        
    except Exception as e:
        console.print(f"[red]Error during model evaluation: {e}[/red]")
        if verbose:
            console.print_exception()
        raise typer.Exit(1)


@app.command()
def update_taxonomy(
    action: str = typer.Argument(..., help="Action: list, add, remove, validate"),
    category: Optional[str] = typer.Option(None, "--category", help="Document category"),
    domain: Optional[str] = typer.Option(None, "--domain", help="Legal domain"),
    output_file: Optional[Path] = typer.Option(None, "--output", "-o", help="Save taxonomy to file"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output")
) -> None:
    """
    Manage legal document taxonomy and categories.
    
    This command allows you to view, validate, and modify the legal document
    classification taxonomy used by the system.
    """
    setup_logging(verbose, False)
    
    try:
        if action == "list":
            _display_taxonomy()
        
        elif action == "validate":
            if not category:
                console.print("[red]Error: Category required for validation[/red]")
                raise typer.Exit(1)
            
            is_valid = validate_category(category, domain)
            if is_valid:
                console.print(f"[green]✓ Valid category:[/green] {category}")
                if domain:
                    console.print(f"[green]✓ Valid domain:[/green] {domain}")
            else:
                console.print(f"[red]✗ Invalid category:[/red] {category}")
                if domain:
                    console.print(f"[red]✗ Invalid domain combination[/red]")
        
        elif action == "add":
            console.print("[yellow]Note: Adding categories requires code modification[/yellow]")
            console.print("See legal_taxonomy.py for category definitions")
        
        elif action == "remove":
            console.print("[yellow]Note: Removing categories requires code modification[/yellow]")
            console.print("See legal_taxonomy.py for category definitions")
        
        else:
            console.print(f"[red]Error: Unknown action '{action}'[/red]")
            console.print("Available actions: list, add, remove, validate")
            raise typer.Exit(1)
        
        # Save taxonomy if requested
        if output_file and action == "list":
            hierarchy = get_category_hierarchy()
            with open(output_file, 'w') as f:
                json.dump(hierarchy.dict(), f, indent=2, default=str)
            console.print(f"[green]Taxonomy saved to:[/green] {output_file}")
        
    except Exception as e:
        console.print(f"[red]Error managing taxonomy: {e}[/red]")
        if verbose:
            console.print_exception()
        raise typer.Exit(1)


@app.command()
def info() -> None:
    """Display system information and supported categories."""
    
    # System information
    console.print(Panel(
        "[bold blue]Lemkin Legal Document Classifier[/bold blue]\n\n"
        "Advanced AI system for automated classification of legal documents\n"
        "Built with BERT transformers and fine-tuned for legal domains",
        title="System Information",
        border_style="blue"
    ))
    
    # Supported categories
    categories = get_supported_categories()
    console.print(f"\n[bold]Supported Document Types:[/bold] {len(categories)}")
    
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Document Type", style="cyan")
    table.add_column("Description", style="white")
    
    # Add some example descriptions
    descriptions = {
        "witness_statement": "Witness testimonies and statements",
        "police_report": "Law enforcement incident reports",
        "medical_record": "Medical documentation and health records",
        "court_filing": "Legal documents filed with courts",
        "government_document": "Official government communications",
        "military_report": "Military and defense-related documents",
        "email": "Email communications and correspondence",
        "expert_testimony": "Expert witness reports and analysis",
        "forensic_report": "Forensic analysis and laboratory reports",
        "financial_record": "Financial documents and transaction records"
    }
    
    for category in categories:
        desc = descriptions.get(category.value, "Legal document classification")
        table.add_row(category.value.replace('_', ' ').title(), desc)
    
    console.print(table)
    
    # Legal domains
    console.print(f"\n[bold]Legal Domains:[/bold]")
    domains = [
        "Criminal Law", "Civil Rights", "International Humanitarian Law",
        "Human Rights Law", "Administrative Law", "Constitutional Law",
        "Corporate Law", "Family Law", "Immigration Law", "Environmental Law"
    ]
    
    console.print(", ".join(domains))


def _display_classification_table(result: ClassificationResult) -> None:
    """Display classification result as a formatted table"""
    
    table = Table(title="Document Classification Result", show_header=True, header_style="bold magenta")
    table.add_column("Property", style="cyan", width=20)
    table.add_column("Value", style="white")
    
    table.add_row("Document Type", result.classification.document_type.value.replace('_', ' ').title())
    table.add_row("Legal Domain", result.classification.legal_domain.value.replace('_', ' ').title())
    table.add_row("Confidence Score", f"{result.classification.confidence_score:.3f}")
    table.add_row("Urgency Level", result.urgency_level.title())
    table.add_row("Sensitivity Level", result.sensitivity_level.title())
    table.add_row("Requires Review", "Yes" if result.requires_review else "No")
    table.add_row("Processing Time", f"{result.processing_time:.3f}s")
    table.add_row("Document Length", f"{result.document_content.length:,} chars")
    
    if result.review_reasons:
        table.add_row("Review Reasons", "\n".join(result.review_reasons))
    
    if result.recommended_actions:
        table.add_row("Recommendations", "\n".join(result.recommended_actions))
    
    console.print(table)


def _display_classification_json(result: ClassificationResult, output_file: Optional[Path] = None) -> None:
    """Display classification result as JSON"""
    
    result_dict = result.dict()
    json_obj = JSON(json.dumps(result_dict, indent=2, default=str))
    
    console.print("\n[bold]Classification Result:[/bold]")
    console.print(json_obj)


def _display_classification_csv(result: ClassificationResult, output_file: Optional[Path] = None) -> None:
    """Display classification result in CSV format"""
    
    # Flatten result for CSV
    row = {
        'file_path': result.document_content.file_path,
        'document_type': result.classification.document_type.value,
        'legal_domain': result.classification.legal_domain.value,
        'confidence_score': result.classification.confidence_score,
        'urgency_level': result.urgency_level,
        'sensitivity_level': result.sensitivity_level,
        'requires_review': result.requires_review,
        'review_reasons': '; '.join(result.review_reasons),
        'processing_time': result.processing_time
    }
    
    if output_file:
        df = pd.DataFrame([row])
        df.to_csv(output_file, index=False)
        console.print(f"[green]CSV results saved to:[/green] {output_file}")
    else:
        # Display as table
        table = Table(show_header=True, header_style="bold magenta")
        for key in row.keys():
            table.add_column(key.replace('_', ' ').title())
        
        table.add_row(*[str(v) for v in row.values()])
        console.print(table)


def _display_batch_summary(result) -> None:  # BatchProcessingResult
    """Display batch processing summary"""
    
    console.print("\n[bold green]Batch Processing Summary[/bold green]")
    
    # Main statistics table
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="white")
    
    metrics = result.metrics
    table.add_row("Total Documents", str(metrics.total_documents))
    table.add_row("Successful", f"{metrics.successful_documents} ({metrics.successful_documents/metrics.total_documents:.1%})")
    table.add_row("Failed", f"{metrics.failed_documents} ({metrics.error_rate:.1%})")
    table.add_row("Processing Time", f"{metrics.total_duration:.1f}s")
    table.add_row("Throughput", f"{metrics.documents_per_second:.1f} docs/sec" if metrics.documents_per_second else "N/A")
    table.add_row("Avg Confidence", f"{metrics.average_confidence:.3f}" if metrics.average_confidence else "N/A")
    table.add_row("Review Required", str(metrics.review_required_count))
    
    console.print(table)
    
    # Confidence distribution
    if metrics.confidence_distribution:
        console.print("\n[bold]Confidence Distribution:[/bold]")
        conf_table = Table(show_header=True, header_style="bold magenta")
        conf_table.add_column("Confidence Level", style="cyan")
        conf_table.add_column("Count", style="white")
        conf_table.add_column("Percentage", style="green")
        
        total = sum(metrics.confidence_distribution.values())
        for level, count in metrics.confidence_distribution.items():
            percentage = f"{count/total:.1%}" if total > 0 else "0%"
            conf_table.add_row(level.replace('_', ' ').title(), str(count), percentage)
        
        console.print(conf_table)


def _display_training_metrics(metrics) -> None:  # ModelMetrics
    """Display training metrics"""
    
    console.print("\n[bold green]Training Results[/bold green]")
    
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="white")
    
    table.add_row("Accuracy", f"{metrics.accuracy:.3f}")
    table.add_row("Precision", f"{metrics.precision:.3f}")
    table.add_row("Recall", f"{metrics.recall:.3f}")
    table.add_row("F1 Score", f"{metrics.f1_score:.3f}")
    table.add_row("Test Set Size", str(metrics.test_set_size))
    
    console.print(table)


def _display_evaluation_metrics(metrics) -> None:  # ModelMetrics
    """Display evaluation metrics"""
    
    console.print("\n[bold green]Evaluation Results[/bold green]")
    
    # Overall metrics
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="white")
    
    table.add_row("Accuracy", f"{metrics.accuracy:.3f}")
    table.add_row("Precision (Macro)", f"{metrics.precision:.3f}")
    table.add_row("Recall (Macro)", f"{metrics.recall:.3f}")
    table.add_row("F1 Score (Macro)", f"{metrics.f1_score:.3f}")
    table.add_row("Test Set Size", str(metrics.test_set_size))
    
    console.print(table)
    
    # Per-class metrics
    if metrics.class_metrics:
        console.print("\n[bold]Per-Class Performance:[/bold]")
        class_table = Table(show_header=True, header_style="bold magenta")
        class_table.add_column("Class", style="cyan")
        class_table.add_column("Precision", style="white")
        class_table.add_column("Recall", style="white")
        class_table.add_column("F1-Score", style="white")
        
        for class_name, class_metrics in metrics.class_metrics.items():
            class_table.add_row(
                class_name.replace('_', ' ').title(),
                f"{class_metrics['precision']:.3f}",
                f"{class_metrics['recall']:.3f}",
                f"{class_metrics['f1-score']:.3f}"
            )
        
        console.print(class_table)


def _display_taxonomy() -> None:
    """Display the complete legal document taxonomy"""
    
    console.print("[bold blue]Legal Document Taxonomy[/bold blue]\n")
    
    hierarchy = get_category_hierarchy()
    
    # Document types
    console.print("[bold]Supported Document Types:[/bold]")
    doc_table = Table(show_header=True, header_style="bold magenta")
    doc_table.add_column("Type", style="cyan")
    doc_table.add_column("Domain", style="white")
    doc_table.add_column("Urgency", style="yellow")
    doc_table.add_column("Sensitivity", style="red")
    doc_table.add_column("Review Required", style="green")
    
    for doc_type, category in hierarchy.primary_categories.items():
        doc_table.add_row(
            doc_type.value.replace('_', ' ').title(),
            category.legal_domain.value.replace('_', ' ').title(),
            category.urgency_level.title(),
            category.sensitivity_level.title(),
            "Yes" if category.requires_human_review else "No"
        )
    
    console.print(doc_table)
    
    # Legal domains
    console.print(f"\n[bold]Legal Domains ({len(hierarchy.domain_mapping)}):[/bold]")
    for domain, doc_types in hierarchy.domain_mapping.items():
        type_names = [dt.value.replace('_', ' ').title() for dt in doc_types]
        console.print(f"• {domain.value.replace('_', ' ').title()}: {', '.join(type_names)}")


if __name__ == "__main__":
    app()