"""
Command-line interface for the Lemkin Document Processing and OCR Toolkit.

Provides comprehensive CLI commands for document processing, OCR, layout analysis,
and text extraction across multiple formats and languages.
"""

import json
import sys
from pathlib import Path
from typing import List, Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.text import Text

from .core import (
    DocumentProcessor,
    DocumentType,
    OCREngine,
    LanguageCode,
    DocumentAnalysis,
    OCRResult,
    LayoutAnalysis,
    ExtractionResult,
)

app = typer.Typer(
    name="lemkin-ocr",
    help="Document processing and OCR toolkit for legal investigations",
    no_args_is_help=True,
)

console = Console()


def validate_document_file(file_path: str) -> Path:
    """Validate that the file exists and is a supported document format."""
    path = Path(file_path)

    if not path.exists():
        raise typer.BadParameter(f"Document file does not exist: {file_path}")

    supported_extensions = {
        '.pdf', '.docx', '.doc', '.xlsx', '.xls', '.txt', '.rtf', '.html', '.htm',
        '.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif'
    }
    if path.suffix.lower() not in supported_extensions:
        raise typer.BadParameter(
            f"Unsupported document format: {path.suffix}. "
            f"Supported formats: {', '.join(supported_extensions)}"
        )

    return path


def save_results(results: dict, output_path: Optional[Path]) -> None:
    """Save analysis results to JSON file."""
    if output_path:
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, default=str, ensure_ascii=False)
            console.print(f"✅ Results saved to: {output_path}")
        except Exception as e:
            console.print(f"❌ Failed to save results: {e}", style="red")


@app.command()
def process_document(
    document_file: str = typer.Argument(..., help="Path to document file to process"),
    output: Optional[str] = typer.Option(
        None, "--output", "-o", help="Output file path for results (JSON)"
    ),
    ocr_engine: OCREngine = typer.Option(
        OCREngine.TESSERACT, "--engine", "-e", help="OCR engine to use"
    ),
    languages: Optional[str] = typer.Option(
        None, "--languages", "-l", help="Comma-separated list of languages (e.g., 'eng,spa,fra')"
    ),
    include_layout: bool = typer.Option(
        True, "--layout/--no-layout", help="Include layout analysis"
    ),
    include_ocr: bool = typer.Option(
        True, "--ocr/--no-ocr", help="Include OCR processing"
    ),
    include_extraction: bool = typer.Option(
        True, "--extraction/--no-extraction", help="Include native text extraction"
    ),
    show_details: bool = typer.Option(
        False, "--details", help="Show detailed analysis results"
    ),
) -> None:
    """Process document with comprehensive analysis including OCR, layout, and text extraction."""

    try:
        document_path = validate_document_file(document_file)
        output_path = Path(output) if output else None

        # Parse languages
        language_list = None
        if languages:
            try:
                language_list = [LanguageCode(lang.strip()) for lang in languages.split(',')]
            except ValueError as e:
                console.print(f"❌ Invalid language code: {e}", style="red")
                raise typer.Exit(1)

        console.print(f"📄 Processing document: {document_path.name}")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Processing document...", total=None)

            processor = DocumentProcessor(default_engine=ocr_engine)
            analysis = processor.process_document(
                document_path=document_path,
                ocr_engine=ocr_engine,
                languages=language_list,
                include_layout=include_layout,
                include_ocr=include_ocr,
                include_extraction=include_extraction
            )

            progress.update(task, description="Document processing completed!")

        # Display results
        console.print("\n" + "="*60)
        console.print(Panel(
            f"[bold green]Document Processing Results[/bold green]\n\n"
            f"📁 File: {document_path.name}\n"
            f"📋 Type: {analysis.document_type.value}\n"
            f"📊 Quality Score: {analysis.quality_score:.1%}\n"
            f"📝 Text Length: {len(analysis.final_text)} characters\n"
            f"⏱️ Processed: {analysis.analysis_timestamp.strftime('%Y-%m-%d %H:%M:%S')}",
            title="Document Analysis"
        ))

        # Show extracted text preview
        if analysis.final_text:
            preview_text = analysis.final_text[:500] + "..." if len(analysis.final_text) > 500 else analysis.final_text
            console.print(f"\n[bold blue]Extracted Text Preview:[/bold blue]")
            console.print(Panel(preview_text))
        else:
            console.print("\n[yellow]⚠️ No text was extracted from the document[/yellow]")

        # Show detailed results if requested
        if show_details:
            if analysis.ocr_result:
                console.print(f"\n[bold blue]OCR Results:[/bold blue]")
                console.print(f"Engine: {analysis.ocr_result.engine_used.value}")
                console.print(f"Confidence: {analysis.ocr_result.total_confidence:.1%}")
                console.print(f"Pages: {analysis.ocr_result.pages_processed}")
                console.print(f"Text blocks: {len(analysis.ocr_result.text_blocks)}")
                console.print(f"Languages: {', '.join([lang.value for lang in analysis.ocr_result.languages_detected])}")

            if analysis.extraction_result:
                console.print(f"\n[bold blue]Text Extraction Results:[/bold blue]")
                console.print(f"Method: {analysis.extraction_result.extraction_method}")
                console.print(f"Success: {'✅' if analysis.extraction_result.success else '❌'}")
                console.print(f"Tables found: {len(analysis.extraction_result.tables)}")
                console.print(f"Images found: {len(analysis.extraction_result.images)}")

            if analysis.layout_analysis:
                console.print(f"\n[bold blue]Layout Analysis Results:[/bold blue]")
                console.print(f"Elements detected: {len(analysis.layout_analysis.elements)}")
                console.print(f"Pages analyzed: {len(analysis.layout_analysis.page_layouts)}")

        # Show recommendations
        if analysis.recommendations:
            console.print(f"\n[bold yellow]Recommendations:[/bold yellow]")
            for rec in analysis.recommendations:
                console.print(f"• {rec}")

        # Save results
        if output_path:
            results = {
                "document_analysis": analysis.dict(),
                "analysis_type": "comprehensive_document_processing"
            }
            save_results(results, output_path)

        console.print(f"\n✅ Document processing completed successfully!")

    except Exception as e:
        console.print(f"❌ Document processing failed: {str(e)}", style="red")
        raise typer.Exit(1)


@app.command()
def ocr_only(
    document_file: str = typer.Argument(..., help="Path to document or image file"),
    output: Optional[str] = typer.Option(
        None, "--output", "-o", help="Output file path for OCR results (JSON)"
    ),
    engine: OCREngine = typer.Option(
        OCREngine.TESSERACT, "--engine", "-e", help="OCR engine to use"
    ),
    languages: Optional[str] = typer.Option(
        None, "--languages", "-l", help="Comma-separated list of languages"
    ),
    show_blocks: bool = typer.Option(
        False, "--blocks", help="Show individual text blocks with coordinates"
    ),
    confidence_threshold: float = typer.Option(
        0.5, "--threshold", "-t", help="Minimum confidence threshold", min=0.0, max=1.0
    ),
) -> None:
    """Perform OCR on document or image file."""

    try:
        document_path = validate_document_file(document_file)
        output_path = Path(output) if output else None

        # Parse languages
        language_list = None
        if languages:
            try:
                language_list = [LanguageCode(lang.strip()) for lang in languages.split(',')]
            except ValueError as e:
                console.print(f"❌ Invalid language code: {e}", style="red")
                raise typer.Exit(1)

        console.print(f"👁️ Performing OCR on: {document_path.name}")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Running OCR...", total=None)

            processor = DocumentProcessor()
            ocr_result = processor.ocr_engine.perform_ocr(
                document_path=document_path,
                engine=engine,
                languages=language_list
            )

            progress.update(task, description="OCR completed!")

        # Display results
        console.print("\n" + "="*60)
        console.print(Panel(
            f"[bold green]OCR Results[/bold green]\n\n"
            f"📁 File: {document_path.name}\n"
            f"🔧 Engine: {ocr_result.engine_used.value}\n"
            f"📊 Confidence: {ocr_result.total_confidence:.1%}\n"
            f"📄 Pages: {ocr_result.pages_processed}\n"
            f"🔤 Text blocks: {len(ocr_result.text_blocks)}\n"
            f"🌐 Languages: {', '.join([lang.value for lang in ocr_result.languages_detected])}\n"
            f"⏱️ Processing time: {ocr_result.processing_time:.2f}s",
            title="OCR Analysis"
        ))

        # Show extracted text
        if ocr_result.full_text:
            console.print(f"\n[bold blue]Extracted Text:[/bold blue]")
            preview_text = ocr_result.full_text[:1000] + "..." if len(ocr_result.full_text) > 1000 else ocr_result.full_text
            console.print(Panel(preview_text))
        else:
            console.print("\n[yellow]⚠️ No text was extracted by OCR[/yellow]")

        # Show text blocks if requested
        if show_blocks and ocr_result.text_blocks:
            console.print(f"\n[bold blue]Text Blocks (confidence >= {confidence_threshold:.1%}):[/bold blue]")

            table = Table(show_header=True, header_style="bold blue")
            table.add_column("Block", width=6)
            table.add_column("Page", width=6)
            table.add_column("Confidence", width=10)
            table.add_column("Coordinates", width=15)
            table.add_column("Text", min_width=30)

            for i, block in enumerate(ocr_result.text_blocks):
                if block.confidence >= confidence_threshold:
                    confidence_color = "green" if block.confidence >= 0.8 else "yellow" if block.confidence >= 0.6 else "red"

                    table.add_row(
                        str(i + 1),
                        str(block.page_number),
                        Text(f"{block.confidence:.1%}", style=confidence_color),
                        f"({block.bbox[0]},{block.bbox[1]},{block.bbox[2]},{block.bbox[3]})",
                        block.text[:50] + "..." if len(block.text) > 50 else block.text
                    )

            console.print(table)

        # Save results
        if output_path:
            results = {
                "ocr_result": ocr_result.dict(),
                "analysis_type": "ocr_only"
            }
            save_results(results, output_path)

        console.print(f"\n✅ OCR processing completed successfully!")

    except Exception as e:
        console.print(f"❌ OCR processing failed: {str(e)}", style="red")
        raise typer.Exit(1)


@app.command()
def extract_text(
    document_file: str = typer.Argument(..., help="Path to document file"),
    output: Optional[str] = typer.Option(
        None, "--output", "-o", help="Output file path for extracted text"
    ),
    format: str = typer.Option(
        "txt", "--format", "-f", help="Output format (txt, json)",
        click_type=typer.Choice(["txt", "json"])
    ),
    include_tables: bool = typer.Option(
        False, "--tables", help="Extract and display tables separately"
    ),
    include_metadata: bool = typer.Option(
        False, "--metadata", help="Include document metadata"
    ),
) -> None:
    """Extract text from document using native format parsing (no OCR)."""

    try:
        document_path = validate_document_file(document_file)
        output_path = Path(output) if output else None

        console.print(f"📝 Extracting text from: {document_path.name}")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Extracting text...", total=None)

            processor = DocumentProcessor()

            # Determine document type
            doc_type = processor._determine_document_type(document_path)

            # Extract text
            extraction_result = processor.text_extractor.extract_text(document_path, doc_type)

            progress.update(task, description="Text extraction completed!")

        # Display results
        console.print("\n" + "="*60)
        console.print(Panel(
            f"[bold green]Text Extraction Results[/bold green]\n\n"
            f"📁 File: {document_path.name}\n"
            f"📋 Type: {doc_type.value}\n"
            f"✅ Success: {'Yes' if extraction_result.success else 'No'}\n"
            f"📝 Text length: {len(extraction_result.extracted_text)} characters\n"
            f"📊 Tables: {len(extraction_result.tables)}\n"
            f"🖼️ Images: {len(extraction_result.images)}\n"
            f"⏱️ Processing time: {extraction_result.processing_time:.2f}s",
            title="Text Extraction"
        ))

        # Show extracted text
        if extraction_result.extracted_text:
            preview_text = extraction_result.extracted_text[:1000] + "..." if len(extraction_result.extracted_text) > 1000 else extraction_result.extracted_text
            console.print(f"\n[bold blue]Extracted Text Preview:[/bold blue]")
            console.print(Panel(preview_text))
        else:
            console.print("\n[yellow]⚠️ No text was extracted from the document[/yellow]")

        # Show metadata if requested
        if include_metadata and extraction_result.metadata:
            console.print(f"\n[bold blue]Document Metadata:[/bold blue]")
            metadata = extraction_result.metadata

            table = Table(show_header=True, header_style="bold blue")
            table.add_column("Property", width=20)
            table.add_column("Value", min_width=30)

            if metadata.title:
                table.add_row("Title", metadata.title)
            if metadata.author:
                table.add_row("Author", metadata.author)
            if metadata.subject:
                table.add_row("Subject", metadata.subject)
            if metadata.creator:
                table.add_row("Creator", metadata.creator)
            if metadata.creation_date:
                table.add_row("Created", metadata.creation_date.strftime('%Y-%m-%d %H:%M:%S'))
            table.add_row("Pages", str(metadata.page_count))

            console.print(table)

        # Show tables if requested
        if include_tables and extraction_result.tables:
            console.print(f"\n[bold blue]Extracted Tables:[/bold blue]")
            for i, table_df in enumerate(extraction_result.tables):
                console.print(f"\nTable {i + 1}:")
                # Show first few rows of each table
                preview_df = table_df.head(3)
                console.print(preview_df.to_string(index=False))
                if len(table_df) > 3:
                    console.print(f"... and {len(table_df) - 3} more rows")

        # Save results
        if output_path:
            if format == "txt":
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(extraction_result.extracted_text)
                console.print(f"✅ Text saved to: {output_path}")
            else:  # json
                results = {
                    "extraction_result": extraction_result.dict(),
                    "analysis_type": "text_extraction"
                }
                save_results(results, output_path)

        console.print(f"\n✅ Text extraction completed successfully!")

    except Exception as e:
        console.print(f"❌ Text extraction failed: {str(e)}", style="red")
        raise typer.Exit(1)


@app.command()
def batch_process(
    input_dir: str = typer.Argument(..., help="Directory containing documents to process"),
    output_dir: str = typer.Argument(..., help="Directory to save processing results"),
    file_pattern: str = typer.Option(
        "*.*", "--pattern", "-p", help="File pattern to match (e.g., '*.pdf', '*.docx')"
    ),
    ocr_engine: OCREngine = typer.Option(
        OCREngine.TESSERACT, "--engine", "-e", help="OCR engine to use"
    ),
    languages: Optional[str] = typer.Option(
        None, "--languages", "-l", help="Comma-separated list of languages"
    ),
    processing_mode: str = typer.Option(
        "comprehensive", "--mode", "-m", help="Processing mode",
        click_type=typer.Choice(["comprehensive", "ocr-only", "extract-only"])
    ),
) -> None:
    """Process multiple documents in batch mode."""

    try:
        input_path = Path(input_dir)
        output_path = Path(output_dir)

        if not input_path.exists():
            raise typer.BadParameter(f"Input directory does not exist: {input_dir}")

        output_path.mkdir(parents=True, exist_ok=True)

        # Find documents
        document_files = list(input_path.glob(file_pattern))

        if not document_files:
            console.print(f"❌ No documents found matching pattern '{file_pattern}' in {input_path}")
            raise typer.Exit(1)

        # Parse languages
        language_list = None
        if languages:
            try:
                language_list = [LanguageCode(lang.strip()) for lang in languages.split(',')]
            except ValueError as e:
                console.print(f"❌ Invalid language code: {e}", style="red")
                raise typer.Exit(1)

        console.print(f"🔄 Processing {len(document_files)} documents...")
        console.print(f"📂 Input: {input_path}")
        console.print(f"📁 Output: {output_path}")
        console.print(f"⚙️ Mode: {processing_mode}")

        # Process each document
        results_summary = []
        processor = DocumentProcessor(default_engine=ocr_engine)

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:

            main_task = progress.add_task(f"Processing documents...", total=len(document_files))

            for i, document_file in enumerate(document_files):
                try:
                    progress.update(main_task, description=f"Processing {document_file.name}...")

                    # Determine output file
                    output_file = output_path / f"{document_file.stem}_{processing_mode}.json"

                    # Perform processing based on mode
                    if processing_mode == "comprehensive":
                        result = processor.process_document(
                            document_path=document_file,
                            ocr_engine=ocr_engine,
                            languages=language_list
                        )
                        analysis_data = {"document_analysis": result.dict()}

                    elif processing_mode == "ocr-only":
                        result = processor.ocr_engine.perform_ocr(
                            document_path=document_file,
                            engine=ocr_engine,
                            languages=language_list
                        )
                        analysis_data = {"ocr_result": result.dict()}

                    elif processing_mode == "extract-only":
                        doc_type = processor._determine_document_type(document_file)
                        result = processor.text_extractor.extract_text(document_file, doc_type)
                        analysis_data = {"extraction_result": result.dict()}

                    # Save results
                    analysis_data["analysis_type"] = processing_mode
                    analysis_data["processed_file"] = str(document_file)

                    with open(output_file, 'w', encoding='utf-8') as f:
                        json.dump(analysis_data, f, indent=2, default=str, ensure_ascii=False)

                    results_summary.append({
                        "file": document_file.name,
                        "status": "success",
                        "output": output_file.name
                    })

                except Exception as e:
                    console.print(f"❌ Failed to process {document_file.name}: {e}")
                    results_summary.append({
                        "file": document_file.name,
                        "status": "error",
                        "error": str(e)
                    })

                progress.advance(main_task)

        # Show summary
        successful = sum(1 for r in results_summary if r["status"] == "success")
        failed = len(results_summary) - successful

        console.print(f"\n✅ Batch processing completed!")
        console.print(f"📊 Successfully processed: {successful}/{len(document_files)} documents")

        if failed > 0:
            console.print(f"❌ Failed: {failed} documents")

            # Show failed files
            for result in results_summary:
                if result["status"] == "error":
                    console.print(f"  • {result['file']}: {result['error']}")

        # Save batch summary
        summary_file = output_path / f"batch_summary_{processing_mode}.json"
        with open(summary_file, 'w') as f:
            json.dump({
                "batch_processing_mode": processing_mode,
                "total_files": len(document_files),
                "successful": successful,
                "failed": failed,
                "results": results_summary
            }, f, indent=2)

        console.print(f"📋 Batch summary saved to: {summary_file}")

    except Exception as e:
        console.print(f"❌ Batch processing failed: {str(e)}", style="red")
        raise typer.Exit(1)


@app.command()
def compare_engines(
    document_file: str = typer.Argument(..., help="Path to document file"),
    engines: str = typer.Option(
        "tesseract,easyocr", "--engines", help="Comma-separated list of engines to compare"
    ),
    languages: Optional[str] = typer.Option(
        None, "--languages", "-l", help="Comma-separated list of languages"
    ),
    output: Optional[str] = typer.Option(
        None, "--output", "-o", help="Output file path for comparison results (JSON)"
    ),
) -> None:
    """Compare different OCR engines on the same document."""

    try:
        document_path = validate_document_file(document_file)
        output_path = Path(output) if output else None

        # Parse engines
        try:
            engine_list = [OCREngine(engine.strip()) for engine in engines.split(',')]
        except ValueError as e:
            console.print(f"❌ Invalid OCR engine: {e}", style="red")
            raise typer.Exit(1)

        # Parse languages
        language_list = None
        if languages:
            try:
                language_list = [LanguageCode(lang.strip()) for lang in languages.split(',')]
            except ValueError as e:
                console.print(f"❌ Invalid language code: {e}", style="red")
                raise typer.Exit(1)

        console.print(f"🔍 Comparing OCR engines on: {document_path.name}")
        console.print(f"🔧 Engines: {', '.join([e.value for e in engine_list])}")

        processor = DocumentProcessor()
        comparison_results = {}

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:

            task = progress.add_task("Comparing engines...", total=len(engine_list))

            for engine in engine_list:
                try:
                    progress.update(task, description=f"Running {engine.value}...")

                    result = processor.ocr_engine.perform_ocr(
                        document_path=document_path,
                        engine=engine,
                        languages=language_list
                    )

                    comparison_results[engine.value] = result.dict()
                    progress.advance(task)

                except Exception as e:
                    console.print(f"❌ {engine.value} failed: {e}")
                    comparison_results[engine.value] = {"error": str(e)}
                    progress.advance(task)

        # Display comparison results
        console.print("\n" + "="*80)
        console.print(Panel("OCR Engine Comparison Results", style="bold green"))

        table = Table(show_header=True, header_style="bold blue")
        table.add_column("Engine", width=15)
        table.add_column("Confidence", width=12)
        table.add_column("Pages", width=8)
        table.add_column("Text Blocks", width=12)
        table.add_column("Processing Time", width=15)
        table.add_column("Text Length", width=12)
        table.add_column("Languages", width=15)

        for engine_name, result in comparison_results.items():
            if "error" not in result:
                confidence_color = "green" if result["total_confidence"] >= 0.8 else "yellow" if result["total_confidence"] >= 0.6 else "red"

                table.add_row(
                    engine_name,
                    Text(f"{result['total_confidence']:.1%}", style=confidence_color),
                    str(result["pages_processed"]),
                    str(len(result["text_blocks"])),
                    f"{result['processing_time']:.2f}s",
                    str(len(result["full_text"])),
                    ", ".join(result["languages_detected"])
                )
            else:
                table.add_row(
                    engine_name,
                    Text("ERROR", style="red"),
                    "-", "-", "-", "-", "-"
                )

        console.print(table)

        # Show text comparison preview
        console.print(f"\n[bold blue]Text Extraction Comparison:[/bold blue]")
        for engine_name, result in comparison_results.items():
            if "error" not in result and result["full_text"]:
                preview = result["full_text"][:200] + "..." if len(result["full_text"]) > 200 else result["full_text"]
                console.print(f"\n[bold]{engine_name}:[/bold]")
                console.print(Panel(preview, border_style="dim"))

        # Save results
        if output_path:
            results = {
                "engine_comparison": comparison_results,
                "analysis_type": "engine_comparison",
                "document_file": str(document_path)
            }
            save_results(results, output_path)

        console.print(f"\n✅ Engine comparison completed!")

    except Exception as e:
        console.print(f"❌ Engine comparison failed: {str(e)}", style="red")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()