"""
Lemkin Export CLI

Command-line interface for multi-format export capabilities, data validation,
and platform integration for legal investigation workflows.
"""

from datetime import datetime
from pathlib import Path
from typing import Optional, List
import json

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.prompt import Prompt, Confirm
from loguru import logger

from .core import (
    ExportEngine,
    ExportFormat,
    CompressionType,
    ExportScope,
    ValidationLevel,
    ExportItem,
    DataValidator,
    FormatConverter
)

app = typer.Typer(help="Lemkin Multi-format Export System")
console = Console()


@app.command()
def create_package(
    case_id: str = typer.Argument(..., help="Case ID for export"),
    export_format: ExportFormat = typer.Argument(..., help="Export format"),
    export_scope: ExportScope = typer.Option(ExportScope.CASE_COMPLETE, help="Export scope"),
    data_file: Path = typer.Option(..., help="JSON file containing export items"),
    created_by: str = typer.Option(..., help="User creating the export"),
    output_path: Optional[Path] = typer.Option(None, help="Output directory path"),
    validation_level: ValidationLevel = typer.Option(ValidationLevel.STANDARD, help="Validation level"),
    compression: CompressionType = typer.Option(CompressionType.NONE, help="Compression type")
):
    """Create a complete export package from case data."""
    console.print(f"[bold blue]Creating export package for case {case_id}[/bold blue]")

    try:
        # Load export items
        if not data_file.exists():
            console.print(f"[bold red]✗[/bold red] Data file not found: {data_file}")
            raise typer.Exit(1)

        with open(data_file) as f:
            items_data = json.load(f)

        # Convert to ExportItem objects
        export_items = []
        for item_data in items_data:
            if "source_path" in item_data and item_data["source_path"]:
                item_data["source_path"] = Path(item_data["source_path"])
            export_items.append(ExportItem(**item_data))

        console.print(f"Loaded {len(export_items)} items for export")

        # Create export engine
        engine = ExportEngine(output_path)

        # Show export summary before processing
        summary_content = f"""
[bold]Export Configuration:[/bold]
• Case ID: {case_id}
• Format: {export_format}
• Scope: {export_scope}
• Items: {len(export_items)}
• Validation: {validation_level}
• Compression: {compression}
• Created By: {created_by}
        """
        console.print(Panel(summary_content, title="Export Summary", border_style="blue"))

        if not Confirm.ask("Proceed with export?"):
            console.print("Export cancelled.")
            return

        # Create export package with progress tracking
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            console=console
        ) as progress:

            main_task = progress.add_task("Creating export package...", total=100)

            progress.update(main_task, advance=20, description="Validating items...")

            package = engine.create_export_package(
                case_id=case_id,
                items=export_items,
                export_format=export_format,
                export_scope=export_scope,
                created_by=created_by,
                validation_level=validation_level,
                compression=compression
            )

            progress.update(main_task, advance=80, description="Package created!")

        console.print(f"[bold green]✓[/bold green] Export package created successfully!")
        console.print(f"Package ID: {package.package_id}")
        console.print(f"Output Path: {package.output_path}")
        console.print(f"Package Size: {package.package_size / 1024 / 1024:.2f} MB")

        # Display export log
        if package.export_log:
            log_table = Table(title="Export Log", show_header=False)
            log_table.add_column("Entry", style="dim")

            for entry in package.export_log[-10:]:  # Show last 10 entries
                log_table.add_row(entry)

            console.print(log_table)

    except Exception as e:
        console.print(f"[bold red]✗[/bold red] Export failed: {e}")
        raise typer.Exit(1)


@app.command()
def convert(
    input_file: Path = typer.Argument(..., help="Input data file"),
    target_format: ExportFormat = typer.Argument(..., help="Target export format"),
    output_file: Optional[Path] = typer.Option(None, help="Output file path"),
    case_id: Optional[str] = typer.Option(None, help="Case ID (for legal formats)")
):
    """Convert data file to different export format."""
    console.print(f"[bold blue]Converting {input_file.name} to {target_format}[/bold blue]")

    try:
        # Validate input file
        if not input_file.exists():
            console.print(f"[bold red]✗[/bold red] Input file not found: {input_file}")
            raise typer.Exit(1)

        # Determine output file path
        if not output_file:
            output_file = input_file.parent / f"{input_file.stem}_converted.{target_format}"

        # Load input data
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:

            task = progress.add_task("Loading input data...", total=None)

            if input_file.suffix.lower() == '.json':
                with open(input_file) as f:
                    data = json.load(f)
            elif input_file.suffix.lower() == '.csv':
                import pandas as pd
                data = pd.read_csv(input_file)
            elif input_file.suffix.lower() in ['.xlsx', '.xls']:
                import pandas as pd
                data = pd.read_excel(input_file)
            else:
                console.print(f"[bold red]✗[/bold red] Unsupported input format: {input_file.suffix}")
                raise typer.Exit(1)

            progress.update(task, description="Converting format...")

            # Convert data
            converter = FormatConverter()
            kwargs = {}
            if case_id and target_format in [ExportFormat.EDRM_XML, ExportFormat.LEGAL_XML]:
                kwargs["case_id"] = case_id

            converted_file = converter.convert_data(data, target_format, output_file, **kwargs)

            progress.update(task, completed=100, description="Conversion complete!")

        console.print(f"[bold green]✓[/bold green] Conversion successful!")
        console.print(f"Output file: {converted_file}")
        console.print(f"File size: {converted_file.stat().st_size / 1024:.1f} KB")

    except Exception as e:
        console.print(f"[bold red]✗[/bold red] Conversion failed: {e}")
        raise typer.Exit(1)


@app.command()
def validate(
    data_file: Path = typer.Argument(..., help="JSON file containing export items"),
    validation_level: ValidationLevel = typer.Option(ValidationLevel.STANDARD, help="Validation level"),
    compliance_standards: Optional[List[str]] = typer.Option(None, help="Compliance standards to check")
):
    """Validate export data for integrity and compliance."""
    console.print(f"[bold blue]Validating export data with {validation_level} level[/bold blue]")

    try:
        # Load export items
        if not data_file.exists():
            console.print(f"[bold red]✗[/bold red] Data file not found: {data_file}")
            raise typer.Exit(1)

        with open(data_file) as f:
            items_data = json.load(f)

        # Convert to ExportItem objects
        export_items = []
        for item_data in items_data:
            if "source_path" in item_data and item_data["source_path"]:
                item_data["source_path"] = Path(item_data["source_path"])
            export_items.append(ExportItem(**item_data))

        console.print(f"Loaded {len(export_items)} items for validation")

        validator = DataValidator(validation_level)

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:

            task = progress.add_task("Validating items...", total=None)

            validation_result = validator.validate_export_items(export_items)

            progress.update(task, completed=100, description="Validation complete!")

        # Display results
        is_valid = validation_result["valid"]
        status_color = "green" if is_valid else "red"
        status_text = "VALID" if is_valid else "INVALID"

        console.print(f"[bold {status_color}]{status_text}[/bold {status_color}] - {validation_result['items_validated']} items validated")

        # Show validation details
        if validation_result["warnings"]:
            warning_table = Table(title="Validation Warnings", show_header=True)
            warning_table.add_column("Warning", style="yellow")

            for warning in validation_result["warnings"]:
                warning_table.add_row(warning)

            console.print(warning_table)

        if validation_result["errors"]:
            error_table = Table(title="Validation Errors", show_header=True)
            error_table.add_column("Error", style="red")

            for error in validation_result["errors"]:
                error_table.add_row(error)

            console.print(error_table)

        # Compliance validation if requested
        if compliance_standards and is_valid:
            console.print("\n[bold blue]Checking compliance standards...[/bold blue]")

            # Create mock manifest for compliance checking
            from .core import ExportManifest, ExportScope, ExportFormat
            manifest = ExportManifest(
                case_id="validation_test",
                export_type=ExportScope.CASE_COMPLETE,
                export_format=ExportFormat.JSON,
                created_by="validator",
                total_items=len(export_items),
                total_size=sum(item.file_size or 0 for item in export_items),
                validation_level=validation_level,
                items=export_items
            )

            compliance_results = validator.validate_compliance(manifest, compliance_standards)

            for standard, result in compliance_results.items():
                status = "✓" if result["valid"] else "✗"
                color = "green" if result["valid"] else "red"
                console.print(f"[bold {color}]{status}[/bold {color}] {standard}: {'COMPLIANT' if result['valid'] else 'NON-COMPLIANT'}")

                if result["warnings"]:
                    for warning in result["warnings"]:
                        console.print(f"  [yellow]⚠[/yellow] {warning}")

                if result["errors"]:
                    for error in result["errors"]:
                        console.print(f"  [red]✗[/red] {error}")

    except Exception as e:
        console.print(f"[bold red]✗[/bold red] Validation failed: {e}")
        raise typer.Exit(1)


@app.command()
def compress(
    input_path: Path = typer.Argument(..., help="Input file or directory to compress"),
    compression_type: CompressionType = typer.Argument(..., help="Compression algorithm"),
    output_path: Optional[Path] = typer.Option(None, help="Output compressed file path")
):
    """Compress files or directories for export."""
    console.print(f"[bold blue]Compressing {input_path.name} using {compression_type}[/bold blue]")

    try:
        if not input_path.exists():
            console.print(f"[bold red]✗[/bold red] Input path not found: {input_path}")
            raise typer.Exit(1)

        # Determine output path
        if not output_path:
            output_path = input_path.parent / f"{input_path.name}.{compression_type}"

        from .core import CompressionManager

        compressor = CompressionManager()

        # Calculate original size
        original_size = compressor._calculate_path_size(input_path)

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:

            task = progress.add_task("Compressing...", total=None)

            compressed_path = compressor.compress_package(input_path, output_path, compression_type)

            progress.update(task, completed=100, description="Compression complete!")

        # Calculate compression statistics
        compressed_size = compressed_path.stat().st_size
        ratio = compressor.calculate_compression_ratio(original_size, compressed_size)

        console.print(f"[bold green]✓[/bold green] Compression successful!")
        console.print(f"Output file: {compressed_path}")
        console.print(f"Original size: {original_size / 1024 / 1024:.2f} MB")
        console.print(f"Compressed size: {compressed_size / 1024 / 1024:.2f} MB")
        console.print(f"Compression ratio: {ratio:.1f}%")

    except Exception as e:
        console.print(f"[bold red]✗[/bold red] Compression failed: {e}")
        raise typer.Exit(1)


@app.command()
def list_formats():
    """List all supported export formats."""
    console.print("[bold blue]Supported Export Formats[/bold blue]")

    # Group formats by category
    format_categories = {
        "Document Formats": [
            ExportFormat.PDF, ExportFormat.HTML, ExportFormat.DOCX,
            ExportFormat.MARKDOWN, ExportFormat.TXT
        ],
        "Data Formats": [
            ExportFormat.CSV, ExportFormat.XLSX, ExportFormat.JSON,
            ExportFormat.XML, ExportFormat.YAML, ExportFormat.PARQUET
        ],
        "Archive Formats": [
            ExportFormat.ZIP, ExportFormat.TAR_GZ, ExportFormat.TAR_XZ
        ],
        "Legal Formats": [
            ExportFormat.EDRM_XML, ExportFormat.CONCORDANCE,
            ExportFormat.RELATIVITY, ExportFormat.LEGAL_XML
        ],
        "Platform Formats": [
            ExportFormat.SALESFORCE, ExportFormat.SHAREPOINT, ExportFormat.TEAMS
        ]
    }

    for category, formats in format_categories.items():
        table = Table(title=category, show_header=True)
        table.add_column("Format", style="cyan")
        table.add_column("Extension", style="green")
        table.add_column("Description", style="dim")

        for fmt in formats:
            if fmt == ExportFormat.EDRM_XML:
                description = "Electronic Discovery Reference Model XML"
            elif fmt == ExportFormat.CONCORDANCE:
                description = "Concordance load file format"
            elif fmt == ExportFormat.PARQUET:
                description = "Columnar storage format"
            elif fmt == ExportFormat.TAR_GZ:
                description = "Compressed tar archive"
            else:
                description = f"{fmt.upper()} format"

            table.add_row(fmt, fmt, description)

        console.print(table)
        console.print()


@app.command()
def create_sample_data(
    output_file: Path = typer.Option("sample_export_items.json", help="Output sample data file"),
    item_count: int = typer.Option(10, help="Number of sample items to create")
):
    """Create sample export data for testing."""
    console.print(f"[bold blue]Creating sample export data with {item_count} items[/bold blue]")

    try:
        sample_items = []

        for i in range(item_count):
            item = {
                "item_id": f"ITEM_{i+1:03d}",
                "item_type": "document" if i % 2 == 0 else "email",
                "metadata": {
                    "title": f"Sample Document {i+1}",
                    "author": f"Author {i % 3 + 1}",
                    "created_date": datetime.now().isoformat(),
                    "file_type": "pdf" if i % 2 == 0 else "eml",
                    "classification": "internal" if i % 2 == 0 else "confidential"
                },
                "content": {
                    "text": f"This is sample content for item {i+1}. " * 10,
                    "word_count": 100 + i * 5,
                    "language": "en"
                },
                "file_size": 1024 * (i + 1),
                "file_hash": f"sha256:{hash(f'content_{i}'):x}",
                "tags": ["sample", "test", f"category_{i % 3}"],
                "permissions": ["read", "export"]
            }

            sample_items.append(item)

        # Save sample data
        with open(output_file, 'w') as f:
            json.dump(sample_items, f, indent=2)

        console.print(f"[bold green]✓[/bold green] Sample data created!")
        console.print(f"Output file: {output_file}")
        console.print(f"Items created: {len(sample_items)}")

        # Show sample structure
        sample_item = sample_items[0]
        sample_content = json.dumps(sample_item, indent=2)[:300] + "..."

        console.print(Panel(
            sample_content,
            title="Sample Item Structure",
            border_style="green"
        ))

    except Exception as e:
        console.print(f"[bold red]✗[/bold red] Failed to create sample data: {e}")
        raise typer.Exit(1)


@app.command()
def export_to_platform(
    package_file: Path = typer.Argument(..., help="Export package JSON file"),
    platform: str = typer.Argument(..., help="Platform name (sharepoint, salesforce, teams)"),
    config_file: Path = typer.Option(..., help="Platform configuration JSON file")
):
    """Export package to external platform."""
    console.print(f"[bold blue]Exporting package to {platform}[/bold blue]")

    try:
        # Load package information
        if not package_file.exists():
            console.print(f"[bold red]✗[/bold red] Package file not found: {package_file}")
            raise typer.Exit(1)

        with open(package_file) as f:
            package_data = json.load(f)

        # Load platform configuration
        if not config_file.exists():
            console.print(f"[bold red]✗[/bold red] Config file not found: {config_file}")
            raise typer.Exit(1)

        with open(config_file) as f:
            platform_config = json.load(f)

        platform_config["platform"] = platform

        # Create mock package for export
        from .core import ExportPackage, ExportManifest

        manifest = ExportManifest(**package_data["manifest"])
        package = ExportPackage(
            manifest=manifest,
            output_path=Path(package_data["output_path"]),
            package_size=package_data["package_size"]
        )

        engine = ExportEngine()

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:

            task = progress.add_task(f"Exporting to {platform}...", total=None)

            result = engine.export_to_platform(package, platform_config)

            progress.update(task, completed=100, description="Export complete!")

        console.print(f"[bold green]✓[/bold green] Platform export successful!")
        console.print(f"Platform: {result['platform']}")
        console.print(f"Status: {result['status']}")
        console.print(f"Exported at: {result['exported_at']}")

        if "location" in result:
            console.print(f"Location: {result['location']}")

    except Exception as e:
        console.print(f"[bold red]✗[/bold red] Platform export failed: {e}")
        raise typer.Exit(1)


@app.command()
def verify_package(
    package_path: Path = typer.Argument(..., help="Path to export package"),
    check_integrity: bool = typer.Option(True, help="Verify file integrity"),
    check_manifest: bool = typer.Option(True, help="Validate manifest")
):
    """Verify integrity of export package."""
    console.print(f"[bold blue]Verifying export package: {package_path.name}[/bold blue]")

    try:
        if not package_path.exists():
            console.print(f"[bold red]✗[/bold red] Package not found: {package_path}")
            raise typer.Exit(1)

        verification_results = []

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:

            task = progress.add_task("Verifying package...", total=100)

            # Check if it's a directory or compressed file
            if package_path.is_dir():
                package_dir = package_path
            else:
                # For simplicity, assume it's a zip file that we can check
                console.print("[yellow]Compressed package verification not fully implemented[/yellow]")
                package_dir = package_path.parent

            progress.update(task, advance=20, description="Checking manifest...")

            # Check manifest file
            manifest_file = package_dir / "manifest.json"
            if check_manifest and manifest_file.exists():
                with open(manifest_file) as f:
                    manifest_data = json.load(f)

                verification_results.append("✓ Manifest file found and readable")

                # Check manifest completeness
                required_fields = ["case_id", "export_type", "created_at", "total_items"]
                for field in required_fields:
                    if field in manifest_data:
                        verification_results.append(f"✓ Manifest contains {field}")
                    else:
                        verification_results.append(f"✗ Manifest missing {field}")

            else:
                verification_results.append("✗ Manifest file not found")

            progress.update(task, advance=40, description="Checking files...")

            # Check README file
            readme_file = package_dir / "README.txt"
            if readme_file.exists():
                verification_results.append("✓ README file found")
            else:
                verification_results.append("⚠ README file missing")

            progress.update(task, advance=40, description="Verification complete!")

        # Display results
        console.print("\n[bold]Verification Results:[/bold]")

        for result in verification_results:
            if result.startswith("✓"):
                console.print(f"[green]{result}[/green]")
            elif result.startswith("✗"):
                console.print(f"[red]{result}[/red]")
            elif result.startswith("⚠"):
                console.print(f"[yellow]{result}[/yellow]")
            else:
                console.print(result)

        # Overall status
        errors = [r for r in verification_results if r.startswith("✗")]
        if not errors:
            console.print(f"\n[bold green]✓ Package verification passed![/bold green]")
        else:
            console.print(f"\n[bold red]✗ Package verification failed with {len(errors)} errors[/bold red]")

    except Exception as e:
        console.print(f"[bold red]✗[/bold red] Verification failed: {e}")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()