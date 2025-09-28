"""
Command-line interface for Lemkin OSINT toolkit.
"""

import json
from pathlib import Path
from typing import Optional, List

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.json import JSON

from .core import (
    OSINTCollector,
    WebArchiver,
    MetadataExtractor,
    SourceVerifier,
    Source,
    SourceType,
)

app = typer.Typer(
    name="lemkin-osint",
    help="Systematic open-source intelligence gathering for legal investigations",
    no_args_is_help=True
)
console = Console()


@app.command()
def collect(
    query: str = typer.Argument(..., help="Search query for OSINT collection"),
    platforms: str = typer.Option("wayback", help="Comma-separated platforms"),
    limit: int = typer.Option(100, help="Maximum items to collect"),
    output: Optional[Path] = typer.Option(None, help="Output file path")
):
    """Collect OSINT data from specified platforms"""

    try:
        platform_list = [p.strip() for p in platforms.split(",")]

        console.print(f"[cyan]Starting OSINT collection for: {query}[/cyan]")
        console.print(f"Platforms: {', '.join(platform_list)}")

        collector = OSINTCollector()
        collection = collector.collect_social_media_evidence(
            query=query,
            platforms=platform_list,
            limit=limit
        )

        # Display results
        table = Table(title="OSINT Collection Results")
        table.add_column("Property", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Collection ID", collection.collection_id)
        table.add_row("Query", collection.query)
        table.add_row("Total Items", str(collection.total_items))
        table.add_row("Platforms", ", ".join(collection.platforms))
        table.add_row("Collected Date", collection.collected_date.isoformat())

        console.print(table)

        # Save to file if requested
        if output:
            with open(output, 'w') as f:
                json.dump(collection.model_dump(mode='json'), f, indent=2, default=str)
            console.print(f"[green]✓ Results saved to: {output}[/green]")

    except Exception as e:
        console.print(f"[red]Error during collection: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def archive(
    urls: str = typer.Argument(..., help="Comma-separated URLs to archive"),
    output: Optional[Path] = typer.Option(None, help="Output file path")
):
    """Archive web content using Wayback Machine"""

    try:
        url_list = [url.strip() for url in urls.split(",")]

        console.print(f"[cyan]Archiving {len(url_list)} URLs...[/cyan]")

        archiver = WebArchiver()
        archive_collection = archiver.archive_web_content(url_list)

        # Display results
        table = Table(title="Archive Results")
        table.add_column("URL", style="cyan")
        table.add_column("Status", style="green")
        table.add_column("Archive URL", style="blue")

        for item in archive_collection.archived_items:
            status = "✓" if item.get("status") == "success" else "✗"
            archive_url = item.get("archive_url", "N/A")
            table.add_row(
                item["original_url"][:50],
                status,
                archive_url[:50] if archive_url != "N/A" else "Failed"
            )

        console.print(table)

        # Save to file if requested
        if output:
            with open(output, 'w') as f:
                json.dump(archive_collection.model_dump(mode='json'), f, indent=2, default=str)
            console.print(f"[green]✓ Archive data saved to: {output}[/green]")

    except Exception as e:
        console.print(f"[red]Error during archiving: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def extract_metadata(
    file_path: Path = typer.Argument(..., help="Path to media file"),
    output: Optional[Path] = typer.Option(None, help="Output file path"),
    show_gps: bool = typer.Option(False, help="Show GPS coordinates if available")
):
    """Extract EXIF/XMP metadata from media files"""

    if not file_path.exists():
        console.print(f"[red]Error: File not found: {file_path}[/red]")
        raise typer.Exit(1)

    try:
        extractor = MetadataExtractor()
        metadata = extractor.extract_media_metadata(file_path)

        # Display results
        panel_content = f"""
[bold]File:[/bold] {metadata.file_path.name}
[bold]Hash:[/bold] {metadata.file_hash[:16]}...
[bold]Size:[/bold] {metadata.file_size:,} bytes
[bold]MIME Type:[/bold] {metadata.mime_type}
[bold]Creation Date:[/bold] {metadata.creation_date or 'Unknown'}
[bold]Modification Date:[/bold] {metadata.modification_date}
        """

        if metadata.camera_info:
            panel_content += f"\n[bold]Camera:[/bold] {metadata.camera_info.get('make', '')} {metadata.camera_info.get('model', '')}"

        if metadata.software_info:
            panel_content += f"\n[bold]Software:[/bold] {metadata.software_info}"

        if show_gps and metadata.gps_data:
            if 'decimal_coordinates' in metadata.gps_data:
                coords = metadata.gps_data['decimal_coordinates']
                panel_content += f"\n[bold]GPS:[/bold] {coords['latitude']:.6f}, {coords['longitude']:.6f}"

        console.print(Panel(panel_content, title="Media Metadata"))

        # Show EXIF data count
        if metadata.exif_data:
            console.print(f"\n[cyan]EXIF Tags Found: {len(metadata.exif_data)}[/cyan]")

            # Show first few EXIF tags
            table = Table(title="Sample EXIF Data")
            table.add_column("Tag", style="cyan")
            table.add_column("Value", style="green")

            for i, (tag, value) in enumerate(metadata.exif_data.items()):
                if i >= 10:  # Show only first 10
                    break
                table.add_row(str(tag), str(value)[:50])

            console.print(table)

        # Save to file if requested
        if output:
            # Convert Path to string for JSON serialization
            output_data = metadata.model_dump(mode='json')
            output_data['file_path'] = str(metadata.file_path)

            with open(output, 'w') as f:
                json.dump(output_data, f, indent=2, default=str)
            console.print(f"[green]✓ Metadata saved to: {output}[/green]")

    except Exception as e:
        console.print(f"[red]Error extracting metadata: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def verify_source(
    url: str = typer.Argument(..., help="URL to verify"),
    title: Optional[str] = typer.Option(None, help="Page title"),
    source_type: Optional[str] = typer.Option(None, help="Source type")
):
    """Verify credibility of an OSINT source"""

    try:
        # Create source object
        source = Source(
            url=url,
            title=title,
            source_type=SourceType(source_type) if source_type else SourceType.UNKNOWN
        )

        verifier = SourceVerifier()
        assessment = verifier.verify_source_credibility(source)

        # Display results
        status_color = "green" if assessment.credibility_level in ["high", "verified"] else "yellow"
        if assessment.credibility_level in ["low", "suspicious"]:
            status_color = "red"

        panel_content = f"""
[bold]URL:[/bold] {source.url}
[bold]Domain:[/bold] {source.domain}
[bold]Credibility:[/bold] [{status_color}]{assessment.credibility_level.upper()}[/{status_color}]
[bold]Confidence:[/bold] {assessment.confidence_score:.1%}
[bold]Recommendation:[/bold] {assessment.recommendation}
        """

        console.print(Panel(panel_content, title="Source Credibility Assessment"))

        # Show indicators
        if assessment.indicators:
            table = Table(title="Credibility Indicators")
            table.add_column("Indicator", style="cyan")
            table.add_column("Type", style="green")
            table.add_column("Details", style="white")

            for indicator in assessment.indicators:
                indicator_type = "✓ Positive" if indicator.positive else "✗ Negative"
                table.add_row(
                    indicator.indicator,
                    indicator_type,
                    indicator.details
                )

            console.print(table)

        # Show concerns
        if assessment.concerns:
            console.print("\n[yellow]Concerns:[/yellow]")
            for concern in assessment.concerns:
                console.print(f"  • {concern}")

    except Exception as e:
        console.print(f"[red]Error verifying source: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def batch_verify(
    input_file: Path = typer.Argument(..., help="JSON file with URLs to verify"),
    output: Optional[Path] = typer.Option(None, help="Output file path")
):
    """Batch verify multiple sources from a file"""

    if not input_file.exists():
        console.print(f"[red]Error: File not found: {input_file}[/red]")
        raise typer.Exit(1)

    try:
        with open(input_file, 'r') as f:
            data = json.load(f)

        urls = data if isinstance(data, list) else data.get('urls', [])

        console.print(f"[cyan]Verifying {len(urls)} sources...[/cyan]")

        verifier = SourceVerifier()
        results = []

        with console.status("[bold green]Verifying sources...") as status:
            for url in urls:
                if isinstance(url, dict):
                    source = Source(**url)
                else:
                    source = Source(url=url)

                assessment = verifier.verify_source_credibility(source)
                results.append({
                    'url': str(source.url),
                    'credibility': assessment.credibility_level,
                    'confidence': assessment.confidence_score,
                    'recommendation': assessment.recommendation
                })

        # Display summary
        table = Table(title="Verification Summary")
        table.add_column("Credibility Level", style="cyan")
        table.add_column("Count", style="green")

        from collections import Counter
        credibility_counts = Counter(r['credibility'] for r in results)

        for level, count in credibility_counts.items():
            table.add_row(level.upper(), str(count))

        console.print(table)

        # Save results if requested
        if output:
            with open(output, 'w') as f:
                json.dump(results, f, indent=2)
            console.print(f"[green]✓ Verification results saved to: {output}[/green]")

    except Exception as e:
        console.print(f"[red]Error during batch verification: {e}[/red]")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()