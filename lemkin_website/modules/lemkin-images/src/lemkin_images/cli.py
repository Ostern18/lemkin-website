"""
Command-line interface for Lemkin Image Verification Suite.
"""

import json
from pathlib import Path
from typing import Optional, List

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress

from .core import (
    ReverseImageSearcher,
    ManipulationDetector,
    GeolocationHelper,
    MetadataForensics,
    SearchEngine,
    ManipulationType,
    AuthenticityStatus,
)

app = typer.Typer(
    name="lemkin-images",
    help="Image verification suite for legal investigations",
    no_args_is_help=True
)
console = Console()


@app.command()
def reverse_search(
    image_path: Path = typer.Argument(..., help="Path to image file"),
    engines: str = typer.Option("google", help="Comma-separated search engines"),
    limit: int = typer.Option(20, help="Maximum results per engine"),
    output: Optional[Path] = typer.Option(None, help="Output file path")
):
    """Perform reverse image search"""

    if not image_path.exists():
        console.print(f"[red]Error: Image file not found: {image_path}[/red]")
        raise typer.Exit(1)

    try:
        # Parse engines
        engine_list = []
        for engine_name in engines.split(","):
            engine_name = engine_name.strip().lower()
            try:
                engine_list.append(SearchEngine(engine_name))
            except ValueError:
                console.print(f"[yellow]Warning: Unknown search engine: {engine_name}[/yellow]")

        if not engine_list:
            console.print("[red]No valid search engines specified[/red]")
            raise typer.Exit(1)

        console.print(f"[cyan]Performing reverse image search: {image_path}[/cyan]")
        console.print(f"Engines: {', '.join(e.value for e in engine_list)}")

        with Progress() as progress:
            task = progress.add_task("[green]Searching...", total=100)

            searcher = ReverseImageSearcher()
            results = searcher.reverse_search_image(image_path, engine_list, limit)

            progress.update(task, completed=100)

        # Display results
        panel_content = f"""
[bold]Image:[/bold] {image_path.name}
[bold]Search ID:[/bold] {results.search_id}
[bold]Query Hash:[/bold] {results.query_hash}
[bold]Results Found:[/bold] {results.total_results}
[bold]Primary Engine:[/bold] {results.search_engine.value}
        """

        console.print(Panel(panel_content, title="Reverse Search Results"))

        # Show results table
        if results.results:
            table = Table(title="Search Results")
            table.add_column("URL", style="cyan", max_width=50)
            table.add_column("Title", style="green", max_width=40)
            table.add_column("Domain", style="yellow")
            table.add_column("Similarity", style="white")

            for result in results.results:
                title = result.title[:37] + "..." if result.title and len(result.title) > 40 else (result.title or "No title")
                similarity = f"{result.similarity_score:.1%}" if result.similarity_score else "N/A"

                table.add_row(
                    str(result.url)[:47] + "..." if len(str(result.url)) > 50 else str(result.url),
                    title,
                    result.source_domain,
                    similarity
                )

            console.print(table)
        else:
            console.print("[yellow]No results found[/yellow]")

        # Save results if requested
        if output:
            with open(output, 'w') as f:
                json.dump(results.model_dump(mode='json'), f, indent=2, default=str)
            console.print(f"[green]✓ Results saved to: {output}[/green]")

    except Exception as e:
        console.print(f"[red]Error performing reverse search: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def detect_manipulation(
    image_path: Path = typer.Argument(..., help="Path to image file"),
    output: Optional[Path] = typer.Option(None, help="Output file path"),
    detailed: bool = typer.Option(False, help="Show detailed analysis")
):
    """Detect image manipulation and forgery"""

    if not image_path.exists():
        console.print(f"[red]Error: Image file not found: {image_path}[/red]")
        raise typer.Exit(1)

    try:
        console.print(f"[cyan]Detecting manipulation in: {image_path}[/cyan]")

        with Progress() as progress:
            task = progress.add_task("[green]Analyzing image...", total=100)

            detector = ManipulationDetector()
            analysis = detector.detect_image_manipulation(image_path)

            progress.update(task, completed=100)

        # Display results
        status_color = "red" if analysis.is_manipulated else "green"
        panel_content = f"""
[bold]Image:[/bold] {image_path.name}
[bold]Manipulated:[/bold] [{status_color}]{"YES" if analysis.is_manipulated else "NO"}[/{status_color}]
[bold]Confidence:[/bold] {analysis.confidence_score:.1%}
[bold]Manipulation Types:[/bold] {len(analysis.manipulation_types)}
[bold]Suspicious Regions:[/bold] {len(analysis.manipulation_regions)}
        """

        console.print(Panel(panel_content, title="Manipulation Detection Results"))

        # Show manipulation types
        if analysis.manipulation_types:
            console.print(f"\n[red]Manipulation Types Detected:[/red]")
            for manipulation_type in analysis.manipulation_types:
                console.print(f"  • {manipulation_type.value.replace('_', ' ').title()}")

        # Show detailed analysis if requested
        if detailed and analysis.analysis_details:
            table = Table(title="Analysis Details")
            table.add_column("Detection", style="cyan")
            table.add_column("Result", style="green")

            for key, value in analysis.analysis_details.items():
                result = "✓ Detected" if value else "✗ Not detected"
                table.add_row(key.replace('_', ' ').title(), result)

            console.print(table)

        # Show manipulation regions
        if analysis.manipulation_regions:
            console.print(f"\n[yellow]Suspicious Regions Found:[/yellow] {len(analysis.manipulation_regions)}")

            region_table = Table(title="Copy-Move Regions")
            region_table.add_column("Source", style="cyan")
            region_table.add_column("Target", style="yellow")
            region_table.add_column("Distance", style="green")

            for region in analysis.manipulation_regions[:10]:  # Show first 10
                source = f"({region['source_point'][0]:.0f}, {region['source_point'][1]:.0f})"
                target = f"({region['target_point'][0]:.0f}, {region['target_point'][1]:.0f})"
                distance = f"{region['distance']:.0f} px"

                region_table.add_row(source, target, distance)

            console.print(region_table)

        # Save results if requested
        if output:
            with open(output, 'w') as f:
                json.dump(analysis.model_dump(mode='json'), f, indent=2, default=str)
            console.print(f"[green]✓ Analysis saved to: {output}[/green]")

    except Exception as e:
        console.print(f"[red]Error detecting manipulation: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def geolocate(
    image_path: Path = typer.Argument(..., help="Path to image file"),
    output: Optional[Path] = typer.Option(None, help="Output file path"),
    show_features: bool = typer.Option(True, help="Show visual features")
):
    """Attempt to geolocate image from visual content"""

    if not image_path.exists():
        console.print(f"[red]Error: Image file not found: {image_path}[/red]")
        raise typer.Exit(1)

    try:
        console.print(f"[cyan]Geolocating image: {image_path}[/cyan]")

        with Progress() as progress:
            task = progress.add_task("[green]Analyzing location clues...", total=100)

            helper = GeolocationHelper()
            result = helper.geolocate_image(image_path)

            progress.update(task, completed=100)

        # Display results
        location_str = "Unknown"
        if result.estimated_location:
            lat, lon = result.estimated_location
            location_str = f"{lat:.6f}, {lon:.6f}"

        metadata_str = "Not available"
        if result.metadata_location:
            lat, lon = result.metadata_location
            metadata_str = f"{lat:.6f}, {lon:.6f}"

        panel_content = f"""
[bold]Image:[/bold] {image_path.name}
[bold]Estimated Location:[/bold] {location_str}
[bold]Confidence:[/bold] {result.location_confidence:.1%}
[bold]Metadata Location:[/bold] {metadata_str}
[bold]Visual Features:[/bold] {len(result.visual_features)}
[bold]Landmark Matches:[/bold] {len(result.landmark_matches)}
        """

        console.print(Panel(panel_content, title="Geolocation Results"))

        # Show visual features
        if show_features and result.visual_features:
            console.print(f"\n[cyan]Visual Features Detected:[/cyan]")
            for feature in result.visual_features:
                console.print(f"  • {feature.replace('_', ' ').title()}")

        # Show landmark matches
        if result.landmark_matches:
            table = Table(title="Landmark Matches")
            table.add_column("Landmark", style="cyan")
            table.add_column("Location", style="green")
            table.add_column("Confidence", style="yellow")

            for match in result.landmark_matches:
                table.add_row(
                    match.get('name', 'Unknown'),
                    f"{match.get('latitude', 0):.4f}, {match.get('longitude', 0):.4f}",
                    f"{match.get('confidence', 0):.1%}"
                )

            console.print(table)

        # Show recommendations
        if result.location_confidence < 0.5:
            console.print(f"\n[yellow]Recommendations:[/yellow]")
            console.print("  • Low confidence - consider additional verification")
            console.print("  • Check for text/signs that might provide location clues")
            console.print("  • Cross-reference with witness statements or other evidence")

        # Save results if requested
        if output:
            with open(output, 'w') as f:
                json.dump(result.model_dump(mode='json'), f, indent=2, default=str)
            console.print(f"[green]✓ Results saved to: {output}[/green]")

    except Exception as e:
        console.print(f"[red]Error geolocating image: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def analyze_metadata(
    image_path: Path = typer.Argument(..., help="Path to image file"),
    output: Optional[Path] = typer.Option(None, help="Output file path"),
    show_exif: bool = typer.Option(False, help="Show all EXIF data")
):
    """Analyze image metadata for forensic information"""

    if not image_path.exists():
        console.print(f"[red]Error: Image file not found: {image_path}[/red]")
        raise typer.Exit(1)

    try:
        console.print(f"[cyan]Analyzing metadata: {image_path}[/cyan]")

        forensics = MetadataForensics()
        metadata = forensics.analyze_image_metadata(image_path)

        # Display basic info
        creation_str = metadata.creation_date.strftime("%Y-%m-%d %H:%M:%S") if metadata.creation_date else "Unknown"
        modification_str = metadata.modification_date.strftime("%Y-%m-%d %H:%M:%S") if metadata.modification_date else "Unknown"
        gps_str = f"{metadata.gps_coordinates[0]:.6f}, {metadata.gps_coordinates[1]:.6f}" if metadata.gps_coordinates else "Not available"

        panel_content = f"""
[bold]File:[/bold] {metadata.file_path.name}
[bold]Size:[/bold] {metadata.file_size:,} bytes
[bold]Dimensions:[/bold] {metadata.dimensions[0]} × {metadata.dimensions[1]}
[bold]Format:[/bold] {metadata.format}
[bold]Mode:[/bold] {metadata.mode}
[bold]Created:[/bold] {creation_str}
[bold]Modified:[/bold] {modification_str}
[bold]Camera:[/bold] {metadata.camera_make or 'Unknown'} {metadata.camera_model or ''}
[bold]GPS Coordinates:[/bold] {gps_str}
[bold]File Hash:[/bold] {metadata.file_hash[:32]}...
        """

        console.print(Panel(panel_content, title="Image Metadata Analysis"))

        # Show EXIF data if requested
        if show_exif and metadata.exif_data:
            table = Table(title="EXIF Data")
            table.add_column("Tag", style="cyan")
            table.add_column("Value", style="green", max_width=50)

            # Show most important EXIF tags first
            important_tags = [
                'DateTime', 'DateTimeOriginal', 'DateTimeDigitized',
                'Make', 'Model', 'Software',
                'ExposureTime', 'FNumber', 'ISO',
                'FocalLength', 'WhiteBalance',
                'ImageWidth', 'ImageLength'
            ]

            # Show important tags first
            shown_tags = set()
            for tag in important_tags:
                if tag in metadata.exif_data:
                    value = metadata.exif_data[tag]
                    if len(str(value)) > 50:
                        value = str(value)[:47] + "..."
                    table.add_row(tag, str(value))
                    shown_tags.add(tag)

            # Show remaining tags
            remaining_count = 0
            for tag, value in metadata.exif_data.items():
                if tag not in shown_tags and remaining_count < 10:  # Limit additional tags
                    if len(str(value)) > 50:
                        value = str(value)[:47] + "..."
                    table.add_row(tag, str(value))
                    remaining_count += 1

            console.print(table)

            if len(metadata.exif_data) > len(shown_tags) + remaining_count:
                remaining = len(metadata.exif_data) - len(shown_tags) - remaining_count
                console.print(f"[dim]... and {remaining} more EXIF tags[/dim]")

        # Show forensic indicators
        console.print(f"\n[cyan]Forensic Indicators:[/cyan]")

        # Check for potential issues
        issues = []
        if metadata.creation_date and metadata.modification_date:
            time_diff = (metadata.modification_date - metadata.creation_date).total_seconds()
            if time_diff < 0:
                issues.append("Modification date before creation date")
            elif time_diff > 86400 * 30:  # More than 30 days
                issues.append("Long gap between creation and modification")

        if not metadata.camera_make and not metadata.camera_model:
            issues.append("No camera information in metadata")

        if metadata.file_size < 50000:  # Less than 50KB
            issues.append("Unusually small file size for image dimensions")

        if issues:
            for issue in issues:
                console.print(f"  [yellow]⚠[/yellow] {issue}")
        else:
            console.print("  [green]✓[/green] No obvious metadata inconsistencies detected")

        # Save results if requested
        if output:
            with open(output, 'w') as f:
                json.dump(metadata.model_dump(mode='json'), f, indent=2, default=str)
            console.print(f"[green]✓ Metadata saved to: {output}[/green]")

    except Exception as e:
        console.print(f"[red]Error analyzing metadata: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def comprehensive_analysis(
    image_path: Path = typer.Argument(..., help="Path to image file"),
    output: Optional[Path] = typer.Option(None, help="Output file path"),
    include_search: bool = typer.Option(False, help="Include reverse image search"),
    include_geolocation: bool = typer.Option(False, help="Include geolocation analysis")
):
    """Perform comprehensive image verification analysis"""

    if not image_path.exists():
        console.print(f"[red]Error: Image file not found: {image_path}[/red]")
        raise typer.Exit(1)

    try:
        console.print(f"[cyan]Performing comprehensive analysis: {image_path}[/cyan]")

        import uuid
        from .core import ImageAnalysis

        analysis_id = str(uuid.uuid4())

        with Progress() as progress:
            main_task = progress.add_task("[green]Running analysis...", total=100)

            # Metadata analysis
            progress.update(main_task, completed=20)
            forensics = MetadataForensics()
            metadata_analysis = forensics.analyze_image_metadata(image_path)

            # Manipulation detection
            progress.update(main_task, completed=40)
            detector = ManipulationDetector()
            manipulation_analysis = detector.detect_image_manipulation(image_path)

            # Optional reverse search
            reverse_search = None
            if include_search:
                progress.update(main_task, completed=60)
                searcher = ReverseImageSearcher()
                reverse_search = searcher.reverse_search_image(image_path, [SearchEngine.GOOGLE], 10)

            # Optional geolocation
            geolocation_analysis = None
            if include_geolocation:
                progress.update(main_task, completed=80)
                helper = GeolocationHelper()
                geolocation_analysis = helper.geolocate_image(image_path)

            progress.update(main_task, completed=100)

        # Determine overall authenticity status
        if manipulation_analysis.is_manipulated:
            if ManipulationType.AIGENERATED in manipulation_analysis.manipulation_types:
                authenticity_status = AuthenticityStatus.AI_GENERATED
            else:
                authenticity_status = AuthenticityStatus.MANIPULATED
        elif manipulation_analysis.confidence_score > 0.3:
            authenticity_status = AuthenticityStatus.SUSPICIOUS
        else:
            authenticity_status = AuthenticityStatus.AUTHENTIC

        # Calculate overall confidence
        confidence_factors = [manipulation_analysis.confidence_score]
        if reverse_search and reverse_search.results:
            confidence_factors.append(0.7)  # Finding results increases confidence
        if geolocation_analysis and geolocation_analysis.location_confidence > 0.5:
            confidence_factors.append(geolocation_analysis.location_confidence)

        overall_confidence = sum(confidence_factors) / len(confidence_factors)

        # Generate findings and recommendations
        findings = []
        recommendations = []

        if manipulation_analysis.is_manipulated:
            findings.append(f"Image shows signs of manipulation: {', '.join(t.value for t in manipulation_analysis.manipulation_types)}")
            recommendations.append("Verify with original source and examine metadata carefully")

        if metadata_analysis.gps_coordinates:
            findings.append("GPS coordinates found in metadata")
        else:
            findings.append("No GPS coordinates in metadata")
            recommendations.append("Consider geolocation analysis from visual content")

        if reverse_search and reverse_search.results:
            findings.append(f"Found {len(reverse_search.results)} similar images online")
            recommendations.append("Examine reverse search results for context and original source")

        # Create comprehensive analysis
        comprehensive = ImageAnalysis(
            analysis_id=analysis_id,
            image_path=image_path,
            authenticity_status=authenticity_status,
            confidence_score=overall_confidence,
            metadata_analysis=metadata_analysis,
            manipulation_analysis=manipulation_analysis,
            reverse_search=reverse_search,
            geolocation_analysis=geolocation_analysis,
            findings=findings,
            recommendations=recommendations
        )

        # Display summary
        status_color = "green" if authenticity_status == AuthenticityStatus.AUTHENTIC else "red"
        panel_content = f"""
[bold]Image:[/bold] {image_path.name}
[bold]Authenticity Status:[/bold] [{status_color}]{authenticity_status.value.upper()}[/{status_color}]
[bold]Overall Confidence:[/bold] {overall_confidence:.1%}
[bold]Manipulation Detected:[/bold] {"YES" if manipulation_analysis.is_manipulated else "NO"}
[bold]GPS Available:[/bold] {"YES" if metadata_analysis.gps_coordinates else "NO"}
[bold]Reverse Search Results:[/bold] {len(reverse_search.results) if reverse_search else "Not performed"}
[bold]Location Estimated:[/bold] {"YES" if geolocation_analysis and geolocation_analysis.estimated_location else "NO"}
        """

        console.print(Panel(panel_content, title="Comprehensive Image Analysis"))

        # Show key findings
        if findings:
            console.print(f"\n[cyan]Key Findings:[/cyan]")
            for finding in findings:
                console.print(f"  • {finding}")

        # Show recommendations
        if recommendations:
            console.print(f"\n[yellow]Recommendations:[/yellow]")
            for recommendation in recommendations:
                console.print(f"  • {recommendation}")

        # Save comprehensive results
        if output:
            with open(output, 'w') as f:
                json.dump(comprehensive.model_dump(mode='json'), f, indent=2, default=str)
            console.print(f"[green]✓ Comprehensive analysis saved to: {output}[/green]")

    except Exception as e:
        console.print(f"[red]Error in comprehensive analysis: {e}[/red]")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()