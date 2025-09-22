"""
Command-line interface for Lemkin Geospatial Analysis Suite.
"""

import json
from pathlib import Path
from typing import Optional, List
from datetime import datetime

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from .core import (
    CoordinateConverter,
    SatelliteAnalyzer,
    GeofenceProcessor,
    MappingGenerator,
    StandardCoordinate,
    BoundingBox,
    DateRange,
    Event,
    Evidence,
    EventType,
    CoordinateFormat,
)

app = typer.Typer(
    name="lemkin-geo",
    help="Geospatial analysis suite for legal investigations",
    no_args_is_help=True
)
console = Console()


@app.command()
def convert_coords(
    coordinates: str = typer.Argument(..., help="Coordinates to convert"),
    input_format: str = typer.Option("decimal", help="Input format: decimal, dms, ddm, utm"),
    geocode: bool = typer.Option(False, help="Also perform reverse geocoding")
):
    """Convert coordinates between different formats"""

    try:
        converter = CoordinateConverter()
        coord_format = CoordinateFormat(input_format.lower())

        # Standardize coordinates
        standard = converter.standardize_coordinates(coordinates, coord_format)

        # Display results
        panel_content = f"""
[bold]Standardized Coordinates (WGS84):[/bold]
Latitude: {standard.latitude:.6f}
Longitude: {standard.longitude:.6f}
Original Format: {standard.original_format.value}
        """

        # Add reverse geocoding if requested
        if geocode:
            address = converter.reverse_geocode(standard)
            if address:
                panel_content += f"\n[bold]Address:[/bold] {address}"

        console.print(Panel(panel_content, title="Coordinate Conversion"))

        # Show in different formats
        table = Table(title="Alternative Formats")
        table.add_column("Format", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Decimal", f"{standard.latitude:.6f}, {standard.longitude:.6f}")

        # DMS format
        lat_dms = _decimal_to_dms(standard.latitude, 'lat')
        lon_dms = _decimal_to_dms(standard.longitude, 'lon')
        table.add_row("DMS", f"{lat_dms} {lon_dms}")

        console.print(table)

    except Exception as e:
        console.print(f"[red]Error converting coordinates: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def geocode(
    address: str = typer.Argument(..., help="Address to geocode"),
    output: Optional[Path] = typer.Option(None, help="Output file path")
):
    """Geocode an address to coordinates"""

    try:
        converter = CoordinateConverter()
        coordinate = converter.geocode_address(address)

        if coordinate:
            console.print(f"[green]✓ Successfully geocoded address[/green]")

            panel_content = f"""
[bold]Address:[/bold] {address}
[bold]Latitude:[/bold] {coordinate.latitude:.6f}
[bold]Longitude:[/bold] {coordinate.longitude:.6f}
[bold]Coordinate System:[/bold] {coordinate.coordinate_system}
            """

            console.print(Panel(panel_content, title="Geocoding Result"))

            # Save to file if requested
            if output:
                with open(output, 'w') as f:
                    json.dump(coordinate.model_dump(), f, indent=2)
                console.print(f"[green]✓ Coordinates saved to: {output}[/green]")

        else:
            console.print(f"[yellow]Could not geocode address: {address}[/yellow]")

    except Exception as e:
        console.print(f"[red]Error geocoding address: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def analyze_satellite(
    min_lat: float = typer.Option(..., help="Minimum latitude"),
    max_lat: float = typer.Option(..., help="Maximum latitude"),
    min_lon: float = typer.Option(..., help="Minimum longitude"),
    max_lon: float = typer.Option(..., help="Maximum longitude"),
    start_date: str = typer.Option(..., help="Start date (YYYY-MM-DD)"),
    end_date: str = typer.Option(..., help="End date (YYYY-MM-DD)"),
    output: Optional[Path] = typer.Option(None, help="Output file path")
):
    """Analyze satellite imagery for an area"""

    try:
        # Parse dates
        start = datetime.fromisoformat(start_date)
        end = datetime.fromisoformat(end_date)

        # Create bounding box and date range
        bbox = BoundingBox(
            min_lat=min_lat,
            max_lat=max_lat,
            min_lon=min_lon,
            max_lon=max_lon
        )
        date_range = DateRange(start_date=start, end_date=end)

        # Perform analysis
        analyzer = SatelliteAnalyzer()
        analysis = analyzer.analyze_satellite_imagery(bbox, date_range)

        # Display results
        panel_content = f"""
[bold]Analysis ID:[/bold] {analysis.analysis_id}
[bold]Area:[/bold] {bbox.min_lat:.3f},{bbox.min_lon:.3f} to {bbox.max_lat:.3f},{bbox.max_lon:.3f}
[bold]Period:[/bold] {start_date} to {end_date}
[bold]Images Found:[/bold] {analysis.images_found}
[bold]Changes Detected:[/bold] {len(analysis.changes_detected)}
        """

        console.print(Panel(panel_content, title="Satellite Analysis Results"))

        # Show detected changes
        if analysis.changes_detected:
            table = Table(title="Detected Changes")
            table.add_column("Type", style="cyan")
            table.add_column("Location", style="green")
            table.add_column("Confidence", style="yellow")

            for change in analysis.changes_detected:
                table.add_row(
                    change.get("type", "Unknown"),
                    f"{change['location']['lat']:.4f}, {change['location']['lon']:.4f}",
                    f"{change.get('confidence', 0):.1%}"
                )

            console.print(table)

        # Save to file if requested
        if output:
            with open(output, 'w') as f:
                json.dump(analysis.model_dump(mode='json'), f, indent=2, default=str)
            console.print(f"[green]✓ Analysis saved to: {output}[/green]")

    except Exception as e:
        console.print(f"[red]Error analyzing satellite imagery: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def correlate_events(
    input_file: Path = typer.Argument(..., help="JSON file with events"),
    radius: float = typer.Option(1000.0, help="Correlation radius in meters"),
    output: Optional[Path] = typer.Option(None, help="Output file path")
):
    """Correlate events by geographic proximity"""

    if not input_file.exists():
        console.print(f"[red]Error: File not found: {input_file}[/red]")
        raise typer.Exit(1)

    try:
        # Load events
        with open(input_file, 'r') as f:
            data = json.load(f)

        # Parse events
        events = []
        for item in data:
            # Handle both Event objects and simple dicts
            if 'location' in item:
                loc_data = item['location']
                if isinstance(loc_data, dict):
                    location = StandardCoordinate(**loc_data)
                else:
                    location = StandardCoordinate(
                        latitude=loc_data[0],
                        longitude=loc_data[1]
                    )

                event = Event(
                    event_id=item.get('event_id', f"event_{len(events)}"),
                    event_type=EventType(item.get('event_type', 'incident')),
                    location=location,
                    timestamp=datetime.fromisoformat(item['timestamp']) if isinstance(item['timestamp'], str) else item['timestamp'],
                    description=item.get('description', '')
                )
                events.append(event)

        console.print(f"[cyan]Correlating {len(events)} events...[/cyan]")

        # Correlate events
        processor = GeofenceProcessor()
        correlation = processor.correlate_events_by_location(events, radius)

        # Display results
        panel_content = f"""
[bold]Correlation ID:[/bold] {correlation.correlation_id}
[bold]Total Events:[/bold] {len(correlation.events)}
[bold]Radius:[/bold] {correlation.radius_meters} meters
[bold]Clusters Found:[/bold] {len(correlation.clusters)}
[bold]Correlation Strength:[/bold] {correlation.correlation_strength:.1%}
        """

        console.print(Panel(panel_content, title="Event Correlation Results"))

        # Show clusters
        if correlation.clusters:
            table = Table(title="Event Clusters")
            table.add_column("Cluster", style="cyan")
            table.add_column("Events", style="green")
            table.add_column("Time Span", style="yellow")

            for cluster in correlation.clusters:
                time_span = f"{cluster['time_span']['start'][:10]} to {cluster['time_span']['end'][:10]}"
                table.add_row(
                    cluster['cluster_id'],
                    str(cluster['size']),
                    time_span
                )

            console.print(table)

        # Save to file if requested
        if output:
            with open(output, 'w') as f:
                json.dump(correlation.model_dump(mode='json'), f, indent=2, default=str)
            console.print(f"[green]✓ Correlation saved to: {output}[/green]")

    except Exception as e:
        console.print(f"[red]Error correlating events: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def create_map(
    input_file: Path = typer.Argument(..., help="JSON file with evidence"),
    output: Path = typer.Option("evidence_map.html", help="Output HTML file"),
    title: str = typer.Option("Evidence Map", help="Map title")
):
    """Create interactive map from evidence data"""

    if not input_file.exists():
        console.print(f"[red]Error: File not found: {input_file}[/red]")
        raise typer.Exit(1)

    try:
        # Load evidence
        with open(input_file, 'r') as f:
            data = json.load(f)

        # Parse evidence
        evidence_list = []
        for item in data:
            # Handle location data
            location = None
            locations = []

            if 'location' in item and item['location']:
                loc_data = item['location']
                if isinstance(loc_data, dict):
                    location = StandardCoordinate(**loc_data)
                else:
                    location = StandardCoordinate(
                        latitude=loc_data[0],
                        longitude=loc_data[1]
                    )

            if 'locations' in item:
                for loc_data in item['locations']:
                    if isinstance(loc_data, dict):
                        locations.append(StandardCoordinate(**loc_data))
                    else:
                        locations.append(StandardCoordinate(
                            latitude=loc_data[0],
                            longitude=loc_data[1]
                        ))

            evidence = Evidence(
                evidence_id=item.get('evidence_id', f"evidence_{len(evidence_list)}"),
                location=location,
                locations=locations,
                timestamp=datetime.fromisoformat(item['timestamp']) if 'timestamp' in item and isinstance(item['timestamp'], str) else None,
                description=item.get('description', 'No description'),
                evidence_type=item.get('evidence_type', 'unknown')
            )
            evidence_list.append(evidence)

        console.print(f"[cyan]Creating map with {len(evidence_list)} evidence items...[/cyan]")

        # Generate map
        generator = MappingGenerator()
        interactive_map = generator.generate_evidence_map(evidence_list, output)

        # Display summary
        located_count = sum(1 for e in evidence_list if e.location or e.locations)

        panel_content = f"""
[bold]Map ID:[/bold] {interactive_map.map_id}
[bold]Total Evidence:[/bold] {len(evidence_list)}
[bold]Located Evidence:[/bold] {located_count}
[bold]Center:[/bold] {interactive_map.center.latitude:.4f}, {interactive_map.center.longitude:.4f}
[bold]Zoom Level:[/bold] {interactive_map.zoom_level}
[bold]Output:[/bold] {output}
        """

        console.print(Panel(panel_content, title="Map Generation Complete"))
        console.print(f"[green]✓ Interactive map saved to: {output}[/green]")
        console.print(f"[cyan]Open {output} in a web browser to view the map[/cyan]")

    except Exception as e:
        console.print(f"[red]Error creating map: {e}[/red]")
        raise typer.Exit(1)


def _decimal_to_dms(decimal: float, coord_type: str) -> str:
    """Convert decimal degrees to DMS format"""
    is_negative = decimal < 0
    decimal = abs(decimal)

    degrees = int(decimal)
    minutes_decimal = (decimal - degrees) * 60
    minutes = int(minutes_decimal)
    seconds = (minutes_decimal - minutes) * 60

    if coord_type == 'lat':
        direction = 'S' if is_negative else 'N'
    else:
        direction = 'W' if is_negative else 'E'

    return f"{degrees}°{minutes}'{seconds:.1f}\"{direction}"


if __name__ == "__main__":
    app()