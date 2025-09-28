"""
Lemkin Geospatial Analysis Suite - CLI Module

Command-line interface for geospatial analysis tools designed for legal professionals.
Provides user-friendly commands for coordinate conversion, satellite analysis, 
location correlation, and interactive mapping without requiring GIS expertise.

Commands:
- convert-coordinates: Convert between coordinate formats
- analyze-satellite: Satellite imagery analysis
- correlate-locations: Location-based event correlation  
- generate-map: Create interactive evidence maps
- create-geofence: Set up location monitoring zones
"""

import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional
import logging

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich import print as rprint

from .core import (
    GeospatialAnalyzer, 
    GeoConfig, 
    Coordinate, 
    BoundingBox, 
    DateRange, 
    Evidence,
    SpatialEvent,
    EvidenceType,
    CoordinateFormat,
    SatelliteProvider
)
from .coordinate_converter import CoordinateConverter, standardize_coordinates
from .satellite_analyzer import SatelliteAnalyzer, analyze_satellite_imagery
from .geofence_processor import GeofenceProcessor, correlate_events_by_location, GeofenceType
from .mapping_generator import MappingGenerator, generate_evidence_map, ExportFormat

# Initialize Rich console
console = Console()

# Create Typer app
app = typer.Typer(
    name="lemkin-geo",
    help="Lemkin Geospatial Analysis Suite - Geographic analysis for legal evidence",
    add_completion=False
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@app.command()
def convert_coordinates(
    coords: str = typer.Argument(..., help="Coordinates to convert"),
    input_format: Optional[str] = typer.Option(None, "--input-format", "-i", help="Input coordinate format (dd, dms, utm, mgrs)"),
    output_format: str = typer.Option("dd", "--output-format", "-o", help="Output coordinate format"),
    output_file: Optional[Path] = typer.Option(None, "--output", "-f", help="Output file path"),
    precision: int = typer.Option(6, "--precision", "-p", help="Decimal precision for output"),
    validate: bool = typer.Option(True, "--validate/--no-validate", help="Validate coordinate precision")
):
    """
    Convert coordinates between different formats (DD, DMS, UTM, MGRS).
    
    Examples:
        lemkin-geo convert-coordinates "40.7128, -74.0060"
        lemkin-geo convert-coordinates "40°42'46.0\"N 74°00'21.6\"W" -o dms
        lemkin-geo convert-coordinates "18T 585628 4511322" -i utm -o dd
    """
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Converting coordinates...", total=None)
            
            converter = CoordinateConverter()
            result = converter.standardize_coordinates(coords, input_format)
            
            if not result.success:
                console.print(f"[red]Error:[/red] {result.error_message}")
                raise typer.Exit(1)
            
            # Convert to output format if different from decimal degrees
            if output_format != "dd":
                try:
                    target_format = CoordinateFormat(output_format)
                    conversion_result = converter.convert_to_format(result.coordinate, target_format)
                    if not conversion_result.success:
                        console.print(f"[red]Conversion Error:[/red] {conversion_result.error_message}")
                        raise typer.Exit(1)
                    result = conversion_result
                except ValueError:
                    console.print(f"[red]Error:[/red] Unsupported output format: {output_format}")
                    raise typer.Exit(1)
            
            progress.update(task, description="Validating precision...")
            
            # Validate precision if requested
            validation_info = None
            if validate and result.coordinate:
                validation_info = converter.validate_coordinate_precision(result.coordinate)
            
            progress.remove_task(task)
        
        # Display results
        console.print("\n[bold green]Coordinate Conversion Results[/bold green]")
        
        if result.coordinate:
            # Create results table
            table = Table(show_header=True, header_style="bold blue")
            table.add_column("Property")
            table.add_column("Value")
            
            table.add_row("Original Format", result.original_format or "Auto-detected")
            table.add_row("Target Format", result.target_format)
            table.add_row("Latitude", f"{result.coordinate.latitude:.{precision}f}")
            table.add_row("Longitude", f"{result.coordinate.longitude:.{precision}f}")
            
            if result.coordinate.altitude:
                table.add_row("Altitude (m)", f"{result.coordinate.altitude}")
            
            console.print(table)
            
            # Display validation info
            if validation_info:
                console.print(f"\n[bold]Precision Validation:[/bold]")
                meets_req = "✓" if validation_info['meets_precision'] else "✗"
                console.print(f"  Meets 1m precision: {meets_req}")
                console.print(f"  Estimated precision: {validation_info['estimated_precision_meters']:.2f} meters")
                console.print(f"  Decimal places: {validation_info['decimal_places']}")
                
                if validation_info['recommendations']:
                    console.print(f"\n[yellow]Recommendations:[/yellow]")
                    for rec in validation_info['recommendations']:
                        console.print(f"  • {rec}")
            
            # Save to file if requested
            if output_file:
                output_data = {
                    'original_input': coords,
                    'converted_coordinate': {
                        'latitude': result.coordinate.latitude,
                        'longitude': result.coordinate.longitude,
                        'format': result.target_format,
                        'precision_meters': validation_info.get('estimated_precision_meters') if validation_info else None
                    },
                    'conversion_metadata': {
                        'original_format': result.original_format,
                        'target_format': result.target_format,
                        'converted_at': datetime.utcnow().isoformat(),
                        'tool_version': '1.0.0'
                    }
                }
                
                output_file.parent.mkdir(parents=True, exist_ok=True)
                with open(output_file, 'w') as f:
                    json.dump(output_data, f, indent=2, default=str)
                
                console.print(f"\n[green]Results saved to:[/green] {output_file}")
        
    except Exception as e:
        console.print(f"[red]Unexpected error:[/red] {str(e)}")
        raise typer.Exit(1)


@app.command()
def analyze_satellite(
    bbox: str = typer.Argument(..., help="Bounding box as 'north,south,east,west'"),
    start_date: str = typer.Option(..., "--start", "-s", help="Start date (YYYY-MM-DD)"),
    end_date: str = typer.Option(..., "--end", "-e", help="End date (YYYY-MM-DD)"),
    provider: str = typer.Option("landsat", "--provider", "-p", help="Satellite provider (landsat, sentinel, modis)"),
    cloud_cover: float = typer.Option(10.0, "--cloud-cover", "-c", help="Maximum cloud cover percentage"),
    analysis_type: str = typer.Option("change_detection", "--analysis-type", "-a", help="Analysis type"),
    output_file: Optional[Path] = typer.Option(None, "--output", "-f", help="Output file path")
):
    """
    Analyze satellite imagery for a specified area and time period.
    
    Examples:
        lemkin-geo analyze-satellite "40.8,40.6,-73.9,-74.1" --start 2024-01-01 --end 2024-01-31
        lemkin-geo analyze-satellite "45.5,45.3,2.3,2.5" --provider sentinel --cloud-cover 5
    """
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Analyzing satellite imagery...", total=None)
            
            # Parse bounding box
            try:
                bbox_values = [float(x.strip()) for x in bbox.split(',')]
                if len(bbox_values) != 4:
                    raise ValueError("Bounding box must have 4 values")
                
                bounding_box = BoundingBox(
                    north=bbox_values[0],
                    south=bbox_values[1], 
                    east=bbox_values[2],
                    west=bbox_values[3]
                )
            except (ValueError, IndexError):
                console.print(f"[red]Error:[/red] Invalid bounding box format. Use: north,south,east,west")
                raise typer.Exit(1)
            
            progress.update(task, description="Parsing date range...")
            
            # Parse dates
            try:
                start_dt = datetime.strptime(start_date, "%Y-%m-%d")
                end_dt = datetime.strptime(end_date, "%Y-%m-%d")
                date_range = DateRange(start_date=start_dt, end_date=end_dt)
            except ValueError:
                console.print(f"[red]Error:[/red] Invalid date format. Use YYYY-MM-DD")
                raise typer.Exit(1)
            
            # Parse provider
            try:
                sat_provider = SatelliteProvider(provider.lower())
            except ValueError:
                console.print(f"[red]Error:[/red] Unsupported provider: {provider}")
                console.print("Supported providers: landsat, sentinel, modis")
                raise typer.Exit(1)
            
            progress.update(task, description="Searching satellite imagery...")
            
            # Create analyzer with custom config
            config = GeoConfig(
                preferred_satellite=sat_provider,
                cloud_cover_threshold=cloud_cover
            )
            analyzer = SatelliteAnalyzer(config)
            
            progress.update(task, description="Performing analysis...")
            
            # Perform analysis
            analysis = analyzer.analyze_satellite_imagery(
                bbox=bounding_box,
                date_range=date_range,
                provider=sat_provider,
                analysis_type=analysis_type
            )
            
            progress.remove_task(task)
        
        # Display results
        console.print("\n[bold green]Satellite Imagery Analysis Results[/bold green]")
        
        # Analysis summary table
        summary_table = Table(show_header=True, header_style="bold blue")
        summary_table.add_column("Property")
        summary_table.add_column("Value")
        
        summary_table.add_row("Analysis Name", analysis.analysis_name)
        summary_table.add_row("Provider", analysis.satellite_provider)
        summary_table.add_row("Images Analyzed", str(len(analysis.images_analyzed)))
        summary_table.add_row("Cloud Cover", f"{analysis.cloud_cover_percentage:.1f}%" if analysis.cloud_cover_percentage else "N/A")
        summary_table.add_row("Resolution", f"{analysis.resolution_meters}m" if analysis.resolution_meters else "N/A")
        summary_table.add_row("Area", f"{bounding_box.area_km2():.2f} km²")
        
        console.print(summary_table)
        
        # Display changes detected
        if analysis.changes_detected:
            console.print(f"\n[bold]Changes Detected:[/bold] {len(analysis.changes_detected)}")
            
            changes_table = Table(show_header=True, header_style="bold yellow")
            changes_table.add_column("Change Type")
            changes_table.add_column("Confidence")
            changes_table.add_column("Location")
            changes_table.add_column("Description")
            
            for change in analysis.changes_detected:
                if isinstance(change, dict) and 'type' in change:
                    location_str = "N/A"
                    if 'location' in change and isinstance(change['location'], dict):
                        loc = change['location']
                        location_str = f"{loc.get('latitude', 0):.4f}, {loc.get('longitude', 0):.4f}"
                    
                    changes_table.add_row(
                        change.get('type', 'Unknown'),
                        f"{change.get('confidence', 0):.2f}",
                        location_str,
                        change.get('description', '')
                    )
            
            console.print(changes_table)
        
        # Display points of interest
        if analysis.points_of_interest:
            console.print(f"\n[bold]Points of Interest:[/bold] {len(analysis.points_of_interest)}")
            for i, poi in enumerate(analysis.points_of_interest, 1):
                console.print(f"  {i}. {poi.latitude:.6f}, {poi.longitude:.6f}")
        
        # Display area measurements
        if analysis.area_measurements:
            console.print(f"\n[bold]Area Measurements:[/bold]")
            for area_type, area_value in analysis.area_measurements.items():
                console.print(f"  {area_type}: {area_value:.2f} m²")
        
        # Save results if requested
        if output_file:
            output_file.parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, 'w') as f:
                json.dump(analysis.dict(), f, indent=2, default=str)
            
            console.print(f"\n[green]Analysis results saved to:[/green] {output_file}")
        
    except Exception as e:
        console.print(f"[red]Unexpected error:[/red] {str(e)}")
        raise typer.Exit(1)


@app.command()
def correlate_locations(
    evidence_file: Path = typer.Argument(..., help="JSON file containing evidence data"),
    radius: float = typer.Option(100.0, "--radius", "-r", help="Search radius in meters"),
    method: str = typer.Option("distance_based", "--method", "-m", help="Correlation method"),
    time_window: Optional[int] = typer.Option(None, "--time-window", "-t", help="Time window in hours"),
    output_file: Optional[Path] = typer.Option(None, "--output", "-f", help="Output file path")
):
    """
    Correlate evidence and events based on location proximity.
    
    Examples:
        lemkin-geo correlate-locations evidence.json --radius 500
        lemkin-geo correlate-locations events.json --method temporal_spatial --time-window 24
    """
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Loading evidence data...", total=None)
            
            # Load evidence data
            if not evidence_file.exists():
                console.print(f"[red]Error:[/red] Evidence file not found: {evidence_file}")
                raise typer.Exit(1)
            
            try:
                with open(evidence_file, 'r') as f:
                    evidence_data = json.load(f)
            except json.JSONDecodeError as e:
                console.print(f"[red]Error:[/red] Invalid JSON file: {e}")
                raise typer.Exit(1)
            
            progress.update(task, description="Processing evidence...")
            
            # Convert evidence data to Evidence objects
            evidence_items = []
            if isinstance(evidence_data, list):
                for item in evidence_data:
                    try:
                        # Create Evidence object from dict
                        evidence = Evidence(**item)
                        evidence_items.append(evidence)
                    except Exception as e:
                        console.print(f"[yellow]Warning:[/yellow] Skipping invalid evidence item: {e}")
                        continue
            elif isinstance(evidence_data, dict) and 'evidence' in evidence_data:
                for item in evidence_data['evidence']:
                    try:
                        evidence = Evidence(**item)
                        evidence_items.append(evidence)
                    except Exception as e:
                        console.print(f"[yellow]Warning:[/yellow] Skipping invalid evidence item: {e}")
                        continue
            else:
                console.print(f"[red]Error:[/red] Invalid evidence file format")
                raise typer.Exit(1)
            
            located_items = [e for e in evidence_items if e.location]
            console.print(f"Loaded {len(evidence_items)} evidence items ({len(located_items)} with location data)")
            
            if len(located_items) < 2:
                console.print(f"[yellow]Warning:[/yellow] Need at least 2 items with location data for correlation")
                raise typer.Exit(1)
            
            progress.update(task, description="Performing location correlation...")
            
            # Create processor and perform correlation
            processor = GeofenceProcessor()
            
            from .geofence_processor import CorrelationMethod
            try:
                correlation_method = CorrelationMethod(method)
            except ValueError:
                console.print(f"[red]Error:[/red] Unsupported correlation method: {method}")
                console.print("Supported methods: distance_based, density_clustering, temporal_spatial, hotspot_analysis")
                raise typer.Exit(1)
            
            correlation = processor.correlate_events_by_location(
                events=located_items,
                radius=radius,
                method=correlation_method,
                time_window_hours=time_window
            )
            
            progress.remove_task(task)
        
        # Display results
        console.print("\n[bold green]Location Correlation Results[/bold green]")
        
        # Summary table
        summary_table = Table(show_header=True, header_style="bold blue")
        summary_table.add_column("Property")
        summary_table.add_column("Value")
        
        summary_table.add_row("Analysis Name", correlation.analysis_name)
        summary_table.add_row("Search Radius", f"{correlation.search_radius_meters} meters")
        summary_table.add_row("Time Window", f"{correlation.time_window_hours} hours" if correlation.time_window_hours else "N/A")
        summary_table.add_row("Evidence Analyzed", str(len(correlation.evidence_analyzed)))
        summary_table.add_row("Correlation Strength", f"{correlation.correlation_strength:.3f}" if correlation.correlation_strength else "N/A")
        
        console.print(summary_table)
        
        # Display proximity matches
        if correlation.proximity_matches:
            console.print(f"\n[bold]Proximity Matches:[/bold] {len(correlation.proximity_matches)}")
            
            matches_table = Table(show_header=True, header_style="bold yellow")
            matches_table.add_column("Object 1")
            matches_table.add_column("Object 2")
            matches_table.add_column("Distance (m)")
            matches_table.add_column("Confidence")
            
            for match in correlation.proximity_matches[:10]:  # Limit display
                if isinstance(match, dict):
                    matches_table.add_row(
                        match.get('object1_name', match.get('object1_id', 'Unknown'))[:20],
                        match.get('object2_name', match.get('object2_id', 'Unknown'))[:20],
                        f"{match.get('distance_meters', 0):.1f}",
                        f"{match.get('confidence', 0):.3f}"
                    )
            
            console.print(matches_table)
            
            if len(correlation.proximity_matches) > 10:
                console.print(f"... and {len(correlation.proximity_matches) - 10} more matches")
        
        # Display spatial clusters
        if correlation.spatial_clusters:
            console.print(f"\n[bold]Spatial Clusters:[/bold] {len(correlation.spatial_clusters)}")
            for i, cluster in enumerate(correlation.spatial_clusters, 1):
                if isinstance(cluster, dict):
                    if cluster.get('analysis_type') == 'hotspot':
                        console.print(f"  Hotspot Analysis: {cluster.get('total_events', 0)} events in {cluster.get('analysis_area_km2', 0):.2f} km²")
                        hotspots = cluster.get('hotspots', [])
                        for j, hotspot in enumerate(hotspots[:3], 1):  # Show top 3
                            console.print(f"    {j}. {hotspot.get('event_count', 0)} events (density: {hotspot.get('density_score', 0):.1f})")
                    else:
                        console.print(f"  Cluster {i}: {cluster.get('object_count', 0)} objects, radius: {cluster.get('radius_meters', 0):.1f}m")
        
        # Display temporal correlations
        if correlation.temporal_correlations:
            console.print(f"\n[bold]Temporal-Spatial Correlations:[/bold] {len(correlation.temporal_correlations)}")
            for i, temp_corr in enumerate(correlation.temporal_correlations[:5], 1):  # Show top 5
                if isinstance(temp_corr, dict):
                    console.print(
                        f"  {i}. Distance: {temp_corr.get('spatial_distance_meters', 0):.1f}m, "
                        f"Time: {temp_corr.get('temporal_distance_seconds', 0):.0f}s, "
                        f"Strength: {temp_corr.get('correlation_strength', 0):.3f}"
                    )
        
        # Save results if requested
        if output_file:
            output_file.parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, 'w') as f:
                json.dump(correlation.dict(), f, indent=2, default=str)
            
            console.print(f"\n[green]Correlation results saved to:[/green] {output_file}")
        
    except Exception as e:
        console.print(f"[red]Unexpected error:[/red] {str(e)}")
        raise typer.Exit(1)


@app.command() 
def generate_map(
    evidence_file: Path = typer.Argument(..., help="JSON file containing evidence data"),
    title: str = typer.Option("Evidence Location Map", "--title", "-t", help="Map title"),
    output_file: Path = typer.Option(Path("evidence_map.html"), "--output", "-f", help="Output file path"),
    format: str = typer.Option("html", "--format", help="Export format (html, pdf, kml, geojson)"),
    cluster: bool = typer.Option(True, "--cluster/--no-cluster", help="Enable marker clustering"),
    center: Optional[str] = typer.Option(None, "--center", "-c", help="Map center as 'lat,lon'"),
    zoom: Optional[int] = typer.Option(None, "--zoom", "-z", help="Zoom level (1-20)")
):
    """
    Generate interactive map with evidence overlay.
    
    Examples:
        lemkin-geo generate-map evidence.json --title "Crime Scene Evidence"
        lemkin-geo generate-map evidence.json --format kml --output evidence.kml
        lemkin-geo generate-map evidence.json --center "40.7,-74.0" --zoom 15
    """
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Loading evidence data...", total=None)
            
            # Load evidence data
            if not evidence_file.exists():
                console.print(f"[red]Error:[/red] Evidence file not found: {evidence_file}")
                raise typer.Exit(1)
            
            try:
                with open(evidence_file, 'r') as f:
                    evidence_data = json.load(f)
            except json.JSONDecodeError as e:
                console.print(f"[red]Error:[/red] Invalid JSON file: {e}")
                raise typer.Exit(1)
            
            progress.update(task, description="Processing evidence...")
            
            # Convert evidence data to Evidence objects
            evidence_items = []
            if isinstance(evidence_data, list):
                for item in evidence_data:
                    try:
                        evidence = Evidence(**item)
                        evidence_items.append(evidence)
                    except Exception as e:
                        console.print(f"[yellow]Warning:[/yellow] Skipping invalid evidence item: {e}")
                        continue
            elif isinstance(evidence_data, dict) and 'evidence' in evidence_data:
                for item in evidence_data['evidence']:
                    try:
                        evidence = Evidence(**item)
                        evidence_items.append(evidence)
                    except Exception as e:
                        console.print(f"[yellow]Warning:[/yellow] Skipping invalid evidence item: {e}")
                        continue
            
            located_items = [e for e in evidence_items if e.location]
            console.print(f"Loaded {len(evidence_items)} evidence items ({len(located_items)} with location data)")
            
            if not located_items:
                console.print(f"[yellow]Warning:[/yellow] No evidence items with location data found")
                raise typer.Exit(1)
            
            progress.update(task, description="Generating map...")
            
            # Parse center coordinates if provided
            center_coord = None
            if center:
                try:
                    lat, lon = [float(x.strip()) for x in center.split(',')]
                    center_coord = Coordinate(latitude=lat, longitude=lon)
                except ValueError:
                    console.print(f"[red]Error:[/red] Invalid center format. Use: lat,lon")
                    raise typer.Exit(1)
            
            # Generate map
            generator = MappingGenerator()
            interactive_map = generator.generate_evidence_map(
                evidence=located_items,
                title=title,
                center=center_coord,
                zoom_level=zoom,
                cluster_markers=cluster
            )
            
            progress.update(task, description="Exporting map...")
            
            # Export map
            try:
                export_format = ExportFormat(format.lower())
            except ValueError:
                console.print(f"[red]Error:[/red] Unsupported export format: {format}")
                console.print("Supported formats: html, pdf, kml, geojson, gpx")
                raise typer.Exit(1)
            
            output_file.parent.mkdir(parents=True, exist_ok=True)
            success = generator.export_map(
                interactive_map=interactive_map,
                output_path=output_file,
                format=export_format,
                include_metadata=True
            )
            
            progress.remove_task(task)
            
            if not success:
                console.print(f"[red]Error:[/red] Failed to export map")
                raise typer.Exit(1)
        
        # Display results
        console.print("\n[bold green]Interactive Map Generated[/bold green]")
        
        # Map summary
        summary_table = Table(show_header=True, header_style="bold blue")
        summary_table.add_column("Property")
        summary_table.add_column("Value")
        
        summary_table.add_row("Title", interactive_map.title)
        summary_table.add_row("Evidence Points", str(len(interactive_map.evidence_points)))
        summary_table.add_row("Map Layers", str(len(interactive_map.layers)))
        summary_table.add_row("Center", f"{interactive_map.center.latitude:.6f}, {interactive_map.center.longitude:.6f}")
        summary_table.add_row("Zoom Level", str(interactive_map.zoom_level))
        summary_table.add_row("Export Format", format.upper())
        summary_table.add_row("Output File", str(output_file))
        
        console.print(summary_table)
        
        # Evidence type breakdown
        if located_items:
            evidence_types = {}
            for evidence in located_items:
                evidence_types[evidence.evidence_type] = evidence_types.get(evidence.evidence_type, 0) + 1
            
            console.print(f"\n[bold]Evidence Types:[/bold]")
            for evidence_type, count in evidence_types.items():
                console.print(f"  {evidence_type}: {count}")
        
        console.print(f"\n[green]Map exported to:[/green] {output_file}")
        
        # Open file suggestion
        if format == "html":
            console.print(f"[dim]Open the file in a web browser to view the interactive map[/dim]")
        
    except Exception as e:
        console.print(f"[red]Unexpected error:[/red] {str(e)}")
        raise typer.Exit(1)


@app.command()
def create_geofence(
    name: str = typer.Argument(..., help="Geofence name"),
    geofence_type: str = typer.Option("circular", "--type", "-t", help="Geofence type (circular, rectangular, polygon)"),
    center: Optional[str] = typer.Option(None, "--center", "-c", help="Center coordinates as 'lat,lon'"),
    radius: Optional[float] = typer.Option(None, "--radius", "-r", help="Radius in meters (for circular)"),
    bbox: Optional[str] = typer.Option(None, "--bbox", "-b", help="Bounding box as 'north,south,east,west'"),
    vertices: Optional[str] = typer.Option(None, "--vertices", "-v", help="Polygon vertices as 'lat1,lon1;lat2,lon2;...'"),
    evidence_file: Optional[Path] = typer.Option(None, "--evidence", "-e", help="Evidence file to analyze"),
    output_file: Optional[Path] = typer.Option(None, "--output", "-f", help="Output file path")
):
    """
    Create geofence and analyze evidence violations.
    
    Examples:
        lemkin-geo create-geofence "School Zone" --center "40.7,-74.0" --radius 500
        lemkin-geo create-geofence "Restricted Area" --type rectangular --bbox "40.8,40.6,-73.9,-74.1"
        lemkin-geo create-geofence "Crime Scene" --center "40.7,-74.0" --radius 100 --evidence evidence.json
    """
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Creating geofence...", total=None)
            
            # Parse geofence type
            try:
                gf_type = GeofenceType(geofence_type.lower())
            except ValueError:
                console.print(f"[red]Error:[/red] Unsupported geofence type: {geofence_type}")
                console.print("Supported types: circular, rectangular, polygon")
                raise typer.Exit(1)
            
            # Parse parameters based on type
            center_coord = None
            bounding_box = None
            polygon_vertices = None
            
            if gf_type == GeofenceType.CIRCULAR:
                if not center or not radius:
                    console.print(f"[red]Error:[/red] Circular geofence requires --center and --radius")
                    raise typer.Exit(1)
                
                try:
                    lat, lon = [float(x.strip()) for x in center.split(',')]
                    center_coord = Coordinate(latitude=lat, longitude=lon)
                except ValueError:
                    console.print(f"[red]Error:[/red] Invalid center format. Use: lat,lon")
                    raise typer.Exit(1)
            
            elif gf_type == GeofenceType.RECTANGULAR:
                if not bbox:
                    console.print(f"[red]Error:[/red] Rectangular geofence requires --bbox")
                    raise typer.Exit(1)
                
                try:
                    bbox_values = [float(x.strip()) for x in bbox.split(',')]
                    if len(bbox_values) != 4:
                        raise ValueError("Bounding box must have 4 values")
                    
                    bounding_box = BoundingBox(
                        north=bbox_values[0],
                        south=bbox_values[1],
                        east=bbox_values[2],
                        west=bbox_values[3]
                    )
                except (ValueError, IndexError):
                    console.print(f"[red]Error:[/red] Invalid bounding box format. Use: north,south,east,west")
                    raise typer.Exit(1)
            
            elif gf_type == GeofenceType.POLYGON:
                if not vertices:
                    console.print(f"[red]Error:[/red] Polygon geofence requires --vertices")
                    raise typer.Exit(1)
                
                try:
                    polygon_vertices = []
                    for vertex_str in vertices.split(';'):
                        lat, lon = [float(x.strip()) for x in vertex_str.split(',')]
                        polygon_vertices.append(Coordinate(latitude=lat, longitude=lon))
                    
                    if len(polygon_vertices) < 3:
                        raise ValueError("Polygon must have at least 3 vertices")
                        
                except (ValueError, IndexError):
                    console.print(f"[red]Error:[/red] Invalid vertices format. Use: lat1,lon1;lat2,lon2;...")
                    raise typer.Exit(1)
            
            # Create geofence
            processor = GeofenceProcessor()
            geofence_id = processor.create_geofence(
                name=name,
                geofence_type=gf_type,
                center=center_coord,
                radius_meters=radius,
                bounding_box=bounding_box,
                polygon_vertices=polygon_vertices
            )
            
            progress.remove_task(task)
        
        console.print(f"\n[bold green]Geofence Created:[/bold green] {name}")
        console.print(f"Geofence ID: {geofence_id}")
        
        # Display geofence details
        geofence_table = Table(show_header=True, header_style="bold blue")
        geofence_table.add_column("Property")
        geofence_table.add_column("Value")
        
        geofence_table.add_row("Name", name)
        geofence_table.add_row("Type", gf_type)
        
        if center_coord:
            geofence_table.add_row("Center", f"{center_coord.latitude:.6f}, {center_coord.longitude:.6f}")
        if radius:
            geofence_table.add_row("Radius", f"{radius} meters")
        if bounding_box:
            geofence_table.add_row("Area", f"{bounding_box.area_km2():.4f} km²")
        if polygon_vertices:
            geofence_table.add_row("Vertices", str(len(polygon_vertices)))
        
        console.print(geofence_table)
        
        # Analyze evidence if provided
        if evidence_file:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console
            ) as progress:
                task = progress.add_task("Analyzing evidence against geofence...", total=None)
                
                # Load evidence data
                if not evidence_file.exists():
                    console.print(f"[red]Error:[/red] Evidence file not found: {evidence_file}")
                    raise typer.Exit(1)
                
                try:
                    with open(evidence_file, 'r') as f:
                        evidence_data = json.load(f)
                    
                    evidence_items = []
                    if isinstance(evidence_data, list):
                        for item in evidence_data:
                            try:
                                evidence = Evidence(**item)
                                evidence_items.append(evidence)
                            except Exception:
                                continue
                    
                    located_items = [e for e in evidence_items if e.location]
                    
                    if located_items:
                        result = processor.analyze_geofence_violations(
                            geofence_id=geofence_id,
                            events=located_items
                        )
                        
                        progress.remove_task(task)
                        
                        # Display analysis results
                        console.print(f"\n[bold]Geofence Analysis Results:[/bold]")
                        
                        analysis_table = Table(show_header=True, header_style="bold yellow")
                        analysis_table.add_column("Metric")
                        analysis_table.add_column("Count")
                        
                        analysis_table.add_row("Evidence Inside", str(len(result.events_inside)))
                        analysis_table.add_row("Evidence Outside", str(len(result.events_outside)))
                        analysis_table.add_row("Boundary Crossings", str(len(result.boundary_crossings)))
                        
                        if result.duration_inside_minutes:
                            analysis_table.add_row("Time Inside", f"{result.duration_inside_minutes:.1f} minutes")
                        
                        console.print(analysis_table)
                        
                        # Show evidence inside geofence
                        if result.events_inside:
                            console.print(f"\n[bold]Evidence Inside Geofence:[/bold]")
                            for event_id in result.events_inside[:5]:  # Show first 5
                                evidence_item = next((e for e in located_items if str(e.id) == event_id), None)
                                if evidence_item:
                                    console.print(f"  • {evidence_item.title} ({evidence_item.evidence_type})")
                            
                            if len(result.events_inside) > 5:
                                console.print(f"  ... and {len(result.events_inside) - 5} more")
                        
                        # Save analysis results
                        if output_file:
                            output_data = {
                                'geofence': {
                                    'id': geofence_id,
                                    'name': name,
                                    'type': gf_type,
                                    'center': center_coord.dict() if center_coord else None,
                                    'radius_meters': radius,
                                    'bounding_box': bounding_box.dict() if bounding_box else None
                                },
                                'analysis_results': result.dict(),
                                'created_at': datetime.utcnow().isoformat()
                            }
                            
                            output_file.parent.mkdir(parents=True, exist_ok=True)
                            with open(output_file, 'w') as f:
                                json.dump(output_data, f, indent=2, default=str)
                            
                            console.print(f"\n[green]Analysis results saved to:[/green] {output_file}")
                    
                except Exception as e:
                    console.print(f"[red]Error analyzing evidence:[/red] {str(e)}")
        
    except Exception as e:
        console.print(f"[red]Unexpected error:[/red] {str(e)}")
        raise typer.Exit(1)


@app.command()
def version():
    """Show version information."""
    console.print("[bold blue]Lemkin Geospatial Analysis Suite[/bold blue]")
    console.print("Version: 1.0.0")
    console.print("Author: Lemkin AI Contributors")
    console.print("License: Apache 2.0")
    console.print("\nComponents:")
    console.print("  • Coordinate Converter")
    console.print("  • Satellite Analyzer") 
    console.print("  • Geofence Processor")
    console.print("  • Interactive Mapping")


@app.command()
def example_data():
    """Generate example evidence data file for testing."""
    example_evidence = [
        {
            "title": "Photograph of incident location",
            "description": "Digital photograph taken at scene",
            "evidence_type": "photograph",
            "location": {
                "latitude": 40.7128,
                "longitude": -74.0060,
                "source": "GPS camera"
            },
            "timestamp": "2024-01-15T14:30:00Z",
            "source": "Crime scene photographer",
            "collector": "Detective Smith",
            "verified": True
        },
        {
            "title": "Witness video recording",
            "description": "Mobile phone video of events",
            "evidence_type": "video", 
            "location": {
                "latitude": 40.7130,
                "longitude": -74.0058,
                "source": "Mobile GPS"
            },
            "timestamp": "2024-01-15T14:32:00Z",
            "source": "Eyewitness",
            "collector": "Officer Jones",
            "verified": True
        },
        {
            "title": "Social media post",
            "description": "Public post about incident",
            "evidence_type": "social_media_post",
            "location": {
                "latitude": 40.7125,
                "longitude": -74.0062,
                "source": "Social media geotag"
            },
            "timestamp": "2024-01-15T14:35:00Z",
            "source": "Twitter",
            "collector": "Digital forensics team",
            "verified": False
        }
    ]
    
    output_file = Path("example_evidence.json")
    with open(output_file, 'w') as f:
        json.dump(example_evidence, f, indent=2)
    
    console.print(f"[green]Example evidence data saved to:[/green] {output_file}")
    console.print("\nYou can now use this file with other commands:")
    console.print(f"  lemkin-geo correlate-locations {output_file}")
    console.print(f"  lemkin-geo generate-map {output_file}")


if __name__ == "__main__":
    app()