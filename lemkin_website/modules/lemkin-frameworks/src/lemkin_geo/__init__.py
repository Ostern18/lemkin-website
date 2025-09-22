"""
Lemkin Geospatial Analysis Suite

Geographic analysis tools for legal evidence investigation without requiring GIS expertise.
Designed for legal professionals to analyze spatial relationships, correlate events by location,
and generate interactive maps for legal proceedings.

Key Components:
- Coordinate Converter: GPS format standardization and projection handling
- Satellite Analyzer: Satellite imagery analysis using public datasets  
- Geofence Processor: Location-based event correlation and geofencing
- Mapping Generator: Interactive map creation with evidence overlay

Main Classes:
- GeospatialAnalyzer: Main analysis coordination class
- CoordinateConverter: Coordinate format conversion and validation
- SatelliteAnalyzer: Satellite imagery search and analysis
- GeofenceProcessor: Spatial correlation and geofencing
- MappingGenerator: Interactive map generation and export

Data Models:
- Coordinate: Geographic coordinate with validation
- BoundingBox: Geographic area definition
- Evidence: Evidence item with location data
- SpatialEvent: Event with spatial and temporal dimensions
- InteractiveMap: Interactive map with layers and evidence overlay

Example Usage:
    ```python
    from lemkin_geo import GeospatialAnalyzer, Evidence, Coordinate
    
    # Create analyzer
    analyzer = GeospatialAnalyzer()
    
    # Add evidence with location
    evidence = Evidence(
        title="Crime scene photo",
        evidence_type="photograph",
        location=Coordinate(latitude=40.7128, longitude=-74.0060)
    )
    analyzer.add_evidence(evidence)
    
    # Generate interactive map
    from lemkin_geo import generate_evidence_map
    map_obj = generate_evidence_map([evidence])
    ```

CLI Usage:
    ```bash
    # Convert coordinates
    lemkin-geo convert-coordinates "40.7128, -74.0060" --output-format dms
    
    # Analyze satellite imagery
    lemkin-geo analyze-satellite "40.8,40.6,-73.9,-74.1" --start 2024-01-01 --end 2024-01-31
    
    # Correlate evidence by location  
    lemkin-geo correlate-locations evidence.json --radius 500
    
    # Generate interactive map
    lemkin-geo generate-map evidence.json --title "Evidence Locations"
    
    # Create geofence
    lemkin-geo create-geofence "Crime Scene" --center "40.7,-74.0" --radius 100
    ```
"""

from .core import (
    # Main analyzer class
    GeospatialAnalyzer,
    
    # Configuration
    GeoConfig,
    
    # Core data models
    Coordinate,
    BoundingBox,
    DateRange,
    Evidence,
    SpatialEvent,
    
    # Analysis result models
    SatelliteAnalysis,
    LocationCorrelation,
    GeofenceResult,
    
    # Map models
    InteractiveMap,
    MapLayer,
    
    # Enums
    CoordinateFormat,
    ProjectionSystem,
    SatelliteProvider,
    MapLayerType,
    EvidenceType
)

from .coordinate_converter import (
    CoordinateConverter,
    ConversionResult,
    standardize_coordinates,
    convert_coordinate_format
)

from .satellite_analyzer import (
    SatelliteAnalyzer,
    SatelliteImageQuery,
    SatelliteImageMetadata,
    ChangeDetectionResult,
    analyze_satellite_imagery
)

from .geofence_processor import (
    GeofenceProcessor,
    GeofenceDefinition,
    ProximityMatch,
    SpatialCluster,
    correlate_events_by_location,
    # Enums
    GeofenceType,
    ProximityRelation,
    CorrelationMethod
)

from .mapping_generator import (
    MappingGenerator,
    MarkerStyle,
    LayerConfiguration,
    generate_evidence_map,
    # Enums
    MapProvider,
    ExportFormat
)

# Package metadata
__version__ = "1.0.0"
__author__ = "Lemkin AI Contributors"
__email__ = "contributors@lemkin.org"
__license__ = "Apache-2.0"
__description__ = "Geospatial analysis suite for legal evidence investigation"

# Package-level convenience functions
def create_analyzer(config=None):
    """
    Create a new GeospatialAnalyzer instance
    
    Args:
        config: Optional GeoConfig instance
        
    Returns:
        GeospatialAnalyzer instance
    """
    return GeospatialAnalyzer(config)


def load_evidence_from_json(file_path, encoding='utf-8'):
    """
    Load evidence data from JSON file
    
    Args:
        file_path: Path to JSON file
        encoding: File encoding (default: utf-8)
        
    Returns:
        List of Evidence objects
        
    Raises:
        FileNotFoundError: If file doesn't exist
        json.JSONDecodeError: If file contains invalid JSON
        ValidationError: If evidence data is invalid
    """
    import json
    from pathlib import Path
    
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Evidence file not found: {file_path}")
    
    with open(file_path, 'r', encoding=encoding) as f:
        data = json.load(f)
    
    evidence_list = []
    
    # Handle different JSON structures
    if isinstance(data, list):
        evidence_data = data
    elif isinstance(data, dict) and 'evidence' in data:
        evidence_data = data['evidence']
    elif isinstance(data, dict) and 'evidence_collection' in data:
        evidence_data = data['evidence_collection']
    else:
        raise ValueError("Invalid JSON structure. Expected list or object with 'evidence' key.")
    
    for item in evidence_data:
        try:
            evidence = Evidence(**item)
            evidence_list.append(evidence)
        except Exception as e:
            import warnings
            warnings.warn(f"Skipping invalid evidence item: {e}")
            continue
    
    return evidence_list


def save_evidence_to_json(evidence_list, file_path, include_metadata=True, encoding='utf-8'):
    """
    Save evidence data to JSON file
    
    Args:
        evidence_list: List of Evidence objects
        file_path: Output file path
        include_metadata: Include package metadata in output
        encoding: File encoding (default: utf-8)
    """
    import json
    from pathlib import Path
    from datetime import datetime
    
    output_data = {
        'evidence': [evidence.dict() for evidence in evidence_list]
    }
    
    if include_metadata:
        output_data['metadata'] = {
            'exported_at': datetime.utcnow().isoformat(),
            'package_version': __version__,
            'evidence_count': len(evidence_list),
            'located_evidence_count': len([e for e in evidence_list if e.location])
        }
    
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(file_path, 'w', encoding=encoding) as f:
        json.dump(output_data, f, indent=2, default=str)


def quick_analysis(evidence_list, output_dir=None):
    """
    Perform quick analysis on evidence list
    
    Args:
        evidence_list: List of Evidence objects
        output_dir: Optional output directory for results
        
    Returns:
        Dictionary with analysis results
    """
    from pathlib import Path
    from datetime import datetime
    
    analyzer = GeospatialAnalyzer()
    
    # Add evidence to analyzer
    for evidence in evidence_list:
        analyzer.add_evidence(evidence)
    
    located_evidence = [e for e in evidence_list if e.location]
    
    results = {
        'summary': {
            'total_evidence': len(evidence_list),
            'located_evidence': len(located_evidence),
            'evidence_types': {},
            'analysis_timestamp': datetime.utcnow().isoformat()
        }
    }
    
    # Evidence type breakdown
    for evidence in evidence_list:
        evidence_type = evidence.evidence_type
        results['summary']['evidence_types'][evidence_type] = (
            results['summary']['evidence_types'].get(evidence_type, 0) + 1
        )
    
    if len(located_evidence) >= 2:
        # Spatial correlation analysis
        processor = GeofenceProcessor()
        correlation = processor.correlate_events_by_location(
            events=located_evidence,
            radius=100.0  # 100 meter radius
        )
        results['spatial_correlation'] = correlation.dict()
        
        # Generate map
        generator = MappingGenerator()
        interactive_map = generator.generate_evidence_map(
            evidence=located_evidence,
            title="Quick Analysis Evidence Map"
        )
        results['map_generated'] = True
        
        # Save outputs if directory provided
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Save correlation results
            correlation_file = output_path / "spatial_correlation.json"
            with open(correlation_file, 'w') as f:
                import json
                json.dump(correlation.dict(), f, indent=2, default=str)
            
            # Save map
            map_file = output_path / "evidence_map.html"
            generator.export_map(interactive_map, map_file, ExportFormat.HTML)
            
            results['output_files'] = {
                'correlation': str(correlation_file),
                'map': str(map_file)
            }
    
    return results


# Export main functions for easy access
__all__ = [
    # Main classes
    'GeospatialAnalyzer',
    'CoordinateConverter', 
    'SatelliteAnalyzer',
    'GeofenceProcessor',
    'MappingGenerator',
    
    # Configuration
    'GeoConfig',
    
    # Core models
    'Coordinate',
    'BoundingBox', 
    'DateRange',
    'Evidence',
    'SpatialEvent',
    
    # Analysis models
    'SatelliteAnalysis',
    'LocationCorrelation',
    'GeofenceResult',
    'InteractiveMap',
    'MapLayer',
    
    # Result models
    'ConversionResult',
    'SatelliteImageMetadata',
    'ChangeDetectionResult',
    'GeofenceDefinition',
    'ProximityMatch',
    'SpatialCluster',
    'MarkerStyle',
    'LayerConfiguration',
    
    # Enums
    'CoordinateFormat',
    'ProjectionSystem', 
    'SatelliteProvider',
    'MapLayerType',
    'EvidenceType',
    'GeofenceType',
    'ProximityRelation',
    'CorrelationMethod',
    'MapProvider',
    'ExportFormat',
    
    # Convenience functions
    'standardize_coordinates',
    'convert_coordinate_format',
    'analyze_satellite_imagery',
    'correlate_events_by_location', 
    'generate_evidence_map',
    'create_analyzer',
    'load_evidence_from_json',
    'save_evidence_to_json',
    'quick_analysis'
]