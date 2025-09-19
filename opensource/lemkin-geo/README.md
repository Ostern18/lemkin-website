# Lemkin Geospatial Analysis Suite

## Purpose

The Lemkin Geospatial Analysis Suite provides geographic analysis capabilities for legal investigations without requiring GIS expertise. This toolkit enables investigators to work with coordinates, satellite imagery, and location-based evidence correlation using simple, accessible interfaces.

## Safety & Ethics Notice

⚠️ **IMPORTANT**: This toolkit is designed for legitimate legal investigations and human rights documentation. Users must:
- Respect privacy and surveillance laws
- Obtain proper authorization for satellite imagery analysis
- Protect location privacy of witnesses and victims
- Use geospatial data only for lawful purposes
- Verify coordinate accuracy before legal proceedings

## Key Features

- **Coordinate Conversion**: Support for multiple coordinate formats (Decimal, DMS, DDM, UTM)
- **Geocoding**: Convert addresses to coordinates and vice versa
- **Satellite Analysis**: Analyze satellite imagery for temporal changes
- **Event Correlation**: Find geographic patterns in incidents
- **Interactive Mapping**: Generate evidence maps for case presentation
- **Geofencing**: Location-based event clustering and analysis

## Quick Start

```bash
# Install the toolkit
pip install lemkin-geo

# Convert coordinates
lemkin-geo convert-coords "40.7128, -74.0060" --input-format decimal

# Geocode an address
lemkin-geo geocode "United Nations Headquarters, New York"

# Analyze satellite imagery
lemkin-geo analyze-satellite --min-lat 40.0 --max-lat 41.0 \
    --min-lon -74.5 --max-lon -73.5 \
    --start-date 2024-01-01 --end-date 2024-02-01

# Create evidence map
lemkin-geo create-map evidence.json --output case_map.html
```

## Usage Examples

### 1. Converting Coordinates Between Formats

```bash
# Convert DMS to decimal
lemkin-geo convert-coords "40°42'46\"N 74°0'21\"W" --input-format dms --geocode

# Convert UTM to decimal
lemkin-geo convert-coords "18T 589633 4511322" --input-format utm
```

### 2. Analyzing Satellite Imagery for Changes

```bash
# Analyze area for structural changes
lemkin-geo analyze-satellite \
    --min-lat 36.0 --max-lat 37.0 \
    --min-lon 37.0 --max-lon 38.0 \
    --start-date 2023-01-01 \
    --end-date 2024-01-01 \
    --output satellite_analysis.json
```

### 3. Correlating Events by Location

```bash
# Create events file
cat > events.json << EOF
[
    {
        "event_id": "incident_001",
        "event_type": "incident",
        "location": {"latitude": 40.7128, "longitude": -74.0060},
        "timestamp": "2024-01-15T10:30:00Z",
        "description": "Witness report"
    },
    {
        "event_id": "incident_002",
        "event_type": "evidence",
        "location": {"latitude": 40.7130, "longitude": -74.0058},
        "timestamp": "2024-01-15T11:00:00Z",
        "description": "Physical evidence found"
    }
]
EOF

# Correlate events within 500 meters
lemkin-geo correlate-events events.json --radius 500 --output correlation.json
```

### 4. Creating Interactive Evidence Maps

```bash
# Create evidence data
cat > evidence.json << EOF
[
    {
        "evidence_id": "DOC001",
        "evidence_type": "document",
        "location": {"latitude": 40.7589, "longitude": -73.9851},
        "description": "Witness statement collected at Times Square"
    },
    {
        "evidence_id": "PHOTO001",
        "evidence_type": "photo",
        "location": {"latitude": 40.7614, "longitude": -73.9776},
        "description": "Photographic evidence from location"
    }
]
EOF

# Generate interactive map
lemkin-geo create-map evidence.json --output evidence_map.html --title "Case Evidence Locations"
```

## Input/Output Specifications

### Coordinate Format
```python
{
    "latitude": 40.7128,
    "longitude": -74.0060,
    "elevation": 10.0,  # Optional, in meters
    "coordinate_system": "WGS84",
    "accuracy_meters": 5.0  # Optional
}
```

### Event Format
```python
{
    "event_id": "unique_identifier",
    "event_type": "incident",  # incident, witness, evidence, report, observation
    "location": {
        "latitude": 40.7128,
        "longitude": -74.0060
    },
    "timestamp": "2024-01-15T10:30:00Z",
    "description": "Event description",
    "metadata": {}  # Optional additional data
}
```

### Evidence Format
```python
{
    "evidence_id": "unique_identifier",
    "evidence_type": "document",  # document, photo, video, witness, report
    "location": {  # Optional single location
        "latitude": 40.7128,
        "longitude": -74.0060
    },
    "locations": [  # Optional multiple locations
        {"latitude": 40.7128, "longitude": -74.0060},
        {"latitude": 40.7130, "longitude": -74.0058}
    ],
    "timestamp": "2024-01-15T10:30:00Z",  # Optional
    "description": "Evidence description"
}
```

## API Reference

### Core Classes

#### CoordinateConverter
Handles coordinate format conversions and geocoding.

```python
from lemkin_geo import CoordinateConverter, CoordinateFormat

converter = CoordinateConverter()

# Convert coordinates
standard = converter.standardize_coordinates(
    "40°42'46\"N 74°0'21\"W",
    CoordinateFormat.DMS
)

# Geocode address
coords = converter.geocode_address("UN Headquarters, New York")

# Reverse geocode
address = converter.reverse_geocode(standard)
```

#### GeofenceProcessor
Correlates events based on geographic proximity.

```python
from lemkin_geo import GeofenceProcessor, Event

processor = GeofenceProcessor()
correlation = processor.correlate_events_by_location(
    events=event_list,
    radius=1000.0  # meters
)

print(f"Found {len(correlation.clusters)} clusters")
print(f"Correlation strength: {correlation.correlation_strength:.1%}")
```

#### MappingGenerator
Creates interactive maps for evidence visualization.

```python
from lemkin_geo import MappingGenerator, Evidence

generator = MappingGenerator()
interactive_map = generator.generate_evidence_map(
    evidence=evidence_list,
    output_path=Path("map.html")
)
```

## Evaluation & Limitations

### Performance Metrics
- Coordinate conversion: <100ms per coordinate
- Geocoding: 1-2 seconds per address (API dependent)
- Event correlation: ~1000 events/second
- Map generation: ~100 markers/second

### Known Limitations
- Geocoding requires internet connection
- Satellite imagery analysis is simulated (requires provider integration)
- Coordinate accuracy depends on input format
- Map complexity limited by browser performance
- UTM conversion requires zone information

### Failure Modes
- Invalid coordinates: Clear error messages with format examples
- Geocoding failures: Returns None, check internet connection
- Large datasets: Consider chunking for >10,000 points
- Map rendering: Browser limitations with >1000 markers

## Safety Guidelines

### Location Privacy
1. **Witness Protection**: Never publish exact witness locations
2. **Victim Privacy**: Generalize sensitive locations
3. **Safe Houses**: Exclude protected locations from maps
4. **Operational Security**: Consider location fuzzing for sensitive data

### Coordinate Accuracy
1. **Verify Sources**: Cross-reference coordinate sources
2. **Document Precision**: Record coordinate accuracy/uncertainty
3. **Legal Standards**: Ensure accuracy meets court requirements
4. **Error Margins**: Document and communicate uncertainty

### Data Security
1. **Encryption**: Encrypt location data at rest and in transit
2. **Access Control**: Limit access to location information
3. **Audit Trails**: Log all access to geographic data
4. **Data Retention**: Follow legal requirements for location data

## Contributing

We welcome contributions that enhance geospatial analysis capabilities for legal investigations.

### Development Setup
```bash
# Clone repository
git clone https://github.com/lemkin-org/lemkin-geo.git
cd lemkin-geo

# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Format code
black src/ tests/
```

## License

Apache License 2.0 - see LICENSE file for details.

This toolkit is designed for legitimate legal investigations and human rights documentation. Users are responsible for ensuring compliance with all applicable laws regarding location data and surveillance.

---

*Part of the Lemkin AI open-source legal technology ecosystem.*