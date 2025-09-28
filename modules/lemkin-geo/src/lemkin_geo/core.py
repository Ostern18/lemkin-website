"""
Core geospatial analysis functionality for legal investigations.
"""

import json
import re
from datetime import datetime, timezone, date
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union
from enum import Enum

import folium
from folium import plugins
import geopy
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
import pyproj
from shapely.geometry import Point, Polygon
from pydantic import BaseModel, Field, field_validator
from loguru import logger


class CoordinateFormat(str, Enum):
    """Supported coordinate formats"""
    DECIMAL = "decimal"
    DMS = "dms"  # Degrees Minutes Seconds
    DDM = "ddm"  # Degrees Decimal Minutes
    UTM = "utm"  # Universal Transverse Mercator
    MGRS = "mgrs"  # Military Grid Reference System


class EventType(str, Enum):
    """Types of events for correlation"""
    INCIDENT = "incident"
    WITNESS = "witness"
    EVIDENCE = "evidence"
    REPORT = "report"
    OBSERVATION = "observation"


class StandardCoordinate(BaseModel):
    """Standardized coordinate representation"""
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)
    elevation: Optional[float] = None
    coordinate_system: str = "WGS84"
    original_format: CoordinateFormat = CoordinateFormat.DECIMAL
    accuracy_meters: Optional[float] = None

    @field_validator('latitude')
    def validate_latitude(cls, v):
        if not -90 <= v <= 90:
            raise ValueError(f"Latitude must be between -90 and 90, got {v}")
        return v

    @field_validator('longitude')
    def validate_longitude(cls, v):
        if not -180 <= v <= 180:
            raise ValueError(f"Longitude must be between -180 and 180, got {v}")
        return v


class BoundingBox(BaseModel):
    """Geographic bounding box"""
    min_lat: float = Field(ge=-90, le=90)
    max_lat: float = Field(ge=-90, le=90)
    min_lon: float = Field(ge=-180, le=180)
    max_lon: float = Field(ge=-180, le=180)

    @field_validator('max_lat')
    def validate_lat_order(cls, v, values):
        if 'min_lat' in values.data and v < values.data['min_lat']:
            raise ValueError("max_lat must be greater than min_lat")
        return v

    @field_validator('max_lon')
    def validate_lon_order(cls, v, values):
        if 'min_lon' in values.data and v < values.data['min_lon']:
            raise ValueError("max_lon must be greater than min_lon")
        return v


class DateRange(BaseModel):
    """Date range for temporal queries"""
    start_date: datetime
    end_date: datetime

    @field_validator('end_date')
    def validate_date_order(cls, v, values):
        if 'start_date' in values.data and v < values.data['start_date']:
            raise ValueError("end_date must be after start_date")
        return v


class Event(BaseModel):
    """Event with location and time"""
    event_id: str
    event_type: EventType
    location: StandardCoordinate
    timestamp: datetime
    description: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class Evidence(BaseModel):
    """Evidence with geographic component"""
    evidence_id: str
    location: Optional[StandardCoordinate] = None
    locations: List[StandardCoordinate] = Field(default_factory=list)
    timestamp: Optional[datetime] = None
    description: str
    evidence_type: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class SatelliteAnalysis(BaseModel):
    """Satellite imagery analysis results"""
    analysis_id: str
    bbox: BoundingBox
    date_range: DateRange
    images_found: int
    changes_detected: List[Dict[str, Any]] = Field(default_factory=list)
    analysis_date: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = Field(default_factory=dict)


class LocationCorrelation(BaseModel):
    """Correlation of events by location"""
    correlation_id: str
    center_point: StandardCoordinate
    radius_meters: float
    events: List[Event]
    clusters: List[Dict[str, Any]] = Field(default_factory=list)
    correlation_strength: float = Field(ge=0.0, le=1.0)


class InteractiveMap(BaseModel):
    """Interactive map configuration"""
    map_id: str
    center: StandardCoordinate
    zoom_level: int = 10
    evidence_items: List[Evidence]
    layers: List[str] = Field(default_factory=list)
    html_path: Optional[Path] = None


class CoordinateConverter:
    """Convert between coordinate formats and projections"""

    def __init__(self):
        """Initialize coordinate converter"""
        self.geocoder = Nominatim(user_agent="lemkin-geo/0.1.0")
        logger.info("Initialized coordinate converter")

    def standardize_coordinates(
        self,
        coords: str,
        format: CoordinateFormat = CoordinateFormat.DECIMAL
    ) -> StandardCoordinate:
        """
        Standardize coordinates from various formats.

        Args:
            coords: Coordinate string
            format: Input coordinate format

        Returns:
            StandardCoordinate object
        """
        try:
            if format == CoordinateFormat.DECIMAL:
                return self._parse_decimal(coords)
            elif format == CoordinateFormat.DMS:
                return self._parse_dms(coords)
            elif format == CoordinateFormat.DDM:
                return self._parse_ddm(coords)
            elif format == CoordinateFormat.UTM:
                return self._parse_utm(coords)
            else:
                raise ValueError(f"Unsupported format: {format}")

        except Exception as e:
            logger.error(f"Failed to parse coordinates: {e}")
            raise ValueError(f"Invalid coordinate format: {coords}")

    def _parse_decimal(self, coords: str) -> StandardCoordinate:
        """Parse decimal degree coordinates"""
        # Handle various decimal formats
        # Examples: "40.7128, -74.0060" or "40.7128 N 74.0060 W"

        # Remove extra spaces and normalize
        coords = coords.strip()

        # Try comma-separated format first
        if ',' in coords:
            parts = coords.split(',')
            if len(parts) == 2:
                lat = float(parts[0].strip())
                lon = float(parts[1].strip())
                return StandardCoordinate(
                    latitude=lat,
                    longitude=lon,
                    original_format=CoordinateFormat.DECIMAL
                )

        # Try space-separated with N/S E/W indicators
        pattern = r'([0-9.]+)\s*([NS])\s*([0-9.]+)\s*([EW])'
        match = re.match(pattern, coords, re.IGNORECASE)
        if match:
            lat = float(match.group(1))
            if match.group(2).upper() == 'S':
                lat = -lat
            lon = float(match.group(3))
            if match.group(4).upper() == 'W':
                lon = -lon
            return StandardCoordinate(
                latitude=lat,
                longitude=lon,
                original_format=CoordinateFormat.DECIMAL
            )

        # Try simple space-separated
        parts = coords.split()
        if len(parts) == 2:
            lat = float(parts[0])
            lon = float(parts[1])
            return StandardCoordinate(
                latitude=lat,
                longitude=lon,
                original_format=CoordinateFormat.DECIMAL
            )

        raise ValueError(f"Cannot parse decimal coordinates: {coords}")

    def _parse_dms(self, coords: str) -> StandardCoordinate:
        """Parse Degrees Minutes Seconds format"""
        # Example: "40°42'46"N 74°0'21"W"

        pattern = r'(\d+)[°]\s*(\d+)[\']\s*([0-9.]+)[\"]\s*([NS])\s*(\d+)[°]\s*(\d+)[\']\s*([0-9.]+)[\"]\s*([EW])'
        match = re.match(pattern, coords, re.IGNORECASE)

        if match:
            lat_deg = float(match.group(1))
            lat_min = float(match.group(2))
            lat_sec = float(match.group(3))
            lat = lat_deg + lat_min/60 + lat_sec/3600
            if match.group(4).upper() == 'S':
                lat = -lat

            lon_deg = float(match.group(5))
            lon_min = float(match.group(6))
            lon_sec = float(match.group(7))
            lon = lon_deg + lon_min/60 + lon_sec/3600
            if match.group(8).upper() == 'W':
                lon = -lon

            return StandardCoordinate(
                latitude=lat,
                longitude=lon,
                original_format=CoordinateFormat.DMS
            )

        raise ValueError(f"Cannot parse DMS coordinates: {coords}")

    def _parse_ddm(self, coords: str) -> StandardCoordinate:
        """Parse Degrees Decimal Minutes format"""
        # Example: "40 42.767' N 74 0.35' W"

        pattern = r'(\d+)\s+([0-9.]+)[\']\s*([NS])\s+(\d+)\s+([0-9.]+)[\']\s*([EW])'
        match = re.match(pattern, coords, re.IGNORECASE)

        if match:
            lat_deg = float(match.group(1))
            lat_min = float(match.group(2))
            lat = lat_deg + lat_min/60
            if match.group(3).upper() == 'S':
                lat = -lat

            lon_deg = float(match.group(4))
            lon_min = float(match.group(5))
            lon = lon_deg + lon_min/60
            if match.group(6).upper() == 'W':
                lon = -lon

            return StandardCoordinate(
                latitude=lat,
                longitude=lon,
                original_format=CoordinateFormat.DDM
            )

        raise ValueError(f"Cannot parse DDM coordinates: {coords}")

    def _parse_utm(self, coords: str) -> StandardCoordinate:
        """Parse UTM coordinates"""
        # Example: "18T 589633 4511322"

        pattern = r'(\d+)([A-Z])\s+(\d+)\s+(\d+)'
        match = re.match(pattern, coords, re.IGNORECASE)

        if match:
            zone = int(match.group(1))
            letter = match.group(2).upper()
            easting = float(match.group(3))
            northing = float(match.group(4))

            # Determine if northern or southern hemisphere
            northern = letter >= 'N'

            # Create UTM projection
            proj_utm = pyproj.Proj(proj='utm', zone=zone, ellps='WGS84', north=northern)
            proj_latlon = pyproj.Proj(proj='latlong', datum='WGS84')

            # Transform to lat/lon
            lon, lat = pyproj.transform(proj_utm, proj_latlon, easting, northing)

            return StandardCoordinate(
                latitude=lat,
                longitude=lon,
                original_format=CoordinateFormat.UTM
            )

        raise ValueError(f"Cannot parse UTM coordinates: {coords}")

    def geocode_address(self, address: str) -> Optional[StandardCoordinate]:
        """
        Geocode an address to coordinates.

        Args:
            address: Address string

        Returns:
            StandardCoordinate or None if not found
        """
        try:
            location = self.geocoder.geocode(address)
            if location:
                return StandardCoordinate(
                    latitude=location.latitude,
                    longitude=location.longitude,
                    original_format=CoordinateFormat.DECIMAL
                )
            return None
        except Exception as e:
            logger.error(f"Geocoding failed for {address}: {e}")
            return None

    def reverse_geocode(self, coordinate: StandardCoordinate) -> Optional[str]:
        """
        Reverse geocode coordinates to address.

        Args:
            coordinate: StandardCoordinate object

        Returns:
            Address string or None
        """
        try:
            location = self.geocoder.reverse(
                (coordinate.latitude, coordinate.longitude)
            )
            if location:
                return location.address
            return None
        except Exception as e:
            logger.error(f"Reverse geocoding failed: {e}")
            return None


class SatelliteAnalyzer:
    """Analyze satellite imagery for investigations"""

    def __init__(self):
        """Initialize satellite analyzer"""
        logger.info("Initialized satellite analyzer")

    def analyze_satellite_imagery(
        self,
        bbox: BoundingBox,
        date_range: DateRange
    ) -> SatelliteAnalysis:
        """
        Analyze satellite imagery for an area and time period.

        Args:
            bbox: Bounding box for area of interest
            date_range: Time period for analysis

        Returns:
            SatelliteAnalysis with findings
        """
        import uuid

        analysis_id = str(uuid.uuid4())

        # In production, this would integrate with satellite data providers
        # (Sentinel Hub, Planet Labs, etc.)
        # For demonstration, we simulate the analysis

        logger.info(f"Analyzing satellite imagery for bbox: {bbox}")

        # Simulate finding images
        images_found = 12  # Would be actual count from API

        # Simulate change detection
        changes_detected = []

        # Example change detection result
        if images_found > 0:
            changes_detected.append({
                "type": "structure_appearance",
                "location": {
                    "lat": (bbox.min_lat + bbox.max_lat) / 2,
                    "lon": (bbox.min_lon + bbox.max_lon) / 2
                },
                "date_detected": date_range.start_date.isoformat(),
                "confidence": 0.85,
                "description": "New structure detected in imagery"
            })

        return SatelliteAnalysis(
            analysis_id=analysis_id,
            bbox=bbox,
            date_range=date_range,
            images_found=images_found,
            changes_detected=changes_detected,
            metadata={
                "provider": "simulation",
                "resolution_meters": 10
            }
        )


class GeofenceProcessor:
    """Process location-based event correlations"""

    def __init__(self):
        """Initialize geofence processor"""
        logger.info("Initialized geofence processor")

    def correlate_events_by_location(
        self,
        events: List[Event],
        radius: float = 1000.0
    ) -> LocationCorrelation:
        """
        Correlate events within a geographic radius.

        Args:
            events: List of events to correlate
            radius: Correlation radius in meters

        Returns:
            LocationCorrelation with clustered events
        """
        import uuid
        from collections import defaultdict

        correlation_id = str(uuid.uuid4())

        if not events:
            return LocationCorrelation(
                correlation_id=correlation_id,
                center_point=StandardCoordinate(latitude=0, longitude=0),
                radius_meters=radius,
                events=[],
                clusters=[],
                correlation_strength=0.0
            )

        # Calculate center point (centroid)
        avg_lat = sum(e.location.latitude for e in events) / len(events)
        avg_lon = sum(e.location.longitude for e in events) / len(events)
        center = StandardCoordinate(latitude=avg_lat, longitude=avg_lon)

        # Cluster events by proximity
        clusters = []
        clustered_events = set()

        for i, event1 in enumerate(events):
            if event1.event_id in clustered_events:
                continue

            cluster = [event1]
            clustered_events.add(event1.event_id)

            for event2 in events[i+1:]:
                if event2.event_id in clustered_events:
                    continue

                # Calculate distance
                dist = geodesic(
                    (event1.location.latitude, event1.location.longitude),
                    (event2.location.latitude, event2.location.longitude)
                ).meters

                if dist <= radius:
                    cluster.append(event2)
                    clustered_events.add(event2.event_id)

            if len(cluster) > 1:
                clusters.append({
                    "cluster_id": f"cluster_{len(clusters)}",
                    "events": [e.event_id for e in cluster],
                    "size": len(cluster),
                    "center": {
                        "lat": sum(e.location.latitude for e in cluster) / len(cluster),
                        "lon": sum(e.location.longitude for e in cluster) / len(cluster)
                    },
                    "time_span": {
                        "start": min(e.timestamp for e in cluster).isoformat(),
                        "end": max(e.timestamp for e in cluster).isoformat()
                    }
                })

        # Calculate correlation strength
        if len(events) > 1:
            clustered_count = sum(c["size"] for c in clusters)
            correlation_strength = clustered_count / len(events)
        else:
            correlation_strength = 0.0

        return LocationCorrelation(
            correlation_id=correlation_id,
            center_point=center,
            radius_meters=radius,
            events=events,
            clusters=clusters,
            correlation_strength=correlation_strength
        )


class MappingGenerator:
    """Generate interactive maps for evidence visualization"""

    def __init__(self):
        """Initialize mapping generator"""
        logger.info("Initialized mapping generator")

    def generate_evidence_map(
        self,
        evidence: List[Evidence],
        output_path: Optional[Path] = None
    ) -> InteractiveMap:
        """
        Generate interactive map with evidence overlay.

        Args:
            evidence: List of evidence items to map
            output_path: Optional path to save HTML map

        Returns:
            InteractiveMap configuration
        """
        import uuid

        map_id = str(uuid.uuid4())

        # Filter evidence with locations
        located_evidence = [
            e for e in evidence
            if e.location or e.locations
        ]

        if not located_evidence:
            # Default map centered on null island
            center = StandardCoordinate(latitude=0, longitude=0)
            zoom = 2
        else:
            # Calculate center from evidence locations
            all_coords = []
            for e in located_evidence:
                if e.location:
                    all_coords.append((e.location.latitude, e.location.longitude))
                for loc in e.locations:
                    all_coords.append((loc.latitude, loc.longitude))

            avg_lat = sum(c[0] for c in all_coords) / len(all_coords)
            avg_lon = sum(c[1] for c in all_coords) / len(all_coords)
            center = StandardCoordinate(latitude=avg_lat, longitude=avg_lon)

            # Calculate appropriate zoom level
            if len(all_coords) > 1:
                lat_range = max(c[0] for c in all_coords) - min(c[0] for c in all_coords)
                lon_range = max(c[1] for c in all_coords) - min(c[1] for c in all_coords)
                max_range = max(lat_range, lon_range)

                if max_range > 10:
                    zoom = 5
                elif max_range > 1:
                    zoom = 8
                elif max_range > 0.1:
                    zoom = 11
                else:
                    zoom = 14
            else:
                zoom = 12

        # Create Folium map
        m = folium.Map(
            location=[center.latitude, center.longitude],
            zoom_start=zoom,
            control_scale=True
        )

        # Add evidence markers
        for e in located_evidence:
            if e.location:
                self._add_evidence_marker(m, e, e.location)
            for loc in e.locations:
                self._add_evidence_marker(m, e, loc)

        # Add layers
        folium.TileLayer('openstreetmap').add_to(m)
        folium.TileLayer('Stamen Terrain').add_to(m)
        folium.LayerControl().add_to(m)

        # Add plugins
        plugins.Fullscreen().add_to(m)
        plugins.MeasureControl().add_to(m)
        plugins.MousePosition().add_to(m)

        # Save if path provided
        if output_path:
            m.save(str(output_path))
            logger.info(f"Map saved to {output_path}")

        return InteractiveMap(
            map_id=map_id,
            center=center,
            zoom_level=zoom,
            evidence_items=evidence,
            layers=["openstreetmap", "terrain"],
            html_path=output_path
        )

    def _add_evidence_marker(
        self,
        map_obj: folium.Map,
        evidence: Evidence,
        location: StandardCoordinate
    ):
        """Add evidence marker to map"""
        # Determine marker color based on evidence type
        color_map = {
            "witness": "blue",
            "document": "green",
            "photo": "orange",
            "video": "red",
            "report": "purple"
        }
        color = color_map.get(evidence.evidence_type.lower(), "gray")

        # Create popup content
        popup_html = f"""
        <div style="width: 200px;">
            <h4>{evidence.evidence_id}</h4>
            <p><b>Type:</b> {evidence.evidence_type}</p>
            <p><b>Description:</b> {evidence.description[:100]}...</p>
            {f'<p><b>Time:</b> {evidence.timestamp}</p>' if evidence.timestamp else ''}
        </div>
        """

        # Add marker
        folium.Marker(
            location=[location.latitude, location.longitude],
            popup=folium.Popup(popup_html, max_width=300),
            tooltip=f"{evidence.evidence_type}: {evidence.evidence_id}",
            icon=folium.Icon(color=color, icon='info-sign')
        ).add_to(map_obj)