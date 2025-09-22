"""
Lemkin Geospatial Analysis Suite - Core Module

This module provides the core data models and GeospatialAnalyzer class for 
geographic analysis of evidence without requiring GIS expertise.

Designed for legal professionals to analyze spatial relationships,
correlate events by location, and generate interactive maps.
"""

from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, validator
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CoordinateFormat(str, Enum):
    """Supported coordinate formats"""
    DECIMAL_DEGREES = "dd"          # 40.7128, -74.0060
    DEGREES_MINUTES_SECONDS = "dms" # 40°42'46.0"N 74°00'21.6"W
    UTM = "utm"                     # Universal Transverse Mercator
    MGRS = "mgrs"                   # Military Grid Reference System
    PLUS_CODES = "plus_codes"       # Open Location Code


class ProjectionSystem(str, Enum):
    """Supported coordinate projection systems"""
    WGS84 = "EPSG:4326"            # World Geodetic System 1984
    WEB_MERCATOR = "EPSG:3857"     # Web Mercator projection
    UTM_NORTH = "UTM_N"            # UTM Northern Hemisphere
    UTM_SOUTH = "UTM_S"            # UTM Southern Hemisphere


class SatelliteProvider(str, Enum):
    """Supported satellite imagery providers"""
    LANDSAT = "landsat"
    SENTINEL = "sentinel"
    MODIS = "modis"
    PLANET = "planet"
    MAXAR = "maxar"


class MapLayerType(str, Enum):
    """Types of map layers"""
    SATELLITE = "satellite"
    STREET = "street"
    TERRAIN = "terrain"
    HEAT_MAP = "heat_map"
    EVIDENCE = "evidence"
    GEOFENCE = "geofence"
    TRAJECTORY = "trajectory"


class EvidenceType(str, Enum):
    """Types of evidence with location data"""
    PHOTOGRAPH = "photograph"
    VIDEO = "video"
    WITNESS_TESTIMONY = "witness_testimony"
    DOCUMENT = "document"
    SOCIAL_MEDIA_POST = "social_media_post"
    SATELLITE_IMAGE = "satellite_image"
    MOBILE_DATA = "mobile_data"
    VEHICLE_TRACKING = "vehicle_tracking"


class GeoConfig(BaseModel):
    """Configuration for geospatial analysis operations"""
    
    # Coordinate handling
    default_projection: ProjectionSystem = Field(default=ProjectionSystem.WGS84)
    precision_meters: float = Field(default=1.0, ge=0.1, le=1000.0)
    
    # Analysis settings
    default_search_radius_km: float = Field(default=1.0, ge=0.01, le=100.0)
    temporal_tolerance_hours: int = Field(default=24, ge=1, le=8760)
    
    # Mapping settings
    default_zoom_level: int = Field(default=15, ge=1, le=20)
    max_points_per_layer: int = Field(default=1000, ge=10, le=10000)
    
    # Satellite imagery
    preferred_satellite: SatelliteProvider = Field(default=SatelliteProvider.LANDSAT)
    cloud_cover_threshold: float = Field(default=10.0, ge=0.0, le=100.0)
    
    # Privacy and ethics
    anonymize_sensitive_locations: bool = Field(default=True)
    blur_residential_areas: bool = Field(default=True)
    
    class Config:
        schema_extra = {
            "example": {
                "default_projection": "EPSG:4326",
                "precision_meters": 1.0,
                "default_search_radius_km": 1.0,
                "anonymize_sensitive_locations": True
            }
        }


class Coordinate(BaseModel):
    """Represents a geographic coordinate"""
    
    latitude: float = Field(..., ge=-90.0, le=90.0)
    longitude: float = Field(..., ge=-180.0, le=180.0)
    altitude: Optional[float] = Field(None, description="Altitude in meters")
    
    # Coordinate metadata
    format: CoordinateFormat = Field(default=CoordinateFormat.DECIMAL_DEGREES)
    projection: ProjectionSystem = Field(default=ProjectionSystem.WGS84)
    precision_meters: Optional[float] = Field(None, ge=0.0)
    
    # Source information
    source: Optional[str] = None
    collected_at: datetime = Field(default_factory=datetime.utcnow)
    
    @validator('latitude', 'longitude')
    def validate_coordinates(cls, v, field):
        if field.name == 'latitude' and not -90.0 <= v <= 90.0:
            raise ValueError('Latitude must be between -90 and 90 degrees')
        if field.name == 'longitude' and not -180.0 <= v <= 180.0:
            raise ValueError('Longitude must be between -180 and 180 degrees')
        return v
    
    def to_decimal_degrees(self) -> Tuple[float, float]:
        """Convert to decimal degrees format"""
        return (self.latitude, self.longitude)
    
    def distance_to(self, other: 'Coordinate') -> float:
        """Calculate distance to another coordinate in meters using Haversine formula"""
        import math
        
        # Convert to radians
        lat1_rad = math.radians(self.latitude)
        lon1_rad = math.radians(self.longitude)
        lat2_rad = math.radians(other.latitude)
        lon2_rad = math.radians(other.longitude)
        
        # Haversine formula
        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad
        
        a = (math.sin(dlat/2)**2 + 
             math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon/2)**2)
        c = 2 * math.asin(math.sqrt(a))
        
        # Earth's radius in meters
        earth_radius = 6371000
        
        return earth_radius * c

    class Config:
        schema_extra = {
            "example": {
                "latitude": 40.7128,
                "longitude": -74.0060,
                "format": "dd",
                "projection": "EPSG:4326",
                "source": "GPS device"
            }
        }


class BoundingBox(BaseModel):
    """Represents a geographic bounding box"""
    
    north: float = Field(..., ge=-90.0, le=90.0)
    south: float = Field(..., ge=-90.0, le=90.0)
    east: float = Field(..., ge=-180.0, le=180.0)
    west: float = Field(..., ge=-180.0, le=180.0)
    
    @validator('south')
    def validate_south_north(cls, v, values):
        if 'north' in values and v >= values['north']:
            raise ValueError('South latitude must be less than north latitude')
        return v
    
    @validator('west')
    def validate_west_east(cls, v, values):
        if 'east' in values and v >= values['east']:
            raise ValueError('West longitude must be less than east longitude')
        return v
    
    def center(self) -> Coordinate:
        """Get the center point of the bounding box"""
        center_lat = (self.north + self.south) / 2
        center_lon = (self.east + self.west) / 2
        return Coordinate(latitude=center_lat, longitude=center_lon)
    
    def area_km2(self) -> float:
        """Calculate approximate area in square kilometers"""
        # Simplified calculation - for more precision use proper geodesic calculations
        lat_diff = abs(self.north - self.south)
        lon_diff = abs(self.east - self.west)
        
        # Average latitude for longitude correction
        avg_lat = (self.north + self.south) / 2
        lon_correction = abs(math.cos(math.radians(avg_lat)))
        
        # Convert degrees to km (approximately)
        lat_km = lat_diff * 111.32
        lon_km = lon_diff * 111.32 * lon_correction
        
        return lat_km * lon_km

    class Config:
        schema_extra = {
            "example": {
                "north": 40.8,
                "south": 40.6,
                "east": -73.9,
                "west": -74.1
            }
        }


class DateRange(BaseModel):
    """Represents a time range for analysis"""
    
    start_date: datetime
    end_date: datetime
    
    @validator('end_date')
    def validate_date_range(cls, v, values):
        if 'start_date' in values and v <= values['start_date']:
            raise ValueError('End date must be after start date')
        return v
    
    def duration_days(self) -> int:
        """Get duration in days"""
        return (self.end_date - self.start_date).days


class Evidence(BaseModel):
    """Represents a piece of evidence with location data"""
    
    id: UUID = Field(default_factory=uuid4)
    title: str = Field(..., min_length=1)
    description: Optional[str] = None
    evidence_type: EvidenceType
    
    # Location data
    location: Optional[Coordinate] = None
    location_description: Optional[str] = None
    location_accuracy_meters: Optional[float] = Field(None, ge=0.0)
    
    # Temporal data
    timestamp: Optional[datetime] = None
    timestamp_precision: Optional[str] = None  # "exact", "approximate", "unknown"
    
    # File information
    file_path: Optional[str] = None
    file_hash: Optional[str] = None
    
    # Metadata
    source: Optional[str] = None
    collector: Optional[str] = None
    collected_at: datetime = Field(default_factory=datetime.utcnow)
    
    # Chain of custody
    custody_log: List[str] = Field(default_factory=list)
    verified: bool = Field(default=False)
    verification_notes: Optional[str] = None
    
    class Config:
        schema_extra = {
            "example": {
                "title": "Photograph of incident location",
                "evidence_type": "photograph",
                "location": {
                    "latitude": 40.7128,
                    "longitude": -74.0060
                },
                "timestamp": "2024-01-15T14:30:00Z",
                "source": "Witness smartphone"
            }
        }


class SpatialEvent(BaseModel):
    """Represents an event with spatial and temporal dimensions"""
    
    id: UUID = Field(default_factory=uuid4)
    name: str = Field(..., min_length=1)
    description: Optional[str] = None
    event_type: str
    
    # Location data
    primary_location: Optional[Coordinate] = None
    secondary_locations: List[Coordinate] = Field(default_factory=list)
    area_of_interest: Optional[BoundingBox] = None
    
    # Temporal data
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    duration_minutes: Optional[int] = Field(None, ge=0)
    
    # Associated evidence
    evidence_ids: List[UUID] = Field(default_factory=list)
    
    # Analysis metadata
    confidence_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    analysis_notes: Optional[str] = None
    
    class Config:
        schema_extra = {
            "example": {
                "name": "Incident at Market Square",
                "event_type": "civil_disturbance",
                "primary_location": {
                    "latitude": 40.7128,
                    "longitude": -74.0060
                },
                "start_time": "2024-01-15T14:00:00Z",
                "end_time": "2024-01-15T16:00:00Z"
            }
        }


class SatelliteAnalysis(BaseModel):
    """Results from satellite imagery analysis"""
    
    id: UUID = Field(default_factory=uuid4)
    analysis_name: str = Field(..., min_length=1)
    
    # Analysis parameters
    bounding_box: BoundingBox
    date_range: DateRange
    satellite_provider: SatelliteProvider
    
    # Imagery metadata
    images_analyzed: List[Dict[str, Any]] = Field(default_factory=list)
    cloud_cover_percentage: Optional[float] = Field(None, ge=0.0, le=100.0)
    resolution_meters: Optional[float] = Field(None, ge=0.1)
    
    # Analysis results
    changes_detected: List[Dict[str, Any]] = Field(default_factory=list)
    points_of_interest: List[Coordinate] = Field(default_factory=list)
    area_measurements: Dict[str, float] = Field(default_factory=dict)
    
    # Processing metadata
    analyzed_at: datetime = Field(default_factory=datetime.utcnow)
    processing_time_seconds: Optional[float] = Field(None, ge=0.0)
    algorithm_version: str = Field(default="1.0")
    
    class Config:
        schema_extra = {
            "example": {
                "analysis_name": "Infrastructure Change Detection",
                "satellite_provider": "landsat",
                "cloud_cover_percentage": 5.2,
                "changes_detected": [
                    {"type": "new_construction", "confidence": 0.85}
                ]
            }
        }


class LocationCorrelation(BaseModel):
    """Results from location-based event correlation"""
    
    id: UUID = Field(default_factory=uuid4)
    analysis_name: str = Field(..., min_length=1)
    
    # Analysis parameters
    search_radius_meters: float = Field(..., ge=1.0)
    time_window_hours: Optional[int] = Field(None, ge=1)
    
    # Input data
    events_analyzed: List[UUID] = Field(default_factory=list)
    evidence_analyzed: List[UUID] = Field(default_factory=list)
    
    # Correlation results
    spatial_clusters: List[Dict[str, Any]] = Field(default_factory=list)
    temporal_correlations: List[Dict[str, Any]] = Field(default_factory=list)
    proximity_matches: List[Dict[str, Any]] = Field(default_factory=list)
    
    # Statistical analysis
    correlation_strength: Optional[float] = Field(None, ge=0.0, le=1.0)
    statistical_significance: Optional[float] = Field(None, ge=0.0, le=1.0)
    
    # Analysis metadata
    analyzed_at: datetime = Field(default_factory=datetime.utcnow)
    analysis_method: str = Field(default="proximity_clustering")
    
    class Config:
        schema_extra = {
            "example": {
                "analysis_name": "Event Proximity Analysis",
                "search_radius_meters": 500.0,
                "correlation_strength": 0.75,
                "spatial_clusters": [
                    {"cluster_id": 1, "event_count": 3, "center": [40.7128, -74.0060]}
                ]
            }
        }


class GeofenceResult(BaseModel):
    """Results from geofencing analysis"""
    
    id: UUID = Field(default_factory=uuid4)
    geofence_name: str = Field(..., min_length=1)
    
    # Geofence definition
    center: Coordinate
    radius_meters: float = Field(..., ge=1.0)
    polygon_vertices: Optional[List[Coordinate]] = None
    
    # Analysis results
    events_inside: List[UUID] = Field(default_factory=list)
    events_outside: List[UUID] = Field(default_factory=list)
    boundary_crossings: List[Dict[str, Any]] = Field(default_factory=list)
    
    # Temporal analysis
    time_periods: List[Dict[str, Any]] = Field(default_factory=list)
    duration_inside_minutes: Optional[float] = Field(None, ge=0.0)
    
    # Analysis metadata
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        schema_extra = {
            "example": {
                "geofence_name": "Restricted Area Monitor",
                "radius_meters": 1000.0,
                "events_inside": ["event-uuid-1", "event-uuid-2"],
                "boundary_crossings": [
                    {"timestamp": "2024-01-15T14:30:00Z", "direction": "entry"}
                ]
            }
        }


class MapLayer(BaseModel):
    """Represents a layer on an interactive map"""
    
    name: str = Field(..., min_length=1)
    layer_type: MapLayerType
    visible: bool = Field(default=True)
    opacity: float = Field(default=1.0, ge=0.0, le=1.0)
    
    # Layer data
    coordinates: List[Coordinate] = Field(default_factory=list)
    properties: Dict[str, Any] = Field(default_factory=dict)
    style: Dict[str, Any] = Field(default_factory=dict)
    
    # Layer metadata
    created_at: datetime = Field(default_factory=datetime.utcnow)
    data_source: Optional[str] = None


class InteractiveMap(BaseModel):
    """Represents an interactive map with evidence overlay"""
    
    id: UUID = Field(default_factory=uuid4)
    title: str = Field(..., min_length=1)
    description: Optional[str] = None
    
    # Map configuration
    center: Coordinate
    zoom_level: int = Field(default=15, ge=1, le=20)
    map_provider: str = Field(default="OpenStreetMap")
    
    # Map layers
    layers: List[MapLayer] = Field(default_factory=list)
    evidence_points: List[Evidence] = Field(default_factory=list)
    
    # Interactive features
    clustering_enabled: bool = Field(default=True)
    popup_templates: Dict[str, str] = Field(default_factory=dict)
    legend_enabled: bool = Field(default=True)
    
    # Export formats
    supported_formats: List[str] = Field(default_factory=lambda: ["html", "pdf", "kml", "geojson"])
    
    # Creation metadata
    created_at: datetime = Field(default_factory=datetime.utcnow)
    created_by: Optional[str] = None
    
    class Config:
        schema_extra = {
            "example": {
                "title": "Evidence Location Map",
                "center": {"latitude": 40.7128, "longitude": -74.0060},
                "zoom_level": 15,
                "clustering_enabled": True
            }
        }


class GeospatialAnalyzer:
    """
    Main geospatial analysis class providing geographic analysis capabilities
    for legal evidence without requiring GIS expertise.
    
    Designed to be user-friendly for legal professionals while providing
    powerful spatial analysis capabilities.
    """
    
    def __init__(self, config: Optional[GeoConfig] = None):
        """Initialize geospatial analyzer with configuration"""
        self.config = config or GeoConfig()
        self.logger = logging.getLogger(f"{__name__}.GeospatialAnalyzer")
        
        # Initialize component modules (will be populated by specific modules)
        self._coordinate_converter = None
        self._satellite_analyzer = None
        self._geofence_processor = None
        self._mapping_generator = None
        
        # Storage for analysis data
        self.evidence_collection: List[Evidence] = []
        self.spatial_events: List[SpatialEvent] = []
        self.analysis_cache: Dict[str, Any] = {}
        
        self.logger.info("Geospatial Analyzer initialized")
    
    def add_evidence(self, evidence: Evidence) -> None:
        """Add evidence to the collection for analysis"""
        self.evidence_collection.append(evidence)
        evidence.custody_log.append(f"Added to analysis collection at {datetime.utcnow().isoformat()}")
        self.logger.info(f"Added evidence: {evidence.title} (ID: {evidence.id})")
    
    def add_spatial_event(self, event: SpatialEvent) -> None:
        """Add a spatial event to the collection"""
        self.spatial_events.append(event)
        self.logger.info(f"Added spatial event: {event.name} (ID: {event.id})")
    
    def get_evidence_in_radius(
        self, 
        center: Coordinate, 
        radius_meters: float
    ) -> List[Evidence]:
        """Get all evidence within a specified radius of a center point"""
        evidence_in_radius = []
        
        for evidence in self.evidence_collection:
            if evidence.location:
                distance = center.distance_to(evidence.location)
                if distance <= radius_meters:
                    evidence_in_radius.append(evidence)
        
        self.logger.info(f"Found {len(evidence_in_radius)} evidence items within {radius_meters}m of center")
        return evidence_in_radius
    
    def get_evidence_in_timeframe(
        self, 
        start_time: datetime, 
        end_time: datetime
    ) -> List[Evidence]:
        """Get all evidence within a specified time frame"""
        evidence_in_timeframe = []
        
        for evidence in self.evidence_collection:
            if evidence.timestamp and start_time <= evidence.timestamp <= end_time:
                evidence_in_timeframe.append(evidence)
        
        self.logger.info(f"Found {len(evidence_in_timeframe)} evidence items in timeframe")
        return evidence_in_timeframe
    
    def calculate_center_point(self, coordinates: List[Coordinate]) -> Coordinate:
        """Calculate the geographic center of a list of coordinates"""
        if not coordinates:
            raise ValueError("Cannot calculate center of empty coordinate list")
        
        total_lat = sum(coord.latitude for coord in coordinates)
        total_lon = sum(coord.longitude for coord in coordinates)
        count = len(coordinates)
        
        center = Coordinate(
            latitude=total_lat / count,
            longitude=total_lon / count,
            source="calculated_center"
        )
        
        return center
    
    def generate_bounding_box(
        self, 
        coordinates: List[Coordinate], 
        buffer_meters: float = 0
    ) -> BoundingBox:
        """Generate a bounding box that encompasses all coordinates with optional buffer"""
        if not coordinates:
            raise ValueError("Cannot generate bounding box from empty coordinate list")
        
        lats = [coord.latitude for coord in coordinates]
        lons = [coord.longitude for coord in coordinates]
        
        # Add buffer if specified (rough conversion from meters to degrees)
        buffer_degrees = buffer_meters / 111320  # Approximate meters per degree
        
        return BoundingBox(
            north=max(lats) + buffer_degrees,
            south=min(lats) - buffer_degrees,
            east=max(lons) + buffer_degrees,
            west=min(lons) - buffer_degrees
        )
    
    def validate_coordinates(self, coordinate: Coordinate) -> bool:
        """Validate coordinate data for common issues"""
        try:
            # Check coordinate bounds
            if not (-90 <= coordinate.latitude <= 90):
                self.logger.warning(f"Invalid latitude: {coordinate.latitude}")
                return False
            
            if not (-180 <= coordinate.longitude <= 180):
                self.logger.warning(f"Invalid longitude: {coordinate.longitude}")
                return False
            
            # Check for null island (0,0) which is often an error
            if coordinate.latitude == 0 and coordinate.longitude == 0:
                self.logger.warning("Coordinate at (0,0) - possible GPS error")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Coordinate validation failed: {str(e)}")
            return False
    
    def export_analysis(
        self, 
        output_path: Path, 
        format: str = "json",
        include_chain_of_custody: bool = True
    ) -> bool:
        """Export analysis results with full documentation"""
        try:
            export_data = {
                'analysis_metadata': {
                    'analyzer_version': '1.0.0',
                    'export_timestamp': datetime.utcnow().isoformat(),
                    'config': self.config.dict()
                },
                'evidence_collection': [evidence.dict() for evidence in self.evidence_collection],
                'spatial_events': [event.dict() for event in self.spatial_events],
                'analysis_cache': self.analysis_cache
            }
            
            if include_chain_of_custody:
                export_data['chain_of_custody'] = self._generate_custody_report()
            
            if format.lower() == "json":
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(export_data, f, indent=2, default=str)
                
                self.logger.info(f"Analysis exported to {output_path}")
                return True
            else:
                self.logger.error(f"Unsupported export format: {format}")
                return False
                
        except Exception as e:
            self.logger.error(f"Export failed: {str(e)}")
            return False
    
    def _generate_custody_report(self) -> Dict[str, Any]:
        """Generate chain of custody report for analysis integrity"""
        return {
            'evidence_count': len(self.evidence_collection),
            'events_count': len(self.spatial_events),
            'analysis_config': self.config.dict(),
            'creation_timestamp': datetime.utcnow().isoformat(),
            'evidence_integrity': {
                evidence.id: {
                    'title': evidence.title,
                    'custody_log': evidence.custody_log,
                    'verified': evidence.verified,
                    'file_hash': evidence.file_hash
                }
                for evidence in self.evidence_collection
            }
        }

# Import math for calculations
import math