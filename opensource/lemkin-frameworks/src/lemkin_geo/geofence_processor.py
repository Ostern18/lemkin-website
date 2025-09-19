"""
Lemkin Geospatial Analysis Suite - Geofence Processor Module

This module provides location-based event correlation and geofencing capabilities
for legal evidence analysis. Enables analysis of spatial relationships between
events, evidence, and defined geographic boundaries.

Features: Proximity analysis, geofencing, trajectory analysis, spatial clustering.
"""

import math
import json
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union
from enum import Enum
from dataclasses import dataclass
import logging

from .core import (
    Coordinate, 
    BoundingBox, 
    Evidence, 
    SpatialEvent, 
    LocationCorrelation, 
    GeofenceResult,
    GeoConfig
)

logger = logging.getLogger(__name__)


class GeofenceType(str, Enum):
    """Types of geofences"""
    CIRCULAR = "circular"
    RECTANGULAR = "rectangular"
    POLYGON = "polygon"
    BUFFER_ZONE = "buffer_zone"


class ProximityRelation(str, Enum):
    """Types of proximity relationships"""
    WITHIN = "within"
    OVERLAPPING = "overlapping"
    ADJACENT = "adjacent"
    DISTANT = "distant"


class CorrelationMethod(str, Enum):
    """Methods for spatial correlation analysis"""
    DISTANCE_BASED = "distance_based"
    DENSITY_CLUSTERING = "density_clustering"
    TEMPORAL_SPATIAL = "temporal_spatial"
    HOTSPOT_ANALYSIS = "hotspot_analysis"


@dataclass
class GeofenceDefinition:
    """Definition of a geographic fence"""
    name: str
    geofence_type: GeofenceType
    center: Optional[Coordinate] = None
    radius_meters: Optional[float] = None
    bounding_box: Optional[BoundingBox] = None
    polygon_vertices: Optional[List[Coordinate]] = None
    description: Optional[str] = None
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()


@dataclass
class ProximityMatch:
    """Result of proximity analysis between two spatial objects"""
    object1_id: str
    object2_id: str
    distance_meters: float
    relation: ProximityRelation
    confidence: float
    matched_at: datetime = None
    
    def __post_init__(self):
        if self.matched_at is None:
            self.matched_at = datetime.utcnow()


@dataclass  
class SpatialCluster:
    """A cluster of spatially related objects"""
    cluster_id: str
    center: Coordinate
    radius_meters: float
    object_ids: List[str]
    cluster_strength: float
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()


class GeofenceProcessor:
    """
    Location-based event correlation and geofencing processor.
    
    Provides spatial analysis capabilities for legal evidence investigation
    including proximity analysis, geofencing, and spatial clustering.
    """
    
    def __init__(self, config: Optional[GeoConfig] = None):
        """Initialize geofence processor"""
        self.config = config or GeoConfig()
        self.logger = logging.getLogger(f"{__name__}.GeofenceProcessor")
        
        # Storage for geofences and analysis results
        self.geofences: Dict[str, GeofenceDefinition] = {}
        self.analysis_cache: Dict[str, Any] = {}
        
        self.logger.info("Geofence Processor initialized")
    
    def correlate_events_by_location(
        self, 
        events: List[Union[SpatialEvent, Evidence]], 
        radius: float,
        method: CorrelationMethod = CorrelationMethod.DISTANCE_BASED,
        time_window_hours: Optional[int] = None
    ) -> LocationCorrelation:
        """
        Correlate events based on spatial proximity and optional temporal constraints
        
        Args:
            events: List of spatial events or evidence to correlate
            radius: Search radius in meters
            method: Correlation analysis method
            time_window_hours: Optional temporal constraint in hours
            
        Returns:
            LocationCorrelation with analysis results
        """
        try:
            correlation = LocationCorrelation(
                analysis_name=f"Location Correlation ({method})",
                search_radius_meters=radius,
                time_window_hours=time_window_hours
            )
            
            # Extract coordinates and IDs from events
            spatial_objects = []
            for event in events:
                obj_data = self._extract_spatial_data(event)
                if obj_data:
                    spatial_objects.append(obj_data)
                    
                    # Record analyzed object IDs
                    if hasattr(event, 'id'):
                        if isinstance(event, SpatialEvent):
                            correlation.events_analyzed.append(event.id)
                        elif isinstance(event, Evidence):
                            correlation.evidence_analyzed.append(event.id)
            
            if len(spatial_objects) < 2:
                self.logger.warning("Insufficient spatial objects for correlation analysis")
                return correlation
            
            # Perform correlation analysis based on method
            if method == CorrelationMethod.DISTANCE_BASED:
                correlation = self._distance_based_correlation(spatial_objects, correlation, radius)
            elif method == CorrelationMethod.DENSITY_CLUSTERING:
                correlation = self._density_clustering_correlation(spatial_objects, correlation, radius)
            elif method == CorrelationMethod.TEMPORAL_SPATIAL:
                correlation = self._temporal_spatial_correlation(spatial_objects, correlation, radius, time_window_hours)
            elif method == CorrelationMethod.HOTSPOT_ANALYSIS:
                correlation = self._hotspot_analysis_correlation(spatial_objects, correlation, radius)
            
            # Calculate correlation strength
            correlation.correlation_strength = self._calculate_correlation_strength(correlation)
            correlation.statistical_significance = self._calculate_statistical_significance(correlation)
            
            self.logger.info(f"Location correlation completed: {len(correlation.proximity_matches)} matches found")
            return correlation
            
        except Exception as e:
            self.logger.error(f"Location correlation failed: {str(e)}")
            correlation = LocationCorrelation(
                analysis_name=f"Failed Correlation ({method})",
                search_radius_meters=radius,
                time_window_hours=time_window_hours
            )
            correlation.proximity_matches = [{'error': str(e)}]
            return correlation
    
    def create_geofence(
        self,
        name: str,
        geofence_type: GeofenceType,
        center: Optional[Coordinate] = None,
        radius_meters: Optional[float] = None,
        bounding_box: Optional[BoundingBox] = None,
        polygon_vertices: Optional[List[Coordinate]] = None
    ) -> str:
        """
        Create a new geofence definition
        
        Args:
            name: Name for the geofence
            geofence_type: Type of geofence
            center: Center point (required for circular geofences)
            radius_meters: Radius in meters (required for circular geofences)
            bounding_box: Bounding box (for rectangular geofences)
            polygon_vertices: Vertices for polygon geofences
            
        Returns:
            Geofence ID
        """
        try:
            # Validate geofence parameters
            if geofence_type == GeofenceType.CIRCULAR:
                if not center or not radius_meters:
                    raise ValueError("Circular geofence requires center and radius")
            
            elif geofence_type == GeofenceType.RECTANGULAR:
                if not bounding_box:
                    raise ValueError("Rectangular geofence requires bounding_box")
            
            elif geofence_type == GeofenceType.POLYGON:
                if not polygon_vertices or len(polygon_vertices) < 3:
                    raise ValueError("Polygon geofence requires at least 3 vertices")
            
            # Create geofence definition
            geofence = GeofenceDefinition(
                name=name,
                geofence_type=geofence_type,
                center=center,
                radius_meters=radius_meters,
                bounding_box=bounding_box,
                polygon_vertices=polygon_vertices
            )
            
            # Generate unique ID
            geofence_id = f"geofence_{len(self.geofences)}_{hash(name) % 10000}"
            self.geofences[geofence_id] = geofence
            
            self.logger.info(f"Created geofence: {name} (ID: {geofence_id})")
            return geofence_id
            
        except Exception as e:
            self.logger.error(f"Geofence creation failed: {str(e)}")
            raise
    
    def analyze_geofence_violations(
        self,
        geofence_id: str,
        events: List[Union[SpatialEvent, Evidence]],
        time_range: Optional[Tuple[datetime, datetime]] = None
    ) -> GeofenceResult:
        """
        Analyze events relative to a geofence
        
        Args:
            geofence_id: ID of the geofence to analyze
            events: List of events to check against geofence
            time_range: Optional time range filter
            
        Returns:
            GeofenceResult with analysis
        """
        try:
            if geofence_id not in self.geofences:
                raise ValueError(f"Geofence {geofence_id} not found")
            
            geofence = self.geofences[geofence_id]
            
            result = GeofenceResult(
                geofence_name=geofence.name,
                center=geofence.center or Coordinate(latitude=0, longitude=0),
                radius_meters=geofence.radius_meters or 0.0,
                polygon_vertices=geofence.polygon_vertices
            )
            
            # Filter events by time range if specified
            filtered_events = events
            if time_range:
                start_time, end_time = time_range
                filtered_events = [
                    event for event in events
                    if self._event_in_time_range(event, start_time, end_time)
                ]
            
            # Analyze each event relative to geofence
            total_time_inside = 0.0
            boundary_crossings = []
            
            for event in filtered_events:
                spatial_data = self._extract_spatial_data(event)
                if not spatial_data:
                    continue
                
                is_inside = self._point_in_geofence(spatial_data['location'], geofence)
                event_id = str(spatial_data['id'])
                
                if is_inside:
                    result.events_inside.append(event_id)
                    
                    # Calculate time spent inside (simplified)
                    if hasattr(event, 'duration_minutes') and event.duration_minutes:
                        total_time_inside += event.duration_minutes
                    else:
                        total_time_inside += 60  # Default 1 hour
                else:
                    result.events_outside.append(event_id)
                
                # Detect boundary crossings (simplified implementation)
                if hasattr(event, 'secondary_locations'):
                    for secondary_loc in event.secondary_locations or []:
                        secondary_inside = self._point_in_geofence(secondary_loc, geofence)
                        if is_inside != secondary_inside:
                            boundary_crossings.append({
                                'event_id': event_id,
                                'timestamp': getattr(event, 'start_time', datetime.utcnow()).isoformat(),
                                'direction': 'exit' if is_inside else 'entry',
                                'location': secondary_loc.dict()
                            })
            
            result.boundary_crossings = boundary_crossings
            result.duration_inside_minutes = total_time_inside
            
            # Generate time period analysis
            result.time_periods = self._analyze_time_periods(result.events_inside, filtered_events)
            
            self.logger.info(
                f"Geofence analysis completed: {len(result.events_inside)} inside, "
                f"{len(result.events_outside)} outside"
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Geofence analysis failed: {str(e)}")
            # Return empty result with error info
            result = GeofenceResult(
                geofence_name=f"Error: {geofence_id}",
                center=Coordinate(latitude=0, longitude=0),
                radius_meters=0.0
            )
            result.boundary_crossings = [{'error': str(e)}]
            return result
    
    def find_spatial_clusters(
        self,
        events: List[Union[SpatialEvent, Evidence]],
        min_cluster_size: int = 3,
        max_distance_meters: float = 100.0
    ) -> List[SpatialCluster]:
        """
        Find spatial clusters of events using density-based clustering
        
        Args:
            events: Events to cluster
            min_cluster_size: Minimum number of events in a cluster
            max_distance_meters: Maximum distance between cluster members
            
        Returns:
            List of spatial clusters
        """
        try:
            # Extract spatial data
            spatial_objects = []
            for event in events:
                obj_data = self._extract_spatial_data(event)
                if obj_data:
                    spatial_objects.append(obj_data)
            
            if len(spatial_objects) < min_cluster_size:
                return []
            
            clusters = []
            unassigned = spatial_objects.copy()
            cluster_id = 0
            
            while len(unassigned) >= min_cluster_size:
                # Start new cluster with first unassigned point
                seed_point = unassigned[0]
                cluster_members = [seed_point]
                unassigned.remove(seed_point)
                
                # Find all points within distance of cluster members
                expanded = True
                while expanded:
                    expanded = False
                    new_members = []
                    
                    for member in cluster_members:
                        for candidate in unassigned[:]:  # Use slice to allow modification
                            distance = member['location'].distance_to(candidate['location'])
                            if distance <= max_distance_meters:
                                new_members.append(candidate)
                                unassigned.remove(candidate)
                                expanded = True
                    
                    cluster_members.extend(new_members)
                
                # Create cluster if it meets minimum size requirement
                if len(cluster_members) >= min_cluster_size:
                    cluster_center = self._calculate_cluster_center(cluster_members)
                    cluster_radius = self._calculate_cluster_radius(cluster_members, cluster_center)
                    
                    cluster = SpatialCluster(
                        cluster_id=f"cluster_{cluster_id}",
                        center=cluster_center,
                        radius_meters=cluster_radius,
                        object_ids=[str(member['id']) for member in cluster_members],
                        cluster_strength=self._calculate_cluster_strength(cluster_members)
                    )
                    
                    clusters.append(cluster)
                    cluster_id += 1
                else:
                    # Return members to unassigned if cluster too small
                    unassigned.extend(cluster_members[1:])  # Keep seed point removed
            
            self.logger.info(f"Found {len(clusters)} spatial clusters")
            return clusters
            
        except Exception as e:
            self.logger.error(f"Spatial clustering failed: {str(e)}")
            return []
    
    def calculate_event_density(
        self,
        events: List[Union[SpatialEvent, Evidence]],
        grid_size_meters: float = 100.0,
        bbox: Optional[BoundingBox] = None
    ) -> Dict[str, Any]:
        """
        Calculate event density across a geographic area
        
        Args:
            events: Events to analyze
            grid_size_meters: Size of density grid cells
            bbox: Bounding box for analysis (auto-calculated if not provided)
            
        Returns:
            Dictionary with density analysis results
        """
        try:
            spatial_objects = []
            for event in events:
                obj_data = self._extract_spatial_data(event)
                if obj_data:
                    spatial_objects.append(obj_data)
            
            if not spatial_objects:
                return {'error': 'No spatial objects found'}
            
            # Calculate bounding box if not provided
            if not bbox:
                coordinates = [obj['location'] for obj in spatial_objects]
                bbox = self._calculate_bounding_box(coordinates)
            
            # Create density grid
            grid_size_degrees = grid_size_meters / 111320  # Rough conversion
            
            lat_steps = int((bbox.north - bbox.south) / grid_size_degrees) + 1
            lon_steps = int((bbox.east - bbox.west) / grid_size_degrees) + 1
            
            density_grid = {}
            hotspots = []
            
            for i in range(lat_steps):
                for j in range(lon_steps):
                    cell_south = bbox.south + i * grid_size_degrees
                    cell_north = min(bbox.north, cell_south + grid_size_degrees)
                    cell_west = bbox.west + j * grid_size_degrees
                    cell_east = min(bbox.east, cell_west + grid_size_degrees)
                    
                    cell_bbox = BoundingBox(
                        north=cell_north,
                        south=cell_south,
                        east=cell_east,
                        west=cell_west
                    )
                    
                    # Count events in cell
                    cell_count = 0
                    for obj in spatial_objects:
                        if self._point_in_bbox(obj['location'], cell_bbox):
                            cell_count += 1
                    
                    if cell_count > 0:
                        cell_key = f"{i}_{j}"
                        cell_center = cell_bbox.center()
                        
                        density_grid[cell_key] = {
                            'center': cell_center.dict(),
                            'count': cell_count,
                            'density_per_km2': cell_count / (grid_size_meters/1000)**2
                        }
                        
                        # Identify hotspots (cells with high density)
                        if cell_count >= 3:  # Threshold for hotspot
                            hotspots.append({
                                'location': cell_center.dict(),
                                'event_count': cell_count,
                                'density_score': cell_count / max(1, len(spatial_objects)) * 100
                            })
            
            return {
                'total_events': len(spatial_objects),
                'analysis_area_km2': bbox.area_km2(),
                'grid_cells': len(density_grid),
                'hotspots': sorted(hotspots, key=lambda x: x['event_count'], reverse=True),
                'density_grid': density_grid,
                'analysis_parameters': {
                    'grid_size_meters': grid_size_meters,
                    'bounding_box': bbox.dict()
                }
            }
            
        except Exception as e:
            self.logger.error(f"Density analysis failed: {str(e)}")
            return {'error': str(e)}
    
    def _extract_spatial_data(self, event: Union[SpatialEvent, Evidence]) -> Optional[Dict[str, Any]]:
        """Extract spatial data from event or evidence"""
        try:
            if isinstance(event, SpatialEvent):
                if event.primary_location:
                    return {
                        'id': event.id,
                        'location': event.primary_location,
                        'timestamp': event.start_time,
                        'type': 'spatial_event',
                        'name': event.name
                    }
            elif isinstance(event, Evidence):
                if event.location:
                    return {
                        'id': event.id,
                        'location': event.location,
                        'timestamp': event.timestamp,
                        'type': 'evidence',
                        'name': event.title
                    }
        except Exception as e:
            self.logger.warning(f"Failed to extract spatial data: {str(e)}")
        
        return None
    
    def _distance_based_correlation(
        self,
        spatial_objects: List[Dict[str, Any]],
        correlation: LocationCorrelation,
        radius: float
    ) -> LocationCorrelation:
        """Perform distance-based correlation analysis"""
        proximity_matches = []
        
        for i, obj1 in enumerate(spatial_objects):
            for obj2 in spatial_objects[i+1:]:
                distance = obj1['location'].distance_to(obj2['location'])
                
                if distance <= radius:
                    relation = ProximityRelation.WITHIN
                    confidence = max(0.1, 1.0 - (distance / radius))
                    
                    match = {
                        'object1_id': str(obj1['id']),
                        'object2_id': str(obj2['id']),
                        'distance_meters': distance,
                        'relation': relation,
                        'confidence': confidence,
                        'object1_name': obj1.get('name', 'Unknown'),
                        'object2_name': obj2.get('name', 'Unknown')
                    }
                    
                    proximity_matches.append(match)
        
        correlation.proximity_matches = proximity_matches
        return correlation
    
    def _density_clustering_correlation(
        self,
        spatial_objects: List[Dict[str, Any]],
        correlation: LocationCorrelation,
        radius: float
    ) -> LocationCorrelation:
        """Perform density-based clustering correlation"""
        # Use spatial clustering to find correlations
        events_for_clustering = []
        for obj in spatial_objects:
            # Create mock event for clustering
            mock_event = type('MockEvent', (), {
                'id': obj['id'],
                'location': obj['location']
            })()
            events_for_clustering.append(mock_event)
        
        clusters = self.find_spatial_clusters(events_for_clustering, min_cluster_size=2, max_distance_meters=radius)
        
        cluster_data = []
        for cluster in clusters:
            cluster_info = {
                'cluster_id': cluster.cluster_id,
                'center': cluster.center.dict(),
                'radius_meters': cluster.radius_meters,
                'object_count': len(cluster.object_ids),
                'strength': cluster.cluster_strength,
                'object_ids': cluster.object_ids
            }
            cluster_data.append(cluster_info)
        
        correlation.spatial_clusters = cluster_data
        return correlation
    
    def _temporal_spatial_correlation(
        self,
        spatial_objects: List[Dict[str, Any]],
        correlation: LocationCorrelation,
        radius: float,
        time_window_hours: Optional[int]
    ) -> LocationCorrelation:
        """Perform temporal-spatial correlation analysis"""
        if not time_window_hours:
            return self._distance_based_correlation(spatial_objects, correlation, radius)
        
        temporal_correlations = []
        time_window = timedelta(hours=time_window_hours)
        
        for i, obj1 in enumerate(spatial_objects):
            for obj2 in spatial_objects[i+1:]:
                # Check spatial proximity
                distance = obj1['location'].distance_to(obj2['location'])
                if distance > radius:
                    continue
                
                # Check temporal proximity
                time1 = obj1.get('timestamp')
                time2 = obj2.get('timestamp')
                
                if time1 and time2:
                    time_diff = abs((time1 - time2).total_seconds())
                    if time_diff <= time_window.total_seconds():
                        temporal_correlation = {
                            'object1_id': str(obj1['id']),
                            'object2_id': str(obj2['id']),
                            'spatial_distance_meters': distance,
                            'temporal_distance_seconds': time_diff,
                            'correlation_strength': max(0.1, 1.0 - (distance / radius) - (time_diff / time_window.total_seconds()))
                        }
                        temporal_correlations.append(temporal_correlation)
        
        correlation.temporal_correlations = temporal_correlations
        return correlation
    
    def _hotspot_analysis_correlation(
        self,
        spatial_objects: List[Dict[str, Any]],
        correlation: LocationCorrelation,
        radius: float
    ) -> LocationCorrelation:
        """Perform hotspot analysis correlation"""
        # Calculate event density to identify hotspots
        events_for_density = []
        for obj in spatial_objects:
            mock_event = type('MockEvent', (), {
                'id': obj['id'],
                'location': obj['location']
            })()
            events_for_density.append(mock_event)
        
        density_analysis = self.calculate_event_density(events_for_density, grid_size_meters=radius)
        
        # Store hotspot information in correlation
        correlation.spatial_clusters = [{
            'analysis_type': 'hotspot',
            'hotspots': density_analysis.get('hotspots', []),
            'total_events': density_analysis.get('total_events', 0),
            'analysis_area_km2': density_analysis.get('analysis_area_km2', 0)
        }]
        
        return correlation
    
    def _calculate_correlation_strength(self, correlation: LocationCorrelation) -> float:
        """Calculate overall correlation strength"""
        total_strength = 0.0
        count = 0
        
        # Factor in proximity matches
        for match in correlation.proximity_matches:
            if isinstance(match, dict) and 'confidence' in match:
                total_strength += match['confidence']
                count += 1
        
        # Factor in temporal correlations
        for temp_corr in correlation.temporal_correlations:
            if isinstance(temp_corr, dict) and 'correlation_strength' in temp_corr:
                total_strength += temp_corr['correlation_strength']
                count += 1
        
        return total_strength / max(count, 1)
    
    def _calculate_statistical_significance(self, correlation: LocationCorrelation) -> float:
        """Calculate statistical significance of correlation"""
        # Simplified significance calculation
        total_matches = len(correlation.proximity_matches) + len(correlation.temporal_correlations)
        total_analyzed = len(correlation.events_analyzed) + len(correlation.evidence_analyzed)
        
        if total_analyzed < 2:
            return 0.0
        
        match_ratio = total_matches / (total_analyzed * (total_analyzed - 1) / 2)
        return min(1.0, match_ratio * 2)  # Simplified significance score
    
    def _point_in_geofence(self, point: Coordinate, geofence: GeofenceDefinition) -> bool:
        """Check if point is inside geofence"""
        try:
            if geofence.geofence_type == GeofenceType.CIRCULAR:
                if geofence.center and geofence.radius_meters:
                    distance = point.distance_to(geofence.center)
                    return distance <= geofence.radius_meters
            
            elif geofence.geofence_type == GeofenceType.RECTANGULAR:
                if geofence.bounding_box:
                    return self._point_in_bbox(point, geofence.bounding_box)
            
            elif geofence.geofence_type == GeofenceType.POLYGON:
                if geofence.polygon_vertices:
                    return self._point_in_polygon(point, geofence.polygon_vertices)
            
        except Exception as e:
            self.logger.error(f"Point-in-geofence check failed: {str(e)}")
        
        return False
    
    def _point_in_bbox(self, point: Coordinate, bbox: BoundingBox) -> bool:
        """Check if point is inside bounding box"""
        return (bbox.south <= point.latitude <= bbox.north and
                bbox.west <= point.longitude <= bbox.east)
    
    def _point_in_polygon(self, point: Coordinate, vertices: List[Coordinate]) -> bool:
        """Check if point is inside polygon using ray casting algorithm"""
        x, y = point.longitude, point.latitude
        n = len(vertices)
        inside = False
        
        p1x, p1y = vertices[0].longitude, vertices[0].latitude
        for i in range(1, n + 1):
            p2x, p2y = vertices[i % n].longitude, vertices[i % n].latitude
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        
        return inside
    
    def _event_in_time_range(self, event: Union[SpatialEvent, Evidence], start_time: datetime, end_time: datetime) -> bool:
        """Check if event falls within time range"""
        event_time = None
        
        if isinstance(event, SpatialEvent):
            event_time = event.start_time
        elif isinstance(event, Evidence):
            event_time = event.timestamp
        
        if not event_time:
            return True  # Include events with no timestamp
        
        return start_time <= event_time <= end_time
    
    def _calculate_cluster_center(self, cluster_members: List[Dict[str, Any]]) -> Coordinate:
        """Calculate center point of cluster"""
        total_lat = sum(member['location'].latitude for member in cluster_members)
        total_lon = sum(member['location'].longitude for member in cluster_members)
        count = len(cluster_members)
        
        return Coordinate(
            latitude=total_lat / count,
            longitude=total_lon / count,
            source="cluster_center_calculation"
        )
    
    def _calculate_cluster_radius(self, cluster_members: List[Dict[str, Any]], center: Coordinate) -> float:
        """Calculate radius of cluster"""
        max_distance = 0.0
        
        for member in cluster_members:
            distance = center.distance_to(member['location'])
            max_distance = max(max_distance, distance)
        
        return max_distance
    
    def _calculate_cluster_strength(self, cluster_members: List[Dict[str, Any]]) -> float:
        """Calculate cluster strength based on density and cohesion"""
        if len(cluster_members) < 2:
            return 0.0
        
        # Calculate average pairwise distance
        total_distance = 0.0
        pair_count = 0
        
        for i, member1 in enumerate(cluster_members):
            for member2 in cluster_members[i+1:]:
                distance = member1['location'].distance_to(member2['location'])
                total_distance += distance
                pair_count += 1
        
        avg_distance = total_distance / max(pair_count, 1)
        
        # Strength is inversely related to average distance and positively related to size
        size_factor = min(1.0, len(cluster_members) / 10)  # Normalize to max 10 members
        distance_factor = max(0.1, 1.0 - min(avg_distance / 1000, 1.0))  # Normalize to max 1km
        
        return (size_factor + distance_factor) / 2
    
    def _calculate_bounding_box(self, coordinates: List[Coordinate]) -> BoundingBox:
        """Calculate bounding box for list of coordinates"""
        lats = [coord.latitude for coord in coordinates]
        lons = [coord.longitude for coord in coordinates]
        
        return BoundingBox(
            north=max(lats),
            south=min(lats),
            east=max(lons),
            west=min(lons)
        )
    
    def _analyze_time_periods(self, event_ids: List[str], events: List[Union[SpatialEvent, Evidence]]) -> List[Dict[str, Any]]:
        """Analyze time periods for events inside geofence"""
        time_periods = []
        
        # Group events by time periods (simplified)
        events_with_time = []
        for event in events:
            if hasattr(event, 'id') and str(event.id) in event_ids:
                timestamp = None
                if isinstance(event, SpatialEvent):
                    timestamp = event.start_time
                elif isinstance(event, Evidence):
                    timestamp = event.timestamp
                
                if timestamp:
                    events_with_time.append((timestamp, event))
        
        # Sort by time
        events_with_time.sort(key=lambda x: x[0])
        
        # Group into periods (simplified - daily periods)
        if events_with_time:
            current_date = events_with_time[0][0].date()
            period_events = []
            
            for timestamp, event in events_with_time:
                if timestamp.date() == current_date:
                    period_events.append(str(event.id))
                else:
                    # Save current period
                    time_periods.append({
                        'date': current_date.isoformat(),
                        'event_count': len(period_events),
                        'event_ids': period_events
                    })
                    
                    # Start new period
                    current_date = timestamp.date()
                    period_events = [str(event.id)]
            
            # Add last period
            if period_events:
                time_periods.append({
                    'date': current_date.isoformat(),
                    'event_count': len(period_events),
                    'event_ids': period_events
                })
        
        return time_periods


def correlate_events_by_location(events: List[Union[SpatialEvent, Evidence]], radius: float) -> LocationCorrelation:
    """
    Convenience function to correlate events by location
    
    Args:
        events: List of spatial events or evidence to correlate
        radius: Search radius in meters
        
    Returns:
        LocationCorrelation with analysis results
    """
    processor = GeofenceProcessor()
    return processor.correlate_events_by_location(events, radius)