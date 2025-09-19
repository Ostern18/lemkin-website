"""
Lemkin Geospatial Analysis Suite - Mapping Generator Module

This module provides interactive map creation with evidence overlay capabilities
for legal professionals. Creates user-friendly interactive maps without requiring
GIS expertise, supporting multiple export formats.

Features: Interactive maps, evidence overlay, clustering, multiple export formats.
"""

import json
import base64
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path
from enum import Enum
import logging

from .core import (
    Coordinate,
    BoundingBox,
    Evidence,
    SpatialEvent,
    InteractiveMap,
    MapLayer,
    MapLayerType,
    EvidenceType,
    GeoConfig
)

logger = logging.getLogger(__name__)


class MapProvider(str, Enum):
    """Supported map providers"""
    OPENSTREETMAP = "OpenStreetMap"
    STAMEN_TERRAIN = "Stamen Terrain"
    STAMEN_TONER = "Stamen Toner"
    CARTODB_POSITRON = "CartoDB Positron"
    CARTODB_DARK_MATTER = "CartoDB Dark_Matter"


class ExportFormat(str, Enum):
    """Supported export formats"""
    HTML = "html"
    PDF = "pdf"
    PNG = "png"
    KML = "kml"
    GEOJSON = "geojson"
    GPX = "gpx"


class MarkerStyle:
    """Style configuration for map markers"""
    
    def __init__(
        self,
        color: str = "blue",
        size: int = 10,
        icon: Optional[str] = None,
        popup_template: Optional[str] = None
    ):
        self.color = color
        self.size = size
        self.icon = icon
        self.popup_template = popup_template or self._default_popup_template()
    
    def _default_popup_template(self) -> str:
        """Default popup template"""
        return """
        <div style="font-family: Arial, sans-serif; max-width: 300px;">
            <h4>{title}</h4>
            <p><strong>Type:</strong> {type}</p>
            <p><strong>Location:</strong> {latitude:.6f}, {longitude:.6f}</p>
            <p><strong>Timestamp:</strong> {timestamp}</p>
            <p><strong>Description:</strong> {description}</p>
        </div>
        """


class LayerConfiguration:
    """Configuration for map layers"""
    
    def __init__(
        self,
        name: str,
        layer_type: MapLayerType,
        visible: bool = True,
        opacity: float = 1.0,
        style: Optional[Dict[str, Any]] = None
    ):
        self.name = name
        self.layer_type = layer_type
        self.visible = visible
        self.opacity = opacity
        self.style = style or self._default_style()
    
    def _default_style(self) -> Dict[str, Any]:
        """Default layer style based on type"""
        styles = {
            MapLayerType.EVIDENCE: {
                "color": "#FF0000",
                "fillColor": "#FF0000",
                "fillOpacity": 0.6,
                "weight": 2
            },
            MapLayerType.GEOFENCE: {
                "color": "#0000FF",
                "fillColor": "#0000FF", 
                "fillOpacity": 0.2,
                "weight": 3,
                "dashArray": "5, 5"
            },
            MapLayerType.TRAJECTORY: {
                "color": "#00FF00",
                "weight": 3,
                "opacity": 0.8
            },
            MapLayerType.HEAT_MAP: {
                "radius": 25,
                "blur": 15,
                "maxZoom": 17
            }
        }
        return styles.get(self.layer_type, {})


class MappingGenerator:
    """
    Interactive map generator for legal evidence visualization.
    
    Creates user-friendly interactive maps with evidence overlay,
    clustering, and multiple export options without requiring GIS expertise.
    """
    
    def __init__(self, config: Optional[GeoConfig] = None):
        """Initialize mapping generator"""
        self.config = config or GeoConfig()
        self.logger = logging.getLogger(f"{__name__}.MappingGenerator")
        
        # Style configurations for different evidence types
        self.evidence_styles = {
            EvidenceType.PHOTOGRAPH: MarkerStyle(color="red", icon="camera"),
            EvidenceType.VIDEO: MarkerStyle(color="orange", icon="video"),
            EvidenceType.WITNESS_TESTIMONY: MarkerStyle(color="blue", icon="user"),
            EvidenceType.DOCUMENT: MarkerStyle(color="green", icon="file"),
            EvidenceType.SOCIAL_MEDIA_POST: MarkerStyle(color="purple", icon="share"),
            EvidenceType.SATELLITE_IMAGE: MarkerStyle(color="yellow", icon="globe"),
            EvidenceType.MOBILE_DATA: MarkerStyle(color="pink", icon="mobile"),
            EvidenceType.VEHICLE_TRACKING: MarkerStyle(color="brown", icon="car")
        }
        
        self.logger.info("Mapping Generator initialized")
    
    def generate_evidence_map(
        self, 
        evidence: List[Evidence],
        title: str = "Evidence Location Map",
        description: Optional[str] = None,
        center: Optional[Coordinate] = None,
        zoom_level: Optional[int] = None,
        provider: MapProvider = MapProvider.OPENSTREETMAP,
        cluster_markers: bool = True
    ) -> InteractiveMap:
        """
        Generate interactive map with evidence overlay
        
        Args:
            evidence: List of evidence items with location data
            title: Map title
            description: Optional map description
            center: Map center (auto-calculated if not provided)
            zoom_level: Zoom level (auto-calculated if not provided)
            provider: Map tile provider
            cluster_markers: Whether to cluster nearby markers
            
        Returns:
            InteractiveMap object
        """
        try:
            # Filter evidence with location data
            located_evidence = [e for e in evidence if e.location]
            
            if not located_evidence:
                self.logger.warning("No evidence with location data found")
                # Return empty map
                default_center = Coordinate(latitude=40.7128, longitude=-74.0060)
                return InteractiveMap(
                    title=title,
                    description=description or "No evidence locations available",
                    center=default_center,
                    zoom_level=zoom_level or 10,
                    map_provider=provider
                )
            
            # Calculate map center if not provided
            if not center:
                coordinates = [e.location for e in located_evidence]
                center = self._calculate_center(coordinates)
            
            # Calculate zoom level if not provided
            if not zoom_level:
                coordinates = [e.location for e in located_evidence]
                zoom_level = self._calculate_optimal_zoom(coordinates)
            
            # Create interactive map
            interactive_map = InteractiveMap(
                title=title,
                description=description or f"Map showing {len(located_evidence)} evidence locations",
                center=center,
                zoom_level=zoom_level,
                map_provider=provider,
                clustering_enabled=cluster_markers
            )
            
            # Add evidence to map
            interactive_map.evidence_points = located_evidence
            
            # Create evidence layer
            evidence_layer = self._create_evidence_layer(located_evidence)
            interactive_map.layers.append(evidence_layer)
            
            # Add legend
            if len(located_evidence) > 0:
                interactive_map.legend_enabled = True
                interactive_map.popup_templates = self._generate_popup_templates()
            
            self.logger.info(f"Generated evidence map with {len(located_evidence)} locations")
            return interactive_map
            
        except Exception as e:
            self.logger.error(f"Evidence map generation failed: {str(e)}")
            # Return error map
            default_center = Coordinate(latitude=0, longitude=0)
            error_map = InteractiveMap(
                title=f"Error: {title}",
                description=f"Map generation failed: {str(e)}",
                center=default_center,
                zoom_level=2,
                map_provider=provider
            )
            return error_map
    
    def add_geofence_layer(
        self, 
        interactive_map: InteractiveMap,
        geofences: List[Dict[str, Any]],
        layer_name: str = "Geofences"
    ) -> InteractiveMap:
        """
        Add geofence layer to interactive map
        
        Args:
            interactive_map: Existing interactive map
            geofences: List of geofence definitions
            layer_name: Name for the geofence layer
            
        Returns:
            Updated InteractiveMap
        """
        try:
            # Create geofence coordinates
            geofence_coordinates = []
            
            for geofence in geofences:
                if geofence.get('type') == 'circular':
                    # Create circle approximation
                    center = geofence.get('center')
                    radius_meters = geofence.get('radius_meters', 100)
                    
                    if center and isinstance(center, (dict, Coordinate)):
                        if isinstance(center, dict):
                            center_coord = Coordinate(**center)
                        else:
                            center_coord = center
                        
                        # Create circle points
                        circle_points = self._create_circle_points(center_coord, radius_meters)
                        geofence_coordinates.extend(circle_points)
                
                elif geofence.get('type') == 'polygon':
                    vertices = geofence.get('vertices', [])
                    for vertex in vertices:
                        if isinstance(vertex, dict):
                            geofence_coordinates.append(Coordinate(**vertex))
                        elif isinstance(vertex, Coordinate):
                            geofence_coordinates.append(vertex)
            
            # Create geofence layer
            if geofence_coordinates:
                geofence_layer = MapLayer(
                    name=layer_name,
                    layer_type=MapLayerType.GEOFENCE,
                    coordinates=geofence_coordinates,
                    style=LayerConfiguration("geofence", MapLayerType.GEOFENCE).style
                )
                
                interactive_map.layers.append(geofence_layer)
                self.logger.info(f"Added geofence layer with {len(geofence_coordinates)} points")
            
            return interactive_map
            
        except Exception as e:
            self.logger.error(f"Adding geofence layer failed: {str(e)}")
            return interactive_map
    
    def add_trajectory_layer(
        self,
        interactive_map: InteractiveMap,
        trajectory_points: List[Coordinate],
        layer_name: str = "Trajectory",
        style: Optional[Dict[str, Any]] = None
    ) -> InteractiveMap:
        """
        Add trajectory layer to interactive map
        
        Args:
            interactive_map: Existing interactive map
            trajectory_points: List of coordinates forming trajectory
            layer_name: Name for the trajectory layer
            style: Optional custom style
            
        Returns:
            Updated InteractiveMap
        """
        try:
            if not trajectory_points:
                return interactive_map
            
            trajectory_style = style or LayerConfiguration("trajectory", MapLayerType.TRAJECTORY).style
            
            trajectory_layer = MapLayer(
                name=layer_name,
                layer_type=MapLayerType.TRAJECTORY,
                coordinates=trajectory_points,
                style=trajectory_style,
                properties={
                    'trajectory_length_km': self._calculate_trajectory_length(trajectory_points),
                    'point_count': len(trajectory_points)
                }
            )
            
            interactive_map.layers.append(trajectory_layer)
            self.logger.info(f"Added trajectory layer with {len(trajectory_points)} points")
            
            return interactive_map
            
        except Exception as e:
            self.logger.error(f"Adding trajectory layer failed: {str(e)}")
            return interactive_map
    
    def create_heat_map_layer(
        self,
        interactive_map: InteractiveMap,
        coordinates: List[Coordinate],
        weights: Optional[List[float]] = None,
        layer_name: str = "Heat Map"
    ) -> InteractiveMap:
        """
        Create heat map layer from coordinates
        
        Args:
            interactive_map: Existing interactive map
            coordinates: List of coordinates for heat map
            weights: Optional weights for each coordinate
            layer_name: Name for the heat map layer
            
        Returns:
            Updated InteractiveMap
        """
        try:
            if not coordinates:
                return interactive_map
            
            # Prepare heat map data
            heat_data = []
            for i, coord in enumerate(coordinates):
                weight = weights[i] if weights and i < len(weights) else 1.0
                heat_data.append({
                    'lat': coord.latitude,
                    'lng': coord.longitude,
                    'weight': weight
                })
            
            heat_layer = MapLayer(
                name=layer_name,
                layer_type=MapLayerType.HEAT_MAP,
                coordinates=coordinates,
                style=LayerConfiguration("heat", MapLayerType.HEAT_MAP).style,
                properties={
                    'heat_data': heat_data,
                    'max_intensity': max(weights) if weights else 1.0
                }
            )
            
            interactive_map.layers.append(heat_layer)
            self.logger.info(f"Added heat map layer with {len(coordinates)} points")
            
            return interactive_map
            
        except Exception as e:
            self.logger.error(f"Creating heat map layer failed: {str(e)}")
            return interactive_map
    
    def export_map(
        self,
        interactive_map: InteractiveMap,
        output_path: Path,
        format: ExportFormat = ExportFormat.HTML,
        include_metadata: bool = True
    ) -> bool:
        """
        Export interactive map to various formats
        
        Args:
            interactive_map: Map to export
            output_path: Output file path
            format: Export format
            include_metadata: Whether to include metadata
            
        Returns:
            True if export successful
        """
        try:
            if format == ExportFormat.HTML:
                return self._export_html(interactive_map, output_path, include_metadata)
            elif format == ExportFormat.GEOJSON:
                return self._export_geojson(interactive_map, output_path, include_metadata)
            elif format == ExportFormat.KML:
                return self._export_kml(interactive_map, output_path, include_metadata)
            elif format == ExportFormat.GPX:
                return self._export_gpx(interactive_map, output_path, include_metadata)
            elif format == ExportFormat.PDF:
                return self._export_pdf(interactive_map, output_path, include_metadata)
            else:
                self.logger.error(f"Unsupported export format: {format}")
                return False
                
        except Exception as e:
            self.logger.error(f"Map export failed: {str(e)}")
            return False
    
    def create_comparison_map(
        self,
        before_evidence: List[Evidence],
        after_evidence: List[Evidence],
        title: str = "Before/After Comparison Map"
    ) -> InteractiveMap:
        """
        Create comparison map showing before/after evidence
        
        Args:
            before_evidence: Evidence from before period
            after_evidence: Evidence from after period
            title: Map title
            
        Returns:
            InteractiveMap with comparison layers
        """
        try:
            # Combine all evidence for center calculation
            all_evidence = before_evidence + after_evidence
            located_evidence = [e for e in all_evidence if e.location]
            
            if not located_evidence:
                self.logger.warning("No evidence with location data for comparison")
                default_center = Coordinate(latitude=40.7128, longitude=-74.0060)
                return InteractiveMap(
                    title=title,
                    center=default_center,
                    zoom_level=10
                )
            
            # Calculate map parameters
            coordinates = [e.location for e in located_evidence]
            center = self._calculate_center(coordinates)
            zoom_level = self._calculate_optimal_zoom(coordinates)
            
            # Create comparison map
            comparison_map = InteractiveMap(
                title=title,
                description=f"Comparison showing {len(before_evidence)} before and {len(after_evidence)} after evidence items",
                center=center,
                zoom_level=zoom_level,
                clustering_enabled=False  # Disable clustering for comparison
            )
            
            # Add before evidence layer
            if before_evidence:
                before_layer = self._create_evidence_layer(
                    [e for e in before_evidence if e.location],
                    layer_name="Before Evidence",
                    color_override="blue"
                )
                comparison_map.layers.append(before_layer)
            
            # Add after evidence layer
            if after_evidence:
                after_layer = self._create_evidence_layer(
                    [e for e in after_evidence if e.location],
                    layer_name="After Evidence", 
                    color_override="red"
                )
                comparison_map.layers.append(after_layer)
            
            # Add all evidence points for popup functionality
            comparison_map.evidence_points = located_evidence
            
            self.logger.info(f"Created comparison map with {len(located_evidence)} total evidence points")
            return comparison_map
            
        except Exception as e:
            self.logger.error(f"Comparison map creation failed: {str(e)}")
            default_center = Coordinate(latitude=0, longitude=0)
            error_map = InteractiveMap(
                title=f"Error: {title}",
                description=f"Comparison map creation failed: {str(e)}",
                center=default_center,
                zoom_level=2
            )
            return error_map
    
    def _create_evidence_layer(
        self, 
        evidence: List[Evidence], 
        layer_name: str = "Evidence",
        color_override: Optional[str] = None
    ) -> MapLayer:
        """Create map layer from evidence items"""
        coordinates = [e.location for e in evidence if e.location]
        
        # Create style based on evidence types
        style = {
            "markers": [],
            "default_color": color_override or "red"
        }
        
        for e in evidence:
            if e.location:
                marker_style = self.evidence_styles.get(e.evidence_type, MarkerStyle())
                if color_override:
                    marker_style.color = color_override
                
                style["markers"].append({
                    "id": str(e.id),
                    "lat": e.location.latitude,
                    "lng": e.location.longitude,
                    "color": marker_style.color,
                    "icon": marker_style.icon,
                    "popup": self._generate_evidence_popup(e)
                })
        
        layer = MapLayer(
            name=layer_name,
            layer_type=MapLayerType.EVIDENCE,
            coordinates=coordinates,
            style=style,
            properties={
                "evidence_count": len(evidence),
                "evidence_types": list(set(e.evidence_type for e in evidence))
            }
        )
        
        return layer
    
    def _calculate_center(self, coordinates: List[Coordinate]) -> Coordinate:
        """Calculate center point of coordinates"""
        if not coordinates:
            return Coordinate(latitude=40.7128, longitude=-74.0060)  # Default to NYC
        
        total_lat = sum(c.latitude for c in coordinates)
        total_lon = sum(c.longitude for c in coordinates)
        count = len(coordinates)
        
        return Coordinate(
            latitude=total_lat / count,
            longitude=total_lon / count,
            source="calculated_map_center"
        )
    
    def _calculate_optimal_zoom(self, coordinates: List[Coordinate]) -> int:
        """Calculate optimal zoom level for coordinates"""
        if not coordinates or len(coordinates) == 1:
            return self.config.default_zoom_level
        
        # Calculate bounding box
        lats = [c.latitude for c in coordinates]
        lons = [c.longitude for c in coordinates]
        
        lat_range = max(lats) - min(lats)
        lon_range = max(lons) - min(lons)
        
        # Simple zoom calculation based on coordinate span
        max_range = max(lat_range, lon_range)
        
        if max_range > 10:
            return 3
        elif max_range > 5:
            return 5
        elif max_range > 2:
            return 7
        elif max_range > 1:
            return 9
        elif max_range > 0.5:
            return 11
        elif max_range > 0.1:
            return 13
        else:
            return 15
    
    def _create_circle_points(self, center: Coordinate, radius_meters: float, num_points: int = 32) -> List[Coordinate]:
        """Create points approximating a circle"""
        points = []
        
        # Convert radius to degrees (rough approximation)
        lat_radius = radius_meters / 111320  # meters per degree latitude
        lon_radius = radius_meters / (111320 * abs(math.cos(math.radians(center.latitude))))
        
        for i in range(num_points + 1):  # +1 to close the circle
            angle = 2 * math.pi * i / num_points
            lat = center.latitude + lat_radius * math.sin(angle)
            lon = center.longitude + lon_radius * math.cos(angle)
            
            points.append(Coordinate(
                latitude=lat,
                longitude=lon,
                source=f"circle_point_{i}"
            ))
        
        return points
    
    def _calculate_trajectory_length(self, points: List[Coordinate]) -> float:
        """Calculate total length of trajectory in kilometers"""
        if len(points) < 2:
            return 0.0
        
        total_distance = 0.0
        for i in range(len(points) - 1):
            distance = points[i].distance_to(points[i + 1])
            total_distance += distance
        
        return total_distance / 1000  # Convert to kilometers
    
    def _generate_evidence_popup(self, evidence: Evidence) -> str:
        """Generate popup content for evidence marker"""
        template = self.evidence_styles.get(evidence.evidence_type, MarkerStyle()).popup_template
        
        return template.format(
            title=evidence.title,
            type=evidence.evidence_type,
            latitude=evidence.location.latitude if evidence.location else 0,
            longitude=evidence.location.longitude if evidence.location else 0,
            timestamp=evidence.timestamp.strftime("%Y-%m-%d %H:%M:%S") if evidence.timestamp else "Unknown",
            description=evidence.description or "No description available"
        )
    
    def _generate_popup_templates(self) -> Dict[str, str]:
        """Generate popup templates for different evidence types"""
        templates = {}
        for evidence_type, style in self.evidence_styles.items():
            templates[evidence_type] = style.popup_template
        return templates
    
    def _export_html(self, interactive_map: InteractiveMap, output_path: Path, include_metadata: bool) -> bool:
        """Export map as HTML file"""
        try:
            html_content = self._generate_html_map(interactive_map, include_metadata)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            self.logger.info(f"Exported HTML map to {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"HTML export failed: {str(e)}")
            return False
    
    def _export_geojson(self, interactive_map: InteractiveMap, output_path: Path, include_metadata: bool) -> bool:
        """Export map as GeoJSON file"""
        try:
            geojson = {
                "type": "FeatureCollection",
                "features": []
            }
            
            # Add evidence points as features
            for evidence in interactive_map.evidence_points:
                if evidence.location:
                    feature = {
                        "type": "Feature",
                        "geometry": {
                            "type": "Point",
                            "coordinates": [evidence.location.longitude, evidence.location.latitude]
                        },
                        "properties": {
                            "title": evidence.title,
                            "evidence_type": evidence.evidence_type,
                            "timestamp": evidence.timestamp.isoformat() if evidence.timestamp else None,
                            "description": evidence.description,
                            "id": str(evidence.id)
                        }
                    }
                    
                    if include_metadata:
                        feature["properties"]["metadata"] = {
                            "collector": evidence.collector,
                            "source": evidence.source,
                            "verified": evidence.verified,
                            "collected_at": evidence.collected_at.isoformat()
                        }
                    
                    geojson["features"].append(feature)
            
            # Add layer features
            for layer in interactive_map.layers:
                if layer.layer_type == MapLayerType.TRAJECTORY and layer.coordinates:
                    feature = {
                        "type": "Feature",
                        "geometry": {
                            "type": "LineString",
                            "coordinates": [[c.longitude, c.latitude] for c in layer.coordinates]
                        },
                        "properties": {
                            "name": layer.name,
                            "layer_type": layer.layer_type,
                            "style": layer.style
                        }
                    }
                    geojson["features"].append(feature)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(geojson, f, indent=2, default=str)
            
            self.logger.info(f"Exported GeoJSON to {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"GeoJSON export failed: {str(e)}")
            return False
    
    def _export_kml(self, interactive_map: InteractiveMap, output_path: Path, include_metadata: bool) -> bool:
        """Export map as KML file"""
        try:
            kml_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
    <Document>
        <name>{interactive_map.title}</name>
        <description>{interactive_map.description or ""}</description>
        
        <!-- Evidence Points -->
"""
            
            for evidence in interactive_map.evidence_points:
                if evidence.location:
                    kml_content += f"""        <Placemark>
            <name>{evidence.title}</name>
            <description><![CDATA[
                <strong>Type:</strong> {evidence.evidence_type}<br/>
                <strong>Timestamp:</strong> {evidence.timestamp or "Unknown"}<br/>
                <strong>Description:</strong> {evidence.description or "No description"}
            ]]></description>
            <Point>
                <coordinates>{evidence.location.longitude},{evidence.location.latitude},0</coordinates>
            </Point>
        </Placemark>
"""
            
            kml_content += """    </Document>
</kml>"""
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(kml_content)
            
            self.logger.info(f"Exported KML to {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"KML export failed: {str(e)}")
            return False
    
    def _export_gpx(self, interactive_map: InteractiveMap, output_path: Path, include_metadata: bool) -> bool:
        """Export map as GPX file"""
        try:
            gpx_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<gpx version="1.1" creator="Lemkin Geo Analysis Suite">
    <metadata>
        <name>{interactive_map.title}</name>
        <desc>{interactive_map.description or ""}</desc>
        <time>{datetime.utcnow().isoformat()}Z</time>
    </metadata>
"""
            
            # Add waypoints for evidence
            for evidence in interactive_map.evidence_points:
                if evidence.location:
                    gpx_content += f"""    <wpt lat="{evidence.location.latitude}" lon="{evidence.location.longitude}">
        <name>{evidence.title}</name>
        <desc>{evidence.description or ""}</desc>
        <type>{evidence.evidence_type}</type>
"""
                    if evidence.timestamp:
                        gpx_content += f"        <time>{evidence.timestamp.isoformat()}Z</time>\n"
                    
                    gpx_content += "    </wpt>\n"
            
            # Add tracks for trajectories
            for layer in interactive_map.layers:
                if layer.layer_type == MapLayerType.TRAJECTORY and layer.coordinates:
                    gpx_content += f"""    <trk>
        <name>{layer.name}</name>
        <trkseg>
"""
                    for coord in layer.coordinates:
                        gpx_content += f"""            <trkpt lat="{coord.latitude}" lon="{coord.longitude}"></trkpt>
"""
                    gpx_content += """        </trkseg>
    </trk>
"""
            
            gpx_content += "</gpx>"
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(gpx_content)
            
            self.logger.info(f"Exported GPX to {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"GPX export failed: {str(e)}")
            return False
    
    def _export_pdf(self, interactive_map: InteractiveMap, output_path: Path, include_metadata: bool) -> bool:
        """Export map as PDF file (placeholder implementation)"""
        try:
            # This would require a library like weasyprint, reportlab, or matplotlib
            # For now, create a simple text-based report
            
            content = f"""EVIDENCE LOCATION MAP REPORT
Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC

Title: {interactive_map.title}
Description: {interactive_map.description or "No description"}
Map Center: {interactive_map.center.latitude:.6f}, {interactive_map.center.longitude:.6f}
Zoom Level: {interactive_map.zoom_level}
Evidence Points: {len(interactive_map.evidence_points)}

EVIDENCE LOCATIONS:
"""
            
            for i, evidence in enumerate(interactive_map.evidence_points, 1):
                if evidence.location:
                    content += f"""
{i}. {evidence.title}
   Type: {evidence.evidence_type}
   Location: {evidence.location.latitude:.6f}, {evidence.location.longitude:.6f}
   Timestamp: {evidence.timestamp or "Unknown"}
   Description: {evidence.description or "No description"}
"""
            
            # Save as text file (in production, would generate actual PDF)
            text_path = output_path.with_suffix('.txt')
            with open(text_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            self.logger.info(f"Exported map report to {text_path} (PDF export requires additional libraries)")
            return True
            
        except Exception as e:
            self.logger.error(f"PDF export failed: {str(e)}")
            return False
    
    def _generate_html_map(self, interactive_map: InteractiveMap, include_metadata: bool) -> str:
        """Generate HTML content for interactive map"""
        # This is a simplified HTML map generator
        # In production, would use Folium, Leaflet, or similar library
        
        html_template = f"""<!DOCTYPE html>
<html>
<head>
    <title>{interactive_map.title}</title>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.7.1/dist/leaflet.css" />
    <script src="https://unpkg.com/leaflet@1.7.1/dist/leaflet.js"></script>
    <style>
        body {{ margin: 0; padding: 0; }}
        #map {{ width: 100%; height: 100vh; }}
        .info {{ padding: 6px 8px; font: 14px/16px Arial, Helvetica, sans-serif; background: white; background: rgba(255,255,255,0.8); box-shadow: 0 0 15px rgba(0,0,0,0.2); border-radius: 5px; }}
        .legend {{ text-align: left; line-height: 18px; color: #555; }}
        .legend i {{ width: 18px; height: 18px; float: left; margin-right: 8px; opacity: 0.7; }}
    </style>
</head>
<body>
    <div id="map"></div>
    
    <script>
        var map = L.map('map').setView([{interactive_map.center.latitude}, {interactive_map.center.longitude}], {interactive_map.zoom_level});
        
        L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
            attribution: '© OpenStreetMap contributors'
        }}).addTo(map);
        
        // Add evidence markers
"""
        
        # Add evidence markers to HTML
        for evidence in interactive_map.evidence_points:
            if evidence.location:
                marker_color = self.evidence_styles.get(evidence.evidence_type, MarkerStyle()).color
                popup_content = self._generate_evidence_popup(evidence).replace('\n', '').replace("'", "\\'")
                
                html_template += f"""        L.marker([{evidence.location.latitude}, {evidence.location.longitude}])
            .addTo(map)
            .bindPopup('{popup_content}');
"""
        
        html_template += """    </script>
</body>
</html>"""
        
        return html_template


def generate_evidence_map(evidence: List[Evidence]) -> InteractiveMap:
    """
    Convenience function to generate evidence map
    
    Args:
        evidence: List of evidence items with location data
        
    Returns:
        InteractiveMap object
    """
    generator = MappingGenerator()
    return generator.generate_evidence_map(evidence)


# Import math for calculations
import math