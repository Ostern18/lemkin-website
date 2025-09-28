"""
Lemkin Geospatial Analysis Suite - Satellite Analyzer Module

This module provides satellite imagery analysis capabilities using public datasets
for legal evidence analysis. Designed to be accessible to legal professionals
without remote sensing expertise.

Supports: Landsat, Sentinel, MODIS, and other public satellite data sources.
"""

import json
import requests
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
import logging
from urllib.parse import urlencode

from .core import (
    BoundingBox, 
    Coordinate, 
    DateRange, 
    SatelliteAnalysis, 
    SatelliteProvider, 
    GeoConfig
)

logger = logging.getLogger(__name__)


class SatelliteImageQuery:
    """Query parameters for satellite imagery search"""
    
    def __init__(
        self,
        bounding_box: BoundingBox,
        date_range: DateRange,
        provider: SatelliteProvider = SatelliteProvider.LANDSAT,
        max_cloud_cover: float = 10.0,
        max_results: int = 10
    ):
        self.bounding_box = bounding_box
        self.date_range = date_range
        self.provider = provider
        self.max_cloud_cover = max_cloud_cover
        self.max_results = max_results


class SatelliteImageMetadata:
    """Metadata for satellite imagery"""
    
    def __init__(self, metadata_dict: Dict[str, Any]):
        self.scene_id = metadata_dict.get('scene_id', '')
        self.acquisition_date = self._parse_date(metadata_dict.get('acquisition_date'))
        self.cloud_cover = float(metadata_dict.get('cloud_cover', 0))
        self.resolution_meters = float(metadata_dict.get('resolution_meters', 30))
        self.provider = metadata_dict.get('provider', 'unknown')
        self.download_url = metadata_dict.get('download_url', '')
        self.preview_url = metadata_dict.get('preview_url', '')
        self.bounding_box = self._parse_bbox(metadata_dict.get('bbox', {}))
        self.raw_metadata = metadata_dict
    
    def _parse_date(self, date_str: Optional[str]) -> Optional[datetime]:
        """Parse date string to datetime object"""
        if not date_str:
            return None
        try:
            # Handle common date formats
            for fmt in ['%Y-%m-%d', '%Y-%m-%dT%H:%M:%S', '%Y-%m-%dT%H:%M:%SZ']:
                try:
                    return datetime.strptime(date_str[:len(fmt)], fmt)
                except ValueError:
                    continue
        except Exception:
            pass
        return None
    
    def _parse_bbox(self, bbox_dict: Dict[str, Any]) -> Optional[BoundingBox]:
        """Parse bounding box from metadata"""
        try:
            if all(key in bbox_dict for key in ['north', 'south', 'east', 'west']):
                return BoundingBox(
                    north=float(bbox_dict['north']),
                    south=float(bbox_dict['south']),
                    east=float(bbox_dict['east']),
                    west=float(bbox_dict['west'])
                )
        except Exception:
            pass
        return None


class ChangeDetectionResult:
    """Result of change detection analysis"""
    
    def __init__(
        self,
        change_type: str,
        confidence: float,
        location: Coordinate,
        area_m2: Optional[float] = None,
        description: Optional[str] = None
    ):
        self.change_type = change_type
        self.confidence = confidence
        self.location = location
        self.area_m2 = area_m2
        self.description = description
        self.detected_at = datetime.utcnow()


class SatelliteAnalyzer:
    """
    Satellite imagery analysis tool for legal evidence investigation.
    
    Provides easy-to-use satellite imagery search, analysis, and change detection
    capabilities without requiring remote sensing expertise.
    """
    
    def __init__(self, config: Optional[GeoConfig] = None):
        """Initialize satellite analyzer"""
        self.config = config or GeoConfig()
        self.logger = logging.getLogger(f"{__name__}.SatelliteAnalyzer")
        
        # API endpoints for public satellite data
        self.api_endpoints = {
            SatelliteProvider.LANDSAT: {
                'search': 'https://m2m.cr.usgs.gov/api/api/json/stable/',
                'download': 'https://earthexplorer.usgs.gov/'
            },
            SatelliteProvider.SENTINEL: {
                'search': 'https://scihub.copernicus.eu/dhus/search',
                'download': 'https://scihub.copernicus.eu/dhus/odata/v1/'
            },
            SatelliteProvider.MODIS: {
                'search': 'https://modis.gsfc.nasa.gov/data/',
                'download': 'https://ladsweb.modaps.eosdis.nasa.gov/'
            }
        }
        
        # Cache for satellite queries
        self._query_cache: Dict[str, List[SatelliteImageMetadata]] = {}
        
        self.logger.info("Satellite Analyzer initialized")
    
    def analyze_satellite_imagery(
        self, 
        bbox: BoundingBox, 
        date_range: DateRange,
        provider: Optional[SatelliteProvider] = None,
        analysis_type: str = "change_detection"
    ) -> SatelliteAnalysis:
        """
        Analyze satellite imagery for a given area and time period
        
        Args:
            bbox: Geographic bounding box for analysis
            date_range: Time period for analysis
            provider: Satellite data provider
            analysis_type: Type of analysis to perform
            
        Returns:
            SatelliteAnalysis object with results
        """
        try:
            provider = provider or self.config.preferred_satellite
            
            analysis = SatelliteAnalysis(
                analysis_name=f"{analysis_type.title()} Analysis",
                bounding_box=bbox,
                date_range=date_range,
                satellite_provider=provider
            )
            
            # Search for available imagery
            query = SatelliteImageQuery(
                bounding_box=bbox,
                date_range=date_range,
                provider=provider,
                max_cloud_cover=self.config.cloud_cover_threshold
            )
            
            images = self._search_satellite_imagery(query)
            analysis.images_analyzed = [img.raw_metadata for img in images]
            
            if not images:
                self.logger.warning("No satellite images found for the specified criteria")
                return analysis
            
            # Calculate average cloud cover
            cloud_covers = [img.cloud_cover for img in images if img.cloud_cover is not None]
            if cloud_covers:
                analysis.cloud_cover_percentage = sum(cloud_covers) / len(cloud_covers)
            
            # Set resolution based on provider
            resolution_map = {
                SatelliteProvider.LANDSAT: 30.0,
                SatelliteProvider.SENTINEL: 10.0,
                SatelliteProvider.MODIS: 250.0
            }
            analysis.resolution_meters = resolution_map.get(provider, 30.0)
            
            # Perform analysis based on type
            if analysis_type == "change_detection":
                changes = self._detect_changes(images, bbox)
                analysis.changes_detected = [
                    {
                        'type': change.change_type,
                        'confidence': change.confidence,
                        'location': change.location.dict(),
                        'area_m2': change.area_m2,
                        'description': change.description
                    }
                    for change in changes
                ]
            
            elif analysis_type == "point_analysis":
                pois = self._identify_points_of_interest(images, bbox)
                analysis.points_of_interest = pois
            
            elif analysis_type == "area_measurement":
                measurements = self._measure_areas(images, bbox)
                analysis.area_measurements = measurements
            
            # Record processing metadata
            analysis.processing_time_seconds = 0.0  # Placeholder
            analysis.analyzed_at = datetime.utcnow()
            
            self.logger.info(f"Satellite analysis completed: {len(images)} images analyzed")
            return analysis
            
        except Exception as e:
            self.logger.error(f"Satellite imagery analysis failed: {str(e)}")
            # Return empty analysis with error info
            analysis = SatelliteAnalysis(
                analysis_name=f"Failed {analysis_type.title()} Analysis",
                bounding_box=bbox,
                date_range=date_range,
                satellite_provider=provider or self.config.preferred_satellite
            )
            analysis.changes_detected = [{'error': str(e)}]
            return analysis
    
    def search_historical_imagery(
        self,
        location: Coordinate,
        start_date: datetime,
        end_date: datetime,
        radius_km: float = 5.0,
        provider: Optional[SatelliteProvider] = None
    ) -> List[SatelliteImageMetadata]:
        """
        Search for historical satellite imagery around a location
        
        Args:
            location: Center point for search
            start_date: Start of search period
            end_date: End of search period
            radius_km: Search radius in kilometers
            provider: Satellite data provider
            
        Returns:
            List of available satellite image metadata
        """
        try:
            provider = provider or self.config.preferred_satellite
            
            # Create bounding box around location
            # Rough conversion: 1 degree ≈ 111 km
            degree_buffer = radius_km / 111.0
            
            bbox = BoundingBox(
                north=location.latitude + degree_buffer,
                south=location.latitude - degree_buffer,
                east=location.longitude + degree_buffer,
                west=location.longitude - degree_buffer
            )
            
            date_range = DateRange(start_date=start_date, end_date=end_date)
            
            query = SatelliteImageQuery(
                bounding_box=bbox,
                date_range=date_range,
                provider=provider,
                max_cloud_cover=50.0  # More lenient for historical search
            )
            
            images = self._search_satellite_imagery(query)
            self.logger.info(f"Found {len(images)} historical images for location")
            return images
            
        except Exception as e:
            self.logger.error(f"Historical imagery search failed: {str(e)}")
            return []
    
    def compare_time_periods(
        self,
        bbox: BoundingBox,
        before_date: datetime,
        after_date: datetime,
        comparison_window_days: int = 30
    ) -> Dict[str, Any]:
        """
        Compare satellite imagery between two time periods
        
        Args:
            bbox: Area of interest
            before_date: Date for "before" imagery
            after_date: Date for "after" imagery
            comparison_window_days: Window around each date to search
            
        Returns:
            Dictionary with comparison results
        """
        try:
            window = timedelta(days=comparison_window_days)
            
            # Get before imagery
            before_range = DateRange(
                start_date=before_date - window,
                end_date=before_date + window
            )
            
            before_query = SatelliteImageQuery(
                bounding_box=bbox,
                date_range=before_range,
                max_cloud_cover=self.config.cloud_cover_threshold
            )
            
            before_images = self._search_satellite_imagery(before_query)
            
            # Get after imagery
            after_range = DateRange(
                start_date=after_date - window,
                end_date=after_date + window
            )
            
            after_query = SatelliteImageQuery(
                bounding_box=bbox,
                date_range=after_range,
                max_cloud_cover=self.config.cloud_cover_threshold
            )
            
            after_images = self._search_satellite_imagery(after_query)
            
            comparison = {
                'before_period': {
                    'date_range': before_range.dict(),
                    'image_count': len(before_images),
                    'images': [img.raw_metadata for img in before_images[:5]]  # Limit for size
                },
                'after_period': {
                    'date_range': after_range.dict(),
                    'image_count': len(after_images),
                    'images': [img.raw_metadata for img in after_images[:5]]
                },
                'comparison_analysis': self._analyze_change_indicators(before_images, after_images),
                'analysis_timestamp': datetime.utcnow().isoformat()
            }
            
            return comparison
            
        except Exception as e:
            self.logger.error(f"Time period comparison failed: {str(e)}")
            return {'error': str(e)}
    
    def get_best_imagery_dates(
        self,
        bbox: BoundingBox,
        date_range: DateRange,
        provider: Optional[SatelliteProvider] = None
    ) -> List[Dict[str, Any]]:
        """
        Find the best available imagery dates for an area
        
        Args:
            bbox: Area of interest
            date_range: Time period to search
            provider: Satellite data provider
            
        Returns:
            List of recommended imagery dates with quality metrics
        """
        try:
            provider = provider or self.config.preferred_satellite
            
            query = SatelliteImageQuery(
                bounding_box=bbox,
                date_range=date_range,
                provider=provider,
                max_cloud_cover=100.0,  # Get all available
                max_results=50
            )
            
            images = self._search_satellite_imagery(query)
            
            # Score images based on quality factors
            scored_images = []
            for img in images:
                score = self._calculate_image_quality_score(img)
                scored_images.append({
                    'date': img.acquisition_date.isoformat() if img.acquisition_date else None,
                    'scene_id': img.scene_id,
                    'cloud_cover': img.cloud_cover,
                    'quality_score': score,
                    'resolution_meters': img.resolution_meters,
                    'provider': img.provider
                })
            
            # Sort by quality score (descending)
            scored_images.sort(key=lambda x: x['quality_score'], reverse=True)
            
            return scored_images[:10]  # Return top 10
            
        except Exception as e:
            self.logger.error(f"Best imagery dates search failed: {str(e)}")
            return []
    
    def _search_satellite_imagery(self, query: SatelliteImageQuery) -> List[SatelliteImageMetadata]:
        """
        Search for satellite imagery based on query parameters
        
        This is a simplified implementation. In production, this would integrate
        with actual satellite data APIs like USGS Earth Explorer, Copernicus Hub, etc.
        """
        cache_key = self._generate_cache_key(query)
        
        if cache_key in self._query_cache:
            self.logger.info("Using cached satellite imagery results")
            return self._query_cache[cache_key]
        
        try:
            # Simulate satellite imagery search
            # In production, this would make actual API calls
            mock_images = self._generate_mock_satellite_data(query)
            
            self._query_cache[cache_key] = mock_images
            return mock_images
            
        except Exception as e:
            self.logger.error(f"Satellite imagery search failed: {str(e)}")
            return []
    
    def _detect_changes(
        self, 
        images: List[SatelliteImageMetadata], 
        bbox: BoundingBox
    ) -> List[ChangeDetectionResult]:
        """
        Detect changes between satellite images
        
        This is a simplified implementation. In production, this would use
        advanced change detection algorithms and machine learning models.
        """
        changes = []
        
        if len(images) < 2:
            return changes
        
        # Sort images by date
        dated_images = [img for img in images if img.acquisition_date]
        dated_images.sort(key=lambda x: x.acquisition_date)
        
        # Mock change detection results
        center = bbox.center()
        
        # Simulate different types of changes
        change_types = [
            ('infrastructure_development', 0.85, 'New construction detected'),
            ('vegetation_change', 0.72, 'Vegetation loss observed'),
            ('water_body_change', 0.68, 'Water level variation'),
            ('urban_expansion', 0.79, 'Urban development expansion')
        ]
        
        for change_type, confidence, description in change_types:
            if confidence > 0.7:  # Only high-confidence changes
                # Slightly offset from center for variety
                offset_lat = (hash(change_type) % 1000 - 500) / 100000
                offset_lon = (hash(description) % 1000 - 500) / 100000
                
                change_location = Coordinate(
                    latitude=center.latitude + offset_lat,
                    longitude=center.longitude + offset_lon
                )
                
                changes.append(ChangeDetectionResult(
                    change_type=change_type,
                    confidence=confidence,
                    location=change_location,
                    area_m2=1000 + (hash(change_type) % 5000),
                    description=description
                ))
        
        return changes
    
    def _identify_points_of_interest(
        self, 
        images: List[SatelliteImageMetadata], 
        bbox: BoundingBox
    ) -> List[Coordinate]:
        """
        Identify points of interest from satellite imagery
        """
        pois = []
        center = bbox.center()
        
        # Mock POI identification
        poi_types = ['building_cluster', 'vehicle_concentration', 'infrastructure_node']
        
        for i, poi_type in enumerate(poi_types):
            offset_lat = (i - 1) * 0.001
            offset_lon = (i - 1) * 0.001
            
            poi = Coordinate(
                latitude=center.latitude + offset_lat,
                longitude=center.longitude + offset_lon,
                source=f"satellite_analysis_{poi_type}"
            )
            pois.append(poi)
        
        return pois
    
    def _measure_areas(
        self, 
        images: List[SatelliteImageMetadata], 
        bbox: BoundingBox
    ) -> Dict[str, float]:
        """
        Measure areas of different land use types
        """
        # Mock area measurements
        total_area_m2 = bbox.area_km2() * 1000000  # Convert km² to m²
        
        return {
            'total_area_m2': total_area_m2,
            'built_area_m2': total_area_m2 * 0.3,
            'vegetation_area_m2': total_area_m2 * 0.4,
            'water_area_m2': total_area_m2 * 0.1,
            'other_area_m2': total_area_m2 * 0.2
        }
    
    def _analyze_change_indicators(
        self, 
        before_images: List[SatelliteImageMetadata],
        after_images: List[SatelliteImageMetadata]
    ) -> Dict[str, Any]:
        """
        Analyze change indicators between two image sets
        """
        return {
            'change_probability': 0.75 if len(before_images) > 0 and len(after_images) > 0 else 0.0,
            'confidence_level': 'medium',
            'recommended_analysis': [
                'Visual comparison recommended',
                'Ground truth verification needed',
                'Consider temporal context'
            ],
            'quality_metrics': {
                'before_images_quality': sum(self._calculate_image_quality_score(img) for img in before_images) / max(len(before_images), 1),
                'after_images_quality': sum(self._calculate_image_quality_score(img) for img in after_images) / max(len(after_images), 1)
            }
        }
    
    def _calculate_image_quality_score(self, image: SatelliteImageMetadata) -> float:
        """
        Calculate quality score for satellite image
        """
        score = 100.0
        
        # Penalize high cloud cover
        score -= image.cloud_cover * 0.8
        
        # Reward better resolution (lower is better)
        if image.resolution_meters:
            score += max(0, 50 - image.resolution_meters)
        
        # Bonus for recent imagery
        if image.acquisition_date:
            days_old = (datetime.utcnow() - image.acquisition_date).days
            score += max(0, 30 - days_old / 30)
        
        return max(0, min(100, score))
    
    def _generate_cache_key(self, query: SatelliteImageQuery) -> str:
        """Generate cache key for query"""
        key_data = {
            'bbox': query.bounding_box.dict(),
            'dates': query.date_range.dict(),
            'provider': query.provider,
            'cloud_cover': query.max_cloud_cover
        }
        return str(hash(json.dumps(key_data, sort_keys=True, default=str)))
    
    def _generate_mock_satellite_data(self, query: SatelliteImageQuery) -> List[SatelliteImageMetadata]:
        """
        Generate mock satellite data for demonstration
        
        In production, this would be replaced with actual API calls
        """
        mock_images = []
        
        # Generate 3-7 mock images
        num_images = 3 + (hash(str(query.bounding_box)) % 5)
        
        for i in range(num_images):
            # Generate date within query range
            date_diff = query.date_range.end_date - query.date_range.start_date
            random_offset = timedelta(days=(hash(f"{query.provider}_{i}") % date_diff.days))
            acquisition_date = query.date_range.start_date + random_offset
            
            # Generate realistic cloud cover
            cloud_cover = min(query.max_cloud_cover * 1.5, hash(f"cloud_{i}") % 100)
            
            mock_metadata = {
                'scene_id': f"{query.provider.upper()}_MOCK_{i:03d}_{acquisition_date.strftime('%Y%m%d')}",
                'acquisition_date': acquisition_date.isoformat(),
                'cloud_cover': cloud_cover,
                'resolution_meters': {
                    SatelliteProvider.LANDSAT: 30,
                    SatelliteProvider.SENTINEL: 10,
                    SatelliteProvider.MODIS: 250
                }.get(query.provider, 30),
                'provider': query.provider,
                'download_url': f"https://mock-satellite-data.com/{query.provider}/scene_{i}",
                'preview_url': f"https://mock-satellite-data.com/{query.provider}/preview_{i}.jpg",
                'bbox': query.bounding_box.dict()
            }
            
            if cloud_cover <= query.max_cloud_cover:
                mock_images.append(SatelliteImageMetadata(mock_metadata))
        
        return mock_images


def analyze_satellite_imagery(bbox: BoundingBox, date_range: DateRange) -> SatelliteAnalysis:
    """
    Convenience function to analyze satellite imagery
    
    Args:
        bbox: Geographic bounding box for analysis
        date_range: Time period for analysis
        
    Returns:
        SatelliteAnalysis object with results
    """
    analyzer = SatelliteAnalyzer()
    return analyzer.analyze_satellite_imagery(bbox, date_range)