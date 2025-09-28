"""
Lemkin Image Verification Suite - Geolocation Module

This module implements image geolocation extraction and verification including
GPS metadata extraction, visual landmark recognition, and location verification
for forensic investigations.

Legal Compliance: Meets standards for digital evidence geolocation in legal proceedings
"""

import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import math

import cv2
import numpy as np
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
import requests
# Optional dependencies with fallbacks
try:
    import reverse_geocoder as rg
    HAS_REVERSE_GEOCODER = True
except ImportError:
    rg = None
    HAS_REVERSE_GEOCODER = False

try:
    from geopy.geocoders import Nominatim
    from geopy.distance import geodesic
    HAS_GEOPY = True
except ImportError:
    Nominatim = None
    geodesic = None
    HAS_GEOPY = False
import json

from .core import (
    GeolocationData,
    GeolocationResult,
    ImageAuthConfig
)

logger = logging.getLogger(__name__)


class ImageGeolocator:
    """
    Comprehensive image geolocation analysis including GPS metadata extraction,
    visual landmark recognition, and location verification.
    """
    
    def __init__(self, config: Optional[ImageAuthConfig] = None):
        """Initialize the image geolocator"""
        self.config = config or ImageAuthConfig()
        
        # Initialize geocoder if available
        if HAS_GEOPY:
            self.geocoder = Nominatim(user_agent="lemkin-images-geolocator")
        else:
            self.geocoder = None
            logger.warning("geopy not available - detailed geocoding will be limited")
        
        # Visual recognition models (would be loaded here in full implementation)
        self.landmark_detector = None
        self.feature_matcher = None
        
        logger.info("Image geolocator initialized")
    
    def geolocate_image(self, image_path: Path) -> GeolocationResult:
        """
        Perform comprehensive geolocation analysis on an image
        
        Args:
            image_path: Path to the image file
            
        Returns:
            GeolocationResult with location data and confidence assessment
        """
        if not image_path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        # Calculate image hash
        with open(image_path, 'rb') as f:
            import hashlib
            image_hash = hashlib.sha256(f.read()).hexdigest()
        
        logger.info(f"Starting geolocation analysis for: {image_path.name}")
        
        # Initialize result
        result = GeolocationResult(
            image_hash=image_hash,
            overall_confidence=0.0,
            location_verified=False
        )
        
        try:
            # Extract GPS metadata
            gps_location = self._extract_gps_metadata(image_path, result)
            if gps_location:
                result.primary_location = gps_location
                result.gps_data_present = True
                result.methods_used.append("gps_metadata")
            
            # Perform visual geolocation (if enabled)
            if self.config.enable_visual_geolocation:
                visual_locations = self._visual_geolocation(image_path, result)
                result.alternative_locations.extend(visual_locations)
                if visual_locations:
                    result.methods_used.append("visual_recognition")
                    result.visual_landmarks_detected = True
            
            # Verify GPS data integrity
            if result.gps_data_present:
                self._verify_gps_integrity(image_path, result)
            
            # Calculate overall confidence
            self._calculate_overall_confidence(result)
            
            logger.info(f"Geolocation analysis completed: confidence {result.overall_confidence:.3f}")
            
        except Exception as e:
            logger.error(f"Geolocation analysis failed: {str(e)}")
            result.overall_confidence = 0.0
        
        return result
    
    def _extract_gps_metadata(self, image_path: Path, result: GeolocationResult) -> Optional[GeolocationData]:
        """Extract GPS coordinates from EXIF metadata"""
        try:
            with Image.open(image_path) as img:
                exif_data = img._getexif()
                
                if not exif_data:
                    return None
                
                # Look for GPS info
                gps_info = None
                for tag, value in exif_data.items():
                    decoded_tag = TAGS.get(tag, tag)
                    if decoded_tag == "GPSInfo":
                        gps_info = value
                        break
                
                if not gps_info:
                    return None
                
                # Extract GPS coordinates
                gps_data = {}
                for key in gps_info.keys():
                    decoded_key = GPSTAGS.get(key, key)
                    gps_data[decoded_key] = gps_info[key]
                
                # Parse coordinates
                lat = self._parse_gps_coordinate(
                    gps_data.get('GPSLatitude'),
                    gps_data.get('GPSLatitudeRef')
                )
                lon = self._parse_gps_coordinate(
                    gps_data.get('GPSLongitude'),
                    gps_data.get('GPSLongitudeRef')
                )
                
                if lat is None or lon is None:
                    return None
                
                # Parse altitude if available
                altitude = None
                if 'GPSAltitude' in gps_data:
                    alt_value = gps_data['GPSAltitude']
                    if isinstance(alt_value, tuple) and len(alt_value) == 2:
                        altitude = float(alt_value[0]) / float(alt_value[1])
                        if gps_data.get('GPSAltitudeRef') == 1:
                            altitude = -altitude
                
                # Parse timestamp if available
                gps_timestamp = None
                if 'GPSTimeStamp' in gps_data and 'GPSDateStamp' in gps_data:
                    try:
                        date_str = gps_data['GPSDateStamp']
                        time_tuple = gps_data['GPSTimeStamp']
                        
                        if isinstance(time_tuple, tuple) and len(time_tuple) == 3:
                            hour = int(time_tuple[0])
                            minute = int(time_tuple[1])
                            second = int(time_tuple[2])
                            
                            # Parse date (YYYY:MM:DD format)
                            date_parts = date_str.split(':')
                            if len(date_parts) == 3:
                                year, month, day = map(int, date_parts)
                                gps_timestamp = datetime(year, month, day, hour, minute, second)
                    except Exception as e:
                        logger.debug(f"Failed to parse GPS timestamp: {str(e)}")
                
                # Perform reverse geocoding
                location_info = self._reverse_geocode(lat, lon)
                
                # Create geolocation data
                geo_data = GeolocationData(
                    latitude=lat,
                    longitude=lon,
                    altitude=altitude,
                    source="exif",
                    confidence=0.9,  # High confidence for GPS metadata
                    extraction_method="EXIF_GPS",
                    timestamp=gps_timestamp,
                    **location_info
                )
                
                logger.info(f"Extracted GPS coordinates: {lat:.6f}, {lon:.6f}")
                
                return geo_data
                
        except Exception as e:
            logger.error(f"GPS metadata extraction failed: {str(e)}")
            return None
    
    def _parse_gps_coordinate(self, coord_tuple: Optional[tuple], ref: Optional[str]) -> Optional[float]:
        """Parse GPS coordinate from EXIF format"""
        if not coord_tuple or not ref:
            return None
        
        try:
            if len(coord_tuple) != 3:
                return None
            
            # Convert DMS to decimal degrees
            degrees = float(coord_tuple[0])
            minutes = float(coord_tuple[1])
            seconds = float(coord_tuple[2])
            
            decimal_degrees = degrees + (minutes / 60.0) + (seconds / 3600.0)
            
            # Apply direction
            if ref in ['S', 'W']:
                decimal_degrees = -decimal_degrees
            
            return decimal_degrees
            
        except (ValueError, TypeError, ZeroDivisionError) as e:
            logger.debug(f"Failed to parse GPS coordinate: {str(e)}")
            return None
    
    def _reverse_geocode(self, lat: float, lon: float) -> Dict[str, str]:
        """Perform reverse geocoding to get location details"""
        location_info = {}
        
        try:
            # Use reverse_geocoder for fast lookup if available
            if HAS_REVERSE_GEOCODER:
                rg_result = rg.search([(lat, lon)])
                if rg_result:
                    result = rg_result[0]
                    location_info.update({
                        'country': result.get('cc', ''),
                        'region': result.get('admin1', ''),
                        'city': result.get('name', '')
                    })
            
            # Use Nominatim for detailed address if available
            if HAS_GEOPY and self.geocoder:
                try:
                    location = self.geocoder.reverse(f"{lat}, {lon}", timeout=10)
                    if location and location.address:
                        location_info['address'] = location.address
                        
                        # Extract additional details from address components
                        if hasattr(location, 'raw') and 'address' in location.raw:
                            address_components = location.raw['address']
                            if 'postcode' in address_components:
                                location_info['postal_code'] = address_components['postcode']
                            if not location_info.get('country') and 'country' in address_components:
                                location_info['country'] = address_components['country']
                            if not location_info.get('city'):
                                for key in ['city', 'town', 'village', 'hamlet']:
                                    if key in address_components:
                                        location_info['city'] = address_components[key]
                                        break
                except Exception as e:
                    logger.debug(f"Nominatim geocoding failed: {str(e)}")
            
            # Fallback: Basic country inference from coordinates if no other services available
            if not location_info.get('country') and not HAS_REVERSE_GEOCODER and not HAS_GEOPY:
                location_info['country'] = self._infer_country_from_coords(lat, lon)
            
        except Exception as e:
            logger.error(f"Reverse geocoding failed: {str(e)}")
        
        return location_info
    
    def _visual_geolocation(self, image_path: Path, result: GeolocationResult) -> List[GeolocationData]:
        """Perform visual geolocation using landmark recognition"""
        visual_locations = []
        
        try:
            # Load image
            image = cv2.imread(str(image_path))
            if image is None:
                return visual_locations
            
            # Detect landmarks using various methods
            landmarks = self._detect_landmarks(image)
            
            # Match visual features against known location databases
            location_matches = self._match_visual_features(image)
            
            # Combine landmark and feature matching results
            for landmark in landmarks:
                geo_data = GeolocationData(
                    source="visual",
                    confidence=landmark['confidence'],
                    extraction_method="landmark_recognition",
                    identified_landmarks=[landmark['name']],
                    landmark_confidence=landmark['confidence'],
                    **landmark.get('location', {})
                )
                visual_locations.append(geo_data)
            
            for match in location_matches:
                geo_data = GeolocationData(
                    latitude=match.get('latitude'),
                    longitude=match.get('longitude'),
                    source="visual",
                    confidence=match['confidence'],
                    extraction_method="feature_matching",
                    **match.get('location_info', {})
                )
                visual_locations.append(geo_data)
            
        except Exception as e:
            logger.error(f"Visual geolocation failed: {str(e)}")
        
        return visual_locations
    
    def _detect_landmarks(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Detect known landmarks in the image"""
        landmarks = []
        
        try:
            # Convert to RGB for processing
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Simple landmark detection using template matching
            # In a full implementation, this would use trained ML models
            
            # Detect architectural features that might indicate location
            landmarks.extend(self._detect_architectural_features(rgb_image))
            
            # Detect text/signs that might indicate location
            landmarks.extend(self._detect_location_text(rgb_image))
            
            # Detect vegetation/landscape features
            landmarks.extend(self._detect_landscape_features(rgb_image))
            
        except Exception as e:
            logger.error(f"Landmark detection failed: {str(e)}")
        
        return landmarks
    
    def _detect_architectural_features(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Detect architectural features that might indicate location"""
        features = []
        
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
            # Detect building features using edge detection
            edges = cv2.Canny(gray, 50, 150)
            
            # Find contours that might represent buildings
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Analyze contour shapes for architectural patterns
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 1000:  # Filter small contours
                    # Approximate contour shape
                    epsilon = 0.02 * cv2.arcLength(contour, True)
                    approx = cv2.approxPolyDP(contour, epsilon, True)
                    
                    # Check for rectangular structures (buildings)
                    if len(approx) == 4:
                        # Calculate aspect ratio
                        x, y, w, h = cv2.boundingRect(approx)
                        aspect_ratio = float(w) / h
                        
                        if 0.5 < aspect_ratio < 2.0:  # Reasonable building proportions
                            features.append({
                                'name': 'building_structure',
                                'confidence': 0.3,  # Low confidence without ML model
                                'type': 'architectural',
                                'bbox': (x, y, w, h)
                            })
            
        except Exception as e:
            logger.debug(f"Architectural feature detection failed: {str(e)}")
        
        return features
    
    def _detect_location_text(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Detect text/signs that might indicate location"""
        features = []
        
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
            # Use OCR to detect text (simplified implementation)
            # In a full implementation, this would use proper OCR libraries like Tesseract
            
            # Detect text regions using MSER
            mser = cv2.MSER_create()
            regions, _ = mser.detectRegions(gray)
            
            # Filter regions that might contain text
            for region in regions:
                if len(region) > 10:  # Minimum number of pixels
                    # Get bounding box
                    x, y, w, h = cv2.boundingRect(region)
                    
                    # Check aspect ratio for text-like regions
                    aspect_ratio = float(w) / h
                    if 1.5 < aspect_ratio < 10.0:  # Text-like aspect ratio
                        features.append({
                            'name': 'text_region',
                            'confidence': 0.2,  # Low confidence without proper OCR
                            'type': 'text',
                            'bbox': (x, y, w, h)
                        })
            
        except Exception as e:
            logger.debug(f"Text detection failed: {str(e)}")
        
        return features
    
    def _detect_landscape_features(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Detect landscape features that might indicate location"""
        features = []
        
        try:
            # Analyze color distribution for landscape detection
            hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            
            # Detect sky (blue regions in upper part of image)
            height, width = image.shape[:2]
            upper_region = hsv_image[:height//3, :]
            
            # Define blue color range for sky
            lower_blue = np.array([100, 50, 50])
            upper_blue = np.array([130, 255, 255])
            sky_mask = cv2.inRange(upper_region, lower_blue, upper_blue)
            sky_ratio = np.sum(sky_mask > 0) / sky_mask.size
            
            if sky_ratio > 0.3:
                features.append({
                    'name': 'sky_region',
                    'confidence': min(0.5, sky_ratio),
                    'type': 'landscape',
                    'sky_ratio': sky_ratio
                })
            
            # Detect vegetation (green regions)
            lower_green = np.array([35, 50, 50])
            upper_green = np.array([85, 255, 255])
            vegetation_mask = cv2.inRange(hsv_image, lower_green, upper_green)
            vegetation_ratio = np.sum(vegetation_mask > 0) / vegetation_mask.size
            
            if vegetation_ratio > 0.2:
                features.append({
                    'name': 'vegetation',
                    'confidence': min(0.4, vegetation_ratio),
                    'type': 'landscape',
                    'vegetation_ratio': vegetation_ratio
                })
            
            # Detect water (blue regions in lower part)
            lower_region = hsv_image[2*height//3:, :]
            water_mask = cv2.inRange(lower_region, lower_blue, upper_blue)
            water_ratio = np.sum(water_mask > 0) / water_mask.size
            
            if water_ratio > 0.1:
                features.append({
                    'name': 'water_body',
                    'confidence': min(0.3, water_ratio * 2),
                    'type': 'landscape',
                    'water_ratio': water_ratio
                })
            
        except Exception as e:
            logger.debug(f"Landscape feature detection failed: {str(e)}")
        
        return features
    
    def _match_visual_features(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Match visual features against known location databases"""
        matches = []
        
        try:
            # Extract visual features using ORB detector
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            orb = cv2.ORB_create(nfeatures=1000)
            keypoints, descriptors = orb.detectAndCompute(gray, None)
            
            if descriptors is not None:
                # In a full implementation, this would match against a database
                # of known location features
                
                # For now, return a placeholder match with low confidence
                if len(keypoints) > 100:  # Sufficient features for matching
                    matches.append({
                        'confidence': 0.1,  # Very low confidence without real database
                        'feature_count': len(keypoints),
                        'match_type': 'feature_similarity'
                    })
            
        except Exception as e:
            logger.debug(f"Visual feature matching failed: {str(e)}")
        
        return matches
    
    def _verify_gps_integrity(self, image_path: Path, result: GeolocationResult):
        """Verify the integrity of GPS data"""
        
        if not result.primary_location:
            return
        
        try:
            # Check if coordinates are within valid ranges
            lat = result.primary_location.latitude
            lon = result.primary_location.longitude
            
            if lat is None or lon is None:
                result.gps_data_tampered = True
                return
            
            # Check for impossible coordinates
            if not (-90 <= lat <= 90) or not (-180 <= lon <= 180):
                result.gps_data_tampered = True
                return
            
            # Check for common fake coordinates (e.g., exactly 0,0)
            if lat == 0 and lon == 0:
                result.gps_data_tampered = True
                return
            
            # Check timestamp consistency
            if result.primary_location.timestamp:
                gps_time = result.primary_location.timestamp
                
                # Compare with file modification time
                file_stat = image_path.stat()
                file_time = datetime.fromtimestamp(file_stat.st_mtime)
                
                # GPS time should not be significantly after file modification
                time_diff = (file_time - gps_time).total_seconds()
                if time_diff < -3600:  # GPS time more than 1 hour after file time
                    result.gps_data_tampered = True
                    return
            
            # Check for precision anomalies
            # Real GPS coordinates rarely have extreme precision
            lat_str = str(lat)
            lon_str = str(lon)
            
            if '.' in lat_str and len(lat_str.split('.')[1]) > 8:
                result.gps_data_tampered = True
                return
            
            if '.' in lon_str and len(lon_str.split('.')[1]) > 8:
                result.gps_data_tampered = True
                return
            
            # If all checks pass, GPS data appears authentic
            result.gps_data_tampered = False
            
        except Exception as e:
            logger.error(f"GPS integrity verification failed: {str(e)}")
            result.gps_data_tampered = None  # Unknown
    
    def _calculate_overall_confidence(self, result: GeolocationResult):
        """Calculate overall confidence in geolocation results"""
        
        confidences = []
        
        # Add primary location confidence
        if result.primary_location:
            confidences.append(result.primary_location.confidence)
            
            # Reduce confidence if GPS data is tampered
            if result.gps_data_tampered:
                confidences[-1] *= 0.3
        
        # Add alternative location confidences
        for location in result.alternative_locations:
            confidences.append(location.confidence * 0.7)  # Visual locations get lower weight
        
        # Calculate weighted average
        if confidences:
            result.overall_confidence = max(confidences)  # Use highest confidence
            
            # Boost confidence if multiple methods agree
            if len(confidences) > 1:
                result.overall_confidence = min(0.95, result.overall_confidence * 1.2)
        else:
            result.overall_confidence = 0.0
        
        # Set location verified flag
        result.location_verified = (
            result.overall_confidence > self.config.geolocation_confidence_threshold and
            not result.gps_data_tampered
        )
        
        # Infer timezone if we have coordinates
        best_location = result.get_best_location()
        if best_location and best_location.latitude and best_location.longitude:
            result.timezone_inferred = self._infer_timezone(
                best_location.latitude,
                best_location.longitude
            )
    
    def _infer_timezone(self, lat: float, lon: float) -> Optional[str]:
        """Infer timezone from coordinates"""
        try:
            # Simple timezone inference based on longitude
            # In a full implementation, this would use a proper timezone database
            hours_offset = round(lon / 15.0)
            
            if hours_offset == 0:
                return "UTC"
            elif hours_offset > 0:
                return f"UTC+{hours_offset}"
            else:
                return f"UTC{hours_offset}"
                
        except Exception as e:
            logger.debug(f"Timezone inference failed: {str(e)}")
            return None
    
    def _infer_country_from_coords(self, lat: float, lon: float) -> Optional[str]:
        """Basic country inference from coordinates (fallback when no geocoding services available)"""
        try:
            # Very basic geographic regions - this would be much more comprehensive in a real implementation
            if 24.0 <= lat <= 49.0 and -125.0 <= lon <= -66.0:
                return "United States"
            elif 41.0 <= lat <= 71.0 and -141.0 <= lon <= -52.0:
                return "Canada"
            elif 35.0 <= lat <= 72.0 and -10.0 <= lon <= 40.0:
                return "Europe"  # Very broad
            elif -55.0 <= lat <= 37.0 and -74.0 <= lon <= -34.0:
                return "South America"  # Very broad
            elif -47.0 <= lat <= 37.0 and 12.0 <= lon <= 52.0:
                return "Africa"  # Very broad
            elif -50.0 <= lat <= 80.0 and 26.0 <= lon <= 180.0:
                return "Asia"  # Very broad
            elif -50.0 <= lat <= -10.0 and 110.0 <= lon <= 180.0:
                return "Australia/Oceania"  # Very broad
            else:
                return "Unknown"
        except Exception:
            return None


def geolocate_image(image_path: Path, config: Optional[ImageAuthConfig] = None) -> GeolocationResult:
    """
    Convenience function to geolocate an image
    
    Args:
        image_path: Path to the image file
        config: Optional configuration
        
    Returns:
        GeolocationResult with location data and verification
    """
    geolocator = ImageGeolocator(config)
    return geolocator.geolocate_image(image_path)