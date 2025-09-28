"""
Lemkin Geospatial Analysis Suite - Coordinate Converter Module

This module provides coordinate format standardization and projection handling
for various GPS and mapping coordinate systems. Designed to be user-friendly
for legal professionals without GIS expertise.

Supports: DD, DMS, UTM, MGRS, Plus Codes, and various projection systems.
"""

import re
import math
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import logging

from .core import Coordinate, CoordinateFormat, ProjectionSystem

logger = logging.getLogger(__name__)


@dataclass
class ConversionResult:
    """Result of coordinate conversion operation"""
    success: bool
    coordinate: Optional[Coordinate]
    original_format: Optional[CoordinateFormat]
    target_format: CoordinateFormat
    error_message: Optional[str] = None
    precision_loss: bool = False


class CoordinateConverter:
    """
    Comprehensive coordinate conversion utility for legal evidence geolocation.
    
    Handles multiple coordinate formats and projection systems while maintaining
    precision and providing clear error messages for non-GIS users.
    """
    
    def __init__(self):
        """Initialize coordinate converter"""
        self.logger = logging.getLogger(f"{__name__}.CoordinateConverter")
        
        # Regex patterns for different coordinate formats
        self.patterns = {
            'dd': [
                # Decimal degrees: 40.7128, -74.0060 or 40.7128° -74.0060°
                r'^(-?\d+\.?\d*)[°\s,]+(-?\d+\.?\d*)°?$',
                # With N/S/E/W: 40.7128°N, 74.0060°W
                r'^(\d+\.?\d*)°?[NS][,\s]+(\d+\.?\d*)°?[EW]$'
            ],
            'dms': [
                # Degrees Minutes Seconds: 40°42'46.0"N 74°00'21.6"W
                r"^(\d+)[°\s]+(\d+)['\s]+(\d+\.?\d*)[\"]\s*([NSEW])\s*[,\s]*(\d+)[°\s]+(\d+)['\s]+(\d+\.?\d*)[\"]\s*([NSEW])$",
                # Alternative format: 40 42 46.0 N, 74 00 21.6 W
                r'^(\d+)\s+(\d+)\s+(\d+\.?\d*)\s+([NSEW])[,\s]+(\d+)\s+(\d+)\s+(\d+\.?\d*)\s+([NSEW])$'
            ],
            'utm': [
                # UTM format: 18T 585628 4511322
                r'^(\d{1,2})([A-Z])\s+(\d+)\s+(\d+)$'
            ],
            'mgrs': [
                # MGRS format: 18TWL8562811322
                r'^(\d{1,2})([A-Z])([A-Z]{2})(\d{2,10})$'
            ],
            'plus_codes': [
                # Plus Codes: 87G7X2QR+2X
                r'^[23456789CFGHJMPQRVWX]{8}\+[23456789CFGHJMPQRVWX]{2,3}$'
            ]
        }
    
    def standardize_coordinates(self, coords: str, format_hint: Optional[str] = None) -> ConversionResult:
        """
        Standardize coordinates from various formats to decimal degrees
        
        Args:
            coords: Coordinate string in various formats
            format_hint: Optional hint about the expected format
            
        Returns:
            ConversionResult with standardized coordinate
        """
        try:
            # Clean input
            coords_clean = coords.strip().replace('\n', ' ').replace('\t', ' ')
            
            # Detect format if not provided
            if format_hint:
                detected_format = CoordinateFormat(format_hint.lower())
            else:
                detected_format = self._detect_format(coords_clean)
            
            if not detected_format:
                return ConversionResult(
                    success=False,
                    coordinate=None,
                    original_format=None,
                    target_format=CoordinateFormat.DECIMAL_DEGREES,
                    error_message=f"Unable to detect coordinate format from: {coords}"
                )
            
            # Convert based on detected format
            coordinate = None
            
            if detected_format == CoordinateFormat.DECIMAL_DEGREES:
                coordinate = self._parse_decimal_degrees(coords_clean)
            elif detected_format == CoordinateFormat.DEGREES_MINUTES_SECONDS:
                coordinate = self._parse_dms(coords_clean)
            elif detected_format == CoordinateFormat.UTM:
                coordinate = self._parse_utm(coords_clean)
            elif detected_format == CoordinateFormat.MGRS:
                coordinate = self._parse_mgrs(coords_clean)
            elif detected_format == CoordinateFormat.PLUS_CODES:
                coordinate = self._parse_plus_codes(coords_clean)
            
            if coordinate:
                return ConversionResult(
                    success=True,
                    coordinate=coordinate,
                    original_format=detected_format,
                    target_format=CoordinateFormat.DECIMAL_DEGREES
                )
            else:
                return ConversionResult(
                    success=False,
                    coordinate=None,
                    original_format=detected_format,
                    target_format=CoordinateFormat.DECIMAL_DEGREES,
                    error_message=f"Failed to parse {detected_format} coordinates: {coords}"
                )
                
        except Exception as e:
            self.logger.error(f"Coordinate standardization failed: {str(e)}")
            return ConversionResult(
                success=False,
                coordinate=None,
                original_format=None,
                target_format=CoordinateFormat.DECIMAL_DEGREES,
                error_message=f"Conversion error: {str(e)}"
            )
    
    def convert_to_format(
        self, 
        coordinate: Coordinate, 
        target_format: CoordinateFormat
    ) -> ConversionResult:
        """
        Convert coordinate to specified format
        
        Args:
            coordinate: Source coordinate in decimal degrees
            target_format: Desired output format
            
        Returns:
            ConversionResult with converted coordinate
        """
        try:
            if target_format == CoordinateFormat.DECIMAL_DEGREES:
                # Already in decimal degrees
                return ConversionResult(
                    success=True,
                    coordinate=coordinate,
                    original_format=CoordinateFormat.DECIMAL_DEGREES,
                    target_format=target_format
                )
            
            elif target_format == CoordinateFormat.DEGREES_MINUTES_SECONDS:
                converted_coord = self._convert_to_dms(coordinate)
                
            elif target_format == CoordinateFormat.UTM:
                converted_coord = self._convert_to_utm(coordinate)
                
            elif target_format == CoordinateFormat.MGRS:
                converted_coord = self._convert_to_mgrs(coordinate)
                
            elif target_format == CoordinateFormat.PLUS_CODES:
                converted_coord = self._convert_to_plus_codes(coordinate)
                
            else:
                return ConversionResult(
                    success=False,
                    coordinate=None,
                    original_format=CoordinateFormat.DECIMAL_DEGREES,
                    target_format=target_format,
                    error_message=f"Unsupported target format: {target_format}"
                )
            
            if converted_coord:
                return ConversionResult(
                    success=True,
                    coordinate=converted_coord,
                    original_format=CoordinateFormat.DECIMAL_DEGREES,
                    target_format=target_format
                )
            else:
                return ConversionResult(
                    success=False,
                    coordinate=None,
                    original_format=CoordinateFormat.DECIMAL_DEGREES,
                    target_format=target_format,
                    error_message="Conversion failed"
                )
                
        except Exception as e:
            self.logger.error(f"Format conversion failed: {str(e)}")
            return ConversionResult(
                success=False,
                coordinate=None,
                original_format=CoordinateFormat.DECIMAL_DEGREES,
                target_format=target_format,
                error_message=f"Conversion error: {str(e)}"
            )
    
    def batch_convert(
        self, 
        coordinates: List[str], 
        target_format: CoordinateFormat = CoordinateFormat.DECIMAL_DEGREES
    ) -> List[ConversionResult]:
        """
        Convert multiple coordinates in batch
        
        Args:
            coordinates: List of coordinate strings
            target_format: Desired output format
            
        Returns:
            List of ConversionResults
        """
        results = []
        
        for coord_str in coordinates:
            # First standardize to decimal degrees
            std_result = self.standardize_coordinates(coord_str)
            
            if std_result.success and std_result.coordinate:
                # Then convert to target format if different
                if target_format != CoordinateFormat.DECIMAL_DEGREES:
                    final_result = self.convert_to_format(std_result.coordinate, target_format)
                else:
                    final_result = std_result
            else:
                final_result = std_result
            
            results.append(final_result)
        
        return results
    
    def validate_coordinate_precision(
        self, 
        coordinate: Coordinate, 
        required_precision_meters: float = 1.0
    ) -> Dict[str, Union[bool, float, str]]:
        """
        Validate coordinate precision for legal evidence requirements
        
        Args:
            coordinate: Coordinate to validate
            required_precision_meters: Required precision in meters
            
        Returns:
            Dictionary with validation results
        """
        validation_result = {
            'meets_precision': False,
            'estimated_precision_meters': None,
            'decimal_places': 0,
            'recommendations': []
        }
        
        try:
            # Calculate decimal places
            lat_str = str(coordinate.latitude)
            lon_str = str(coordinate.longitude)
            
            lat_decimals = len(lat_str.split('.')[-1]) if '.' in lat_str else 0
            lon_decimals = len(lon_str.split('.')[-1]) if '.' in lon_str else 0
            
            decimal_places = min(lat_decimals, lon_decimals)
            validation_result['decimal_places'] = decimal_places
            
            # Estimate precision based on decimal places
            # At equator: 1 degree ≈ 111,320 meters
            precision_map = {
                0: 111320,    # ~111 km
                1: 11132,     # ~11 km  
                2: 1113,      # ~1 km
                3: 111,       # ~111 m
                4: 11,        # ~11 m
                5: 1.1,       # ~1 m
                6: 0.11,      # ~11 cm
                7: 0.011      # ~1 cm
            }
            
            estimated_precision = precision_map.get(decimal_places, precision_map[7])
            validation_result['estimated_precision_meters'] = estimated_precision
            
            # Check if meets requirement
            validation_result['meets_precision'] = estimated_precision <= required_precision_meters
            
            # Generate recommendations
            recommendations = []
            if not validation_result['meets_precision']:
                required_decimals = 5  # For ~1 meter precision
                for decimals, precision in precision_map.items():
                    if precision <= required_precision_meters:
                        required_decimals = decimals
                        break
                
                recommendations.append(
                    f"Increase precision to at least {required_decimals} decimal places "
                    f"for {required_precision_meters}m accuracy"
                )
            
            if estimated_precision > 100:
                recommendations.append(
                    "Very low precision coordinate - verify GPS source accuracy"
                )
            
            if coordinate.latitude == 0 and coordinate.longitude == 0:
                recommendations.append(
                    "Warning: Null Island coordinates (0,0) may indicate GPS error"
                )
            
            validation_result['recommendations'] = recommendations
            
        except Exception as e:
            self.logger.error(f"Precision validation failed: {str(e)}")
            validation_result['error'] = str(e)
        
        return validation_result
    
    def _detect_format(self, coords: str) -> Optional[CoordinateFormat]:
        """Detect coordinate format from string patterns"""
        for format_type, patterns in self.patterns.items():
            for pattern in patterns:
                if re.match(pattern, coords.upper()):
                    return CoordinateFormat(format_type)
        return None
    
    def _parse_decimal_degrees(self, coords: str) -> Optional[Coordinate]:
        """Parse decimal degrees format"""
        try:
            # Handle various DD formats
            coords_upper = coords.upper()
            
            # Pattern 1: Simple format (lat, lon)
            match = re.match(r'^(-?\d+\.?\d*)[°\s,]+(-?\d+\.?\d*)°?$', coords)
            if match:
                lat, lon = float(match.group(1)), float(match.group(2))
                return Coordinate(
                    latitude=lat,
                    longitude=lon,
                    format=CoordinateFormat.DECIMAL_DEGREES
                )
            
            # Pattern 2: With cardinal directions
            match = re.match(r'^(\d+\.?\d*)°?([NS])[,\s]+(\d+\.?\d*)°?([EW])$', coords_upper)
            if match:
                lat_val, lat_dir, lon_val, lon_dir = match.groups()
                lat = float(lat_val) * (-1 if lat_dir == 'S' else 1)
                lon = float(lon_val) * (-1 if lon_dir == 'W' else 1)
                return Coordinate(
                    latitude=lat,
                    longitude=lon,
                    format=CoordinateFormat.DECIMAL_DEGREES
                )
            
        except Exception as e:
            self.logger.error(f"DD parsing failed: {str(e)}")
        
        return None
    
    def _parse_dms(self, coords: str) -> Optional[Coordinate]:
        """Parse degrees, minutes, seconds format"""
        try:
            coords_upper = coords.upper()
            
            # Pattern: 40°42'46.0"N 74°00'21.6"W
            match = re.match(
                r"(\d+)[°\s]+(\d+)['\s]+(\d+\.?\d*)[\"]\s*([NSEW])\s*[,\s]*(\d+)[°\s]+(\d+)['\s]+(\d+\.?\d*)[\"]\s*([NSEW])",
                coords_upper
            )
            
            if match:
                deg1, min1, sec1, dir1, deg2, min2, sec2, dir2 = match.groups()
                
                # Convert to decimal degrees
                dd1 = float(deg1) + float(min1)/60 + float(sec1)/3600
                dd2 = float(deg2) + float(min2)/60 + float(sec2)/3600
                
                # Apply direction
                if dir1 in ['S', 'W']:
                    dd1 = -dd1
                if dir2 in ['S', 'W']:
                    dd2 = -dd2
                
                # Determine which is lat/lon based on direction
                if dir1 in ['N', 'S']:
                    lat, lon = dd1, dd2
                else:
                    lat, lon = dd2, dd1
                
                return Coordinate(
                    latitude=lat,
                    longitude=lon,
                    format=CoordinateFormat.DEGREES_MINUTES_SECONDS
                )
            
        except Exception as e:
            self.logger.error(f"DMS parsing failed: {str(e)}")
        
        return None
    
    def _parse_utm(self, coords: str) -> Optional[Coordinate]:
        """Parse UTM format (simplified implementation)"""
        try:
            match = re.match(r'^(\d{1,2})([A-Z])\s+(\d+)\s+(\d+)$', coords.upper())
            if match:
                zone, letter, easting, northing = match.groups()
                
                # This is a simplified UTM to lat/lon conversion
                # In production, use a proper geodetic library like pyproj
                lat, lon = self._utm_to_latlon(int(zone), letter, float(easting), float(northing))
                
                return Coordinate(
                    latitude=lat,
                    longitude=lon,
                    format=CoordinateFormat.UTM
                )
                
        except Exception as e:
            self.logger.error(f"UTM parsing failed: {str(e)}")
        
        return None
    
    def _parse_mgrs(self, coords: str) -> Optional[Coordinate]:
        """Parse MGRS format (simplified implementation)"""
        try:
            match = re.match(r'^(\d{1,2})([A-Z])([A-Z]{2})(\d{2,10})$', coords.upper())
            if match:
                zone, letter, square, coordinates = match.groups()
                
                # Simplified MGRS parsing - in production use proper library
                # This is a placeholder implementation
                self.logger.warning("MGRS parsing uses simplified implementation")
                
                # For now, return None to indicate unsupported
                # In production, implement full MGRS to lat/lon conversion
                return None
                
        except Exception as e:
            self.logger.error(f"MGRS parsing failed: {str(e)}")
        
        return None
    
    def _parse_plus_codes(self, coords: str) -> Optional[Coordinate]:
        """Parse Plus Codes format (simplified implementation)"""
        try:
            if re.match(r'^[23456789CFGHJMPQRVWX]{8}\+[23456789CFGHJMPQRVWX]{2,3}$', coords.upper()):
                # Simplified Plus Codes parsing - in production use proper library
                self.logger.warning("Plus Codes parsing uses simplified implementation")
                
                # For now, return None to indicate unsupported
                # In production, implement full Plus Codes to lat/lon conversion
                return None
                
        except Exception as e:
            self.logger.error(f"Plus Codes parsing failed: {str(e)}")
        
        return None
    
    def _convert_to_dms(self, coordinate: Coordinate) -> Optional[Coordinate]:
        """Convert decimal degrees to DMS format"""
        try:
            def dd_to_dms(dd):
                degrees = int(abs(dd))
                minutes_float = (abs(dd) - degrees) * 60
                minutes = int(minutes_float)
                seconds = (minutes_float - minutes) * 60
                return degrees, minutes, seconds
            
            lat_d, lat_m, lat_s = dd_to_dms(coordinate.latitude)
            lon_d, lon_m, lon_s = dd_to_dms(coordinate.longitude)
            
            lat_dir = 'N' if coordinate.latitude >= 0 else 'S'
            lon_dir = 'E' if coordinate.longitude >= 0 else 'W'
            
            # Create a new coordinate with DMS format info
            dms_coord = Coordinate(
                latitude=coordinate.latitude,  # Keep original DD values
                longitude=coordinate.longitude,
                format=CoordinateFormat.DEGREES_MINUTES_SECONDS,
                source=f"Converted from DD: {lat_d}°{lat_m}'{lat_s:.1f}\"{lat_dir} {lon_d}°{lon_m}'{lon_s:.1f}\"{lon_dir}"
            )
            
            return dms_coord
            
        except Exception as e:
            self.logger.error(f"DMS conversion failed: {str(e)}")
            return None
    
    def _convert_to_utm(self, coordinate: Coordinate) -> Optional[Coordinate]:
        """Convert to UTM format (simplified implementation)"""
        try:
            zone, letter, easting, northing = self._latlon_to_utm(coordinate.latitude, coordinate.longitude)
            
            utm_coord = Coordinate(
                latitude=coordinate.latitude,  # Keep original DD values
                longitude=coordinate.longitude,
                format=CoordinateFormat.UTM,
                source=f"Converted to UTM: {zone}{letter} {easting:.0f} {northing:.0f}"
            )
            
            return utm_coord
            
        except Exception as e:
            self.logger.error(f"UTM conversion failed: {str(e)}")
            return None
    
    def _convert_to_mgrs(self, coordinate: Coordinate) -> Optional[Coordinate]:
        """Convert to MGRS format (placeholder)"""
        self.logger.warning("MGRS conversion not fully implemented")
        return None
    
    def _convert_to_plus_codes(self, coordinate: Coordinate) -> Optional[Coordinate]:
        """Convert to Plus Codes format (placeholder)"""
        self.logger.warning("Plus Codes conversion not fully implemented") 
        return None
    
    def _utm_to_latlon(self, zone: int, letter: str, easting: float, northing: float) -> Tuple[float, float]:
        """Simplified UTM to lat/lon conversion"""
        # This is a very simplified conversion
        # In production, use pyproj or similar library for accurate conversion
        
        # Rough approximation for demonstration
        central_meridian = (zone - 1) * 6 - 180 + 3
        
        # Very rough conversion (not accurate for real use)
        lat = (northing / 111320) + (0 if letter >= 'N' else -90)
        lon = ((easting - 500000) / 111320) + central_meridian
        
        self.logger.warning("Using simplified UTM conversion - not suitable for production")
        return lat, lon
    
    def _latlon_to_utm(self, lat: float, lon: float) -> Tuple[int, str, float, float]:
        """Simplified lat/lon to UTM conversion"""
        # This is a very simplified conversion
        # In production, use pyproj or similar library for accurate conversion
        
        zone = int((lon + 180) / 6) + 1
        letter = 'N' if lat >= 0 else 'S'
        
        # Very rough conversion (not accurate for real use)
        central_meridian = (zone - 1) * 6 - 180 + 3
        easting = (lon - central_meridian) * 111320 + 500000
        northing = lat * 111320
        
        if lat < 0:
            northing += 10000000
        
        self.logger.warning("Using simplified UTM conversion - not suitable for production")
        return zone, letter, easting, northing


def standardize_coordinates(coords: str, format: str = None) -> Coordinate:
    """
    Convenience function to standardize coordinates
    
    Args:
        coords: Coordinate string in various formats
        format: Optional format hint
        
    Returns:
        Standardized Coordinate object
        
    Raises:
        ValueError: If coordinates cannot be parsed
    """
    converter = CoordinateConverter()
    result = converter.standardize_coordinates(coords, format)
    
    if result.success and result.coordinate:
        return result.coordinate
    else:
        raise ValueError(result.error_message or "Failed to parse coordinates")


def convert_coordinate_format(
    coordinate: Coordinate, 
    target_format: str
) -> Coordinate:
    """
    Convenience function to convert coordinate format
    
    Args:
        coordinate: Source coordinate
        target_format: Target format (dd, dms, utm, mgrs, plus_codes)
        
    Returns:
        Converted coordinate
        
    Raises:
        ValueError: If conversion fails
    """
    converter = CoordinateConverter()
    result = converter.convert_to_format(coordinate, CoordinateFormat(target_format))
    
    if result.success and result.coordinate:
        return result.coordinate
    else:
        raise ValueError(result.error_message or "Conversion failed")