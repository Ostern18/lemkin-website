"""
Lemkin Image Verification Suite - Metadata Forensics Module

This module implements comprehensive EXIF metadata forensic analysis for
image authenticity verification including camera fingerprinting, timestamp
validation, and manipulation detection through metadata inconsistencies.

Legal Compliance: Meets standards for digital evidence metadata analysis in legal proceedings
"""

import hashlib
import json
import logging
import re
import struct
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union

import cv2
import numpy as np
from PIL import Image, ExifTags
from PIL.ExifTags import TAGS, GPSTAGS

# Optional dependencies
try:
    import exifread
    HAS_EXIFREAD = True
except ImportError:
    exifread = None
    HAS_EXIFREAD = False

try:
    import magic
    HAS_MAGIC = True
except ImportError:
    magic = None
    HAS_MAGIC = False

from .core import (
    ImageMetadata,
    MetadataForensics,
    ImageFormat,
    ImageAuthConfig
)

logger = logging.getLogger(__name__)


class MetadataForensicsAnalyzer:
    """
    Comprehensive metadata forensics analyzer for image authenticity verification.
    
    Performs deep analysis of EXIF metadata, camera fingerprinting, timestamp
    validation, and detection of metadata manipulation or fabrication.
    """
    
    def __init__(self, config: Optional[ImageAuthConfig] = None):
        """Initialize the metadata forensics analyzer"""
        self.config = config or ImageAuthConfig()
        
        # Known camera databases (in full implementation, this would be extensive)
        self.camera_database = self._load_camera_database()
        self.software_signatures = self._load_software_signatures()
        
        logger.info("Metadata forensics analyzer initialized")
    
    def analyze_metadata(self, image_path: Path) -> MetadataForensics:
        """
        Perform comprehensive forensic analysis of image metadata
        
        Args:
            image_path: Path to the image file
            
        Returns:
            MetadataForensics with detailed authenticity assessment
        """
        if not image_path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        logger.info(f"Starting metadata forensics analysis for: {image_path.name}")
        
        # Extract comprehensive metadata
        metadata = self._extract_comprehensive_metadata(image_path)
        
        # Calculate image hash
        with open(image_path, 'rb') as f:
            image_hash = hashlib.sha256(f.read()).hexdigest()
        
        # Initialize forensics analysis
        forensics = MetadataForensics(
            image_hash=image_hash,
            metadata_source=metadata,
            metadata_authentic=True,
            metadata_confidence=0.8
        )
        
        try:
            # Perform various forensic analyses
            self._analyze_timestamp_consistency(metadata, forensics)
            self._analyze_camera_consistency(metadata, forensics)
            self._analyze_software_consistency(metadata, forensics)
            self._analyze_gps_consistency(metadata, forensics)
            self._analyze_exif_integrity(image_path, metadata, forensics)
            self._analyze_thumbnail_consistency(image_path, metadata, forensics)
            self._detect_metadata_manipulation(metadata, forensics)
            self._validate_camera_fingerprint(metadata, forensics)
            
            # Calculate overall authenticity assessment
            self._calculate_authenticity_assessment(forensics)
            
            logger.info(f"Metadata forensics completed: authentic={forensics.metadata_authentic}")
            
        except Exception as e:
            logger.error(f"Metadata forensics analysis failed: {str(e)}")
            forensics.metadata_authentic = False
            forensics.metadata_confidence = 0.0
            forensics.requires_deeper_analysis = True
        
        return forensics
    
    def _extract_comprehensive_metadata(self, image_path: Path) -> ImageMetadata:
        """Extract comprehensive metadata from image file"""
        
        # Get basic file information
        file_stat = image_path.stat()
        file_size = file_stat.st_size
        
        # Detect file format
        file_format = self._detect_image_format(image_path)
        
        # Calculate file hashes
        with open(image_path, 'rb') as f:
            file_data = f.read()
            md5_hash = hashlib.md5(file_data).hexdigest()
            sha256_hash = hashlib.sha256(file_data).hexdigest()
        
        # Initialize metadata with basic information
        metadata = ImageMetadata(
            file_name=image_path.name,
            file_path=str(image_path),
            file_size_bytes=file_size,
            file_format=file_format,
            width=1,  # Will be updated
            height=1,  # Will be updated
            md5_hash=md5_hash,
            sha256_hash=sha256_hash,
            modification_time=datetime.fromtimestamp(file_stat.st_mtime),
            exif_data={}
        )
        
        try:
            # Extract EXIF data using PIL
            with Image.open(image_path) as img:
                metadata.width = img.width
                metadata.height = img.height
                metadata.color_space = img.mode
                
                # Extract EXIF data
                exif_data = img._getexif()
                if exif_data:
                    metadata.exif_data = self._process_exif_data(exif_data)
                    self._populate_metadata_from_exif(metadata, exif_data)
            
            # Extract additional metadata using exifread if available
            if HAS_EXIFREAD:
                with open(image_path, 'rb') as f:
                    additional_tags = exifread.process_file(f, details=True)
                    self._process_additional_exif(metadata, additional_tags)
            
            # Calculate perceptual hash
            metadata.perceptual_hash = self._calculate_perceptual_hash(image_path)
            
        except Exception as e:
            logger.error(f"Metadata extraction failed: {str(e)}")
        
        return metadata
    
    def _detect_image_format(self, image_path: Path) -> ImageFormat:
        """Detect image format using multiple methods"""
        try:
            # Try using python-magic if available
            if HAS_MAGIC:
                file_type = magic.from_file(str(image_path), mime=True)
                
                format_map = {
                    'image/jpeg': ImageFormat.JPEG,
                    'image/png': ImageFormat.PNG,
                    'image/tiff': ImageFormat.TIFF,
                    'image/bmp': ImageFormat.BMP,
                    'image/gif': ImageFormat.GIF,
                    'image/webp': ImageFormat.WEBP
                }
                
                if file_type in format_map:
                    return format_map[file_type]
            
            # Fallback to PIL
            with Image.open(image_path) as img:
                format_str = img.format.lower() if img.format else 'unknown'
                
                if format_str in ['jpeg', 'jpg']:
                    return ImageFormat.JPEG
                elif format_str == 'png':
                    return ImageFormat.PNG
                elif format_str in ['tiff', 'tif']:
                    return ImageFormat.TIFF
                elif format_str == 'bmp':
                    return ImageFormat.BMP
                elif format_str == 'gif':
                    return ImageFormat.GIF
                elif format_str == 'webp':
                    return ImageFormat.WEBP
            
        except Exception as e:
            logger.debug(f"Format detection failed: {str(e)}")
        
        # Default fallback
        extension = image_path.suffix.lower()
        if extension in ['.jpg', '.jpeg']:
            return ImageFormat.JPEG
        elif extension == '.png':
            return ImageFormat.PNG
        else:
            return ImageFormat.JPEG  # Default assumption
    
    def _process_exif_data(self, exif_data: dict) -> Dict[str, Any]:
        """Process raw EXIF data into readable format"""
        processed = {}
        
        for tag_id, value in exif_data.items():
            try:
                tag_name = TAGS.get(tag_id, f"Tag{tag_id}")
                
                # Handle special cases
                if tag_name == "GPSInfo":
                    processed[tag_name] = self._process_gps_info(value)
                elif isinstance(value, bytes):
                    try:
                        processed[tag_name] = value.decode('utf-8', errors='ignore')
                    except:
                        processed[tag_name] = str(value)
                else:
                    processed[tag_name] = str(value)
                    
            except Exception as e:
                logger.debug(f"Failed to process EXIF tag {tag_id}: {str(e)}")
        
        return processed
    
    def _process_gps_info(self, gps_info: dict) -> Dict[str, Any]:
        """Process GPS information from EXIF"""
        processed_gps = {}
        
        for key, value in gps_info.items():
            try:
                tag_name = GPSTAGS.get(key, f"GPSTag{key}")
                processed_gps[tag_name] = str(value)
            except Exception as e:
                logger.debug(f"Failed to process GPS tag {key}: {str(e)}")
        
        return processed_gps
    
    def _populate_metadata_from_exif(self, metadata: ImageMetadata, exif_data: dict):
        """Populate metadata fields from EXIF data"""
        
        # Camera information
        if 'Make' in exif_data:
            metadata.camera_make = str(exif_data['Make']).strip()
        if 'Model' in exif_data:
            metadata.camera_model = str(exif_data['Model']).strip()
        if 'LensModel' in exif_data:
            metadata.lens_model = str(exif_data['LensModel']).strip()
        
        # Capture settings
        if 'ISOSpeedRatings' in exif_data:
            try:
                metadata.iso_speed = int(exif_data['ISOSpeedRatings'])
            except:
                pass
        
        if 'FNumber' in exif_data:
            metadata.aperture = str(exif_data['FNumber'])
        
        if 'ExposureTime' in exif_data:
            metadata.shutter_speed = str(exif_data['ExposureTime'])
        
        if 'FocalLength' in exif_data:
            metadata.focal_length = str(exif_data['FocalLength'])
        
        if 'Flash' in exif_data:
            try:
                flash_value = int(exif_data['Flash'])
                metadata.flash_used = (flash_value & 1) != 0
            except:
                pass
        
        # Timestamps
        if 'DateTime' in exif_data:
            metadata.creation_time = self._parse_exif_datetime(str(exif_data['DateTime']))
        
        if 'DateTimeOriginal' in exif_data:
            metadata.creation_time = self._parse_exif_datetime(str(exif_data['DateTimeOriginal']))
        
        if 'DateTimeDigitized' in exif_data:
            metadata.digitized_time = self._parse_exif_datetime(str(exif_data['DateTimeDigitized']))
        
        # Software information
        if 'Software' in exif_data:
            metadata.software_used = str(exif_data['Software']).strip()
        
        # GPS data
        if 'GPSInfo' in exif_data:
            gps_info = exif_data['GPSInfo']
            if isinstance(gps_info, dict):
                # Extract coordinates (simplified)
                if 'GPSLatitude' in gps_info and 'GPSLatitudeRef' in gps_info:
                    metadata.gps_latitude = self._parse_gps_coordinate(
                        gps_info['GPSLatitude'], gps_info['GPSLatitudeRef']
                    )
                
                if 'GPSLongitude' in gps_info and 'GPSLongitudeRef' in gps_info:
                    metadata.gps_longitude = self._parse_gps_coordinate(
                        gps_info['GPSLongitude'], gps_info['GPSLongitudeRef']
                    )
    
    def _parse_exif_datetime(self, datetime_str: str) -> Optional[datetime]:
        """Parse EXIF datetime string"""
        try:
            # EXIF datetime format: "YYYY:MM:DD HH:MM:SS"
            return datetime.strptime(datetime_str, "%Y:%m:%d %H:%M:%S")
        except ValueError:
            try:
                # Alternative format
                return datetime.strptime(datetime_str, "%Y-%m-%d %H:%M:%S")
            except ValueError:
                logger.debug(f"Failed to parse datetime: {datetime_str}")
                return None
    
    def _parse_gps_coordinate(self, coord_data: Any, ref: str) -> Optional[float]:
        """Parse GPS coordinate from EXIF format"""
        try:
            if isinstance(coord_data, (list, tuple)) and len(coord_data) == 3:
                degrees = float(coord_data[0])
                minutes = float(coord_data[1])
                seconds = float(coord_data[2])
                
                decimal = degrees + (minutes / 60.0) + (seconds / 3600.0)
                
                if ref in ['S', 'W']:
                    decimal = -decimal
                
                return decimal
        except Exception as e:
            logger.debug(f"Failed to parse GPS coordinate: {str(e)}")
        
        return None
    
    def _process_additional_exif(self, metadata: ImageMetadata, tags: dict):
        """Process additional EXIF tags using exifread"""
        
        # Look for hidden or manufacturer-specific tags
        hidden_data = {}
        
        for tag_name, tag_value in tags.items():
            try:
                tag_str = str(tag_name)
                value_str = str(tag_value)
                
                # Look for manufacturer-specific tags
                if any(maker in tag_str.upper() for maker in ['CANON', 'NIKON', 'SONY', 'FUJI']):
                    hidden_data[tag_str] = value_str
                
                # Look for software tags
                if 'SOFTWARE' in tag_str.upper() or 'PROCESSING' in tag_str.upper():
                    if value_str not in metadata.editing_software:
                        metadata.editing_software.append(value_str)
                
            except Exception as e:
                logger.debug(f"Failed to process additional tag {tag_name}: {str(e)}")
        
        if hidden_data:
            metadata.hidden_metadata = hidden_data
    
    def _calculate_perceptual_hash(self, image_path: Path) -> Optional[str]:
        """Calculate perceptual hash of the image"""
        try:
            import imagehash
            with Image.open(image_path) as img:
                phash = imagehash.phash(img)
                return str(phash)
        except ImportError:
            logger.debug("imagehash library not available")
        except Exception as e:
            logger.debug(f"Perceptual hash calculation failed: {str(e)}")
        
        return None
    
    def _analyze_timestamp_consistency(self, metadata: ImageMetadata, forensics: MetadataForensics):
        """Analyze timestamp consistency and detect anomalies"""
        
        timestamps = []
        
        if metadata.creation_time:
            timestamps.append(('creation', metadata.creation_time))
        if metadata.digitized_time:
            timestamps.append(('digitized', metadata.digitized_time))
        if metadata.modification_time:
            timestamps.append(('modification', metadata.modification_time))
        if metadata.gps_timestamp:
            timestamps.append(('gps', metadata.gps_timestamp))
        
        # Check for inconsistencies
        inconsistencies = []
        
        # Check if creation time is after modification time
        if metadata.creation_time and metadata.modification_time:
            if metadata.creation_time > metadata.modification_time:
                inconsistencies.append("Creation time is after file modification time")
        
        # Check if digitized time is significantly different from creation time
        if metadata.creation_time and metadata.digitized_time:
            time_diff = abs((metadata.creation_time - metadata.digitized_time).total_seconds())
            if time_diff > 86400:  # More than 1 day difference
                inconsistencies.append("Large difference between creation and digitized times")
        
        # Check for impossible dates
        current_time = datetime.now()
        for timestamp_type, timestamp in timestamps:
            if timestamp > current_time:
                inconsistencies.append(f"{timestamp_type} timestamp is in the future")
            
            # Check for dates before digital photography era
            if timestamp.year < 1990:
                inconsistencies.append(f"{timestamp_type} timestamp predates digital photography")
        
        # Check GPS timestamp consistency
        if metadata.gps_timestamp and metadata.creation_time:
            gps_diff = abs((metadata.gps_timestamp - metadata.creation_time).total_seconds())
            if gps_diff > 3600:  # More than 1 hour difference
                inconsistencies.append("GPS timestamp differs significantly from creation time")
        
        forensics.timestamp_inconsistencies = inconsistencies
    
    def _analyze_camera_consistency(self, metadata: ImageMetadata, forensics: MetadataForensics):
        """Analyze camera metadata consistency"""
        
        inconsistencies = []
        
        # Check camera make/model combination
        if metadata.camera_make and metadata.camera_model:
            if not self._validate_camera_combination(metadata.camera_make, metadata.camera_model):
                inconsistencies.append("Invalid camera make/model combination")
        
        # Check camera settings consistency
        if metadata.iso_speed and metadata.iso_speed > 0:
            # Check for impossible ISO values
            if metadata.iso_speed > 3200000:  # Extremely high ISO
                inconsistencies.append("Unusually high ISO value")
            elif metadata.iso_speed < 25:  # Very low ISO
                inconsistencies.append("Unusually low ISO value")
        
        # Check focal length consistency with camera model
        if metadata.focal_length and metadata.camera_model:
            if not self._validate_focal_length(metadata.camera_model, metadata.focal_length):
                inconsistencies.append("Focal length inconsistent with camera model")
        
        forensics.camera_inconsistencies = inconsistencies
    
    def _analyze_software_consistency(self, metadata: ImageMetadata, forensics: MetadataForensics):
        """Analyze software metadata consistency"""
        
        inconsistencies = []
        editing_signatures = []
        
        # Check software field
        if metadata.software_used:
            software = metadata.software_used.lower()
            
            # Detect editing software
            editing_software = [
                'photoshop', 'gimp', 'paint.net', 'lightroom', 'capture one',
                'affinity', 'canva', 'pixlr', 'snapseed', 'vsco'
            ]
            
            for editor in editing_software:
                if editor in software:
                    editing_signatures.append(f"Editing software detected: {editor}")
                    forensics.editing_history.append(editor)
            
            # Check for inconsistent software versions
            if 'photoshop' in software:
                version_match = re.search(r'(\d+\.\d+)', software)
                if version_match:
                    version = float(version_match.group(1))
                    if version > 25.0:  # Future version check
                        inconsistencies.append("Software version appears to be from the future")
        
        # Check for multiple editing software signatures
        if len(editing_signatures) > 1:
            inconsistencies.append("Multiple editing software signatures detected")
        
        forensics.software_inconsistencies = inconsistencies
        forensics.software_signatures = editing_signatures
    
    def _analyze_gps_consistency(self, metadata: ImageMetadata, forensics: MetadataForensics):
        """Analyze GPS metadata consistency"""
        
        inconsistencies = []
        
        if metadata.gps_latitude and metadata.gps_longitude:
            lat, lon = metadata.gps_latitude, metadata.gps_longitude
            
            # Check for invalid coordinates
            if not (-90 <= lat <= 90):
                inconsistencies.append("Invalid GPS latitude")
            
            if not (-180 <= lon <= 180):
                inconsistencies.append("Invalid GPS longitude")
            
            # Check for suspicious precision
            lat_str = str(lat)
            lon_str = str(lon)
            
            if '.' in lat_str and len(lat_str.split('.')[1]) > 8:
                inconsistencies.append("GPS latitude has suspicious precision")
            
            if '.' in lon_str and len(lon_str.split('.')[1]) > 8:
                inconsistencies.append("GPS longitude has suspicious precision")
            
            # Check for common fake coordinates
            if lat == 0 and lon == 0:
                inconsistencies.append("GPS coordinates at null island (0,0)")
            
            # Check for rounded coordinates (suspicious)
            if lat == round(lat) and lon == round(lon):
                inconsistencies.append("GPS coordinates are suspiciously rounded")
        
        forensics.gps_inconsistencies = inconsistencies
    
    def _analyze_exif_integrity(self, image_path: Path, metadata: ImageMetadata, forensics: MetadataForensics):
        """Analyze EXIF data integrity"""
        
        try:
            with open(image_path, 'rb') as f:
                # Read file header
                header = f.read(10)
                
                # Check JPEG structure
                if metadata.file_format == ImageFormat.JPEG:
                    if not header.startswith(b'\xff\xd8'):
                        forensics.exif_integrity = False
                        return
                    
                    # Look for EXIF marker
                    f.seek(0)
                    content = f.read(65536)  # Read first 64KB
                    
                    if b'Exif\x00\x00' not in content:
                        forensics.metadata_stripped = True
                    
                    # Check for multiple EXIF segments (suspicious)
                    exif_count = content.count(b'Exif\x00\x00')
                    if exif_count > 1:
                        forensics.metadata_modified = True
            
            # Check for EXIF data consistency with file size
            exif_size = len(json.dumps(metadata.exif_data).encode())
            file_size = metadata.file_size_bytes
            
            # EXIF should not be more than 10% of file size
            if exif_size > file_size * 0.1:
                forensics.metadata_modified = True
        
        except Exception as e:
            logger.error(f"EXIF integrity analysis failed: {str(e)}")
            forensics.exif_integrity = False
    
    def _analyze_thumbnail_consistency(self, image_path: Path, metadata: ImageMetadata, forensics: MetadataForensics):
        """Analyze embedded thumbnail consistency"""
        
        try:
            with Image.open(image_path) as img:
                # Check if thumbnail exists
                if hasattr(img, '_getexif') and img._getexif():
                    exif = img._getexif()
                    
                    # Look for thumbnail data
                    thumbnail_data = None
                    for tag, value in exif.items():
                        if TAGS.get(tag) == 'JPEGThumbnail':
                            thumbnail_data = value
                            break
                    
                    if thumbnail_data:
                        # Verify thumbnail matches main image
                        # This is a simplified check
                        try:
                            import io
                            thumbnail = Image.open(io.BytesIO(thumbnail_data))
                            
                            # Compare aspect ratios
                            main_ratio = img.width / img.height
                            thumb_ratio = thumbnail.width / thumbnail.height
                            
                            if abs(main_ratio - thumb_ratio) > 0.1:
                                forensics.thumbnail_consistency = False
                        
                        except Exception as e:
                            logger.debug(f"Thumbnail analysis failed: {str(e)}")
                            forensics.thumbnail_consistency = False
        
        except Exception as e:
            logger.debug(f"Thumbnail consistency analysis failed: {str(e)}")
    
    def _detect_metadata_manipulation(self, metadata: ImageMetadata, forensics: MetadataForensics):
        """Detect signs of metadata manipulation"""
        
        # Check for missing typical metadata
        typical_fields = ['camera_make', 'camera_model', 'creation_time']
        missing_fields = [field for field in typical_fields if not getattr(metadata, field)]
        
        if len(missing_fields) == len(typical_fields):
            forensics.metadata_stripped = True
        
        # Check for fabricated metadata patterns
        if metadata.camera_make and metadata.camera_model:
            # Look for generic or placeholder values
            generic_patterns = ['unknown', 'camera', 'device', 'digital']
            
            make_lower = metadata.camera_make.lower()
            model_lower = metadata.camera_model.lower()
            
            if any(pattern in make_lower for pattern in generic_patterns):
                forensics.metadata_fabricated = True
            
            if any(pattern in model_lower for pattern in generic_patterns):
                forensics.metadata_fabricated = True
        
        # Check for impossible combinations
        if metadata.iso_speed and metadata.aperture and metadata.shutter_speed:
            # This would require complex camera physics validation
            # For now, check for obvious impossibilities
            if metadata.iso_speed > 1000000:  # Impossibly high ISO
                forensics.metadata_fabricated = True
    
    def _validate_camera_fingerprint(self, metadata: ImageMetadata, forensics: MetadataForensics):
        """Validate camera fingerprint against known database"""
        
        if not metadata.camera_make or not metadata.camera_model:
            return
        
        camera_key = f"{metadata.camera_make}_{metadata.camera_model}".lower()
        
        # Check against known camera database
        if camera_key in self.camera_database:
            camera_info = self.camera_database[camera_key]
            forensics.camera_database_match = True
            
            # Validate specifications
            if metadata.iso_speed:
                iso_range = camera_info.get('iso_range', [50, 25600])
                if not (iso_range[0] <= metadata.iso_speed <= iso_range[1]):
                    forensics.camera_inconsistencies.append("ISO outside camera's range")
            
            # Check release date consistency
            if metadata.creation_time and 'release_year' in camera_info:
                release_year = camera_info['release_year']
                if metadata.creation_time.year < release_year:
                    forensics.camera_inconsistencies.append("Photo predates camera release")
        
        # Validate lens characteristics if available
        if metadata.lens_model:
            forensics.lens_characteristics_match = self._validate_lens_characteristics(
                metadata.camera_make, metadata.lens_model, metadata.focal_length
            )
    
    def _calculate_authenticity_assessment(self, forensics: MetadataForensics):
        """Calculate overall metadata authenticity assessment"""
        
        # Start with base confidence
        confidence = 0.8
        
        # Reduce confidence for each inconsistency type
        inconsistency_weights = {
            'timestamp_inconsistencies': 0.1,
            'camera_inconsistencies': 0.15,
            'software_inconsistencies': 0.1,
            'gps_inconsistencies': 0.1
        }
        
        for inconsistency_type, weight in inconsistency_weights.items():
            inconsistencies = getattr(forensics, inconsistency_type, [])
            confidence -= len(inconsistencies) * weight
        
        # Major integrity issues
        if not forensics.exif_integrity:
            confidence -= 0.3
        
        if forensics.metadata_stripped:
            confidence -= 0.2
        
        if forensics.metadata_modified:
            confidence -= 0.4
        
        if forensics.metadata_fabricated:
            confidence -= 0.5
        
        if not forensics.thumbnail_consistency:
            confidence -= 0.1
        
        # Positive indicators
        if forensics.camera_database_match:
            confidence += 0.1
        
        if forensics.camera_fingerprint_valid:
            confidence += 0.1
        
        # Set final values
        forensics.metadata_confidence = max(0.0, min(1.0, confidence))
        forensics.metadata_authentic = (
            forensics.metadata_confidence > 0.6 and
            not forensics.metadata_fabricated and
            forensics.exif_integrity
        )
        
        # Determine if deeper analysis is needed
        forensics.requires_deeper_analysis = (
            forensics.metadata_confidence < 0.5 or
            len(forensics.timestamp_inconsistencies) > 2 or
            len(forensics.camera_inconsistencies) > 1
        )
        
        # Determine if expert validation is needed
        forensics.expert_validation_needed = (
            not forensics.metadata_authentic or
            forensics.metadata_fabricated or
            not forensics.exif_integrity
        )
        
        # Check for chain of custody concerns
        if forensics.metadata_modified or forensics.metadata_stripped:
            forensics.chain_of_custody_concerns.append("Metadata has been altered")
        
        if forensics.metadata_fabricated:
            forensics.chain_of_custody_concerns.append("Metadata may be fabricated")
    
    def _load_camera_database(self) -> Dict[str, Dict[str, Any]]:
        """Load known camera database (simplified for demonstration)"""
        return {
            'canon_eos 5d mark iv': {
                'release_year': 2016,
                'iso_range': [100, 32000],
                'max_resolution': (6720, 4480)
            },
            'nikon_d850': {
                'release_year': 2017,
                'iso_range': [64, 25600],
                'max_resolution': (8256, 5504)
            },
            'sony_a7r iv': {
                'release_year': 2019,
                'iso_range': [100, 32000],
                'max_resolution': (9504, 6336)
            }
        }
    
    def _load_software_signatures(self) -> Dict[str, List[str]]:
        """Load known software signatures"""
        return {
            'photoshop': ['Adobe Photoshop', 'Photoshop'],
            'lightroom': ['Adobe Lightroom', 'Lightroom'],
            'gimp': ['GIMP'],
            'canon_dpp': ['Canon Digital Photo Professional'],
            'nikon_capture': ['Nikon Capture']
        }
    
    def _validate_camera_combination(self, make: str, model: str) -> bool:
        """Validate if camera make/model combination is realistic"""
        make_lower = make.lower()
        model_lower = model.lower()
        
        # Check for obvious mismatches
        canon_keywords = ['canon', 'eos']
        nikon_keywords = ['nikon', 'd']
        sony_keywords = ['sony', 'alpha', 'a7']
        
        if 'canon' in make_lower:
            return any(keyword in model_lower for keyword in canon_keywords)
        elif 'nikon' in make_lower:
            return any(keyword in model_lower for keyword in nikon_keywords)
        elif 'sony' in make_lower:
            return any(keyword in model_lower for keyword in sony_keywords)
        
        return True  # Allow unknown combinations
    
    def _validate_focal_length(self, camera_model: str, focal_length: str) -> bool:
        """Validate focal length against camera capabilities"""
        try:
            # Extract numeric focal length
            focal_match = re.search(r'(\d+(?:\.\d+)?)', focal_length)
            if not focal_match:
                return True  # Can't validate
            
            focal_mm = float(focal_match.group(1))
            
            # Basic sanity check (most cameras: 10-800mm)
            return 10 <= focal_mm <= 800
            
        except Exception:
            return True  # Can't validate, assume valid
    
    def _validate_lens_characteristics(self, camera_make: str, lens_model: str, focal_length: str) -> bool:
        """Validate lens characteristics"""
        # Simplified validation
        if not lens_model or not camera_make:
            return False
        
        # Check if lens brand matches camera brand (simplified)
        make_lower = camera_make.lower()
        lens_lower = lens_model.lower()
        
        if 'canon' in make_lower and 'canon' in lens_lower:
            return True
        elif 'nikon' in make_lower and 'nikon' in lens_lower:
            return True
        elif 'sony' in make_lower and 'sony' in lens_lower:
            return True
        
        # Third-party lenses are common
        third_party = ['sigma', 'tamron', 'tokina', 'zeiss']
        if any(brand in lens_lower for brand in third_party):
            return True
        
        return False


def analyze_image_metadata(image_path: Path, config: Optional[ImageAuthConfig] = None) -> MetadataForensics:
    """
    Convenience function to analyze image metadata
    
    Args:
        image_path: Path to the image file
        config: Optional configuration
        
    Returns:
        MetadataForensics with comprehensive analysis
    """
    analyzer = MetadataForensicsAnalyzer(config)
    return analyzer.analyze_metadata(image_path)