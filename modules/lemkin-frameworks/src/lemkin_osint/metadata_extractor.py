"""
Lemkin OSINT Collection Toolkit - Metadata Extractor

EXIF/XMP extraction from images and videos for digital forensics and evidence preservation.
Implements comprehensive metadata extraction while maintaining chain of custody.

Compliance: Berkeley Protocol for Digital Investigations
"""

import hashlib
import mimetypes
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple
import logging
import json
import struct

from PIL import Image, ExifTags
from PIL.ExifTags import TAGS, GPSTAGS

from .core import MediaMetadata, ContentType, OSINTConfig

logger = logging.getLogger(__name__)


class MetadataExtractionError(Exception):
    """Raised when metadata extraction fails"""
    pass


class MetadataExtractor:
    """
    Comprehensive metadata extractor for images, videos, and documents.
    Extracts EXIF, XMP, IPTC, and other metadata while preserving
    chain of custody for digital evidence.
    """
    
    def __init__(self, config: OSINTConfig):
        self.config = config
        
        # Supported file types
        self.image_extensions = {'.jpg', '.jpeg', '.png', '.tiff', '.tif', '.bmp', '.gif', '.webp'}
        self.video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm'}
        self.audio_extensions = {'.mp3', '.wav', '.flac', '.aac', '.ogg', '.m4a'}
        self.document_extensions = {'.pdf', '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx'}
        
        logger.info("Metadata extractor initialized")
    
    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA-256 hash of file"""
        hash_sha256 = hashlib.sha256()
        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_sha256.update(chunk)
            return hash_sha256.hexdigest()
        except Exception as e:
            logger.error(f"Error calculating file hash: {e}")
            raise MetadataExtractionError(f"Failed to calculate hash: {e}")
    
    def _determine_content_type(self, file_path: Path) -> ContentType:
        """Determine content type from file extension and MIME type"""
        extension = file_path.suffix.lower()
        
        if extension in self.image_extensions:
            return ContentType.IMAGE
        elif extension in self.video_extensions:
            return ContentType.VIDEO
        elif extension in self.audio_extensions:
            return ContentType.AUDIO
        elif extension in self.document_extensions:
            return ContentType.DOCUMENT
        else:
            # Try MIME type detection
            mime_type, _ = mimetypes.guess_type(str(file_path))
            if mime_type:
                if mime_type.startswith('image/'):
                    return ContentType.IMAGE
                elif mime_type.startswith('video/'):
                    return ContentType.VIDEO
                elif mime_type.startswith('audio/'):
                    return ContentType.AUDIO
                elif mime_type.startswith('application/'):
                    return ContentType.DOCUMENT
            
            return ContentType.TEXT  # Default fallback
    
    def extract_media_metadata(self, file_path: Union[str, Path]) -> MediaMetadata:
        """
        Extract comprehensive metadata from media file
        
        Args:
            file_path: Path to the media file
            
        Returns:
            MediaMetadata object containing extracted metadata
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise MetadataExtractionError(f"File not found: {file_path}")
        
        try:
            # Basic file information
            stat_info = file_path.stat()
            file_hash = self._calculate_file_hash(file_path)
            content_type = self._determine_content_type(file_path)
            
            # Initialize metadata object
            metadata = MediaMetadata(
                file_path=str(file_path.absolute()),
                file_hash=file_hash,
                file_size=stat_info.st_size,
                content_type=content_type,
                creation_date=datetime.fromtimestamp(stat_info.st_ctime),
                modification_date=datetime.fromtimestamp(stat_info.st_mtime)
            )
            
            # Extract type-specific metadata
            if content_type == ContentType.IMAGE:
                self._extract_image_metadata(file_path, metadata)
            elif content_type == ContentType.VIDEO:
                self._extract_video_metadata(file_path, metadata)
            elif content_type == ContentType.AUDIO:
                self._extract_audio_metadata(file_path, metadata)
            elif content_type == ContentType.DOCUMENT:
                self._extract_document_metadata(file_path, metadata)
            
            logger.info(f"Successfully extracted metadata from {file_path}")
            return metadata
            
        except Exception as e:
            logger.error(f"Error extracting metadata from {file_path}: {e}")
            raise MetadataExtractionError(f"Metadata extraction failed: {e}")
    
    def _extract_image_metadata(self, file_path: Path, metadata: MediaMetadata):
        """Extract EXIF and other metadata from image files"""
        try:
            with Image.open(file_path) as img:
                # Basic image properties
                metadata.width = img.width
                metadata.height = img.height
                
                # Color space information
                metadata.color_space = img.mode
                
                # DPI information
                if hasattr(img, 'info') and 'dpi' in img.info:
                    metadata.dpi = img.info['dpi'][0]  # Take first DPI value
                
                # Extract EXIF data
                if hasattr(img, '_getexif') and img._getexif():
                    exif_data = img._getexif()
                    metadata.raw_metadata['exif'] = {}
                    
                    for tag_id, value in exif_data.items():
                        tag = TAGS.get(tag_id, tag_id)
                        
                        # Handle special EXIF tags
                        if tag == 'Make':
                            metadata.camera_make = str(value)
                        elif tag == 'Model':
                            metadata.camera_model = str(value)
                        elif tag == 'DateTime':
                            try:
                                metadata.creation_date = datetime.strptime(
                                    str(value), '%Y:%m:%d %H:%M:%S'
                                )
                            except ValueError:
                                pass
                        elif tag == 'GPSInfo':
                            metadata.gps_coordinates = self._extract_gps_coordinates(value)
                        
                        # Store all EXIF data
                        metadata.raw_metadata['exif'][tag] = self._serialize_exif_value(value)
                
                # Extract XMP data if available
                if hasattr(img, 'info') and 'xmp' in img.info:
                    metadata.raw_metadata['xmp'] = img.info['xmp']
                
                # Extract ICC profile if available
                if hasattr(img, 'info') and 'icc_profile' in img.info:
                    metadata.raw_metadata['icc_profile_size'] = len(img.info['icc_profile'])
                
        except Exception as e:
            logger.warning(f"Error extracting image metadata: {e}")
            metadata.raw_metadata['extraction_error'] = str(e)
    
    def _extract_video_metadata(self, file_path: Path, metadata: MediaMetadata):
        """Extract metadata from video files"""
        try:
            # Note: This is a basic implementation. For production use,
            # consider using ffmpeg-python or similar libraries
            
            # Try to use ffprobe if available
            import subprocess
            import json
            
            try:
                # Run ffprobe to get video metadata
                cmd = [
                    'ffprobe', '-v', 'quiet', '-print_format', 'json',
                    '-show_format', '-show_streams', str(file_path)
                ]
                
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
                
                if result.returncode == 0:
                    video_info = json.loads(result.stdout)
                    
                    # Extract format information
                    format_info = video_info.get('format', {})
                    metadata.duration_seconds = float(format_info.get('duration', 0))
                    
                    # Extract video stream information
                    for stream in video_info.get('streams', []):
                        if stream.get('codec_type') == 'video':
                            metadata.width = stream.get('width')
                            metadata.height = stream.get('height')
                            metadata.codec = stream.get('codec_name')
                            
                            # Extract frame rate
                            r_frame_rate = stream.get('r_frame_rate', '0/1')
                            if '/' in r_frame_rate:
                                num, den = map(int, r_frame_rate.split('/'))
                                if den != 0:
                                    metadata.frame_rate = num / den
                            
                            break
                    
                    # Store raw metadata
                    metadata.raw_metadata['ffprobe'] = video_info
                
            except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError):
                logger.warning("ffprobe not available or failed, using basic extraction")
                
                # Basic file-based metadata extraction
                self._extract_basic_video_metadata(file_path, metadata)
                
        except Exception as e:
            logger.warning(f"Error extracting video metadata: {e}")
            metadata.raw_metadata['extraction_error'] = str(e)
    
    def _extract_basic_video_metadata(self, file_path: Path, metadata: MediaMetadata):
        """Extract basic video metadata without external tools"""
        try:
            # This is a very basic implementation
            # For production, use proper video analysis libraries
            
            with open(file_path, 'rb') as f:
                # Read first few bytes to identify format
                header = f.read(32)
                
                if header[4:8] == b'ftyp':  # MP4/MOV format
                    metadata.codec = 'mp4'
                    metadata.raw_metadata['container'] = 'mp4'
                elif header[:4] == b'RIFF' and header[8:12] == b'AVI ':  # AVI format
                    metadata.codec = 'avi'
                    metadata.raw_metadata['container'] = 'avi'
                elif header[:3] == b'\x1A\x45\xDF':  # MKV format
                    metadata.codec = 'mkv'
                    metadata.raw_metadata['container'] = 'mkv'
                
        except Exception as e:
            logger.warning(f"Error in basic video metadata extraction: {e}")
    
    def _extract_audio_metadata(self, file_path: Path, metadata: MediaMetadata):
        """Extract metadata from audio files"""
        try:
            # Try to use mutagen library if available for comprehensive audio metadata
            try:
                from mutagen import File
                
                audio_file = File(file_path)
                if audio_file:
                    # Duration
                    if hasattr(audio_file, 'info') and hasattr(audio_file.info, 'length'):
                        metadata.duration_seconds = audio_file.info.length
                    
                    # Codec information
                    if hasattr(audio_file, 'mime'):
                        metadata.codec = audio_file.mime[0].split('/')[-1]
                    
                    # Extract tags
                    audio_tags = {}
                    if audio_file.tags:
                        for key, value in audio_file.tags.items():
                            audio_tags[key] = str(value[0]) if isinstance(value, list) else str(value)
                    
                    metadata.raw_metadata['audio_tags'] = audio_tags
                
            except ImportError:
                logger.warning("mutagen library not available, using basic audio extraction")
                self._extract_basic_audio_metadata(file_path, metadata)
                
        except Exception as e:
            logger.warning(f"Error extracting audio metadata: {e}")
            metadata.raw_metadata['extraction_error'] = str(e)
    
    def _extract_basic_audio_metadata(self, file_path: Path, metadata: MediaMetadata):
        """Extract basic audio metadata without external libraries"""
        try:
            extension = file_path.suffix.lower()
            
            # Set codec based on extension
            if extension == '.mp3':
                metadata.codec = 'mp3'
                # Try to read ID3 tags from MP3
                self._extract_mp3_id3_tags(file_path, metadata)
            elif extension == '.wav':
                metadata.codec = 'wav'
            elif extension == '.flac':
                metadata.codec = 'flac'
            else:
                metadata.codec = extension[1:] if extension else 'unknown'
                
        except Exception as e:
            logger.warning(f"Error in basic audio metadata extraction: {e}")
    
    def _extract_mp3_id3_tags(self, file_path: Path, metadata: MediaMetadata):
        """Extract ID3 tags from MP3 files"""
        try:
            with open(file_path, 'rb') as f:
                # Check for ID3v2 header
                header = f.read(10)
                if header[:3] == b'ID3':
                    # ID3v2 tag present
                    version = header[3:5]
                    size_bytes = header[6:10]
                    
                    # Calculate tag size (synchsafe integer)
                    tag_size = (
                        (size_bytes[0] & 0x7f) << 21 |
                        (size_bytes[1] & 0x7f) << 14 |
                        (size_bytes[2] & 0x7f) << 7 |
                        (size_bytes[3] & 0x7f)
                    )
                    
                    metadata.raw_metadata['id3v2'] = {
                        'version': f"{version[0]}.{version[1]}",
                        'size': tag_size
                    }
                
        except Exception as e:
            logger.warning(f"Error extracting MP3 ID3 tags: {e}")
    
    def _extract_document_metadata(self, file_path: Path, metadata: MediaMetadata):
        """Extract metadata from document files"""
        try:
            extension = file_path.suffix.lower()
            
            if extension == '.pdf':
                self._extract_pdf_metadata(file_path, metadata)
            elif extension in {'.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx'}:
                self._extract_office_metadata(file_path, metadata)
            else:
                logger.warning(f"Document metadata extraction not implemented for {extension}")
                
        except Exception as e:
            logger.warning(f"Error extracting document metadata: {e}")
            metadata.raw_metadata['extraction_error'] = str(e)
    
    def _extract_pdf_metadata(self, file_path: Path, metadata: MediaMetadata):
        """Extract metadata from PDF files"""
        try:
            # Try to use PyPDF2 if available
            try:
                import PyPDF2
                
                with open(file_path, 'rb') as f:
                    pdf_reader = PyPDF2.PdfReader(f)
                    
                    # Extract document info
                    if pdf_reader.metadata:
                        pdf_metadata = {}
                        for key, value in pdf_reader.metadata.items():
                            pdf_metadata[key] = str(value)
                        
                        metadata.raw_metadata['pdf_metadata'] = pdf_metadata
                        
                        # Extract creation date if available
                        if '/CreationDate' in pdf_reader.metadata:
                            try:
                                # PDF date format: D:YYYYMMDDHHmmSSOHH'mm
                                date_str = str(pdf_reader.metadata['/CreationDate'])
                                if date_str.startswith('D:'):
                                    date_part = date_str[2:16]  # YYYYMMDDHHMMSS
                                    metadata.creation_date = datetime.strptime(
                                        date_part, '%Y%m%d%H%M%S'
                                    )
                            except (ValueError, IndexError):
                                pass
                    
                    # Number of pages
                    metadata.raw_metadata['pdf_pages'] = len(pdf_reader.pages)
                
            except ImportError:
                logger.warning("PyPDF2 not available for PDF metadata extraction")
                
        except Exception as e:
            logger.warning(f"Error extracting PDF metadata: {e}")
    
    def _extract_office_metadata(self, file_path: Path, metadata: MediaMetadata):
        """Extract metadata from Microsoft Office documents"""
        try:
            # Try to use python-docx, openpyxl, or similar libraries
            # This is a placeholder for Office document metadata extraction
            
            extension = file_path.suffix.lower()
            
            if extension == '.docx':
                try:
                    from docx import Document
                    doc = Document(file_path)
                    
                    # Extract core properties
                    core_props = doc.core_properties
                    office_metadata = {
                        'title': core_props.title,
                        'author': core_props.author,
                        'subject': core_props.subject,
                        'created': core_props.created.isoformat() if core_props.created else None,
                        'modified': core_props.modified.isoformat() if core_props.modified else None,
                        'last_modified_by': core_props.last_modified_by
                    }
                    
                    metadata.raw_metadata['office_metadata'] = office_metadata
                    
                    if core_props.created:
                        metadata.creation_date = core_props.created
                        
                except ImportError:
                    logger.warning("python-docx not available for DOCX metadata extraction")
                    
            else:
                logger.warning(f"Office metadata extraction not implemented for {extension}")
                
        except Exception as e:
            logger.warning(f"Error extracting Office metadata: {e}")
    
    def _extract_gps_coordinates(self, gps_info: Dict) -> Optional[Dict[str, float]]:
        """Extract GPS coordinates from EXIF GPS info"""
        try:
            if not gps_info:
                return None
            
            def convert_to_degrees(value):
                """Convert GPS coordinate to decimal degrees"""
                d, m, s = value
                return d + (m / 60.0) + (s / 3600.0)
            
            gps_data = {}
            
            for tag_id in gps_info:
                tag = GPSTAGS.get(tag_id, tag_id)
                gps_data[tag] = gps_info[tag_id]
            
            # Extract latitude
            if 'GPSLatitude' in gps_data and 'GPSLatitudeRef' in gps_data:
                lat = convert_to_degrees(gps_data['GPSLatitude'])
                if gps_data['GPSLatitudeRef'] == 'S':
                    lat = -lat
                
                # Extract longitude
                if 'GPSLongitude' in gps_data and 'GPSLongitudeRef' in gps_data:
                    lon = convert_to_degrees(gps_data['GPSLongitude'])
                    if gps_data['GPSLongitudeRef'] == 'W':
                        lon = -lon
                    
                    coordinates = {
                        'latitude': lat,
                        'longitude': lon
                    }
                    
                    # Add altitude if available
                    if 'GPSAltitude' in gps_data:
                        altitude = float(gps_data['GPSAltitude'])
                        if 'GPSAltitudeRef' in gps_data and gps_data['GPSAltitudeRef'] == 1:
                            altitude = -altitude  # Below sea level
                        coordinates['altitude'] = altitude
                    
                    return coordinates
            
        except Exception as e:
            logger.warning(f"Error extracting GPS coordinates: {e}")
        
        return None
    
    def _serialize_exif_value(self, value: Any) -> Any:
        """Serialize EXIF values for JSON storage"""
        try:
            if isinstance(value, bytes):
                return value.decode('utf-8', errors='ignore')
            elif isinstance(value, tuple):
                return [self._serialize_exif_value(v) for v in value]
            elif hasattr(value, '__iter__') and not isinstance(value, str):
                return [self._serialize_exif_value(v) for v in value]
            else:
                return str(value)
        except Exception:
            return str(value)
    
    def batch_extract_metadata(
        self, 
        file_paths: List[Union[str, Path]],
        output_file: Optional[Path] = None
    ) -> List[MediaMetadata]:
        """
        Extract metadata from multiple files in batch
        
        Args:
            file_paths: List of file paths to process
            output_file: Optional file to save results
            
        Returns:
            List of MediaMetadata objects
        """
        results = []
        
        for file_path in file_paths:
            try:
                metadata = self.extract_media_metadata(file_path)
                results.append(metadata)
                logger.info(f"Processed {file_path}")
            except Exception as e:
                logger.error(f"Failed to process {file_path}: {e}")
        
        # Save results if output file specified
        if output_file and results:
            try:
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(
                        [metadata.dict() for metadata in results],
                        f,
                        indent=2,
                        default=str
                    )
                logger.info(f"Batch results saved to {output_file}")
            except Exception as e:
                logger.error(f"Failed to save batch results: {e}")
        
        return results
    
    def verify_file_integrity(self, file_path: Path, expected_hash: str) -> bool:
        """
        Verify file integrity using hash comparison
        
        Args:
            file_path: Path to file to verify
            expected_hash: Expected SHA-256 hash
            
        Returns:
            bool: True if file integrity is verified
        """
        try:
            actual_hash = self._calculate_file_hash(file_path)
            return actual_hash.lower() == expected_hash.lower()
        except Exception as e:
            logger.error(f"Error verifying file integrity: {e}")
            return False
    
    def create_metadata_report(
        self, 
        metadata_list: List[MediaMetadata],
        output_path: Path
    ) -> bool:
        """
        Create comprehensive metadata report
        
        Args:
            metadata_list: List of metadata to include in report
            output_path: Path to save report
            
        Returns:
            bool: True if report created successfully
        """
        try:
            report = {
                'report_created': datetime.utcnow().isoformat(),
                'tool': 'lemkin-osint-metadata-extractor',
                'version': '1.0',
                'total_files': len(metadata_list),
                'files': []
            }
            
            # Group by content type
            content_type_counts = {}
            
            for metadata in metadata_list:
                # Count content types
                content_type = metadata.content_type
                content_type_counts[content_type] = content_type_counts.get(content_type, 0) + 1
                
                # Add file info to report
                file_info = {
                    'file_path': metadata.file_path,
                    'file_hash': metadata.file_hash,
                    'file_size': metadata.file_size,
                    'content_type': metadata.content_type,
                    'creation_date': metadata.creation_date.isoformat() if metadata.creation_date else None,
                    'modification_date': metadata.modification_date.isoformat() if metadata.modification_date else None,
                    'extracted_at': metadata.extracted_at.isoformat(),
                    'has_gps': bool(metadata.gps_coordinates),
                    'has_camera_info': bool(metadata.camera_make or metadata.camera_model)
                }
                
                # Add type-specific info
                if metadata.content_type == ContentType.IMAGE:
                    file_info['dimensions'] = f"{metadata.width}x{metadata.height}" if metadata.width and metadata.height else None
                elif metadata.content_type == ContentType.VIDEO:
                    file_info['duration'] = metadata.duration_seconds
                    file_info['codec'] = metadata.codec
                
                report['files'].append(file_info)
            
            # Add summary statistics
            report['summary'] = {
                'content_type_distribution': content_type_counts,
                'files_with_gps': sum(1 for m in metadata_list if m.gps_coordinates),
                'files_with_camera_info': sum(1 for m in metadata_list if m.camera_make or m.camera_model)
            }
            
            # Write report
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, default=str)
            
            logger.info(f"Metadata report created: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error creating metadata report: {e}")
            return False