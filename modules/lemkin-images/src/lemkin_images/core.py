"""
Core image verification functionality for legal investigations.
"""

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
from urllib.parse import urlencode

import cv2
import numpy as np
import requests
from PIL import Image, ExifTags
from PIL.ExifTags import TAGS, GPSTAGS
import imagehash
from pydantic import BaseModel, Field, HttpUrl, field_validator
from loguru import logger
from skimage import feature, measure, filters, segmentation
from bs4 import BeautifulSoup


class ManipulationType(str, Enum):
    """Types of image manipulation"""
    SPLICING = "splicing"
    COPY_MOVE = "copy_move"
    ENHANCEMENT = "enhancement"
    UPSCALING = "upscaling"
    DEEPFAKE = "deepfake"
    AIGENERATED = "ai_generated"
    UNKNOWN = "unknown"


class AuthenticityStatus(str, Enum):
    """Image authenticity status"""
    AUTHENTIC = "authentic"
    MANIPULATED = "manipulated"
    SUSPICIOUS = "suspicious"
    AI_GENERATED = "ai_generated"
    UNKNOWN = "unknown"


class SearchEngine(str, Enum):
    """Supported reverse image search engines"""
    GOOGLE = "google"
    BING = "bing"
    YANDEX = "yandex"
    TINEYE = "tineye"


class ImageMetadata(BaseModel):
    """Image metadata for forensic analysis"""
    file_path: Path
    file_size: int
    dimensions: Tuple[int, int]
    format: str
    mode: str
    creation_date: Optional[datetime] = None
    modification_date: Optional[datetime] = None
    camera_make: Optional[str] = None
    camera_model: Optional[str] = None
    gps_coordinates: Optional[Tuple[float, float]] = None
    exif_data: Dict[str, Any] = Field(default_factory=dict)
    file_hash: str = ""

    def model_post_init(self, __context: Any) -> None:
        """Calculate file hash after initialization"""
        if self.file_path.exists():
            self.file_hash = self._calculate_hash()

    def _calculate_hash(self) -> str:
        """Calculate SHA-256 hash of image file"""
        sha256 = hashlib.sha256()
        with open(self.file_path, 'rb') as f:
            while chunk := f.read(8192):
                sha256.update(chunk)
        return sha256.hexdigest()


class ReverseSearchResult(BaseModel):
    """Single reverse search result"""
    url: HttpUrl
    title: Optional[str] = None
    source_domain: str
    thumbnail_url: Optional[HttpUrl] = None
    similarity_score: Optional[float] = None
    first_seen: Optional[datetime] = None


class ReverseSearchResults(BaseModel):
    """Collection of reverse search results"""
    search_id: str
    query_image: Path
    search_engine: SearchEngine
    results: List[ReverseSearchResult]
    total_results: int
    search_date: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    query_hash: str = ""


class ManipulationAnalysis(BaseModel):
    """Image manipulation analysis results"""
    analysis_id: str
    image_path: Path
    is_manipulated: bool
    manipulation_types: List[ManipulationType] = Field(default_factory=list)
    confidence_score: float = Field(ge=0.0, le=1.0)
    manipulation_regions: List[Dict[str, Any]] = Field(default_factory=list)
    analysis_details: Dict[str, Any] = Field(default_factory=dict)
    analysis_date: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class GeolocationResult(BaseModel):
    """Image geolocation analysis result"""
    analysis_id: str
    image_path: Path
    estimated_location: Optional[Tuple[float, float]] = None
    location_confidence: float = Field(ge=0.0, le=1.0, default=0.0)
    visual_features: List[str] = Field(default_factory=list)
    landmark_matches: List[Dict[str, Any]] = Field(default_factory=list)
    metadata_location: Optional[Tuple[float, float]] = None
    analysis_date: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ImageAnalysis(BaseModel):
    """Comprehensive image analysis report"""
    analysis_id: str
    image_path: Path
    authenticity_status: AuthenticityStatus
    confidence_score: float = Field(ge=0.0, le=1.0)
    metadata_analysis: Optional[ImageMetadata] = None
    manipulation_analysis: Optional[ManipulationAnalysis] = None
    reverse_search: Optional[ReverseSearchResults] = None
    geolocation_analysis: Optional[GeolocationResult] = None
    findings: List[str] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)
    analysis_date: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ReverseImageSearcher:
    """Perform reverse image searches across multiple engines"""

    def __init__(self, user_agent: Optional[str] = None):
        """Initialize reverse image searcher"""
        self.user_agent = user_agent or "Lemkin-Images/0.1.0 (Legal Investigation Tool)"
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": self.user_agent})
        logger.info("Initialized reverse image searcher")

    def reverse_search_image(
        self,
        image_path: Path,
        engines: List[SearchEngine] = None,
        limit: int = 20
    ) -> ReverseSearchResults:
        """
        Perform reverse image search across multiple engines.

        Args:
            image_path: Path to image file
            engines: List of search engines to use
            limit: Maximum results per engine

        Returns:
            ReverseSearchResults with findings
        """
        import uuid

        if engines is None:
            engines = [SearchEngine.GOOGLE]

        search_id = str(uuid.uuid4())

        if not image_path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")

        logger.info(f"Starting reverse image search for {image_path}")

        # Calculate image hash for query
        with Image.open(image_path) as img:
            query_hash = str(imagehash.phash(img))

        all_results = []
        total_results = 0

        for engine in engines:
            try:
                if engine == SearchEngine.GOOGLE:
                    results = self._search_google(image_path, limit)
                elif engine == SearchEngine.BING:
                    results = self._search_bing(image_path, limit)
                elif engine == SearchEngine.YANDEX:
                    results = self._search_yandex(image_path, limit)
                elif engine == SearchEngine.TINEYE:
                    results = self._search_tineye(image_path, limit)
                else:
                    logger.warning(f"Unsupported search engine: {engine}")
                    results = []

                all_results.extend(results)
                total_results += len(results)

            except Exception as e:
                logger.error(f"Error searching {engine.value}: {e}")
                continue

        return ReverseSearchResults(
            search_id=search_id,
            query_image=image_path,
            search_engine=engines[0],  # Primary engine
            results=all_results[:limit],
            total_results=total_results,
            query_hash=query_hash
        )

    def _search_google(self, image_path: Path, limit: int) -> List[ReverseSearchResult]:
        """Search Google Images (simulation)"""
        # Note: This would require Google Custom Search API in production
        # For demonstration, we simulate results

        logger.info("Searching Google Images (simulated)")

        results = []

        # Simulate some results
        if limit > 0:
            results.append(ReverseSearchResult(
                url="https://example.com/news/article1",
                title="Sample news article with image",
                source_domain="example.com",
                similarity_score=0.95
            ))

        return results

    def _search_bing(self, image_path: Path, limit: int) -> List[ReverseSearchResult]:
        """Search Bing Images (simulation)"""
        logger.info("Searching Bing Images (simulated)")
        return []

    def _search_yandex(self, image_path: Path, limit: int) -> List[ReverseSearchResult]:
        """Search Yandex Images (simulation)"""
        logger.info("Searching Yandex Images (simulated)")
        return []

    def _search_tineye(self, image_path: Path, limit: int) -> List[ReverseSearchResult]:
        """Search TinEye (simulation)"""
        logger.info("Searching TinEye (simulated)")
        return []


class ManipulationDetector:
    """Detect various types of image manipulation"""

    def __init__(self):
        """Initialize manipulation detector"""
        logger.info("Initialized manipulation detector")

    def detect_image_manipulation(self, image_path: Path) -> ManipulationAnalysis:
        """
        Detect image manipulation using multiple techniques.

        Args:
            image_path: Path to image file

        Returns:
            ManipulationAnalysis with findings
        """
        import uuid

        analysis_id = str(uuid.uuid4())

        if not image_path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")

        logger.info(f"Starting manipulation detection for {image_path}")

        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")

        manipulation_types = []
        manipulation_regions = []
        analysis_details = {}

        # Check for splicing
        if self._detect_splicing(image):
            manipulation_types.append(ManipulationType.SPLICING)
            analysis_details['splicing_detected'] = True

        # Check for copy-move forgery
        copy_move_regions = self._detect_copy_move(image)
        if copy_move_regions:
            manipulation_types.append(ManipulationType.COPY_MOVE)
            manipulation_regions.extend(copy_move_regions)

        # Check for enhancement/filtering
        if self._detect_enhancement(image):
            manipulation_types.append(ManipulationType.ENHANCEMENT)
            analysis_details['enhancement_detected'] = True

        # Check for upscaling artifacts
        if self._detect_upscaling(image):
            manipulation_types.append(ManipulationType.UPSCALING)
            analysis_details['upscaling_detected'] = True

        # Check for AI generation indicators
        if self._detect_ai_generation(image):
            manipulation_types.append(ManipulationType.AIGENERATED)
            analysis_details['ai_generated_detected'] = True

        # Calculate confidence score
        confidence_score = self._calculate_manipulation_confidence(
            manipulation_types, analysis_details
        )

        is_manipulated = len(manipulation_types) > 0 or confidence_score > 0.5

        return ManipulationAnalysis(
            analysis_id=analysis_id,
            image_path=image_path,
            is_manipulated=is_manipulated,
            manipulation_types=manipulation_types,
            confidence_score=confidence_score,
            manipulation_regions=manipulation_regions,
            analysis_details=analysis_details
        )

    def _detect_splicing(self, image: np.ndarray) -> bool:
        """Detect image splicing based on noise analysis"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Divide image into blocks and analyze noise patterns
        h, w = gray.shape
        block_size = min(h // 8, w // 8, 64)  # Adaptive block size

        if block_size < 16:
            return False

        noise_levels = []

        for y in range(0, h - block_size, block_size):
            for x in range(0, w - block_size, block_size):
                block = gray[y:y+block_size, x:x+block_size]

                # Apply high-pass filter to isolate noise
                blur = cv2.GaussianBlur(block, (3, 3), 0)
                noise = cv2.subtract(block, blur)
                noise_level = np.std(noise)
                noise_levels.append(noise_level)

        if len(noise_levels) < 2:
            return False

        # High variance in noise levels suggests splicing
        noise_variance = np.var(noise_levels)
        return noise_variance > 25  # Threshold for splicing detection

    def _detect_copy_move(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Detect copy-move forgery using feature matching"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Extract keypoints and descriptors using ORB
        orb = cv2.ORB_create(nfeatures=5000)
        keypoints, descriptors = orb.detectAndCompute(gray, None)

        if descriptors is None or len(descriptors) < 10:
            return []

        # Match features with themselves to find duplicates
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = matcher.match(descriptors, descriptors)

        # Filter matches by distance and spatial separation
        good_matches = []
        for match in matches:
            # Skip self-matches
            if match.queryIdx == match.trainIdx:
                continue

            # Check spatial separation
            pt1 = keypoints[match.queryIdx].pt
            pt2 = keypoints[match.trainIdx].pt
            distance = np.sqrt((pt1[0] - pt2[0])**2 + (pt1[1] - pt2[1])**2)

            # Must be spatially separated but have similar features
            if distance > 50 and match.distance < 50:
                good_matches.append({
                    'source_point': pt1,
                    'target_point': pt2,
                    'distance': float(distance),
                    'feature_distance': float(match.distance)
                })

        # If we have many similar features in different locations, likely copy-move
        if len(good_matches) > 10:
            return good_matches[:20]  # Return top 20 matches

        return []

    def _detect_enhancement(self, image: np.ndarray) -> bool:
        """Detect image enhancement/filtering artifacts"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Check for over-sharpening artifacts
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        sharpness = np.var(laplacian)

        # Check for artificial saturation
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        saturation = hsv[:, :, 1]
        sat_mean = np.mean(saturation)
        sat_std = np.std(saturation)

        # Over-sharpened images have very high variance
        over_sharpened = sharpness > 2000

        # Over-saturated images have high mean saturation with low variance
        over_saturated = sat_mean > 180 and sat_std < 30

        return over_sharpened or over_saturated

    def _detect_upscaling(self, image: np.ndarray) -> bool:
        """Detect upscaling/interpolation artifacts"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Look for regular patterns that suggest interpolation
        # Calculate second-order derivatives
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

        # Second derivatives
        sobelxx = cv2.Sobel(sobelx, cv2.CV_64F, 1, 0, ksize=3)
        sobelyy = cv2.Sobel(sobely, cv2.CV_64F, 0, 1, ksize=3)

        # Low variance in second derivatives suggests interpolation
        second_deriv_var = np.var(sobelxx) + np.var(sobelyy)

        # Check for regular patterns using FFT
        h, w = gray.shape
        if min(h, w) > 128:
            # Sample a region for FFT analysis
            sample = gray[h//4:3*h//4, w//4:3*w//4]
            fft = np.fft.fft2(sample)
            fft_magnitude = np.abs(fft)

            # Look for peaks that suggest regular patterns
            peaks = feature.peak_local_maxima(fft_magnitude.flatten(), height=np.mean(fft_magnitude) * 2)
            regular_patterns = len(peaks[0]) > 20
        else:
            regular_patterns = False

        return second_deriv_var < 100 or regular_patterns

    def _detect_ai_generation(self, image: np.ndarray) -> bool:
        """Detect AI-generated image indicators"""
        # This is a simplified detection - real implementation would use
        # specialized models trained to detect AI-generated content

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Check for unnatural smoothness patterns
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        high_freq = cv2.subtract(gray, blur)
        high_freq_energy = np.mean(np.abs(high_freq))

        # AI images often have less high-frequency content
        unnatural_smoothness = high_freq_energy < 5

        # Check for repetitive patterns
        # Calculate local binary patterns
        try:
            from skimage.feature import local_binary_pattern
            lbp = local_binary_pattern(gray, 8, 1, method='uniform')
            lbp_hist, _ = np.histogram(lbp.ravel(), bins=10)
            lbp_uniformity = np.max(lbp_hist) / np.sum(lbp_hist)

            # AI images might have more uniform texture patterns
            repetitive_patterns = lbp_uniformity > 0.3
        except ImportError:
            repetitive_patterns = False

        return unnatural_smoothness or repetitive_patterns

    def _calculate_manipulation_confidence(
        self,
        manipulation_types: List[ManipulationType],
        analysis_details: Dict[str, Any]
    ) -> float:
        """Calculate overall confidence in manipulation detection"""
        if not manipulation_types:
            return 0.0

        # Base confidence from number of detected manipulation types
        base_confidence = min(len(manipulation_types) * 0.3, 0.9)

        # Adjust based on specific indicators
        confidence_adjustments = []

        if ManipulationType.SPLICING in manipulation_types:
            confidence_adjustments.append(0.8)

        if ManipulationType.COPY_MOVE in manipulation_types:
            confidence_adjustments.append(0.7)

        if ManipulationType.AIGENERATED in manipulation_types:
            confidence_adjustments.append(0.9)

        # Return weighted average
        if confidence_adjustments:
            return min((base_confidence + np.mean(confidence_adjustments)) / 2, 1.0)
        else:
            return base_confidence


class GeolocationHelper:
    """Help geolocate images from visual content"""

    def __init__(self):
        """Initialize geolocation helper"""
        logger.info("Initialized geolocation helper")

    def geolocate_image(self, image_path: Path) -> GeolocationResult:
        """
        Attempt to geolocate image from visual content.

        Args:
            image_path: Path to image file

        Returns:
            GeolocationResult with findings
        """
        import uuid

        analysis_id = str(uuid.uuid4())

        if not image_path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")

        logger.info(f"Starting geolocation analysis for {image_path}")

        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")

        # Extract visual features
        visual_features = self._extract_visual_features(image)

        # Look for text/signs
        text_clues = self._extract_text_clues(image)
        visual_features.extend(text_clues)

        # Check for architectural features
        architectural_features = self._detect_architectural_features(image)
        visual_features.extend(architectural_features)

        # Get metadata location if available
        metadata_location = self._extract_gps_from_metadata(image_path)

        # Simulate landmark matching (would use computer vision APIs in production)
        landmark_matches = self._match_landmarks(image)

        # Estimate location confidence
        location_confidence = self._calculate_location_confidence(
            visual_features, landmark_matches, metadata_location
        )

        # Estimated location (would use more sophisticated methods in production)
        estimated_location = None
        if metadata_location:
            estimated_location = metadata_location
        elif landmark_matches:
            # Use first landmark match as estimate
            match = landmark_matches[0]
            estimated_location = (match.get('latitude'), match.get('longitude'))

        return GeolocationResult(
            analysis_id=analysis_id,
            image_path=image_path,
            estimated_location=estimated_location,
            location_confidence=location_confidence,
            visual_features=visual_features,
            landmark_matches=landmark_matches,
            metadata_location=metadata_location
        )

    def _extract_visual_features(self, image: np.ndarray) -> List[str]:
        """Extract visual features that might indicate location"""
        features = []

        # Analyze color distribution for climate/environment clues
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Dominant colors
        h_mean = np.mean(hsv[:, :, 0])
        s_mean = np.mean(hsv[:, :, 1])
        v_mean = np.mean(hsv[:, :, 2])

        if v_mean > 180:
            features.append("bright_lighting")
        elif v_mean < 80:
            features.append("low_lighting")

        if s_mean > 150:
            features.append("vibrant_colors")
        elif s_mean < 50:
            features.append("muted_colors")

        # Detect vegetation (green areas)
        green_mask = cv2.inRange(hsv, np.array([35, 50, 50]), np.array([85, 255, 255]))
        green_ratio = np.sum(green_mask > 0) / green_mask.size

        if green_ratio > 0.3:
            features.append("heavy_vegetation")
        elif green_ratio > 0.1:
            features.append("moderate_vegetation")
        else:
            features.append("little_vegetation")

        return features

    def _extract_text_clues(self, image: np.ndarray) -> List[str]:
        """Extract text that might provide location clues"""
        # This would integrate with OCR in a real implementation
        # For now, we simulate text detection

        clues = []

        # In production, would use OCR to extract text
        # and then look for:
        # - Street names
        # - Business names
        # - License plates
        # - Signs with place names

        return clues

    def _detect_architectural_features(self, image: np.ndarray) -> List[str]:
        """Detect architectural features that might indicate region"""
        features = []

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Detect lines (could indicate buildings, bridges, etc.)
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=50, maxLineGap=10)

        if lines is not None and len(lines) > 20:
            features.append("structured_architecture")

            # Analyze line orientations
            angles = []
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
                angles.append(abs(angle))

            # Predominantly vertical/horizontal suggests urban architecture
            if np.sum(np.array(angles) < 10) + np.sum(np.array(angles) > 80) > len(angles) * 0.6:
                features.append("urban_grid_pattern")

        return features

    def _extract_gps_from_metadata(self, image_path: Path) -> Optional[Tuple[float, float]]:
        """Extract GPS coordinates from image metadata"""
        try:
            with Image.open(image_path) as img:
                exif = img._getexif()

                if exif is not None:
                    gps_info = exif.get(ExifTags.TAGS.get('GPSInfo'))

                    if gps_info:
                        lat = self._convert_gps_coord(gps_info.get(2), gps_info.get(1))
                        lon = self._convert_gps_coord(gps_info.get(4), gps_info.get(3))

                        if lat is not None and lon is not None:
                            return (lat, lon)

        except Exception as e:
            logger.debug(f"Could not extract GPS from metadata: {e}")

        return None

    def _convert_gps_coord(self, coord_data, direction):
        """Convert GPS coordinate from EXIF format to decimal degrees"""
        if not coord_data or not direction:
            return None

        try:
            degrees = float(coord_data[0])
            minutes = float(coord_data[1])
            seconds = float(coord_data[2])

            decimal = degrees + minutes/60 + seconds/3600

            if direction in ['S', 'W']:
                decimal = -decimal

            return decimal

        except (IndexError, TypeError, ValueError):
            return None

    def _match_landmarks(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Match image against known landmarks (simulation)"""
        # In production, this would use computer vision APIs like
        # Google Cloud Vision, AWS Rekognition, or specialized landmark databases

        landmarks = []

        # Simulate landmark detection
        # Would analyze image features and match against database

        return landmarks

    def _calculate_location_confidence(
        self,
        visual_features: List[str],
        landmark_matches: List[Dict[str, Any]],
        metadata_location: Optional[Tuple[float, float]]
    ) -> float:
        """Calculate confidence in location estimation"""
        confidence = 0.0

        # GPS metadata provides high confidence
        if metadata_location:
            confidence += 0.8

        # Landmark matches provide medium confidence
        if landmark_matches:
            confidence += 0.6 * min(len(landmark_matches) / 3, 1.0)

        # Visual features provide low confidence
        if visual_features:
            confidence += 0.3 * min(len(visual_features) / 5, 1.0)

        return min(confidence, 1.0)


class MetadataForensics:
    """Analyze image metadata for authenticity indicators"""

    def __init__(self):
        """Initialize metadata forensics analyzer"""
        logger.info("Initialized metadata forensics analyzer")

    def analyze_image_metadata(self, image_path: Path) -> ImageMetadata:
        """
        Analyze image metadata for forensic information.

        Args:
            image_path: Path to image file

        Returns:
            ImageMetadata with forensic analysis
        """
        if not image_path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")

        logger.info(f"Analyzing metadata for {image_path}")

        # Get file statistics
        stat = image_path.stat()
        file_size = stat.st_size
        modification_date = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc)

        # Open image and extract basic info
        with Image.open(image_path) as img:
            dimensions = img.size
            format = img.format or "Unknown"
            mode = img.mode

            # Extract EXIF data
            exif_data = {}
            creation_date = None
            camera_make = None
            camera_model = None
            gps_coordinates = None

            try:
                exif_raw = img._getexif()
                if exif_raw:
                    for tag_id, value in exif_raw.items():
                        tag = TAGS.get(tag_id, tag_id)

                        if tag == "DateTime" and not creation_date:
                            try:
                                creation_date = datetime.strptime(
                                    str(value), "%Y:%m:%d %H:%M:%S"
                                ).replace(tzinfo=timezone.utc)
                            except ValueError:
                                pass
                        elif tag == "Make":
                            camera_make = str(value)
                        elif tag == "Model":
                            camera_model = str(value)
                        elif tag == "GPSInfo":
                            gps_coordinates = self._extract_gps_coordinates(value)

                        # Store all EXIF data (with length limits)
                        if isinstance(value, bytes):
                            value = value.decode('utf-8', errors='ignore')
                        exif_data[str(tag)] = str(value)[:500]  # Limit length

            except Exception as e:
                logger.debug(f"Could not extract EXIF data: {e}")

        return ImageMetadata(
            file_path=image_path,
            file_size=file_size,
            dimensions=dimensions,
            format=format,
            mode=mode,
            creation_date=creation_date,
            modification_date=modification_date,
            camera_make=camera_make,
            camera_model=camera_model,
            gps_coordinates=gps_coordinates,
            exif_data=exif_data
        )

    def _extract_gps_coordinates(self, gps_info: Dict) -> Optional[Tuple[float, float]]:
        """Extract GPS coordinates from EXIF GPS info"""
        try:
            # Parse GPS data
            lat = gps_info.get(2)  # GPSLatitude
            lat_ref = gps_info.get(1)  # GPSLatitudeRef
            lon = gps_info.get(4)  # GPSLongitude
            lon_ref = gps_info.get(3)  # GPSLongitudeRef

            if lat and lon and lat_ref and lon_ref:
                # Convert to decimal degrees
                lat_decimal = self._dms_to_decimal(lat, lat_ref)
                lon_decimal = self._dms_to_decimal(lon, lon_ref)

                if lat_decimal is not None and lon_decimal is not None:
                    return (lat_decimal, lon_decimal)

        except Exception as e:
            logger.debug(f"Could not extract GPS coordinates: {e}")

        return None

    def _dms_to_decimal(self, dms_tuple, reference):
        """Convert DMS (Degrees, Minutes, Seconds) to decimal degrees"""
        try:
            degrees, minutes, seconds = dms_tuple
            decimal = float(degrees) + float(minutes)/60 + float(seconds)/3600

            if reference in ['S', 'W']:
                decimal = -decimal

            return decimal

        except (ValueError, TypeError, IndexError):
            return None