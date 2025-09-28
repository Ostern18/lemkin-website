"""
Core OSINT collection functionality for legal investigations.
"""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from enum import Enum
from urllib.parse import urlparse

import httpx
from pydantic import BaseModel, Field, HttpUrl, field_validator
from loguru import logger
import waybackpy
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
import exif


class SourceType(str, Enum):
    """Types of OSINT sources"""
    SOCIAL_MEDIA = "social_media"
    NEWS_ARTICLE = "news_article"
    GOVERNMENT = "government"
    ACADEMIC = "academic"
    BLOG = "blog"
    FORUM = "forum"
    WEBSITE = "website"
    UNKNOWN = "unknown"


class CredibilityLevel(str, Enum):
    """Source credibility levels"""
    VERIFIED = "verified"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    UNVERIFIED = "unverified"
    SUSPICIOUS = "suspicious"


class Source(BaseModel):
    """OSINT source information"""
    url: HttpUrl
    title: Optional[str] = None
    source_type: SourceType = SourceType.UNKNOWN
    domain: str = Field(default="")
    collected_date: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    content_hash: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def model_post_init(self, __context: Any) -> None:
        """Extract domain from URL"""
        if self.url:
            parsed = urlparse(str(self.url))
            self.domain = parsed.netloc


class OSINTCollection(BaseModel):
    """Collection of OSINT data"""
    collection_id: str
    query: str
    platforms: List[str]
    sources: List[Source]
    collected_date: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    total_items: int = 0
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def model_post_init(self, __context: Any) -> None:
        """Update total items count"""
        self.total_items = len(self.sources)


class ArchiveCollection(BaseModel):
    """Collection of archived web content"""
    archive_id: str
    urls: List[HttpUrl]
    archived_items: List[Dict[str, Any]]
    archive_date: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    wayback_snapshots: Dict[str, List[str]] = Field(default_factory=dict)


class MediaMetadata(BaseModel):
    """Extracted metadata from media files"""
    file_path: Path
    file_hash: str
    file_size: int
    mime_type: str
    creation_date: Optional[datetime] = None
    modification_date: Optional[datetime] = None
    exif_data: Dict[str, Any] = Field(default_factory=dict)
    gps_data: Optional[Dict[str, Any]] = None
    camera_info: Optional[Dict[str, Any]] = None
    software_info: Optional[str] = None


class CredibilityIndicator(BaseModel):
    """Individual credibility indicator"""
    indicator: str
    weight: float
    positive: bool
    details: str


class CredibilityAssessment(BaseModel):
    """Source credibility assessment"""
    source: Source
    credibility_level: CredibilityLevel
    confidence_score: float = Field(ge=0.0, le=1.0)
    indicators: List[CredibilityIndicator]
    assessment_date: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    recommendation: str
    concerns: List[str] = Field(default_factory=list)


class OSINTCollector:
    """Ethical OSINT collection within platform ToS"""

    def __init__(self, user_agent: Optional[str] = None):
        """Initialize OSINT collector"""
        self.user_agent = user_agent or "Lemkin-OSINT/0.1.0 (Legal Investigation Tool)"
        self.client = httpx.Client(
            headers={"User-Agent": self.user_agent},
            timeout=30.0,
            follow_redirects=True
        )
        logger.info("Initialized OSINT collector")

    def collect_social_media_evidence(
        self,
        query: str,
        platforms: List[str],
        limit: int = 100
    ) -> OSINTCollection:
        """
        Collect social media evidence within ToS limits.

        Args:
            query: Search query
            platforms: List of platforms to search
            limit: Maximum items to collect

        Returns:
            OSINTCollection with results
        """
        import uuid

        collection_id = str(uuid.uuid4())
        sources = []

        logger.info(f"Starting OSINT collection for query: {query}")

        # Note: In production, would integrate with platform APIs
        # This is a demonstration of the structure
        for platform in platforms:
            if platform.lower() == "wayback":
                # Example: Check Wayback Machine
                sources.extend(self._search_wayback(query, limit // len(platforms)))
            else:
                logger.warning(f"Platform {platform} not yet implemented")

        collection = OSINTCollection(
            collection_id=collection_id,
            query=query,
            platforms=platforms,
            sources=sources,
            metadata={
                "limit": limit,
                "user_agent": self.user_agent
            }
        )

        logger.info(f"Collected {len(sources)} sources for query: {query}")
        return collection

    def _search_wayback(self, query: str, limit: int) -> List[Source]:
        """Search Wayback Machine for archived content"""
        sources = []

        # This would integrate with Wayback Machine CDX API
        # For demonstration, returning empty list
        logger.info(f"Searching Wayback Machine for: {query}")

        return sources

    def __del__(self):
        """Cleanup client connection"""
        if hasattr(self, 'client'):
            self.client.close()


class WebArchiver:
    """Web content preservation using archives"""

    def __init__(self):
        """Initialize web archiver"""
        self.user_agent = "Lemkin-WebArchiver/0.1.0"
        logger.info("Initialized web archiver")

    def archive_web_content(self, urls: List[str]) -> ArchiveCollection:
        """
        Archive web content using Wayback Machine.

        Args:
            urls: List of URLs to archive

        Returns:
            ArchiveCollection with archive information
        """
        import uuid

        archive_id = str(uuid.uuid4())
        archived_items = []
        wayback_snapshots = {}

        for url in urls:
            try:
                logger.info(f"Archiving URL: {url}")

                # Use waybackpy to interact with Wayback Machine
                wayback = waybackpy.Url(url, self.user_agent)

                # Save to Wayback Machine
                archive = wayback.save()

                # Get available snapshots
                cdx = wayback.cdx_api()
                snapshots = []

                for item in cdx:
                    snapshots.append({
                        "timestamp": item.timestamp,
                        "url": item.archive_url
                    })

                archived_items.append({
                    "original_url": url,
                    "archive_url": archive.archive_url,
                    "timestamp": archive.timestamp,
                    "status": "success"
                })

                wayback_snapshots[url] = snapshots[:10]  # Keep last 10 snapshots

                logger.info(f"Successfully archived: {url}")

            except Exception as e:
                logger.error(f"Failed to archive {url}: {e}")
                archived_items.append({
                    "original_url": url,
                    "error": str(e),
                    "status": "failed"
                })

        return ArchiveCollection(
            archive_id=archive_id,
            urls=[HttpUrl(url) for url in urls],
            archived_items=archived_items,
            wayback_snapshots=wayback_snapshots
        )


class MetadataExtractor:
    """Extract metadata from media files"""

    def __init__(self):
        """Initialize metadata extractor"""
        logger.info("Initialized metadata extractor")

    def extract_media_metadata(self, file_path: Path) -> MediaMetadata:
        """
        Extract EXIF and XMP metadata from media files.

        Args:
            file_path: Path to media file

        Returns:
            MediaMetadata with extracted information
        """
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # Calculate file hash
        file_hash = self._calculate_file_hash(file_path)
        file_size = file_path.stat().st_size

        # Detect MIME type
        mime_type = self._detect_mime_type(file_path)

        # Extract EXIF data
        exif_data = {}
        gps_data = None
        camera_info = None
        software_info = None
        creation_date = None

        if mime_type.startswith("image/"):
            try:
                # Using PIL for EXIF extraction
                image = Image.open(file_path)
                exif_raw = image._getexif()

                if exif_raw:
                    # Parse EXIF tags
                    for tag_id, value in exif_raw.items():
                        tag = TAGS.get(tag_id, tag_id)

                        if tag == "GPSInfo":
                            gps_data = self._parse_gps_info(value)
                        elif tag == "Make":
                            if not camera_info:
                                camera_info = {}
                            camera_info["make"] = value
                        elif tag == "Model":
                            if not camera_info:
                                camera_info = {}
                            camera_info["model"] = value
                        elif tag == "Software":
                            software_info = value
                        elif tag == "DateTimeOriginal":
                            try:
                                creation_date = datetime.strptime(
                                    value, "%Y:%m:%d %H:%M:%S"
                                ).replace(tzinfo=timezone.utc)
                            except:
                                pass

                        # Store cleaned value
                        if isinstance(value, bytes):
                            value = value.decode('utf-8', errors='ignore')
                        exif_data[tag] = str(value)[:1000]  # Limit length

                image.close()

            except Exception as e:
                logger.warning(f"Failed to extract EXIF from {file_path}: {e}")

        # Get file timestamps
        stat = file_path.stat()
        modification_date = datetime.fromtimestamp(
            stat.st_mtime, tz=timezone.utc
        )

        return MediaMetadata(
            file_path=file_path,
            file_hash=file_hash,
            file_size=file_size,
            mime_type=mime_type,
            creation_date=creation_date,
            modification_date=modification_date,
            exif_data=exif_data,
            gps_data=gps_data,
            camera_info=camera_info,
            software_info=software_info
        )

    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA-256 hash of file"""
        sha256 = hashlib.sha256()
        with open(file_path, 'rb') as f:
            while chunk := f.read(8192):
                sha256.update(chunk)
        return sha256.hexdigest()

    def _detect_mime_type(self, file_path: Path) -> str:
        """Detect MIME type of file"""
        import mimetypes
        mime_type, _ = mimetypes.guess_type(str(file_path))
        return mime_type or "application/octet-stream"

    def _parse_gps_info(self, gps_info: Dict) -> Dict[str, Any]:
        """Parse GPS information from EXIF"""
        parsed = {}

        for key, value in gps_info.items():
            tag = GPSTAGS.get(key, key)
            parsed[tag] = value

        # Convert GPS coordinates to decimal degrees
        if all(k in parsed for k in ['GPSLatitude', 'GPSLongitude',
                                      'GPSLatitudeRef', 'GPSLongitudeRef']):
            lat = self._convert_to_degrees(parsed['GPSLatitude'])
            if parsed['GPSLatitudeRef'] == 'S':
                lat = -lat

            lon = self._convert_to_degrees(parsed['GPSLongitude'])
            if parsed['GPSLongitudeRef'] == 'W':
                lon = -lon

            parsed['decimal_coordinates'] = {
                'latitude': lat,
                'longitude': lon
            }

        return parsed

    def _convert_to_degrees(self, value) -> float:
        """Convert GPS coordinates to decimal degrees"""
        d, m, s = value
        return d + (m / 60.0) + (s / 3600.0)


class SourceVerifier:
    """Verify and assess source credibility"""

    def __init__(self):
        """Initialize source verifier"""
        self.trusted_domains = {
            'reuters.com', 'apnews.com', 'bbc.com', 'npr.org',
            'un.org', 'icj-cij.org', 'icc-cpi.int', 'echr.coe.int'
        }
        self.suspicious_indicators = [
            'fake', 'hoax', 'satire', 'clickbait', 'conspiracy'
        ]
        logger.info("Initialized source verifier")

    def verify_source_credibility(self, source: Source) -> CredibilityAssessment:
        """
        Assess credibility of an OSINT source.

        Args:
            source: Source to verify

        Returns:
            CredibilityAssessment with credibility analysis
        """
        indicators = []
        positive_score = 0.0
        negative_score = 0.0

        # Check domain trust
        domain = source.domain.lower()
        if any(trusted in domain for trusted in self.trusted_domains):
            indicators.append(CredibilityIndicator(
                indicator="trusted_domain",
                weight=0.3,
                positive=True,
                details=f"Source from trusted domain: {domain}"
            ))
            positive_score += 0.3

        # Check for suspicious patterns
        if source.title:
            title_lower = source.title.lower()
            for suspicious in self.suspicious_indicators:
                if suspicious in title_lower:
                    indicators.append(CredibilityIndicator(
                        indicator="suspicious_content",
                        weight=0.2,
                        positive=False,
                        details=f"Suspicious keyword found: {suspicious}"
                    ))
                    negative_score += 0.2
                    break

        # Check HTTPS
        if str(source.url).startswith("https://"):
            indicators.append(CredibilityIndicator(
                indicator="secure_connection",
                weight=0.1,
                positive=True,
                details="Uses HTTPS encryption"
            ))
            positive_score += 0.1

        # Check source type
        if source.source_type in [SourceType.GOVERNMENT, SourceType.ACADEMIC]:
            indicators.append(CredibilityIndicator(
                indicator="authoritative_type",
                weight=0.2,
                positive=True,
                details=f"Authoritative source type: {source.source_type.value}"
            ))
            positive_score += 0.2

        # Calculate confidence score
        total_weight = positive_score + negative_score
        if total_weight > 0:
            confidence_score = positive_score / total_weight
        else:
            confidence_score = 0.5

        # Determine credibility level
        if confidence_score >= 0.8:
            credibility_level = CredibilityLevel.HIGH
        elif confidence_score >= 0.6:
            credibility_level = CredibilityLevel.MEDIUM
        elif confidence_score >= 0.4:
            credibility_level = CredibilityLevel.LOW
        else:
            credibility_level = CredibilityLevel.SUSPICIOUS

        # Generate recommendation
        if credibility_level in [CredibilityLevel.HIGH, CredibilityLevel.VERIFIED]:
            recommendation = "Source appears credible for use in investigation"
        elif credibility_level == CredibilityLevel.MEDIUM:
            recommendation = "Source requires additional verification"
        else:
            recommendation = "Source should be treated with caution"

        # Identify concerns
        concerns = []
        if not str(source.url).startswith("https://"):
            concerns.append("No HTTPS encryption")
        if negative_score > 0:
            concerns.append("Suspicious content indicators detected")
        if source.source_type == SourceType.UNKNOWN:
            concerns.append("Unknown source type")

        return CredibilityAssessment(
            source=source,
            credibility_level=credibility_level,
            confidence_score=confidence_score,
            indicators=indicators,
            recommendation=recommendation,
            concerns=concerns
        )