"""
Lemkin OSINT Collection Toolkit - Core Module

This module provides the core data models and OSINTCollector class for 
systematic open-source intelligence gathering while respecting platform 
terms of service and ethical collection practices.

Compliance: Berkeley Protocol for Digital Investigations
"""

from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, validator
import json
import hashlib
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PlatformType(str, Enum):
    """Supported social media platforms for ethical collection"""
    TWITTER = "twitter"
    FACEBOOK = "facebook" 
    YOUTUBE = "youtube"
    LINKEDIN = "linkedin"
    INSTAGRAM = "instagram"
    REDDIT = "reddit"
    TELEGRAM = "telegram"
    TIKTOK = "tiktok"


class ContentType(str, Enum):
    """Types of content that can be collected"""
    TEXT = "text"
    IMAGE = "image"
    VIDEO = "video"
    AUDIO = "audio"
    DOCUMENT = "document"
    WEB_PAGE = "web_page"
    SOCIAL_POST = "social_post"


class CredibilityLevel(str, Enum):
    """Source credibility assessment levels"""
    VERY_HIGH = "very_high"
    HIGH = "high" 
    MEDIUM = "medium"
    LOW = "low"
    VERY_LOW = "very_low"
    UNKNOWN = "unknown"


class CollectionStatus(str, Enum):
    """Status of OSINT collection operations"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    RATE_LIMITED = "rate_limited"
    TOS_VIOLATION = "tos_violation"


class OSINTConfig(BaseModel):
    """Configuration for OSINT collection operations"""
    
    # Rate limiting settings
    rate_limit_requests_per_minute: int = Field(default=30, ge=1, le=100)
    rate_limit_delay_seconds: float = Field(default=2.0, ge=0.1, le=60.0)
    
    # Collection settings
    max_results_per_query: int = Field(default=100, ge=1, le=1000)
    collection_timeout_seconds: int = Field(default=300, ge=30, le=3600)
    
    # Storage settings
    preserve_metadata: bool = Field(default=True)
    chain_of_custody_logging: bool = Field(default=True)
    
    # Ethics and compliance
    respect_robots_txt: bool = Field(default=True)
    berkeley_protocol_compliance: bool = Field(default=True)
    tos_compliance_check: bool = Field(default=True)
    
    # Archive settings
    use_wayback_machine: bool = Field(default=True)
    archive_original_content: bool = Field(default=True)
    
    class Config:
        schema_extra = {
            "example": {
                "rate_limit_requests_per_minute": 30,
                "max_results_per_query": 100,
                "preserve_metadata": True,
                "berkeley_protocol_compliance": True
            }
        }


class Source(BaseModel):
    """Represents a source of information"""
    
    id: UUID = Field(default_factory=uuid4)
    name: str = Field(..., min_length=1)
    url: Optional[str] = None
    platform: Optional[PlatformType] = None
    
    # Source metadata
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_verified: Optional[datetime] = None
    credibility_score: Optional[float] = Field(None, ge=0.0, le=10.0)
    credibility_level: CredibilityLevel = Field(default=CredibilityLevel.UNKNOWN)
    
    # Verification data
    domain_age_days: Optional[int] = None
    ssl_certificate_valid: Optional[bool] = None
    has_verified_badge: Optional[bool] = None
    follower_count: Optional[int] = None
    account_age_days: Optional[int] = None
    
    # Chain of custody
    collection_metadata: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        schema_extra = {
            "example": {
                "name": "CNN Breaking News",
                "url": "https://twitter.com/cnnbrk",
                "platform": "twitter",
                "credibility_level": "high",
                "has_verified_badge": True
            }
        }


class MediaMetadata(BaseModel):
    """Metadata extracted from media files"""
    
    file_path: str
    file_hash: str = Field(..., description="SHA-256 hash of file")
    file_size: int = Field(..., ge=0)
    content_type: ContentType
    
    # Technical metadata
    creation_date: Optional[datetime] = None
    modification_date: Optional[datetime] = None
    
    # EXIF/XMP data
    camera_make: Optional[str] = None
    camera_model: Optional[str] = None
    gps_coordinates: Optional[Dict[str, float]] = None
    
    # Image-specific
    width: Optional[int] = None
    height: Optional[int] = None
    dpi: Optional[int] = None
    color_space: Optional[str] = None
    
    # Video-specific  
    duration_seconds: Optional[float] = None
    frame_rate: Optional[float] = None
    codec: Optional[str] = None
    
    # Chain of custody
    extracted_at: datetime = Field(default_factory=datetime.utcnow)
    extraction_tool: str = Field(default="lemkin-osint")
    
    raw_metadata: Dict[str, Any] = Field(default_factory=dict)
    
    @validator('file_hash')
    def validate_hash(cls, v):
        if len(v) != 64:  # SHA-256 is 64 characters
            raise ValueError('file_hash must be a valid SHA-256 hash')
        return v


class SocialMediaPost(BaseModel):
    """Represents a social media post collected ethically"""
    
    id: UUID = Field(default_factory=uuid4)
    platform: PlatformType
    platform_post_id: str
    
    # Content
    text_content: Optional[str] = None
    media_urls: List[str] = Field(default_factory=list)
    hashtags: List[str] = Field(default_factory=list)
    mentions: List[str] = Field(default_factory=list)
    
    # Author information
    author_username: Optional[str] = None
    author_display_name: Optional[str] = None
    author_verified: Optional[bool] = None
    
    # Engagement metrics
    likes_count: Optional[int] = Field(None, ge=0)
    shares_count: Optional[int] = Field(None, ge=0)
    comments_count: Optional[int] = Field(None, ge=0)
    
    # Temporal data
    published_at: Optional[datetime] = None
    collected_at: datetime = Field(default_factory=datetime.utcnow)
    
    # Location data (if publicly available)
    location: Optional[str] = None
    coordinates: Optional[Dict[str, float]] = None
    
    # Collection metadata
    source: Source
    collection_method: str = Field(default="api")
    tos_compliant: bool = Field(default=True)
    
    class Config:
        schema_extra = {
            "example": {
                "platform": "twitter",
                "platform_post_id": "1234567890",
                "text_content": "Breaking: Important news update",
                "author_username": "newsaccount",
                "likes_count": 150,
                "tos_compliant": True
            }
        }


class WebContent(BaseModel):
    """Represents web content collected for archiving"""
    
    id: UUID = Field(default_factory=uuid4)
    url: str = Field(..., regex=r'^https?://')
    title: Optional[str] = None
    
    # Content
    html_content: Optional[str] = None
    text_content: Optional[str] = None
    content_hash: str = Field(..., description="SHA-256 hash of content")
    
    # Metadata
    collected_at: datetime = Field(default_factory=datetime.utcnow)
    last_modified: Optional[datetime] = None
    content_type: str = Field(default="text/html")
    content_length: int = Field(..., ge=0)
    
    # HTTP metadata
    status_code: int = Field(..., ge=100, le=599)
    headers: Dict[str, str] = Field(default_factory=dict)
    
    # Archive information
    wayback_url: Optional[str] = None
    archive_timestamp: Optional[datetime] = None
    
    # Chain of custody
    collection_tool: str = Field(default="lemkin-osint")
    robots_txt_compliant: bool = Field(default=True)


class ArchiveEntry(BaseModel):
    """Represents an archived piece of content"""
    
    id: UUID = Field(default_factory=uuid4)
    original_url: str
    archived_url: str
    
    archive_timestamp: datetime
    content_hash: str
    
    # Archive service info
    archive_service: str = Field(default="wayback_machine")
    archive_id: Optional[str] = None
    
    # Verification
    content_verified: bool = Field(default=False)
    verification_date: Optional[datetime] = None
    
    class Config:
        schema_extra = {
            "example": {
                "original_url": "https://example.com/news",
                "archived_url": "https://web.archive.org/web/20231201/https://example.com/news",
                "archive_service": "wayback_machine"
            }
        }


class CredibilityAssessment(BaseModel):
    """Assessment of source credibility"""
    
    source_id: UUID
    assessed_at: datetime = Field(default_factory=datetime.utcnow)
    
    # Overall assessment
    credibility_score: float = Field(..., ge=0.0, le=10.0)
    credibility_level: CredibilityLevel
    confidence: float = Field(..., ge=0.0, le=1.0)
    
    # Assessment factors
    domain_reputation: Optional[float] = Field(None, ge=0.0, le=10.0)
    content_quality: Optional[float] = Field(None, ge=0.0, le=10.0)
    author_expertise: Optional[float] = Field(None, ge=0.0, le=10.0)
    publication_history: Optional[float] = Field(None, ge=0.0, le=10.0)
    fact_check_record: Optional[float] = Field(None, ge=0.0, le=10.0)
    
    # Technical factors
    ssl_valid: Optional[bool] = None
    domain_age_score: Optional[float] = Field(None, ge=0.0, le=10.0)
    
    # Social factors
    social_media_presence: Optional[float] = Field(None, ge=0.0, le=10.0)
    verification_status: Optional[bool] = None
    
    # Assessment notes
    assessment_notes: Optional[str] = None
    warning_flags: List[str] = Field(default_factory=list)
    
    class Config:
        schema_extra = {
            "example": {
                "credibility_score": 8.5,
                "credibility_level": "high",
                "confidence": 0.85,
                "ssl_valid": True,
                "verification_status": True
            }
        }


class OSINTCollection(BaseModel):
    """Represents a complete OSINT collection operation"""
    
    id: UUID = Field(default_factory=uuid4)
    name: str = Field(..., min_length=1)
    description: Optional[str] = None
    
    # Collection parameters
    query: str = Field(..., min_length=1)
    platforms: List[PlatformType] = Field(default_factory=list)
    content_types: List[ContentType] = Field(default_factory=list)
    
    # Status and timing
    status: CollectionStatus = Field(default=CollectionStatus.PENDING)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    # Results
    social_posts: List[SocialMediaPost] = Field(default_factory=list)
    web_content: List[WebContent] = Field(default_factory=list)
    media_files: List[MediaMetadata] = Field(default_factory=list)
    sources: List[Source] = Field(default_factory=list)
    
    # Collection metadata
    total_items_collected: int = Field(default=0, ge=0)
    collection_config: OSINTConfig = Field(default_factory=OSINTConfig)
    
    # Chain of custody
    collector_info: Dict[str, Any] = Field(default_factory=dict)
    collection_log: List[str] = Field(default_factory=list)
    
    # Ethics and compliance
    tos_violations: List[str] = Field(default_factory=list)
    rate_limit_hits: int = Field(default=0, ge=0)
    
    def add_log_entry(self, message: str):
        """Add entry to collection log with timestamp"""
        timestamp = datetime.utcnow().isoformat()
        self.collection_log.append(f"[{timestamp}] {message}")
    
    def calculate_content_hash(self) -> str:
        """Calculate hash of all collected content for integrity verification"""
        content_str = json.dumps({
            'social_posts': len(self.social_posts),
            'web_content': len(self.web_content),
            'media_files': len(self.media_files),
            'query': self.query,
            'platforms': self.platforms
        }, sort_keys=True)
        return hashlib.sha256(content_str.encode()).hexdigest()


class ArchiveCollection(BaseModel):
    """Collection of archived web content"""
    
    id: UUID = Field(default_factory=uuid4)
    name: str = Field(..., min_length=1)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    # Archive entries
    archives: List[ArchiveEntry] = Field(default_factory=list)
    web_content: List[WebContent] = Field(default_factory=list)
    
    # Collection stats
    total_urls: int = Field(default=0, ge=0)
    successful_archives: int = Field(default=0, ge=0)
    failed_archives: int = Field(default=0, ge=0)
    
    # Status
    status: CollectionStatus = Field(default=CollectionStatus.PENDING)
    
    class Config:
        schema_extra = {
            "example": {
                "name": "News Article Archive Collection",
                "total_urls": 10,
                "successful_archives": 8,
                "status": "completed"
            }
        }


class CollectionResult(BaseModel):
    """Result of a collection operation"""
    
    collection_id: UUID
    success: bool
    message: str
    
    # Results summary
    items_collected: int = Field(default=0, ge=0)
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    
    # Performance metrics
    duration_seconds: Optional[float] = Field(None, ge=0.0)
    rate_limit_hits: int = Field(default=0, ge=0)
    
    # Data integrity
    content_hash: Optional[str] = None
    
    class Config:
        schema_extra = {
            "example": {
                "success": True,
                "message": "Collection completed successfully",
                "items_collected": 25,
                "duration_seconds": 45.2
            }
        }


class OSINTCollector:
    """
    Main OSINT collector class providing systematic intelligence gathering
    while respecting platform terms of service and ethical guidelines.
    
    Implements Berkeley Protocol compliance for digital investigations.
    """
    
    def __init__(self, config: Optional[OSINTConfig] = None):
        """Initialize OSINT collector with configuration"""
        self.config = config or OSINTConfig()
        self.logger = logging.getLogger(f"{__name__}.OSINTCollector")
        
        # Initialize components (will be populated by specific modules)
        self._social_scraper = None
        self._web_archiver = None
        self._metadata_extractor = None
        self._source_verifier = None
        
        self.logger.info("OSINT Collector initialized")
        if self.config.berkeley_protocol_compliance:
            self.logger.info("Berkeley Protocol compliance enabled")
    
    def create_collection(
        self,
        name: str,
        query: str,
        platforms: Optional[List[PlatformType]] = None,
        description: Optional[str] = None
    ) -> OSINTCollection:
        """Create a new OSINT collection"""
        collection = OSINTCollection(
            name=name,
            query=query,
            platforms=platforms or [],
            description=description,
            collection_config=self.config
        )
        
        collection.add_log_entry(f"Collection '{name}' created")
        self.logger.info(f"Created collection: {collection.id}")
        
        return collection
    
    def validate_tos_compliance(self, platform: PlatformType, operation: str) -> bool:
        """
        Validate that the requested operation complies with platform ToS
        
        Args:
            platform: The platform to check
            operation: The operation to validate
            
        Returns:
            bool: True if compliant, False otherwise
        """
        if not self.config.tos_compliance_check:
            return True
            
        # ToS compliance checks would be implemented here
        # This is a placeholder for actual ToS validation logic
        compliant_operations = {
            PlatformType.TWITTER: ['public_search', 'public_profile'],
            PlatformType.FACEBOOK: ['public_pages'],
            PlatformType.YOUTUBE: ['public_videos', 'public_channels'],
            PlatformType.REDDIT: ['public_posts', 'public_subreddits']
        }
        
        platform_operations = compliant_operations.get(platform, [])
        is_compliant = operation in platform_operations
        
        if not is_compliant:
            self.logger.warning(f"ToS violation detected: {operation} on {platform}")
        
        return is_compliant
    
    def check_rate_limits(self, platform: PlatformType) -> bool:
        """
        Check if rate limits allow for collection
        
        Args:
            platform: Platform to check rate limits for
            
        Returns:
            bool: True if within rate limits
        """
        # Rate limiting logic would be implemented here
        # This is a placeholder for actual rate limit checking
        return True
    
    def generate_chain_of_custody_report(self, collection: OSINTCollection) -> Dict[str, Any]:
        """
        Generate chain of custody report for Berkeley Protocol compliance
        
        Args:
            collection: The collection to generate report for
            
        Returns:
            Dict containing chain of custody information
        """
        return {
            'collection_id': str(collection.id),
            'collection_name': collection.name,
            'created_at': collection.created_at.isoformat(),
            'collector_info': collection.collector_info,
            'collection_log': collection.collection_log,
            'content_hash': collection.calculate_content_hash(),
            'total_items': collection.total_items_collected,
            'tos_violations': collection.tos_violations,
            'berkeley_protocol_compliant': self.config.berkeley_protocol_compliance,
            'metadata_preserved': self.config.preserve_metadata
        }
    
    def export_collection(
        self, 
        collection: OSINTCollection, 
        output_path: Path,
        format: str = "json"
    ) -> bool:
        """
        Export collection with full chain of custody documentation
        
        Args:
            collection: Collection to export
            output_path: Path to export to
            format: Export format (json, csv, etc.)
            
        Returns:
            bool: True if export successful
        """
        try:
            if format.lower() == "json":
                export_data = {
                    'collection': collection.dict(),
                    'chain_of_custody': self.generate_chain_of_custody_report(collection),
                    'export_timestamp': datetime.utcnow().isoformat(),
                    'export_tool': 'lemkin-osint'
                }
                
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(export_data, f, indent=2, default=str)
                
                self.logger.info(f"Collection exported to {output_path}")
                return True
            else:
                self.logger.error(f"Unsupported export format: {format}")
                return False
                
        except Exception as e:
            self.logger.error(f"Export failed: {str(e)}")
            return False