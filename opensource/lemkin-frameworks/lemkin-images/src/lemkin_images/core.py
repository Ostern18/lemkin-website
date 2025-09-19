"""
Lemkin Image Verification Suite Core Module

This module provides the core data models and ImageAuthenticator class for
comprehensive image verification and authenticity analysis. It implements
multi-engine reverse search, manipulation detection, geolocation, and
metadata forensics for legal investigations.

Legal Compliance: Meets standards for digital evidence handling in legal proceedings
"""

from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, validator
import json
import hashlib
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ImageFormat(str, Enum):
    """Supported image formats for analysis"""
    JPEG = "jpeg"
    PNG = "png"
    TIFF = "tiff"
    BMP = "bmp"
    GIF = "gif"
    WEBP = "webp"
    RAW = "raw"
    HEIC = "heic"


class ManipulationType(str, Enum):
    """Types of image manipulation that can be detected"""
    COPY_MOVE = "copy_move"
    SPLICING = "splicing"
    CLONING = "cloning"
    RESAMPLING = "resampling"
    COMPRESSION_INCONSISTENCY = "compression_inconsistency"
    NOISE_INCONSISTENCY = "noise_inconsistency"
    LIGHTING_INCONSISTENCY = "lighting_inconsistency"
    EDGE_INCONSISTENCY = "edge_inconsistency"
    METADATA_MANIPULATION = "metadata_manipulation"
    DEEPFAKE = "deepfake"


class AnalysisStatus(str, Enum):
    """Status of image analysis operations"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIALLY_COMPLETED = "partially_completed"
    REQUIRES_MANUAL_REVIEW = "requires_manual_review"


class SearchEngine(str, Enum):
    """Supported reverse image search engines"""
    GOOGLE = "google"
    TINEYE = "tineye"
    BING = "bing"
    YANDEX = "yandex"
    BAIDU = "baidu"


class ImageAuthConfig(BaseModel):
    """Configuration for image authentication and verification operations"""
    
    # Reverse search settings
    enable_reverse_search: bool = Field(default=True)
    search_engines: List[SearchEngine] = Field(default=[SearchEngine.GOOGLE, SearchEngine.TINEYE, SearchEngine.BING])
    max_search_results: int = Field(default=50, ge=1, le=500)
    search_timeout_seconds: int = Field(default=30, ge=5, le=300)
    
    # Manipulation detection settings
    enable_manipulation_detection: bool = Field(default=True)
    manipulation_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    detailed_analysis: bool = Field(default=True)
    enable_compression_analysis: bool = Field(default=True)
    
    # Geolocation settings
    enable_geolocation: bool = Field(default=True)
    enable_visual_geolocation: bool = Field(default=False)
    geolocation_confidence_threshold: float = Field(default=0.6, ge=0.0, le=1.0)
    
    # Metadata analysis settings
    enable_metadata_forensics: bool = Field(default=True)
    extract_hidden_metadata: bool = Field(default=True)
    verify_metadata_integrity: bool = Field(default=True)
    
    # Processing settings
    max_image_size_mb: int = Field(default=100, ge=1, le=1000)
    analysis_timeout_minutes: int = Field(default=15, ge=1, le=120)
    preserve_original: bool = Field(default=True)
    
    # Output settings
    generate_visual_reports: bool = Field(default=True)
    include_technical_details: bool = Field(default=True)
    confidence_scoring: bool = Field(default=True)


class GeolocationData(BaseModel):
    """Represents geolocation information extracted from or inferred for an image"""
    
    # GPS coordinates
    latitude: Optional[float] = Field(None, ge=-90.0, le=90.0)
    longitude: Optional[float] = Field(None, ge=-180.0, le=180.0)
    altitude: Optional[float] = None
    accuracy: Optional[float] = Field(None, ge=0.0)
    
    # Location details
    country: Optional[str] = None
    region: Optional[str] = None
    city: Optional[str] = None
    postal_code: Optional[str] = None
    address: Optional[str] = None
    
    # Data source
    source: str = Field(..., description="Source of geolocation data")  # "exif", "visual", "reverse_search"
    confidence: float = Field(..., ge=0.0, le=1.0)
    extraction_method: Optional[str] = None
    
    # Visual landmarks (if using visual geolocation)
    identified_landmarks: List[str] = Field(default_factory=list)
    landmark_confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    
    # Timestamp
    timestamp: Optional[datetime] = None


class SearchResult(BaseModel):
    """Individual result from reverse image search"""
    
    id: UUID = Field(default_factory=uuid4)
    search_engine: SearchEngine
    
    # Result details
    url: str = Field(..., min_length=1)
    title: Optional[str] = None
    description: Optional[str] = None
    thumbnail_url: Optional[str] = None
    
    # Similarity metrics
    similarity_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    visual_similarity: Optional[float] = Field(None, ge=0.0, le=1.0)
    
    # Page information
    domain: Optional[str] = None
    page_title: Optional[str] = None
    publication_date: Optional[datetime] = None
    
    # Image metadata from search result
    image_size: Optional[Tuple[int, int]] = None
    file_format: Optional[str] = None
    file_size_bytes: Optional[int] = None
    
    # Context information
    context_text: Optional[str] = None
    language: Optional[str] = None
    location_mentioned: Optional[str] = None


class ReverseSearchResults(BaseModel):
    """Complete results from reverse image search across multiple engines"""
    
    id: UUID = Field(default_factory=uuid4)
    image_hash: str = Field(..., min_length=1)
    
    # Search metadata
    search_timestamp: datetime = Field(default_factory=datetime.utcnow)
    engines_used: List[SearchEngine]
    total_results_found: int = Field(default=0, ge=0)
    
    # Results by engine
    results: List[SearchResult] = Field(default_factory=list)
    
    # Analysis summary
    oldest_result_date: Optional[datetime] = None
    most_recent_result_date: Optional[datetime] = None
    unique_domains: List[str] = Field(default_factory=list)
    potential_source_urls: List[str] = Field(default_factory=list)
    
    # Authenticity indicators
    widespread_usage: bool = Field(default=False)
    stock_photo_indicators: List[str] = Field(default_factory=list)
    social_media_presence: bool = Field(default=False)
    
    # Geographic distribution
    countries_found: List[str] = Field(default_factory=list)
    languages_found: List[str] = Field(default_factory=list)
    
    def get_earliest_appearance(self) -> Optional[SearchResult]:
        """Get the earliest dated appearance of the image"""
        dated_results = [r for r in self.results if r.publication_date]
        if not dated_results:
            return None
        return min(dated_results, key=lambda r: r.publication_date)
    
    def get_highest_similarity(self) -> Optional[SearchResult]:
        """Get the result with highest similarity score"""
        scored_results = [r for r in self.results if r.similarity_score is not None]
        if not scored_results:
            return None
        return max(scored_results, key=lambda r: r.similarity_score)


class ManipulationIndicator(BaseModel):
    """Individual indicator of image manipulation"""
    
    id: UUID = Field(default_factory=uuid4)
    manipulation_type: ManipulationType
    
    # Detection details
    confidence: float = Field(..., ge=0.0, le=1.0)
    severity: str = Field(..., description="low, medium, high, critical")
    
    # Location in image
    affected_regions: List[Tuple[int, int, int, int]] = Field(default_factory=list)  # (x, y, width, height)
    
    # Technical details
    detection_method: str = Field(..., min_length=1)
    algorithm_used: Optional[str] = None
    parameters: Dict[str, Any] = Field(default_factory=dict)
    
    # Evidence
    description: str = Field(..., min_length=1)
    technical_explanation: Optional[str] = None
    visual_evidence_path: Optional[str] = None
    
    # Validation
    false_positive_likelihood: Optional[float] = Field(None, ge=0.0, le=1.0)
    requires_expert_review: bool = Field(default=False)


class ManipulationAnalysis(BaseModel):
    """Complete analysis of potential image manipulation"""
    
    id: UUID = Field(default_factory=uuid4)
    image_hash: str = Field(..., min_length=1)
    
    # Analysis metadata
    analysis_timestamp: datetime = Field(default_factory=datetime.utcnow)
    analysis_duration_seconds: Optional[float] = Field(None, ge=0.0)
    
    # Overall assessment
    is_manipulated: bool = Field(default=False)
    overall_confidence: float = Field(..., ge=0.0, le=1.0)
    manipulation_probability: float = Field(..., ge=0.0, le=1.0)
    
    # Specific indicators
    indicators: List[ManipulationIndicator] = Field(default_factory=list)
    
    # Analysis methods used
    methods_applied: List[str] = Field(default_factory=list)
    algorithms_used: List[str] = Field(default_factory=list)
    
    # Technical findings
    compression_analysis: Optional[Dict[str, Any]] = None
    noise_analysis: Optional[Dict[str, Any]] = None
    frequency_analysis: Optional[Dict[str, Any]] = None
    edge_analysis: Optional[Dict[str, Any]] = None
    
    # Quality assessments
    image_quality_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    authenticity_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    
    # Recommendations
    expert_review_recommended: bool = Field(default=False)
    additional_analysis_needed: List[str] = Field(default_factory=list)
    
    def get_highest_confidence_indicator(self) -> Optional[ManipulationIndicator]:
        """Get the manipulation indicator with highest confidence"""
        if not self.indicators:
            return None
        return max(self.indicators, key=lambda i: i.confidence)
    
    def get_critical_indicators(self) -> List[ManipulationIndicator]:
        """Get all indicators marked as critical severity"""
        return [i for i in self.indicators if i.severity == "critical"]


class GeolocationResult(BaseModel):
    """Complete geolocation analysis result"""
    
    id: UUID = Field(default_factory=uuid4)
    image_hash: str = Field(..., min_length=1)
    
    # Analysis metadata
    analysis_timestamp: datetime = Field(default_factory=datetime.utcnow)
    methods_used: List[str] = Field(default_factory=list)
    
    # Location data
    primary_location: Optional[GeolocationData] = None
    alternative_locations: List[GeolocationData] = Field(default_factory=list)
    
    # Confidence assessment
    overall_confidence: float = Field(..., ge=0.0, le=1.0)
    location_verified: bool = Field(default=False)
    
    # Technical details
    gps_data_present: bool = Field(default=False)
    gps_data_tampered: Optional[bool] = None
    visual_landmarks_detected: bool = Field(default=False)
    
    # Additional context
    timezone_inferred: Optional[str] = None
    elevation_data: Optional[float] = None
    weather_conditions: Optional[Dict[str, Any]] = None
    
    def get_best_location(self) -> Optional[GeolocationData]:
        """Get the location data with highest confidence"""
        if self.primary_location:
            return self.primary_location
        if self.alternative_locations:
            return max(self.alternative_locations, key=lambda l: l.confidence)
        return None


class ImageMetadata(BaseModel):
    """Comprehensive image metadata extracted from file"""
    
    # Basic file information
    file_name: str = Field(..., min_length=1)
    file_path: str = Field(..., min_length=1)
    file_size_bytes: int = Field(..., ge=0)
    file_format: ImageFormat
    
    # Image properties
    width: int = Field(..., ge=1)
    height: int = Field(..., ge=1)
    color_depth: Optional[int] = None
    color_space: Optional[str] = None
    compression: Optional[str] = None
    
    # Timestamps
    creation_time: Optional[datetime] = None
    modification_time: Optional[datetime] = None
    digitized_time: Optional[datetime] = None
    
    # Camera information
    camera_make: Optional[str] = None
    camera_model: Optional[str] = None
    lens_model: Optional[str] = None
    
    # Capture settings
    iso_speed: Optional[int] = None
    aperture: Optional[str] = None
    shutter_speed: Optional[str] = None
    focal_length: Optional[str] = None
    flash_used: Optional[bool] = None
    
    # GPS data
    gps_latitude: Optional[float] = None
    gps_longitude: Optional[float] = None
    gps_altitude: Optional[float] = None
    gps_timestamp: Optional[datetime] = None
    
    # Software information
    software_used: Optional[str] = None
    editing_software: List[str] = Field(default_factory=list)
    
    # Hashes
    md5_hash: str = Field(..., min_length=32, max_length=32)
    sha256_hash: str = Field(..., min_length=64, max_length=64)
    perceptual_hash: Optional[str] = None
    
    # Additional EXIF data
    exif_data: Dict[str, Any] = Field(default_factory=dict)
    hidden_metadata: Dict[str, Any] = Field(default_factory=dict)


class MetadataForensics(BaseModel):
    """Forensic analysis of image metadata for authenticity verification"""
    
    id: UUID = Field(default_factory=uuid4)
    image_hash: str = Field(..., min_length=1)
    
    # Analysis metadata
    analysis_timestamp: datetime = Field(default_factory=datetime.utcnow)
    metadata_source: ImageMetadata
    
    # Authenticity assessment
    metadata_authentic: bool = Field(default=True)
    metadata_confidence: float = Field(..., ge=0.0, le=1.0)
    
    # Inconsistency detection
    timestamp_inconsistencies: List[str] = Field(default_factory=list)
    camera_inconsistencies: List[str] = Field(default_factory=list)
    software_inconsistencies: List[str] = Field(default_factory=list)
    gps_inconsistencies: List[str] = Field(default_factory=list)
    
    # Manipulation indicators
    metadata_stripped: bool = Field(default=False)
    metadata_modified: bool = Field(default=False)
    metadata_fabricated: bool = Field(default=False)
    
    # Technical findings
    exif_integrity: bool = Field(default=True)
    thumbnail_consistency: bool = Field(default=True)
    color_profile_authentic: bool = Field(default=True)
    
    # Camera validation
    camera_database_match: bool = Field(default=False)
    camera_fingerprint_valid: bool = Field(default=False)
    lens_characteristics_match: bool = Field(default=False)
    
    # Additional evidence
    editing_history: List[str] = Field(default_factory=list)
    software_signatures: List[str] = Field(default_factory=list)
    hidden_data_found: bool = Field(default=False)
    
    # Recommendations
    requires_deeper_analysis: bool = Field(default=False)
    expert_validation_needed: bool = Field(default=False)
    chain_of_custody_concerns: List[str] = Field(default_factory=list)


class AuthenticityReport(BaseModel):
    """Comprehensive authenticity report combining all analysis results"""
    
    id: UUID = Field(default_factory=uuid4)
    image_path: str = Field(..., min_length=1)
    image_hash: str = Field(..., min_length=1)
    
    # Report metadata
    generated_at: datetime = Field(default_factory=datetime.utcnow)
    analysis_duration_minutes: Optional[float] = Field(None, ge=0.0)
    analyst: Optional[str] = None
    
    # Overall assessment
    authenticity_verdict: str = Field(..., description="authentic, manipulated, inconclusive, suspicious")
    overall_confidence: float = Field(..., ge=0.0, le=1.0)
    manipulation_likelihood: float = Field(..., ge=0.0, le=1.0)
    
    # Component analyses
    reverse_search_results: Optional[ReverseSearchResults] = None
    manipulation_analysis: Optional[ManipulationAnalysis] = None
    geolocation_result: Optional[GeolocationResult] = None
    metadata_forensics: Optional[MetadataForensics] = None
    
    # Key findings
    critical_findings: List[str] = Field(default_factory=list)
    supporting_evidence: List[str] = Field(default_factory=list)
    red_flags: List[str] = Field(default_factory=list)
    
    # Legal considerations
    admissibility_assessment: Optional[str] = None
    chain_of_custody_status: str = Field(default="maintained")
    expert_testimony_required: bool = Field(default=False)
    
    # Technical summary
    methods_used: List[str] = Field(default_factory=list)
    tools_applied: List[str] = Field(default_factory=list)
    limitations: List[str] = Field(default_factory=list)
    
    # Recommendations
    additional_analysis_recommended: List[str] = Field(default_factory=list)
    expert_review_areas: List[str] = Field(default_factory=list)
    
    def get_executive_summary(self) -> str:
        """Generate executive summary for legal professionals"""
        summary = f"Image authenticity analysis of {Path(self.image_path).name}:\n\n"
        summary += f"VERDICT: {self.authenticity_verdict.upper()}\n"
        summary += f"CONFIDENCE: {self.overall_confidence:.1%}\n"
        summary += f"MANIPULATION LIKELIHOOD: {self.manipulation_likelihood:.1%}\n\n"
        
        if self.critical_findings:
            summary += "CRITICAL FINDINGS:\n"
            for finding in self.critical_findings:
                summary += f"• {finding}\n"
            summary += "\n"
        
        if self.red_flags:
            summary += "RED FLAGS:\n"
            for flag in self.red_flags:
                summary += f"• {flag}\n"
            summary += "\n"
        
        if self.expert_testimony_required:
            summary += "EXPERT TESTIMONY RECOMMENDED\n"
        
        return summary


class ImageAuthenticator:
    """
    Main coordinator class for comprehensive image verification and authenticity analysis.
    
    Provides a unified interface for:
    - Multi-engine reverse image search
    - Advanced manipulation detection
    - Geolocation extraction and verification
    - Forensic metadata analysis
    - Comprehensive authenticity reporting
    """
    
    def __init__(self, config: Optional[ImageAuthConfig] = None):
        """Initialize the image authenticator with configuration"""
        self.config = config or ImageAuthConfig()
        self.logger = logging.getLogger(f"{__name__}.ImageAuthenticator")
        
        # Initialize component analyzers (will be set by specific modules)
        self._reverse_searcher = None
        self._manipulation_detector = None
        self._geolocation_helper = None
        self._metadata_forensics = None
        
        self.logger.info("Image Authenticator initialized")
        
    def authenticate_image(
        self,
        image_path: Path,
        include_reverse_search: bool = True,
        include_manipulation_detection: bool = True,
        include_geolocation: bool = True,
        include_metadata_forensics: bool = True
    ) -> AuthenticityReport:
        """
        Perform comprehensive authenticity analysis on an image
        
        Args:
            image_path: Path to the image file
            include_reverse_search: Whether to perform reverse image search
            include_manipulation_detection: Whether to detect manipulation
            include_geolocation: Whether to extract geolocation
            include_metadata_forensics: Whether to analyze metadata
            
        Returns:
            AuthenticityReport with comprehensive analysis results
        """
        start_time = datetime.utcnow()
        
        if not image_path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        # Calculate image hash
        with open(image_path, 'rb') as f:
            image_data = f.read()
            image_hash = hashlib.sha256(image_data).hexdigest()
        
        self.logger.info(f"Starting authentication analysis for: {image_path.name}")
        
        # Initialize report
        report = AuthenticityReport(
            image_path=str(image_path),
            image_hash=image_hash,
            authenticity_verdict="inconclusive",
            overall_confidence=0.0,
            manipulation_likelihood=0.5
        )
        
        try:
            # Perform reverse image search
            if include_reverse_search and self.config.enable_reverse_search:
                self.logger.info("Performing reverse image search...")
                report.reverse_search_results = self.reverse_search_image(image_path)
                report.methods_used.append("reverse_image_search")
            
            # Detect image manipulation
            if include_manipulation_detection and self.config.enable_manipulation_detection:
                self.logger.info("Detecting image manipulation...")
                report.manipulation_analysis = self.detect_image_manipulation(image_path)
                report.methods_used.append("manipulation_detection")
            
            # Extract geolocation
            if include_geolocation and self.config.enable_geolocation:
                self.logger.info("Extracting geolocation data...")
                report.geolocation_result = self.geolocate_image(image_path)
                report.methods_used.append("geolocation_analysis")
            
            # Analyze metadata
            if include_metadata_forensics and self.config.enable_metadata_forensics:
                self.logger.info("Analyzing metadata forensics...")
                report.metadata_forensics = self.analyze_image_metadata(image_path)
                report.methods_used.append("metadata_forensics")
            
            # Synthesize results
            self._synthesize_authenticity_verdict(report)
            
            # Calculate duration
            end_time = datetime.utcnow()
            duration = end_time - start_time
            report.analysis_duration_minutes = duration.total_seconds() / 60
            
            self.logger.info(f"Authentication analysis completed: {report.authenticity_verdict}")
            
        except Exception as e:
            self.logger.error(f"Authentication analysis failed: {str(e)}")
            report.authenticity_verdict = "error"
            report.critical_findings.append(f"Analysis failed: {str(e)}")
        
        return report
    
    def reverse_search_image(self, image_path: Path) -> ReverseSearchResults:
        """Perform multi-engine reverse image search"""
        if not self._reverse_searcher:
            from .reverse_search import ReverseImageSearcher
            self._reverse_searcher = ReverseImageSearcher(self.config)
        
        return self._reverse_searcher.search_image(image_path)
    
    def detect_image_manipulation(self, image_path: Path) -> ManipulationAnalysis:
        """Detect image manipulation using multiple algorithms"""
        if not self._manipulation_detector:
            from .manipulation_detector import ImageManipulationDetector
            self._manipulation_detector = ImageManipulationDetector(self.config)
        
        return self._manipulation_detector.detect_manipulation(image_path)
    
    def geolocate_image(self, image_path: Path) -> GeolocationResult:
        """Extract or infer image geolocation"""
        if not self._geolocation_helper:
            from .geolocation_helper import ImageGeolocator
            self._geolocation_helper = ImageGeolocator(self.config)
        
        return self._geolocation_helper.geolocate_image(image_path)
    
    def analyze_image_metadata(self, image_path: Path) -> MetadataForensics:
        """Perform forensic analysis of image metadata"""
        if not self._metadata_forensics:
            from .metadata_forensics import MetadataForensicsAnalyzer
            self._metadata_forensics = MetadataForensicsAnalyzer(self.config)
        
        return self._metadata_forensics.analyze_metadata(image_path)
    
    def _synthesize_authenticity_verdict(self, report: AuthenticityReport):
        """Synthesize overall authenticity verdict from component analyses"""
        confidence_scores = []
        manipulation_indicators = []
        
        # Analyze manipulation detection results
        if report.manipulation_analysis:
            manipulation_indicators.append(report.manipulation_analysis.manipulation_probability)
            confidence_scores.append(report.manipulation_analysis.overall_confidence)
            
            if report.manipulation_analysis.is_manipulated:
                report.red_flags.append("Image manipulation detected")
                if report.manipulation_analysis.get_critical_indicators():
                    report.critical_findings.append("Critical manipulation indicators found")
        
        # Analyze metadata forensics results
        if report.metadata_forensics:
            confidence_scores.append(report.metadata_forensics.metadata_confidence)
            
            if not report.metadata_forensics.metadata_authentic:
                report.red_flags.append("Metadata authenticity concerns")
            
            if report.metadata_forensics.metadata_modified:
                report.red_flags.append("Metadata has been modified")
        
        # Analyze reverse search results
        if report.reverse_search_results:
            if report.reverse_search_results.total_results_found > 100:
                report.supporting_evidence.append("Image widely distributed online")
            
            earliest = report.reverse_search_results.get_earliest_appearance()
            if earliest and earliest.publication_date:
                report.supporting_evidence.append(f"Earliest online appearance: {earliest.publication_date.strftime('%Y-%m-%d')}")
        
        # Analyze geolocation results
        if report.geolocation_result:
            confidence_scores.append(report.geolocation_result.overall_confidence)
            
            if report.geolocation_result.gps_data_tampered:
                report.red_flags.append("GPS data may have been tampered with")
        
        # Calculate overall scores
        if confidence_scores:
            report.overall_confidence = sum(confidence_scores) / len(confidence_scores)
        
        if manipulation_indicators:
            report.manipulation_likelihood = max(manipulation_indicators)
        
        # Determine verdict
        if report.manipulation_likelihood > 0.8:
            report.authenticity_verdict = "manipulated"
        elif report.manipulation_likelihood < 0.2 and report.overall_confidence > 0.7:
            report.authenticity_verdict = "authentic"
        elif len(report.red_flags) >= 3:
            report.authenticity_verdict = "suspicious"
        else:
            report.authenticity_verdict = "inconclusive"
        
        # Set expert review requirements
        if report.authenticity_verdict in ["manipulated", "suspicious"]:
            report.expert_testimony_required = True
        
        if report.overall_confidence < 0.6:
            report.additional_analysis_recommended.append("Low confidence - additional analysis recommended")