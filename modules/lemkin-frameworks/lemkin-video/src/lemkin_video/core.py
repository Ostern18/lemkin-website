"""
Lemkin Video Authentication Toolkit - Core Module

This module provides the core data models and VideoAuthenticator class for 
comprehensive video authenticity verification and manipulation detection.

Compliance: Berkeley Protocol for Digital Investigations
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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AuthenticityLevel(str, Enum):
    """Video authenticity assessment levels"""
    AUTHENTIC = "authentic"
    LIKELY_AUTHENTIC = "likely_authentic"
    SUSPICIOUS = "suspicious"
    LIKELY_MANIPULATED = "likely_manipulated"
    MANIPULATED = "manipulated"
    UNKNOWN = "unknown"


class TamperingType(str, Enum):
    """Types of video tampering that can be detected"""
    DEEPFAKE = "deepfake"
    FACE_SWAP = "face_swap"
    FRAME_INSERTION = "frame_insertion"
    FRAME_DELETION = "frame_deletion"
    COMPRESSION_INCONSISTENCY = "compression_inconsistency"
    METADATA_MANIPULATION = "metadata_manipulation"
    TEMPORAL_INCONSISTENCY = "temporal_inconsistency"
    LIGHTING_INCONSISTENCY = "lighting_inconsistency"
    AUDIO_VIDEO_MISMATCH = "audio_video_mismatch"
    EDGE_ARTIFACTS = "edge_artifacts"
    PIXEL_LEVEL_EDITING = "pixel_level_editing"


class AnalysisStatus(str, Enum):
    """Status of video analysis operations"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIAL = "partial"
    TIMEOUT = "timeout"


class CompressionLevel(str, Enum):
    """Video compression quality levels"""
    UNCOMPRESSED = "uncompressed"
    LOSSLESS = "lossless"
    HIGH_QUALITY = "high_quality"
    MEDIUM_QUALITY = "medium_quality"
    LOW_QUALITY = "low_quality"
    HEAVILY_COMPRESSED = "heavily_compressed"


class FrameType(str, Enum):
    """Types of video frames"""
    I_FRAME = "i_frame"  # Intra-frame (keyframe)
    P_FRAME = "p_frame"  # Predicted frame
    B_FRAME = "b_frame"  # Bidirectional frame


class VideoAuthConfig(BaseModel):
    """Configuration for video authentication operations"""
    
    # Analysis settings
    deepfake_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    authenticity_threshold: float = Field(default=0.8, ge=0.0, le=1.0)
    frame_analysis_interval: int = Field(default=30, ge=1, le=300)  # frames
    
    # Model settings
    deepfake_model_path: Optional[str] = None
    use_gpu: bool = Field(default=True)
    model_confidence_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    
    # Fingerprinting settings
    fingerprint_algorithm: str = Field(default="dhash")
    hash_size: int = Field(default=8, ge=4, le=32)
    perceptual_hash_threshold: float = Field(default=0.8, ge=0.0, le=1.0)
    
    # Compression analysis
    compression_analysis_enabled: bool = Field(default=True)
    metadata_extraction_enabled: bool = Field(default=True)
    frame_level_analysis_enabled: bool = Field(default=True)
    
    # Performance settings
    max_video_duration_seconds: int = Field(default=3600, ge=1)  # 1 hour max
    max_file_size_mb: int = Field(default=1024, ge=1)  # 1GB max
    processing_timeout_seconds: int = Field(default=1800, ge=60)  # 30 minutes
    
    # Chain of custody
    preserve_original: bool = Field(default=True)
    chain_of_custody_logging: bool = Field(default=True)
    generate_detailed_report: bool = Field(default=True)
    
    # Quality settings
    minimum_resolution: Tuple[int, int] = Field(default=(240, 160))
    minimum_frame_rate: float = Field(default=1.0, ge=0.1)
    
    class Config:
        schema_extra = {
            "example": {
                "deepfake_threshold": 0.7,
                "authenticity_threshold": 0.8,
                "use_gpu": True,
                "fingerprint_algorithm": "dhash",
                "preserve_original": True
            }
        }


class VideoMetadata(BaseModel):
    """Video file metadata"""
    
    file_path: str
    file_hash: str = Field(..., description="SHA-256 hash of file")
    file_size: int = Field(..., ge=0)
    
    # Technical metadata
    duration_seconds: float = Field(..., ge=0.0)
    frame_rate: float = Field(..., ge=0.0)
    width: int = Field(..., ge=1)
    height: int = Field(..., ge=1)
    bit_rate: Optional[int] = Field(None, ge=0)
    
    # Codec information
    video_codec: Optional[str] = None
    audio_codec: Optional[str] = None
    container_format: Optional[str] = None
    
    # Creation metadata
    creation_date: Optional[datetime] = None
    modification_date: Optional[datetime] = None
    camera_make: Optional[str] = None
    camera_model: Optional[str] = None
    
    # Location data
    gps_coordinates: Optional[Dict[str, float]] = None
    location_name: Optional[str] = None
    
    # Technical details
    color_space: Optional[str] = None
    color_depth: Optional[int] = None
    aspect_ratio: Optional[str] = None
    
    # Compression metadata
    compression_level: Optional[CompressionLevel] = None
    quality_metrics: Dict[str, float] = Field(default_factory=dict)
    
    # Chain of custody
    extracted_at: datetime = Field(default_factory=datetime.utcnow)
    extraction_tool: str = Field(default="lemkin-video")
    
    raw_metadata: Dict[str, Any] = Field(default_factory=dict)
    
    @validator('file_hash')
    def validate_hash(cls, v):
        if len(v) != 64:  # SHA-256 is 64 characters
            raise ValueError('file_hash must be a valid SHA-256 hash')
        return v


class TamperingIndicator(BaseModel):
    """Indicator of potential video tampering"""
    
    id: UUID = Field(default_factory=uuid4)
    tampering_type: TamperingType
    confidence: float = Field(..., ge=0.0, le=1.0)
    
    # Location information
    frame_numbers: List[int] = Field(default_factory=list)
    timestamp_range: Optional[Tuple[float, float]] = None  # start, end in seconds
    spatial_coordinates: Optional[Tuple[int, int, int, int]] = None  # x, y, w, h
    
    # Analysis details
    description: str
    evidence: Dict[str, Any] = Field(default_factory=dict)
    analysis_method: str
    
    # Severity
    severity_score: float = Field(..., ge=0.0, le=10.0)
    is_critical: bool = Field(default=False)
    
    # Supporting data
    before_after_hashes: Optional[Tuple[str, str]] = None
    statistical_metrics: Dict[str, float] = Field(default_factory=dict)
    
    detected_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        schema_extra = {
            "example": {
                "tampering_type": "deepfake",
                "confidence": 0.85,
                "frame_numbers": [120, 121, 122],
                "description": "Facial features show inconsistent lighting patterns",
                "severity_score": 7.5,
                "is_critical": True
            }
        }


class KeyFrame(BaseModel):
    """Represents a key frame in video analysis"""
    
    frame_number: int = Field(..., ge=0)
    timestamp: float = Field(..., ge=0.0)
    frame_type: FrameType
    
    # Frame data
    frame_hash: str = Field(..., description="Hash of frame content")
    frame_size: int = Field(..., ge=0)
    
    # Analysis results
    authenticity_score: float = Field(..., ge=0.0, le=1.0)
    tampering_indicators: List[TamperingIndicator] = Field(default_factory=list)
    
    # Technical metrics
    compression_ratio: Optional[float] = Field(None, ge=0.0)
    quality_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    motion_vector_consistency: Optional[float] = Field(None, ge=0.0, le=1.0)
    
    # Visual features
    dominant_colors: List[str] = Field(default_factory=list)
    edge_density: Optional[float] = Field(None, ge=0.0, le=1.0)
    texture_features: Dict[str, float] = Field(default_factory=dict)
    
    # Face analysis (if applicable)
    faces_detected: int = Field(default=0, ge=0)
    face_regions: List[Dict[str, int]] = Field(default_factory=list)
    deepfake_probability: Optional[float] = Field(None, ge=0.0, le=1.0)
    
    extracted_at: datetime = Field(default_factory=datetime.utcnow)


class DeepfakeAnalysis(BaseModel):
    """Results of deepfake detection analysis"""
    
    id: UUID = Field(default_factory=uuid4)
    video_hash: str
    
    # Overall assessment
    is_deepfake: bool
    confidence: float = Field(..., ge=0.0, le=1.0)
    deepfake_probability: float = Field(..., ge=0.0, le=1.0)
    
    # Analysis details
    model_used: str
    model_version: str
    analysis_method: str = Field(default="frame_by_frame")
    
    # Frame-level results
    total_frames_analyzed: int = Field(..., ge=0)
    positive_frames: int = Field(default=0, ge=0)
    negative_frames: int = Field(default=0, ge=0)
    uncertain_frames: int = Field(default=0, ge=0)
    
    # Face analysis
    faces_detected: int = Field(default=0, ge=0)
    face_consistency_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    identity_changes_detected: int = Field(default=0, ge=0)
    
    # Technical indicators
    compression_artifacts: List[str] = Field(default_factory=list)
    temporal_inconsistencies: List[str] = Field(default_factory=list)
    pixel_level_anomalies: List[str] = Field(default_factory=list)
    
    # Performance metrics
    processing_time_seconds: float = Field(..., ge=0.0)
    gpu_used: bool = Field(default=False)
    
    # Detailed results
    frame_analyses: List[Dict[str, Any]] = Field(default_factory=list)
    suspicious_regions: List[Dict[str, Any]] = Field(default_factory=list)
    
    analyzed_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        schema_extra = {
            "example": {
                "is_deepfake": True,
                "confidence": 0.92,
                "deepfake_probability": 0.88,
                "model_used": "EfficientNet-B4",
                "total_frames_analyzed": 300,
                "positive_frames": 278,
                "faces_detected": 1
            }
        }


class VideoFingerprint(BaseModel):
    """Video content fingerprint for duplicate detection"""
    
    id: UUID = Field(default_factory=uuid4)
    video_hash: str
    
    # Fingerprint data
    perceptual_hash: str = Field(..., description="Perceptual hash of video content")
    temporal_hash: str = Field(..., description="Hash based on temporal features")
    audio_fingerprint: Optional[str] = None
    
    # Content features
    frame_count: int = Field(..., ge=0)
    duration_seconds: float = Field(..., ge=0.0)
    resolution: Tuple[int, int]
    average_brightness: float = Field(..., ge=0.0, le=1.0)
    
    # Fingerprinting method
    algorithm_used: str = Field(default="dhash")
    hash_size: int = Field(default=8, ge=4, le=32)
    sample_rate: int = Field(default=1, ge=1)  # frames per second sampled
    
    # Similarity thresholds
    exact_match_threshold: float = Field(default=0.95, ge=0.0, le=1.0)
    similar_match_threshold: float = Field(default=0.8, ge=0.0, le=1.0)
    
    # Technical metadata
    key_frames_hashes: List[str] = Field(default_factory=list)
    motion_vectors: List[float] = Field(default_factory=list)
    color_histograms: List[List[float]] = Field(default_factory=list)
    
    # Processing info
    processing_time_seconds: float = Field(..., ge=0.0)
    quality_score: float = Field(..., ge=0.0, le=1.0)
    
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    def calculate_similarity(self, other_fingerprint: 'VideoFingerprint') -> float:
        """Calculate similarity with another video fingerprint"""
        # Hamming distance for perceptual hashes
        if len(self.perceptual_hash) != len(other_fingerprint.perceptual_hash):
            return 0.0
        
        hamming_distance = sum(c1 != c2 for c1, c2 in 
                              zip(self.perceptual_hash, other_fingerprint.perceptual_hash))
        max_distance = len(self.perceptual_hash)
        
        similarity = 1.0 - (hamming_distance / max_distance)
        return similarity


class CompressionAnalysis(BaseModel):
    """Analysis of video compression for authenticity verification"""
    
    id: UUID = Field(default_factory=uuid4)
    video_hash: str
    
    # Compression assessment
    compression_level: CompressionLevel
    is_recompressed: bool
    recompression_count: int = Field(default=0, ge=0)
    
    # Quality metrics
    overall_quality_score: float = Field(..., ge=0.0, le=1.0)
    bitrate_consistency: float = Field(..., ge=0.0, le=1.0)
    compression_efficiency: float = Field(..., ge=0.0, le=1.0)
    
    # Artifact detection
    blocking_artifacts: float = Field(default=0.0, ge=0.0, le=1.0)
    ringing_artifacts: float = Field(default=0.0, ge=0.0, le=1.0)
    mosquito_noise: float = Field(default=0.0, ge=0.0, le=1.0)
    
    # Technical details
    codec_sequence: List[str] = Field(default_factory=list)
    quantization_parameters: List[int] = Field(default_factory=list)
    motion_compensation_vectors: Dict[str, Any] = Field(default_factory=dict)
    
    # Inconsistency detection
    inconsistent_regions: List[Dict[str, Any]] = Field(default_factory=list)
    compression_boundaries: List[Tuple[int, int]] = Field(default_factory=list)
    quality_variations: List[Dict[str, float]] = Field(default_factory=list)
    
    # Analysis metadata
    analysis_method: str = Field(default="multi_scale")
    tools_used: List[str] = Field(default_factory=list)
    processing_time_seconds: float = Field(..., ge=0.0)
    
    analyzed_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        schema_extra = {
            "example": {
                "compression_level": "medium_quality",
                "is_recompressed": True,
                "recompression_count": 2,
                "overall_quality_score": 0.75,
                "blocking_artifacts": 0.15,
                "codec_sequence": ["H.264", "H.264"]
            }
        }


class AnalysisResult(BaseModel):
    """Combined result of video authentication analysis"""
    
    id: UUID = Field(default_factory=uuid4)
    video_metadata: VideoMetadata
    
    # Overall assessment
    authenticity_level: AuthenticityLevel
    overall_confidence: float = Field(..., ge=0.0, le=1.0)
    authenticity_score: float = Field(..., ge=0.0, le=1.0)
    
    # Component analyses
    deepfake_analysis: Optional[DeepfakeAnalysis] = None
    fingerprint_analysis: Optional[VideoFingerprint] = None
    compression_analysis: Optional[CompressionAnalysis] = None
    key_frames: List[KeyFrame] = Field(default_factory=list)
    
    # Tampering assessment
    tampering_indicators: List[TamperingIndicator] = Field(default_factory=list)
    tampering_probability: float = Field(default=0.0, ge=0.0, le=1.0)
    critical_issues_found: int = Field(default=0, ge=0)
    
    # Analysis summary
    total_analysis_time_seconds: float = Field(..., ge=0.0)
    analysis_status: AnalysisStatus
    analysis_config: VideoAuthConfig
    
    # Warnings and notes
    warnings: List[str] = Field(default_factory=list)
    analysis_notes: Optional[str] = None
    limitations: List[str] = Field(default_factory=list)
    
    # Chain of custody
    analyzed_by: str = Field(default="lemkin-video")
    analyzed_at: datetime = Field(default_factory=datetime.utcnow)
    analysis_environment: Dict[str, Any] = Field(default_factory=dict)
    
    def calculate_final_score(self) -> float:
        """Calculate weighted final authenticity score"""
        weights = {
            'deepfake': 0.4,
            'compression': 0.3,
            'metadata': 0.2,
            'temporal': 0.1
        }
        
        scores = []
        
        if self.deepfake_analysis:
            deepfake_score = 1.0 - self.deepfake_analysis.deepfake_probability
            scores.append(deepfake_score * weights['deepfake'])
        
        if self.compression_analysis:
            compression_score = self.compression_analysis.overall_quality_score
            scores.append(compression_score * weights['compression'])
        
        # Add other component scores as available
        if not scores:
            return self.authenticity_score
        
        return sum(scores) / sum(weights[k] for k in ['deepfake', 'compression'] 
                                if getattr(self, f'{k}_analysis', None))


class AuthenticityReport(BaseModel):
    """Comprehensive authenticity report for legal use"""
    
    id: UUID = Field(default_factory=uuid4)
    case_number: Optional[str] = None
    investigator: Optional[str] = None
    
    # Video information
    video_file_path: str
    original_filename: str
    file_hash: str
    
    # Analysis summary
    analysis_result: AnalysisResult
    executive_summary: str
    key_findings: List[str] = Field(default_factory=list)
    
    # Legal considerations
    chain_of_custody_verified: bool = Field(default=True)
    admissibility_notes: Optional[str] = None
    technical_limitations: List[str] = Field(default_factory=list)
    confidence_interval: Tuple[float, float] = Field(default=(0.0, 1.0))
    
    # Supporting evidence
    evidence_files: List[str] = Field(default_factory=list)
    reference_samples: List[str] = Field(default_factory=list)
    comparison_results: List[Dict[str, Any]] = Field(default_factory=list)
    
    # Report metadata
    report_version: str = Field(default="1.0")
    generated_at: datetime = Field(default_factory=datetime.utcnow)
    report_format: str = Field(default="comprehensive")
    
    # Quality assurance
    peer_reviewed: bool = Field(default=False)
    reviewer_notes: Optional[str] = None
    methodology_verification: bool = Field(default=False)
    
    class Config:
        schema_extra = {
            "example": {
                "case_number": "CASE-2024-001",
                "investigator": "Jane Doe",
                "original_filename": "evidence_video.mp4",
                "executive_summary": "Video analysis indicates high probability of manipulation",
                "chain_of_custody_verified": True,
                "confidence_interval": [0.85, 0.95]
            }
        }


class VideoAuthenticator:
    """
    Main video authentication class providing comprehensive analysis
    for detecting manipulation and verifying authenticity.
    
    Implements Berkeley Protocol compliance for digital investigations.
    """
    
    def __init__(self, config: Optional[VideoAuthConfig] = None):
        """Initialize video authenticator with configuration"""
        self.config = config or VideoAuthConfig()
        self.logger = logging.getLogger(f"{__name__}.VideoAuthenticator")
        
        # Initialize analysis components
        self._deepfake_detector = None
        self._video_fingerprinter = None
        self._compression_analyzer = None
        self._frame_analyzer = None
        
        self.logger.info("Video Authenticator initialized")
        if self.config.chain_of_custody_logging:
            self.logger.info("Chain of custody logging enabled")
    
    def authenticate_video(
        self,
        video_path: Path,
        case_number: Optional[str] = None
    ) -> AuthenticityReport:
        """
        Perform comprehensive video authentication analysis
        
        Args:
            video_path: Path to video file to analyze
            case_number: Optional case number for legal tracking
            
        Returns:
            AuthenticityReport: Comprehensive analysis report
        """
        start_time = datetime.utcnow()
        self.logger.info(f"Starting video authentication for: {video_path}")
        
        try:
            # Extract metadata
            metadata = self._extract_video_metadata(video_path)
            
            # Initialize analysis result
            analysis_result = AnalysisResult(
                video_metadata=metadata,
                authenticity_level=AuthenticityLevel.UNKNOWN,
                overall_confidence=0.0,
                authenticity_score=0.0,
                analysis_status=AnalysisStatus.IN_PROGRESS,
                analysis_config=self.config,
                total_analysis_time_seconds=0.0
            )
            
            # Perform component analyses
            if self.config.deepfake_model_path:
                analysis_result.deepfake_analysis = self.detect_deepfake(video_path)
            
            if self.config.compression_analysis_enabled:
                analysis_result.compression_analysis = self.analyze_compression_artifacts(video_path)
            
            analysis_result.fingerprint_analysis = self.fingerprint_video(video_path)
            
            if self.config.frame_level_analysis_enabled:
                analysis_result.key_frames = self.extract_key_frames(video_path)
            
            # Calculate final assessment
            analysis_result.authenticity_score = analysis_result.calculate_final_score()
            analysis_result.authenticity_level = self._determine_authenticity_level(
                analysis_result.authenticity_score
            )
            analysis_result.overall_confidence = self._calculate_overall_confidence(analysis_result)
            
            # Update timing
            end_time = datetime.utcnow()
            analysis_result.total_analysis_time_seconds = (end_time - start_time).total_seconds()
            analysis_result.analysis_status = AnalysisStatus.COMPLETED
            
            # Generate report
            report = AuthenticityReport(
                case_number=case_number,
                video_file_path=str(video_path),
                original_filename=video_path.name,
                file_hash=metadata.file_hash,
                analysis_result=analysis_result,
                executive_summary=self._generate_executive_summary(analysis_result),
                key_findings=self._extract_key_findings(analysis_result)
            )
            
            self.logger.info(f"Video authentication completed: {analysis_result.authenticity_level}")
            return report
            
        except Exception as e:
            self.logger.error(f"Video authentication failed: {str(e)}")
            raise
    
    def detect_deepfake(self, video_path: Path) -> DeepfakeAnalysis:
        """Detect deepfake manipulation in video"""
        # This will be implemented by the DeepfakeDetector
        if not self._deepfake_detector:
            from .deepfake_detector import DeepfakeDetector
            self._deepfake_detector = DeepfakeDetector(self.config)
        
        return self._deepfake_detector.detect_deepfake(video_path)
    
    def fingerprint_video(self, video_path: Path) -> VideoFingerprint:
        """Generate content-based video fingerprint"""
        # This will be implemented by the VideoFingerprinter
        if not self._video_fingerprinter:
            from .video_fingerprinter import VideoFingerprinter
            self._video_fingerprinter = VideoFingerprinter(self.config)
        
        return self._video_fingerprinter.fingerprint_video(video_path)
    
    def analyze_compression_artifacts(self, video_path: Path) -> CompressionAnalysis:
        """Analyze compression artifacts for authenticity verification"""
        # This will be implemented by the CompressionAnalyzer
        if not self._compression_analyzer:
            from .compression_analyzer import CompressionAnalyzer
            self._compression_analyzer = CompressionAnalyzer(self.config)
        
        return self._compression_analyzer.analyze_compression_artifacts(video_path)
    
    def extract_key_frames(self, video_path: Path) -> List[KeyFrame]:
        """Extract and analyze key frames"""
        # This will be implemented by the FrameAnalyzer
        if not self._frame_analyzer:
            from .frame_analyzer import FrameAnalyzer
            self._frame_analyzer = FrameAnalyzer(self.config)
        
        return self._frame_analyzer.extract_key_frames(video_path)
    
    def _extract_video_metadata(self, video_path: Path) -> VideoMetadata:
        """Extract comprehensive video metadata"""
        # Placeholder implementation - would use ffprobe/OpenCV
        import hashlib
        
        file_hash = hashlib.sha256()
        with open(video_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                file_hash.update(chunk)
        
        return VideoMetadata(
            file_path=str(video_path),
            file_hash=file_hash.hexdigest(),
            file_size=video_path.stat().st_size,
            duration_seconds=60.0,  # Placeholder
            frame_rate=30.0,  # Placeholder
            width=1920,  # Placeholder
            height=1080,  # Placeholder
            video_codec="H.264",  # Placeholder
            container_format="MP4"  # Placeholder
        )
    
    def _determine_authenticity_level(self, score: float) -> AuthenticityLevel:
        """Determine authenticity level from score"""
        if score >= 0.9:
            return AuthenticityLevel.AUTHENTIC
        elif score >= 0.7:
            return AuthenticityLevel.LIKELY_AUTHENTIC
        elif score >= 0.5:
            return AuthenticityLevel.SUSPICIOUS
        elif score >= 0.3:
            return AuthenticityLevel.LIKELY_MANIPULATED
        else:
            return AuthenticityLevel.MANIPULATED
    
    def _calculate_overall_confidence(self, result: AnalysisResult) -> float:
        """Calculate overall confidence in the analysis"""
        # Simplified confidence calculation
        confidence_factors = []
        
        if result.deepfake_analysis:
            confidence_factors.append(result.deepfake_analysis.confidence)
        
        if result.compression_analysis:
            confidence_factors.append(result.compression_analysis.overall_quality_score)
        
        if confidence_factors:
            return sum(confidence_factors) / len(confidence_factors)
        
        return 0.5  # Default moderate confidence
    
    def _generate_executive_summary(self, result: AnalysisResult) -> str:
        """Generate executive summary for legal report"""
        summary_parts = [
            f"Video authenticity assessment: {result.authenticity_level.value.upper()}",
            f"Overall confidence: {result.overall_confidence:.1%}",
            f"Authenticity score: {result.authenticity_score:.1%}"
        ]
        
        if result.tampering_indicators:
            critical_count = sum(1 for indicator in result.tampering_indicators if indicator.is_critical)
            summary_parts.append(f"Critical tampering indicators found: {critical_count}")
        
        return ". ".join(summary_parts) + "."
    
    def _extract_key_findings(self, result: AnalysisResult) -> List[str]:
        """Extract key findings for the report"""
        findings = []
        
        if result.deepfake_analysis and result.deepfake_analysis.is_deepfake:
            findings.append(f"Deepfake detected with {result.deepfake_analysis.confidence:.1%} confidence")
        
        if result.compression_analysis and result.compression_analysis.is_recompressed:
            findings.append(f"Video shows signs of {result.compression_analysis.recompression_count} recompression cycles")
        
        critical_tampering = [t for t in result.tampering_indicators if t.is_critical]
        if critical_tampering:
            findings.append(f"Critical tampering indicators detected: {len(critical_tampering)}")
        
        return findings
    
    def compare_videos(
        self,
        video1_path: Path,
        video2_path: Path
    ) -> Dict[str, Any]:
        """Compare two videos for similarity and detect potential duplicates"""
        fingerprint1 = self.fingerprint_video(video1_path)
        fingerprint2 = self.fingerprint_video(video2_path)
        
        similarity = fingerprint1.calculate_similarity(fingerprint2)
        
        return {
            'similarity_score': similarity,
            'is_exact_match': similarity >= fingerprint1.exact_match_threshold,
            'is_similar': similarity >= fingerprint1.similar_match_threshold,
            'fingerprint1': fingerprint1,
            'fingerprint2': fingerprint2,
            'comparison_timestamp': datetime.utcnow()
        }
    
    def export_report(
        self,
        report: AuthenticityReport,
        output_path: Path,
        format: str = "json"
    ) -> bool:
        """Export authentication report in specified format"""
        try:
            if format.lower() == "json":
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(report.dict(), f, indent=2, default=str)
                return True
            else:
                self.logger.error(f"Unsupported export format: {format}")
                return False
        except Exception as e:
            self.logger.error(f"Export failed: {str(e)}")
            return False