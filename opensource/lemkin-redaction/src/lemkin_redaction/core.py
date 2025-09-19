"""
Core classes and functionality for PII redaction pipeline.

This module provides the main PIIRedactor class and supporting data models
for automated redaction of personally identifiable information.
"""

from enum import Enum
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime, timezone
import uuid

from pydantic import BaseModel, Field, validator
from loguru import logger


class EntityType(str, Enum):
    """Types of PII entities that can be detected and redacted."""
    PERSON = "PERSON"
    ORGANIZATION = "ORGANIZATION" 
    LOCATION = "LOCATION"
    EMAIL = "EMAIL"
    PHONE = "PHONE"
    SSN = "SSN"
    CREDIT_CARD = "CREDIT_CARD"
    IP_ADDRESS = "IP_ADDRESS"
    DATE = "DATE"
    ADDRESS = "ADDRESS"
    MEDICAL = "MEDICAL"
    FINANCIAL = "FINANCIAL"
    CUSTOM = "CUSTOM"


class ConfidenceLevel(str, Enum):
    """Confidence levels for PII detection."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


class RedactionType(str, Enum):
    """Types of redaction methods available."""
    MASK = "mask"  # Replace with asterisks or X's
    BLUR = "blur"  # Blur visual content
    REPLACE = "replace"  # Replace with generic placeholder
    DELETE = "delete"  # Remove completely
    ANONYMIZE = "anonymize"  # Replace with synthetic data


@dataclass
class PIIEntity:
    """Represents a detected PII entity."""
    entity_type: EntityType
    text: str
    start_pos: int
    end_pos: int
    confidence: float
    confidence_level: ConfidenceLevel
    replacement: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class RedactionConfig(BaseModel):
    """Configuration for PII redaction operations."""
    
    # Entity types to redact
    entity_types: List[EntityType] = Field(
        default=[EntityType.PERSON, EntityType.EMAIL, EntityType.PHONE],
        description="List of entity types to redact"
    )
    
    # Redaction method by entity type
    redaction_methods: Dict[EntityType, RedactionType] = Field(
        default={
            EntityType.PERSON: RedactionType.REPLACE,
            EntityType.EMAIL: RedactionType.MASK,
            EntityType.PHONE: RedactionType.MASK,
            EntityType.SSN: RedactionType.MASK,
            EntityType.CREDIT_CARD: RedactionType.MASK,
            EntityType.ADDRESS: RedactionType.REPLACE,
        },
        description="Redaction method for each entity type"
    )
    
    # Minimum confidence threshold
    min_confidence: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Minimum confidence threshold for redaction"
    )
    
    # Language for processing
    language: str = Field(
        default="en",
        description="Language code for processing"
    )
    
    # Custom patterns
    custom_patterns: Dict[str, str] = Field(
        default={},
        description="Custom regex patterns for detection"
    )
    
    # Preserve case and formatting
    preserve_formatting: bool = Field(
        default=True,
        description="Whether to preserve original text formatting"
    )
    
    # Output options
    generate_report: bool = Field(
        default=True,
        description="Whether to generate redaction report"
    )
    
    # Audit trail
    track_changes: bool = Field(
        default=True,
        description="Whether to maintain audit trail of changes"
    )


class RedactionResult(BaseModel):
    """Result of a redaction operation."""
    
    # Unique identifier
    operation_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for this operation"
    )
    
    # Input information
    original_content_hash: str = Field(
        description="Hash of original content for integrity"
    )
    content_type: str = Field(
        description="Type of content processed (text, image, audio, video)"
    )
    
    # Processing results
    entities_detected: List[PIIEntity] = Field(
        description="List of PII entities detected"
    )
    entities_redacted: List[PIIEntity] = Field(
        description="List of entities actually redacted"
    )
    
    # Statistics
    total_entities: int = Field(
        description="Total number of entities detected"
    )
    redacted_count: int = Field(
        description="Number of entities redacted"
    )
    confidence_scores: Dict[str, float] = Field(
        description="Average confidence by entity type"
    )
    
    # Processing metadata
    processing_time: float = Field(
        description="Time taken for processing in seconds"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When processing completed"
    )
    config_used: RedactionConfig = Field(
        description="Configuration used for redaction"
    )
    
    # Output paths
    redacted_content_path: Optional[str] = Field(
        default=None,
        description="Path to redacted content file"
    )
    report_path: Optional[str] = Field(
        default=None, 
        description="Path to redaction report"
    )
    
    # Quality metrics
    redaction_quality: Dict[str, Any] = Field(
        default={},
        description="Quality metrics for redaction"
    )
    
    # Warnings and issues
    warnings: List[str] = Field(
        default=[],
        description="Warnings encountered during processing"
    )
    errors: List[str] = Field(
        default=[],
        description="Errors encountered during processing"
    )


class PIIRedactor:
    """
    Main PII redaction coordinator that manages different redaction types.
    
    This class coordinates text, image, audio, and video redaction while
    maintaining audit trails and ensuring legal compliance.
    """
    
    def __init__(self, config: Optional[RedactionConfig] = None):
        """Initialize PII redactor with configuration."""
        self.config = config or RedactionConfig()
        self.logger = logger
        
        # Initialize component redactors
        self._text_redactor = None
        self._image_redactor = None
        self._audio_redactor = None
        self._video_redactor = None
        
        self.logger.info(f"PIIRedactor initialized with config: {self.config.dict()}")
    
    @property
    def text_redactor(self):
        """Lazy loading of text redactor."""
        if self._text_redactor is None:
            from .text_redactor import TextRedactor
            self._text_redactor = TextRedactor(self.config)
        return self._text_redactor
    
    @property  
    def image_redactor(self):
        """Lazy loading of image redactor."""
        if self._image_redactor is None:
            from .image_redactor import ImageRedactor
            self._image_redactor = ImageRedactor(self.config)
        return self._image_redactor
    
    @property
    def audio_redactor(self):
        """Lazy loading of audio redactor."""
        if self._audio_redactor is None:
            from .audio_redactor import AudioRedactor
            self._audio_redactor = AudioRedactor(self.config)
        return self._audio_redactor
    
    @property
    def video_redactor(self):
        """Lazy loading of video redactor."""
        if self._video_redactor is None:
            from .video_redactor import VideoRedactor
            self._video_redactor = VideoRedactor(self.config)
        return self._video_redactor
    
    def redact_text(self, text: str, output_path: Optional[Path] = None) -> RedactionResult:
        """
        Redact PII from text content.
        
        Args:
            text: Text content to redact
            output_path: Optional path to save redacted text
            
        Returns:
            RedactionResult with processing details
        """
        self.logger.info(f"Starting text redaction, length: {len(text)} characters")
        
        try:
            result = self.text_redactor.redact(text, output_path)
            self.logger.info(f"Text redaction completed: {result.redacted_count} entities redacted")
            return result
            
        except Exception as e:
            self.logger.error(f"Text redaction failed: {e}")
            raise
    
    def redact_image(self, image_path: Path, output_path: Optional[Path] = None) -> RedactionResult:
        """
        Redact PII from image content.
        
        Args:
            image_path: Path to image file
            output_path: Optional path to save redacted image
            
        Returns:
            RedactionResult with processing details
        """
        self.logger.info(f"Starting image redaction: {image_path}")
        
        try:
            result = self.image_redactor.redact(image_path, output_path)
            self.logger.info(f"Image redaction completed: {result.redacted_count} entities redacted")
            return result
            
        except Exception as e:
            self.logger.error(f"Image redaction failed: {e}")
            raise
    
    def redact_audio(self, audio_path: Path, output_path: Optional[Path] = None) -> RedactionResult:
        """
        Redact PII from audio content.
        
        Args:
            audio_path: Path to audio file
            output_path: Optional path to save redacted audio
            
        Returns:
            RedactionResult with processing details
        """
        self.logger.info(f"Starting audio redaction: {audio_path}")
        
        try:
            result = self.audio_redactor.redact(audio_path, output_path)
            self.logger.info(f"Audio redaction completed: {result.redacted_count} entities redacted")
            return result
            
        except Exception as e:
            self.logger.error(f"Audio redaction failed: {e}")
            raise
    
    def redact_video(self, video_path: Path, output_path: Optional[Path] = None) -> RedactionResult:
        """
        Redact PII from video content.
        
        Args:
            video_path: Path to video file
            output_path: Optional path to save redacted video
            
        Returns:
            RedactionResult with processing details
        """
        self.logger.info(f"Starting video redaction: {video_path}")
        
        try:
            result = self.video_redactor.redact(video_path, output_path)
            self.logger.info(f"Video redaction completed: {result.redacted_count} entities redacted")
            return result
            
        except Exception as e:
            self.logger.error(f"Video redaction failed: {e}")
            raise
    
    def redact_file(self, file_path: Path, output_path: Optional[Path] = None) -> RedactionResult:
        """
        Automatically detect file type and apply appropriate redaction.
        
        Args:
            file_path: Path to file to redact
            output_path: Optional path to save redacted file
            
        Returns:
            RedactionResult with processing details
        """
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Detect file type
        suffix = file_path.suffix.lower()
        
        if suffix in ['.txt', '.md', '.doc', '.docx', '.pdf']:
            # Handle text-based files
            content = file_path.read_text(encoding='utf-8')
            return self.redact_text(content, output_path)
            
        elif suffix in ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff']:
            return self.redact_image(file_path, output_path)
            
        elif suffix in ['.wav', '.mp3', '.flac', '.ogg', '.m4a']:
            return self.redact_audio(file_path, output_path)
            
        elif suffix in ['.mp4', '.avi', '.mov', '.mkv', '.wmv']:
            return self.redact_video(file_path, output_path)
            
        else:
            raise ValueError(f"Unsupported file type: {suffix}")
    
    def batch_redact(self, file_paths: List[Path], output_dir: Path) -> List[RedactionResult]:
        """
        Redact multiple files in batch.
        
        Args:
            file_paths: List of file paths to redact
            output_dir: Directory to save redacted files
            
        Returns:
            List of RedactionResult objects
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        results = []
        
        for file_path in file_paths:
            try:
                output_path = output_dir / f"redacted_{file_path.name}"
                result = self.redact_file(file_path, output_path)
                results.append(result)
                
            except Exception as e:
                self.logger.error(f"Failed to redact {file_path}: {e}")
                # Create error result
                error_result = RedactionResult(
                    original_content_hash="",
                    content_type="unknown",
                    entities_detected=[],
                    entities_redacted=[],
                    total_entities=0,
                    redacted_count=0,
                    confidence_scores={},
                    processing_time=0.0,
                    config_used=self.config,
                    errors=[str(e)]
                )
                results.append(error_result)
        
        return results
    
    def update_config(self, config: RedactionConfig) -> None:
        """Update redaction configuration."""
        self.config = config
        
        # Reset component redactors to use new config
        self._text_redactor = None
        self._image_redactor = None
        self._audio_redactor = None
        self._video_redactor = None
        
        self.logger.info("Configuration updated, component redactors reset")
    
    def get_supported_formats(self) -> Dict[str, List[str]]:
        """Get supported file formats by category."""
        return {
            "text": [".txt", ".md", ".doc", ".docx", ".pdf"],
            "image": [".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff"], 
            "audio": [".wav", ".mp3", ".flac", ".ogg", ".m4a"],
            "video": [".mp4", ".avi", ".mov", ".mkv", ".wmv"]
        }