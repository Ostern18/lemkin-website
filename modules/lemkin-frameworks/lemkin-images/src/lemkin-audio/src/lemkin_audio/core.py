"""
Core audio analysis components and data models for lemkin-audio.
Provides forensic-grade audio processing with chain of custody support.
"""

import hashlib
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import librosa
import numpy as np
import soundfile as sf
from pydantic import BaseModel, Field, validator
from scipy import stats


class AudioFormat(str, Enum):
    """Supported audio formats."""
    WAV = "wav"
    MP3 = "mp3"
    FLAC = "flac"
    M4A = "m4a"
    OGG = "ogg"
    AAC = "aac"


class ConfidenceLevel(str, Enum):
    """Confidence levels for analysis results."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    UNKNOWN = "unknown"


class ManipulationType(str, Enum):
    """Types of audio manipulation detected."""
    NONE = "none"
    SPLICE = "splice"
    COPY_MOVE = "copy_move"
    DEEPFAKE = "deepfake"
    NOISE_REDUCTION = "noise_reduction"
    PITCH_SHIFT = "pitch_shift"
    TIME_STRETCH = "time_stretch"
    COMPRESSION_ARTIFACTS = "compression_artifacts"
    UNKNOWN = "unknown"


class AudioMetadata(BaseModel):
    """Audio file metadata and technical properties."""
    file_path: Path
    file_size: int = Field(description="File size in bytes")
    duration: float = Field(description="Audio duration in seconds")
    sample_rate: int = Field(description="Sample rate in Hz")
    channels: int = Field(description="Number of audio channels")
    bit_depth: Optional[int] = Field(None, description="Bit depth")
    format: AudioFormat = Field(description="Audio format")
    codec: Optional[str] = Field(None, description="Audio codec")
    bitrate: Optional[int] = Field(None, description="Bitrate in kbps")
    created_date: Optional[datetime] = Field(None, description="File creation date")
    modified_date: Optional[datetime] = Field(None, description="File modification date")
    md5_hash: str = Field(description="MD5 hash for integrity verification")
    sha256_hash: str = Field(description="SHA256 hash for integrity verification")

    class Config:
        json_encoders = {
            Path: str,
            datetime: lambda v: v.isoformat() if v else None
        }


class AudioQualityMetrics(BaseModel):
    """Audio quality assessment metrics."""
    snr_db: Optional[float] = Field(None, description="Signal-to-noise ratio in dB")
    thd_percent: Optional[float] = Field(None, description="Total harmonic distortion")
    dynamic_range_db: float = Field(description="Dynamic range in dB")
    peak_level_db: float = Field(description="Peak level in dB")
    rms_level_db: float = Field(description="RMS level in dB")
    spectral_centroid: float = Field(description="Spectral centroid")
    spectral_rolloff: float = Field(description="Spectral rolloff frequency")
    zero_crossing_rate: float = Field(description="Zero crossing rate")
    mfcc_features: List[float] = Field(description="MFCC coefficients")
    spectral_features: Dict[str, float] = Field(description="Additional spectral features")


class TranscriptionSegment(BaseModel):
    """Individual transcription segment with timing and confidence."""
    start_time: float = Field(description="Segment start time in seconds")
    end_time: float = Field(description="Segment end time in seconds")
    text: str = Field(description="Transcribed text")
    confidence: float = Field(ge=0.0, le=1.0, description="Transcription confidence")
    language: str = Field(description="Detected/specified language")
    speaker_id: Optional[str] = Field(None, description="Speaker identifier if available")
    words: List[Dict[str, Any]] = Field(default_factory=list, description="Word-level details")


class AudioTranscription(BaseModel):
    """Complete audio transcription with metadata."""
    audio_metadata: AudioMetadata
    segments: List[TranscriptionSegment] = Field(description="Transcription segments")
    full_text: str = Field(description="Complete transcribed text")
    language: str = Field(description="Primary detected/specified language")
    confidence: float = Field(ge=0.0, le=1.0, description="Overall confidence")
    processing_time: float = Field(description="Processing time in seconds")
    model_used: str = Field(description="Transcription model used")
    timestamp: datetime = Field(default_factory=datetime.now)


class VoicePrint(BaseModel):
    """Voice biometric characteristics for speaker identification."""
    speaker_id: str = Field(description="Unique speaker identifier")
    mfcc_features: List[float] = Field(description="MFCC feature vector")
    pitch_mean: float = Field(description="Mean fundamental frequency")
    pitch_std: float = Field(description="Pitch standard deviation")
    formant_frequencies: List[float] = Field(description="Formant frequencies F1-F4")
    spectral_features: Dict[str, float] = Field(description="Spectral characteristics")
    prosodic_features: Dict[str, float] = Field(description="Prosodic characteristics")
    voice_quality_features: Dict[str, float] = Field(description="Voice quality metrics")
    embedding_vector: List[float] = Field(description="Deep learning embedding")
    confidence: float = Field(ge=0.0, le=1.0, description="Extraction confidence")


class SpeakerProfile(BaseModel):
    """Speaker profile with multiple voice samples."""
    speaker_id: str = Field(description="Unique speaker identifier")
    name: Optional[str] = Field(None, description="Speaker name if known")
    voice_prints: List[VoicePrint] = Field(description="Voice biometric samples")
    average_features: Dict[str, float] = Field(description="Averaged features")
    quality_score: float = Field(ge=0.0, le=1.0, description="Profile quality")
    sample_count: int = Field(description="Number of voice samples")
    created_date: datetime = Field(default_factory=datetime.now)
    last_updated: datetime = Field(default_factory=datetime.now)


class SpeakerAnalysis(BaseModel):
    """Speaker identification and diarization results."""
    audio_metadata: AudioMetadata
    speakers: List[SpeakerProfile] = Field(description="Identified speakers")
    diarization_segments: List[Dict[str, Any]] = Field(description="Speaker diarization")
    speaker_changes: List[float] = Field(description="Speaker change timestamps")
    confidence_scores: Dict[str, float] = Field(description="Per-speaker confidence")
    total_speakers: int = Field(description="Number of unique speakers detected")
    processing_time: float = Field(description="Analysis processing time")
    model_used: str = Field(description="Speaker recognition model")
    timestamp: datetime = Field(default_factory=datetime.now)


class EnhancedAudio(BaseModel):
    """Audio enhancement results and metadata."""
    original_metadata: AudioMetadata
    enhanced_file_path: Path
    enhancement_applied: List[str] = Field(description="Enhancement techniques applied")
    quality_improvement: Dict[str, float] = Field(description="Quality metrics improvement")
    processing_parameters: Dict[str, Any] = Field(description="Enhancement parameters used")
    before_metrics: AudioQualityMetrics
    after_metrics: AudioQualityMetrics
    processing_time: float = Field(description="Enhancement processing time")
    timestamp: datetime = Field(default_factory=datetime.now)

    class Config:
        json_encoders = {
            Path: str,
            datetime: lambda v: v.isoformat() if v else None
        }


class AudioAuthenticity(BaseModel):
    """Audio authenticity and manipulation detection results."""
    audio_metadata: AudioMetadata
    is_authentic: bool = Field(description="Overall authenticity assessment")
    confidence: ConfidenceLevel = Field(description="Confidence in assessment")
    manipulations_detected: List[ManipulationType] = Field(description="Detected manipulations")
    manipulation_locations: List[Dict[str, Any]] = Field(description="Manipulation timestamps")
    authenticity_score: float = Field(ge=0.0, le=1.0, description="Authenticity score")
    technical_analysis: Dict[str, Any] = Field(description="Technical analysis results")
    compression_analysis: Dict[str, Any] = Field(description="Compression artifact analysis")
    temporal_analysis: Dict[str, Any] = Field(description="Temporal consistency analysis")
    spectral_analysis: Dict[str, Any] = Field(description="Spectral analysis results")
    processing_time: float = Field(description="Analysis processing time")
    model_used: str = Field(description="Detection model used")
    timestamp: datetime = Field(default_factory=datetime.now)


class AudioAnalysisConfig(BaseModel):
    """Configuration for audio analysis operations."""
    # Transcription settings
    transcription_model: str = Field(default="whisper-large-v3")
    target_language: Optional[str] = Field(None, description="Target language for transcription")
    auto_detect_language: bool = Field(default=True)
    
    # Speaker analysis settings
    speaker_min_duration: float = Field(default=1.0, description="Minimum speaker segment duration")
    speaker_similarity_threshold: float = Field(default=0.8, description="Speaker similarity threshold")
    
    # Enhancement settings
    noise_reduction_enabled: bool = Field(default=True)
    normalize_audio: bool = Field(default=True)
    target_sample_rate: int = Field(default=16000)
    
    # Authenticity detection settings
    authenticity_threshold: float = Field(default=0.7, description="Authenticity confidence threshold")
    enable_deepfake_detection: bool = Field(default=True)
    enable_compression_analysis: bool = Field(default=True)
    
    # Processing settings
    chunk_duration: float = Field(default=30.0, description="Processing chunk duration in seconds")
    parallel_processing: bool = Field(default=True)
    preserve_originals: bool = Field(default=True)
    output_directory: Optional[Path] = Field(None, description="Output directory for results")

    class Config:
        json_encoders = {
            Path: str
        }


class AudioAnalyzer:
    """
    Main audio analysis engine providing comprehensive forensic audio processing.
    
    Features:
    - Multi-language speech transcription
    - Speaker identification and diarization
    - Audio quality enhancement
    - Authenticity and manipulation detection
    - Forensic-grade reporting with chain of custody
    """

    def __init__(self, config: Optional[AudioAnalysisConfig] = None):
        """Initialize the audio analyzer with configuration."""
        self.config = config or AudioAnalysisConfig()
        self._initialize_models()

    def _initialize_models(self):
        """Initialize ML models and processors."""
        # Lazy loading to avoid startup overhead
        self._whisper_model = None
        self._speaker_model = None
        self._enhancement_models = None
        self._authenticity_models = None

    def load_audio(self, file_path: Union[str, Path]) -> tuple[np.ndarray, AudioMetadata]:
        """
        Load and analyze audio file, extracting metadata and signal.
        
        Args:
            file_path: Path to audio file
            
        Returns:
            Tuple of (audio_signal, metadata)
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Audio file not found: {file_path}")

        # Load audio with librosa
        try:
            audio_signal, sample_rate = librosa.load(
                str(file_path), 
                sr=None,  # Keep original sample rate
                mono=False  # Preserve channel count
            )
        except Exception as e:
            raise ValueError(f"Failed to load audio file: {e}")

        # Get file info
        info = sf.info(str(file_path))
        file_stat = file_path.stat()

        # Calculate hashes for integrity
        md5_hash = self._calculate_md5(file_path)
        sha256_hash = self._calculate_sha256(file_path)

        # Extract format information
        format_name = file_path.suffix.lower().lstrip('.')
        audio_format = AudioFormat(format_name) if format_name in AudioFormat.__members__.values() else AudioFormat.WAV

        metadata = AudioMetadata(
            file_path=file_path,
            file_size=file_stat.st_size,
            duration=info.duration,
            sample_rate=info.samplerate,
            channels=info.channels,
            format=audio_format,
            created_date=datetime.fromtimestamp(file_stat.st_ctime),
            modified_date=datetime.fromtimestamp(file_stat.st_mtime),
            md5_hash=md5_hash,
            sha256_hash=sha256_hash
        )

        return audio_signal, metadata

    def calculate_quality_metrics(self, audio_signal: np.ndarray, sample_rate: int) -> AudioQualityMetrics:
        """
        Calculate comprehensive audio quality metrics.
        
        Args:
            audio_signal: Audio signal array
            sample_rate: Sample rate in Hz
            
        Returns:
            AudioQualityMetrics object
        """
        # Ensure mono for analysis
        if audio_signal.ndim > 1:
            audio_mono = librosa.to_mono(audio_signal)
        else:
            audio_mono = audio_signal

        # Basic level measurements
        peak_level = 20 * np.log10(np.max(np.abs(audio_mono)) + 1e-10)
        rms_level = 20 * np.log10(np.sqrt(np.mean(audio_mono**2)) + 1e-10)
        dynamic_range = peak_level - rms_level

        # Spectral features
        spectral_centroids = librosa.feature.spectral_centroid(y=audio_mono, sr=sample_rate)[0]
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio_mono, sr=sample_rate)[0]
        zero_crossings = librosa.zero_crossings(audio_mono, pad=False)
        zcr = np.sum(zero_crossings) / len(audio_mono)

        # MFCC features
        mfccs = librosa.feature.mfcc(y=audio_mono, sr=sample_rate, n_mfcc=13)
        mfcc_means = np.mean(mfccs, axis=1).tolist()

        # Additional spectral features
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio_mono, sr=sample_rate)[0]
        spectral_contrast = librosa.feature.spectral_contrast(y=audio_mono, sr=sample_rate)
        
        spectral_features = {
            "spectral_bandwidth_mean": float(np.mean(spectral_bandwidth)),
            "spectral_contrast_mean": float(np.mean(spectral_contrast)),
            "spectral_flatness": float(np.mean(librosa.feature.spectral_flatness(y=audio_mono)[0])),
        }

        return AudioQualityMetrics(
            dynamic_range_db=dynamic_range,
            peak_level_db=peak_level,
            rms_level_db=rms_level,
            spectral_centroid=float(np.mean(spectral_centroids)),
            spectral_rolloff=float(np.mean(spectral_rolloff)),
            zero_crossing_rate=zcr,
            mfcc_features=mfcc_means,
            spectral_features=spectral_features
        )

    def _calculate_md5(self, file_path: Path) -> str:
        """Calculate MD5 hash of file."""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

    def _calculate_sha256(self, file_path: Path) -> str:
        """Calculate SHA256 hash of file."""
        hash_sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()

    def analyze_comprehensive(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Perform comprehensive audio analysis including all available features.
        
        Args:
            file_path: Path to audio file
            
        Returns:
            Dictionary containing all analysis results
        """
        from .speech_transcriber import transcribe_audio
        from .speaker_analyzer import identify_speakers
        from .audio_enhancer import enhance_audio_quality
        from .authenticity_detector import detect_audio_manipulation

        file_path = Path(file_path)
        
        # Load audio and get metadata
        audio_signal, metadata = self.load_audio(file_path)
        
        # Calculate quality metrics
        quality_metrics = self.calculate_quality_metrics(audio_signal, metadata.sample_rate)

        results = {
            "metadata": metadata,
            "quality_metrics": quality_metrics,
            "analysis_timestamp": datetime.now(),
            "analyzer_version": "1.0.0"
        }

        # Perform transcription
        try:
            transcription = transcribe_audio(file_path, self.config.target_language)
            results["transcription"] = transcription
        except Exception as e:
            results["transcription_error"] = str(e)

        # Perform speaker analysis
        try:
            speaker_analysis = identify_speakers(file_path)
            results["speaker_analysis"] = speaker_analysis
        except Exception as e:
            results["speaker_analysis_error"] = str(e)

        # Perform enhancement (if enabled)
        if self.config.noise_reduction_enabled:
            try:
                enhanced_audio = enhance_audio_quality(file_path)
                results["enhanced_audio"] = enhanced_audio
            except Exception as e:
                results["enhancement_error"] = str(e)

        # Perform authenticity detection
        try:
            authenticity = detect_audio_manipulation(file_path)
            results["authenticity"] = authenticity
        except Exception as e:
            results["authenticity_error"] = str(e)

        return results