"""
Lemkin Audio Analysis Toolkit
============================

Professional audio evidence processing and authentication toolkit for forensic applications.

Core Components:
- Audio transcription with multi-language support
- Speaker identification and voice biometrics
- Audio quality enhancement and noise reduction
- Audio authenticity detection and manipulation analysis

Key Features:
- Forensic-grade processing with chain of custody
- Multi-language speech-to-text transcription
- Speaker diarization and identification
- Voice biometric analysis and matching
- Professional audio enhancement and cleanup
- Deepfake and manipulation detection
- Comprehensive technical reporting

Usage:
    >>> from lemkin_audio import AudioAnalyzer, transcribe_audio
    >>> analyzer = AudioAnalyzer()
    >>> results = analyzer.analyze_comprehensive("audio_file.wav")
    >>> transcription = transcribe_audio("speech.wav", language="en")

For detailed examples and documentation, see the individual module documentation.
"""

__version__ = "0.1.0"
__author__ = "Lemkin AI Contributors"
__license__ = "Apache-2.0"

# Core components
from .core import (
    AudioAnalyzer,
    AudioAnalysisConfig,
    AudioMetadata,
    AudioQualityMetrics,
    AudioTranscription,
    TranscriptionSegment,
    SpeakerAnalysis,
    SpeakerProfile,
    VoicePrint,
    EnhancedAudio,
    AudioAuthenticity,
    AudioFormat,
    ConfidenceLevel,
    ManipulationType
)

# Main analysis functions
from .speech_transcriber import (
    transcribe_audio,
    SpeechTranscriber,
    get_supported_languages,
    validate_language_code,
    SUPPORTED_LANGUAGES
)

from .speaker_analyzer import (
    identify_speakers,
    SpeakerAnalyzer
)

from .audio_enhancer import (
    enhance_audio_quality,
    AudioEnhancer
)

from .authenticity_detector import (
    detect_audio_manipulation,
    AuthenticityDetector
)

# Convenience imports
__all__ = [
    # Version info
    "__version__",
    "__author__", 
    "__license__",
    
    # Core classes
    "AudioAnalyzer",
    "AudioAnalysisConfig",
    
    # Data models
    "AudioMetadata",
    "AudioQualityMetrics", 
    "AudioTranscription",
    "TranscriptionSegment",
    "SpeakerAnalysis",
    "SpeakerProfile",
    "VoicePrint",
    "EnhancedAudio",
    "AudioAuthenticity",
    
    # Enums
    "AudioFormat",
    "ConfidenceLevel", 
    "ManipulationType",
    
    # Analysis engines
    "SpeechTranscriber",
    "SpeakerAnalyzer",
    "AudioEnhancer",
    "AuthenticityDetector",
    
    # Main functions
    "transcribe_audio",
    "identify_speakers", 
    "enhance_audio_quality",
    "detect_audio_manipulation",
    
    # Utilities
    "get_supported_languages",
    "validate_language_code",
    "SUPPORTED_LANGUAGES"
]


def get_version() -> str:
    """Get the current version of lemkin-audio."""
    return __version__


def get_system_info() -> dict:
    """Get system information for debugging and support."""
    import platform
    import sys
    
    try:
        import torch
        torch_version = torch.__version__
        cuda_available = torch.cuda.is_available()
    except ImportError:
        torch_version = "Not installed"
        cuda_available = False
    
    try:
        import librosa
        librosa_version = librosa.__version__
    except ImportError:
        librosa_version = "Not installed"
    
    return {
        "lemkin_audio_version": __version__,
        "python_version": sys.version,
        "platform": platform.platform(),
        "torch_version": torch_version,
        "cuda_available": cuda_available,
        "librosa_version": librosa_version
    }


# Module-level configuration
DEFAULT_CONFIG = AudioAnalysisConfig()


def set_default_config(config: AudioAnalysisConfig) -> None:
    """
    Set the default configuration for all analysis operations.
    
    Args:
        config: AudioAnalysisConfig object with desired settings
    """
    global DEFAULT_CONFIG
    DEFAULT_CONFIG = config


def get_default_config() -> AudioAnalysisConfig:
    """Get the current default configuration."""
    return DEFAULT_CONFIG


# Quick analysis functions for common use cases
def quick_transcribe(audio_path: str, language: str = None) -> str:
    """
    Quick transcription - returns just the text.
    
    Args:
        audio_path: Path to audio file
        language: Target language (None for auto-detection)
        
    Returns:
        Transcribed text string
    """
    result = transcribe_audio(audio_path, language)
    return result.full_text


def quick_identify_speakers(audio_path: str) -> int:
    """
    Quick speaker identification - returns just the speaker count.
    
    Args:
        audio_path: Path to audio file
        
    Returns:
        Number of unique speakers detected
    """
    result = identify_speakers(audio_path)
    return result.total_speakers


def quick_authenticity_check(audio_path: str) -> bool:
    """
    Quick authenticity check - returns True if authentic.
    
    Args:
        audio_path: Path to audio file
        
    Returns:
        True if audio appears authentic, False otherwise
    """
    result = detect_audio_manipulation(audio_path)
    return result.is_authentic


# Add quick functions to __all__
__all__.extend([
    "get_version",
    "get_system_info", 
    "set_default_config",
    "get_default_config",
    "quick_transcribe",
    "quick_identify_speakers", 
    "quick_authenticity_check"
])


# Package initialization
def _initialize_package():
    """Initialize package with logging and dependency checks."""
    import logging
    
    # Set up basic logging
    logging.getLogger("lemkin_audio").addHandler(logging.NullHandler())
    
    # Optional: Check critical dependencies
    missing_deps = []
    
    try:
        import librosa
    except ImportError:
        missing_deps.append("librosa")
    
    try:
        import torch
    except ImportError:
        missing_deps.append("torch")
    
    try:
        import whisper
    except ImportError:
        missing_deps.append("openai-whisper")
    
    if missing_deps:
        import warnings
        warnings.warn(
            f"Missing optional dependencies: {', '.join(missing_deps)}. "
            "Some functionality may be limited.",
            ImportWarning
        )


# Run initialization
_initialize_package()