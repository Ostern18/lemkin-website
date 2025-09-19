"""
Core audio analysis functionality for the Lemkin Audio Analysis Toolkit.

This module provides comprehensive audio analysis capabilities including:
- Speech transcription and language detection
- Speaker identification and voice authentication
- Audio authenticity verification and tamper detection
- Audio enhancement and noise reduction
- Forensic audio analysis for legal evidence
"""

import logging
import tempfile
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from uuid import uuid4

import librosa
import numpy as np
import scipy.signal
import soundfile as sf
import speech_recognition as sr
import torch
import torchaudio
from pydantic import BaseModel, Field, validator
from scipy.fft import fft, fftfreq
from scipy.stats import entropy

logger = logging.getLogger(__name__)


class AudioFormat(str, Enum):
    """Supported audio file formats."""
    WAV = "wav"
    MP3 = "mp3"
    FLAC = "flac"
    M4A = "m4a"
    OGG = "ogg"
    OPUS = "opus"


class LanguageCode(str, Enum):
    """Supported language codes for transcription."""
    EN_US = "en-US"
    EN_GB = "en-GB"
    ES_ES = "es-ES"
    FR_FR = "fr-FR"
    DE_DE = "de-DE"
    IT_IT = "it-IT"
    PT_BR = "pt-BR"
    RU_RU = "ru-RU"
    ZH_CN = "zh-CN"
    JA_JP = "ja-JP"
    AR_SA = "ar-SA"


class SpeechSegment(BaseModel):
    """Represents a segment of transcribed speech."""
    start_time: float = Field(description="Start time in seconds")
    end_time: float = Field(description="End time in seconds")
    text: str = Field(description="Transcribed text")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score")
    speaker_id: Optional[str] = Field(default=None, description="Speaker identifier")
    language: Optional[LanguageCode] = Field(default=None, description="Detected language")


class TranscriptionResult(BaseModel):
    """Complete transcription result with metadata."""
    transcription_id: str = Field(default_factory=lambda: str(uuid4()))
    audio_path: str = Field(description="Path to source audio file")
    segments: List[SpeechSegment] = Field(description="Transcribed speech segments")
    full_text: str = Field(description="Complete transcribed text")
    total_duration: float = Field(description="Total audio duration in seconds")
    detected_language: Optional[LanguageCode] = Field(default=None)
    analysis_timestamp: datetime = Field(default_factory=datetime.utcnow)
    processing_time: float = Field(description="Processing time in seconds")


class SpeakerProfile(BaseModel):
    """Speaker voice profile for identification."""
    speaker_id: str = Field(description="Unique speaker identifier")
    voice_features: Dict[str, float] = Field(description="Extracted voice features")
    fundamental_frequency: float = Field(description="Average F0 in Hz")
    formant_frequencies: List[float] = Field(description="Formant frequencies")
    voice_quality_metrics: Dict[str, float] = Field(description="Voice quality measures")
    sample_count: int = Field(ge=1, description="Number of audio samples used")
    confidence_threshold: float = Field(ge=0.0, le=1.0, default=0.8)
    created_at: datetime = Field(default_factory=datetime.utcnow)


class SpeakerIdentification(BaseModel):
    """Result of speaker identification analysis."""
    audio_path: str = Field(description="Path to analyzed audio")
    identified_speakers: List[Dict[str, Any]] = Field(description="Identified speakers")
    speaker_segments: List[Dict[str, Any]] = Field(description="Time segments by speaker")
    total_speakers: int = Field(ge=0, description="Total number of detected speakers")
    analysis_confidence: float = Field(ge=0.0, le=1.0, description="Overall confidence")


class AuthenticityIndicator(BaseModel):
    """Individual authenticity indicator result."""
    indicator_name: str = Field(description="Name of the authenticity test")
    is_authentic: bool = Field(description="Whether this indicator suggests authenticity")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence in the result")
    details: Dict[str, Any] = Field(default_factory=dict, description="Additional details")


class AuthenticityReport(BaseModel):
    """Comprehensive audio authenticity analysis report."""
    analysis_id: str = Field(default_factory=lambda: str(uuid4()))
    audio_path: str = Field(description="Path to analyzed audio file")
    overall_authenticity: bool = Field(description="Overall authenticity assessment")
    authenticity_confidence: float = Field(ge=0.0, le=1.0, description="Overall confidence")
    indicators: List[AuthenticityIndicator] = Field(description="Individual test results")
    technical_analysis: Dict[str, Any] = Field(description="Technical analysis details")
    metadata_analysis: Dict[str, Any] = Field(description="File metadata analysis")
    tampering_detected: bool = Field(description="Whether tampering was detected")
    analysis_timestamp: datetime = Field(default_factory=datetime.utcnow)


class EnhancementSettings(BaseModel):
    """Audio enhancement processing settings."""
    noise_reduction: bool = Field(default=True, description="Apply noise reduction")
    gain_normalization: bool = Field(default=True, description="Normalize audio gain")
    frequency_filtering: bool = Field(default=False, description="Apply frequency filtering")
    low_freq_cutoff: float = Field(default=80.0, ge=0.0, description="Low frequency cutoff (Hz)")
    high_freq_cutoff: float = Field(default=8000.0, ge=0.0, description="High frequency cutoff (Hz)")
    spectral_subtraction: bool = Field(default=False, description="Apply spectral subtraction")
    echo_cancellation: bool = Field(default=False, description="Apply echo cancellation")


class EnhancementResult(BaseModel):
    """Result of audio enhancement processing."""
    enhancement_id: str = Field(default_factory=lambda: str(uuid4()))
    original_path: str = Field(description="Path to original audio file")
    enhanced_path: str = Field(description="Path to enhanced audio file")
    settings_used: EnhancementSettings = Field(description="Enhancement settings applied")
    quality_improvement: Dict[str, float] = Field(description="Quality metrics improvement")
    processing_time: float = Field(description="Processing time in seconds")
    enhancement_timestamp: datetime = Field(default_factory=datetime.utcnow)


class AudioAnalysis(BaseModel):
    """Comprehensive audio analysis results."""
    analysis_id: str = Field(default_factory=lambda: str(uuid4()))
    audio_path: str = Field(description="Path to analyzed audio file")
    file_metadata: Dict[str, Any] = Field(description="File metadata and properties")
    transcription: Optional[TranscriptionResult] = Field(default=None)
    speaker_analysis: Optional[SpeakerIdentification] = Field(default=None)
    authenticity_report: Optional[AuthenticityReport] = Field(default=None)
    enhancement_result: Optional[EnhancementResult] = Field(default=None)
    technical_specs: Dict[str, Any] = Field(description="Technical audio specifications")
    analysis_timestamp: datetime = Field(default_factory=datetime.utcnow)


class SpeechTranscriber:
    """Advanced speech transcription with multi-language support."""

    def __init__(self, model_size: str = "base", device: str = "auto"):
        """Initialize the speech transcriber.

        Args:
            model_size: Size of the Whisper model ("tiny", "base", "small", "medium", "large")
            device: Device to use ("auto", "cpu", "cuda")
        """
        self.model_size = model_size
        self.device = device
        self.recognizer = sr.Recognizer()

        # Set up device
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        logger.info(f"Initialized SpeechTranscriber with model {model_size} on {self.device}")

    def transcribe_audio(
        self,
        audio_path: Path,
        language: Optional[LanguageCode] = None,
        segment_length: float = 30.0,
        enable_timestamps: bool = True
    ) -> TranscriptionResult:
        """Transcribe audio file to text with timestamps.

        Args:
            audio_path: Path to audio file
            language: Target language for transcription
            segment_length: Length of segments for processing (seconds)
            enable_timestamps: Whether to include timestamp information

        Returns:
            TranscriptionResult with full transcription and segments
        """
        start_time = datetime.utcnow()

        try:
            # Load audio file
            audio_data, sample_rate = librosa.load(str(audio_path), sr=16000)
            total_duration = len(audio_data) / sample_rate

            logger.info(f"Transcribing audio: {audio_path} ({total_duration:.2f}s)")

            # Process in segments for better accuracy
            segments = []
            full_text_parts = []

            segment_samples = int(segment_length * sample_rate)

            for i in range(0, len(audio_data), segment_samples):
                segment_data = audio_data[i:i + segment_samples]
                start_sec = i / sample_rate
                end_sec = min((i + segment_samples) / sample_rate, total_duration)

                # Save segment to temporary file for transcription
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                    sf.write(temp_file.name, segment_data, sample_rate)

                    # Transcribe segment
                    try:
                        with sr.AudioFile(temp_file.name) as source:
                            audio_segment = self.recognizer.record(source)

                        # Use Google Speech Recognition for now
                        # In production, replace with Whisper or other advanced models
                        text = self.recognizer.recognize_google(
                            audio_segment,
                            language=language.value if language else None
                        )

                        if text.strip():
                            segment = SpeechSegment(
                                start_time=start_sec,
                                end_time=end_sec,
                                text=text.strip(),
                                confidence=0.85,  # Default confidence
                                language=language
                            )
                            segments.append(segment)
                            full_text_parts.append(text.strip())

                    except sr.UnknownValueError:
                        logger.warning(f"Could not transcribe segment {start_sec:.1f}-{end_sec:.1f}s")
                    except sr.RequestError as e:
                        logger.error(f"Transcription service error: {e}")

                    # Clean up temporary file
                    Path(temp_file.name).unlink(missing_ok=True)

            full_text = " ".join(full_text_parts)
            processing_time = (datetime.utcnow() - start_time).total_seconds()

            result = TranscriptionResult(
                audio_path=str(audio_path),
                segments=segments,
                full_text=full_text,
                total_duration=total_duration,
                detected_language=language,
                processing_time=processing_time
            )

            logger.info(f"Transcription completed: {len(segments)} segments, {len(full_text)} characters")
            return result

        except Exception as e:
            logger.error(f"Transcription failed for {audio_path}: {e}")
            raise


class SpeakerIdentifier:
    """Speaker identification and voice analysis system."""

    def __init__(self):
        """Initialize the speaker identifier."""
        self.speaker_profiles: Dict[str, SpeakerProfile] = {}
        logger.info("Initialized SpeakerIdentifier")

    def extract_voice_features(self, audio_path: Path) -> Dict[str, float]:
        """Extract voice features from audio file.

        Args:
            audio_path: Path to audio file

        Returns:
            Dictionary of extracted voice features
        """
        try:
            # Load audio
            y, sr = librosa.load(str(audio_path))

            # Extract features
            features = {}

            # Fundamental frequency (F0)
            pitches, magnitudes = librosa.piptrack(y=y, sr=sr, threshold=0.3)
            f0_values = []
            for t in range(pitches.shape[1]):
                index = magnitudes[:, t].argmax()
                pitch = pitches[index, t]
                if pitch > 0:
                    f0_values.append(pitch)

            features["f0_mean"] = np.mean(f0_values) if f0_values else 0.0
            features["f0_std"] = np.std(f0_values) if f0_values else 0.0

            # Spectral features
            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            features["spectral_centroid_mean"] = np.mean(spectral_centroids)
            features["spectral_centroid_std"] = np.std(spectral_centroids)

            # MFCCs
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            for i in range(13):
                features[f"mfcc_{i}_mean"] = np.mean(mfccs[i])
                features[f"mfcc_{i}_std"] = np.std(mfccs[i])

            # Zero crossing rate
            zcr = librosa.feature.zero_crossing_rate(y)[0]
            features["zcr_mean"] = np.mean(zcr)
            features["zcr_std"] = np.std(zcr)

            # Formant estimation (simplified)
            # In practice, use more sophisticated formant analysis
            spectrum = np.abs(fft(y))
            freqs = fftfreq(len(spectrum), 1/sr)

            # Find peaks for formant estimation
            peaks, _ = scipy.signal.find_peaks(spectrum[:len(spectrum)//2], height=np.max(spectrum)*0.1)
            formant_freqs = freqs[peaks][:5]  # First 5 formants

            features["formant_count"] = len(formant_freqs)
            for i, freq in enumerate(formant_freqs):
                features[f"formant_{i}"] = freq

            return features

        except Exception as e:
            logger.error(f"Feature extraction failed for {audio_path}: {e}")
            return {}

    def create_speaker_profile(
        self,
        speaker_id: str,
        audio_samples: List[Path],
        confidence_threshold: float = 0.8
    ) -> SpeakerProfile:
        """Create a speaker voice profile from audio samples.

        Args:
            speaker_id: Unique identifier for the speaker
            audio_samples: List of audio file paths for training
            confidence_threshold: Minimum confidence for speaker identification

        Returns:
            SpeakerProfile object
        """
        try:
            logger.info(f"Creating speaker profile for {speaker_id} from {len(audio_samples)} samples")

            all_features = []
            f0_values = []
            all_formants = []

            for audio_path in audio_samples:
                features = self.extract_voice_features(audio_path)
                if features:
                    all_features.append(features)
                    f0_values.append(features.get("f0_mean", 0.0))

                    # Collect formants
                    formants = [features.get(f"formant_{i}", 0.0) for i in range(5)]
                    all_formants.extend([f for f in formants if f > 0])

            if not all_features:
                raise ValueError("No valid features extracted from audio samples")

            # Average features across samples
            averaged_features = {}
            feature_keys = all_features[0].keys()

            for key in feature_keys:
                values = [f[key] for f in all_features if key in f]
                averaged_features[key] = np.mean(values) if values else 0.0

            # Voice quality metrics
            quality_metrics = {
                "f0_stability": 1.0 - (np.std(f0_values) / (np.mean(f0_values) + 1e-6)),
                "formant_clarity": len(all_formants) / (len(audio_samples) * 5),
                "feature_consistency": len(all_features) / len(audio_samples)
            }

            profile = SpeakerProfile(
                speaker_id=speaker_id,
                voice_features=averaged_features,
                fundamental_frequency=np.mean(f0_values),
                formant_frequencies=all_formants[:5],  # Top 5 formants
                voice_quality_metrics=quality_metrics,
                sample_count=len(audio_samples),
                confidence_threshold=confidence_threshold
            )

            self.speaker_profiles[speaker_id] = profile
            logger.info(f"Created speaker profile for {speaker_id}")

            return profile

        except Exception as e:
            logger.error(f"Failed to create speaker profile for {speaker_id}: {e}")
            raise

    def identify_speaker(
        self,
        audio_path: Path,
        known_profiles: Optional[List[str]] = None
    ) -> SpeakerIdentification:
        """Identify speakers in an audio file.

        Args:
            audio_path: Path to audio file to analyze
            known_profiles: List of speaker IDs to compare against

        Returns:
            SpeakerIdentification results
        """
        try:
            logger.info(f"Identifying speakers in {audio_path}")

            # Extract features from the audio
            features = self.extract_voice_features(audio_path)

            if not features:
                return SpeakerIdentification(
                    audio_path=str(audio_path),
                    identified_speakers=[],
                    speaker_segments=[],
                    total_speakers=0,
                    analysis_confidence=0.0
                )

            # Compare against known profiles
            profiles_to_check = known_profiles or list(self.speaker_profiles.keys())
            identified_speakers = []

            for profile_id in profiles_to_check:
                if profile_id not in self.speaker_profiles:
                    continue

                profile = self.speaker_profiles[profile_id]

                # Calculate similarity score
                similarity = self._calculate_voice_similarity(features, profile.voice_features)

                if similarity >= profile.confidence_threshold:
                    identified_speakers.append({
                        "speaker_id": profile_id,
                        "similarity_score": similarity,
                        "confidence": min(similarity * 1.2, 1.0)  # Boost confidence slightly
                    })

            # Sort by similarity score
            identified_speakers.sort(key=lambda x: x["similarity_score"], reverse=True)

            # Create basic segment (entire audio for now)
            # In practice, implement voice activity detection and speaker diarization
            audio_duration = librosa.get_duration(path=str(audio_path))
            speaker_segments = []

            if identified_speakers:
                speaker_segments.append({
                    "start_time": 0.0,
                    "end_time": audio_duration,
                    "speaker_id": identified_speakers[0]["speaker_id"],
                    "confidence": identified_speakers[0]["confidence"]
                })

            result = SpeakerIdentification(
                audio_path=str(audio_path),
                identified_speakers=identified_speakers,
                speaker_segments=speaker_segments,
                total_speakers=len(identified_speakers),
                analysis_confidence=identified_speakers[0]["confidence"] if identified_speakers else 0.0
            )

            logger.info(f"Speaker identification completed: {len(identified_speakers)} matches found")
            return result

        except Exception as e:
            logger.error(f"Speaker identification failed for {audio_path}: {e}")
            raise

    def _calculate_voice_similarity(self, features1: Dict[str, float], features2: Dict[str, float]) -> float:
        """Calculate similarity between two voice feature sets.

        Args:
            features1: First feature set
            features2: Second feature set

        Returns:
            Similarity score between 0 and 1
        """
        try:
            # Find common features
            common_features = set(features1.keys()) & set(features2.keys())

            if not common_features:
                return 0.0

            # Calculate normalized differences
            differences = []

            for feature in common_features:
                val1 = features1[feature]
                val2 = features2[feature]

                # Normalize difference
                max_val = max(abs(val1), abs(val2), 1e-6)
                diff = abs(val1 - val2) / max_val
                differences.append(1.0 - diff)  # Convert to similarity

            # Return average similarity
            return np.mean(differences)

        except Exception as e:
            logger.error(f"Similarity calculation error: {e}")
            return 0.0


class AudioAuthenticator:
    """Audio authenticity verification and tamper detection."""

    def __init__(self):
        """Initialize the audio authenticator."""
        logger.info("Initialized AudioAuthenticator")

    def verify_audio_authenticity(self, audio_path: Path) -> AuthenticityReport:
        """Perform comprehensive audio authenticity verification.

        Args:
            audio_path: Path to audio file to verify

        Returns:
            AuthenticityReport with detailed analysis
        """
        try:
            logger.info(f"Verifying authenticity of {audio_path}")

            indicators = []
            technical_analysis = {}
            metadata_analysis = {}

            # Load audio for analysis
            y, sr = librosa.load(str(audio_path))

            # 1. Continuity Analysis
            continuity_result = self._analyze_continuity(y, sr)
            indicators.append(AuthenticityIndicator(
                indicator_name="continuity_analysis",
                is_authentic=continuity_result["is_continuous"],
                confidence=continuity_result["confidence"],
                details=continuity_result
            ))

            # 2. Compression Artifact Analysis
            compression_result = self._analyze_compression_artifacts(audio_path)
            indicators.append(AuthenticityIndicator(
                indicator_name="compression_analysis",
                is_authentic=compression_result["authentic_compression"],
                confidence=compression_result["confidence"],
                details=compression_result
            ))

            # 3. Noise Floor Analysis
            noise_result = self._analyze_noise_floor(y, sr)
            indicators.append(AuthenticityIndicator(
                indicator_name="noise_floor_analysis",
                is_authentic=noise_result["consistent_noise_floor"],
                confidence=noise_result["confidence"],
                details=noise_result
            ))

            # 4. Frequency Spectrum Analysis
            spectrum_result = self._analyze_frequency_spectrum(y, sr)
            indicators.append(AuthenticityIndicator(
                indicator_name="spectrum_analysis",
                is_authentic=spectrum_result["natural_spectrum"],
                confidence=spectrum_result["confidence"],
                details=spectrum_result
            ))

            # 5. File Metadata Analysis
            metadata_analysis = self._analyze_file_metadata(audio_path)
            indicators.append(AuthenticityIndicator(
                indicator_name="metadata_analysis",
                is_authentic=metadata_analysis["metadata_consistent"],
                confidence=metadata_analysis["confidence"],
                details=metadata_analysis
            ))

            # Calculate overall authenticity
            authentic_count = sum(1 for indicator in indicators if indicator.is_authentic)
            overall_authenticity = authentic_count >= len(indicators) * 0.6  # 60% threshold

            # Calculate weighted confidence
            total_confidence = sum(indicator.confidence for indicator in indicators)
            avg_confidence = total_confidence / len(indicators) if indicators else 0.0

            # Detect tampering
            tampering_detected = any(
                not indicator.is_authentic and indicator.confidence > 0.7
                for indicator in indicators
            )

            technical_analysis = {
                "sample_rate": sr,
                "duration_seconds": len(y) / sr,
                "channels": 1,  # librosa loads as mono by default
                "bit_depth": "unknown",  # Would need file analysis
                "total_samples": len(y)
            }

            report = AuthenticityReport(
                audio_path=str(audio_path),
                overall_authenticity=overall_authenticity,
                authenticity_confidence=avg_confidence,
                indicators=indicators,
                technical_analysis=technical_analysis,
                metadata_analysis=metadata_analysis,
                tampering_detected=tampering_detected
            )

            logger.info(f"Authenticity verification completed: {'AUTHENTIC' if overall_authenticity else 'SUSPICIOUS'}")
            return report

        except Exception as e:
            logger.error(f"Authenticity verification failed for {audio_path}: {e}")
            raise

    def _analyze_continuity(self, y: np.ndarray, sr: int) -> Dict[str, Any]:
        """Analyze audio continuity for signs of splicing."""
        try:
            # Calculate short-time energy
            frame_length = int(0.025 * sr)  # 25ms frames
            hop_length = int(0.010 * sr)   # 10ms hop

            energy = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]

            # Look for sudden energy changes
            energy_diff = np.diff(energy)
            energy_threshold = np.std(energy_diff) * 3

            sudden_changes = np.sum(np.abs(energy_diff) > energy_threshold)
            total_frames = len(energy_diff)

            # Calculate continuity score
            discontinuity_ratio = sudden_changes / total_frames
            is_continuous = discontinuity_ratio < 0.05  # Less than 5% sudden changes
            confidence = 1.0 - discontinuity_ratio * 10  # Scale to confidence
            confidence = max(0.0, min(1.0, confidence))

            return {
                "is_continuous": is_continuous,
                "confidence": confidence,
                "discontinuity_ratio": discontinuity_ratio,
                "sudden_changes": int(sudden_changes),
                "total_frames": int(total_frames)
            }

        except Exception as e:
            logger.error(f"Continuity analysis error: {e}")
            return {
                "is_continuous": False,
                "confidence": 0.0,
                "error": str(e)
            }

    def _analyze_compression_artifacts(self, audio_path: Path) -> Dict[str, Any]:
        """Analyze compression artifacts in the audio file."""
        try:
            # Load with original sample rate to detect compression
            y, sr = librosa.load(str(audio_path), sr=None)

            # Analyze frequency content for compression artifacts
            stft = librosa.stft(y)
            magnitude = np.abs(stft)

            # Look for typical compression cutoffs
            freqs = librosa.fft_frequencies(sr=sr)

            # Check for high-frequency cutoff (typical in compressed audio)
            high_freq_energy = np.mean(magnitude[freqs > sr * 0.4])  # Above 40% of Nyquist
            total_energy = np.mean(magnitude)

            high_freq_ratio = high_freq_energy / (total_energy + 1e-6)

            # MP3 and other lossy formats typically cut high frequencies
            expected_ratio = 0.1  # Threshold for natural audio
            authentic_compression = high_freq_ratio > expected_ratio * 0.5

            confidence = min(high_freq_ratio / expected_ratio, 1.0)

            return {
                "authentic_compression": authentic_compression,
                "confidence": confidence,
                "high_freq_ratio": high_freq_ratio,
                "sample_rate": sr,
                "detected_cutoff_freq": None  # Could implement more sophisticated detection
            }

        except Exception as e:
            logger.error(f"Compression analysis error: {e}")
            return {
                "authentic_compression": False,
                "confidence": 0.0,
                "error": str(e)
            }

    def _analyze_noise_floor(self, y: np.ndarray, sr: int) -> Dict[str, Any]:
        """Analyze the noise floor consistency."""
        try:
            # Segment audio into chunks
            chunk_duration = 1.0  # 1 second chunks
            chunk_samples = int(chunk_duration * sr)

            noise_floors = []

            for i in range(0, len(y) - chunk_samples, chunk_samples):
                chunk = y[i:i + chunk_samples]

                # Calculate noise floor (bottom 10% of energy values)
                energy = chunk ** 2
                sorted_energy = np.sort(energy)
                noise_floor = np.mean(sorted_energy[:int(len(sorted_energy) * 0.1)])
                noise_floors.append(noise_floor)

            if not noise_floors:
                return {
                    "consistent_noise_floor": False,
                    "confidence": 0.0,
                    "error": "No valid chunks found"
                }

            # Check consistency
            noise_floor_std = np.std(noise_floors)
            noise_floor_mean = np.mean(noise_floors)

            # Coefficient of variation
            cv = noise_floor_std / (noise_floor_mean + 1e-6)

            # Consistent noise floor should have low variation
            consistent_noise_floor = cv < 0.5
            confidence = 1.0 - min(cv / 0.5, 1.0)

            return {
                "consistent_noise_floor": consistent_noise_floor,
                "confidence": confidence,
                "coefficient_of_variation": cv,
                "noise_floor_values": noise_floors,
                "chunks_analyzed": len(noise_floors)
            }

        except Exception as e:
            logger.error(f"Noise floor analysis error: {e}")
            return {
                "consistent_noise_floor": False,
                "confidence": 0.0,
                "error": str(e)
            }

    def _analyze_frequency_spectrum(self, y: np.ndarray, sr: int) -> Dict[str, Any]:
        """Analyze frequency spectrum for naturalness."""
        try:
            # Calculate power spectral density
            frequencies, psd = scipy.signal.welch(y, sr, nperseg=1024)

            # Analyze spectrum characteristics
            # Natural speech/audio typically has certain frequency distribution patterns

            # Check for unnatural peaks or gaps
            psd_db = 10 * np.log10(psd + 1e-10)

            # Look for sudden drops or peaks
            psd_diff = np.diff(psd_db)
            large_changes = np.sum(np.abs(psd_diff) > 20)  # >20dB changes

            # Check frequency distribution entropy
            psd_normalized = psd / np.sum(psd)
            spectrum_entropy = entropy(psd_normalized)

            # Natural spectrum should have reasonable entropy
            natural_spectrum = (large_changes < len(psd_diff) * 0.05 and
                              spectrum_entropy > 3.0)  # Threshold for natural audio

            confidence = min(spectrum_entropy / 5.0, 1.0)  # Scale entropy to confidence

            return {
                "natural_spectrum": natural_spectrum,
                "confidence": confidence,
                "spectrum_entropy": spectrum_entropy,
                "large_changes": int(large_changes),
                "frequency_range": [float(frequencies[0]), float(frequencies[-1])]
            }

        except Exception as e:
            logger.error(f"Spectrum analysis error: {e}")
            return {
                "natural_spectrum": False,
                "confidence": 0.0,
                "error": str(e)
            }

    def _analyze_file_metadata(self, audio_path: Path) -> Dict[str, Any]:
        """Analyze file metadata for consistency indicators."""
        try:
            # Basic file information
            file_stats = audio_path.stat()
            file_size = file_stats.st_size

            # Load audio to get technical specs
            y, sr = librosa.load(str(audio_path), sr=None)
            duration = len(y) / sr

            # Calculate expected file size for common formats
            # This is a simplified analysis - real implementation would be more sophisticated
            samples_per_second = sr
            bytes_per_sample = 2  # Assume 16-bit
            channels = 1  # Assume mono

            expected_size_wav = duration * samples_per_second * bytes_per_sample * channels
            size_ratio = file_size / expected_size_wav

            # WAV files should be close to expected size, compressed formats much smaller
            metadata_consistent = True  # Default assumption

            # Check for reasonable compression ratio
            if audio_path.suffix.lower() in ['.mp3', '.m4a', '.ogg']:
                # Compressed formats should be significantly smaller
                if size_ratio > 0.5:  # Too large for compressed format
                    metadata_consistent = False
            elif audio_path.suffix.lower() in ['.wav', '.flac']:
                # Uncompressed formats should be close to expected size
                if size_ratio < 0.8 or size_ratio > 1.2:
                    metadata_consistent = False

            confidence = 0.8 if metadata_consistent else 0.3

            return {
                "metadata_consistent": metadata_consistent,
                "confidence": confidence,
                "file_size": file_size,
                "expected_size_wav": expected_size_wav,
                "size_ratio": size_ratio,
                "file_extension": audio_path.suffix.lower(),
                "duration_seconds": duration
            }

        except Exception as e:
            logger.error(f"Metadata analysis error: {e}")
            return {
                "metadata_consistent": False,
                "confidence": 0.0,
                "error": str(e)
            }


class AudioEnhancer:
    """Audio enhancement and noise reduction for improved analysis."""

    def __init__(self):
        """Initialize the audio enhancer."""
        logger.info("Initialized AudioEnhancer")

    def enhance_audio(
        self,
        audio_path: Path,
        output_path: Optional[Path] = None,
        settings: Optional[EnhancementSettings] = None
    ) -> EnhancementResult:
        """Enhance audio quality using various processing techniques.

        Args:
            audio_path: Path to input audio file
            output_path: Path for enhanced output (optional)
            settings: Enhancement settings to apply

        Returns:
            EnhancementResult with processing information
        """
        if settings is None:
            settings = EnhancementSettings()

        if output_path is None:
            output_path = audio_path.parent / f"{audio_path.stem}_enhanced{audio_path.suffix}"

        start_time = datetime.utcnow()

        try:
            logger.info(f"Enhancing audio: {audio_path}")

            # Load audio
            y_original, sr = librosa.load(str(audio_path))
            y_enhanced = y_original.copy()

            # Quality metrics for original
            original_metrics = self._calculate_quality_metrics(y_original, sr)

            # Apply enhancement steps
            if settings.noise_reduction:
                y_enhanced = self._apply_noise_reduction(y_enhanced, sr)

            if settings.gain_normalization:
                y_enhanced = self._normalize_gain(y_enhanced)

            if settings.frequency_filtering:
                y_enhanced = self._apply_frequency_filtering(
                    y_enhanced, sr,
                    settings.low_freq_cutoff,
                    settings.high_freq_cutoff
                )

            if settings.spectral_subtraction:
                y_enhanced = self._apply_spectral_subtraction(y_enhanced, sr)

            if settings.echo_cancellation:
                y_enhanced = self._apply_echo_cancellation(y_enhanced, sr)

            # Save enhanced audio
            sf.write(str(output_path), y_enhanced, sr)

            # Calculate quality improvement
            enhanced_metrics = self._calculate_quality_metrics(y_enhanced, sr)
            quality_improvement = {
                key: enhanced_metrics[key] - original_metrics[key]
                for key in original_metrics.keys()
            }

            processing_time = (datetime.utcnow() - start_time).total_seconds()

            result = EnhancementResult(
                original_path=str(audio_path),
                enhanced_path=str(output_path),
                settings_used=settings,
                quality_improvement=quality_improvement,
                processing_time=processing_time
            )

            logger.info(f"Audio enhancement completed in {processing_time:.2f}s")
            return result

        except Exception as e:
            logger.error(f"Audio enhancement failed for {audio_path}: {e}")
            raise

    def _calculate_quality_metrics(self, y: np.ndarray, sr: int) -> Dict[str, float]:
        """Calculate audio quality metrics."""
        try:
            metrics = {}

            # Signal-to-noise ratio estimate
            # Use spectral subtraction approach to estimate noise
            stft = librosa.stft(y)
            magnitude = np.abs(stft)

            # Estimate noise floor from quieter frames
            frame_energy = np.sum(magnitude, axis=0)
            noise_threshold = np.percentile(frame_energy, 20)  # Bottom 20%
            noise_frames = magnitude[:, frame_energy <= noise_threshold]

            if noise_frames.size > 0:
                noise_spectrum = np.mean(noise_frames, axis=1)
                signal_spectrum = np.mean(magnitude, axis=1)
                snr = 10 * np.log10(np.sum(signal_spectrum) / np.sum(noise_spectrum))
                metrics["snr_db"] = snr
            else:
                metrics["snr_db"] = 0.0

            # Dynamic range
            rms = librosa.feature.rms(y=y)[0]
            dynamic_range = 20 * np.log10(np.max(rms) / (np.mean(rms) + 1e-10))
            metrics["dynamic_range_db"] = dynamic_range

            # Spectral clarity (high frequency content)
            freqs = librosa.fft_frequencies(sr=sr)
            high_freq_energy = np.mean(magnitude[freqs > sr * 0.3])
            total_energy = np.mean(magnitude)
            metrics["spectral_clarity"] = high_freq_energy / (total_energy + 1e-10)

            return metrics

        except Exception as e:
            logger.error(f"Quality metrics calculation error: {e}")
            return {"snr_db": 0.0, "dynamic_range_db": 0.0, "spectral_clarity": 0.0}

    def _apply_noise_reduction(self, y: np.ndarray, sr: int) -> np.ndarray:
        """Apply noise reduction using spectral subtraction."""
        try:
            # Simple spectral subtraction
            stft = librosa.stft(y)
            magnitude = np.abs(stft)
            phase = np.angle(stft)

            # Estimate noise spectrum from quiet segments
            frame_energy = np.sum(magnitude, axis=0)
            noise_threshold = np.percentile(frame_energy, 20)
            noise_frames = magnitude[:, frame_energy <= noise_threshold]

            if noise_frames.size > 0:
                noise_spectrum = np.mean(noise_frames, axis=1, keepdims=True)

                # Spectral subtraction with oversubtraction factor
                alpha = 2.0  # Oversubtraction factor
                enhanced_magnitude = magnitude - alpha * noise_spectrum

                # Prevent over-subtraction
                enhanced_magnitude = np.maximum(enhanced_magnitude, 0.1 * magnitude)
            else:
                enhanced_magnitude = magnitude

            # Reconstruct signal
            enhanced_stft = enhanced_magnitude * np.exp(1j * phase)
            y_enhanced = librosa.istft(enhanced_stft)

            return y_enhanced

        except Exception as e:
            logger.error(f"Noise reduction error: {e}")
            return y

    def _normalize_gain(self, y: np.ndarray) -> np.ndarray:
        """Normalize audio gain to optimal level."""
        try:
            # Calculate RMS
            rms = np.sqrt(np.mean(y ** 2))

            if rms > 1e-6:  # Avoid division by zero
                # Target RMS level (adjust as needed)
                target_rms = 0.1
                gain = target_rms / rms

                # Prevent clipping
                peak = np.max(np.abs(y))
                max_gain = 0.95 / peak
                gain = min(gain, max_gain)

                y_normalized = y * gain
            else:
                y_normalized = y

            return y_normalized

        except Exception as e:
            logger.error(f"Gain normalization error: {e}")
            return y

    def _apply_frequency_filtering(
        self,
        y: np.ndarray,
        sr: int,
        low_cutoff: float,
        high_cutoff: float
    ) -> np.ndarray:
        """Apply frequency filtering to remove unwanted frequencies."""
        try:
            # Design bandpass filter
            nyquist = sr / 2
            low = low_cutoff / nyquist
            high = min(high_cutoff / nyquist, 0.99)  # Prevent aliasing

            # Use Butterworth filter
            order = 4
            b, a = scipy.signal.butter(order, [low, high], btype='band')

            # Apply filter
            y_filtered = scipy.signal.filtfilt(b, a, y)

            return y_filtered

        except Exception as e:
            logger.error(f"Frequency filtering error: {e}")
            return y

    def _apply_spectral_subtraction(self, y: np.ndarray, sr: int) -> np.ndarray:
        """Apply advanced spectral subtraction for noise reduction."""
        try:
            # More sophisticated spectral subtraction
            stft = librosa.stft(y, hop_length=512, n_fft=2048)
            magnitude = np.abs(stft)
            phase = np.angle(stft)

            # Estimate noise using voice activity detection
            # Simple VAD based on energy threshold
            frame_energy = np.sum(magnitude ** 2, axis=0)
            energy_threshold = np.percentile(frame_energy, 30)

            # Estimate noise from low-energy frames
            noise_frames = magnitude[:, frame_energy <= energy_threshold]

            if noise_frames.size > 0:
                noise_spectrum = np.mean(noise_frames, axis=1, keepdims=True)

                # Multi-band spectral subtraction
                alpha = 2.5  # Over-subtraction factor
                beta = 0.01  # Spectral floor factor

                enhanced_magnitude = magnitude - alpha * noise_spectrum
                enhanced_magnitude = np.maximum(
                    enhanced_magnitude,
                    beta * magnitude
                )
            else:
                enhanced_magnitude = magnitude

            # Reconstruct
            enhanced_stft = enhanced_magnitude * np.exp(1j * phase)
            y_enhanced = librosa.istft(enhanced_stft, hop_length=512)

            return y_enhanced

        except Exception as e:
            logger.error(f"Spectral subtraction error: {e}")
            return y

    def _apply_echo_cancellation(self, y: np.ndarray, sr: int) -> np.ndarray:
        """Apply echo cancellation (simplified implementation)."""
        try:
            # Simple echo cancellation using delay and subtract
            # In practice, this would use more sophisticated adaptive filtering

            # Detect potential echo delay
            autocorr = np.correlate(y, y, mode='full')
            autocorr = autocorr[len(autocorr)//2:]

            # Look for peaks that might indicate echo
            peaks, _ = scipy.signal.find_peaks(autocorr[int(0.05*sr):], height=np.max(autocorr)*0.3)

            if len(peaks) > 0:
                # Use first significant peak as echo delay
                echo_delay = peaks[0] + int(0.05*sr)  # Add offset

                if echo_delay < len(y):
                    # Simple echo cancellation
                    echo_strength = 0.3  # Assume 30% echo strength
                    y_delayed = np.zeros_like(y)
                    y_delayed[echo_delay:] = y[:-echo_delay]

                    y_enhanced = y - echo_strength * y_delayed
                else:
                    y_enhanced = y
            else:
                y_enhanced = y

            return y_enhanced

        except Exception as e:
            logger.error(f"Echo cancellation error: {e}")
            return y