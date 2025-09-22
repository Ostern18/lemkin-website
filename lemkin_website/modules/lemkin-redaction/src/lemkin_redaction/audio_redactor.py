"""
Audio-based PII redaction for voice anonymization and content removal.

This module provides automated detection and redaction of personally
identifiable information in audio content, including voice anonymization.
"""

import hashlib
import time
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

import librosa
import soundfile as sf
import numpy as np
from loguru import logger

from .core import (
    PIIEntity, EntityType, ConfidenceLevel, RedactionConfig,
    RedactionResult, RedactionType
)


class AudioRedactor:
    """Audio-based PII redaction using signal processing."""
    
    def __init__(self, config: RedactionConfig):
        """Initialize audio redactor with configuration."""
        self.config = config
        self.logger = logger
        
        # Audio processing parameters
        self.sample_rate = 22050
        self.hop_length = 512
        self.win_length = 1024
        
        self.logger.info("AudioRedactor initialized")
    
    def detect_speech_segments(self, audio: np.ndarray, sr: int) -> List[PIIEntity]:
        """Detect speech segments in audio that may contain PII."""
        entities = []
        
        try:
            # Use spectral features to detect speech
            # This is a simplified approach - production would use speech detection models
            
            # Calculate spectral centroid
            spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
            
            # Calculate RMS energy
            rms = librosa.feature.rms(y=audio)[0]
            
            # Simple voice activity detection based on energy and spectral content
            frame_length = len(audio) // len(rms)
            
            for i, (centroid, energy) in enumerate(zip(spectral_centroids, rms)):
                # Thresholds for speech detection (simplified)
                if energy > 0.01 and 1000 < centroid < 5000:  # Typical speech range
                    start_time = i * frame_length / sr
                    end_time = (i + 1) * frame_length / sr
                    
                    confidence = min(0.8, energy * 10)  # Simple confidence based on energy
                    
                    entity = PIIEntity(
                        entity_type=EntityType.PERSON,  # Speech implies person
                        text=f"speech_segment_{i}",
                        start_pos=int(start_time * sr),
                        end_pos=int(end_time * sr),
                        confidence=confidence,
                        confidence_level=self._get_confidence_level(confidence),
                        metadata={
                            "source": "speech_detection",
                            "start_time": start_time,
                            "end_time": end_time,
                            "energy": float(energy),
                            "spectral_centroid": float(centroid)
                        }
                    )
                    entities.append(entity)
                    
        except Exception as e:
            self.logger.error(f"Speech segment detection failed: {e}")
            
        return entities
    
    def detect_silence_segments(self, audio: np.ndarray, sr: int) -> List[Tuple[float, float]]:
        """Detect silence segments for audio processing."""
        # Simple silence detection based on energy threshold
        frame_length = 2048
        hop_length = 1024
        
        # Calculate RMS energy
        rms = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
        
        # Threshold for silence (relative to max energy)
        silence_threshold = 0.01 * np.max(rms)
        
        silence_segments = []
        in_silence = False
        silence_start = 0
        
        for i, energy in enumerate(rms):
            time_pos = i * hop_length / sr
            
            if energy < silence_threshold and not in_silence:
                # Start of silence
                in_silence = True
                silence_start = time_pos
            elif energy >= silence_threshold and in_silence:
                # End of silence
                in_silence = False
                silence_segments.append((silence_start, time_pos))
        
        return silence_segments
    
    def apply_voice_anonymization(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Apply voice anonymization using pitch shifting and formant modification."""
        try:
            # Simple pitch shifting for voice anonymization
            # Production version would use more sophisticated voice conversion
            
            # Random pitch shift between -3 to +3 semitones
            pitch_shift = np.random.uniform(-3, 3)
            
            # Apply pitch shift
            anonymized = librosa.effects.pitch_shift(audio, sr=sr, n_steps=pitch_shift)
            
            # Add slight formant shifting by spectral manipulation
            stft = librosa.stft(anonymized)
            
            # Modify formants by frequency warping
            freq_warp = np.random.uniform(0.9, 1.1)
            magnitude = np.abs(stft)
            phase = np.angle(stft)
            
            # Simple frequency domain manipulation
            warped_magnitude = np.zeros_like(magnitude)
            for i in range(magnitude.shape[0]):
                new_idx = int(i * freq_warp)
                if new_idx < magnitude.shape[0]:
                    warped_magnitude[new_idx] = magnitude[i]
            
            # Reconstruct audio
            warped_stft = warped_magnitude * np.exp(1j * phase)
            anonymized = librosa.istft(warped_stft)
            
            return anonymized
            
        except Exception as e:
            self.logger.error(f"Voice anonymization failed: {e}")
            return audio
    
    def apply_redaction(self, audio: np.ndarray, entities: List[PIIEntity], sr: int) -> Tuple[np.ndarray, List[PIIEntity]]:
        """Apply redaction to audio based on detected entities."""
        redacted_audio = audio.copy()
        redacted_entities = []
        
        for entity in entities:
            # Check if entity meets confidence threshold
            if entity.confidence < self.config.min_confidence:
                continue
            
            # Get redaction method for this entity type
            redaction_method = self.config.redaction_methods.get(
                entity.entity_type, RedactionType.ANONYMIZE
            )
            
            # Get audio segment
            start_sample = entity.start_pos
            end_sample = entity.end_pos
            
            if redaction_method == RedactionType.ANONYMIZE:
                # Apply voice anonymization to segment
                segment = redacted_audio[start_sample:end_sample]
                anonymized_segment = self.apply_voice_anonymization(segment, sr)
                redacted_audio[start_sample:end_sample] = anonymized_segment
                entity.replacement = "anonymized_voice"
                
            elif redaction_method == RedactionType.DELETE:
                # Replace with silence
                redacted_audio[start_sample:end_sample] = 0
                entity.replacement = "silence"
                
            elif redaction_method == RedactionType.MASK:
                # Replace with white noise
                noise = np.random.normal(0, 0.01, end_sample - start_sample)
                redacted_audio[start_sample:end_sample] = noise
                entity.replacement = "white_noise"
            
            redacted_entities.append(entity)
        
        return redacted_audio, redacted_entities
    
    def redact(self, audio_path: Path, output_path: Optional[Path] = None) -> RedactionResult:
        """
        Main redaction method for audio content.
        
        Args:
            audio_path: Path to audio file
            output_path: Optional path to save redacted audio
            
        Returns:
            RedactionResult with processing details
        """
        start_time = time.time()
        
        # Load audio
        audio, sr = librosa.load(str(audio_path), sr=self.sample_rate)
        
        # Generate content hash for integrity
        with open(audio_path, 'rb') as f:
            content_hash = hashlib.sha256(f.read()).hexdigest()
        
        # Detect entities
        speech_entities = self.detect_speech_segments(audio, sr)
        
        # Apply redaction
        redacted_audio, redacted_entities = self.apply_redaction(audio, speech_entities, sr)
        
        # Save redacted audio if output path provided
        if output_path:
            sf.write(str(output_path), redacted_audio, sr)
        
        # Calculate statistics
        processing_time = time.time() - start_time
        confidence_scores = self._calculate_confidence_scores(speech_entities)
        
        # Create result
        result = RedactionResult(
            original_content_hash=content_hash,
            content_type="audio",
            entities_detected=speech_entities,
            entities_redacted=redacted_entities,
            total_entities=len(speech_entities),
            redacted_count=len(redacted_entities),
            confidence_scores=confidence_scores,
            processing_time=processing_time,
            config_used=self.config,
            redacted_content_path=str(output_path) if output_path else None,
            redaction_quality={
                "duration_seconds": len(audio) / sr,
                "sample_rate": sr,
                "redacted_duration": sum([
                    (e.end_pos - e.start_pos) / sr for e in redacted_entities
                ])
            }
        )
        
        return result
    
    def _get_confidence_level(self, confidence: float) -> ConfidenceLevel:
        """Convert numeric confidence to confidence level."""
        if confidence >= 0.9:
            return ConfidenceLevel.VERY_HIGH
        elif confidence >= 0.75:
            return ConfidenceLevel.HIGH
        elif confidence >= 0.5:
            return ConfidenceLevel.MEDIUM
        else:
            return ConfidenceLevel.LOW
    
    def _calculate_confidence_scores(self, entities: List[PIIEntity]) -> Dict[str, float]:
        """Calculate average confidence scores by entity type."""
        scores = {}
        type_counts = {}
        
        for entity in entities:
            entity_type = entity.entity_type.value
            if entity_type not in scores:
                scores[entity_type] = 0.0
                type_counts[entity_type] = 0
            
            scores[entity_type] += entity.confidence
            type_counts[entity_type] += 1
        
        # Calculate averages
        for entity_type in scores:
            if type_counts[entity_type] > 0:
                scores[entity_type] /= type_counts[entity_type]
        
        return scores