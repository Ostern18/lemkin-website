"""
Multi-language speech transcription using OpenAI Whisper.
Provides forensic-grade transcription with timing, confidence, and language detection.
"""

import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Union

import librosa
import numpy as np
import torch
import whisper
from loguru import logger

from .core import (
    AudioAnalyzer,
    AudioTranscription,
    TranscriptionSegment,
    AudioAnalysisConfig
)


# Supported languages with their codes
SUPPORTED_LANGUAGES = {
    'en': 'english',
    'es': 'spanish', 
    'fr': 'french',
    'de': 'german',
    'it': 'italian',
    'pt': 'portuguese',
    'ru': 'russian',
    'ja': 'japanese',
    'ko': 'korean',
    'zh': 'chinese',
    'ar': 'arabic',
    'hi': 'hindi',
    'nl': 'dutch',
    'sv': 'swedish',
    'no': 'norwegian',
    'da': 'danish',
    'fi': 'finnish',
    'pl': 'polish',
    'tr': 'turkish',
    'th': 'thai',
    'vi': 'vietnamese',
    'uk': 'ukrainian',
    'cs': 'czech',
    'hu': 'hungarian',
    'ro': 'romanian',
    'bg': 'bulgarian',
    'hr': 'croatian',
    'sk': 'slovak',
    'sl': 'slovenian',
    'et': 'estonian',
    'lv': 'latvian',
    'lt': 'lithuanian',
    'mt': 'maltese',
    'cy': 'welsh',
    'ga': 'irish',
    'eu': 'basque',
    'ca': 'catalan',
    'gl': 'galician'
}


class SpeechTranscriber:
    """
    Advanced speech transcription engine using Whisper models.
    
    Features:
    - Multi-language support with auto-detection
    - Word-level timestamps and confidence scores
    - Speaker-aware transcription integration
    - Forensic-quality output with chain of custody
    - Support for various Whisper model sizes
    """

    def __init__(self, model_name: str = "large-v3", device: Optional[str] = None):
        """
        Initialize the speech transcriber.
        
        Args:
            model_name: Whisper model size ('tiny', 'base', 'small', 'medium', 'large', 'large-v2', 'large-v3')
            device: Device to run on ('cpu', 'cuda', or None for auto-detect)
        """
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model = None
        self._load_model()

    def _load_model(self):
        """Load the Whisper model."""
        try:
            logger.info(f"Loading Whisper model '{self.model_name}' on device '{self.device}'")
            self._model = whisper.load_model(self.model_name, device=self.device)
            logger.success(f"Successfully loaded Whisper model '{self.model_name}'")
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}")
            raise

    def detect_language(self, audio_signal: np.ndarray, sample_rate: int = 16000) -> tuple[str, float]:
        """
        Detect the primary language in the audio.
        
        Args:
            audio_signal: Audio signal array
            sample_rate: Sample rate in Hz
            
        Returns:
            Tuple of (language_code, confidence)
        """
        if self._model is None:
            raise RuntimeError("Model not loaded")

        # Ensure correct format for Whisper
        if sample_rate != 16000:
            audio_signal = librosa.resample(audio_signal, orig_sr=sample_rate, target_sr=16000)

        # Use a snippet for language detection (first 30 seconds)
        snippet_length = min(len(audio_signal), 16000 * 30)
        audio_snippet = audio_signal[:snippet_length]

        try:
            # Detect language
            mel = whisper.log_mel_spectrogram(audio_snippet).to(self._model.device)
            _, probs = self._model.detect_language(mel)
            
            # Get the most likely language
            detected_language = max(probs, key=probs.get)
            confidence = probs[detected_language]
            
            logger.info(f"Detected language: {detected_language} (confidence: {confidence:.3f})")
            return detected_language, confidence
            
        except Exception as e:
            logger.error(f"Language detection failed: {e}")
            return "en", 0.5  # Default to English with low confidence

    def transcribe_with_timestamps(
        self, 
        audio_signal: np.ndarray, 
        sample_rate: int = 16000,
        language: Optional[str] = None,
        word_timestamps: bool = True
    ) -> Dict:
        """
        Transcribe audio with detailed timestamps and confidence scores.
        
        Args:
            audio_signal: Audio signal array
            sample_rate: Sample rate in Hz
            language: Target language code (None for auto-detection)
            word_timestamps: Whether to extract word-level timestamps
            
        Returns:
            Detailed transcription results
        """
        if self._model is None:
            raise RuntimeError("Model not loaded")

        # Ensure correct format for Whisper
        if sample_rate != 16000:
            audio_signal = librosa.resample(audio_signal, orig_sr=sample_rate, target_sr=16000)

        # Prepare transcription options
        options = {
            "word_timestamps": word_timestamps,
            "task": "transcribe",
            "fp16": torch.cuda.is_available(),
        }

        if language:
            if language in SUPPORTED_LANGUAGES:
                options["language"] = language
            else:
                logger.warning(f"Language '{language}' not supported, using auto-detection")

        try:
            logger.info("Starting transcription...")
            start_time = time.time()
            
            # Perform transcription
            result = self._model.transcribe(audio_signal, **options)
            
            processing_time = time.time() - start_time
            logger.success(f"Transcription completed in {processing_time:.2f} seconds")
            
            return result
            
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            raise

    def process_transcription_result(
        self, 
        result: Dict, 
        audio_metadata,
        processing_time: float
    ) -> AudioTranscription:
        """
        Process raw Whisper results into structured transcription data.
        
        Args:
            result: Raw Whisper transcription result
            audio_metadata: AudioMetadata object
            processing_time: Processing time in seconds
            
        Returns:
            AudioTranscription object
        """
        segments = []
        full_text_parts = []

        for segment_data in result.get("segments", []):
            # Extract word-level details if available
            words = []
            if "words" in segment_data:
                words = [
                    {
                        "word": word.get("word", ""),
                        "start": word.get("start", 0.0),
                        "end": word.get("end", 0.0),
                        "confidence": word.get("probability", 0.0)
                    }
                    for word in segment_data["words"]
                ]

            segment = TranscriptionSegment(
                start_time=segment_data.get("start", 0.0),
                end_time=segment_data.get("end", 0.0),
                text=segment_data.get("text", "").strip(),
                confidence=segment_data.get("avg_logprob", 0.0),  # Convert logprob to confidence
                language=result.get("language", "unknown"),
                words=words
            )
            segments.append(segment)
            full_text_parts.append(segment.text)

        # Calculate overall confidence
        if segments:
            overall_confidence = np.mean([seg.confidence for seg in segments])
        else:
            overall_confidence = 0.0

        return AudioTranscription(
            audio_metadata=audio_metadata,
            segments=segments,
            full_text=" ".join(full_text_parts),
            language=result.get("language", "unknown"),
            confidence=max(0.0, min(1.0, overall_confidence)),  # Normalize to [0,1]
            processing_time=processing_time,
            model_used=self.model_name
        )

    def transcribe_file(
        self, 
        file_path: Union[str, Path], 
        language: Optional[str] = None,
        config: Optional[AudioAnalysisConfig] = None
    ) -> AudioTranscription:
        """
        Transcribe an entire audio file.
        
        Args:
            file_path: Path to audio file
            language: Target language (None for auto-detection)
            config: Analysis configuration
            
        Returns:
            AudioTranscription object
        """
        analyzer = AudioAnalyzer(config)
        
        # Load audio and get metadata
        audio_signal, metadata = analyzer.load_audio(file_path)
        
        start_time = time.time()
        
        # Auto-detect language if not specified
        if not language and (not config or config.auto_detect_language):
            detected_lang, confidence = self.detect_language(audio_signal, metadata.sample_rate)
            if confidence > 0.7:  # Use detection if confident
                language = detected_lang
                logger.info(f"Using auto-detected language: {language}")

        # Perform transcription
        result = self.transcribe_with_timestamps(
            audio_signal, 
            metadata.sample_rate,
            language=language
        )
        
        processing_time = time.time() - start_time
        
        # Process and structure results
        return self.process_transcription_result(result, metadata, processing_time)

    def transcribe_chunk(
        self, 
        audio_signal: np.ndarray, 
        sample_rate: int,
        start_offset: float = 0.0,
        language: Optional[str] = None
    ) -> List[TranscriptionSegment]:
        """
        Transcribe a chunk of audio with offset adjustment.
        
        Args:
            audio_signal: Audio signal chunk
            sample_rate: Sample rate in Hz
            start_offset: Time offset for this chunk in seconds
            language: Target language
            
        Returns:
            List of TranscriptionSegment objects
        """
        result = self.transcribe_with_timestamps(audio_signal, sample_rate, language)
        
        segments = []
        for segment_data in result.get("segments", []):
            # Adjust timestamps with offset
            words = []
            if "words" in segment_data:
                words = [
                    {
                        "word": word.get("word", ""),
                        "start": word.get("start", 0.0) + start_offset,
                        "end": word.get("end", 0.0) + start_offset,
                        "confidence": word.get("probability", 0.0)
                    }
                    for word in segment_data["words"]
                ]

            segment = TranscriptionSegment(
                start_time=segment_data.get("start", 0.0) + start_offset,
                end_time=segment_data.get("end", 0.0) + start_offset,
                text=segment_data.get("text", "").strip(),
                confidence=segment_data.get("avg_logprob", 0.0),
                language=result.get("language", "unknown"),
                words=words
            )
            segments.append(segment)
        
        return segments

    def batch_transcribe(
        self, 
        file_paths: List[Union[str, Path]],
        language: Optional[str] = None,
        config: Optional[AudioAnalysisConfig] = None
    ) -> List[AudioTranscription]:
        """
        Transcribe multiple audio files in batch.
        
        Args:
            file_paths: List of audio file paths
            language: Target language (None for auto-detection)
            config: Analysis configuration
            
        Returns:
            List of AudioTranscription objects
        """
        results = []
        
        for file_path in file_paths:
            try:
                logger.info(f"Transcribing {file_path}")
                transcription = self.transcribe_file(file_path, language, config)
                results.append(transcription)
                logger.success(f"Successfully transcribed {file_path}")
            except Exception as e:
                logger.error(f"Failed to transcribe {file_path}: {e}")
                # Create error transcription
                analyzer = AudioAnalyzer(config)
                _, metadata = analyzer.load_audio(file_path)
                error_transcription = AudioTranscription(
                    audio_metadata=metadata,
                    segments=[],
                    full_text=f"[ERROR: {str(e)}]",
                    language="unknown",
                    confidence=0.0,
                    processing_time=0.0,
                    model_used=self.model_name
                )
                results.append(error_transcription)
        
        return results


# Factory function for easy import
def transcribe_audio(
    audio_path: Path, 
    language: Optional[str] = None,
    model_name: str = "large-v3",
    config: Optional[AudioAnalysisConfig] = None
) -> AudioTranscription:
    """
    Transcribe audio file with multi-language support.
    
    Args:
        audio_path: Path to audio file
        language: Target language code (None for auto-detection)
        model_name: Whisper model to use
        config: Analysis configuration
        
    Returns:
        AudioTranscription object with detailed results
    """
    transcriber = SpeechTranscriber(model_name=model_name)
    return transcriber.transcribe_file(audio_path, language, config)


def get_supported_languages() -> Dict[str, str]:
    """Get dictionary of supported language codes and names."""
    return SUPPORTED_LANGUAGES.copy()


def validate_language_code(language_code: str) -> bool:
    """Validate if language code is supported."""
    return language_code in SUPPORTED_LANGUAGES