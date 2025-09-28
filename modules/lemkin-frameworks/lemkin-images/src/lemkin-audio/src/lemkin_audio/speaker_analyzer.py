"""
Speaker identification, diarization, and voice biometric analysis.
Provides forensic-grade speaker recognition and voice fingerprinting.
"""

import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import librosa
import numpy as np
from resemblyzer import VoiceEncoder, preprocess_wav
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist, squareform
from scipy.stats import zscore
from sklearn.cluster import AgglomerativeClustering, DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from loguru import logger

from .core import (
    AudioAnalyzer,
    AudioAnalysisConfig,
    SpeakerAnalysis,
    SpeakerProfile,
    VoicePrint
)


class SpeakerAnalyzer:
    """
    Advanced speaker analysis engine for identification and diarization.
    
    Features:
    - Speaker diarization (who spoke when)
    - Voice biometric extraction and matching
    - Speaker identification and verification
    - Voice quality assessment
    - Forensic-grade speaker profiling
    """

    def __init__(self, config: Optional[AudioAnalysisConfig] = None):
        """Initialize the speaker analyzer."""
        self.config = config or AudioAnalysisConfig()
        self._voice_encoder = None
        self._initialize_models()

    def _initialize_models(self):
        """Initialize voice encoding and analysis models."""
        try:
            logger.info("Initializing voice encoder...")
            self._voice_encoder = VoiceEncoder()
            logger.success("Voice encoder initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize voice encoder: {e}")
            raise

    def extract_voice_features(
        self, 
        audio_signal: np.ndarray, 
        sample_rate: int,
        start_time: float = 0.0,
        end_time: Optional[float] = None
    ) -> Dict[str, Union[float, List[float]]]:
        """
        Extract comprehensive voice biometric features.
        
        Args:
            audio_signal: Audio signal array
            sample_rate: Sample rate in Hz
            start_time: Start time in seconds
            end_time: End time in seconds (None for full signal)
            
        Returns:
            Dictionary of voice features
        """
        # Extract segment if specified
        if end_time is not None:
            start_idx = int(start_time * sample_rate)
            end_idx = int(end_time * sample_rate)
            audio_segment = audio_signal[start_idx:end_idx]
        else:
            audio_segment = audio_signal

        if len(audio_segment) < sample_rate * 0.5:  # Minimum 0.5 seconds
            raise ValueError("Audio segment too short for reliable feature extraction")

        features = {}

        try:
            # Fundamental frequency (pitch) analysis
            f0, voiced_flag, voiced_probs = librosa.pyin(
                audio_segment, 
                fmin=librosa.note_to_hz('C2'), 
                fmax=librosa.note_to_hz('C7'),
                sr=sample_rate
            )
            
            # Remove NaN values for statistics
            f0_clean = f0[~np.isnan(f0)]
            if len(f0_clean) > 0:
                features['pitch_mean'] = float(np.mean(f0_clean))
                features['pitch_std'] = float(np.std(f0_clean))
                features['pitch_min'] = float(np.min(f0_clean))
                features['pitch_max'] = float(np.max(f0_clean))
                features['pitch_range'] = features['pitch_max'] - features['pitch_min']
            else:
                features.update({
                    'pitch_mean': 0.0, 'pitch_std': 0.0,
                    'pitch_min': 0.0, 'pitch_max': 0.0, 'pitch_range': 0.0
                })

            # MFCC features (voice timbre)
            mfccs = librosa.feature.mfcc(y=audio_segment, sr=sample_rate, n_mfcc=13)
            features['mfcc_features'] = np.mean(mfccs, axis=1).tolist()
            features['mfcc_std'] = np.std(mfccs, axis=1).tolist()

            # Spectral features
            spectral_centroid = librosa.feature.spectral_centroid(y=audio_segment, sr=sample_rate)[0]
            spectral_rolloff = librosa.feature.spectral_rolloff(y=audio_segment, sr=sample_rate)[0]
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio_segment, sr=sample_rate)[0]
            spectral_flatness = librosa.feature.spectral_flatness(y=audio_segment)[0]
            
            features.update({
                'spectral_centroid_mean': float(np.mean(spectral_centroid)),
                'spectral_centroid_std': float(np.std(spectral_centroid)),
                'spectral_rolloff_mean': float(np.mean(spectral_rolloff)),
                'spectral_bandwidth_mean': float(np.mean(spectral_bandwidth)),
                'spectral_flatness_mean': float(np.mean(spectral_flatness)),
            })

            # Formant estimation (approximate)
            # Using spectral peaks as formant approximation
            stft = librosa.stft(audio_segment)
            magnitude = np.abs(stft)
            freqs = librosa.fft_frequencies(sr=sample_rate)
            
            # Find spectral peaks for formant estimation
            formants = []
            for frame in magnitude.T:
                peaks = np.where(frame > 0.1 * np.max(frame))[0]
                if len(peaks) >= 2:
                    # Sort by magnitude and take top frequencies
                    peak_freqs = freqs[peaks[np.argsort(frame[peaks])[-4:]]]
                    formants.append(sorted(peak_freqs)[:4])  # F1-F4
            
            if formants:
                formant_means = np.mean(formants, axis=0)
                features['formant_frequencies'] = formant_means.tolist()[:4]
            else:
                features['formant_frequencies'] = [0.0, 0.0, 0.0, 0.0]

            # Voice quality features
            zero_crossing_rate = librosa.feature.zero_crossing_rate(audio_segment)[0]
            features['zero_crossing_rate'] = float(np.mean(zero_crossing_rate))
            
            # Jitter and shimmer approximations
            if len(f0_clean) > 1:
                f0_diff = np.diff(f0_clean)
                features['jitter_estimate'] = float(np.std(f0_diff) / np.mean(f0_clean)) if np.mean(f0_clean) > 0 else 0.0
            else:
                features['jitter_estimate'] = 0.0
                
            # Energy-based features
            rms_energy = librosa.feature.rms(y=audio_segment)[0]
            features['energy_mean'] = float(np.mean(rms_energy))
            features['energy_std'] = float(np.std(rms_energy))

        except Exception as e:
            logger.error(f"Error extracting voice features: {e}")
            # Return default features in case of error
            features = {
                'pitch_mean': 0.0, 'pitch_std': 0.0, 'pitch_min': 0.0, 
                'pitch_max': 0.0, 'pitch_range': 0.0,
                'mfcc_features': [0.0] * 13,
                'mfcc_std': [0.0] * 13,
                'formant_frequencies': [0.0, 0.0, 0.0, 0.0],
                'spectral_centroid_mean': 0.0, 'spectral_centroid_std': 0.0,
                'spectral_rolloff_mean': 0.0, 'spectral_bandwidth_mean': 0.0,
                'spectral_flatness_mean': 0.0, 'zero_crossing_rate': 0.0,
                'jitter_estimate': 0.0, 'energy_mean': 0.0, 'energy_std': 0.0
            }

        return features

    def create_voice_print(
        self, 
        audio_signal: np.ndarray, 
        sample_rate: int,
        speaker_id: str,
        start_time: float = 0.0,
        end_time: Optional[float] = None
    ) -> VoicePrint:
        """
        Create a voice biometric fingerprint from audio.
        
        Args:
            audio_signal: Audio signal array
            sample_rate: Sample rate in Hz
            speaker_id: Unique speaker identifier
            start_time: Start time in seconds
            end_time: End time in seconds
            
        Returns:
            VoicePrint object
        """
        # Extract voice features
        features = self.extract_voice_features(audio_signal, sample_rate, start_time, end_time)
        
        # Extract segment for embedding
        if end_time is not None:
            start_idx = int(start_time * sample_rate)
            end_idx = int(end_time * sample_rate)
            audio_segment = audio_signal[start_idx:end_idx]
        else:
            audio_segment = audio_signal

        # Create deep learning embedding using resemblyzer
        try:
            # Preprocess audio for resemblyzer
            if sample_rate != 16000:
                audio_segment = librosa.resample(audio_segment, orig_sr=sample_rate, target_sr=16000)
            
            # Get embedding
            embedding = self._voice_encoder.embed_utterance(audio_segment)
            embedding_vector = embedding.tolist()
            
            # Calculate confidence based on audio quality
            confidence = min(1.0, len(audio_segment) / (16000 * 2.0))  # Higher for longer segments
            
        except Exception as e:
            logger.error(f"Failed to create voice embedding: {e}")
            embedding_vector = [0.0] * 256  # Default embedding size
            confidence = 0.0

        # Organize spectral and prosodic features
        spectral_features = {
            'spectral_centroid_mean': features.get('spectral_centroid_mean', 0.0),
            'spectral_centroid_std': features.get('spectral_centroid_std', 0.0),
            'spectral_rolloff_mean': features.get('spectral_rolloff_mean', 0.0),
            'spectral_bandwidth_mean': features.get('spectral_bandwidth_mean', 0.0),
            'spectral_flatness_mean': features.get('spectral_flatness_mean', 0.0),
        }
        
        prosodic_features = {
            'pitch_range': features.get('pitch_range', 0.0),
            'energy_mean': features.get('energy_mean', 0.0),
            'energy_std': features.get('energy_std', 0.0),
        }
        
        voice_quality_features = {
            'zero_crossing_rate': features.get('zero_crossing_rate', 0.0),
            'jitter_estimate': features.get('jitter_estimate', 0.0),
        }

        return VoicePrint(
            speaker_id=speaker_id,
            mfcc_features=features.get('mfcc_features', [0.0] * 13),
            pitch_mean=features.get('pitch_mean', 0.0),
            pitch_std=features.get('pitch_std', 0.0),
            formant_frequencies=features.get('formant_frequencies', [0.0, 0.0, 0.0, 0.0]),
            spectral_features=spectral_features,
            prosodic_features=prosodic_features,
            voice_quality_features=voice_quality_features,
            embedding_vector=embedding_vector,
            confidence=confidence
        )

    def detect_voice_activity(
        self, 
        audio_signal: np.ndarray, 
        sample_rate: int,
        frame_length: float = 0.025,
        frame_step: float = 0.01
    ) -> List[Tuple[float, float]]:
        """
        Detect voice activity in audio signal.
        
        Args:
            audio_signal: Audio signal array
            sample_rate: Sample rate in Hz
            frame_length: Frame length in seconds
            frame_step: Frame step in seconds
            
        Returns:
            List of (start_time, end_time) tuples for voice segments
        """
        # Simple energy-based VAD
        frame_length_samples = int(frame_length * sample_rate)
        frame_step_samples = int(frame_step * sample_rate)
        
        frames = []
        for i in range(0, len(audio_signal) - frame_length_samples, frame_step_samples):
            frame = audio_signal[i:i + frame_length_samples]
            energy = np.sum(frame ** 2)
            frames.append(energy)
        
        frames = np.array(frames)
        
        # Threshold based on statistics
        mean_energy = np.mean(frames)
        std_energy = np.std(frames)
        threshold = mean_energy + 0.5 * std_energy
        
        # Find voice segments
        voice_frames = frames > threshold
        segments = []
        
        in_voice = False
        start_frame = 0
        
        for i, is_voice in enumerate(voice_frames):
            if is_voice and not in_voice:
                start_frame = i
                in_voice = True
            elif not is_voice and in_voice:
                start_time = start_frame * frame_step
                end_time = i * frame_step
                if end_time - start_time >= self.config.speaker_min_duration:
                    segments.append((start_time, end_time))
                in_voice = False
        
        # Handle case where audio ends during voice
        if in_voice:
            start_time = start_frame * frame_step
            end_time = len(voice_frames) * frame_step
            if end_time - start_time >= self.config.speaker_min_duration:
                segments.append((start_time, end_time))
        
        return segments

    def perform_diarization(
        self, 
        audio_signal: np.ndarray, 
        sample_rate: int,
        n_speakers: Optional[int] = None
    ) -> Tuple[List[Dict], List[VoicePrint]]:
        """
        Perform speaker diarization on audio signal.
        
        Args:
            audio_signal: Audio signal array
            sample_rate: Sample rate in Hz
            n_speakers: Number of speakers (None for auto-detection)
            
        Returns:
            Tuple of (diarization_segments, voice_prints)
        """
        # Detect voice activity
        voice_segments = self.detect_voice_activity(audio_signal, sample_rate)
        
        if not voice_segments:
            logger.warning("No voice activity detected")
            return [], []

        # Extract embeddings for each voice segment
        embeddings = []
        segment_info = []
        
        for start_time, end_time in voice_segments:
            try:
                start_idx = int(start_time * sample_rate)
                end_idx = int(end_time * sample_rate)
                segment = audio_signal[start_idx:end_idx]
                
                # Resample for resemblyzer if needed
                if sample_rate != 16000:
                    segment = librosa.resample(segment, orig_sr=sample_rate, target_sr=16000)
                
                # Get embedding
                embedding = self._voice_encoder.embed_utterance(segment)
                embeddings.append(embedding)
                segment_info.append({
                    'start_time': start_time,
                    'end_time': end_time,
                    'duration': end_time - start_time
                })
                
            except Exception as e:
                logger.error(f"Failed to process segment {start_time}-{end_time}: {e}")
                continue

        if not embeddings:
            logger.error("No valid embeddings extracted")
            return [], []

        embeddings = np.array(embeddings)
        
        # Clustering to identify speakers
        if n_speakers is None:
            # Auto-detect number of speakers using silhouette score
            best_score = -1
            best_n_speakers = 2
            
            for n in range(2, min(len(embeddings) + 1, 10)):  # Try 2-9 speakers
                try:
                    clusterer = AgglomerativeClustering(n_clusters=n, linkage='ward')
                    labels = clusterer.fit_predict(embeddings)
                    
                    if len(set(labels)) > 1:  # Need at least 2 clusters for silhouette score
                        score = silhouette_score(embeddings, labels)
                        if score > best_score:
                            best_score = score
                            best_n_speakers = n
                except:
                    continue
            
            n_speakers = best_n_speakers
            logger.info(f"Auto-detected {n_speakers} speakers (silhouette score: {best_score:.3f})")

        # Final clustering
        clusterer = AgglomerativeClustering(n_clusters=n_speakers, linkage='ward')
        speaker_labels = clusterer.fit_predict(embeddings)

        # Create diarization segments and voice prints
        diarization_segments = []
        speaker_voice_prints = {}
        
        for i, (segment, label) in enumerate(zip(segment_info, speaker_labels)):
            speaker_id = f"speaker_{label}"
            
            diarization_segments.append({
                'start_time': segment['start_time'],
                'end_time': segment['end_time'],
                'speaker_id': speaker_id,
                'confidence': 0.8  # Default confidence
            })
            
            # Create voice print for this segment
            if speaker_id not in speaker_voice_prints:
                voice_print = self.create_voice_print(
                    audio_signal, sample_rate, speaker_id,
                    segment['start_time'], segment['end_time']
                )
                speaker_voice_prints[speaker_id] = voice_print

        return diarization_segments, list(speaker_voice_prints.values())

    def analyze_speakers(self, file_path: Union[str, Path]) -> SpeakerAnalysis:
        """
        Perform comprehensive speaker analysis on audio file.
        
        Args:
            file_path: Path to audio file
            
        Returns:
            SpeakerAnalysis object
        """
        analyzer = AudioAnalyzer(self.config)
        
        # Load audio and get metadata
        audio_signal, metadata = analyzer.load_audio(file_path)
        
        start_time = time.time()
        
        # Perform diarization
        diarization_segments, voice_prints = self.perform_diarization(
            audio_signal, metadata.sample_rate
        )
        
        # Create speaker profiles
        speaker_profiles = {}
        for voice_print in voice_prints:
            if voice_print.speaker_id not in speaker_profiles:
                speaker_profiles[voice_print.speaker_id] = SpeakerProfile(
                    speaker_id=voice_print.speaker_id,
                    voice_prints=[voice_print],
                    average_features={
                        'pitch_mean': voice_print.pitch_mean,
                        'pitch_std': voice_print.pitch_std,
                        **voice_print.spectral_features,
                        **voice_print.prosodic_features
                    },
                    quality_score=voice_print.confidence,
                    sample_count=1
                )
            else:
                # Add to existing profile
                profile = speaker_profiles[voice_print.speaker_id]
                profile.voice_prints.append(voice_print)
                profile.sample_count += 1
                
                # Update averaged features
                all_prints = profile.voice_prints
                profile.average_features['pitch_mean'] = np.mean([vp.pitch_mean for vp in all_prints])
                profile.average_features['pitch_std'] = np.mean([vp.pitch_std for vp in all_prints])
                profile.quality_score = np.mean([vp.confidence for vp in all_prints])

        # Extract speaker change points
        speaker_changes = []
        current_speaker = None
        for segment in diarization_segments:
            if segment['speaker_id'] != current_speaker:
                speaker_changes.append(segment['start_time'])
                current_speaker = segment['speaker_id']

        # Calculate confidence scores
        confidence_scores = {}
        for speaker_id, profile in speaker_profiles.items():
            confidence_scores[speaker_id] = profile.quality_score

        processing_time = time.time() - start_time

        return SpeakerAnalysis(
            audio_metadata=metadata,
            speakers=list(speaker_profiles.values()),
            diarization_segments=diarization_segments,
            speaker_changes=speaker_changes,
            confidence_scores=confidence_scores,
            total_speakers=len(speaker_profiles),
            processing_time=processing_time,
            model_used="resemblyzer+clustering"
        )

    def compare_voice_prints(self, voice_print1: VoicePrint, voice_print2: VoicePrint) -> float:
        """
        Compare two voice prints and return similarity score.
        
        Args:
            voice_print1: First voice print
            voice_print2: Second voice print
            
        Returns:
            Similarity score (0.0 to 1.0)
        """
        try:
            # Compare embeddings (most important)
            embed1 = np.array(voice_print1.embedding_vector)
            embed2 = np.array(voice_print2.embedding_vector)
            
            if len(embed1) > 0 and len(embed2) > 0:
                # Cosine similarity
                cosine_sim = np.dot(embed1, embed2) / (np.linalg.norm(embed1) * np.linalg.norm(embed2))
                embedding_similarity = (cosine_sim + 1) / 2  # Convert to 0-1 range
            else:
                embedding_similarity = 0.0
            
            # Compare MFCC features
            mfcc1 = np.array(voice_print1.mfcc_features)
            mfcc2 = np.array(voice_print2.mfcc_features)
            mfcc_similarity = 1.0 - np.linalg.norm(mfcc1 - mfcc2) / (np.linalg.norm(mfcc1) + np.linalg.norm(mfcc2) + 1e-10)
            
            # Compare pitch characteristics
            pitch_diff = abs(voice_print1.pitch_mean - voice_print2.pitch_mean)
            pitch_similarity = max(0.0, 1.0 - pitch_diff / 200.0)  # Normalize by 200 Hz
            
            # Weighted combination
            total_similarity = (
                0.6 * embedding_similarity +
                0.3 * mfcc_similarity +
                0.1 * pitch_similarity
            )
            
            return max(0.0, min(1.0, total_similarity))
            
        except Exception as e:
            logger.error(f"Error comparing voice prints: {e}")
            return 0.0


# Factory function for easy import
def identify_speakers(
    audio_path: Path,
    n_speakers: Optional[int] = None,
    config: Optional[AudioAnalysisConfig] = None
) -> SpeakerAnalysis:
    """
    Identify and analyze speakers in audio file.
    
    Args:
        audio_path: Path to audio file
        n_speakers: Number of speakers (None for auto-detection)
        config: Analysis configuration
        
    Returns:
        SpeakerAnalysis object with detailed results
    """
    analyzer = SpeakerAnalyzer(config)
    return analyzer.analyze_speakers(audio_path)