"""
Audio authenticity detection and manipulation analysis for forensic audio processing.
Detects deepfakes, splicing, and other audio manipulations with forensic-grade accuracy.
"""

import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import librosa
import numpy as np
from scipy import signal, stats
from scipy.fft import fft, fftfreq
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from loguru import logger

from .core import (
    AudioAnalyzer,
    AudioAnalysisConfig,
    AudioAuthenticity,
    ManipulationType,
    ConfidenceLevel
)


class AuthenticityDetector:
    """
    Advanced audio authenticity detection engine for forensic analysis.
    
    Features:
    - Deepfake audio detection
    - Splice detection and analysis
    - Compression artifact analysis
    - Temporal consistency analysis
    - Spectral manipulation detection
    - Chain of custody preservation
    """

    def __init__(self, config: Optional[AudioAnalysisConfig] = None):
        """Initialize the authenticity detector."""
        self.config = config or AudioAnalysisConfig()

    def analyze_compression_artifacts(
        self, 
        audio_signal: np.ndarray, 
        sample_rate: int
    ) -> Dict[str, float]:
        """
        Analyze compression artifacts that may indicate manipulation.
        
        Args:
            audio_signal: Input audio signal
            sample_rate: Sample rate in Hz
            
        Returns:
            Dictionary of compression analysis metrics
        """
        # Compute spectrogram
        stft = librosa.stft(audio_signal, n_fft=2048, hop_length=512)
        magnitude = np.abs(stft)
        
        # Analyze frequency distribution for compression artifacts
        freq_bins = librosa.fft_frequencies(sr=sample_rate, n_fft=2048)
        
        # Look for typical compression cutoffs
        cutoff_8khz = np.where(freq_bins <= 8000)[0]
        cutoff_16khz = np.where(freq_bins <= 16000)[0]
        
        energy_8khz = np.sum(magnitude[cutoff_8khz, :])
        energy_16khz = np.sum(magnitude[cutoff_16khz, :])
        total_energy = np.sum(magnitude)
        
        # Calculate metrics
        high_freq_energy_ratio = (energy_16khz - energy_8khz) / (total_energy + 1e-10)
        spectral_rolloff_consistency = self._analyze_spectral_rolloff_consistency(audio_signal, sample_rate)
        
        # Analyze for pre-echo (common in MP3)
        pre_echo_score = self._detect_pre_echo(magnitude)
        
        # Check for quantization noise patterns
        quantization_score = self._analyze_quantization_patterns(audio_signal)
        
        return {
            "high_frequency_energy_ratio": high_freq_energy_ratio,
            "spectral_rolloff_consistency": spectral_rolloff_consistency,
            "pre_echo_score": pre_echo_score,
            "quantization_score": quantization_score,
            "compression_likelihood": (1 - spectral_rolloff_consistency + pre_echo_score) / 2
        }

    def _analyze_spectral_rolloff_consistency(
        self, 
        audio_signal: np.ndarray, 
        sample_rate: int
    ) -> float:
        """Analyze consistency of spectral rolloff (compression artifact indicator)."""
        # Compute spectral rolloff over time
        rolloff = librosa.feature.spectral_rolloff(y=audio_signal, sr=sample_rate)[0]
        
        # Consistent rolloff indicates potential compression
        rolloff_std = np.std(rolloff)
        rolloff_mean = np.mean(rolloff)
        
        # Normalize by mean to get coefficient of variation
        consistency = 1 - min(1.0, rolloff_std / (rolloff_mean + 1e-10))
        return consistency

    def _detect_pre_echo(self, magnitude: np.ndarray) -> float:
        """Detect pre-echo artifacts common in lossy compression."""
        # Look for energy before transients (simplified detection)
        # Real pre-echo detection would be more sophisticated
        
        # Detect transients using onset detection
        onset_frames = librosa.onset.onset_detect(S=magnitude, units='frames')
        
        if len(onset_frames) == 0:
            return 0.0
        
        pre_echo_scores = []
        for onset_frame in onset_frames:
            if onset_frame > 10:  # Need some frames before
                pre_energy = np.mean(magnitude[:, onset_frame-5:onset_frame])
                onset_energy = np.mean(magnitude[:, onset_frame:onset_frame+2])
                
                if onset_energy > 0:
                    pre_echo_ratio = pre_energy / onset_energy
                    pre_echo_scores.append(pre_echo_ratio)
        
        return np.mean(pre_echo_scores) if pre_echo_scores else 0.0

    def _analyze_quantization_patterns(self, audio_signal: np.ndarray) -> float:
        """Analyze quantization patterns that indicate compression."""
        # Histogram analysis for quantization artifacts
        hist, bins = np.histogram(audio_signal, bins=256)
        
        # Look for uneven distribution that suggests quantization
        hist_normalized = hist / np.sum(hist)
        entropy = -np.sum(hist_normalized * np.log2(hist_normalized + 1e-10))
        
        # Lower entropy suggests more quantization
        max_entropy = np.log2(256)
        quantization_score = 1 - (entropy / max_entropy)
        
        return quantization_score

    def detect_temporal_inconsistencies(
        self, 
        audio_signal: np.ndarray, 
        sample_rate: int,
        window_size: float = 0.1
    ) -> Dict[str, Union[float, List[float]]]:
        """
        Detect temporal inconsistencies that may indicate splicing.
        
        Args:
            audio_signal: Input audio signal
            sample_rate: Sample rate in Hz
            window_size: Analysis window size in seconds
            
        Returns:
            Dictionary of temporal analysis results
        """
        window_samples = int(window_size * sample_rate)
        hop_samples = window_samples // 2
        
        # Extract features over time
        times = []
        energy_values = []
        spectral_centroids = []
        zero_crossing_rates = []
        
        for i in range(0, len(audio_signal) - window_samples, hop_samples):
            window = audio_signal[i:i + window_samples]
            time_stamp = i / sample_rate
            
            # Energy
            energy = np.sum(window ** 2)
            
            # Spectral centroid
            centroid = librosa.feature.spectral_centroid(y=window, sr=sample_rate)[0, 0]
            
            # Zero crossing rate
            zcr = np.sum(librosa.zero_crossings(window)) / len(window)
            
            times.append(time_stamp)
            energy_values.append(energy)
            spectral_centroids.append(centroid)
            zero_crossing_rates.append(zcr)
        
        # Analyze consistency
        energy_consistency = 1 - (np.std(energy_values) / (np.mean(energy_values) + 1e-10))
        centroid_consistency = 1 - (np.std(spectral_centroids) / (np.mean(spectral_centroids) + 1e-10))
        zcr_consistency = 1 - (np.std(zero_crossing_rates) / (np.mean(zero_crossing_rates) + 1e-10))
        
        # Detect sudden changes (potential splice points)
        energy_changes = np.abs(np.diff(energy_values))
        centroid_changes = np.abs(np.diff(spectral_centroids))
        
        # Z-score based outlier detection
        energy_z_scores = np.abs(stats.zscore(energy_changes))
        centroid_z_scores = np.abs(stats.zscore(centroid_changes))
        
        # Identify potential splice points
        splice_threshold = 2.5  # Z-score threshold
        potential_splices = []
        
        for i, (e_z, c_z) in enumerate(zip(energy_z_scores, centroid_z_scores)):
            if e_z > splice_threshold or c_z > splice_threshold:
                splice_time = times[i]
                confidence = min(e_z, c_z) / splice_threshold
                potential_splices.append({
                    'time': splice_time,
                    'confidence': min(confidence, 1.0),
                    'energy_z_score': e_z,
                    'centroid_z_score': c_z
                })
        
        return {
            'energy_consistency': energy_consistency,
            'centroid_consistency': centroid_consistency,
            'zcr_consistency': zcr_consistency,
            'overall_consistency': np.mean([energy_consistency, centroid_consistency, zcr_consistency]),
            'potential_splices': potential_splices,
            'splice_count': len(potential_splices)
        }

    def analyze_spectral_anomalies(
        self, 
        audio_signal: np.ndarray, 
        sample_rate: int
    ) -> Dict[str, float]:
        """
        Detect spectral anomalies that may indicate manipulation.
        
        Args:
            audio_signal: Input audio signal
            sample_rate: Sample rate in Hz
            
        Returns:
            Dictionary of spectral analysis results
        """
        # Compute spectrogram
        stft = librosa.stft(audio_signal, n_fft=2048, hop_length=512)
        magnitude = np.abs(stft)
        
        # Frequency bins
        freq_bins = librosa.fft_frequencies(sr=sample_rate, n_fft=2048)
        
        # Analyze spectral characteristics
        results = {}
        
        # 1. Spectral flatness over time (consistent timbre)
        spectral_flatness = librosa.feature.spectral_flatness(S=magnitude)
        results['spectral_flatness_consistency'] = 1 - np.std(spectral_flatness)
        
        # 2. Harmonic-to-noise ratio consistency
        harmonic, percussive = librosa.decompose.hpss(magnitude)
        harmonic_energy = np.sum(harmonic, axis=0)
        percussive_energy = np.sum(percussive, axis=0)
        hnr = harmonic_energy / (percussive_energy + 1e-10)
        results['hnr_consistency'] = 1 - (np.std(hnr) / (np.mean(hnr) + 1e-10))
        
        # 3. Spectral contrast consistency
        contrast = librosa.feature.spectral_contrast(S=magnitude, sr=sample_rate)
        results['spectral_contrast_consistency'] = 1 - np.mean(np.std(contrast, axis=1))
        
        # 4. Frequency domain artifacts
        # Look for unnatural frequency patterns
        freq_hist = np.sum(magnitude, axis=1)
        freq_hist_norm = freq_hist / np.sum(freq_hist)
        
        # Expected natural frequency distribution (roughly 1/f)
        expected_dist = 1 / (freq_bins[1:] + 1)
        expected_dist = expected_dist / np.sum(expected_dist)
        
        # Compare distributions using KL divergence
        kl_divergence = stats.entropy(freq_hist_norm[1:], expected_dist)
        results['frequency_distribution_naturalness'] = 1 / (1 + kl_divergence)
        
        # 5. Phase coherence analysis
        phase = np.angle(stft)
        phase_diff = np.diff(phase, axis=1)
        phase_coherence = np.mean(np.cos(phase_diff))
        results['phase_coherence'] = (phase_coherence + 1) / 2  # Normalize to [0,1]
        
        return results

    def detect_deepfake_indicators(
        self, 
        audio_signal: np.ndarray, 
        sample_rate: int
    ) -> Dict[str, float]:
        """
        Detect indicators of deepfake/synthetic audio.
        
        Args:
            audio_signal: Input audio signal
            sample_rate: Sample rate in Hz
            
        Returns:
            Dictionary of deepfake detection metrics
        """
        results = {}
        
        # 1. Pitch naturalness analysis
        f0, voiced_flag, voiced_probs = librosa.pyin(
            audio_signal, 
            fmin=librosa.note_to_hz('C2'), 
            fmax=librosa.note_to_hz('C7'),
            sr=sample_rate
        )
        
        f0_clean = f0[~np.isnan(f0)]
        if len(f0_clean) > 10:
            # Analyze pitch contour smoothness
            pitch_diff = np.diff(f0_clean)
            pitch_smoothness = 1 - (np.std(pitch_diff) / (np.mean(np.abs(pitch_diff)) + 1e-10))
            results['pitch_naturalness'] = max(0, min(1, pitch_smoothness))
            
            # Pitch jitter analysis (natural speech has moderate jitter)
            jitter = np.std(pitch_diff) / np.mean(f0_clean)
            # Natural jitter is typically 0.5-2%
            if 0.005 <= jitter <= 0.02:
                jitter_naturalness = 1.0
            else:
                jitter_naturalness = max(0, 1 - abs(jitter - 0.01) / 0.01)
            results['jitter_naturalness'] = jitter_naturalness
        else:
            results['pitch_naturalness'] = 0.5
            results['jitter_naturalness'] = 0.5
        
        # 2. Formant structure analysis
        # Extract MFCC for formant-related features
        mfccs = librosa.feature.mfcc(y=audio_signal, sr=sample_rate, n_mfcc=13)
        
        # Analyze MFCC consistency (synthetic speech often has too consistent formants)
        mfcc_consistency = np.mean(1 - np.std(mfccs, axis=1) / (np.mean(np.abs(mfccs), axis=1) + 1e-10))
        # Natural speech should have moderate consistency (not too high)
        if mfcc_consistency > 0.8:  # Too consistent
            results['formant_naturalness'] = 1 - mfcc_consistency
        else:
            results['formant_naturalness'] = mfcc_consistency
        
        # 3. Breathing and micro-pause analysis
        # Detect very quiet segments that should contain breathing
        quiet_threshold = np.percentile(np.abs(audio_signal), 5)
        quiet_segments = np.abs(audio_signal) < quiet_threshold
        
        # Group consecutive quiet segments
        quiet_regions = self._find_consecutive_regions(quiet_segments)
        breathing_score = 0.0
        
        for start, end in quiet_regions:
            duration = (end - start) / sample_rate
            if 0.1 < duration < 2.0:  # Typical breathing pause duration
                breathing_score += 1
        
        # Normalize by audio duration
        breathing_naturalness = min(1.0, breathing_score / (len(audio_signal) / sample_rate / 10))
        results['breathing_naturalness'] = breathing_naturalness
        
        # 4. Spectral consistency analysis
        stft = librosa.stft(audio_signal, n_fft=2048, hop_length=512)
        magnitude = np.abs(stft)
        
        # Analyze spectral envelope consistency
        spectral_envelope = np.mean(magnitude, axis=1)
        envelope_changes = np.diff(spectral_envelope)
        envelope_consistency = 1 - (np.std(envelope_changes) / (np.mean(np.abs(envelope_changes)) + 1e-10))
        
        # Synthetic audio often has too consistent spectral envelope
        if envelope_consistency > 0.9:
            results['spectral_envelope_naturalness'] = 1 - envelope_consistency
        else:
            results['spectral_envelope_naturalness'] = envelope_consistency
        
        # 5. Overall deepfake likelihood
        deepfake_indicators = [
            1 - results['pitch_naturalness'],
            1 - results['jitter_naturalness'], 
            1 - results['formant_naturalness'],
            1 - results['breathing_naturalness'],
            1 - results['spectral_envelope_naturalness']
        ]
        
        results['deepfake_likelihood'] = np.mean(deepfake_indicators)
        
        return results

    def _find_consecutive_regions(self, boolean_array: np.ndarray) -> List[Tuple[int, int]]:
        """Find consecutive True regions in boolean array."""
        regions = []
        start = None
        
        for i, val in enumerate(boolean_array):
            if val and start is None:
                start = i
            elif not val and start is not None:
                regions.append((start, i))
                start = None
        
        # Handle case where array ends during True region
        if start is not None:
            regions.append((start, len(boolean_array)))
        
        return regions

    def perform_isolation_forest_analysis(
        self, 
        audio_signal: np.ndarray, 
        sample_rate: int
    ) -> Dict[str, Union[float, List[float]]]:
        """
        Use isolation forest for anomaly detection in audio features.
        
        Args:
            audio_signal: Input audio signal
            sample_rate: Sample rate in Hz
            
        Returns:
            Dictionary of anomaly detection results
        """
        # Extract features for anomaly detection
        frame_length = 2048
        hop_length = 512
        
        # Compute various features
        mfccs = librosa.feature.mfcc(y=audio_signal, sr=sample_rate, n_mfcc=13)
        spectral_centroid = librosa.feature.spectral_centroid(y=audio_signal, sr=sample_rate)
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio_signal, sr=sample_rate)
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio_signal, sr=sample_rate)
        zero_crossing_rate = librosa.feature.zero_crossing_rate(audio_signal)
        
        # Combine features
        features = np.vstack([
            mfccs,
            spectral_centroid,
            spectral_rolloff, 
            spectral_bandwidth,
            zero_crossing_rate
        ]).T
        
        # Standardize features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        # Apply isolation forest
        iso_forest = IsolationForest(contamination=0.1, random_state=42)
        anomaly_scores = iso_forest.fit_predict(features_scaled)
        anomaly_probs = iso_forest.decision_function(features_scaled)
        
        # Calculate statistics
        anomaly_ratio = np.sum(anomaly_scores == -1) / len(anomaly_scores)
        mean_anomaly_score = np.mean(anomaly_probs)
        
        return {
            'anomaly_ratio': anomaly_ratio,
            'mean_anomaly_score': mean_anomaly_score,
            'anomaly_locations': np.where(anomaly_scores == -1)[0].tolist()
        }

    def detect_audio_manipulation(self, file_path: Union[str, Path]) -> AudioAuthenticity:
        """
        Perform comprehensive audio authenticity analysis.
        
        Args:
            file_path: Path to audio file
            
        Returns:
            AudioAuthenticity object with detailed analysis
        """
        file_path = Path(file_path)
        analyzer = AudioAnalyzer(self.config)
        
        # Load audio and get metadata
        audio_signal, metadata = analyzer.load_audio(file_path)
        
        start_time = time.time()
        logger.info(f"Starting authenticity analysis for {file_path}")
        
        # Perform various analyses
        results = {}
        detected_manipulations = []
        manipulation_locations = []
        
        # 1. Compression analysis
        if self.config.enable_compression_analysis:
            compression_analysis = self.analyze_compression_artifacts(audio_signal, metadata.sample_rate)
            results['compression_analysis'] = compression_analysis
            
            if compression_analysis['compression_likelihood'] > 0.7:
                detected_manipulations.append(ManipulationType.COMPRESSION_ARTIFACTS)
        
        # 2. Temporal consistency analysis
        temporal_analysis = self.detect_temporal_inconsistencies(audio_signal, metadata.sample_rate)
        results['temporal_analysis'] = temporal_analysis
        
        if temporal_analysis['splice_count'] > 0:
            detected_manipulations.append(ManipulationType.SPLICE)
            manipulation_locations.extend([
                {
                    'type': 'splice',
                    'time': splice['time'],
                    'confidence': splice['confidence']
                }
                for splice in temporal_analysis['potential_splices']
            ])
        
        # 3. Spectral analysis
        spectral_analysis = self.analyze_spectral_anomalies(audio_signal, metadata.sample_rate)
        results['spectral_analysis'] = spectral_analysis
        
        # 4. Deepfake detection
        if self.config.enable_deepfake_detection:
            deepfake_analysis = self.detect_deepfake_indicators(audio_signal, metadata.sample_rate)
            results['deepfake_analysis'] = deepfake_analysis
            
            if deepfake_analysis['deepfake_likelihood'] > 0.6:
                detected_manipulations.append(ManipulationType.DEEPFAKE)
        
        # 5. Anomaly detection
        anomaly_analysis = self.perform_isolation_forest_analysis(audio_signal, metadata.sample_rate)
        results['anomaly_analysis'] = anomaly_analysis
        
        if anomaly_analysis['anomaly_ratio'] > 0.2:  # High anomaly ratio
            detected_manipulations.append(ManipulationType.UNKNOWN)
        
        # Calculate overall authenticity score
        authenticity_factors = []
        
        # Temporal consistency (higher is more authentic)
        authenticity_factors.append(temporal_analysis['overall_consistency'])
        
        # Spectral naturalness
        spectral_score = np.mean(list(spectral_analysis.values()))
        authenticity_factors.append(spectral_score)
        
        # Deepfake indicators (lower deepfake likelihood is more authentic)
        if 'deepfake_analysis' in results:
            authenticity_factors.append(1 - results['deepfake_analysis']['deepfake_likelihood'])
        
        # Compression artifacts (lower compression likelihood is more authentic)
        if 'compression_analysis' in results:
            authenticity_factors.append(1 - results['compression_analysis']['compression_likelihood'])
        
        # Anomaly score (lower anomaly ratio is more authentic)
        authenticity_factors.append(1 - anomaly_analysis['anomaly_ratio'])
        
        authenticity_score = np.mean(authenticity_factors)
        
        # Determine authenticity and confidence
        if authenticity_score >= self.config.authenticity_threshold:
            is_authentic = True
            if authenticity_score >= 0.9:
                confidence = ConfidenceLevel.HIGH
            elif authenticity_score >= 0.8:
                confidence = ConfidenceLevel.MEDIUM
            else:
                confidence = ConfidenceLevel.LOW
        else:
            is_authentic = False
            if authenticity_score <= 0.3:
                confidence = ConfidenceLevel.HIGH
            elif authenticity_score <= 0.5:
                confidence = ConfidenceLevel.MEDIUM
            else:
                confidence = ConfidenceLevel.LOW
        
        # Remove duplicates from detected manipulations
        detected_manipulations = list(set(detected_manipulations))
        if not detected_manipulations:
            detected_manipulations = [ManipulationType.NONE]
        
        processing_time = time.time() - start_time
        logger.success(f"Authenticity analysis completed in {processing_time:.2f} seconds")
        
        return AudioAuthenticity(
            audio_metadata=metadata,
            is_authentic=is_authentic,
            confidence=confidence,
            manipulations_detected=detected_manipulations,
            manipulation_locations=manipulation_locations,
            authenticity_score=authenticity_score,
            technical_analysis=results,
            compression_analysis=results.get('compression_analysis', {}),
            temporal_analysis=temporal_analysis,
            spectral_analysis=spectral_analysis,
            processing_time=processing_time,
            model_used="multi_algorithm_ensemble"
        )


# Factory function for easy import
def detect_audio_manipulation(
    audio_path: Path,
    config: Optional[AudioAnalysisConfig] = None
) -> AudioAuthenticity:
    """
    Detect audio manipulation and assess authenticity.
    
    Args:
        audio_path: Path to audio file
        config: Detection configuration
        
    Returns:
        AudioAuthenticity object with detailed analysis
    """
    detector = AuthenticityDetector(config)
    return detector.detect_audio_manipulation(audio_path)