"""
Audio quality enhancement and noise reduction for forensic audio processing.
Provides professional-grade audio cleanup while preserving evidential integrity.
"""

import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import librosa
import numpy as np
import scipy.signal as signal
from scipy.io import wavfile
from scipy.ndimage import median_filter
from sklearn.decomposition import FastICA
import soundfile as sf
from loguru import logger

from .core import (
    AudioAnalyzer,
    AudioAnalysisConfig,
    EnhancedAudio,
    AudioQualityMetrics
)


class AudioEnhancer:
    """
    Professional audio enhancement engine for forensic applications.
    
    Features:
    - Noise reduction and suppression
    - Audio normalization and dynamic range compression
    - Spectral filtering and cleanup
    - Artifacts removal
    - Quality enhancement while preserving evidence integrity
    """

    def __init__(self, config: Optional[AudioAnalysisConfig] = None):
        """Initialize the audio enhancer."""
        self.config = config or AudioAnalysisConfig()

    def estimate_noise_profile(
        self, 
        audio_signal: np.ndarray, 
        sample_rate: int,
        noise_duration: float = 1.0
    ) -> Dict[str, np.ndarray]:
        """
        Estimate noise profile from audio signal.
        
        Args:
            audio_signal: Input audio signal
            sample_rate: Sample rate in Hz
            noise_duration: Duration in seconds to analyze for noise
            
        Returns:
            Dictionary containing noise profile information
        """
        # Use first segment as noise profile (common in forensic audio)
        noise_samples = int(noise_duration * sample_rate)
        noise_segment = audio_signal[:min(noise_samples, len(audio_signal) // 4)]
        
        # Compute noise spectrum
        noise_stft = librosa.stft(noise_segment, n_fft=2048, hop_length=512)
        noise_magnitude = np.abs(noise_stft)
        noise_phase = np.angle(noise_stft)
        
        # Statistical noise profile
        noise_mean = np.mean(noise_magnitude, axis=1, keepdims=True)
        noise_std = np.std(noise_magnitude, axis=1, keepdims=True)
        
        return {
            'noise_spectrum': noise_mean,
            'noise_std': noise_std,
            'noise_magnitude': noise_magnitude,
            'noise_phase': noise_phase
        }

    def spectral_subtraction(
        self, 
        audio_signal: np.ndarray,
        noise_profile: Dict[str, np.ndarray],
        alpha: float = 2.0,
        beta: float = 0.01
    ) -> np.ndarray:
        """
        Apply spectral subtraction for noise reduction.
        
        Args:
            audio_signal: Input audio signal
            noise_profile: Noise profile from estimate_noise_profile
            alpha: Over-subtraction factor
            beta: Spectral floor parameter
            
        Returns:
            Enhanced audio signal
        """
        # Compute STFT of input signal
        stft = librosa.stft(audio_signal, n_fft=2048, hop_length=512)
        magnitude = np.abs(stft)
        phase = np.angle(stft)
        
        # Get noise spectrum
        noise_spectrum = noise_profile['noise_spectrum']
        
        # Spectral subtraction
        enhanced_magnitude = magnitude - alpha * noise_spectrum
        
        # Apply spectral floor
        enhanced_magnitude = np.maximum(enhanced_magnitude, beta * magnitude)
        
        # Reconstruct signal
        enhanced_stft = enhanced_magnitude * np.exp(1j * phase)
        enhanced_signal = librosa.istft(enhanced_stft, hop_length=512)
        
        return enhanced_signal

    def wiener_filter(
        self, 
        audio_signal: np.ndarray,
        noise_variance: float = 0.01
    ) -> np.ndarray:
        """
        Apply Wiener filtering for noise reduction.
        
        Args:
            audio_signal: Input audio signal
            noise_variance: Estimated noise variance
            
        Returns:
            Filtered audio signal
        """
        # Compute power spectral density
        f, psd = signal.welch(audio_signal, nperseg=1024)
        
        # Estimate signal power
        signal_power = np.mean(psd)
        
        # Wiener filter transfer function
        H = signal_power / (signal_power + noise_variance)
        
        # Apply filter in frequency domain
        fft = np.fft.fft(audio_signal)
        frequencies = np.fft.fftfreq(len(audio_signal))
        
        # Interpolate H to match FFT length
        H_interp = np.interp(np.abs(frequencies), f, H)
        
        # Apply filter
        filtered_fft = fft * H_interp
        filtered_signal = np.real(np.fft.ifft(filtered_fft))
        
        return filtered_signal

    def adaptive_noise_reduction(
        self, 
        audio_signal: np.ndarray, 
        sample_rate: int,
        frame_length: int = 2048,
        hop_length: int = 512
    ) -> np.ndarray:
        """
        Apply adaptive noise reduction using statistical methods.
        
        Args:
            audio_signal: Input audio signal
            sample_rate: Sample rate in Hz
            frame_length: STFT frame length
            hop_length: STFT hop length
            
        Returns:
            Enhanced audio signal
        """
        # Compute STFT
        stft = librosa.stft(audio_signal, n_fft=frame_length, hop_length=hop_length)
        magnitude = np.abs(stft)
        phase = np.angle(stft)
        
        # Estimate noise floor for each frequency bin
        noise_floor = np.percentile(magnitude, 10, axis=1, keepdims=True)
        
        # Calculate SNR for each time-frequency bin
        snr = magnitude / (noise_floor + 1e-10)
        
        # Adaptive gain function
        gain = np.tanh(snr - 1)  # Smooth transition
        gain = np.maximum(gain, 0.1)  # Minimum gain to preserve signal
        
        # Apply gain
        enhanced_magnitude = magnitude * gain
        enhanced_stft = enhanced_magnitude * np.exp(1j * phase)
        
        # Reconstruct signal
        enhanced_signal = librosa.istft(enhanced_stft, hop_length=hop_length)
        
        return enhanced_signal

    def dynamic_range_compression(
        self, 
        audio_signal: np.ndarray,
        threshold: float = -20.0,
        ratio: float = 4.0,
        attack_time: float = 0.003,
        release_time: float = 0.1,
        sample_rate: int = 44100
    ) -> np.ndarray:
        """
        Apply dynamic range compression to improve audibility.
        
        Args:
            audio_signal: Input audio signal
            threshold: Compression threshold in dB
            ratio: Compression ratio
            attack_time: Attack time in seconds
            release_time: Release time in seconds
            sample_rate: Sample rate in Hz
            
        Returns:
            Compressed audio signal
        """
        # Convert to dB
        signal_db = 20 * np.log10(np.abs(audio_signal) + 1e-10)
        
        # Calculate gain reduction
        over_threshold = signal_db > threshold
        gain_reduction = np.zeros_like(signal_db)
        gain_reduction[over_threshold] = (signal_db[over_threshold] - threshold) * (1 - 1/ratio)
        
        # Apply smooth attack/release
        attack_coeff = np.exp(-1 / (attack_time * sample_rate))
        release_coeff = np.exp(-1 / (release_time * sample_rate))
        
        smooth_gain = np.zeros_like(gain_reduction)
        for i in range(1, len(gain_reduction)):
            if gain_reduction[i] > smooth_gain[i-1]:
                # Attack
                smooth_gain[i] = attack_coeff * smooth_gain[i-1] + (1 - attack_coeff) * gain_reduction[i]
            else:
                # Release
                smooth_gain[i] = release_coeff * smooth_gain[i-1] + (1 - release_coeff) * gain_reduction[i]
        
        # Apply compression
        compressed_signal = audio_signal * np.power(10, -smooth_gain / 20)
        
        return compressed_signal

    def normalize_audio(
        self, 
        audio_signal: np.ndarray,
        target_level: float = -3.0,
        method: str = "peak"
    ) -> np.ndarray:
        """
        Normalize audio to target level.
        
        Args:
            audio_signal: Input audio signal
            target_level: Target level in dB
            method: Normalization method ("peak" or "rms")
            
        Returns:
            Normalized audio signal
        """
        if method == "peak":
            current_peak = np.max(np.abs(audio_signal))
            target_peak = np.power(10, target_level / 20)
            gain = target_peak / (current_peak + 1e-10)
        elif method == "rms":
            current_rms = np.sqrt(np.mean(audio_signal ** 2))
            target_rms = np.power(10, target_level / 20)
            gain = target_rms / (current_rms + 1e-10)
        else:
            raise ValueError(f"Unknown normalization method: {method}")
        
        normalized_signal = audio_signal * gain
        
        # Prevent clipping
        max_val = np.max(np.abs(normalized_signal))
        if max_val > 0.99:
            normalized_signal = normalized_signal * 0.99 / max_val
        
        return normalized_signal

    def remove_dc_offset(self, audio_signal: np.ndarray) -> np.ndarray:
        """Remove DC offset from audio signal."""
        return audio_signal - np.mean(audio_signal)

    def apply_highpass_filter(
        self, 
        audio_signal: np.ndarray, 
        sample_rate: int,
        cutoff: float = 80.0,
        order: int = 4
    ) -> np.ndarray:
        """
        Apply high-pass filter to remove low-frequency noise.
        
        Args:
            audio_signal: Input audio signal
            sample_rate: Sample rate in Hz
            cutoff: Cutoff frequency in Hz
            order: Filter order
            
        Returns:
            Filtered audio signal
        """
        nyquist = sample_rate / 2
        normalized_cutoff = cutoff / nyquist
        b, a = signal.butter(order, normalized_cutoff, btype='high')
        filtered_signal = signal.filtfilt(b, a, audio_signal)
        return filtered_signal

    def apply_lowpass_filter(
        self, 
        audio_signal: np.ndarray, 
        sample_rate: int,
        cutoff: float = 8000.0,
        order: int = 4
    ) -> np.ndarray:
        """
        Apply low-pass filter to remove high-frequency noise.
        
        Args:
            audio_signal: Input audio signal
            sample_rate: Sample rate in Hz
            cutoff: Cutoff frequency in Hz
            order: Filter order
            
        Returns:
            Filtered audio signal
        """
        nyquist = sample_rate / 2
        normalized_cutoff = cutoff / nyquist
        b, a = signal.butter(order, normalized_cutoff, btype='low')
        filtered_signal = signal.filtfilt(b, a, audio_signal)
        return filtered_signal

    def remove_clicks_and_pops(
        self, 
        audio_signal: np.ndarray,
        threshold: float = 0.1,
        window_size: int = 5
    ) -> np.ndarray:
        """
        Remove clicks and pops using median filtering.
        
        Args:
            audio_signal: Input audio signal
            threshold: Detection threshold
            window_size: Median filter window size
            
        Returns:
            Cleaned audio signal
        """
        # Detect clicks using derivative
        derivative = np.abs(np.diff(audio_signal))
        click_threshold = np.mean(derivative) + threshold * np.std(derivative)
        click_locations = np.where(derivative > click_threshold)[0]
        
        # Apply median filter to detected regions
        cleaned_signal = audio_signal.copy()
        for loc in click_locations:
            start = max(0, loc - window_size // 2)
            end = min(len(audio_signal), loc + window_size // 2)
            if end - start >= 3:  # Minimum window for median filter
                cleaned_signal[start:end] = median_filter(audio_signal[start:end], size=3)
        
        return cleaned_signal

    def enhance_speech(
        self, 
        audio_signal: np.ndarray, 
        sample_rate: int
    ) -> np.ndarray:
        """
        Apply speech-specific enhancements.
        
        Args:
            audio_signal: Input audio signal
            sample_rate: Sample rate in Hz
            
        Returns:
            Enhanced speech signal
        """
        enhanced = audio_signal.copy()
        
        # Apply bandpass filter for speech frequencies (300-3400 Hz)
        nyquist = sample_rate / 2
        low_cutoff = 300.0 / nyquist
        high_cutoff = 3400.0 / nyquist
        
        b, a = signal.butter(4, [low_cutoff, high_cutoff], btype='band')
        enhanced = signal.filtfilt(b, a, enhanced)
        
        # Apply mild compression for intelligibility
        enhanced = self.dynamic_range_compression(
            enhanced, threshold=-25.0, ratio=3.0, sample_rate=sample_rate
        )
        
        return enhanced

    def enhance_audio_quality(self, file_path: Union[str, Path]) -> EnhancedAudio:
        """
        Apply comprehensive audio enhancement pipeline.
        
        Args:
            file_path: Path to input audio file
            
        Returns:
            EnhancedAudio object with results and metadata
        """
        file_path = Path(file_path)
        analyzer = AudioAnalyzer(self.config)
        
        # Load original audio
        audio_signal, metadata = analyzer.load_audio(file_path)
        
        # Calculate original quality metrics
        original_metrics = analyzer.calculate_quality_metrics(audio_signal, metadata.sample_rate)
        
        start_time = time.time()
        logger.info(f"Starting audio enhancement for {file_path}")
        
        # Enhancement pipeline
        enhanced_signal = audio_signal.copy()
        applied_enhancements = []
        processing_params = {}
        
        # Step 1: Remove DC offset
        enhanced_signal = self.remove_dc_offset(enhanced_signal)
        applied_enhancements.append("dc_offset_removal")
        
        # Step 2: High-pass filter to remove rumble
        enhanced_signal = self.apply_highpass_filter(enhanced_signal, metadata.sample_rate, cutoff=80.0)
        applied_enhancements.append("highpass_filter")
        processing_params["highpass_cutoff"] = 80.0
        
        # Step 3: Remove clicks and pops
        enhanced_signal = self.remove_clicks_and_pops(enhanced_signal)
        applied_enhancements.append("click_removal")
        
        # Step 4: Noise reduction
        if self.config.noise_reduction_enabled:
            # Estimate noise profile
            noise_profile = self.estimate_noise_profile(enhanced_signal, metadata.sample_rate)
            
            # Apply adaptive noise reduction
            enhanced_signal = self.adaptive_noise_reduction(enhanced_signal, metadata.sample_rate)
            applied_enhancements.append("adaptive_noise_reduction")
            
            # Apply spectral subtraction for additional cleanup
            enhanced_signal = self.spectral_subtraction(enhanced_signal, noise_profile, alpha=1.5)
            applied_enhancements.append("spectral_subtraction")
            processing_params["noise_reduction_alpha"] = 1.5
        
        # Step 5: Speech enhancement (if audio contains speech)
        enhanced_signal = self.enhance_speech(enhanced_signal, metadata.sample_rate)
        applied_enhancements.append("speech_enhancement")
        
        # Step 6: Dynamic range compression
        enhanced_signal = self.dynamic_range_compression(enhanced_signal, sample_rate=metadata.sample_rate)
        applied_enhancements.append("dynamic_range_compression")
        processing_params["compression_threshold"] = -20.0
        processing_params["compression_ratio"] = 4.0
        
        # Step 7: Normalization
        if self.config.normalize_audio:
            enhanced_signal = self.normalize_audio(enhanced_signal, target_level=-3.0)
            applied_enhancements.append("normalization")
            processing_params["normalization_target"] = -3.0
        
        # Step 8: Resample if needed
        if self.config.target_sample_rate != metadata.sample_rate:
            enhanced_signal = librosa.resample(
                enhanced_signal, 
                orig_sr=metadata.sample_rate, 
                target_sr=self.config.target_sample_rate
            )
            applied_enhancements.append("resampling")
            processing_params["target_sample_rate"] = self.config.target_sample_rate
            final_sample_rate = self.config.target_sample_rate
        else:
            final_sample_rate = metadata.sample_rate
        
        # Save enhanced audio
        output_dir = self.config.output_directory or file_path.parent
        output_path = output_dir / f"{file_path.stem}_enhanced.wav"
        
        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save enhanced audio
        sf.write(str(output_path), enhanced_signal, final_sample_rate, subtype='PCM_16')
        
        # Calculate enhanced quality metrics
        enhanced_metrics = analyzer.calculate_quality_metrics(enhanced_signal, final_sample_rate)
        
        # Calculate improvement metrics
        quality_improvement = {
            "snr_improvement": 0.0,  # Would need reference signal
            "dynamic_range_change": enhanced_metrics.dynamic_range_db - original_metrics.dynamic_range_db,
            "peak_level_change": enhanced_metrics.peak_level_db - original_metrics.peak_level_db,
            "rms_level_change": enhanced_metrics.rms_level_db - original_metrics.rms_level_db,
            "spectral_centroid_change": enhanced_metrics.spectral_centroid - original_metrics.spectral_centroid
        }
        
        processing_time = time.time() - start_time
        logger.success(f"Audio enhancement completed in {processing_time:.2f} seconds")
        
        return EnhancedAudio(
            original_metadata=metadata,
            enhanced_file_path=output_path,
            enhancement_applied=applied_enhancements,
            quality_improvement=quality_improvement,
            processing_parameters=processing_params,
            before_metrics=original_metrics,
            after_metrics=enhanced_metrics,
            processing_time=processing_time
        )

    def batch_enhance(
        self, 
        file_paths: List[Union[str, Path]]
    ) -> List[EnhancedAudio]:
        """
        Apply enhancement to multiple audio files.
        
        Args:
            file_paths: List of audio file paths
            
        Returns:
            List of EnhancedAudio objects
        """
        results = []
        
        for file_path in file_paths:
            try:
                logger.info(f"Enhancing {file_path}")
                enhanced = self.enhance_audio_quality(file_path)
                results.append(enhanced)
                logger.success(f"Successfully enhanced {file_path}")
            except Exception as e:
                logger.error(f"Failed to enhance {file_path}: {e}")
                # Could create error result here if needed
        
        return results


# Factory function for easy import
def enhance_audio_quality(
    audio_path: Path,
    config: Optional[AudioAnalysisConfig] = None
) -> EnhancedAudio:
    """
    Enhance audio quality with noise reduction and optimization.
    
    Args:
        audio_path: Path to audio file
        config: Enhancement configuration
        
    Returns:
        EnhancedAudio object with results and enhanced file
    """
    enhancer = AudioEnhancer(config)
    return enhancer.enhance_audio_quality(audio_path)