"""
Lemkin Video Authentication Toolkit - Compression Analysis Module

This module provides compression artifact analysis for video authenticity verification.
It detects signs of recompression, quality inconsistencies, and encoding artifacts
that may indicate video manipulation.

Features:
- Multiple compression artifact detection
- Recompression analysis
- Quality assessment and consistency checking
- Codec sequence analysis
- Quantization parameter analysis
- Motion compensation vector analysis
"""

import cv2
import numpy as np
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import hashlib
import subprocess
import json
import re

try:
    import ffmpeg
    FFMPEG_AVAILABLE = True
except ImportError:
    FFMPEG_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

from .core import (
    VideoAuthConfig, CompressionAnalysis, CompressionLevel, 
    TamperingIndicator, TamperingType
)

logger = logging.getLogger(__name__)


class CompressionAnalyzer:
    """
    Advanced compression analysis system for detecting video manipulation
    through compression artifact analysis and quality assessment.
    """
    
    def __init__(self, config: VideoAuthConfig):
        """Initialize compression analyzer with configuration"""
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.CompressionAnalyzer")
        
        # Analysis settings
        self.block_size = 8  # DCT block size for JPEG/H.264
        self.quality_threshold = 0.7
        self.recompression_threshold = 0.15
        
        # Codec information
        self.common_codecs = {
            'h264': {'quality_range': (18, 51), 'default_crf': 23},
            'h265': {'quality_range': (0, 51), 'default_crf': 28},
            'vp9': {'quality_range': (0, 63), 'default_crf': 31},
            'av1': {'quality_range': (0, 63), 'default_crf': 30}
        }
        
        self.logger.info("Compression analyzer initialized")
    
    def analyze_compression_artifacts(self, video_path: Path) -> CompressionAnalysis:
        """
        Analyze compression artifacts for authenticity verification
        
        Args:
            video_path: Path to video file
            
        Returns:
            CompressionAnalysis: Comprehensive compression analysis results
        """
        start_time = datetime.utcnow()
        self.logger.info(f"Starting compression analysis for: {video_path}")
        
        # Calculate video hash
        video_hash = self._calculate_video_hash(video_path)
        
        try:
            # Extract video metadata and technical details
            video_metadata = self._extract_video_metadata(video_path)
            
            # Extract frames for analysis
            frames_data = self._extract_frames_for_analysis(video_path)
            
            # Analyze compression level and quality
            compression_assessment = self._assess_compression_level(video_metadata, frames_data['frames'])
            
            # Detect recompression indicators
            recompression_analysis = self._detect_recompression(frames_data['frames'], video_metadata)
            
            # Analyze compression artifacts
            artifact_analysis = self._analyze_compression_artifacts(frames_data['frames'])
            
            # Detect quality inconsistencies
            quality_analysis = self._analyze_quality_consistency(frames_data['frames'])
            
            # Analyze codec sequence
            codec_analysis = self._analyze_codec_sequence(video_path)
            
            # Calculate overall quality metrics
            quality_metrics = self._calculate_quality_metrics(
                frames_data['frames'], 
                artifact_analysis,
                quality_analysis
            )
            
            # Create compression analysis result
            analysis = CompressionAnalysis(
                video_hash=video_hash,
                compression_level=compression_assessment['level'],
                is_recompressed=recompression_analysis['is_recompressed'],
                recompression_count=recompression_analysis['count'],
                overall_quality_score=quality_metrics['overall_score'],
                bitrate_consistency=quality_analysis['bitrate_consistency'],
                compression_efficiency=quality_metrics['efficiency'],
                blocking_artifacts=artifact_analysis['blocking_artifacts'],
                ringing_artifacts=artifact_analysis['ringing_artifacts'],
                mosquito_noise=artifact_analysis['mosquito_noise'],
                codec_sequence=codec_analysis['sequence'],
                quantization_parameters=codec_analysis['qp_values'],
                motion_compensation_vectors=codec_analysis['motion_vectors'],
                inconsistent_regions=quality_analysis['inconsistent_regions'],
                compression_boundaries=quality_analysis['boundaries'],
                quality_variations=quality_analysis['variations'],
                analysis_method="multi_scale",
                tools_used=["opencv", "ffmpeg"] if FFMPEG_AVAILABLE else ["opencv"],
                processing_time_seconds=(datetime.utcnow() - start_time).total_seconds()
            )
            
            self.logger.info(f"Compression analysis completed: {analysis.compression_level} "
                           f"(recompressed: {analysis.is_recompressed})")
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Compression analysis failed: {str(e)}")
            # Return minimal analysis on failure
            return CompressionAnalysis(
                video_hash=video_hash,
                compression_level=CompressionLevel.UNKNOWN,
                is_recompressed=False,
                recompression_count=0,
                overall_quality_score=0.0,
                bitrate_consistency=0.0,
                compression_efficiency=0.0,
                processing_time_seconds=(datetime.utcnow() - start_time).total_seconds()
            )
    
    def _calculate_video_hash(self, video_path: Path) -> str:
        """Calculate SHA-256 hash of video file"""
        hash_obj = hashlib.sha256()
        with open(video_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_obj.update(chunk)
        return hash_obj.hexdigest()
    
    def _extract_video_metadata(self, video_path: Path) -> Dict[str, Any]:
        """Extract detailed video metadata including codec information"""
        metadata = {}
        
        if FFMPEG_AVAILABLE:
            try:
                # Use ffprobe to get detailed metadata
                probe = ffmpeg.probe(str(video_path))
                
                for stream in probe.get('streams', []):
                    if stream.get('codec_type') == 'video':
                        metadata.update({
                            'codec_name': stream.get('codec_name'),
                            'codec_long_name': stream.get('codec_long_name'),
                            'profile': stream.get('profile'),
                            'level': stream.get('level'),
                            'width': stream.get('width'),
                            'height': stream.get('height'),
                            'frame_rate': eval(stream.get('r_frame_rate', '0/1')),
                            'bit_rate': int(stream.get('bit_rate', 0)),
                            'duration': float(stream.get('duration', 0)),
                            'avg_frame_rate': eval(stream.get('avg_frame_rate', '0/1')),
                            'pix_fmt': stream.get('pix_fmt'),
                            'color_space': stream.get('color_space'),
                            'color_transfer': stream.get('color_transfer'),
                            'color_primaries': stream.get('color_primaries')
                        })
                        break
                
                # Get format information
                format_info = probe.get('format', {})
                metadata.update({
                    'format_name': format_info.get('format_name'),
                    'format_long_name': format_info.get('format_long_name'),
                    'size': int(format_info.get('size', 0)),
                    'probe_score': format_info.get('probe_score')
                })
                
            except Exception as e:
                self.logger.warning(f"FFmpeg metadata extraction failed: {str(e)}")
        
        # Fallback to OpenCV
        if not metadata:
            cap = cv2.VideoCapture(str(video_path))
            try:
                metadata.update({
                    'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                    'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                    'frame_rate': cap.get(cv2.CAP_PROP_FPS),
                    'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                    'fourcc': int(cap.get(cv2.CAP_PROP_FOURCC))
                })
            finally:
                cap.release()
        
        return metadata
    
    def _extract_frames_for_analysis(self, video_path: Path) -> Dict[str, Any]:
        """Extract frames for compression analysis"""
        cap = cv2.VideoCapture(str(video_path))
        frames = []
        frame_info = []
        
        try:
            frame_count = 0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Sample frames throughout the video
            sample_interval = max(1, total_frames // 50)  # Sample ~50 frames max
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_count % sample_interval == 0:
                    frames.append(frame)
                    
                    # Store frame information
                    frame_info.append({
                        'index': frame_count,
                        'timestamp': frame_count / cap.get(cv2.CAP_PROP_FPS),
                        'size': frame.nbytes
                    })
                
                frame_count += 1
                
                # Limit number of frames
                if len(frames) >= 50:
                    break
            
            return {
                'frames': frames,
                'frame_info': frame_info,
                'total_frames': total_frames
            }
            
        finally:
            cap.release()
    
    def _assess_compression_level(self, metadata: Dict[str, Any], frames: List[np.ndarray]) -> Dict[str, Any]:
        """Assess the compression level of the video"""
        
        # Get bitrate information
        bit_rate = metadata.get('bit_rate', 0)
        width = metadata.get('width', 1920)
        height = metadata.get('height', 1080)
        frame_rate = metadata.get('frame_rate', 30)
        
        # Calculate bits per pixel
        if bit_rate > 0 and width > 0 and height > 0 and frame_rate > 0:
            bits_per_pixel = bit_rate / (width * height * frame_rate)
        else:
            bits_per_pixel = 0
        
        # Estimate compression level based on bits per pixel
        if bits_per_pixel >= 1.0:
            level = CompressionLevel.UNCOMPRESSED
        elif bits_per_pixel >= 0.5:
            level = CompressionLevel.HIGH_QUALITY
        elif bits_per_pixel >= 0.2:
            level = CompressionLevel.MEDIUM_QUALITY
        elif bits_per_pixel >= 0.1:
            level = CompressionLevel.LOW_QUALITY
        else:
            level = CompressionLevel.HEAVILY_COMPRESSED
        
        # Analyze actual frame quality
        if frames:
            avg_quality = self._calculate_frame_quality_score(frames[0])
            
            # Adjust level based on actual quality
            if avg_quality < 0.3 and level != CompressionLevel.HEAVILY_COMPRESSED:
                level = CompressionLevel.LOW_QUALITY
            elif avg_quality > 0.8 and level == CompressionLevel.HEAVILY_COMPRESSED:
                level = CompressionLevel.MEDIUM_QUALITY
        
        return {
            'level': level,
            'bits_per_pixel': bits_per_pixel,
            'estimated_quality': avg_quality if frames else 0.0
        }
    
    def _calculate_frame_quality_score(self, frame: np.ndarray) -> float:
        """Calculate quality score for a single frame"""
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate sharpness (Laplacian variance)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        sharpness = laplacian.var()
        
        # Calculate gradient magnitude
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        avg_gradient = np.mean(gradient_magnitude)
        
        # Normalize scores
        sharpness_score = min(1.0, sharpness / 1000.0)
        gradient_score = min(1.0, avg_gradient / 50.0)
        
        # Combined quality score
        quality_score = (sharpness_score + gradient_score) / 2.0
        
        return quality_score
    
    def _detect_recompression(self, frames: List[np.ndarray], metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Detect signs of video recompression"""
        
        recompression_indicators = []
        estimated_count = 0
        
        if not frames:
            return {'is_recompressed': False, 'count': 0, 'indicators': []}
        
        # Analyze DCT coefficient distributions
        dct_analysis = self._analyze_dct_coefficients(frames[:5])  # Sample first 5 frames
        if dct_analysis['double_quantization_detected']:
            recompression_indicators.append("Double quantization detected in DCT coefficients")
            estimated_count += 1
        
        # Check for blocking artifacts patterns
        blocking_analysis = self._analyze_blocking_patterns(frames)
        if blocking_analysis['multiple_block_sizes']:
            recompression_indicators.append("Multiple blocking patterns detected")
            estimated_count += 1
        
        # Analyze quality variations
        quality_variations = self._analyze_frame_quality_variations(frames)
        if quality_variations['high_variation']:
            recompression_indicators.append("High quality variation across frames")
        
        # Check codec metadata for inconsistencies
        if metadata.get('codec_name'):
            codec_inconsistencies = self._check_codec_inconsistencies(metadata)
            if codec_inconsistencies:
                recompression_indicators.extend(codec_inconsistencies)
                estimated_count += len(codec_inconsistencies)
        
        is_recompressed = len(recompression_indicators) >= 2 or estimated_count >= 1
        
        return {
            'is_recompressed': is_recompressed,
            'count': max(1, estimated_count) if is_recompressed else 0,
            'indicators': recompression_indicators
        }
    
    def _analyze_dct_coefficients(self, frames: List[np.ndarray]) -> Dict[str, Any]:
        """Analyze DCT coefficients for double quantization"""
        double_quantization_scores = []
        
        for frame in frames:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Divide into 8x8 blocks and analyze DCT coefficients
            h, w = gray.shape
            blocks_analyzed = 0
            dq_blocks = 0
            
            for y in range(0, h - 8, 8):
                for x in range(0, w - 8, 8):
                    block = gray[y:y+8, x:x+8].astype(np.float32)
                    
                    # Apply DCT
                    dct_block = cv2.dct(block)
                    
                    # Analyze coefficient distribution
                    if self._detect_double_quantization_in_block(dct_block):
                        dq_blocks += 1
                    
                    blocks_analyzed += 1
                    
                    # Limit analysis for performance
                    if blocks_analyzed >= 100:
                        break
                
                if blocks_analyzed >= 100:
                    break
            
            if blocks_analyzed > 0:
                dq_ratio = dq_blocks / blocks_analyzed
                double_quantization_scores.append(dq_ratio)
        
        avg_dq_score = np.mean(double_quantization_scores) if double_quantization_scores else 0.0
        
        return {
            'double_quantization_detected': avg_dq_score > 0.1,
            'score': avg_dq_score
        }
    
    def _detect_double_quantization_in_block(self, dct_block: np.ndarray) -> bool:
        """Detect double quantization in a single DCT block"""
        # Flatten the DCT coefficients
        coeffs = dct_block.flatten()
        
        # Remove DC coefficient
        ac_coeffs = coeffs[1:]
        
        # Look for periodic patterns in coefficient distribution
        # Double quantization often creates regular patterns
        hist, bins = np.histogram(ac_coeffs, bins=50, range=(-100, 100))
        
        # Check for multiple peaks (indication of double quantization)
        peaks = self._find_histogram_peaks(hist)
        
        return len(peaks) > 2
    
    def _find_histogram_peaks(self, hist: np.ndarray) -> List[int]:
        """Find peaks in histogram"""
        peaks = []
        for i in range(1, len(hist) - 1):
            if hist[i] > hist[i-1] and hist[i] > hist[i+1] and hist[i] > np.max(hist) * 0.1:
                peaks.append(i)
        return peaks
    
    def _analyze_blocking_patterns(self, frames: List[np.ndarray]) -> Dict[str, Any]:
        """Analyze blocking artifact patterns"""
        block_sizes_detected = set()
        
        for frame in frames[:3]:  # Analyze first 3 frames
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect 8x8 blocks (common in JPEG/H.264)
            if self._detect_blocking_artifacts(gray, 8):
                block_sizes_detected.add(8)
            
            # Detect 16x16 blocks (common in some codecs)
            if self._detect_blocking_artifacts(gray, 16):
                block_sizes_detected.add(16)
            
            # Detect 4x4 blocks (H.264 intra prediction)
            if self._detect_blocking_artifacts(gray, 4):
                block_sizes_detected.add(4)
        
        return {
            'multiple_block_sizes': len(block_sizes_detected) > 1,
            'detected_sizes': list(block_sizes_detected)
        }
    
    def _detect_blocking_artifacts(self, gray_frame: np.ndarray, block_size: int) -> bool:
        """Detect blocking artifacts of specific block size"""
        h, w = gray_frame.shape
        
        # Calculate gradients at block boundaries
        vertical_gradients = []
        horizontal_gradients = []
        
        # Check vertical block boundaries
        for x in range(block_size, w, block_size):
            if x < w - 1:
                left_col = gray_frame[:, x-1]
                right_col = gray_frame[:, x]
                gradient = np.mean(np.abs(left_col.astype(int) - right_col.astype(int)))
                vertical_gradients.append(gradient)
        
        # Check horizontal block boundaries
        for y in range(block_size, h, block_size):
            if y < h - 1:
                top_row = gray_frame[y-1, :]
                bottom_row = gray_frame[y, :]
                gradient = np.mean(np.abs(top_row.astype(int) - bottom_row.astype(int)))
                horizontal_gradients.append(gradient)
        
        # Calculate average gradients
        avg_vertical = np.mean(vertical_gradients) if vertical_gradients else 0
        avg_horizontal = np.mean(horizontal_gradients) if horizontal_gradients else 0
        
        # Calculate background gradient for comparison
        background_gradient = self._calculate_background_gradient(gray_frame)
        
        # Blocking detected if boundary gradients are significantly higher
        threshold_multiplier = 1.5
        blocking_detected = (avg_vertical > background_gradient * threshold_multiplier or 
                           avg_horizontal > background_gradient * threshold_multiplier)
        
        return blocking_detected
    
    def _calculate_background_gradient(self, gray_frame: np.ndarray) -> float:
        """Calculate average gradient in the image (not at block boundaries)"""
        # Calculate Sobel gradients
        grad_x = cv2.Sobel(gray_frame, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray_frame, cv2.CV_64F, 0, 1, ksize=3)
        
        # Calculate magnitude
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        return np.mean(magnitude)
    
    def _analyze_frame_quality_variations(self, frames: List[np.ndarray]) -> Dict[str, Any]:
        """Analyze quality variations across frames"""
        quality_scores = []
        
        for frame in frames:
            quality = self._calculate_frame_quality_score(frame)
            quality_scores.append(quality)
        
        if len(quality_scores) < 2:
            return {'high_variation': False, 'variation_score': 0.0}
        
        # Calculate variation metrics
        std_dev = np.std(quality_scores)
        mean_quality = np.mean(quality_scores)
        variation_coefficient = std_dev / (mean_quality + 1e-6)
        
        # High variation might indicate recompression
        high_variation = variation_coefficient > 0.3
        
        return {
            'high_variation': high_variation,
            'variation_score': variation_coefficient,
            'quality_scores': quality_scores
        }
    
    def _check_codec_inconsistencies(self, metadata: Dict[str, Any]) -> List[str]:
        """Check for codec-related inconsistencies"""
        inconsistencies = []
        
        codec_name = metadata.get('codec_name', '').lower()
        profile = metadata.get('profile', '')
        level = metadata.get('level', '')
        
        # Check for unusual codec parameters
        if codec_name == 'h264':
            if profile and 'baseline' in profile.lower() and metadata.get('bit_rate', 0) > 10000000:
                inconsistencies.append("High bitrate with baseline profile (unusual)")
            
            if level and int(level) if level.isdigit() else 0 > 51:
                inconsistencies.append("Invalid H.264 level detected")
        
        # Check for format/codec mismatches
        format_name = metadata.get('format_name', '')
        if 'mp4' in format_name and codec_name not in ['h264', 'h265', 'av1']:
            inconsistencies.append(f"Unusual codec {codec_name} in MP4 container")
        
        return inconsistencies
    
    def _analyze_compression_artifacts(self, frames: List[np.ndarray]) -> Dict[str, float]:
        """Analyze various compression artifacts"""
        
        blocking_scores = []
        ringing_scores = []
        mosquito_scores = []
        
        for frame in frames[:10]:  # Analyze first 10 frames
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Blocking artifacts
            blocking_score = self._measure_blocking_artifacts(gray)
            blocking_scores.append(blocking_score)
            
            # Ringing artifacts
            ringing_score = self._measure_ringing_artifacts(gray)
            ringing_scores.append(ringing_score)
            
            # Mosquito noise
            mosquito_score = self._measure_mosquito_noise(gray)
            mosquito_scores.append(mosquito_score)
        
        return {
            'blocking_artifacts': np.mean(blocking_scores) if blocking_scores else 0.0,
            'ringing_artifacts': np.mean(ringing_scores) if ringing_scores else 0.0,
            'mosquito_noise': np.mean(mosquito_scores) if mosquito_scores else 0.0
        }
    
    def _measure_blocking_artifacts(self, gray_frame: np.ndarray) -> float:
        """Measure blocking artifacts intensity"""
        h, w = gray_frame.shape
        
        # Calculate differences at 8x8 block boundaries
        vertical_diffs = []
        horizontal_diffs = []
        
        for x in range(8, w, 8):
            if x < w - 1:
                diff = np.mean(np.abs(gray_frame[:, x-1].astype(int) - gray_frame[:, x].astype(int)))
                vertical_diffs.append(diff)
        
        for y in range(8, h, 8):
            if y < h - 1:
                diff = np.mean(np.abs(gray_frame[y-1, :].astype(int) - gray_frame[y, :].astype(int)))
                horizontal_diffs.append(diff)
        
        # Average blocking score
        all_diffs = vertical_diffs + horizontal_diffs
        if all_diffs:
            return min(1.0, np.mean(all_diffs) / 50.0)  # Normalize to [0, 1]
        return 0.0
    
    def _measure_ringing_artifacts(self, gray_frame: np.ndarray) -> float:
        """Measure ringing artifacts around edges"""
        # Apply edge detection
        edges = cv2.Canny(gray_frame, 50, 150)
        
        # Dilate edges to create regions around edges
        kernel = np.ones((5, 5), np.uint8)
        edge_regions = cv2.dilate(edges, kernel, iterations=1)
        
        # Calculate variance in edge regions
        edge_pixels = gray_frame[edge_regions > 0]
        
        if len(edge_pixels) > 0:
            edge_variance = np.var(edge_pixels)
            return min(1.0, edge_variance / 1000.0)  # Normalize
        
        return 0.0
    
    def _measure_mosquito_noise(self, gray_frame: np.ndarray) -> float:
        """Measure mosquito noise around high-contrast areas"""
        # Find high-contrast regions
        laplacian = cv2.Laplacian(gray_frame, cv2.CV_64F)
        high_contrast = np.abs(laplacian) > np.percentile(np.abs(laplacian), 90)
        
        # Measure noise in surrounding areas
        kernel = np.ones((7, 7), np.uint8)
        noise_regions = cv2.dilate(high_contrast.astype(np.uint8), kernel, iterations=1)
        noise_regions = noise_regions - high_contrast.astype(np.uint8)  # Remove original high-contrast pixels
        
        noise_pixels = gray_frame[noise_regions > 0]
        
        if len(noise_pixels) > 0:
            noise_std = np.std(noise_pixels)
            return min(1.0, noise_std / 30.0)  # Normalize
        
        return 0.0
    
    def _analyze_quality_consistency(self, frames: List[np.ndarray]) -> Dict[str, Any]:
        """Analyze quality consistency across frames"""
        
        quality_scores = []
        bitrate_estimates = []
        inconsistent_regions = []
        boundaries = []
        variations = []
        
        for i, frame in enumerate(frames):
            # Calculate quality score
            quality = self._calculate_frame_quality_score(frame)
            quality_scores.append(quality)
            
            # Estimate bitrate based on frame complexity
            complexity = self._estimate_frame_complexity(frame)
            bitrate_estimates.append(complexity)
            
            # Check for quality boundaries (sudden changes)
            if i > 0 and abs(quality - quality_scores[i-1]) > 0.3:
                boundaries.append((i-1, i))
                inconsistent_regions.append({
                    'frame_range': (i-1, i),
                    'quality_change': abs(quality - quality_scores[i-1]),
                    'type': 'quality_jump'
                })
        
        # Calculate consistency metrics
        quality_std = np.std(quality_scores) if len(quality_scores) > 1 else 0
        bitrate_std = np.std(bitrate_estimates) if len(bitrate_estimates) > 1 else 0
        
        bitrate_consistency = 1.0 - min(1.0, bitrate_std / (np.mean(bitrate_estimates) + 1e-6))
        
        # Analyze variations
        for i in range(len(quality_scores)):
            variations.append({
                'frame_index': i,
                'quality_score': quality_scores[i],
                'bitrate_estimate': bitrate_estimates[i],
                'is_outlier': abs(quality_scores[i] - np.mean(quality_scores)) > 2 * quality_std
            })
        
        return {
            'bitrate_consistency': bitrate_consistency,
            'quality_std': quality_std,
            'inconsistent_regions': inconsistent_regions,
            'boundaries': boundaries,
            'variations': variations
        }
    
    def _estimate_frame_complexity(self, frame: np.ndarray) -> float:
        """Estimate frame complexity for bitrate estimation"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate spatial complexity
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        spatial_complexity = np.var(laplacian)
        
        # Calculate texture complexity
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        texture_complexity = np.mean(np.sqrt(grad_x**2 + grad_y**2))
        
        # Combine complexities
        total_complexity = spatial_complexity + texture_complexity * 10
        
        return total_complexity
    
    def _analyze_codec_sequence(self, video_path: Path) -> Dict[str, Any]:
        """Analyze codec sequence and parameters"""
        
        # Initialize results
        codec_sequence = []
        qp_values = []
        motion_vectors = {}
        
        if FFMPEG_AVAILABLE:
            try:
                # Use ffprobe to get detailed codec information
                result = subprocess.run([
                    'ffprobe', '-v', 'quiet', '-select_streams', 'v:0',
                    '-show_entries', 'packet=codec_type,flags,size,dts,pts',
                    '-of', 'csv=print_section=0', str(video_path)
                ], capture_output=True, text=True, timeout=30)
                
                if result.returncode == 0:
                    lines = result.stdout.strip().split('\n')
                    for line in lines[:100]:  # Limit to first 100 packets
                        parts = line.split(',')
                        if len(parts) >= 4 and parts[0] == 'video':
                            # Analyze packet information
                            packet_size = int(parts[2]) if parts[2].isdigit() else 0
                            if packet_size > 0:
                                # Estimate QP from packet size (rough approximation)
                                estimated_qp = max(0, min(51, 51 - (packet_size / 1000)))
                                qp_values.append(estimated_qp)
                
            except Exception as e:
                self.logger.warning(f"ffprobe analysis failed: {str(e)}")
        
        # Fallback analysis using OpenCV
        cap = cv2.VideoCapture(str(video_path))
        try:
            fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
            codec_name = ''.join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])
            codec_sequence.append(codec_name.strip())
        except:
            codec_sequence.append("unknown")
        finally:
            cap.release()
        
        return {
            'sequence': codec_sequence,
            'qp_values': qp_values,
            'motion_vectors': motion_vectors
        }
    
    def _calculate_quality_metrics(self, 
                                 frames: List[np.ndarray], 
                                 artifact_analysis: Dict[str, float],
                                 quality_analysis: Dict[str, Any]) -> Dict[str, float]:
        """Calculate overall quality metrics"""
        
        if not frames:
            return {'overall_score': 0.0, 'efficiency': 0.0}
        
        # Calculate base quality from frames
        frame_qualities = [self._calculate_frame_quality_score(frame) for frame in frames[:10]]
        avg_frame_quality = np.mean(frame_qualities)
        
        # Adjust for artifacts
        artifact_penalty = (
            artifact_analysis['blocking_artifacts'] * 0.3 +
            artifact_analysis['ringing_artifacts'] * 0.2 +
            artifact_analysis['mosquito_noise'] * 0.2
        )
        
        # Adjust for consistency
        consistency_bonus = quality_analysis['bitrate_consistency'] * 0.1
        
        # Calculate overall score
        overall_score = avg_frame_quality - artifact_penalty + consistency_bonus
        overall_score = max(0.0, min(1.0, overall_score))
        
        # Calculate compression efficiency
        # Higher efficiency = good quality with reasonable file size
        complexity_scores = [self._estimate_frame_complexity(frame) for frame in frames[:5]]
        avg_complexity = np.mean(complexity_scores) if complexity_scores else 1.0
        
        # Efficiency is quality relative to complexity
        efficiency = avg_frame_quality / (1.0 + avg_complexity / 10000.0)
        efficiency = max(0.0, min(1.0, efficiency))
        
        return {
            'overall_score': overall_score,
            'efficiency': efficiency
        }
    
    def create_tampering_indicators(self, analysis: CompressionAnalysis) -> List[TamperingIndicator]:
        """Create tampering indicators from compression analysis"""
        indicators = []
        
        # Recompression indicator
        if analysis.is_recompressed:
            indicator = TamperingIndicator(
                tampering_type=TamperingType.COMPRESSION_INCONSISTENCY,
                confidence=0.8 if analysis.recompression_count > 1 else 0.6,
                description=f"Video shows signs of {analysis.recompression_count} recompression cycles",
                evidence={
                    'recompression_count': analysis.recompression_count,
                    'compression_level': analysis.compression_level.value
                },
                analysis_method="compression_analysis",
                severity_score=min(10.0, analysis.recompression_count * 3.0),
                is_critical=analysis.recompression_count > 2
            )
            indicators.append(indicator)
        
        # Quality inconsistency indicator
        if analysis.bitrate_consistency < 0.5:
            indicator = TamperingIndicator(
                tampering_type=TamperingType.COMPRESSION_INCONSISTENCY,
                confidence=1.0 - analysis.bitrate_consistency,
                description=f"Inconsistent compression quality detected (consistency: {analysis.bitrate_consistency:.2f})",
                evidence={'bitrate_consistency': analysis.bitrate_consistency},
                analysis_method="quality_analysis",
                severity_score=(1.0 - analysis.bitrate_consistency) * 8,
                is_critical=analysis.bitrate_consistency < 0.3
            )
            indicators.append(indicator)
        
        # High artifact levels
        if analysis.blocking_artifacts > 0.5 or analysis.ringing_artifacts > 0.5:
            indicator = TamperingIndicator(
                tampering_type=TamperingType.COMPRESSION_INCONSISTENCY,
                confidence=max(analysis.blocking_artifacts, analysis.ringing_artifacts),
                description="High levels of compression artifacts detected",
                evidence={
                    'blocking_artifacts': analysis.blocking_artifacts,
                    'ringing_artifacts': analysis.ringing_artifacts,
                    'mosquito_noise': analysis.mosquito_noise
                },
                analysis_method="artifact_detection",
                severity_score=max(analysis.blocking_artifacts, analysis.ringing_artifacts) * 7,
                is_critical=False
            )
            indicators.append(indicator)
        
        return indicators