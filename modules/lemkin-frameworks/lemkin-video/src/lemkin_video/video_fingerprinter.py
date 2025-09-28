"""
Lemkin Video Authentication Toolkit - Video Fingerprinting Module

This module provides content-based video duplicate detection and fingerprinting
capabilities for identifying manipulated or copied video content.

Features:
- Perceptual hashing algorithms (dHash, pHash, aHash)
- Temporal feature extraction
- Audio fingerprinting
- Content-based similarity matching
- Robust duplicate detection
"""

import cv2
import numpy as np
import hashlib
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import json

try:
    import imagehash
    from PIL import Image
    IMAGEHASH_AVAILABLE = True
except ImportError:
    IMAGEHASH_AVAILABLE = False

try:
    import moviepy.editor as mp
    MOVIEPY_AVAILABLE = True
except ImportError:
    MOVIEPY_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

from .core import VideoAuthConfig, VideoFingerprint

logger = logging.getLogger(__name__)


class VideoFingerprinter:
    """
    Advanced video fingerprinting system for content-based duplicate detection
    and similarity analysis using multiple algorithms and feature extraction techniques.
    """
    
    def __init__(self, config: VideoAuthConfig):
        """Initialize video fingerprinter with configuration"""
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.VideoFingerprinter")
        
        # Algorithm settings
        self.hash_size = config.hash_size
        self.algorithm = config.fingerprint_algorithm
        self.sample_rate = 1  # Sample every N frames
        
        # Feature extraction settings
        self.frame_analysis_interval = max(1, config.frame_analysis_interval // 5)
        self.color_histogram_bins = 32
        
        # Similarity thresholds
        self.exact_threshold = 0.95
        self.similar_threshold = config.perceptual_hash_threshold
        
        self.logger.info(f"Video fingerprinter initialized (algorithm: {self.algorithm})")
    
    def fingerprint_video(self, video_path: Path) -> VideoFingerprint:
        """
        Generate comprehensive content-based fingerprint for video
        
        Args:
            video_path: Path to video file
            
        Returns:
            VideoFingerprint: Comprehensive video fingerprint
        """
        start_time = datetime.utcnow()
        self.logger.info(f"Starting video fingerprinting for: {video_path}")
        
        # Calculate video hash
        video_hash = self._calculate_video_hash(video_path)
        
        try:
            # Extract video frames and metadata
            video_data = self._extract_video_data(video_path)
            
            if not video_data['frames']:
                raise ValueError("No frames could be extracted from video")
            
            # Generate perceptual hash
            perceptual_hash = self._generate_perceptual_hash(video_data['frames'])
            
            # Generate temporal features hash
            temporal_hash = self._generate_temporal_hash(video_data['frames'])
            
            # Extract audio fingerprint if available
            audio_fingerprint = None
            if MOVIEPY_AVAILABLE:
                audio_fingerprint = self._extract_audio_fingerprint(video_path)
            
            # Calculate content features
            content_features = self._calculate_content_features(video_data['frames'])
            
            # Extract key frame hashes
            key_frame_hashes = self._extract_key_frame_hashes(video_data['frames'])
            
            # Calculate motion vectors
            motion_vectors = self._calculate_motion_vectors(video_data['frames'])
            
            # Generate color histograms
            color_histograms = self._generate_color_histograms(video_data['frames'])
            
            # Create fingerprint
            fingerprint = VideoFingerprint(
                video_hash=video_hash,
                perceptual_hash=perceptual_hash,
                temporal_hash=temporal_hash,
                audio_fingerprint=audio_fingerprint,
                frame_count=len(video_data['frames']),
                duration_seconds=video_data['duration'],
                resolution=(video_data['width'], video_data['height']),
                average_brightness=content_features['avg_brightness'],
                algorithm_used=self.algorithm,
                hash_size=self.hash_size,
                sample_rate=self.sample_rate,
                exact_match_threshold=self.exact_threshold,
                similar_match_threshold=self.similar_threshold,
                key_frames_hashes=key_frame_hashes,
                motion_vectors=motion_vectors,
                color_histograms=color_histograms,
                processing_time_seconds=(datetime.utcnow() - start_time).total_seconds(),
                quality_score=content_features['quality_score']
            )
            
            self.logger.info(f"Video fingerprinting completed: {len(key_frame_hashes)} key frames")
            return fingerprint
            
        except Exception as e:
            self.logger.error(f"Video fingerprinting failed: {str(e)}")
            raise
    
    def _calculate_video_hash(self, video_path: Path) -> str:
        """Calculate SHA-256 hash of video file"""
        hash_obj = hashlib.sha256()
        with open(video_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_obj.update(chunk)
        return hash_obj.hexdigest()
    
    def _extract_video_data(self, video_path: Path) -> Dict[str, Any]:
        """Extract frames and metadata from video"""
        cap = cv2.VideoCapture(str(video_path))
        frames = []
        
        try:
            # Get video properties
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration = total_frames / fps if fps > 0 else 0
            
            frame_count = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Sample frames based on interval
                if frame_count % self.frame_analysis_interval == 0:
                    # Convert BGR to RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(frame_rgb)
                
                frame_count += 1
                
                # Limit frames to prevent memory issues
                if len(frames) >= 500:  # Max 500 sampled frames
                    break
            
            return {
                'frames': frames,
                'total_frames': total_frames,
                'fps': fps,
                'width': width,
                'height': height,
                'duration': duration
            }
            
        finally:
            cap.release()
    
    def _generate_perceptual_hash(self, frames: List[np.ndarray]) -> str:
        """Generate perceptual hash using selected algorithm"""
        if not IMAGEHASH_AVAILABLE:
            return self._fallback_perceptual_hash(frames)
        
        # Sample frames for hashing (use every 10th frame to reduce computation)
        sampled_frames = frames[::max(1, len(frames) // 20)]  # Max 20 frames
        
        hashes = []
        for frame in sampled_frames:
            # Convert to PIL Image
            pil_image = Image.fromarray(frame)
            
            # Generate hash based on algorithm
            if self.algorithm == "dhash":
                frame_hash = imagehash.dhash(pil_image, hash_size=self.hash_size)
            elif self.algorithm == "phash":
                frame_hash = imagehash.phash(pil_image, hash_size=self.hash_size)
            elif self.algorithm == "ahash":
                frame_hash = imagehash.average_hash(pil_image, hash_size=self.hash_size)
            elif self.algorithm == "whash":
                frame_hash = imagehash.whash(pil_image, hash_size=self.hash_size)
            else:
                frame_hash = imagehash.dhash(pil_image, hash_size=self.hash_size)
            
            hashes.append(str(frame_hash))
        
        # Combine hashes into single representation
        combined_hash = hashlib.md5(''.join(hashes).encode()).hexdigest()
        return combined_hash
    
    def _fallback_perceptual_hash(self, frames: List[np.ndarray]) -> str:
        """Fallback perceptual hash when imagehash is not available"""
        self.logger.warning("Using fallback perceptual hash (no imagehash library)")
        
        # Simple average brightness hash
        sampled_frames = frames[::max(1, len(frames) // 20)]
        
        hash_bits = []
        for frame in sampled_frames:
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            
            # Resize to small size
            small = cv2.resize(gray, (self.hash_size, self.hash_size))
            
            # Calculate average
            avg = np.mean(small)
            
            # Generate binary hash
            binary = (small > avg).astype(int)
            hash_bits.extend(binary.flatten())
        
        # Convert to hex string
        hex_hash = hex(int(''.join(map(str, hash_bits[:64])), 2))[2:].zfill(16)
        return hex_hash
    
    def _generate_temporal_hash(self, frames: List[np.ndarray]) -> str:
        """Generate hash based on temporal features"""
        if len(frames) < 2:
            return "0" * 32
        
        temporal_features = []
        
        # Calculate frame differences
        for i in range(1, len(frames)):
            prev_gray = cv2.cvtColor(frames[i-1], cv2.COLOR_RGB2GRAY)
            curr_gray = cv2.cvtColor(frames[i], cv2.COLOR_RGB2GRAY)
            
            # Calculate difference
            diff = cv2.absdiff(prev_gray, curr_gray)
            
            # Reduce to smaller representation
            small_diff = cv2.resize(diff, (8, 8))
            temporal_features.extend(small_diff.flatten())
        
        # Normalize and quantize
        if temporal_features:
            features_array = np.array(temporal_features)
            normalized = (features_array / (np.max(features_array) + 1e-6) * 255).astype(int)
            
            # Generate hash
            hash_input = ''.join(map(str, normalized[:256]))  # Limit size
            temporal_hash = hashlib.md5(hash_input.encode()).hexdigest()
            return temporal_hash
        
        return "0" * 32
    
    def _extract_audio_fingerprint(self, video_path: Path) -> Optional[str]:
        """Extract audio fingerprint from video"""
        try:
            # Load video with moviepy
            video = mp.VideoFileClip(str(video_path))
            
            if video.audio is None:
                return None
            
            # Extract audio samples
            audio = video.audio
            duration = min(audio.duration, 30.0)  # Max 30 seconds
            
            # Get audio array
            audio_array = audio.subclip(0, duration).to_soundarray()
            
            # Simple audio fingerprint based on spectral features
            if len(audio_array.shape) > 1:
                # Convert to mono
                audio_mono = np.mean(audio_array, axis=1)
            else:
                audio_mono = audio_array
            
            # Calculate frequency domain features
            fft = np.fft.fft(audio_mono[:min(len(audio_mono), 44100)])  # 1 second
            magnitude = np.abs(fft)
            
            # Reduce dimensionality
            bins = np.logspace(0, np.log10(len(magnitude)//2), 32, dtype=int)
            binned_magnitude = [np.mean(magnitude[bins[i]:bins[i+1]]) for i in range(len(bins)-1)]
            
            # Generate hash
            normalized = (np.array(binned_magnitude) / (np.max(binned_magnitude) + 1e-6) * 255).astype(int)
            audio_hash = hashlib.md5(''.join(map(str, normalized)).encode()).hexdigest()
            
            video.close()
            return audio_hash
            
        except Exception as e:
            self.logger.warning(f"Failed to extract audio fingerprint: {str(e)}")
            return None
    
    def _calculate_content_features(self, frames: List[np.ndarray]) -> Dict[str, float]:
        """Calculate content-based features"""
        if not frames:
            return {'avg_brightness': 0.0, 'quality_score': 0.0}
        
        brightness_values = []
        sharpness_values = []
        
        for frame in frames[::5]:  # Sample every 5th frame
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            
            # Calculate brightness
            brightness = np.mean(gray) / 255.0
            brightness_values.append(brightness)
            
            # Calculate sharpness (Laplacian variance)
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            sharpness = laplacian.var()
            sharpness_values.append(sharpness)
        
        avg_brightness = np.mean(brightness_values)
        avg_sharpness = np.mean(sharpness_values)
        
        # Quality score based on sharpness and brightness consistency
        brightness_consistency = 1.0 - np.std(brightness_values)
        sharpness_score = min(1.0, avg_sharpness / 1000.0)  # Normalize
        quality_score = (brightness_consistency + sharpness_score) / 2.0
        
        return {
            'avg_brightness': avg_brightness,
            'quality_score': max(0.0, min(1.0, quality_score))
        }
    
    def _extract_key_frame_hashes(self, frames: List[np.ndarray]) -> List[str]:
        """Extract hashes for key frames"""
        if not frames:
            return []
        
        # Use scene change detection to identify key frames
        key_frame_indices = self._detect_scene_changes(frames)
        
        key_hashes = []
        for idx in key_frame_indices:
            if idx < len(frames):
                frame = frames[idx]
                # Generate simple hash for the frame
                gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                small = cv2.resize(gray, (16, 16))
                frame_hash = hashlib.md5(small.tobytes()).hexdigest()
                key_hashes.append(frame_hash)
        
        return key_hashes
    
    def _detect_scene_changes(self, frames: List[np.ndarray]) -> List[int]:
        """Detect scene changes to identify key frames"""
        if len(frames) < 2:
            return [0] if frames else []
        
        scene_changes = [0]  # First frame is always a key frame
        threshold = 30.0  # Threshold for scene change
        
        for i in range(1, len(frames)):
            prev_gray = cv2.cvtColor(frames[i-1], cv2.COLOR_RGB2GRAY)
            curr_gray = cv2.cvtColor(frames[i], cv2.COLOR_RGB2GRAY)
            
            # Calculate histogram difference
            hist1 = cv2.calcHist([prev_gray], [0], None, [256], [0, 256])
            hist2 = cv2.calcHist([curr_gray], [0], None, [256], [0, 256])
            
            # Compare histograms
            correlation = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
            
            # If correlation is low, it's a scene change
            if correlation < 0.7:  # Threshold for scene change
                scene_changes.append(i)
        
        # Limit number of key frames
        if len(scene_changes) > 50:
            # Sample evenly from detected changes
            step = len(scene_changes) // 50
            scene_changes = scene_changes[::step]
        
        return scene_changes
    
    def _calculate_motion_vectors(self, frames: List[np.ndarray]) -> List[float]:
        """Calculate motion vectors between frames"""
        if len(frames) < 2:
            return []
        
        motion_vectors = []
        
        for i in range(1, len(frames)):
            prev_gray = cv2.cvtColor(frames[i-1], cv2.COLOR_RGB2GRAY)
            curr_gray = cv2.cvtColor(frames[i], cv2.COLOR_RGB2GRAY)
            
            # Calculate optical flow
            flow = cv2.calcOpticalFlowPyrLK(
                prev_gray, curr_gray,
                corners=cv2.goodFeaturesToTrack(prev_gray, maxCorners=100, qualityLevel=0.01, minDistance=10),
                nextPts=None
            )[1]
            
            if flow is not None and len(flow) > 0:
                # Calculate average motion magnitude
                motion_magnitude = np.mean(np.sqrt(np.sum(flow**2, axis=2)))
                motion_vectors.append(float(motion_magnitude))
            else:
                motion_vectors.append(0.0)
        
        return motion_vectors
    
    def _generate_color_histograms(self, frames: List[np.ndarray]) -> List[List[float]]:
        """Generate color histograms for frames"""
        histograms = []
        
        # Sample frames for histogram calculation
        sampled_frames = frames[::max(1, len(frames) // 10)]  # Max 10 histograms
        
        for frame in sampled_frames:
            # Calculate histogram for each color channel
            hist_r = cv2.calcHist([frame], [0], None, [self.color_histogram_bins], [0, 256])
            hist_g = cv2.calcHist([frame], [1], None, [self.color_histogram_bins], [0, 256])
            hist_b = cv2.calcHist([frame], [2], None, [self.color_histogram_bins], [0, 256])
            
            # Normalize and flatten
            hist_r = hist_r.flatten() / np.sum(hist_r)
            hist_g = hist_g.flatten() / np.sum(hist_g)
            hist_b = hist_b.flatten() / np.sum(hist_b)
            
            # Combine channels
            combined_hist = np.concatenate([hist_r, hist_g, hist_b])
            histograms.append(combined_hist.tolist())
        
        return histograms
    
    def compare_fingerprints(self, 
                           fingerprint1: VideoFingerprint, 
                           fingerprint2: VideoFingerprint) -> Dict[str, Any]:
        """Compare two video fingerprints for similarity"""
        similarity_score = fingerprint1.calculate_similarity(fingerprint2)
        
        # Additional similarity metrics
        additional_metrics = {}
        
        # Temporal hash similarity
        if fingerprint1.temporal_hash and fingerprint2.temporal_hash:
            temporal_similarity = self._compare_hashes(
                fingerprint1.temporal_hash, 
                fingerprint2.temporal_hash
            )
            additional_metrics['temporal_similarity'] = temporal_similarity
        
        # Audio fingerprint similarity
        if fingerprint1.audio_fingerprint and fingerprint2.audio_fingerprint:
            audio_similarity = self._compare_hashes(
                fingerprint1.audio_fingerprint,
                fingerprint2.audio_fingerprint
            )
            additional_metrics['audio_similarity'] = audio_similarity
        
        # Duration similarity
        duration_diff = abs(fingerprint1.duration_seconds - fingerprint2.duration_seconds)
        duration_similarity = 1.0 - min(1.0, duration_diff / max(fingerprint1.duration_seconds, fingerprint2.duration_seconds))
        additional_metrics['duration_similarity'] = duration_similarity
        
        # Resolution similarity
        res1 = fingerprint1.resolution
        res2 = fingerprint2.resolution
        resolution_similarity = 1.0 - abs(res1[0] * res1[1] - res2[0] * res2[1]) / max(res1[0] * res1[1], res2[0] * res2[1])
        additional_metrics['resolution_similarity'] = resolution_similarity
        
        # Overall similarity (weighted average)
        weights = {
            'perceptual': 0.4,
            'temporal': 0.2,
            'audio': 0.2,
            'duration': 0.1,
            'resolution': 0.1
        }
        
        weighted_similarity = similarity_score * weights['perceptual']
        
        if 'temporal_similarity' in additional_metrics:
            weighted_similarity += additional_metrics['temporal_similarity'] * weights['temporal']
        
        if 'audio_similarity' in additional_metrics:
            weighted_similarity += additional_metrics['audio_similarity'] * weights['audio']
        
        weighted_similarity += duration_similarity * weights['duration']
        weighted_similarity += resolution_similarity * weights['resolution']
        
        return {
            'similarity_score': similarity_score,
            'weighted_similarity': weighted_similarity,
            'is_exact_match': weighted_similarity >= fingerprint1.exact_match_threshold,
            'is_similar_match': weighted_similarity >= fingerprint1.similar_match_threshold,
            'additional_metrics': additional_metrics,
            'comparison_timestamp': datetime.utcnow()
        }
    
    def _compare_hashes(self, hash1: str, hash2: str) -> float:
        """Compare two hash strings for similarity"""
        if len(hash1) != len(hash2):
            return 0.0
        
        # Hamming distance for hex strings
        differences = sum(c1 != c2 for c1, c2 in zip(hash1, hash2))
        similarity = 1.0 - (differences / len(hash1))
        
        return similarity
    
    def find_similar_videos(self, 
                           target_fingerprint: VideoFingerprint,
                           fingerprint_database: List[VideoFingerprint],
                           threshold: float = 0.8) -> List[Dict[str, Any]]:
        """Find similar videos in a database of fingerprints"""
        similar_videos = []
        
        for db_fingerprint in fingerprint_database:
            comparison = self.compare_fingerprints(target_fingerprint, db_fingerprint)
            
            if comparison['weighted_similarity'] >= threshold:
                similar_videos.append({
                    'fingerprint': db_fingerprint,
                    'similarity': comparison['weighted_similarity'],
                    'comparison_details': comparison
                })
        
        # Sort by similarity (highest first)
        similar_videos.sort(key=lambda x: x['similarity'], reverse=True)
        
        return similar_videos
    
    def create_fingerprint_database(self, video_paths: List[Path]) -> List[VideoFingerprint]:
        """Create a database of fingerprints from multiple videos"""
        fingerprints = []
        
        for video_path in video_paths:
            try:
                fingerprint = self.fingerprint_video(video_path)
                fingerprints.append(fingerprint)
                self.logger.info(f"Added fingerprint for: {video_path}")
            except Exception as e:
                self.logger.error(f"Failed to fingerprint {video_path}: {str(e)}")
        
        return fingerprints
    
    def export_fingerprint_database(self, 
                                  fingerprints: List[VideoFingerprint],
                                  output_path: Path) -> bool:
        """Export fingerprint database to file"""
        try:
            database = {
                'created_at': datetime.utcnow().isoformat(),
                'total_fingerprints': len(fingerprints),
                'fingerprints': [fp.dict() for fp in fingerprints]
            }
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(database, f, indent=2, default=str)
            
            self.logger.info(f"Fingerprint database exported to: {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to export database: {str(e)}")
            return False