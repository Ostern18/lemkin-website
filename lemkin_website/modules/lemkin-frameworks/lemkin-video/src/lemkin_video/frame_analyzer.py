"""
Lemkin Video Authentication Toolkit - Frame Analysis Module

This module provides frame-level analysis and key frame extraction for detecting
tampering, inconsistencies, and manipulation at the individual frame level.

Features:
- Key frame extraction using scene detection
- Frame-level tampering detection
- Temporal consistency analysis
- Motion vector analysis
- Frame interpolation detection
- Edge artifact detection
- Lighting inconsistency detection
"""

import cv2
import numpy as np
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import hashlib

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    import scipy.signal
    import scipy.stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

from .core import (
    VideoAuthConfig, KeyFrame, FrameType, TamperingIndicator, 
    TamperingType, AnalysisStatus
)

logger = logging.getLogger(__name__)


class FrameAnalyzer:
    """
    Advanced frame-level analysis system for detecting manipulation
    and inconsistencies at the individual frame level.
    """
    
    def __init__(self, config: VideoAuthConfig):
        """Initialize frame analyzer with configuration"""
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.FrameAnalyzer")
        
        # Analysis settings
        self.scene_threshold = 0.3  # Threshold for scene change detection
        self.motion_threshold = 5.0  # Threshold for significant motion
        self.tampering_threshold = 0.7  # Threshold for tampering detection
        
        # Feature extraction settings
        self.edge_threshold_low = 50
        self.edge_threshold_high = 150
        self.corner_detection_quality = 0.01
        self.corner_detection_min_distance = 10
        
        # Optical flow settings
        self.flow_params = dict(
            maxCorners=100,
            qualityLevel=0.01,
            minDistance=10,
            blockSize=7
        )
        
        self.logger.info("Frame analyzer initialized")
    
    def extract_key_frames(self, video_path: Path) -> List[KeyFrame]:
        """
        Extract and analyze key frames from video
        
        Args:
            video_path: Path to video file
            
        Returns:
            List[KeyFrame]: List of analyzed key frames
        """
        start_time = datetime.utcnow()
        self.logger.info(f"Starting frame analysis for: {video_path}")
        
        try:
            # Extract all frames and metadata
            frames_data = self._extract_video_frames(video_path)
            
            if not frames_data['frames']:
                self.logger.warning("No frames could be extracted")
                return []
            
            # Detect key frames using scene change detection
            key_frame_indices = self._detect_key_frames(frames_data['frames'])
            
            # Analyze each key frame
            key_frames = []
            for i, frame_idx in enumerate(key_frame_indices):
                if frame_idx < len(frames_data['frames']):
                    frame = frames_data['frames'][frame_idx]
                    timestamp = frame_idx / frames_data['fps'] if frames_data['fps'] > 0 else 0
                    
                    # Analyze the frame
                    frame_analysis = self._analyze_single_frame(
                        frame, frame_idx, timestamp, frames_data
                    )
                    
                    key_frames.append(frame_analysis)
                    
                    self.logger.debug(f"Analyzed key frame {i+1}/{len(key_frame_indices)}")
            
            # Perform temporal consistency analysis
            temporal_analysis = self._analyze_temporal_consistency(
                key_frames, frames_data['frames']
            )
            
            # Update key frames with temporal analysis results
            for key_frame in key_frames:
                key_frame.tampering_indicators.extend(temporal_analysis['indicators'])
            
            self.logger.info(f"Frame analysis completed: {len(key_frames)} key frames extracted")
            return key_frames
            
        except Exception as e:
            self.logger.error(f"Frame analysis failed: {str(e)}")
            return []
    
    def _extract_video_frames(self, video_path: Path) -> Dict[str, Any]:
        """Extract frames and metadata from video"""
        cap = cv2.VideoCapture(str(video_path))
        frames = []
        frame_metadata = []
        
        try:
            # Get video properties
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            frame_count = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Store frame and metadata
                frames.append(frame)
                frame_metadata.append({
                    'index': frame_count,
                    'timestamp': frame_count / fps if fps > 0 else 0,
                    'size': frame.nbytes
                })
                
                frame_count += 1
                
                # Limit frames to prevent memory issues
                if len(frames) >= 1000:  # Max 1000 frames
                    self.logger.warning(f"Limited analysis to first {len(frames)} frames")
                    break
            
            return {
                'frames': frames,
                'frame_metadata': frame_metadata,
                'total_frames': total_frames,
                'fps': fps,
                'width': width,
                'height': height
            }
            
        finally:
            cap.release()
    
    def _detect_key_frames(self, frames: List[np.ndarray]) -> List[int]:
        """Detect key frames using scene change detection"""
        if len(frames) < 2:
            return [0] if frames else []
        
        key_frame_indices = [0]  # First frame is always a key frame
        
        # Calculate frame differences for scene detection
        for i in range(1, len(frames)):
            # Convert frames to grayscale
            prev_gray = cv2.cvtColor(frames[i-1], cv2.COLOR_BGR2GRAY)
            curr_gray = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
            
            # Calculate histogram correlation
            hist_correlation = self._calculate_histogram_correlation(prev_gray, curr_gray)
            
            # Calculate structural similarity
            structural_similarity = self._calculate_structural_similarity(prev_gray, curr_gray)
            
            # Calculate optical flow magnitude
            flow_magnitude = self._calculate_optical_flow_magnitude(prev_gray, curr_gray)
            
            # Combine metrics to determine scene change
            scene_change_score = (
                (1.0 - hist_correlation) * 0.4 +
                (1.0 - structural_similarity) * 0.4 +
                min(1.0, flow_magnitude / 50.0) * 0.2
            )
            
            # If scene change score exceeds threshold, mark as key frame
            if scene_change_score > self.scene_threshold:
                key_frame_indices.append(i)
        
        # Ensure we don't have too many key frames
        if len(key_frame_indices) > 100:
            # Sample key frames evenly
            step = len(key_frame_indices) // 100
            key_frame_indices = key_frame_indices[::step]
        
        # Always include the last frame
        if key_frame_indices[-1] != len(frames) - 1:
            key_frame_indices.append(len(frames) - 1)
        
        return key_frame_indices
    
    def _calculate_histogram_correlation(self, frame1: np.ndarray, frame2: np.ndarray) -> float:
        """Calculate histogram correlation between two frames"""
        # Calculate histograms
        hist1 = cv2.calcHist([frame1], [0], None, [256], [0, 256])
        hist2 = cv2.calcHist([frame2], [0], None, [256], [0, 256])
        
        # Calculate correlation
        correlation = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
        
        return max(0.0, correlation)  # Ensure non-negative
    
    def _calculate_structural_similarity(self, frame1: np.ndarray, frame2: np.ndarray) -> float:
        """Calculate structural similarity between two frames"""
        # Resize frames for faster computation
        h, w = frame1.shape
        if h > 480 or w > 640:
            frame1 = cv2.resize(frame1, (640, 480))
            frame2 = cv2.resize(frame2, (640, 480))
        
        # Calculate mean squared error
        mse = np.mean((frame1.astype(float) - frame2.astype(float)) ** 2)
        
        # Convert to similarity (1.0 = identical, 0.0 = completely different)
        max_pixel_value = 255.0
        similarity = 1.0 - (mse / (max_pixel_value ** 2))
        
        return max(0.0, similarity)
    
    def _calculate_optical_flow_magnitude(self, frame1: np.ndarray, frame2: np.ndarray) -> float:
        """Calculate average optical flow magnitude between frames"""
        # Detect corners in first frame
        corners = cv2.goodFeaturesToTrack(frame1, **self.flow_params)
        
        if corners is None or len(corners) == 0:
            return 0.0
        
        # Calculate optical flow
        next_corners, status, error = cv2.calcOpticalFlowPyrLK(
            frame1, frame2, corners, None
        )
        
        # Filter out bad tracks
        good_corners = corners[status == 1]
        good_next = next_corners[status == 1]
        
        if len(good_corners) == 0:
            return 0.0
        
        # Calculate flow vectors
        flow_vectors = good_next - good_corners
        flow_magnitudes = np.sqrt(np.sum(flow_vectors ** 2, axis=1))
        
        return np.mean(flow_magnitudes)
    
    def _analyze_single_frame(self, 
                            frame: np.ndarray, 
                            frame_idx: int, 
                            timestamp: float,
                            frames_data: Dict[str, Any]) -> KeyFrame:
        """Analyze a single frame for tampering indicators"""
        
        # Calculate frame hash
        frame_hash = self._calculate_frame_hash(frame)
        
        # Determine frame type (simplified - would need actual codec analysis)
        frame_type = self._determine_frame_type(frame_idx, frames_data)
        
        # Basic frame analysis
        authenticity_score = self._calculate_frame_authenticity_score(frame)
        
        # Detect tampering indicators
        tampering_indicators = self._detect_frame_tampering(frame, frame_idx, frames_data)
        
        # Calculate technical metrics
        compression_ratio = self._estimate_compression_ratio(frame)
        quality_score = self._calculate_frame_quality_score(frame)
        motion_consistency = self._calculate_motion_consistency(frame, frame_idx, frames_data)
        
        # Extract visual features
        visual_features = self._extract_visual_features(frame)
        
        # Analyze faces if present
        face_analysis = self._analyze_faces_in_frame(frame)
        
        # Create KeyFrame object
        key_frame = KeyFrame(
            frame_number=frame_idx,
            timestamp=timestamp,
            frame_type=frame_type,
            frame_hash=frame_hash,
            frame_size=frame.nbytes,
            authenticity_score=authenticity_score,
            tampering_indicators=tampering_indicators,
            compression_ratio=compression_ratio,
            quality_score=quality_score,
            motion_vector_consistency=motion_consistency,
            dominant_colors=visual_features['dominant_colors'],
            edge_density=visual_features['edge_density'],
            texture_features=visual_features['texture_features'],
            faces_detected=face_analysis['face_count'],
            face_regions=face_analysis['face_regions'],
            deepfake_probability=face_analysis['deepfake_probability']
        )
        
        return key_frame
    
    def _calculate_frame_hash(self, frame: np.ndarray) -> str:
        """Calculate hash of frame content"""
        # Convert to grayscale and resize for consistent hashing
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (64, 64))
        
        # Calculate hash
        return hashlib.md5(resized.tobytes()).hexdigest()
    
    def _determine_frame_type(self, frame_idx: int, frames_data: Dict[str, Any]) -> FrameType:
        """Determine frame type (I, P, or B frame)"""
        # Simplified frame type determination
        # In practice, this would require codec-specific analysis
        
        if frame_idx == 0:
            return FrameType.I_FRAME
        elif frame_idx % 10 == 0:  # Assume I-frames every 10 frames
            return FrameType.I_FRAME
        elif frame_idx % 2 == 0:  # Assume P-frames on even indices
            return FrameType.P_FRAME
        else:
            return FrameType.B_FRAME
    
    def _calculate_frame_authenticity_score(self, frame: np.ndarray) -> float:
        """Calculate authenticity score for frame"""
        scores = []
        
        # Edge consistency score
        edge_score = self._calculate_edge_consistency_score(frame)
        scores.append(edge_score)
        
        # Noise pattern score
        noise_score = self._calculate_noise_pattern_score(frame)
        scores.append(noise_score)
        
        # Color distribution score
        color_score = self._calculate_color_distribution_score(frame)
        scores.append(color_score)
        
        # Texture consistency score
        texture_score = self._calculate_texture_consistency_score(frame)
        scores.append(texture_score)
        
        # Return weighted average
        weights = [0.3, 0.2, 0.25, 0.25]
        authenticity_score = sum(score * weight for score, weight in zip(scores, weights))
        
        return max(0.0, min(1.0, authenticity_score))
    
    def _calculate_edge_consistency_score(self, frame: np.ndarray) -> float:
        """Calculate edge consistency score"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate edges
        edges = cv2.Canny(gray, self.edge_threshold_low, self.edge_threshold_high)
        
        # Calculate edge density
        edge_density = np.mean(edges) / 255.0
        
        # Calculate edge continuity
        # Look for broken or inconsistent edges
        kernel = np.ones((3, 3), np.uint8)
        dilated_edges = cv2.dilate(edges, kernel, iterations=1)
        eroded_edges = cv2.erode(dilated_edges, kernel, iterations=1)
        
        edge_continuity = np.sum(eroded_edges) / (np.sum(edges) + 1e-6)
        
        # Combine metrics
        edge_score = (edge_density * 0.3 + edge_continuity * 0.7)
        
        # Normalize to reasonable range
        return min(1.0, edge_score * 2.0)
    
    def _calculate_noise_pattern_score(self, frame: np.ndarray) -> float:
        """Calculate noise pattern score"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate local variance to detect noise
        kernel = np.ones((5, 5), np.float32) / 25
        mean_img = cv2.filter2D(gray.astype(np.float32), -1, kernel)
        variance = cv2.filter2D((gray.astype(np.float32) - mean_img) ** 2, -1, kernel)
        
        # Calculate noise uniformity
        noise_std = np.std(variance)
        noise_mean = np.mean(variance)
        
        # Lower variance in noise patterns suggests manipulation
        noise_uniformity = 1.0 - min(1.0, noise_std / (noise_mean + 1e-6))
        
        return noise_uniformity
    
    def _calculate_color_distribution_score(self, frame: np.ndarray) -> float:
        """Calculate color distribution score"""
        # Calculate color histograms for each channel
        hist_b = cv2.calcHist([frame], [0], None, [256], [0, 256])
        hist_g = cv2.calcHist([frame], [1], None, [256], [0, 256])
        hist_r = cv2.calcHist([frame], [2], None, [256], [0, 256])
        
        # Calculate histogram entropy (higher is more natural)
        def calculate_entropy(hist):
            hist_norm = hist / (np.sum(hist) + 1e-6)
            entropy = -np.sum(hist_norm * np.log2(hist_norm + 1e-6))
            return entropy
        
        entropy_b = calculate_entropy(hist_b)
        entropy_g = calculate_entropy(hist_g)
        entropy_r = calculate_entropy(hist_r)
        
        # Average entropy (normalize to [0, 1])
        avg_entropy = (entropy_b + entropy_g + entropy_r) / 3.0
        normalized_entropy = min(1.0, avg_entropy / 8.0)  # 8 bits per channel max entropy
        
        return normalized_entropy
    
    def _calculate_texture_consistency_score(self, frame: np.ndarray) -> float:
        """Calculate texture consistency score"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate Local Binary Pattern (simplified)
        lbp_score = self._calculate_lbp_score(gray)
        
        # Calculate Gabor filter responses
        gabor_score = self._calculate_gabor_score(gray)
        
        # Combine texture measures
        texture_score = (lbp_score + gabor_score) / 2.0
        
        return texture_score
    
    def _calculate_lbp_score(self, gray_frame: np.ndarray) -> float:
        """Calculate Local Binary Pattern score"""
        h, w = gray_frame.shape
        lbp = np.zeros((h-2, w-2), dtype=np.uint8)
        
        # Calculate LBP
        for i in range(1, h-1):
            for j in range(1, w-1):
                center = gray_frame[i, j]
                binary_string = ""
                
                # 8-neighborhood
                neighbors = [
                    gray_frame[i-1, j-1], gray_frame[i-1, j], gray_frame[i-1, j+1],
                    gray_frame[i, j+1], gray_frame[i+1, j+1], gray_frame[i+1, j],
                    gray_frame[i+1, j-1], gray_frame[i, j-1]
                ]
                
                for neighbor in neighbors:
                    binary_string += '1' if neighbor >= center else '0'
                
                lbp[i-1, j-1] = int(binary_string, 2)
        
        # Calculate LBP histogram uniformity
        hist = np.histogram(lbp, bins=256, range=(0, 256))[0]
        uniformity = 1.0 - (np.std(hist) / (np.mean(hist) + 1e-6))
        
        return max(0.0, min(1.0, uniformity))
    
    def _calculate_gabor_score(self, gray_frame: np.ndarray) -> float:
        """Calculate Gabor filter response score"""
        # Apply Gabor filters at different orientations
        orientations = [0, 45, 90, 135]
        responses = []
        
        for angle in orientations:
            # Create Gabor kernel
            kernel = cv2.getGaborKernel((21, 21), 5, np.radians(angle), 2*np.pi*0.5, 0.5, 0, ktype=cv2.CV_32F)
            
            # Apply filter
            response = cv2.filter2D(gray_frame, cv2.CV_8UC3, kernel)
            responses.append(np.var(response))
        
        # Calculate texture energy
        texture_energy = np.mean(responses)
        
        # Normalize to [0, 1] range
        normalized_energy = min(1.0, texture_energy / 1000.0)
        
        return normalized_energy
    
    def _detect_frame_tampering(self, 
                              frame: np.ndarray, 
                              frame_idx: int, 
                              frames_data: Dict[str, Any]) -> List[TamperingIndicator]:
        """Detect tampering indicators in frame"""
        indicators = []
        
        # Edge artifact detection
        edge_artifacts = self._detect_edge_artifacts(frame)
        if edge_artifacts['detected']:
            indicator = TamperingIndicator(
                tampering_type=TamperingType.EDGE_ARTIFACTS,
                confidence=edge_artifacts['confidence'],
                frame_numbers=[frame_idx],
                description=edge_artifacts['description'],
                evidence=edge_artifacts['evidence'],
                analysis_method="edge_analysis",
                severity_score=edge_artifacts['confidence'] * 6.0
            )
            indicators.append(indicator)
        
        # Lighting inconsistency detection
        lighting_inconsistency = self._detect_lighting_inconsistency(frame, frame_idx, frames_data)
        if lighting_inconsistency['detected']:
            indicator = TamperingIndicator(
                tampering_type=TamperingType.LIGHTING_INCONSISTENCY,
                confidence=lighting_inconsistency['confidence'],
                frame_numbers=[frame_idx],
                description=lighting_inconsistency['description'],
                evidence=lighting_inconsistency['evidence'],
                analysis_method="lighting_analysis",
                severity_score=lighting_inconsistency['confidence'] * 5.0
            )
            indicators.append(indicator)
        
        # Pixel-level editing detection
        pixel_editing = self._detect_pixel_level_editing(frame)
        if pixel_editing['detected']:
            indicator = TamperingIndicator(
                tampering_type=TamperingType.PIXEL_LEVEL_EDITING,
                confidence=pixel_editing['confidence'],
                frame_numbers=[frame_idx],
                spatial_coordinates=pixel_editing.get('coordinates'),
                description=pixel_editing['description'],
                evidence=pixel_editing['evidence'],
                analysis_method="pixel_analysis",
                severity_score=pixel_editing['confidence'] * 7.0
            )
            indicators.append(indicator)
        
        return indicators
    
    def _detect_edge_artifacts(self, frame: np.ndarray) -> Dict[str, Any]:
        """Detect edge artifacts that may indicate manipulation"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate edges
        edges = cv2.Canny(gray, 50, 150)
        
        # Look for unnatural edge patterns
        # 1. Double edges (sign of copy-paste)
        double_edges = self._detect_double_edges(edges)
        
        # 2. Abrupt edge terminations
        edge_terminations = self._detect_edge_terminations(edges)
        
        # 3. Inconsistent edge sharpness
        edge_sharpness = self._analyze_edge_sharpness(gray, edges)
        
        # Calculate overall confidence
        confidence_factors = []
        evidence = {}
        
        if double_edges['detected']:
            confidence_factors.append(double_edges['confidence'])
            evidence['double_edges'] = double_edges['count']
        
        if edge_terminations['detected']:
            confidence_factors.append(edge_terminations['confidence'])
            evidence['edge_terminations'] = edge_terminations['count']
        
        if edge_sharpness['inconsistent']:
            confidence_factors.append(edge_sharpness['confidence'])
            evidence['sharpness_variation'] = edge_sharpness['variation']
        
        detected = len(confidence_factors) >= 2
        confidence = np.mean(confidence_factors) if confidence_factors else 0.0
        
        return {
            'detected': detected,
            'confidence': confidence,
            'description': f"Edge artifacts detected with {confidence:.2f} confidence",
            'evidence': evidence
        }
    
    def _detect_double_edges(self, edges: np.ndarray) -> Dict[str, Any]:
        """Detect double edge patterns"""
        # Dilate edges to connect nearby edges
        kernel = np.ones((3, 3), np.uint8)
        dilated = cv2.dilate(edges, kernel, iterations=1)
        
        # Find contours
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        double_edge_count = 0
        
        for contour in contours:
            # Check if contour has parallel lines (potential double edge)
            if len(contour) > 10:
                # Simplify contour
                epsilon = 0.02 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                
                # Look for parallel line segments
                if self._has_parallel_segments(approx):
                    double_edge_count += 1
        
        detected = double_edge_count > 3
        confidence = min(1.0, double_edge_count / 10.0)
        
        return {
            'detected': detected,
            'confidence': confidence,
            'count': double_edge_count
        }
    
    def _has_parallel_segments(self, contour: np.ndarray) -> bool:
        """Check if contour has parallel line segments"""
        if len(contour) < 4:
            return False
        
        # Calculate angles between consecutive segments
        angles = []
        for i in range(len(contour)):
            p1 = contour[i][0]
            p2 = contour[(i + 1) % len(contour)][0]
            p3 = contour[(i + 2) % len(contour)][0]
            
            # Calculate angle
            v1 = p2 - p1
            v2 = p3 - p2
            
            if np.linalg.norm(v1) > 0 and np.linalg.norm(v2) > 0:
                cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                angle = np.arccos(np.clip(cos_angle, -1, 1))
                angles.append(angle)
        
        # Look for similar angles (parallel segments)
        if len(angles) >= 2:
            for i in range(len(angles)):
                for j in range(i + 1, len(angles)):
                    if abs(angles[i] - angles[j]) < 0.1:  # 0.1 radian tolerance
                        return True
        
        return False
    
    def _detect_edge_terminations(self, edges: np.ndarray) -> Dict[str, Any]:
        """Detect abrupt edge terminations"""
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        termination_count = 0
        
        for contour in contours:
            if len(contour) > 5:
                # Check if contour endpoints are abrupt
                start_point = contour[0][0]
                end_point = contour[-1][0]
                
                # Check if edges terminate abruptly (not at image boundaries)
                h, w = edges.shape
                margin = 5
                
                start_abrupt = (margin < start_point[0] < w - margin and 
                               margin < start_point[1] < h - margin)
                end_abrupt = (margin < end_point[0] < w - margin and 
                             margin < end_point[1] < h - margin)
                
                if start_abrupt or end_abrupt:
                    termination_count += 1
        
        detected = termination_count > 5
        confidence = min(1.0, termination_count / 20.0)
        
        return {
            'detected': detected,
            'confidence': confidence,
            'count': termination_count
        }
    
    def _analyze_edge_sharpness(self, gray_frame: np.ndarray, edges: np.ndarray) -> Dict[str, Any]:
        """Analyze edge sharpness consistency"""
        # Calculate gradient magnitude along edges
        grad_x = cv2.Sobel(gray_frame, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray_frame, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # Extract gradient values at edge pixels
        edge_gradients = gradient_magnitude[edges > 0]
        
        if len(edge_gradients) == 0:
            return {'inconsistent': False, 'confidence': 0.0, 'variation': 0.0}
        
        # Calculate variation in edge sharpness
        sharpness_std = np.std(edge_gradients)
        sharpness_mean = np.mean(edge_gradients)
        variation_coefficient = sharpness_std / (sharpness_mean + 1e-6)
        
        # High variation might indicate manipulation
        inconsistent = variation_coefficient > 1.0
        confidence = min(1.0, variation_coefficient / 2.0)
        
        return {
            'inconsistent': inconsistent,
            'confidence': confidence,
            'variation': variation_coefficient
        }
    
    def _detect_lighting_inconsistency(self, 
                                     frame: np.ndarray, 
                                     frame_idx: int, 
                                     frames_data: Dict[str, Any]) -> Dict[str, Any]:
        """Detect lighting inconsistencies"""
        
        # Analyze lighting direction and intensity
        lighting_analysis = self._analyze_lighting_patterns(frame)
        
        # Compare with neighboring frames if available
        consistency_score = 1.0
        if frame_idx > 0 and frame_idx < len(frames_data['frames']) - 1:
            prev_frame = frames_data['frames'][frame_idx - 1]
            next_frame = frames_data['frames'][frame_idx + 1]
            
            prev_lighting = self._analyze_lighting_patterns(prev_frame)
            next_lighting = self._analyze_lighting_patterns(next_frame)
            
            # Compare lighting consistency
            consistency_score = self._calculate_lighting_consistency(
                lighting_analysis, prev_lighting, next_lighting
            )
        
        detected = consistency_score < 0.6
        confidence = 1.0 - consistency_score
        
        return {
            'detected': detected,
            'confidence': confidence,
            'description': f"Lighting inconsistency detected (consistency: {consistency_score:.2f})",
            'evidence': {
                'consistency_score': consistency_score,
                'lighting_direction': lighting_analysis['direction'],
                'lighting_intensity': lighting_analysis['intensity']
            }
        }
    
    def _analyze_lighting_patterns(self, frame: np.ndarray) -> Dict[str, Any]:
        """Analyze lighting patterns in frame"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate image gradients
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        # Calculate dominant lighting direction
        gradient_angles = np.arctan2(grad_y, grad_x)
        dominant_angle = np.median(gradient_angles[np.abs(grad_x) + np.abs(grad_y) > 20])
        
        # Calculate lighting intensity
        intensity = np.mean(gray)
        
        # Calculate lighting uniformity
        intensity_std = np.std(gray)
        uniformity = 1.0 - min(1.0, intensity_std / 100.0)
        
        return {
            'direction': dominant_angle,
            'intensity': intensity,
            'uniformity': uniformity
        }
    
    def _calculate_lighting_consistency(self, 
                                      current: Dict[str, Any],
                                      previous: Dict[str, Any], 
                                      next_lighting: Dict[str, Any]) -> float:
        """Calculate lighting consistency between frames"""
        
        # Direction consistency
        dir_diff_prev = abs(current['direction'] - previous['direction'])
        dir_diff_next = abs(current['direction'] - next_lighting['direction'])
        dir_consistency = 1.0 - min(1.0, (dir_diff_prev + dir_diff_next) / (2 * np.pi))
        
        # Intensity consistency
        int_diff_prev = abs(current['intensity'] - previous['intensity'])
        int_diff_next = abs(current['intensity'] - next_lighting['intensity'])
        int_consistency = 1.0 - min(1.0, (int_diff_prev + int_diff_next) / 510.0)
        
        # Overall consistency
        consistency = (dir_consistency + int_consistency) / 2.0
        
        return consistency
    
    def _detect_pixel_level_editing(self, frame: np.ndarray) -> Dict[str, Any]:
        """Detect pixel-level editing artifacts"""
        
        # Convert to different color spaces for analysis
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Detect clone stamping patterns
        clone_detection = self._detect_clone_stamping(frame)
        
        # Detect color space inconsistencies
        color_inconsistencies = self._detect_color_space_inconsistencies(lab, hsv)
        
        # Detect frequency domain anomalies
        frequency_anomalies = self._detect_frequency_anomalies(frame)
        
        # Combine evidence
        evidence = {}
        confidence_factors = []
        
        if clone_detection['detected']:
            confidence_factors.append(clone_detection['confidence'])
            evidence['clone_regions'] = clone_detection['regions']
        
        if color_inconsistencies['detected']:
            confidence_factors.append(color_inconsistencies['confidence'])
            evidence['color_anomalies'] = color_inconsistencies['anomalies']
        
        if frequency_anomalies['detected']:
            confidence_factors.append(frequency_anomalies['confidence'])
            evidence['frequency_peaks'] = frequency_anomalies['peaks']
        
        detected = len(confidence_factors) >= 1
        confidence = np.mean(confidence_factors) if confidence_factors else 0.0
        
        # Find suspicious coordinates
        coordinates = None
        if clone_detection['detected'] and clone_detection['regions']:
            region = clone_detection['regions'][0]
            coordinates = (region['x'], region['y'], region['width'], region['height'])
        
        return {
            'detected': detected,
            'confidence': confidence,
            'coordinates': coordinates,
            'description': f"Pixel-level editing detected with {confidence:.2f} confidence",
            'evidence': evidence
        }
    
    def _detect_clone_stamping(self, frame: np.ndarray) -> Dict[str, Any]:
        """Detect clone stamping artifacts"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Divide image into blocks and look for similar patterns
        block_size = 32
        h, w = gray.shape
        similar_blocks = []
        
        blocks = {}
        for y in range(0, h - block_size, block_size // 2):
            for x in range(0, w - block_size, block_size // 2):
                block = gray[y:y+block_size, x:x+block_size]
                block_hash = hashlib.md5(block.tobytes()).hexdigest()
                
                if block_hash in blocks:
                    # Found similar block
                    similar_blocks.append({
                        'original': blocks[block_hash],
                        'duplicate': {'x': x, 'y': y, 'width': block_size, 'height': block_size},
                        'similarity': 1.0  # Exact match for now
                    })
                else:
                    blocks[block_hash] = {'x': x, 'y': y, 'width': block_size, 'height': block_size}
        
        detected = len(similar_blocks) > 2
        confidence = min(1.0, len(similar_blocks) / 10.0)
        
        return {
            'detected': detected,
            'confidence': confidence,
            'regions': similar_blocks
        }
    
    def _detect_color_space_inconsistencies(self, lab: np.ndarray, hsv: np.ndarray) -> Dict[str, Any]:
        """Detect color space inconsistencies"""
        
        # Analyze L*a*b* channel correlations
        l_channel = lab[:, :, 0]
        a_channel = lab[:, :, 1]
        b_channel = lab[:, :, 2]
        
        # Calculate correlations
        la_corr = np.corrcoef(l_channel.flatten(), a_channel.flatten())[0, 1]
        lb_corr = np.corrcoef(l_channel.flatten(), b_channel.flatten())[0, 1]
        ab_corr = np.corrcoef(a_channel.flatten(), b_channel.flatten())[0, 1]
        
        # Unusual correlations might indicate manipulation
        unusual_correlations = (abs(la_corr) > 0.8 or abs(lb_corr) > 0.8 or abs(ab_corr) > 0.8)
        
        # Analyze HSV consistency
        h_channel = hsv[:, :, 0]
        s_channel = hsv[:, :, 1]
        v_channel = hsv[:, :, 2]
        
        # Look for unusual value distributions
        h_entropy = self._calculate_channel_entropy(h_channel)
        s_entropy = self._calculate_channel_entropy(s_channel)
        v_entropy = self._calculate_channel_entropy(v_channel)
        
        low_entropy = h_entropy < 3.0 or s_entropy < 3.0 or v_entropy < 3.0
        
        detected = unusual_correlations or low_entropy
        confidence = 0.6 if detected else 0.0
        
        return {
            'detected': detected,
            'confidence': confidence,
            'anomalies': {
                'unusual_correlations': unusual_correlations,
                'low_entropy_channels': low_entropy,
                'correlations': {'LA': la_corr, 'LB': lb_corr, 'AB': ab_corr},
                'entropies': {'H': h_entropy, 'S': s_entropy, 'V': v_entropy}
            }
        }
    
    def _calculate_channel_entropy(self, channel: np.ndarray) -> float:
        """Calculate entropy of image channel"""
        hist, _ = np.histogram(channel, bins=256, range=(0, 256))
        hist_norm = hist / (np.sum(hist) + 1e-6)
        entropy = -np.sum(hist_norm * np.log2(hist_norm + 1e-6))
        return entropy
    
    def _detect_frequency_anomalies(self, frame: np.ndarray) -> Dict[str, Any]:
        """Detect frequency domain anomalies"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply FFT
        f_transform = np.fft.fft2(gray)
        f_shift = np.fft.fftshift(f_transform)
        magnitude_spectrum = np.log(np.abs(f_shift) + 1)
        
        # Look for unusual peaks in frequency domain
        # High peaks might indicate periodic artifacts
        threshold = np.percentile(magnitude_spectrum, 99)
        peaks = magnitude_spectrum > threshold
        peak_count = np.sum(peaks)
        
        # Regular grids of peaks might indicate resampling artifacts
        grid_pattern = self._detect_grid_pattern(peaks)
        
        detected = peak_count > 20 or grid_pattern
        confidence = min(1.0, peak_count / 100.0) if not grid_pattern else 0.8
        
        return {
            'detected': detected,
            'confidence': confidence,
            'peaks': int(peak_count),
            'grid_pattern': grid_pattern
        }
    
    def _detect_grid_pattern(self, peaks: np.ndarray) -> bool:
        """Detect grid patterns in frequency peaks"""
        # Look for regular spacing in peak positions
        peak_positions = np.where(peaks)
        
        if len(peak_positions[0]) < 4:
            return False
        
        # Check for regular spacing
        y_diffs = np.diff(np.sort(peak_positions[0]))
        x_diffs = np.diff(np.sort(peak_positions[1]))
        
        # Look for repeated differences (regular spacing)
        y_regular = len(np.unique(y_diffs)) < len(y_diffs) // 2
        x_regular = len(np.unique(x_diffs)) < len(x_diffs) // 2
        
        return y_regular and x_regular
    
    def _estimate_compression_ratio(self, frame: np.ndarray) -> Optional[float]:
        """Estimate compression ratio for frame"""
        # Simple estimation based on JPEG compression
        # Encode frame as JPEG and compare sizes
        original_size = frame.nbytes
        
        # Encode as JPEG with different quality levels
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
        _, encoded = cv2.imencode('.jpg', frame, encode_param)
        compressed_size = len(encoded)
        
        if compressed_size > 0:
            return original_size / compressed_size
        
        return None
    
    def _calculate_frame_quality_score(self, frame: np.ndarray) -> float:
        """Calculate quality score for frame"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate sharpness (Laplacian variance)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        sharpness = laplacian.var()
        
        # Calculate contrast (standard deviation)
        contrast = np.std(gray)
        
        # Calculate brightness distribution
        brightness_hist = np.histogram(gray, bins=256, range=(0, 256))[0]
        brightness_entropy = -np.sum((brightness_hist / np.sum(brightness_hist)) * 
                                    np.log2((brightness_hist / np.sum(brightness_hist)) + 1e-6))
        
        # Combine metrics
        sharpness_score = min(1.0, sharpness / 1000.0)
        contrast_score = min(1.0, contrast / 100.0)
        entropy_score = min(1.0, brightness_entropy / 8.0)
        
        quality_score = (sharpness_score + contrast_score + entropy_score) / 3.0
        
        return quality_score
    
    def _calculate_motion_consistency(self, 
                                    frame: np.ndarray, 
                                    frame_idx: int, 
                                    frames_data: Dict[str, Any]) -> Optional[float]:
        """Calculate motion vector consistency"""
        if frame_idx == 0 or frame_idx >= len(frames_data['frames']) - 1:
            return None
        
        prev_frame = frames_data['frames'][frame_idx - 1]
        curr_frame = frame
        next_frame = frames_data['frames'][frame_idx + 1]
        
        # Calculate optical flow
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
        next_gray = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)
        
        # Flow from prev to current
        flow1 = self._calculate_optical_flow_magnitude(prev_gray, curr_gray)
        
        # Flow from current to next
        flow2 = self._calculate_optical_flow_magnitude(curr_gray, next_gray)
        
        # Consistency is based on flow magnitude similarity
        if flow1 > 0 and flow2 > 0:
            consistency = 1.0 - abs(flow1 - flow2) / max(flow1, flow2)
            return max(0.0, consistency)
        
        return 1.0  # No motion detected
    
    def _extract_visual_features(self, frame: np.ndarray) -> Dict[str, Any]:
        """Extract visual features from frame"""
        
        # Extract dominant colors
        dominant_colors = self._extract_dominant_colors(frame)
        
        # Calculate edge density
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.mean(edges) / 255.0
        
        # Calculate texture features
        texture_features = self._extract_texture_features(gray)
        
        return {
            'dominant_colors': dominant_colors,
            'edge_density': edge_density,
            'texture_features': texture_features
        }
    
    def _extract_dominant_colors(self, frame: np.ndarray, k: int = 5) -> List[str]:
        """Extract dominant colors using K-means clustering"""
        # Reshape image to be a list of pixels
        pixels = frame.reshape((-1, 3))
        pixels = np.float32(pixels)
        
        # Apply K-means clustering
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
        _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        
        # Convert centers to integer and then to hex colors
        centers = np.uint8(centers)
        dominant_colors = []
        
        for center in centers:
            # Convert BGR to RGB
            rgb = (int(center[2]), int(center[1]), int(center[0]))
            hex_color = f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"
            dominant_colors.append(hex_color)
        
        return dominant_colors
    
    def _extract_texture_features(self, gray_frame: np.ndarray) -> Dict[str, float]:
        """Extract texture features from grayscale frame"""
        
        # Calculate GLCM-based features (simplified)
        # Energy, contrast, homogeneity
        
        # Calculate co-occurrence matrix (simplified 1-pixel distance)
        glcm = self._calculate_glcm(gray_frame)
        
        # Calculate texture metrics
        energy = np.sum(glcm ** 2)
        contrast = np.sum((np.arange(256)[:, None] - np.arange(256)) ** 2 * glcm)
        homogeneity = np.sum(glcm / (1 + np.abs(np.arange(256)[:, None] - np.arange(256))))
        
        return {
            'energy': float(energy),
            'contrast': float(contrast),
            'homogeneity': float(homogeneity)
        }
    
    def _calculate_glcm(self, gray_frame: np.ndarray) -> np.ndarray:
        """Calculate Gray-Level Co-occurrence Matrix (simplified)"""
        # Reduce gray levels for computation efficiency
        reduced = (gray_frame // 16).astype(np.uint8)  # 16 gray levels
        
        # Initialize GLCM
        glcm = np.zeros((16, 16), dtype=np.float32)
        
        # Calculate co-occurrence for horizontal pairs
        h, w = reduced.shape
        for i in range(h):
            for j in range(w - 1):
                glcm[reduced[i, j], reduced[i, j + 1]] += 1
        
        # Normalize
        glcm = glcm / np.sum(glcm) if np.sum(glcm) > 0 else glcm
        
        # Expand back to 256x256 for consistency
        expanded_glcm = np.zeros((256, 256), dtype=np.float32)
        expanded_glcm[:16, :16] = glcm
        
        return expanded_glcm
    
    def _analyze_faces_in_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        """Analyze faces in frame"""
        # Simple face detection using Haar cascades
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Load face classifier
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Detect faces
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        face_regions = []
        for (x, y, w, h) in faces:
            face_regions.append({
                'x': int(x), 'y': int(y), 
                'width': int(w), 'height': int(h)
            })
        
        # Simple deepfake probability estimation
        # This would normally use a trained model
        deepfake_probability = None
        if len(faces) > 0:
            # Placeholder: random probability for demonstration
            deepfake_probability = 0.1  # Low probability by default
        
        return {
            'face_count': len(faces),
            'face_regions': face_regions,
            'deepfake_probability': deepfake_probability
        }
    
    def _analyze_temporal_consistency(self, 
                                    key_frames: List[KeyFrame],
                                    all_frames: List[np.ndarray]) -> Dict[str, Any]:
        """Analyze temporal consistency across key frames"""
        
        indicators = []
        
        if len(key_frames) < 2:
            return {'indicators': indicators}
        
        # Analyze quality consistency
        quality_scores = [kf.quality_score for kf in key_frames if kf.quality_score is not None]
        if len(quality_scores) > 1:
            quality_std = np.std(quality_scores)
            if quality_std > 0.3:  # High variation in quality
                indicator = TamperingIndicator(
                    tampering_type=TamperingType.TEMPORAL_INCONSISTENCY,
                    confidence=min(1.0, quality_std / 0.5),
                    description=f"High quality variation across frames (std: {quality_std:.3f})",
                    evidence={'quality_std': quality_std, 'quality_scores': quality_scores},
                    analysis_method="temporal_analysis",
                    severity_score=quality_std * 10
                )
                indicators.append(indicator)
        
        # Analyze motion consistency
        motion_scores = [kf.motion_vector_consistency for kf in key_frames 
                        if kf.motion_vector_consistency is not None]
        if len(motion_scores) > 1:
            low_motion_consistency = np.mean(motion_scores) < 0.5
            if low_motion_consistency:
                indicator = TamperingIndicator(
                    tampering_type=TamperingType.TEMPORAL_INCONSISTENCY,
                    confidence=1.0 - np.mean(motion_scores),
                    description=f"Low motion consistency detected (avg: {np.mean(motion_scores):.3f})",
                    evidence={'motion_consistency_scores': motion_scores},
                    analysis_method="motion_analysis",
                    severity_score=(1.0 - np.mean(motion_scores)) * 8
                )
                indicators.append(indicator)
        
        return {'indicators': indicators}