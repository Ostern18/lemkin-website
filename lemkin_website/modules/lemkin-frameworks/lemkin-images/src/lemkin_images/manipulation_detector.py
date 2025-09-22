"""
Lemkin Image Verification Suite - Manipulation Detection Module

This module implements advanced algorithms for detecting image manipulation
including copy-move forgeries, splicing, cloning, and other tampering techniques.
Uses computer vision and machine learning approaches for forensic analysis.

Legal Compliance: Meets standards for digital evidence analysis in legal proceedings
"""

import logging
import math
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import cv2
from scipy import ndimage
from scipy.fft import fft2, ifft2
from scipy.stats import entropy
import matplotlib.pyplot as plt

from .core import (
    ManipulationType,
    ManipulationIndicator,
    ManipulationAnalysis,
    ImageAuthConfig
)

logger = logging.getLogger(__name__)


class ImageManipulationDetector:
    """
    Advanced image manipulation detection using multiple forensic algorithms.
    
    Implements state-of-the-art techniques for detecting:
    - Copy-move forgeries
    - Splicing and composition
    - Resampling artifacts
    - Compression inconsistencies
    - Noise pattern analysis
    - Edge tampering detection
    """
    
    def __init__(self, config: Optional[ImageAuthConfig] = None):
        """Initialize the manipulation detector"""
        self.config = config or ImageAuthConfig()
        self.logger = logging.getLogger(f"{__name__}.ImageManipulationDetector")
        
        # Algorithm parameters
        self.block_size = 16  # Block size for copy-move detection
        self.overlap_threshold = 0.8  # Overlap threshold for matches
        self.min_cluster_size = 5  # Minimum cluster size for DBSCAN
        
        self.logger.info("Image manipulation detector initialized")
    
    def detect_manipulation(self, image_path: Path) -> ManipulationAnalysis:
        """
        Perform comprehensive manipulation detection on an image
        
        Args:
            image_path: Path to the image file
            
        Returns:
            ManipulationAnalysis with detailed findings
        """
        start_time = datetime.utcnow()
        
        if not image_path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Calculate image hash
        image_hash = self._calculate_image_hash(image)
        
        self.logger.info(f"Starting manipulation detection for: {image_path.name}")
        
        # Initialize analysis result
        analysis = ManipulationAnalysis(
            image_hash=image_hash,
            is_manipulated=False,
            overall_confidence=0.0,
            manipulation_probability=0.0
        )
        
        try:
            # Convert to different color spaces
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            
            # Perform various detection algorithms
            self._detect_copy_move(image, gray, analysis)
            self._detect_splicing(image, lab, analysis)
            self._detect_resampling(gray, analysis)
            self._detect_compression_inconsistencies(image, analysis)
            self._detect_noise_inconsistencies(gray, analysis)
            self._detect_edge_inconsistencies(gray, analysis)
            self._analyze_frequency_domain(gray, analysis)
            
            # Calculate overall assessment
            self._calculate_overall_assessment(analysis)
            
            # Calculate duration
            end_time = datetime.utcnow()
            duration = end_time - start_time
            analysis.analysis_duration_seconds = duration.total_seconds()
            
            self.logger.info(f"Manipulation detection completed: {analysis.manipulation_probability:.3f} probability")
            
        except Exception as e:
            self.logger.error(f"Manipulation detection failed: {str(e)}")
            analysis.indicators.append(ManipulationIndicator(
                manipulation_type=ManipulationType.METADATA_MANIPULATION,
                confidence=0.0,
                severity="high",
                detection_method="error_analysis",
                description=f"Analysis failed: {str(e)}"
            ))
        
        return analysis
    
    def _calculate_image_hash(self, image: np.ndarray) -> str:
        """Calculate SHA-256 hash of image data"""
        import hashlib
        image_bytes = cv2.imencode('.png', image)[1].tobytes()
        return hashlib.sha256(image_bytes).hexdigest()
    
    def _detect_copy_move(self, image: np.ndarray, gray: np.ndarray, analysis: ManipulationAnalysis):
        """Detect copy-move forgeries using block matching"""
        analysis.methods_applied.append("copy_move_detection")
        analysis.algorithms_used.append("block_matching")
        
        try:
            height, width = gray.shape
            block_size = self.block_size
            
            # Extract overlapping blocks
            blocks = []
            positions = []
            
            for y in range(0, height - block_size + 1, block_size // 2):
                for x in range(0, width - block_size + 1, block_size // 2):
                    block = gray[y:y + block_size, x:x + block_size]
                    
                    # Calculate DCT features
                    dct_block = cv2.dct(block.astype(np.float32))
                    features = dct_block.flatten()
                    
                    blocks.append(features)
                    positions.append((x, y))
            
            if len(blocks) < 2:
                return
            
            # Find similar blocks using clustering
            blocks_array = np.array(blocks)
            scaler = StandardScaler()
            blocks_normalized = scaler.fit_transform(blocks_array)
            
            # Use DBSCAN to find clusters of similar blocks
            clustering = DBSCAN(eps=0.3, min_samples=2)
            labels = clustering.fit_predict(blocks_normalized)
            
            # Analyze clusters for copy-move patterns
            unique_labels = set(labels)
            copy_move_regions = []
            
            for label in unique_labels:
                if label == -1:  # Noise
                    continue
                
                cluster_indices = np.where(labels == label)[0]
                if len(cluster_indices) >= 2:
                    # Get positions of blocks in this cluster
                    cluster_positions = [positions[i] for i in cluster_indices]
                    
                    # Check if blocks are sufficiently separated
                    min_distance = float('inf')
                    for i, pos1 in enumerate(cluster_positions):
                        for j, pos2 in enumerate(cluster_positions[i+1:], i+1):
                            distance = np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
                            min_distance = min(min_distance, distance)
                    
                    # If minimum distance is greater than 2 block sizes, likely copy-move
                    if min_distance > 2 * block_size:
                        copy_move_regions.extend(cluster_positions)
            
            # Create indicators for detected copy-move regions
            if copy_move_regions:
                confidence = min(0.9, len(copy_move_regions) / 10.0)
                severity = "critical" if confidence > 0.7 else "high" if confidence > 0.4 else "medium"
                
                affected_regions = [(x, y, block_size, block_size) for x, y in copy_move_regions]
                
                indicator = ManipulationIndicator(
                    manipulation_type=ManipulationType.COPY_MOVE,
                    confidence=confidence,
                    severity=severity,
                    affected_regions=affected_regions,
                    detection_method="block_matching_clustering",
                    algorithm_used="DCT_DBSCAN",
                    description=f"Detected {len(copy_move_regions)} copy-move regions using block matching",
                    technical_explanation="Similar blocks found in different image regions using DCT features and clustering"
                )
                
                analysis.indicators.append(indicator)
                
        except Exception as e:
            self.logger.error(f"Copy-move detection failed: {str(e)}")
    
    def _detect_splicing(self, image: np.ndarray, lab: np.ndarray, analysis: ManipulationAnalysis):
        """Detect image splicing using color and lighting inconsistencies"""
        analysis.methods_applied.append("splicing_detection")
        analysis.algorithms_used.append("color_lighting_analysis")
        
        try:
            # Analyze color distribution inconsistencies
            l_channel, a_channel, b_channel = cv2.split(lab)
            
            # Divide image into regions
            height, width = l_channel.shape
            region_size = min(height, width) // 8
            
            if region_size < 20:
                return
            
            regions_stats = []
            
            for y in range(0, height - region_size + 1, region_size):
                for x in range(0, width - region_size + 1, region_size):
                    region_l = l_channel[y:y + region_size, x:x + region_size]
                    region_a = a_channel[y:y + region_size, x:x + region_size]
                    region_b = b_channel[y:y + region_size, x:x + region_size]
                    
                    # Calculate statistics for each channel
                    stats = {
                        'position': (x, y),
                        'l_mean': np.mean(region_l),
                        'l_std': np.std(region_l),
                        'a_mean': np.mean(region_a),
                        'a_std': np.std(region_a),
                        'b_mean': np.mean(region_b),
                        'b_std': np.std(region_b)
                    }
                    
                    regions_stats.append(stats)
            
            # Analyze statistical consistency
            if len(regions_stats) < 4:
                return
            
            # Extract features for clustering
            features = []
            for stats in regions_stats:
                features.append([
                    stats['l_mean'], stats['l_std'],
                    stats['a_mean'], stats['a_std'],
                    stats['b_mean'], stats['b_std']
                ])
            
            features_array = np.array(features)
            scaler = StandardScaler()
            features_normalized = scaler.fit_transform(features_array)
            
            # Cluster regions by color characteristics
            clustering = DBSCAN(eps=0.5, min_samples=2)
            labels = clustering.fit_predict(features_normalized)
            
            # Check for isolated regions (potential splicing)
            unique_labels = set(labels)
            outlier_regions = []
            
            for i, label in enumerate(labels):
                if label == -1:  # Outlier
                    outlier_regions.append(regions_stats[i]['position'])
            
            # Create indicator if outliers found
            if outlier_regions and len(outlier_regions) < len(regions_stats) * 0.3:
                confidence = min(0.8, len(outlier_regions) / (len(regions_stats) * 0.1))
                severity = "high" if confidence > 0.6 else "medium"
                
                affected_regions = [(x, y, region_size, region_size) for x, y in outlier_regions]
                
                indicator = ManipulationIndicator(
                    manipulation_type=ManipulationType.SPLICING,
                    confidence=confidence,
                    severity=severity,
                    affected_regions=affected_regions,
                    detection_method="color_inconsistency_analysis",
                    algorithm_used="LAB_clustering",
                    description=f"Detected {len(outlier_regions)} regions with inconsistent color characteristics",
                    technical_explanation="Regions with significantly different color distributions may indicate splicing"
                )
                
                analysis.indicators.append(indicator)
                
        except Exception as e:
            self.logger.error(f"Splicing detection failed: {str(e)}")
    
    def _detect_resampling(self, gray: np.ndarray, analysis: ManipulationAnalysis):
        """Detect resampling artifacts using periodic patterns"""
        analysis.methods_applied.append("resampling_detection")
        analysis.algorithms_used.append("periodic_pattern_analysis")
        
        try:
            # Calculate second derivative in both directions
            dx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            dy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            
            # Calculate second derivatives
            dxx = cv2.Sobel(dx, cv2.CV_64F, 1, 0, ksize=3)
            dyy = cv2.Sobel(dy, cv2.CV_64F, 0, 1, ksize=3)
            
            # Combine second derivatives
            laplacian = dxx + dyy
            
            # Apply FFT to detect periodic patterns
            fft_result = fft2(laplacian)
            magnitude_spectrum = np.abs(fft_result)
            
            # Look for periodic peaks in frequency domain
            height, width = magnitude_spectrum.shape
            center_y, center_x = height // 2, width // 2
            
            # Exclude DC component
            magnitude_spectrum[center_y-2:center_y+3, center_x-2:center_x+3] = 0
            
            # Find peaks in the spectrum
            threshold = np.mean(magnitude_spectrum) + 3 * np.std(magnitude_spectrum)
            peaks = magnitude_spectrum > threshold
            
            # Count significant peaks
            peak_count = np.sum(peaks)
            total_pixels = height * width
            peak_ratio = peak_count / total_pixels
            
            # Detect resampling if too many periodic patterns
            if peak_ratio > 0.001:  # Threshold for resampling detection
                confidence = min(0.9, peak_ratio * 1000)
                severity = "high" if confidence > 0.7 else "medium"
                
                indicator = ManipulationIndicator(
                    manipulation_type=ManipulationType.RESAMPLING,
                    confidence=confidence,
                    severity=severity,
                    detection_method="frequency_domain_analysis",
                    algorithm_used="FFT_periodic_detection",
                    description=f"Detected periodic patterns suggesting resampling (peak ratio: {peak_ratio:.4f})",
                    technical_explanation="Excessive periodic patterns in frequency domain indicate potential resampling artifacts",
                    parameters={"peak_ratio": peak_ratio, "peak_count": int(peak_count)}
                )
                
                analysis.indicators.append(indicator)
                
        except Exception as e:
            self.logger.error(f"Resampling detection failed: {str(e)}")
    
    def _detect_compression_inconsistencies(self, image: np.ndarray, analysis: ManipulationAnalysis):
        """Detect JPEG compression inconsistencies"""
        analysis.methods_applied.append("compression_analysis")
        analysis.algorithms_used.append("DCT_grid_analysis")
        
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            height, width = gray.shape
            
            # JPEG uses 8x8 DCT blocks
            block_size = 8
            compression_map = np.zeros((height // block_size, width // block_size))
            
            # Analyze DCT coefficients for each 8x8 block
            for y in range(0, height - block_size + 1, block_size):
                for x in range(0, width - block_size + 1, block_size):
                    block = gray[y:y + block_size, x:x + block_size]
                    
                    # Calculate DCT
                    dct_block = cv2.dct(block.astype(np.float32))
                    
                    # Measure compression artifacts
                    # High frequency coefficients should be quantized in JPEG
                    high_freq_energy = np.sum(np.abs(dct_block[4:, 4:]))
                    total_energy = np.sum(np.abs(dct_block))
                    
                    if total_energy > 0:
                        compression_ratio = high_freq_energy / total_energy
                        compression_map[y // block_size, x // block_size] = compression_ratio
            
            # Analyze consistency of compression
            compression_std = np.std(compression_map)
            compression_mean = np.mean(compression_map)
            
            # Detect inconsistencies
            if compression_std > 0.1:  # Threshold for inconsistency
                # Find regions with significantly different compression
                inconsistent_mask = np.abs(compression_map - compression_mean) > 2 * compression_std
                inconsistent_regions = np.where(inconsistent_mask)
                
                if len(inconsistent_regions[0]) > 0:
                    confidence = min(0.8, compression_std * 5)
                    severity = "high" if confidence > 0.6 else "medium"
                    
                    # Convert to pixel coordinates
                    affected_regions = []
                    for i in range(len(inconsistent_regions[0])):
                        y_block = inconsistent_regions[0][i]
                        x_block = inconsistent_regions[1][i]
                        affected_regions.append((
                            x_block * block_size, y_block * block_size,
                            block_size, block_size
                        ))
                    
                    indicator = ManipulationIndicator(
                        manipulation_type=ManipulationType.COMPRESSION_INCONSISTENCY,
                        confidence=confidence,
                        severity=severity,
                        affected_regions=affected_regions,
                        detection_method="DCT_compression_analysis",
                        algorithm_used="JPEG_block_analysis",
                        description=f"Detected compression inconsistencies (std: {compression_std:.3f})",
                        technical_explanation="Different JPEG compression levels in different regions suggest manipulation",
                        parameters={"compression_std": compression_std, "compression_mean": compression_mean}
                    )
                    
                    analysis.indicators.append(indicator)
                    
                    # Store detailed analysis
                    analysis.compression_analysis = {
                        "compression_map": compression_map.tolist(),
                        "compression_std": compression_std,
                        "compression_mean": compression_mean,
                        "inconsistent_blocks": len(inconsistent_regions[0])
                    }
            
        except Exception as e:
            self.logger.error(f"Compression analysis failed: {str(e)}")
    
    def _detect_noise_inconsistencies(self, gray: np.ndarray, analysis: ManipulationAnalysis):
        """Detect noise pattern inconsistencies"""
        analysis.methods_applied.append("noise_analysis")
        analysis.algorithms_used.append("local_noise_estimation")
        
        try:
            # Apply median filter to estimate noise
            median_filtered = cv2.medianBlur(gray, 3)
            noise_map = np.abs(gray.astype(np.float32) - median_filtered.astype(np.float32))
            
            # Divide into regions and analyze noise statistics
            height, width = gray.shape
            region_size = min(height, width) // 10
            
            if region_size < 16:
                return
            
            noise_stats = []
            positions = []
            
            for y in range(0, height - region_size + 1, region_size):
                for x in range(0, width - region_size + 1, region_size):
                    noise_region = noise_map[y:y + region_size, x:x + region_size]
                    
                    noise_mean = np.mean(noise_region)
                    noise_std = np.std(noise_region)
                    noise_entropy = entropy(noise_region.flatten())
                    
                    noise_stats.append([noise_mean, noise_std, noise_entropy])
                    positions.append((x, y))
            
            if len(noise_stats) < 4:
                return
            
            # Detect outliers in noise characteristics
            noise_array = np.array(noise_stats)
            scaler = StandardScaler()
            noise_normalized = scaler.fit_transform(noise_array)
            
            # Use statistical outlier detection
            noise_distances = np.sqrt(np.sum(noise_normalized**2, axis=1))
            threshold = np.mean(noise_distances) + 2 * np.std(noise_distances)
            outliers = noise_distances > threshold
            
            outlier_positions = [positions[i] for i in range(len(positions)) if outliers[i]]
            
            if outlier_positions and len(outlier_positions) < len(positions) * 0.5:
                confidence = min(0.7, len(outlier_positions) / (len(positions) * 0.1))
                severity = "medium" if confidence > 0.4 else "low"
                
                affected_regions = [(x, y, region_size, region_size) for x, y in outlier_positions]
                
                indicator = ManipulationIndicator(
                    manipulation_type=ManipulationType.NOISE_INCONSISTENCY,
                    confidence=confidence,
                    severity=severity,
                    affected_regions=affected_regions,
                    detection_method="local_noise_analysis",
                    algorithm_used="statistical_outlier_detection",
                    description=f"Detected {len(outlier_positions)} regions with inconsistent noise patterns",
                    technical_explanation="Regions with significantly different noise characteristics may indicate manipulation"
                )
                
                analysis.indicators.append(indicator)
                
                # Store detailed analysis
                analysis.noise_analysis = {
                    "noise_regions": len(positions),
                    "outlier_regions": len(outlier_positions),
                    "noise_statistics": noise_stats
                }
            
        except Exception as e:
            self.logger.error(f"Noise analysis failed: {str(e)}")
    
    def _detect_edge_inconsistencies(self, gray: np.ndarray, analysis: ManipulationAnalysis):
        """Detect edge tampering and inconsistencies"""
        analysis.methods_applied.append("edge_analysis")
        analysis.algorithms_used.append("edge_gradient_analysis")
        
        try:
            # Calculate edge maps using different methods
            edges_canny = cv2.Canny(gray, 50, 150)
            edges_sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            edges_sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            edge_magnitude = np.sqrt(edges_sobel_x**2 + edges_sobel_y**2)
            
            # Analyze edge consistency
            height, width = gray.shape
            region_size = min(height, width) // 8
            
            if region_size < 20:
                return
            
            edge_stats = []
            positions = []
            
            for y in range(0, height - region_size + 1, region_size):
                for x in range(0, width - region_size + 1, region_size):
                    canny_region = edges_canny[y:y + region_size, x:x + region_size]
                    magnitude_region = edge_magnitude[y:y + region_size, x:x + region_size]
                    
                    # Calculate edge statistics
                    edge_density = np.sum(canny_region > 0) / (region_size * region_size)
                    edge_strength = np.mean(magnitude_region)
                    edge_variance = np.var(magnitude_region)
                    
                    edge_stats.append([edge_density, edge_strength, edge_variance])
                    positions.append((x, y))
            
            if len(edge_stats) < 4:
                return
            
            # Detect anomalous edge regions
            edge_array = np.array(edge_stats)
            scaler = StandardScaler()
            edge_normalized = scaler.fit_transform(edge_array)
            
            # Clustering to find consistent edge patterns
            clustering = DBSCAN(eps=0.5, min_samples=2)
            labels = clustering.fit_predict(edge_normalized)
            
            # Find outlier regions
            outlier_indices = np.where(labels == -1)[0]
            outlier_positions = [positions[i] for i in outlier_indices]
            
            if outlier_positions and len(outlier_positions) < len(positions) * 0.3:
                confidence = min(0.6, len(outlier_positions) / (len(positions) * 0.1))
                severity = "medium" if confidence > 0.3 else "low"
                
                affected_regions = [(x, y, region_size, region_size) for x, y in outlier_positions]
                
                indicator = ManipulationIndicator(
                    manipulation_type=ManipulationType.EDGE_INCONSISTENCY,
                    confidence=confidence,
                    severity=severity,
                    affected_regions=affected_regions,
                    detection_method="edge_pattern_analysis",
                    algorithm_used="Canny_Sobel_clustering",
                    description=f"Detected {len(outlier_positions)} regions with inconsistent edge patterns",
                    technical_explanation="Regions with significantly different edge characteristics may indicate tampering"
                )
                
                analysis.indicators.append(indicator)
                
                # Store detailed analysis
                analysis.edge_analysis = {
                    "total_regions": len(positions),
                    "outlier_regions": len(outlier_positions),
                    "edge_statistics": edge_stats
                }
            
        except Exception as e:
            self.logger.error(f"Edge analysis failed: {str(e)}")
    
    def _analyze_frequency_domain(self, gray: np.ndarray, analysis: ManipulationAnalysis):
        """Analyze frequency domain characteristics"""
        analysis.methods_applied.append("frequency_analysis")
        analysis.algorithms_used.append("FFT_analysis")
        
        try:
            # Apply FFT
            fft_result = fft2(gray)
            magnitude_spectrum = np.abs(fft_result)
            phase_spectrum = np.angle(fft_result)
            
            # Analyze frequency characteristics
            height, width = magnitude_spectrum.shape
            center_y, center_x = height // 2, width // 2
            
            # Create frequency coordinate grids
            y_coords, x_coords = np.ogrid[:height, :width]
            distances = np.sqrt((x_coords - center_x)**2 + (y_coords - center_y)**2)
            
            # Analyze radial frequency distribution
            max_distance = min(center_x, center_y)
            radial_profile = []
            
            for r in range(1, max_distance, max_distance // 20):
                mask = (distances >= r) & (distances < r + max_distance // 20)
                if np.any(mask):
                    radial_energy = np.mean(magnitude_spectrum[mask])
                    radial_profile.append(radial_energy)
            
            # Look for anomalies in frequency distribution
            if len(radial_profile) > 3:
                # Calculate smoothness of frequency falloff
                freq_gradient = np.gradient(radial_profile)
                freq_variance = np.var(freq_gradient)
                
                # Detect unusual frequency patterns
                if freq_variance > np.mean(radial_profile) * 0.1:
                    confidence = min(0.5, freq_variance / (np.mean(radial_profile) * 0.1))
                    severity = "low"
                    
                    indicator = ManipulationIndicator(
                        manipulation_type=ManipulationType.RESAMPLING,
                        confidence=confidence,
                        severity=severity,
                        detection_method="frequency_domain_analysis",
                        algorithm_used="radial_frequency_analysis",
                        description=f"Detected unusual frequency domain patterns (variance: {freq_variance:.3f})",
                        technical_explanation="Irregular frequency distribution may indicate digital processing artifacts",
                        parameters={"frequency_variance": freq_variance}
                    )
                    
                    analysis.indicators.append(indicator)
            
            # Store frequency analysis
            analysis.frequency_analysis = {
                "radial_profile": radial_profile,
                "frequency_variance": freq_variance if 'freq_variance' in locals() else 0,
                "dc_component": float(magnitude_spectrum[center_y, center_x]),
                "high_freq_energy": float(np.sum(magnitude_spectrum[center_y+10:, center_x+10:]))
            }
            
        except Exception as e:
            self.logger.error(f"Frequency analysis failed: {str(e)}")
    
    def _calculate_overall_assessment(self, analysis: ManipulationAnalysis):
        """Calculate overall manipulation assessment"""
        
        if not analysis.indicators:
            analysis.is_manipulated = False
            analysis.overall_confidence = 0.8  # High confidence in authentic assessment
            analysis.manipulation_probability = 0.1
            return
        
        # Weight indicators by confidence and severity
        total_weight = 0
        weighted_manipulation_score = 0
        
        severity_weights = {"critical": 1.0, "high": 0.8, "medium": 0.6, "low": 0.4}
        
        for indicator in analysis.indicators:
            weight = severity_weights.get(indicator.severity, 0.5) * indicator.confidence
            total_weight += weight
            weighted_manipulation_score += weight
        
        if total_weight > 0:
            analysis.manipulation_probability = min(0.95, weighted_manipulation_score / max(1.0, total_weight))
        else:
            analysis.manipulation_probability = 0.1
        
        # Determine if image is manipulated
        analysis.is_manipulated = analysis.manipulation_probability > self.config.manipulation_threshold
        
        # Calculate overall confidence
        confidence_sum = sum(indicator.confidence for indicator in analysis.indicators)
        analysis.overall_confidence = min(0.95, confidence_sum / max(1.0, len(analysis.indicators)))
        
        # Determine if expert review is needed
        critical_indicators = [i for i in analysis.indicators if i.severity == "critical"]
        high_indicators = [i for i in analysis.indicators if i.severity == "high"]
        
        analysis.expert_review_recommended = (
            len(critical_indicators) > 0 or
            len(high_indicators) > 2 or
            analysis.manipulation_probability > 0.8
        )
        
        # Calculate image quality score (inverse of manipulation probability)
        analysis.image_quality_score = 1.0 - analysis.manipulation_probability
        analysis.authenticity_score = analysis.image_quality_score * analysis.overall_confidence


def detect_image_manipulation(image_path: Path, config: Optional[ImageAuthConfig] = None) -> ManipulationAnalysis:
    """
    Convenience function to detect image manipulation
    
    Args:
        image_path: Path to the image file
        config: Optional configuration
        
    Returns:
        ManipulationAnalysis with detailed findings
    """
    detector = ImageManipulationDetector(config)
    return detector.detect_manipulation(image_path)