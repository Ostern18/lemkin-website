"""
Core video authentication functionality for legal investigations.
"""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum

import cv2
import numpy as np
from PIL import Image
import imagehash
from pydantic import BaseModel, Field, field_validator
from loguru import logger
from skimage import measure, filters


class AuthenticityStatus(str, Enum):
    """Video authenticity status"""
    AUTHENTIC = "authentic"
    MANIPULATED = "manipulated"
    DEEPFAKE = "deepfake"
    SUSPICIOUS = "suspicious"
    UNKNOWN = "unknown"


class CompressionType(str, Enum):
    """Video compression types"""
    H264 = "h264"
    H265 = "h265"
    MPEG4 = "mpeg4"
    VP8 = "vp8"
    VP9 = "vp9"
    AV1 = "av1"
    UNKNOWN = "unknown"


class VideoMetadata(BaseModel):
    """Video file metadata"""
    file_path: Path
    file_size: int
    duration_seconds: float
    width: int
    height: int
    fps: float
    frame_count: int
    codec: str
    bitrate: Optional[int] = None
    creation_date: Optional[datetime] = None
    file_hash: str = ""

    def model_post_init(self, __context: Any) -> None:
        """Calculate file hash after initialization"""
        if self.file_path.exists():
            self.file_hash = self._calculate_hash()

    def _calculate_hash(self) -> str:
        """Calculate SHA-256 hash of video file"""
        sha256 = hashlib.sha256()
        with open(self.file_path, 'rb') as f:
            while chunk := f.read(8192):
                sha256.update(chunk)
        return sha256.hexdigest()


class KeyFrame(BaseModel):
    """Key frame information"""
    frame_number: int
    timestamp: float
    frame_hash: str
    similarity_score: float = 0.0
    features: Dict[str, Any] = Field(default_factory=dict)
    artifacts: List[str] = Field(default_factory=list)


class DeepfakeAnalysis(BaseModel):
    """Deepfake detection analysis results"""
    analysis_id: str
    video_path: Path
    is_deepfake: bool
    confidence_score: float = Field(ge=0.0, le=1.0)
    frames_analyzed: int
    suspicious_frames: List[int] = Field(default_factory=list)
    detection_metrics: Dict[str, float] = Field(default_factory=dict)
    temporal_consistency: float = Field(ge=0.0, le=1.0, default=0.0)
    analysis_date: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class VideoFingerprint(BaseModel):
    """Video content fingerprint for duplicate detection"""
    fingerprint_id: str
    video_path: Path
    perceptual_hash: str
    temporal_features: List[str] = Field(default_factory=list)
    key_frames: List[KeyFrame] = Field(default_factory=list)
    duration: float
    similarity_threshold: float = 0.8
    created_date: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class CompressionAnalysis(BaseModel):
    """Video compression analysis results"""
    analysis_id: str
    video_path: Path
    compression_type: CompressionType
    quality_metrics: Dict[str, float] = Field(default_factory=dict)
    artifacts_detected: List[str] = Field(default_factory=list)
    recompression_count: int = 0
    authenticity_indicators: List[str] = Field(default_factory=list)
    analysis_date: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class AuthenticityReport(BaseModel):
    """Comprehensive video authenticity report"""
    report_id: str
    video_path: Path
    overall_status: AuthenticityStatus
    confidence_score: float = Field(ge=0.0, le=1.0)
    metadata_analysis: Optional[Dict[str, Any]] = None
    deepfake_analysis: Optional[DeepfakeAnalysis] = None
    compression_analysis: Optional[CompressionAnalysis] = None
    fingerprint_analysis: Optional[VideoFingerprint] = None
    findings: List[str] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)
    analysis_date: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class DeepfakeDetector:
    """Detect deepfake videos using various techniques"""

    def __init__(self):
        """Initialize deepfake detector"""
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        logger.info("Initialized deepfake detector")

    def detect_deepfake(self, video_path: Path) -> DeepfakeAnalysis:
        """
        Detect if video contains deepfake content.

        Args:
            video_path: Path to video file

        Returns:
            DeepfakeAnalysis with detection results
        """
        import uuid

        analysis_id = str(uuid.uuid4())

        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        logger.info(f"Starting deepfake detection for {video_path}")

        # Open video
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")

        frames_analyzed = 0
        suspicious_frames = []
        detection_metrics = {}
        temporal_scores = []

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_analysis = self._analyze_frame_for_deepfake(frame, frames_analyzed)
                frames_analyzed += 1

                # Check for suspicious artifacts
                if frame_analysis['suspicious']:
                    suspicious_frames.append(frames_analyzed)

                temporal_scores.append(frame_analysis['temporal_consistency'])

                # Limit analysis for performance
                if frames_analyzed >= 300:  # Analyze up to 300 frames
                    break

        finally:
            cap.release()

        # Calculate overall metrics
        if temporal_scores:
            temporal_consistency = sum(temporal_scores) / len(temporal_scores)
        else:
            temporal_consistency = 0.0

        suspicious_ratio = len(suspicious_frames) / max(frames_analyzed, 1)

        # Determine if deepfake
        is_deepfake = suspicious_ratio > 0.3  # 30% threshold
        confidence_score = min(suspicious_ratio * 2, 1.0)

        detection_metrics = {
            'suspicious_frame_ratio': suspicious_ratio,
            'temporal_consistency': temporal_consistency,
            'face_detection_rate': detection_metrics.get('face_detection_rate', 0.0)
        }

        return DeepfakeAnalysis(
            analysis_id=analysis_id,
            video_path=video_path,
            is_deepfake=is_deepfake,
            confidence_score=confidence_score,
            frames_analyzed=frames_analyzed,
            suspicious_frames=suspicious_frames,
            detection_metrics=detection_metrics,
            temporal_consistency=temporal_consistency
        )

    def _analyze_frame_for_deepfake(self, frame: np.ndarray, frame_num: int) -> Dict[str, Any]:
        """Analyze individual frame for deepfake indicators"""
        suspicious = False
        temporal_consistency = 1.0

        # Convert to grayscale for analysis
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )

        if len(faces) > 0:
            # Analyze facial regions for artifacts
            for (x, y, w, h) in faces:
                face_region = frame[y:y+h, x:x+w]

                # Check for blurring artifacts around face boundary
                if self._detect_blurring_artifacts(face_region):
                    suspicious = True

                # Check for inconsistent lighting
                if self._detect_lighting_inconsistencies(face_region, frame):
                    suspicious = True

                # Check for compression artifacts in face region
                if self._detect_face_compression_artifacts(face_region):
                    suspicious = True

        # Check overall frame consistency
        temporal_consistency = self._calculate_temporal_consistency(frame)

        return {
            'suspicious': suspicious,
            'temporal_consistency': temporal_consistency,
            'faces_detected': len(faces)
        }

    def _detect_blurring_artifacts(self, face_region: np.ndarray) -> bool:
        """Detect artificial blurring around face boundaries"""
        # Convert to grayscale
        gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)

        # Calculate variance of Laplacian (blur metric)
        variance = cv2.Laplacian(gray, cv2.CV_64F).var()

        # Low variance indicates blur
        return variance < 100  # Threshold for blur detection

    def _detect_lighting_inconsistencies(self, face_region: np.ndarray, full_frame: np.ndarray) -> bool:
        """Detect inconsistent lighting between face and surroundings"""
        # Calculate average brightness of face region
        face_brightness = np.mean(cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY))

        # Calculate average brightness of surrounding area
        h, w = full_frame.shape[:2]
        fh, fw = face_region.shape[:2]

        # Sample surrounding area
        surrounding_samples = []
        if h > fh * 2 and w > fw * 2:
            # Take samples from corners
            sample_size = min(fh // 2, fw // 2)
            samples = [
                full_frame[:sample_size, :sample_size],  # Top-left
                full_frame[:sample_size, -sample_size:],  # Top-right
                full_frame[-sample_size:, :sample_size],  # Bottom-left
                full_frame[-sample_size:, -sample_size:]  # Bottom-right
            ]

            for sample in samples:
                surrounding_samples.append(np.mean(cv2.cvtColor(sample, cv2.COLOR_BGR2GRAY)))

        if surrounding_samples:
            avg_surrounding = np.mean(surrounding_samples)
            brightness_diff = abs(face_brightness - avg_surrounding)

            # Threshold for lighting inconsistency
            return brightness_diff > 50

        return False

    def _detect_face_compression_artifacts(self, face_region: np.ndarray) -> bool:
        """Detect compression artifacts in face region"""
        # Convert to grayscale
        gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)

        # Apply edge detection
        edges = cv2.Canny(gray, 50, 150)

        # Count edge pixels
        edge_density = np.sum(edges > 0) / edges.size

        # High edge density might indicate compression artifacts
        return edge_density > 0.2

    def _calculate_temporal_consistency(self, frame: np.ndarray) -> float:
        """Calculate temporal consistency score for frame"""
        # This would compare with previous frames in a real implementation
        # For now, return a base consistency score
        return 0.8


class VideoFingerprinter:
    """Generate content-based fingerprints for video duplicate detection"""

    def __init__(self):
        """Initialize video fingerprinter"""
        logger.info("Initialized video fingerprinter")

    def fingerprint_video(self, video_path: Path) -> VideoFingerprint:
        """
        Generate perceptual fingerprint of video content.

        Args:
            video_path: Path to video file

        Returns:
            VideoFingerprint with content signature
        """
        import uuid

        fingerprint_id = str(uuid.uuid4())

        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        logger.info(f"Generating fingerprint for {video_path}")

        # Open video
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")

        duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)

        # Extract key frames at regular intervals
        key_frames = self._extract_key_frames(cap)

        # Generate perceptual hash from key frames
        perceptual_hash = self._generate_perceptual_hash(key_frames)

        # Extract temporal features
        temporal_features = self._extract_temporal_features(cap)

        cap.release()

        return VideoFingerprint(
            fingerprint_id=fingerprint_id,
            video_path=video_path,
            perceptual_hash=perceptual_hash,
            temporal_features=temporal_features,
            key_frames=key_frames,
            duration=duration
        )

    def _extract_key_frames(self, cap: cv2.VideoCapture) -> List[KeyFrame]:
        """Extract key frames from video"""
        key_frames = []
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        # Extract frames at regular intervals (every 5 seconds or 10 frames max)
        interval = max(int(fps * 5), frame_count // 10)

        for i in range(0, frame_count, interval):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()

            if ret:
                # Convert to PIL Image for hashing
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(rgb_frame)

                # Generate perceptual hash
                frame_hash = str(imagehash.phash(pil_image))

                key_frame = KeyFrame(
                    frame_number=i,
                    timestamp=i / fps,
                    frame_hash=frame_hash,
                    features=self._extract_frame_features(frame)
                )

                key_frames.append(key_frame)

                if len(key_frames) >= 10:  # Limit to 10 key frames
                    break

        return key_frames

    def _extract_frame_features(self, frame: np.ndarray) -> Dict[str, Any]:
        """Extract features from frame"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Calculate basic statistics
        mean_brightness = float(np.mean(gray))
        std_brightness = float(np.std(gray))

        # Calculate edge density
        edges = cv2.Canny(gray, 50, 150)
        edge_density = float(np.sum(edges > 0) / edges.size)

        return {
            'mean_brightness': mean_brightness,
            'std_brightness': std_brightness,
            'edge_density': edge_density
        }

    def _generate_perceptual_hash(self, key_frames: List[KeyFrame]) -> str:
        """Generate overall perceptual hash from key frames"""
        if not key_frames:
            return "0" * 16

        # Combine frame hashes
        combined_hashes = "".join(frame.frame_hash for frame in key_frames)

        # Hash the combination
        return hashlib.md5(combined_hashes.encode()).hexdigest()[:16]

    def _extract_temporal_features(self, cap: cv2.VideoCapture) -> List[str]:
        """Extract temporal features from video"""
        # Reset to beginning
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        features = []
        prev_frame = None
        motion_vectors = []

        for _ in range(min(50, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))):
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if prev_frame is not None:
                # Calculate optical flow
                flow = cv2.calcOpticalFlowPyrLK(
                    prev_frame, gray, None, None
                )[0] if prev_frame is not None else None

                if flow is not None:
                    motion_magnitude = float(np.mean(np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)))
                    motion_vectors.append(motion_magnitude)

            prev_frame = gray

        # Classify motion patterns
        if motion_vectors:
            avg_motion = np.mean(motion_vectors)
            if avg_motion > 10:
                features.append("high_motion")
            elif avg_motion > 5:
                features.append("medium_motion")
            else:
                features.append("low_motion")

        return features


class CompressionAnalyzer:
    """Analyze video compression artifacts for authenticity"""

    def __init__(self):
        """Initialize compression analyzer"""
        logger.info("Initialized compression analyzer")

    def analyze_compression_artifacts(self, video_path: Path) -> CompressionAnalysis:
        """
        Analyze video compression artifacts for authenticity indicators.

        Args:
            video_path: Path to video file

        Returns:
            CompressionAnalysis with findings
        """
        import uuid

        analysis_id = str(uuid.uuid4())

        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        logger.info(f"Analyzing compression artifacts for {video_path}")

        # Get video metadata
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")

        # Determine compression type
        fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
        codec = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])

        compression_type = self._identify_compression_type(codec)

        # Analyze quality and artifacts
        quality_metrics = self._analyze_quality_metrics(cap)
        artifacts_detected = self._detect_compression_artifacts(cap)
        recompression_count = self._estimate_recompression_count(cap)
        authenticity_indicators = self._identify_authenticity_indicators(
            quality_metrics, artifacts_detected, recompression_count
        )

        cap.release()

        return CompressionAnalysis(
            analysis_id=analysis_id,
            video_path=video_path,
            compression_type=compression_type,
            quality_metrics=quality_metrics,
            artifacts_detected=artifacts_detected,
            recompression_count=recompression_count,
            authenticity_indicators=authenticity_indicators
        )

    def _identify_compression_type(self, codec: str) -> CompressionType:
        """Identify compression type from codec"""
        codec_lower = codec.lower()

        if 'h264' in codec_lower or 'avc' in codec_lower:
            return CompressionType.H264
        elif 'h265' in codec_lower or 'hevc' in codec_lower:
            return CompressionType.H265
        elif 'mp4v' in codec_lower or 'mpeg' in codec_lower:
            return CompressionType.MPEG4
        elif 'vp8' in codec_lower:
            return CompressionType.VP8
        elif 'vp9' in codec_lower:
            return CompressionType.VP9
        elif 'av01' in codec_lower:
            return CompressionType.AV1
        else:
            return CompressionType.UNKNOWN

    def _analyze_quality_metrics(self, cap: cv2.VideoCapture) -> Dict[str, float]:
        """Analyze video quality metrics"""
        metrics = {}

        # Sample frames for analysis
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        sample_frames = min(20, frame_count)

        brightness_values = []
        contrast_values = []
        sharpness_values = []

        for i in range(0, frame_count, frame_count // sample_frames):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()

            if ret:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # Brightness
                brightness = float(np.mean(gray))
                brightness_values.append(brightness)

                # Contrast (standard deviation)
                contrast = float(np.std(gray))
                contrast_values.append(contrast)

                # Sharpness (Laplacian variance)
                sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
                sharpness_values.append(sharpness)

        if brightness_values:
            metrics['avg_brightness'] = float(np.mean(brightness_values))
            metrics['avg_contrast'] = float(np.mean(contrast_values))
            metrics['avg_sharpness'] = float(np.mean(sharpness_values))
            metrics['brightness_stability'] = float(1.0 / (1.0 + np.std(brightness_values)))

        return metrics

    def _detect_compression_artifacts(self, cap: cv2.VideoCapture) -> List[str]:
        """Detect various compression artifacts"""
        artifacts = []

        # Sample a few frames
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        for i in range(0, min(frame_count, 100), 10):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()

            if ret:
                # Check for blocking artifacts
                if self._detect_blocking_artifacts(frame):
                    if "blocking" not in artifacts:
                        artifacts.append("blocking")

                # Check for ringing artifacts
                if self._detect_ringing_artifacts(frame):
                    if "ringing" not in artifacts:
                        artifacts.append("ringing")

                # Check for mosquito noise
                if self._detect_mosquito_noise(frame):
                    if "mosquito_noise" not in artifacts:
                        artifacts.append("mosquito_noise")

        return artifacts

    def _detect_blocking_artifacts(self, frame: np.ndarray) -> bool:
        """Detect blocking artifacts from compression"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Look for regular 8x8 or 16x16 block patterns
        # Apply horizontal and vertical gradients
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

        # Check for regular patterns in gradients
        h, w = gray.shape
        block_size = 8

        # Sample block boundaries
        block_edge_strength = 0
        non_block_edge_strength = 0

        for y in range(block_size, h - block_size, block_size):
            for x in range(block_size, w - block_size, block_size):
                # Check block boundary
                block_edge_strength += abs(float(grad_x[y, x])) + abs(float(grad_y[y, x]))

                # Check non-boundary
                non_block_edge_strength += abs(float(grad_x[y + 4, x + 4])) + abs(float(grad_y[y + 4, x + 4]))

        # If block boundaries have significantly stronger edges, blocking is present
        if non_block_edge_strength > 0:
            blocking_ratio = block_edge_strength / non_block_edge_strength
            return blocking_ratio > 1.5

        return False

    def _detect_ringing_artifacts(self, frame: np.ndarray) -> bool:
        """Detect ringing artifacts near edges"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect edges
        edges = cv2.Canny(gray, 50, 150)

        # Dilate edges to create edge regions
        kernel = np.ones((5, 5), np.uint8)
        edge_regions = cv2.dilate(edges, kernel, iterations=2)

        # Calculate variance in edge regions
        edge_variance = float(np.var(gray[edge_regions > 0]))

        # Calculate variance in non-edge regions
        non_edge_variance = float(np.var(gray[edge_regions == 0]))

        # High variance near edges indicates ringing
        if non_edge_variance > 0:
            ringing_ratio = edge_variance / non_edge_variance
            return ringing_ratio > 2.0

        return False

    def _detect_mosquito_noise(self, frame: np.ndarray) -> bool:
        """Detect mosquito noise artifacts"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply high-pass filter to detect high-frequency noise
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        high_freq = cv2.subtract(gray, blur)

        # Calculate noise level
        noise_level = float(np.std(high_freq))

        # Threshold for mosquito noise detection
        return noise_level > 10

    def _estimate_recompression_count(self, cap: cv2.VideoCapture) -> int:
        """Estimate number of times video has been recompressed"""
        # This is a simplified heuristic based on quality degradation
        quality_metrics = self._analyze_quality_metrics(cap)

        avg_sharpness = quality_metrics.get('avg_sharpness', 1000)

        # Lower sharpness indicates more compression
        if avg_sharpness > 800:
            return 0  # Original or lightly compressed
        elif avg_sharpness > 400:
            return 1  # Once recompressed
        elif avg_sharpness > 200:
            return 2  # Twice recompressed
        else:
            return 3  # Heavily recompressed

    def _identify_authenticity_indicators(
        self,
        quality_metrics: Dict[str, float],
        artifacts: List[str],
        recompression_count: int
    ) -> List[str]:
        """Identify indicators of authenticity or manipulation"""
        indicators = []

        # High quality with minimal artifacts suggests original
        if (quality_metrics.get('avg_sharpness', 0) > 600 and
            len(artifacts) < 2 and
            recompression_count <= 1):
            indicators.append("high_quality_original")

        # Multiple compression artifacts suggest reprocessing
        if len(artifacts) > 2:
            indicators.append("multiple_compression_artifacts")

        # High recompression count suggests manipulation
        if recompression_count > 2:
            indicators.append("multiple_recompression")

        # Low quality with specific artifacts suggests manipulation
        if (quality_metrics.get('avg_sharpness', 1000) < 200 and
            "blocking" in artifacts):
            indicators.append("quality_degradation_suspicious")

        return indicators


class FrameAnalyzer:
    """Analyze individual frames for manipulation detection"""

    def __init__(self):
        """Initialize frame analyzer"""
        logger.info("Initialized frame analyzer")

    def extract_key_frames(self, video_path: Path) -> List[KeyFrame]:
        """
        Extract and analyze key frames from video.

        Args:
            video_path: Path to video file

        Returns:
            List of KeyFrame objects
        """
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        logger.info(f"Extracting key frames from {video_path}")

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")

        key_frames = []
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        # Extract frames at scene changes or regular intervals
        prev_frame = None
        scene_threshold = 30.0  # Threshold for scene change detection

        for i in range(0, frame_count, max(1, frame_count // 50)):  # Max 50 key frames
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()

            if not ret:
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Check for scene change
            is_key_frame = prev_frame is None
            if prev_frame is not None:
                # Calculate frame difference
                diff = cv2.absdiff(prev_frame, gray)
                diff_score = float(np.mean(diff))

                if diff_score > scene_threshold:
                    is_key_frame = True

            if is_key_frame:
                # Generate frame hash
                pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                frame_hash = str(imagehash.phash(pil_image))

                # Detect artifacts
                artifacts = self._detect_frame_artifacts(frame)

                key_frame = KeyFrame(
                    frame_number=i,
                    timestamp=i / fps if fps > 0 else 0,
                    frame_hash=frame_hash,
                    features=self._extract_frame_features(frame),
                    artifacts=artifacts
                )

                key_frames.append(key_frame)

            prev_frame = gray

        cap.release()
        logger.info(f"Extracted {len(key_frames)} key frames")

        return key_frames

    def _extract_frame_features(self, frame: np.ndarray) -> Dict[str, Any]:
        """Extract features from a single frame"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Basic statistics
        mean_brightness = float(np.mean(gray))
        std_brightness = float(np.std(gray))

        # Texture analysis
        edges = cv2.Canny(gray, 50, 150)
        edge_density = float(np.sum(edges > 0) / edges.size)

        # Color histogram
        hist_b = cv2.calcHist([frame], [0], None, [256], [0, 256])
        hist_g = cv2.calcHist([frame], [1], None, [256], [0, 256])
        hist_r = cv2.calcHist([frame], [2], None, [256], [0, 256])

        color_variance = float(np.var(hist_b) + np.var(hist_g) + np.var(hist_r))

        return {
            'mean_brightness': mean_brightness,
            'std_brightness': std_brightness,
            'edge_density': edge_density,
            'color_variance': color_variance,
            'frame_shape': frame.shape[:2]
        }

    def _detect_frame_artifacts(self, frame: np.ndarray) -> List[str]:
        """Detect various artifacts in frame"""
        artifacts = []

        # Convert to different color spaces for analysis
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Check for compression artifacts
        if self._has_compression_artifacts(gray):
            artifacts.append("compression_artifacts")

        # Check for splicing artifacts
        if self._has_splicing_artifacts(frame):
            artifacts.append("splicing_artifacts")

        # Check for inconsistent lighting
        if self._has_lighting_inconsistencies(frame):
            artifacts.append("lighting_inconsistencies")

        # Check for upscaling artifacts
        if self._has_upscaling_artifacts(gray):
            artifacts.append("upscaling_artifacts")

        return artifacts

    def _has_compression_artifacts(self, gray: np.ndarray) -> bool:
        """Check for compression artifacts"""
        # Look for JPEG-like blocking
        return self._detect_blocking_artifacts_frame(gray)

    def _detect_blocking_artifacts_frame(self, gray: np.ndarray) -> bool:
        """Detect blocking artifacts in frame"""
        h, w = gray.shape

        # Check for 8x8 DCT blocking
        block_size = 8
        edge_strength_at_blocks = 0
        edge_strength_elsewhere = 0
        samples = 0

        for y in range(block_size, h - block_size, block_size):
            for x in range(block_size, w - block_size, block_size):
                # Check horizontal edge at block boundary
                edge_at_block = abs(int(gray[y, x]) - int(gray[y-1, x]))
                edge_elsewhere = abs(int(gray[y-4, x]) - int(gray[y-5, x]))

                edge_strength_at_blocks += edge_at_block
                edge_strength_elsewhere += edge_elsewhere
                samples += 1

        if samples > 0 and edge_strength_elsewhere > 0:
            blocking_ratio = edge_strength_at_blocks / edge_strength_elsewhere
            return blocking_ratio > 1.3

        return False

    def _has_splicing_artifacts(self, frame: np.ndarray) -> bool:
        """Check for image splicing artifacts"""
        # Look for inconsistent noise patterns
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Divide frame into regions and analyze noise
        h, w = gray.shape
        region_size = min(h // 4, w // 4)

        if region_size < 50:  # Too small for analysis
            return False

        noise_levels = []

        for y in range(0, h - region_size, region_size):
            for x in range(0, w - region_size, region_size):
                region = gray[y:y+region_size, x:x+region_size]

                # Calculate noise level using high-pass filter
                blur = cv2.GaussianBlur(region, (5, 5), 0)
                noise = cv2.subtract(region, blur)
                noise_level = np.std(noise)
                noise_levels.append(noise_level)

        if len(noise_levels) > 1:
            # High variance in noise levels suggests splicing
            noise_variance = np.var(noise_levels)
            return noise_variance > 20  # Threshold for splicing detection

        return False

    def _has_lighting_inconsistencies(self, frame: np.ndarray) -> bool:
        """Check for inconsistent lighting"""
        # Analyze brightness distribution across frame
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape

        # Divide into 3x3 grid
        grid_h, grid_w = h // 3, w // 3
        brightness_values = []

        for i in range(3):
            for j in range(3):
                y1, y2 = i * grid_h, (i + 1) * grid_h
                x1, x2 = j * grid_w, (j + 1) * grid_w
                region = gray[y1:y2, x1:x2]
                brightness_values.append(np.mean(region))

        # Check for unusual brightness variations
        brightness_std = np.std(brightness_values)
        brightness_range = max(brightness_values) - min(brightness_values)

        # Thresholds for inconsistent lighting
        return brightness_std > 30 or brightness_range > 100

    def _has_upscaling_artifacts(self, gray: np.ndarray) -> bool:
        """Check for upscaling/interpolation artifacts"""
        # Look for regular patterns that suggest interpolation
        # Calculate second derivative to find interpolation artifacts
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)

        # Regular patterns in Laplacian suggest interpolation
        laplacian_std = np.std(laplacian)

        # Low variance in second derivative suggests interpolation
        return laplacian_std < 50  # Threshold for upscaling detection