"""
Lemkin Video Authentication Toolkit - Deepfake Detection Module

This module provides deepfake detection capabilities using state-of-the-art
machine learning models for video authenticity verification.

Features:
- Multiple ML model support (EfficientNet, ResNet, etc.)
- Frame-by-frame analysis
- Face detection and tracking
- Temporal consistency analysis
- Confidence scoring with uncertainty quantification
"""

import cv2
import numpy as np
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import hashlib
import json

try:
    import torch
    import torch.nn.functional as F
    import torchvision.transforms as transforms
    from torchvision.models import efficientnet_b4
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False

try:
    import face_recognition
    FACE_RECOGNITION_AVAILABLE = True
except ImportError:
    FACE_RECOGNITION_AVAILABLE = False

from .core import (
    VideoAuthConfig, DeepfakeAnalysis, TamperingIndicator, 
    TamperingType, AnalysisStatus
)

logger = logging.getLogger(__name__)


class DeepfakeDetector:
    """
    Advanced deepfake detection system using multiple ML models
    and analysis techniques for comprehensive video authenticity assessment.
    """
    
    def __init__(self, config: VideoAuthConfig):
        """Initialize deepfake detector with configuration"""
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.DeepfakeDetector")
        
        # Model management
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() and config.use_gpu else "cpu"
        self.face_detector = None
        self.mediapipe_face_detection = None
        
        # Analysis settings
        self.frame_skip_interval = max(1, config.frame_analysis_interval // 10)
        self.batch_size = 8 if self.device == "cuda" else 4
        
        # Initialize components
        self._initialize_face_detection()
        if config.deepfake_model_path:
            self._load_deepfake_model(config.deepfake_model_path)
        
        self.logger.info(f"Deepfake detector initialized (device: {self.device})")
    
    def _initialize_face_detection(self):
        """Initialize face detection systems"""
        if MEDIAPIPE_AVAILABLE:
            mp_face_detection = mp.solutions.face_detection
            self.mediapipe_face_detection = mp_face_detection.FaceDetection(
                model_selection=1, min_detection_confidence=0.5
            )
            self.logger.info("MediaPipe face detection initialized")
        
        if FACE_RECOGNITION_AVAILABLE:
            self.logger.info("Face recognition library available")
        
        if not MEDIAPIPE_AVAILABLE and not FACE_RECOGNITION_AVAILABLE:
            self.logger.warning("No face detection libraries available")
    
    def _load_deepfake_model(self, model_path: str):
        """Load deepfake detection model"""
        if not TORCH_AVAILABLE:
            self.logger.error("PyTorch not available - cannot load deepfake model")
            return
        
        try:
            # Load pre-trained model or create default
            if Path(model_path).exists():
                self.model = torch.load(model_path, map_location=self.device)
                self.logger.info(f"Loaded deepfake model from {model_path}")
            else:
                # Create a basic model architecture for demonstration
                self.model = self._create_default_model()
                self.logger.warning(f"Model path {model_path} not found, using default model")
            
            self.model.eval()
            
        except Exception as e:
            self.logger.error(f"Failed to load deepfake model: {str(e)}")
            self.model = None
    
    def _create_default_model(self):
        """Create a default deepfake detection model"""
        # Use EfficientNet-B4 as base architecture
        model = efficientnet_b4(pretrained=True)
        
        # Modify classifier for binary classification (real/fake)
        num_features = model.classifier.in_features
        model.classifier = torch.nn.Sequential(
            torch.nn.Dropout(0.3),
            torch.nn.Linear(num_features, 512),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(512, 2)  # Real vs Fake
        )
        
        return model.to(self.device)
    
    def detect_deepfake(self, video_path: Path) -> DeepfakeAnalysis:
        """
        Detect deepfake manipulation in video
        
        Args:
            video_path: Path to video file
            
        Returns:
            DeepfakeAnalysis: Comprehensive deepfake analysis results
        """
        start_time = datetime.utcnow()
        self.logger.info(f"Starting deepfake detection for: {video_path}")
        
        # Calculate video hash
        video_hash = self._calculate_video_hash(video_path)
        
        try:
            # Extract frames and analyze
            frames_data = self._extract_frames_for_analysis(video_path)
            
            if not frames_data['frames']:
                raise ValueError("No frames could be extracted from video")
            
            # Perform deepfake detection
            detection_results = self._analyze_frames_for_deepfake(frames_data['frames'])
            
            # Analyze faces if available
            face_analysis = self._analyze_face_consistency(frames_data['frames'])
            
            # Detect temporal inconsistencies
            temporal_analysis = self._detect_temporal_inconsistencies(frames_data['frames'])
            
            # Calculate overall assessment
            overall_assessment = self._calculate_overall_deepfake_assessment(
                detection_results, face_analysis, temporal_analysis
            )
            
            # Create analysis result
            analysis = DeepfakeAnalysis(
                video_hash=video_hash,
                is_deepfake=overall_assessment['is_deepfake'],
                confidence=overall_assessment['confidence'],
                deepfake_probability=overall_assessment['probability'],
                model_used=self._get_model_name(),
                model_version="1.0",
                total_frames_analyzed=len(frames_data['frames']),
                positive_frames=detection_results['positive_frames'],
                negative_frames=detection_results['negative_frames'],
                uncertain_frames=detection_results['uncertain_frames'],
                faces_detected=face_analysis['total_faces'],
                face_consistency_score=face_analysis['consistency_score'],
                identity_changes_detected=face_analysis['identity_changes'],
                compression_artifacts=temporal_analysis['compression_artifacts'],
                temporal_inconsistencies=temporal_analysis['inconsistencies'],
                pixel_level_anomalies=temporal_analysis['pixel_anomalies'],
                processing_time_seconds=(datetime.utcnow() - start_time).total_seconds(),
                gpu_used=self.device == "cuda",
                frame_analyses=detection_results['frame_details'],
                suspicious_regions=face_analysis['suspicious_regions']
            )
            
            self.logger.info(f"Deepfake detection completed: {analysis.is_deepfake} "
                           f"(confidence: {analysis.confidence:.2f})")
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Deepfake detection failed: {str(e)}")
            # Return failed analysis
            return DeepfakeAnalysis(
                video_hash=video_hash,
                is_deepfake=False,
                confidence=0.0,
                deepfake_probability=0.0,
                model_used="unknown",
                model_version="1.0",
                total_frames_analyzed=0,
                processing_time_seconds=(datetime.utcnow() - start_time).total_seconds(),
                gpu_used=False
            )
    
    def _calculate_video_hash(self, video_path: Path) -> str:
        """Calculate SHA-256 hash of video file"""
        hash_obj = hashlib.sha256()
        with open(video_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_obj.update(chunk)
        return hash_obj.hexdigest()
    
    def _extract_frames_for_analysis(self, video_path: Path) -> Dict[str, Any]:
        """Extract frames from video for deepfake analysis"""
        cap = cv2.VideoCapture(str(video_path))
        frames = []
        frame_indices = []
        
        try:
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            frame_count = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Skip frames based on analysis interval
                if frame_count % self.frame_skip_interval == 0:
                    # Convert BGR to RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(frame_rgb)
                    frame_indices.append(frame_count)
                
                frame_count += 1
                
                # Limit number of frames to prevent memory issues
                if len(frames) >= 1000:  # Max 1000 frames
                    break
            
            return {
                'frames': frames,
                'frame_indices': frame_indices,
                'total_frames': total_frames,
                'fps': fps
            }
            
        finally:
            cap.release()
    
    def _analyze_frames_for_deepfake(self, frames: List[np.ndarray]) -> Dict[str, Any]:
        """Analyze frames for deepfake indicators"""
        if not self.model or not TORCH_AVAILABLE:
            return self._fallback_deepfake_analysis(frames)
        
        positive_frames = 0
        negative_frames = 0
        uncertain_frames = 0
        frame_details = []
        
        # Process frames in batches
        for i in range(0, len(frames), self.batch_size):
            batch_frames = frames[i:i + self.batch_size]
            batch_results = self._process_frame_batch(batch_frames, i)
            
            for result in batch_results:
                if result['deepfake_probability'] > self.config.deepfake_threshold:
                    positive_frames += 1
                elif result['deepfake_probability'] < 1 - self.config.deepfake_threshold:
                    negative_frames += 1
                else:
                    uncertain_frames += 1
                
                frame_details.append(result)
        
        return {
            'positive_frames': positive_frames,
            'negative_frames': negative_frames,
            'uncertain_frames': uncertain_frames,
            'frame_details': frame_details
        }
    
    def _process_frame_batch(self, frames: List[np.ndarray], start_idx: int) -> List[Dict[str, Any]]:
        """Process a batch of frames through the deepfake model"""
        if not self.model:
            return []
        
        # Preprocess frames
        preprocessed = self._preprocess_frames_for_model(frames)
        
        results = []
        with torch.no_grad():
            # Run inference
            outputs = self.model(preprocessed)
            probabilities = F.softmax(outputs, dim=1)
            
            for i, (frame_probs, original_frame) in enumerate(zip(probabilities, frames)):
                # Extract deepfake probability (assuming class 1 is fake)
                deepfake_prob = frame_probs[1].item()
                confidence = max(frame_probs).item()
                
                result = {
                    'frame_index': start_idx + i,
                    'deepfake_probability': deepfake_prob,
                    'confidence': confidence,
                    'is_deepfake': deepfake_prob > self.config.deepfake_threshold,
                    'raw_scores': frame_probs.tolist(),
                    'timestamp': (start_idx + i) / 30.0,  # Assume 30 FPS
                    'analysis_method': 'cnn_classification'
                }
                
                results.append(result)
        
        return results
    
    def _preprocess_frames_for_model(self, frames: List[np.ndarray]) -> torch.Tensor:
        """Preprocess frames for model input"""
        # Standard preprocessing for EfficientNet
        preprocess = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        batch = []
        for frame in frames:
            processed = preprocess(frame)
            batch.append(processed)
        
        return torch.stack(batch).to(self.device)
    
    def _fallback_deepfake_analysis(self, frames: List[np.ndarray]) -> Dict[str, Any]:
        """Fallback analysis when ML model is not available"""
        self.logger.warning("Using fallback deepfake analysis (no ML model)")
        
        # Simple heuristic-based analysis
        suspicious_frames = 0
        frame_details = []
        
        for i, frame in enumerate(frames):
            # Basic checks for obvious manipulation signs
            suspicious_score = self._calculate_basic_suspicious_score(frame)
            
            is_suspicious = suspicious_score > 0.5
            if is_suspicious:
                suspicious_frames += 1
            
            frame_details.append({
                'frame_index': i,
                'deepfake_probability': suspicious_score,
                'confidence': 0.3,  # Low confidence for heuristic method
                'is_deepfake': is_suspicious,
                'analysis_method': 'heuristic_fallback'
            })
        
        return {
            'positive_frames': suspicious_frames,
            'negative_frames': len(frames) - suspicious_frames,
            'uncertain_frames': 0,
            'frame_details': frame_details
        }
    
    def _calculate_basic_suspicious_score(self, frame: np.ndarray) -> float:
        """Calculate basic suspicious score using heuristics"""
        score = 0.0
        
        # Check for unusual edge patterns (common in deepfakes)
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.mean(edges) / 255.0
        
        if edge_density > 0.15:  # High edge density might indicate artifacts
            score += 0.2
        
        # Check for color distribution anomalies
        color_std = np.std(frame, axis=(0, 1))
        if np.max(color_std) > 50:  # High color variance
            score += 0.1
        
        # Check for blurriness (often indicates generation artifacts)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        if laplacian_var < 100:  # Low variance indicates blur
            score += 0.2
        
        return min(score, 1.0)
    
    def _analyze_face_consistency(self, frames: List[np.ndarray]) -> Dict[str, Any]:
        """Analyze face consistency across frames"""
        if not (MEDIAPIPE_AVAILABLE or FACE_RECOGNITION_AVAILABLE):
            return {
                'total_faces': 0,
                'consistency_score': 1.0,
                'identity_changes': 0,
                'suspicious_regions': []
            }
        
        face_data = []
        suspicious_regions = []
        
        for i, frame in enumerate(frames[::10]):  # Sample every 10th frame
            faces = self._detect_faces_in_frame(frame)
            if faces:
                face_data.append({
                    'frame_index': i * 10,
                    'faces': faces,
                    'face_count': len(faces)
                })
        
        # Analyze consistency
        consistency_analysis = self._calculate_face_consistency(face_data)
        
        return {
            'total_faces': sum(data['face_count'] for data in face_data),
            'consistency_score': consistency_analysis['score'],
            'identity_changes': consistency_analysis['changes'],
            'suspicious_regions': suspicious_regions
        }
    
    def _detect_faces_in_frame(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """Detect faces in a single frame"""
        faces = []
        
        if MEDIAPIPE_AVAILABLE and self.mediapipe_face_detection:
            results = self.mediapipe_face_detection.process(frame)
            if results.detections:
                for detection in results.detections:
                    bbox = detection.location_data.relative_bounding_box
                    h, w = frame.shape[:2]
                    
                    faces.append({
                        'bbox': [
                            int(bbox.xmin * w),
                            int(bbox.ymin * h),
                            int(bbox.width * w),
                            int(bbox.height * h)
                        ],
                        'confidence': detection.score[0],
                        'method': 'mediapipe'
                    })
        
        elif FACE_RECOGNITION_AVAILABLE:
            # Use face_recognition library
            face_locations = face_recognition.face_locations(frame)
            for (top, right, bottom, left) in face_locations:
                faces.append({
                    'bbox': [left, top, right - left, bottom - top],
                    'confidence': 0.8,  # Default confidence
                    'method': 'face_recognition'
                })
        
        return faces
    
    def _calculate_face_consistency(self, face_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate face consistency across frames"""
        if len(face_data) < 2:
            return {'score': 1.0, 'changes': 0}
        
        # Simple consistency check based on face count and position
        face_counts = [data['face_count'] for data in face_data]
        count_consistency = 1.0 - (np.std(face_counts) / (np.mean(face_counts) + 1e-6))
        
        # Detect identity changes (simplified)
        identity_changes = 0
        prev_count = face_counts[0]
        for count in face_counts[1:]:
            if abs(count - prev_count) > 1:  # Significant change in face count
                identity_changes += 1
            prev_count = count
        
        return {
            'score': max(0.0, count_consistency),
            'changes': identity_changes
        }
    
    def _detect_temporal_inconsistencies(self, frames: List[np.ndarray]) -> Dict[str, Any]:
        """Detect temporal inconsistencies that may indicate manipulation"""
        inconsistencies = []
        compression_artifacts = []
        pixel_anomalies = []
        
        if len(frames) < 3:
            return {
                'inconsistencies': inconsistencies,
                'compression_artifacts': compression_artifacts,
                'pixel_anomalies': pixel_anomalies
            }
        
        # Analyze frame-to-frame differences
        for i in range(1, len(frames) - 1):
            prev_frame = cv2.cvtColor(frames[i-1], cv2.COLOR_RGB2GRAY)
            curr_frame = cv2.cvtColor(frames[i], cv2.COLOR_RGB2GRAY)
            next_frame = cv2.cvtColor(frames[i+1], cv2.COLOR_RGB2GRAY)
            
            # Calculate optical flow between consecutive frames
            flow = cv2.calcOpticalFlowPyrLK(
                prev_frame, curr_frame, None, None
            )
            
            # Detect sudden motion changes
            if self._detect_motion_inconsistency(prev_frame, curr_frame, next_frame):
                inconsistencies.append(f"Motion inconsistency at frame {i}")
            
            # Check for compression artifacts
            artifacts = self._detect_compression_artifacts(curr_frame)
            if artifacts:
                compression_artifacts.extend(artifacts)
        
        return {
            'inconsistencies': inconsistencies,
            'compression_artifacts': compression_artifacts,
            'pixel_anomalies': pixel_anomalies
        }
    
    def _detect_motion_inconsistency(self, prev_frame: np.ndarray, 
                                   curr_frame: np.ndarray, 
                                   next_frame: np.ndarray) -> bool:
        """Detect motion inconsistencies between frames"""
        # Calculate frame differences
        diff1 = cv2.absdiff(prev_frame, curr_frame)
        diff2 = cv2.absdiff(curr_frame, next_frame)
        
        # Calculate motion magnitude
        motion1 = np.mean(diff1)
        motion2 = np.mean(diff2)
        
        # Check for sudden motion changes
        motion_ratio = abs(motion1 - motion2) / (motion1 + motion2 + 1e-6)
        
        return motion_ratio > 0.5  # Threshold for inconsistency
    
    def _detect_compression_artifacts(self, frame: np.ndarray) -> List[str]:
        """Detect compression artifacts in frame"""
        artifacts = []
        
        # Detect blocking artifacts (8x8 DCT blocks)
        h, w = frame.shape
        block_size = 8
        
        for y in range(0, h - block_size, block_size):
            for x in range(0, w - block_size, block_size):
                block = frame[y:y+block_size, x:x+block_size]
                block_variance = np.var(block)
                
                # Low variance might indicate blocking
                if block_variance < 10:
                    artifacts.append(f"Blocking artifact at ({x},{y})")
        
        return artifacts
    
    def _calculate_overall_deepfake_assessment(self, 
                                             detection_results: Dict[str, Any],
                                             face_analysis: Dict[str, Any],
                                             temporal_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate overall deepfake assessment"""
        
        # Weight different analysis components
        weights = {
            'detection': 0.6,
            'face_consistency': 0.25,
            'temporal': 0.15
        }
        
        # Calculate detection score
        total_frames = (detection_results['positive_frames'] + 
                       detection_results['negative_frames'] + 
                       detection_results['uncertain_frames'])
        
        if total_frames > 0:
            detection_score = detection_results['positive_frames'] / total_frames
        else:
            detection_score = 0.0
        
        # Face consistency score (inverted - low consistency means higher deepfake probability)
        face_score = 1.0 - face_analysis['consistency_score']
        
        # Temporal inconsistency score
        temporal_score = min(1.0, len(temporal_analysis['inconsistencies']) / 10.0)
        
        # Calculate weighted average
        overall_probability = (
            detection_score * weights['detection'] +
            face_score * weights['face_consistency'] +
            temporal_score * weights['temporal']
        )
        
        # Calculate confidence based on agreement between methods
        confidence_factors = []
        if detection_results['positive_frames'] + detection_results['negative_frames'] > 0:
            detection_confidence = max(detection_results['positive_frames'], 
                                     detection_results['negative_frames']) / total_frames
            confidence_factors.append(detection_confidence)
        
        confidence_factors.append(face_analysis['consistency_score'])
        
        overall_confidence = np.mean(confidence_factors) if confidence_factors else 0.5
        
        return {
            'is_deepfake': overall_probability > self.config.deepfake_threshold,
            'probability': overall_probability,
            'confidence': overall_confidence
        }
    
    def _get_model_name(self) -> str:
        """Get the name of the current model"""
        if self.model:
            return "EfficientNet-B4"
        return "Heuristic-Fallback"
    
    def create_tampering_indicators(self, analysis: DeepfakeAnalysis) -> List[TamperingIndicator]:
        """Create tampering indicators from deepfake analysis"""
        indicators = []
        
        if analysis.is_deepfake:
            # Main deepfake indicator
            indicator = TamperingIndicator(
                tampering_type=TamperingType.DEEPFAKE,
                confidence=analysis.confidence,
                description=f"Deepfake detected with {analysis.deepfake_probability:.1%} probability",
                evidence={
                    'positive_frames': analysis.positive_frames,
                    'total_frames': analysis.total_frames_analyzed,
                    'model_used': analysis.model_used
                },
                analysis_method="ml_classification",
                severity_score=analysis.deepfake_probability * 10,
                is_critical=analysis.confidence > 0.8
            )
            indicators.append(indicator)
        
        # Face consistency indicators
        if analysis.face_consistency_score and analysis.face_consistency_score < 0.7:
            indicator = TamperingIndicator(
                tampering_type=TamperingType.FACE_SWAP,
                confidence=1.0 - analysis.face_consistency_score,
                description=f"Face inconsistency detected (score: {analysis.face_consistency_score:.2f})",
                evidence={'consistency_score': analysis.face_consistency_score},
                analysis_method="face_tracking",
                severity_score=(1.0 - analysis.face_consistency_score) * 8,
                is_critical=analysis.face_consistency_score < 0.5
            )
            indicators.append(indicator)
        
        return indicators