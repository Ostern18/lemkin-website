"""
Video-based PII redaction combining image and audio techniques.

This module provides automated redaction of PII in video content by
combining image redaction (faces, license plates) with audio redaction.
"""

import hashlib
import time
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

import cv2
import numpy as np
from moviepy.editor import VideoFileClip, AudioFileClip
from loguru import logger

from .core import (
    PIIEntity, EntityType, ConfidenceLevel, RedactionConfig,
    RedactionResult, RedactionType
)
from .image_redactor import ImageRedactor
from .audio_redactor import AudioRedactor


class VideoRedactor:
    """Video-based PII redaction using combined image and audio processing."""
    
    def __init__(self, config: RedactionConfig):
        """Initialize video redactor with configuration."""
        self.config = config
        self.logger = logger
        
        # Initialize component redactors
        self.image_redactor = ImageRedactor(config)
        self.audio_redactor = AudioRedactor(config)
        
        # Video processing parameters
        self.frame_skip = 1  # Process every frame by default
        self.temp_dir = Path("temp_video_processing")
        
        self.logger.info("VideoRedactor initialized")
    
    def extract_frames(self, video_path: Path, output_dir: Path) -> List[Path]:
        """Extract frames from video for processing."""
        output_dir.mkdir(parents=True, exist_ok=True)
        frame_paths = []
        
        try:
            cap = cv2.VideoCapture(str(video_path))
            frame_count = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_count % self.frame_skip == 0:
                    frame_path = output_dir / f"frame_{frame_count:06d}.jpg"
                    cv2.imwrite(str(frame_path), frame)
                    frame_paths.append(frame_path)
                
                frame_count += 1
            
            cap.release()
            
        except Exception as e:
            self.logger.error(f"Frame extraction failed: {e}")
            
        return frame_paths
    
    def process_video_frames(self, frame_paths: List[Path]) -> List[Tuple[int, List[PIIEntity]]]:
        """Process extracted frames to detect PII entities."""
        frame_entities = []
        
        for i, frame_path in enumerate(frame_paths):
            try:
                # Load frame
                frame = cv2.imread(str(frame_path))
                if frame is None:
                    continue
                
                # Detect entities using image redactor
                face_entities = self.image_redactor.detect_faces(frame)
                plate_entities = self.image_redactor.detect_license_plates(frame)
                text_entities = self.image_redactor.detect_text_regions(frame)
                
                # Combine entities
                all_entities = face_entities + plate_entities + text_entities
                
                # Remove overlapping entities
                all_entities = self.image_redactor._remove_overlapping_entities(all_entities)
                
                # Add frame timestamp metadata
                frame_time = i * self.frame_skip / 30.0  # Assuming 30 FPS
                for entity in all_entities:
                    entity.metadata["frame_number"] = i * self.frame_skip
                    entity.metadata["frame_time"] = frame_time
                
                frame_entities.append((i * self.frame_skip, all_entities))
                
            except Exception as e:
                self.logger.error(f"Failed to process frame {frame_path}: {e}")
        
        return frame_entities
    
    def process_audio_track(self, video_path: Path) -> Tuple[Optional[np.ndarray], List[PIIEntity], int]:
        """Extract and process audio track from video."""
        try:
            # Extract audio using moviepy
            video_clip = VideoFileClip(str(video_path))
            if video_clip.audio is None:
                return None, [], 0
            
            # Get audio data
            audio = video_clip.audio.to_soundarray()
            sr = video_clip.audio.fps
            
            # Convert stereo to mono if needed
            if len(audio.shape) > 1:
                audio = np.mean(audio, axis=1)
            
            # Detect speech entities
            speech_entities = self.audio_redactor.detect_speech_segments(audio, sr)
            
            video_clip.close()
            
            return audio, speech_entities, sr
            
        except Exception as e:
            self.logger.error(f"Audio processing failed: {e}")
            return None, [], 0
    
    def merge_temporal_entities(self, frame_entities: List[Tuple[int, List[PIIEntity]]], 
                              audio_entities: List[PIIEntity]) -> List[PIIEntity]:
        """Merge entities from frames and audio with temporal correlation."""
        all_entities = []
        
        # Add frame entities
        for frame_num, entities in frame_entities:
            for entity in entities:
                # Convert frame position to video timeline
                entity.metadata["content_type"] = "video_frame"
                all_entities.append(entity)
        
        # Add audio entities
        for entity in audio_entities:
            entity.metadata["content_type"] = "video_audio"
            all_entities.append(entity)
        
        # TODO: Implement sophisticated temporal correlation
        # For now, just combine all entities
        
        return all_entities
    
    def apply_video_redaction(self, video_path: Path, entities: List[PIIEntity], 
                            output_path: Path) -> List[PIIEntity]:
        """Apply redaction to video file."""
        redacted_entities = []
        
        try:
            # Load video
            video_clip = VideoFileClip(str(video_path))
            
            # Create frame processing function
            def process_frame(get_frame, t):
                frame = get_frame(t)
                
                # Find entities for this timestamp
                current_entities = []
                for entity in entities:
                    if entity.metadata.get("content_type") == "video_frame":
                        frame_time = entity.metadata.get("frame_time", 0)
                        # Allow some tolerance for frame timing
                        if abs(frame_time - t) < 0.1:  # 100ms tolerance
                            if entity.confidence >= self.config.min_confidence:
                                current_entities.append(entity)
                
                # Apply redaction to current entities
                if current_entities:
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    redacted_frame, _ = self.image_redactor.apply_redaction(frame_bgr, current_entities)
                    frame = cv2.cvtColor(redacted_frame, cv2.COLOR_BGR2RGB)
                    redacted_entities.extend(current_entities)
                
                return frame
            
            # Apply frame processing
            redacted_video = video_clip.fl(process_frame)
            
            # Process audio if available
            if video_clip.audio is not None:
                audio_entities = [e for e in entities if e.metadata.get("content_type") == "video_audio"]
                audio_entities = [e for e in audio_entities if e.confidence >= self.config.min_confidence]
                
                if audio_entities:
                    # Extract audio
                    audio = video_clip.audio.to_soundarray()
                    sr = video_clip.audio.fps
                    
                    # Convert stereo to mono if needed
                    if len(audio.shape) > 1:
                        audio = np.mean(audio, axis=1)
                    
                    # Apply audio redaction
                    redacted_audio, redacted_audio_entities = self.audio_redactor.apply_redaction(
                        audio, audio_entities, sr
                    )
                    redacted_entities.extend(redacted_audio_entities)
                    
                    # Create new audio clip
                    # Note: This is simplified - production would need proper audio handling
                    from moviepy.audio.io.AudioFileClip import AudioArrayClip
                    redacted_audio_clip = AudioArrayClip(redacted_audio.reshape(-1, 1), fps=sr)
                    redacted_video = redacted_video.set_audio(redacted_audio_clip)
            
            # Write output video
            redacted_video.write_videofile(
                str(output_path),
                codec='libx264',
                audio_codec='aac'
            )
            
            video_clip.close()
            redacted_video.close()
            
        except Exception as e:
            self.logger.error(f"Video redaction failed: {e}")
        
        return redacted_entities
    
    def redact(self, video_path: Path, output_path: Optional[Path] = None) -> RedactionResult:
        """
        Main redaction method for video content.
        
        Args:
            video_path: Path to video file
            output_path: Optional path to save redacted video
            
        Returns:
            RedactionResult with processing details
        """
        start_time = time.time()
        
        # Generate content hash for integrity
        with open(video_path, 'rb') as f:
            content_hash = hashlib.sha256(f.read()).hexdigest()
        
        # Create temporary directory for processing
        temp_dir = self.temp_dir / f"video_{int(time.time())}"
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Extract and process frames
            frame_paths = self.extract_frames(video_path, temp_dir / "frames")
            frame_entities = self.process_video_frames(frame_paths)
            
            # Process audio track
            audio, audio_entities, sr = self.process_audio_track(video_path)
            
            # Merge temporal entities
            all_entities = self.merge_temporal_entities(frame_entities, audio_entities)
            
            # Apply redaction if output path provided
            redacted_entities = []
            if output_path:
                redacted_entities = self.apply_video_redaction(video_path, all_entities, output_path)
            else:
                # Just filter by confidence for reporting
                redacted_entities = [e for e in all_entities if e.confidence >= self.config.min_confidence]
            
            # Calculate statistics
            processing_time = time.time() - start_time
            confidence_scores = self._calculate_confidence_scores(all_entities)
            
            # Get video metadata
            video_clip = VideoFileClip(str(video_path))
            duration = video_clip.duration
            fps = video_clip.fps
            resolution = (video_clip.w, video_clip.h)
            video_clip.close()
            
            # Create result
            result = RedactionResult(
                original_content_hash=content_hash,
                content_type="video",
                entities_detected=all_entities,
                entities_redacted=redacted_entities,
                total_entities=len(all_entities),
                redacted_count=len(redacted_entities),
                confidence_scores=confidence_scores,
                processing_time=processing_time,
                config_used=self.config,
                redacted_content_path=str(output_path) if output_path else None,
                redaction_quality={
                    "duration_seconds": duration,
                    "fps": fps,
                    "resolution": resolution,
                    "frames_processed": len(frame_paths),
                    "audio_processed": audio is not None
                }
            )
            
            return result
            
        finally:
            # Clean up temporary files
            try:
                import shutil
                shutil.rmtree(temp_dir, ignore_errors=True)
            except Exception as e:
                self.logger.warning(f"Failed to clean up temp directory: {e}")
    
    def _calculate_confidence_scores(self, entities: List[PIIEntity]) -> Dict[str, float]:
        """Calculate average confidence scores by entity type."""
        scores = {}
        type_counts = {}
        
        for entity in entities:
            entity_type = entity.entity_type.value
            if entity_type not in scores:
                scores[entity_type] = 0.0
                type_counts[entity_type] = 0
            
            scores[entity_type] += entity.confidence
            type_counts[entity_type] += 1
        
        # Calculate averages
        for entity_type in scores:
            if type_counts[entity_type] > 0:
                scores[entity_type] /= type_counts[entity_type]
        
        return scores