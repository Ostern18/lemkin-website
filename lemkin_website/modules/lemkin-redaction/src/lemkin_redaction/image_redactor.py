"""
Image-based PII redaction for face detection and license plate blurring.

This module provides automated detection and redaction of personally
identifiable information in images, including faces and license plates.
"""

import hashlib
import time
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from loguru import logger

from .core import (
    PIIEntity, EntityType, ConfidenceLevel, RedactionConfig,
    RedactionResult, RedactionType
)


class ImageRedactor:
    """Image-based PII redaction using computer vision."""
    
    def __init__(self, config: RedactionConfig):
        """Initialize image redactor with configuration."""
        self.config = config
        self.logger = logger
        
        # Initialize CV models
        self._face_cascade = None
        self._plate_cascade = None
        
        # Redaction parameters
        self.blur_strength = 25
        self.face_scale_factor = 1.1
        self.face_min_neighbors = 5
        self.plate_scale_factor = 1.1
        self.plate_min_neighbors = 3
        
        self.logger.info("ImageRedactor initialized")
    
    @property
    def face_cascade(self):
        """Lazy loading of face detection cascade."""
        if self._face_cascade is None:
            try:
                # Load Haar cascade for face detection
                cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
                self._face_cascade = cv2.CascadeClassifier(cascade_path)
                self.logger.info("Loaded face detection cascade")
            except Exception as e:
                self.logger.error(f"Failed to load face cascade: {e}")
                
        return self._face_cascade
    
    @property
    def plate_cascade(self):
        """Lazy loading of license plate detection cascade."""
        if self._plate_cascade is None:
            try:
                # Load Russian license plate cascade (works well for general plates)
                cascade_path = cv2.data.haarcascades + 'haarcascade_russian_plate_number.xml'
                self._plate_cascade = cv2.CascadeClassifier(cascade_path)
                self.logger.info("Loaded license plate detection cascade")
            except Exception as e:
                self.logger.warning(f"License plate cascade not available: {e}")
                
        return self._plate_cascade
    
    def detect_faces(self, image: np.ndarray) -> List[PIIEntity]:
        """Detect faces in image using Haar cascades."""
        entities = []
        
        if not self.face_cascade or EntityType.PERSON not in self.config.entity_types:
            return entities
        
        try:
            # Convert to grayscale for detection
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
            
            # Detect faces
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=self.face_scale_factor,
                minNeighbors=self.face_min_neighbors,
                minSize=(30, 30)
            )
            
            for i, (x, y, w, h) in enumerate(faces):
                # Calculate confidence based on detection quality
                confidence = min(0.8 + (w * h) / (image.shape[0] * image.shape[1]), 0.95)
                
                entity = PIIEntity(
                    entity_type=EntityType.PERSON,
                    text=f"face_{i}",
                    start_pos=int(x),
                    end_pos=int(x + w),
                    confidence=confidence,
                    confidence_level=self._get_confidence_level(confidence),
                    metadata={
                        "source": "opencv_face",
                        "bbox": [int(x), int(y), int(w), int(h)],
                        "area": int(w * h)
                    }
                )
                entities.append(entity)
                
        except Exception as e:
            self.logger.error(f"Face detection failed: {e}")
            
        return entities
    
    def detect_license_plates(self, image: np.ndarray) -> List[PIIEntity]:
        """Detect license plates in image."""
        entities = []
        
        if not self.plate_cascade or EntityType.CUSTOM not in self.config.entity_types:
            return entities
        
        try:
            # Convert to grayscale for detection
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
            
            # Detect license plates
            plates = self.plate_cascade.detectMultiScale(
                gray,
                scaleFactor=self.plate_scale_factor,
                minNeighbors=self.plate_min_neighbors,
                minSize=(50, 20)
            )
            
            for i, (x, y, w, h) in enumerate(plates):
                # License plates have specific aspect ratio
                aspect_ratio = w / h
                if 2.0 <= aspect_ratio <= 6.0:  # Typical plate ratios
                    confidence = 0.7 + min(0.2, aspect_ratio / 10)
                    
                    entity = PIIEntity(
                        entity_type=EntityType.CUSTOM,
                        text=f"license_plate_{i}",
                        start_pos=int(x),
                        end_pos=int(x + w),
                        confidence=confidence,
                        confidence_level=self._get_confidence_level(confidence),
                        metadata={
                            "source": "opencv_plate",
                            "bbox": [int(x), int(y), int(w), int(h)],
                            "aspect_ratio": aspect_ratio,
                            "area": int(w * h)
                        }
                    )
                    entities.append(entity)
                    
        except Exception as e:
            self.logger.warning(f"License plate detection failed: {e}")
            
        return entities
    
    def detect_text_regions(self, image: np.ndarray) -> List[PIIEntity]:
        """Detect text regions that might contain PII."""
        entities = []
        
        try:
            # Use EAST text detector or similar
            # For now, using simple contour-based text detection
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
            
            # Apply morphological operations to detect text regions
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (18, 18))
            connected = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
            
            # Find contours
            contours, _ = cv2.findContours(connected, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for i, contour in enumerate(contours):
                x, y, w, h = cv2.boundingRect(contour)
                area = cv2.contourArea(contour)
                
                # Filter by size and aspect ratio
                if area > 500 and 0.2 <= h/w <= 5.0:
                    confidence = 0.6  # Lower confidence for text regions
                    
                    entity = PIIEntity(
                        entity_type=EntityType.CUSTOM,
                        text=f"text_region_{i}",
                        start_pos=int(x),
                        end_pos=int(x + w),
                        confidence=confidence,
                        confidence_level=self._get_confidence_level(confidence),
                        metadata={
                            "source": "opencv_text",
                            "bbox": [int(x), int(y), int(w), int(h)],
                            "area": int(area)
                        }
                    )
                    entities.append(entity)
                    
        except Exception as e:
            self.logger.error(f"Text region detection failed: {e}")
            
        return entities
    
    def apply_redaction(self, image: np.ndarray, entities: List[PIIEntity]) -> Tuple[np.ndarray, List[PIIEntity]]:
        """Apply redaction to image based on detected entities."""
        redacted_image = image.copy()
        redacted_entities = []
        
        for entity in entities:
            # Check if entity meets confidence threshold
            if entity.confidence < self.config.min_confidence:
                continue
            
            # Get redaction method for this entity type
            redaction_method = self.config.redaction_methods.get(
                entity.entity_type, RedactionType.BLUR
            )
            
            # Get bounding box from metadata
            bbox = entity.metadata.get("bbox", [entity.start_pos, 0, 50, 50])
            x, y, w, h = bbox
            
            # Apply redaction method
            if redaction_method == RedactionType.BLUR:
                redacted_image = self._apply_blur(redacted_image, x, y, w, h)
                entity.replacement = f"blurred_{entity.entity_type.value}"
                
            elif redaction_method == RedactionType.MASK:
                redacted_image = self._apply_mask(redacted_image, x, y, w, h)
                entity.replacement = f"masked_{entity.entity_type.value}"
                
            elif redaction_method == RedactionType.DELETE:
                redacted_image = self._apply_delete(redacted_image, x, y, w, h)
                entity.replacement = f"deleted_{entity.entity_type.value}"
            
            redacted_entities.append(entity)
        
        return redacted_image, redacted_entities
    
    def redact(self, image_path: Path, output_path: Optional[Path] = None) -> RedactionResult:
        """
        Main redaction method for image content.
        
        Args:
            image_path: Path to image file
            output_path: Optional path to save redacted image
            
        Returns:
            RedactionResult with processing details
        """
        start_time = time.time()
        
        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Generate content hash for integrity
        with open(image_path, 'rb') as f:
            content_hash = hashlib.sha256(f.read()).hexdigest()
        
        # Detect entities using different methods
        face_entities = self.detect_faces(image)
        plate_entities = self.detect_license_plates(image)
        text_entities = self.detect_text_regions(image)
        
        # Merge entities
        all_entities = face_entities + plate_entities + text_entities
        
        # Remove overlapping entities
        all_entities = self._remove_overlapping_entities(all_entities)
        
        # Apply redaction
        redacted_image, redacted_entities = self.apply_redaction(image, all_entities)
        
        # Save redacted image if output path provided
        if output_path:
            cv2.imwrite(str(output_path), redacted_image)
        
        # Calculate statistics
        processing_time = time.time() - start_time
        confidence_scores = self._calculate_confidence_scores(all_entities)
        
        # Create result
        result = RedactionResult(
            original_content_hash=content_hash,
            content_type="image",
            entities_detected=all_entities,
            entities_redacted=redacted_entities,
            total_entities=len(all_entities),
            redacted_count=len(redacted_entities),
            confidence_scores=confidence_scores,
            processing_time=processing_time,
            config_used=self.config,
            redacted_content_path=str(output_path) if output_path else None,
            redaction_quality={
                "image_dimensions": image.shape,
                "total_pixels": image.shape[0] * image.shape[1],
                "redacted_area": sum([
                    e.metadata.get("area", 0) for e in redacted_entities
                ])
            }
        )
        
        return result
    
    def _apply_blur(self, image: np.ndarray, x: int, y: int, w: int, h: int) -> np.ndarray:
        """Apply blur to specific region of image."""
        # Extract region
        region = image[y:y+h, x:x+w]
        
        # Apply Gaussian blur
        blurred_region = cv2.GaussianBlur(region, (self.blur_strength, self.blur_strength), 0)
        
        # Replace in original image
        result = image.copy()
        result[y:y+h, x:x+w] = blurred_region
        
        return result
    
    def _apply_mask(self, image: np.ndarray, x: int, y: int, w: int, h: int) -> np.ndarray:
        """Apply solid mask to specific region of image."""
        result = image.copy()
        cv2.rectangle(result, (x, y), (x+w, y+h), (0, 0, 0), -1)
        return result
    
    def _apply_delete(self, image: np.ndarray, x: int, y: int, w: int, h: int) -> np.ndarray:
        """Delete/replace specific region with background color."""
        result = image.copy()
        
        # Use surrounding pixels to estimate background color
        if x > 0 and y > 0:
            bg_color = image[y-1, x-1]
        else:
            bg_color = np.mean(image, axis=(0, 1))
        
        cv2.rectangle(result, (x, y), (x+w, y+h), bg_color.tolist(), -1)
        return result
    
    def _remove_overlapping_entities(self, entities: List[PIIEntity]) -> List[PIIEntity]:
        """Remove overlapping entities, keeping highest confidence."""
        if not entities:
            return entities
        
        # Sort by confidence (descending)
        entities.sort(key=lambda x: x.confidence, reverse=True)
        
        result = []
        for entity in entities:
            bbox1 = entity.metadata.get("bbox", [0, 0, 0, 0])
            
            # Check for overlap with existing entities
            overlaps = False
            for existing in result:
                bbox2 = existing.metadata.get("bbox", [0, 0, 0, 0])
                if self._bboxes_overlap(bbox1, bbox2):
                    overlaps = True
                    break
            
            if not overlaps:
                result.append(entity)
        
        return result
    
    def _bboxes_overlap(self, bbox1: List[int], bbox2: List[int]) -> bool:
        """Check if two bounding boxes overlap."""
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2
        
        return not (x1 + w1 <= x2 or x2 + w2 <= x1 or y1 + h1 <= y2 or y2 + h2 <= y1)
    
    def _get_confidence_level(self, confidence: float) -> ConfidenceLevel:
        """Convert numeric confidence to confidence level."""
        if confidence >= 0.9:
            return ConfidenceLevel.VERY_HIGH
        elif confidence >= 0.75:
            return ConfidenceLevel.HIGH
        elif confidence >= 0.5:
            return ConfidenceLevel.MEDIUM
        else:
            return ConfidenceLevel.LOW
    
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