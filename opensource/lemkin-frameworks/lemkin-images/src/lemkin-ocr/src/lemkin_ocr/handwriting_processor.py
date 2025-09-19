"""
Handwriting Recognition Processor for Lemkin OCR Suite

This module provides advanced handwriting recognition capabilities for mixed-content
documents. It can detect, isolate, and transcribe handwritten text using state-of-the-art
transformer models and traditional computer vision techniques.

Designed specifically for legal document processing where handwritten annotations,
signatures, and notes are common.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any, Union
from PIL import Image
from datetime import datetime
import logging
from dataclasses import dataclass
import json
import base64
import io

# Deep learning imports
try:
    import torch
    import torch.nn.functional as F
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logging.warning("Transformers not available - handwriting recognition will be limited")

try:
    from torchvision import transforms
    TORCHVISION_AVAILABLE = True
except ImportError:
    TORCHVISION_AVAILABLE = False
    logging.warning("Torchvision not available - some preprocessing features disabled")

# Traditional OCR fallback
try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False

from .core import (
    OCRConfig, HandwritingResult, HandwrittenRegion, BoundingBox,
    ProcessingStatus
)

logger = logging.getLogger(__name__)


@dataclass
class HandwritingSegment:
    """Represents a segment of handwritten text"""
    bbox: BoundingBox
    image_data: np.ndarray
    confidence: float
    stroke_characteristics: Dict[str, Any]


class HandwritingDetector:
    """Detects handwritten regions in documents"""
    
    def __init__(self, config: OCRConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.HandwritingDetector")
    
    def detect_handwriting_regions(self, image: np.ndarray) -> List[HandwritingSegment]:
        """
        Detect regions containing handwritten text
        
        Args:
            image: Input image
            
        Returns:
            List of handwriting segments
        """
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Use multiple detection methods
        stroke_based_regions = self._detect_by_stroke_analysis(gray)
        texture_based_regions = self._detect_by_texture_analysis(gray)
        contour_based_regions = self._detect_by_contour_analysis(gray)
        
        # Combine and filter regions
        all_regions = stroke_based_regions + texture_based_regions + contour_based_regions
        filtered_regions = self._filter_and_merge_regions(all_regions, gray)
        
        return filtered_regions
    
    def _detect_by_stroke_analysis(self, gray: np.ndarray) -> List[HandwritingSegment]:
        """Detect handwriting by analyzing stroke characteristics"""
        segments = []
        
        # Apply adaptive threshold
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                     cv2.THRESH_BINARY_INV, 11, 2)
        
        # Morphological operations to connect broken strokes
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        connected = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(connected, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            # Calculate stroke characteristics
            area = cv2.contourArea(contour)
            if area < 50:  # Too small
                continue
            
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h if h > 0 else 0
            
            # Handwriting typically has varied stroke width
            stroke_width_variation = self._calculate_stroke_width_variation(contour, binary)
            
            # Calculate curvature
            perimeter = cv2.arcLength(contour, True)
            circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
            
            # Handwriting characteristics: low circularity, varied stroke width, moderate aspect ratio
            handwriting_score = 0.0
            
            # Low circularity suggests irregular shapes (handwriting)
            if circularity < 0.3:
                handwriting_score += 0.3
            
            # Varied stroke width suggests handwriting
            if stroke_width_variation > 0.2:
                handwriting_score += 0.4
            
            # Reasonable aspect ratio for text
            if 0.2 < aspect_ratio < 5.0:
                handwriting_score += 0.3
            
            if handwriting_score > 0.6:  # Threshold for handwriting detection
                bbox = BoundingBox(x=x, y=y, width=w, height=h)
                roi = gray[y:y+h, x:x+w]
                
                stroke_chars = {
                    'stroke_width_variation': stroke_width_variation,
                    'circularity': circularity,
                    'aspect_ratio': aspect_ratio,
                    'area': area
                }
                
                segment = HandwritingSegment(
                    bbox=bbox,
                    image_data=roi,
                    confidence=handwriting_score,
                    stroke_characteristics=stroke_chars
                )
                segments.append(segment)
        
        return segments
    
    def _detect_by_texture_analysis(self, gray: np.ndarray) -> List[HandwritingSegment]:
        """Detect handwriting by analyzing texture patterns"""
        segments = []
        
        # Calculate local binary patterns or similar texture features
        # Handwriting has different texture than printed text
        
        # For now, use a simplified approach with edge density
        edges = cv2.Canny(gray, 50, 150)
        
        # Divide image into blocks and analyze edge density
        block_size = 64
        height, width = gray.shape
        
        for y in range(0, height - block_size, block_size // 2):
            for x in range(0, width - block_size, block_size // 2):
                block_edges = edges[y:y+block_size, x:x+block_size]
                edge_density = np.sum(block_edges > 0) / (block_size * block_size)
                
                # Handwriting typically has moderate edge density
                if 0.05 < edge_density < 0.3:
                    # Analyze the block further
                    block_gray = gray[y:y+block_size, x:x+block_size]
                    
                    # Check for irregular patterns
                    std_dev = np.std(block_gray)
                    if std_dev > 30:  # Sufficient variation
                        bbox = BoundingBox(x=x, y=y, width=block_size, height=block_size)
                        
                        stroke_chars = {
                            'edge_density': edge_density,
                            'intensity_variation': std_dev,
                            'detection_method': 'texture_analysis'
                        }
                        
                        segment = HandwritingSegment(
                            bbox=bbox,
                            image_data=block_gray,
                            confidence=0.7,  # Medium confidence for texture-based detection
                            stroke_characteristics=stroke_chars
                        )
                        segments.append(segment)
        
        return segments
    
    def _detect_by_contour_analysis(self, gray: np.ndarray) -> List[HandwritingSegment]:
        """Detect handwriting by analyzing contour properties"""
        segments = []
        
        # Apply threshold
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            # Filter by size
            area = cv2.contourArea(contour)
            if area < 100 or area > 50000:  # Size constraints
                continue
            
            # Calculate hull and convexity defects
            hull = cv2.convexHull(contour, returnPoints=False)
            if len(hull) > 3:
                defects = cv2.convexityDefects(contour, hull)
                
                if defects is not None:
                    # Analyze defects (indentations in the shape)
                    defect_count = len(defects)
                    
                    # Handwriting often has many convexity defects due to curves
                    if defect_count > 3:
                        x, y, w, h = cv2.boundingRect(contour)
                        bbox = BoundingBox(x=x, y=y, width=w, height=h)
                        roi = gray[y:y+h, x:x+w]
                        
                        stroke_chars = {
                            'convexity_defects': defect_count,
                            'area': area,
                            'detection_method': 'contour_analysis'
                        }
                        
                        segment = HandwritingSegment(
                            bbox=bbox,
                            image_data=roi,
                            confidence=0.6,
                            stroke_characteristics=stroke_chars
                        )
                        segments.append(segment)
        
        return segments
    
    def _calculate_stroke_width_variation(self, contour: np.ndarray, binary: np.ndarray) -> float:
        """Calculate variation in stroke width along the contour"""
        if len(contour) < 10:
            return 0.0
        
        # Sample points along the contour
        sample_points = contour[::max(1, len(contour) // 10)]
        
        stroke_widths = []
        for point in sample_points:
            x, y = point[0]
            
            # Find stroke width at this point by scanning perpendicular to the contour
            # This is a simplified approach
            for radius in range(1, 20):
                if (y - radius >= 0 and y + radius < binary.shape[0] and
                    x - radius >= 0 and x + radius < binary.shape[1]):
                    
                    # Check horizontal width
                    left = x
                    while left > 0 and binary[y, left] > 0:
                        left -= 1
                    
                    right = x
                    while right < binary.shape[1] - 1 and binary[y, right] > 0:
                        right += 1
                    
                    width = right - left
                    if width > 0:
                        stroke_widths.append(width)
                        break
        
        if len(stroke_widths) < 2:
            return 0.0
        
        # Calculate coefficient of variation
        mean_width = np.mean(stroke_widths)
        std_width = np.std(stroke_widths)
        
        return std_width / mean_width if mean_width > 0 else 0.0
    
    def _filter_and_merge_regions(self, segments: List[HandwritingSegment], 
                                 image: np.ndarray) -> List[HandwritingSegment]:
        """Filter out false positives and merge overlapping regions"""
        if not segments:
            return []
        
        # Remove duplicates and low-confidence segments
        filtered = []
        for segment in segments:
            if segment.confidence > 0.5:  # Confidence threshold
                # Check if this region is significantly different from existing ones
                is_duplicate = False
                for existing in filtered:
                    overlap_area = segment.bbox.intersection_area(existing.bbox)
                    overlap_ratio = overlap_area / min(segment.bbox.area, existing.bbox.area)
                    
                    if overlap_ratio > 0.7:  # High overlap
                        is_duplicate = True
                        # Keep the one with higher confidence
                        if segment.confidence > existing.confidence:
                            filtered.remove(existing)
                            filtered.append(segment)
                        break
                
                if not is_duplicate:
                    filtered.append(segment)
        
        # Merge nearby segments that likely belong to the same handwritten text
        merged = []
        used_indices = set()
        
        for i, segment in enumerate(filtered):
            if i in used_indices:
                continue
            
            # Find nearby segments to merge
            merge_candidates = [segment]
            used_indices.add(i)
            
            for j, other_segment in enumerate(filtered):
                if j in used_indices or j == i:
                    continue
                
                # Check if segments are close enough to merge
                distance = self._calculate_segment_distance(segment.bbox, other_segment.bbox)
                if distance < 50:  # Distance threshold in pixels
                    merge_candidates.append(other_segment)
                    used_indices.add(j)
            
            # Create merged segment
            if len(merge_candidates) > 1:
                merged_segment = self._merge_segments(merge_candidates, image)
            else:
                merged_segment = merge_candidates[0]
            
            merged.append(merged_segment)
        
        return merged
    
    def _calculate_segment_distance(self, bbox1: BoundingBox, bbox2: BoundingBox) -> float:
        """Calculate distance between two bounding boxes"""
        center1 = bbox1.center
        center2 = bbox2.center
        return np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
    
    def _merge_segments(self, segments: List[HandwritingSegment], 
                       image: np.ndarray) -> HandwritingSegment:
        """Merge multiple segments into one"""
        # Calculate bounding box that encompasses all segments
        min_x = min(seg.bbox.x for seg in segments)
        min_y = min(seg.bbox.y for seg in segments)
        max_x = max(seg.bbox.x + seg.bbox.width for seg in segments)
        max_y = max(seg.bbox.y + seg.bbox.height for seg in segments)
        
        merged_bbox = BoundingBox(
            x=min_x, y=min_y,
            width=max_x - min_x,
            height=max_y - min_y
        )
        
        # Extract merged image region
        merged_image = image[min_y:max_y, min_x:max_x]
        
        # Average confidence
        avg_confidence = sum(seg.confidence for seg in segments) / len(segments)
        
        # Combine stroke characteristics
        merged_chars = {
            'merged_from': len(segments),
            'individual_confidences': [seg.confidence for seg in segments]
        }
        
        return HandwritingSegment(
            bbox=merged_bbox,
            image_data=merged_image,
            confidence=avg_confidence,
            stroke_characteristics=merged_chars
        )


class HandwritingRecognizer:
    """Recognizes handwritten text using transformer models"""
    
    def __init__(self, config: OCRConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.HandwritingRecognizer")
        
        # Initialize models
        self._processor = None
        self._model = None
        self._device = None
        
        # Initialize models if transformers available
        if TRANSFORMERS_AVAILABLE:
            self._initialize_models()
    
    def _initialize_models(self):
        """Initialize the handwriting recognition models"""
        try:
            # Use TrOCR for handwriting recognition
            model_name = self.config.handwriting_model
            if model_name == "trocr":
                model_name = "microsoft/trocr-base-handwritten"
            
            self._processor = TrOCRProcessor.from_pretrained(model_name)
            self._model = VisionEncoderDecoderModel.from_pretrained(model_name)
            
            # Set device
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self._model.to(self._device)
            self._model.eval()
            
            self.logger.info(f"Handwriting recognition model loaded: {model_name}")
            
        except Exception as e:
            self.logger.error(f"Failed to load handwriting model: {str(e)}")
            self._processor = None
            self._model = None
    
    def recognize_handwriting(self, segments: List[HandwritingSegment]) -> List[HandwrittenRegion]:
        """
        Recognize text in handwriting segments
        
        Args:
            segments: List of handwriting segments
            
        Returns:
            List of recognized handwritten regions
        """
        recognized_regions = []
        
        for segment in segments:
            try:
                text, confidence = self._recognize_segment(segment)
                
                # Create handwritten region
                region = HandwrittenRegion(
                    text=text,
                    confidence=confidence,
                    bounding_box=segment.bbox,
                    writing_style=self._analyze_writing_style(segment),
                    legibility_score=self._calculate_legibility_score(segment),
                    recognition_model=self.config.handwriting_model,
                    requires_manual_review=(confidence < self.config.handwriting_confidence_threshold)
                )
                
                recognized_regions.append(region)
                
            except Exception as e:
                self.logger.error(f"Failed to recognize handwriting segment: {str(e)}")
                # Create region with empty text and low confidence
                region = HandwrittenRegion(
                    text="[Recognition Failed]",
                    confidence=0.0,
                    bounding_box=segment.bbox,
                    requires_manual_review=True
                )
                recognized_regions.append(region)
        
        return recognized_regions
    
    def _recognize_segment(self, segment: HandwritingSegment) -> Tuple[str, float]:
        """Recognize text in a single segment"""
        if self._processor is None or self._model is None:
            # Fallback to Tesseract if available
            return self._fallback_recognition(segment)
        
        try:
            # Preprocess image for the model
            pil_image = self._preprocess_for_model(segment.image_data)
            
            # Process with TrOCR
            pixel_values = self._processor(pil_image, return_tensors="pt").pixel_values
            pixel_values = pixel_values.to(self._device)
            
            # Generate text
            with torch.no_grad():
                generated_ids = self._model.generate(pixel_values)
                generated_text = self._processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            # Estimate confidence (TrOCR doesn't provide direct confidence scores)
            confidence = self._estimate_confidence(generated_text, segment)
            
            return generated_text.strip(), confidence
            
        except Exception as e:
            self.logger.error(f"TrOCR recognition failed: {str(e)}")
            return self._fallback_recognition(segment)
    
    def _preprocess_for_model(self, image: np.ndarray) -> Image.Image:
        """Preprocess image for the handwriting recognition model"""
        # Convert to PIL Image
        if len(image.shape) == 3:
            pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        else:
            pil_image = Image.fromarray(image).convert('RGB')
        
        # Resize to model's expected input size (usually 384x384 for TrOCR)
        target_size = (384, 384)
        pil_image = pil_image.resize(target_size, Image.Resampling.LANCZOS)
        
        return pil_image
    
    def _estimate_confidence(self, text: str, segment: HandwritingSegment) -> float:
        """Estimate confidence in recognition result"""
        # Basic heuristics for confidence estimation
        confidence = 0.5  # Base confidence
        
        # Text length factor
        if len(text) > 3:
            confidence += 0.2
        
        # Character diversity
        unique_chars = len(set(text.lower()))
        if unique_chars > 3:
            confidence += 0.1
        
        # Avoid obvious OCR errors
        if not any(char.isalnum() for char in text):
            confidence -= 0.3
        
        # Consider segment detection confidence
        confidence = (confidence + segment.confidence) / 2
        
        return max(0.0, min(1.0, confidence))
    
    def _fallback_recognition(self, segment: HandwritingSegment) -> Tuple[str, float]:
        """Fallback recognition using Tesseract"""
        if not TESSERACT_AVAILABLE:
            return "[Recognition Unavailable]", 0.0
        
        try:
            # Configure Tesseract for handwriting
            config = '--psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789 '
            
            # Get text and confidence
            data = pytesseract.image_to_data(segment.image_data, config=config, output_type=pytesseract.Output.DICT)
            
            # Extract text and confidence
            words = []
            confidences = []
            
            for i in range(len(data['text'])):
                if int(data['conf'][i]) > 0:
                    word = data['text'][i].strip()
                    if word:
                        words.append(word)
                        confidences.append(float(data['conf'][i]) / 100.0)
            
            text = ' '.join(words)
            confidence = sum(confidences) / len(confidences) if confidences else 0.0
            
            return text, confidence
            
        except Exception as e:
            self.logger.error(f"Tesseract fallback failed: {str(e)}")
            return "[Recognition Failed]", 0.0
    
    def _analyze_writing_style(self, segment: HandwritingSegment) -> str:
        """Analyze the writing style of the handwriting"""
        # Simple heuristics based on stroke characteristics
        stroke_chars = segment.stroke_characteristics
        
        # Check aspect ratio and stroke variation for style classification
        aspect_ratio = stroke_chars.get('aspect_ratio', 1.0)
        stroke_variation = stroke_chars.get('stroke_width_variation', 0.0)
        
        if stroke_variation > 0.3:
            return "cursive"
        elif aspect_ratio > 2.0:
            return "print"
        else:
            return "mixed"
    
    def _calculate_legibility_score(self, segment: HandwritingSegment) -> float:
        """Calculate legibility score based on image characteristics"""
        image = segment.image_data
        
        # Factors affecting legibility
        scores = []
        
        # Contrast
        if len(image.shape) == 2:  # Grayscale
            contrast = np.std(image)
            contrast_score = min(contrast / 50.0, 1.0)  # Normalize
            scores.append(contrast_score)
        
        # Sharpness (using Laplacian variance)
        laplacian_var = cv2.Laplacian(image, cv2.CV_64F).var()
        sharpness_score = min(laplacian_var / 500.0, 1.0)  # Normalize
        scores.append(sharpness_score)
        
        # Size (larger text is generally more legible)
        area = segment.bbox.area
        size_score = min(area / 10000.0, 1.0)  # Normalize
        scores.append(size_score)
        
        return sum(scores) / len(scores) if scores else 0.5


class HandwritingProcessor:
    """
    Main handwriting processor that orchestrates detection and recognition
    """
    
    def __init__(self, config: OCRConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.HandwritingProcessor")
        
        # Initialize components
        self.detector = HandwritingDetector(config)
        self.recognizer = HandwritingRecognizer(config)
    
    def process_handwriting(self, image_path: Path) -> HandwritingResult:
        """
        Process handwriting in a document image
        
        Args:
            image_path: Path to the image file
            
        Returns:
            HandwritingResult with recognized text
        """
        start_time = datetime.utcnow()
        
        if not image_path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        self.logger.info(f"Starting handwriting processing for: {image_path.name}")
        
        try:
            # Load image
            if image_path.suffix.lower() in ['.pdf']:
                image = self._load_pdf_page(image_path, 0)
            else:
                image = cv2.imread(str(image_path))
                if image is None:
                    raise ValueError(f"Could not load image: {image_path}")
            
            # Detect handwriting regions
            self.logger.info("Detecting handwriting regions...")
            handwriting_segments = self.detector.detect_handwriting_regions(image)
            
            self.logger.info(f"Found {len(handwriting_segments)} handwriting segments")
            
            # Recognize text in handwriting regions
            if handwriting_segments:
                self.logger.info("Recognizing handwritten text...")
                handwritten_regions = self.recognizer.recognize_handwriting(handwriting_segments)
            else:
                handwritten_regions = []
            
            # Compile results
            total_text = ' '.join(region.text for region in handwritten_regions if region.text)
            average_confidence = (sum(region.confidence for region in handwritten_regions) / 
                                len(handwritten_regions) if handwritten_regions else 0.0)
            
            # Calculate quality metrics
            legibility_score = (sum(region.legibility_score or 0.5 for region in handwritten_regions) / 
                              len(handwritten_regions) if handwritten_regions else 0.0)
            
            recognition_quality = average_confidence  # Simplified
            
            # Determine if manual review is needed
            requires_manual_review = (
                average_confidence < self.config.handwriting_confidence_threshold or
                any(region.requires_manual_review for region in handwritten_regions)
            )
            
            # Analyze writing styles
            writing_styles = list(set(region.writing_style for region in handwritten_regions 
                                    if region.writing_style))
            dominant_style = max(writing_styles, key=lambda style: 
                               sum(1 for region in handwritten_regions 
                                   if region.writing_style == style)) if writing_styles else None
            
            # Calculate processing duration
            end_time = datetime.utcnow()
            duration = (end_time - start_time).total_seconds()
            
            # Create result
            result = HandwritingResult(
                document_path=str(image_path),
                processing_duration_seconds=duration,
                recognition_model=self.config.handwriting_model,
                handwritten_regions=handwritten_regions,
                total_handwritten_text=total_text,
                total_regions=len(handwritten_regions),
                average_confidence=average_confidence,
                legibility_score=legibility_score,
                recognition_quality=recognition_quality,
                requires_manual_review=requires_manual_review,
                writing_styles_detected=writing_styles,
                dominant_writing_style=dominant_style
            )
            
            self.logger.info(f"Handwriting processing completed: {len(handwritten_regions)} regions, "
                           f"{average_confidence:.2%} average confidence")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Handwriting processing failed: {str(e)}")
            # Return failed result
            duration = (datetime.utcnow() - start_time).total_seconds()
            return HandwritingResult(
                document_path=str(image_path),
                processing_duration_seconds=duration,
                recognition_model=self.config.handwriting_model,
                handwritten_regions=[],
                total_handwritten_text="",
                total_regions=0,
                average_confidence=0.0,
                legibility_score=0.0,
                recognition_quality=0.0,
                requires_manual_review=True,
                writing_styles_detected=[],
                dominant_writing_style=None
            )
    
    def _load_pdf_page(self, pdf_path: Path, page_number: int) -> np.ndarray:
        """Load a specific page from a PDF as an image"""
        try:
            import pdf2image
            
            pages = pdf2image.convert_from_path(
                pdf_path, 
                first_page=page_number + 1,
                last_page=page_number + 1,
                dpi=300
            )
            
            if pages:
                pil_image = pages[0]
                return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            else:
                raise ValueError(f"Could not extract page {page_number} from PDF")
                
        except ImportError:
            raise ImportError("pdf2image not available. Install pdf2image to process PDFs.")
    
    def process_mixed_content_document(self, image_path: Path, 
                                     text_regions: List[BoundingBox]) -> HandwritingResult:
        """
        Process handwriting in a document that also contains printed text
        
        Args:
            image_path: Path to the image file
            text_regions: Known text regions to avoid
            
        Returns:
            HandwritingResult focusing on non-text areas
        """
        # Load image
        if image_path.suffix.lower() in ['.pdf']:
            image = self._load_pdf_page(image_path, 0)
        else:
            image = cv2.imread(str(image_path))
            if image is None:
                raise ValueError(f"Could not load image: {image_path}")
        
        # Create mask to exclude known text regions
        mask = np.ones(image.shape[:2], dtype=np.uint8) * 255
        
        for text_region in text_regions:
            cv2.rectangle(mask, 
                         (text_region.x, text_region.y),
                         (text_region.x + text_region.width, text_region.y + text_region.height),
                         0, -1)
        
        # Apply mask to image
        masked_image = cv2.bitwise_and(image, image, mask=mask)
        
        # Detect handwriting in masked image
        handwriting_segments = self.detector.detect_handwriting_regions(masked_image)
        
        # Filter segments that don't overlap with text regions
        filtered_segments = []
        for segment in handwriting_segments:
            overlaps_with_text = False
            for text_region in text_regions:
                if segment.bbox.overlaps_with(text_region):
                    overlap_area = segment.bbox.intersection_area(text_region)
                    overlap_ratio = overlap_area / segment.bbox.area
                    if overlap_ratio > 0.3:  # Significant overlap
                        overlaps_with_text = True
                        break
            
            if not overlaps_with_text:
                filtered_segments.append(segment)
        
        # Recognize handwriting in filtered segments
        if filtered_segments:
            handwritten_regions = self.recognizer.recognize_handwriting(filtered_segments)
        else:
            handwritten_regions = []
        
        # Create result similar to regular processing
        total_text = ' '.join(region.text for region in handwritten_regions if region.text)
        average_confidence = (sum(region.confidence for region in handwritten_regions) / 
                            len(handwritten_regions) if handwritten_regions else 0.0)
        
        return HandwritingResult(
            document_path=str(image_path),
            processing_duration_seconds=0.0,  # Would calculate properly
            recognition_model=self.config.handwriting_model,
            handwritten_regions=handwritten_regions,
            total_handwritten_text=total_text,
            total_regions=len(handwritten_regions),
            average_confidence=average_confidence,
            legibility_score=0.0,  # Would calculate properly
            recognition_quality=average_confidence,
            requires_manual_review=(average_confidence < self.config.handwriting_confidence_threshold),
            writing_styles_detected=[],
            dominant_writing_style=None
        )


# Convenience function for direct usage
def process_handwriting(image_path: Path, config: Optional[OCRConfig] = None) -> HandwritingResult:
    """
    Convenience function to process handwriting in a document
    
    Args:
        image_path: Path to image or PDF file
        config: Optional configuration settings
        
    Returns:
        HandwritingResult with recognized handwritten text
    """
    if config is None:
        config = OCRConfig()
    
    processor = HandwritingProcessor(config)
    return processor.process_handwriting(image_path)