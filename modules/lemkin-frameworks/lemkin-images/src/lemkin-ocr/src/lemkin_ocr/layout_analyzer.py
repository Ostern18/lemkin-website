"""
Document Layout Analyzer for Lemkin OCR Suite

This module provides advanced document layout analysis and structure extraction
capabilities. It identifies headers, paragraphs, tables, images, signatures,
and other document elements with their spatial relationships.

Designed specifically for legal document processing with high accuracy requirements.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any, Union
from PIL import Image
from datetime import datetime
import logging
from dataclasses import dataclass
from collections import defaultdict
import json

# Scientific computing imports
try:
    from sklearn.cluster import DBSCAN
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logging.warning("sklearn not available - some advanced features disabled")

try:
    import scipy.ndimage as ndimage
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    logging.warning("scipy not available - some advanced features disabled")

from .core import (
    OCRConfig, LayoutAnalysis, DocumentStructure, TextRegion, ImageRegion,
    TableRegion, TableCell, RegionType, BoundingBox, DocumentType
)

logger = logging.getLogger(__name__)


@dataclass
class LayoutElement:
    """Represents a detected layout element"""
    element_type: str
    bounding_box: BoundingBox
    confidence: float
    properties: Dict[str, Any]


class TextLineDetector:
    """Detects text lines and their characteristics"""
    
    def __init__(self, config: OCRConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.TextLineDetector")
    
    def detect_text_lines(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect text lines in the image
        
        Args:
            image: Input image
            
        Returns:
            List of text line information
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Apply threshold to create binary image
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Morphological operations to connect text characters
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1))
        morph = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        text_lines = []
        min_width = image.shape[1] * 0.02  # Minimum 2% of image width
        min_height = 5  # Minimum height in pixels
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            
            # Filter by size
            if w >= min_width and h >= min_height:
                # Calculate line characteristics
                aspect_ratio = w / h
                area = cv2.contourArea(contour)
                density = area / (w * h) if w * h > 0 else 0
                
                text_line = {
                    'bbox': BoundingBox(x=x, y=y, width=w, height=h),
                    'aspect_ratio': aspect_ratio,
                    'area': area,
                    'density': density,
                    'center_y': y + h // 2,
                    'baseline_y': y + h  # Approximation
                }
                
                text_lines.append(text_line)
        
        # Sort by vertical position
        text_lines.sort(key=lambda line: line['center_y'])
        
        return text_lines
    
    def group_lines_into_paragraphs(self, text_lines: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """
        Group text lines into paragraphs based on spacing and alignment
        
        Args:
            text_lines: List of text line information
            
        Returns:
            List of paragraph groups (each group is a list of lines)
        """
        if not text_lines:
            return []
        
        paragraphs = []
        current_paragraph = [text_lines[0]]
        
        for i in range(1, len(text_lines)):
            prev_line = text_lines[i-1]
            curr_line = text_lines[i]
            
            # Calculate vertical gap
            vertical_gap = curr_line['bbox'].y - (prev_line['bbox'].y + prev_line['bbox'].height)
            
            # Calculate average line height for context
            avg_line_height = (prev_line['bbox'].height + curr_line['bbox'].height) / 2
            
            # Check if lines should be in same paragraph
            # Criteria: vertical gap is not too large, similar x-alignment
            gap_threshold = avg_line_height * 1.5  # 1.5x average line height
            
            x_alignment_diff = abs(curr_line['bbox'].x - prev_line['bbox'].x)
            alignment_threshold = avg_line_height  # Allow some misalignment
            
            if vertical_gap <= gap_threshold and x_alignment_diff <= alignment_threshold:
                # Same paragraph
                current_paragraph.append(curr_line)
            else:
                # Start new paragraph
                paragraphs.append(current_paragraph)
                current_paragraph = [curr_line]
        
        # Add the last paragraph
        if current_paragraph:
            paragraphs.append(current_paragraph)
        
        return paragraphs


class TableDetector:
    """Detects tables and extracts their structure"""
    
    def __init__(self, config: OCRConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.TableDetector")
    
    def detect_tables(self, image: np.ndarray) -> List[TableRegion]:
        """
        Detect tables in the image
        
        Args:
            image: Input image
            
        Returns:
            List of detected table regions
        """
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Detect horizontal and vertical lines
        horizontal_lines = self._detect_horizontal_lines(gray)
        vertical_lines = self._detect_vertical_lines(gray)
        
        # Find table candidates based on line intersections
        table_candidates = self._find_table_candidates(horizontal_lines, vertical_lines, image.shape)
        
        # Analyze each candidate to extract table structure
        tables = []
        for candidate in table_candidates:
            table = self._analyze_table_structure(candidate, horizontal_lines, vertical_lines, gray)
            if table:
                tables.append(table)
        
        return tables
    
    def _detect_horizontal_lines(self, gray: np.ndarray) -> List[Dict[str, Any]]:
        """Detect horizontal lines that could be table borders"""
        # Create horizontal kernel
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (gray.shape[1]//30, 1))
        
        # Apply threshold
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Detect horizontal lines
        horizontal = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        lines = []
        min_length = gray.shape[1] * 0.1  # Minimum 10% of image width
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w >= min_length and h <= 10:  # Thin horizontal line
                lines.append({
                    'x1': x, 'y1': y + h//2, 'x2': x + w, 'y2': y + h//2,
                    'length': w, 'thickness': h
                })
        
        return lines
    
    def _detect_vertical_lines(self, gray: np.ndarray) -> List[Dict[str, Any]]:
        """Detect vertical lines that could be table borders"""
        # Create vertical kernel
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, gray.shape[0]//30))
        
        # Apply threshold
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Detect vertical lines
        vertical = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(vertical, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        lines = []
        min_length = gray.shape[0] * 0.05  # Minimum 5% of image height
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if h >= min_length and w <= 10:  # Thin vertical line
                lines.append({
                    'x1': x + w//2, 'y1': y, 'x2': x + w//2, 'y2': y + h,
                    'length': h, 'thickness': w
                })
        
        return lines
    
    def _find_table_candidates(self, h_lines: List[Dict[str, Any]], 
                              v_lines: List[Dict[str, Any]], 
                              image_shape: Tuple[int, int]) -> List[BoundingBox]:
        """Find potential table regions based on line intersections"""
        candidates = []
        
        if not h_lines or not v_lines:
            return candidates
        
        # Group nearby lines
        h_groups = self._group_parallel_lines(h_lines, is_horizontal=True)
        v_groups = self._group_parallel_lines(v_lines, is_horizontal=False)
        
        # Find intersections between line groups
        for h_group in h_groups:
            for v_group in v_groups:
                if len(h_group) >= 2 and len(v_group) >= 2:  # At least 2 lines each direction
                    # Calculate bounding box of potential table
                    min_x = min(line['x1'] for line in v_group)
                    max_x = max(line['x1'] for line in v_group)
                    min_y = min(line['y1'] for line in h_group)
                    max_y = max(line['y1'] for line in h_group)
                    
                    width = max_x - min_x
                    height = max_y - min_y
                    
                    # Filter by size
                    if width > 50 and height > 30:  # Minimum table size
                        bbox = BoundingBox(x=min_x, y=min_y, width=width, height=height)
                        candidates.append(bbox)
        
        return candidates
    
    def _group_parallel_lines(self, lines: List[Dict[str, Any]], 
                             is_horizontal: bool) -> List[List[Dict[str, Any]]]:
        """Group parallel lines that are close to each other"""
        if not lines:
            return []
        
        # Sort lines by position
        if is_horizontal:
            lines.sort(key=lambda line: line['y1'])
            position_key = 'y1'
        else:
            lines.sort(key=lambda line: line['x1'])
            position_key = 'x1'
        
        groups = []
        current_group = [lines[0]]
        
        for i in range(1, len(lines)):
            # Check distance between consecutive lines
            distance = abs(lines[i][position_key] - lines[i-1][position_key])
            
            # If lines are close, add to current group
            if distance < 200:  # Threshold for grouping
                current_group.append(lines[i])
            else:
                # Start new group
                groups.append(current_group)
                current_group = [lines[i]]
        
        if current_group:
            groups.append(current_group)
        
        return groups
    
    def _analyze_table_structure(self, bbox: BoundingBox, 
                                h_lines: List[Dict[str, Any]], 
                                v_lines: List[Dict[str, Any]], 
                                image: np.ndarray) -> Optional[TableRegion]:
        """Analyze the structure of a table candidate"""
        # Extract table region
        table_image = image[bbox.y:bbox.y+bbox.height, bbox.x:bbox.x+bbox.width]
        
        # Find lines within the table region
        table_h_lines = [line for line in h_lines 
                        if bbox.y <= line['y1'] <= bbox.y + bbox.height]
        table_v_lines = [line for line in v_lines 
                        if bbox.x <= line['x1'] <= bbox.x + bbox.width]
        
        if len(table_h_lines) < 2 or len(table_v_lines) < 2:
            return None
        
        # Sort lines
        table_h_lines.sort(key=lambda line: line['y1'])
        table_v_lines.sort(key=lambda line: line['x1'])
        
        # Calculate grid dimensions
        rows = len(table_h_lines) - 1
        cols = len(table_v_lines) - 1
        
        if rows < 1 or cols < 1:
            return None
        
        # Create table cells
        cells = []
        for row in range(rows):
            for col in range(cols):
                cell_x = table_v_lines[col]['x1'] - bbox.x
                cell_y = table_h_lines[row]['y1'] - bbox.y
                cell_width = table_v_lines[col+1]['x1'] - table_v_lines[col]['x1']
                cell_height = table_h_lines[row+1]['y1'] - table_h_lines[row]['y1']
                
                cell_bbox = BoundingBox(
                    x=cell_x + bbox.x, y=cell_y + bbox.y, 
                    width=cell_width, height=cell_height
                )
                
                # Extract text from cell (simplified - would integrate with OCR)
                cell_text = ""  # Would be filled by OCR
                
                cell = TableCell(
                    row=row,
                    col=col,
                    text=cell_text,
                    confidence=0.8,  # Would be calculated based on OCR
                    bounding_box=cell_bbox,
                    is_header=(row == 0)  # Simple heuristic
                )
                
                cells.append(cell)
        
        table = TableRegion(
            bounding_box=bbox,
            rows=rows,
            cols=cols,
            cells=cells,
            confidence=0.8,  # Would be calculated based on line detection confidence
            has_header=True  # Simple assumption
        )
        
        return table


class ImageRegionDetector:
    """Detects image regions, signatures, and other non-text elements"""
    
    def __init__(self, config: OCRConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.ImageRegionDetector")
    
    def detect_image_regions(self, image: np.ndarray) -> List[ImageRegion]:
        """
        Detect non-text regions like images, diagrams, signatures
        
        Args:
            image: Input image
            
        Returns:
            List of detected image regions
        """
        regions = []
        
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Detect potential image regions using various methods
        signature_regions = self._detect_signatures(gray)
        diagram_regions = self._detect_diagrams(gray)
        photo_regions = self._detect_photos(image)
        
        regions.extend(signature_regions)
        regions.extend(diagram_regions)
        regions.extend(photo_regions)
        
        # Remove overlapping regions
        regions = self._remove_overlapping_regions(regions)
        
        return regions
    
    def _detect_signatures(self, gray: np.ndarray) -> List[ImageRegion]:
        """Detect handwritten signatures"""
        signatures = []
        
        # Apply threshold
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = cv2.contourArea(contour)
            
            # Signature characteristics: moderate size, irregular shape
            if 1000 < area < 50000:  # Reasonable signature size
                aspect_ratio = w / h
                if 1.5 < aspect_ratio < 8:  # Signatures are typically wider than tall
                    # Calculate contour complexity
                    perimeter = cv2.arcLength(contour, True)
                    circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
                    
                    # Signatures typically have low circularity (irregular)
                    if circularity < 0.3:
                        bbox = BoundingBox(x=x, y=y, width=w, height=h)
                        
                        signature = ImageRegion(
                            region_type=RegionType.SIGNATURE,
                            bounding_box=bbox,
                            description="Potential handwritten signature",
                            image_type="signature",
                            contains_text=True
                        )
                        
                        signatures.append(signature)
        
        return signatures
    
    def _detect_diagrams(self, gray: np.ndarray) -> List[ImageRegion]:
        """Detect diagrams and technical drawings"""
        diagrams = []
        
        # Detect lines using Hough transform
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
        
        if lines is not None and len(lines) > 10:  # Many lines suggest a diagram
            # Find regions with high line density
            line_mask = np.zeros_like(gray)
            
            for line in lines:
                rho, theta = line[0]
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a * rho
                y0 = b * rho
                x1 = int(x0 + 1000 * (-b))
                y1 = int(y0 + 1000 * (a))
                x2 = int(x0 - 1000 * (-b))
                y2 = int(y0 - 1000 * (a))
                
                cv2.line(line_mask, (x1, y1), (x2, y2), 255, 2)
            
            # Find contours in line mask
            contours, _ = cv2.findContours(line_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                area = w * h
                
                if area > 10000:  # Minimum diagram size
                    bbox = BoundingBox(x=x, y=y, width=w, height=h)
                    
                    diagram = ImageRegion(
                        region_type=RegionType.IMAGE,
                        bounding_box=bbox,
                        description="Technical diagram or drawing",
                        image_type="diagram",
                        contains_text=False
                    )
                    
                    diagrams.append(diagram)
        
        return diagrams
    
    def _detect_photos(self, image: np.ndarray) -> List[ImageRegion]:
        """Detect photographic regions"""
        photos = []
        
        # Convert to grayscale for analysis
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Look for regions with high texture variation (characteristic of photos)
        # Calculate local standard deviation
        kernel = np.ones((9, 9), np.float32) / 81
        mean = cv2.filter2D(gray.astype(np.float32), -1, kernel)
        sqr_mean = cv2.filter2D((gray.astype(np.float32))**2, -1, kernel)
        variance = sqr_mean - mean**2
        std_dev = np.sqrt(variance)
        
        # Threshold for high texture regions
        texture_mask = (std_dev > np.mean(std_dev) + 2 * np.std(std_dev)).astype(np.uint8) * 255
        
        # Find contours
        contours, _ = cv2.findContours(texture_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h
            
            if area > 20000:  # Minimum photo size
                bbox = BoundingBox(x=x, y=y, width=w, height=h)
                
                photo = ImageRegion(
                    region_type=RegionType.IMAGE,
                    bounding_box=bbox,
                    description="Photographic content",
                    image_type="photo",
                    contains_text=False
                )
                
                photos.append(photo)
        
        return photos
    
    def _remove_overlapping_regions(self, regions: List[ImageRegion]) -> List[ImageRegion]:
        """Remove overlapping regions, keeping the one with higher confidence or larger area"""
        if len(regions) <= 1:
            return regions
        
        # Sort by area (descending)
        regions.sort(key=lambda r: r.bounding_box.area, reverse=True)
        
        filtered_regions = []
        
        for region in regions:
            # Check if this region overlaps significantly with any kept region
            overlaps = False
            for kept_region in filtered_regions:
                intersection_area = region.bounding_box.intersection_area(kept_region.bounding_box)
                overlap_ratio = intersection_area / min(region.bounding_box.area, kept_region.bounding_box.area)
                
                if overlap_ratio > 0.5:  # 50% overlap threshold
                    overlaps = True
                    break
            
            if not overlaps:
                filtered_regions.append(region)
        
        return filtered_regions


class ReadingOrderAnalyzer:
    """Determines the reading order of document elements"""
    
    def __init__(self, config: OCRConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.ReadingOrderAnalyzer")
    
    def determine_reading_order(self, regions: List[TextRegion]) -> List[int]:
        """
        Determine the logical reading order of text regions
        
        Args:
            regions: List of text regions
            
        Returns:
            List of region indices in reading order
        """
        if not regions:
            return []
        
        # Simple top-to-bottom, left-to-right ordering
        # Could be enhanced with column detection for complex layouts
        
        # Group regions by approximate vertical position (lines/paragraphs)
        lines = self._group_into_lines(regions)
        
        # Sort lines by vertical position
        lines.sort(key=lambda line: min(region.bounding_box.y for region in line))
        
        # Within each line, sort by horizontal position
        reading_order = []
        for line in lines:
            line.sort(key=lambda region: region.bounding_box.x)
            for region in line:
                # Find index of this region in original list
                for i, original_region in enumerate(regions):
                    if region.id == original_region.id:
                        reading_order.append(i)
                        break
        
        return reading_order
    
    def _group_into_lines(self, regions: List[TextRegion]) -> List[List[TextRegion]]:
        """Group regions into horizontal lines"""
        if not regions:
            return []
        
        # Sort by vertical position
        sorted_regions = sorted(regions, key=lambda r: r.bounding_box.center[1])
        
        lines = []
        current_line = [sorted_regions[0]]
        
        for i in range(1, len(sorted_regions)):
            prev_region = sorted_regions[i-1]
            curr_region = sorted_regions[i]
            
            # Check if regions are on the same horizontal line
            prev_center_y = prev_region.bounding_box.center[1]
            curr_center_y = curr_region.bounding_box.center[1]
            
            # Use average height as threshold for line grouping
            avg_height = (prev_region.bounding_box.height + curr_region.bounding_box.height) / 2
            
            if abs(curr_center_y - prev_center_y) <= avg_height * 0.5:
                # Same line
                current_line.append(curr_region)
            else:
                # New line
                lines.append(current_line)
                current_line = [curr_region]
        
        if current_line:
            lines.append(current_line)
        
        return lines


class DocumentTypeClassifier:
    """Classifies document type based on layout characteristics"""
    
    def __init__(self, config: OCRConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.DocumentTypeClassifier")
    
    def classify_document_type(self, structure: DocumentStructure) -> DocumentType:
        """
        Classify document type based on structure analysis
        
        Args:
            structure: Document structure
            
        Returns:
            Detected document type
        """
        # Analyze structure characteristics
        has_tables = len(structure.table_regions) > 0
        has_signatures = any(region.region_type == RegionType.SIGNATURE 
                           for region in structure.image_regions)
        has_forms = any(region.region_type == RegionType.FORM_FIELD 
                       for region in structure.text_regions)
        has_handwriting = len(structure.handwritten_regions) > 0
        
        text_region_count = len(structure.text_regions)
        avg_text_length = (sum(len(region.text) for region in structure.text_regions) / 
                          text_region_count if text_region_count > 0 else 0)
        
        # Simple classification rules (could be enhanced with ML)
        if has_signatures and has_forms:
            return DocumentType.CONTRACT
        elif has_signatures and text_region_count > 10:
            return DocumentType.LEGAL_DOCUMENT
        elif has_forms and not has_signatures:
            return DocumentType.FORM
        elif has_handwriting and text_region_count < 5:
            return DocumentType.HANDWRITTEN_NOTE
        elif has_tables and avg_text_length < 100:
            return DocumentType.INVOICE
        elif text_region_count > 20 and avg_text_length > 200:
            return DocumentType.LEGAL_DOCUMENT
        elif has_handwriting and text_region_count > 5:
            return DocumentType.MIXED_CONTENT
        else:
            return DocumentType.UNKNOWN


class LayoutAnalyzer:
    """
    Main layout analyzer that orchestrates document structure extraction
    """
    
    def __init__(self, config: OCRConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.LayoutAnalyzer")
        
        # Initialize components
        self.text_line_detector = TextLineDetector(config)
        self.table_detector = TableDetector(config)
        self.image_detector = ImageRegionDetector(config)
        self.reading_order_analyzer = ReadingOrderAnalyzer(config)
        self.document_classifier = DocumentTypeClassifier(config)
    
    def analyze_layout(self, image_path: Path) -> LayoutAnalysis:
        """
        Perform comprehensive layout analysis on a document image
        
        Args:
            image_path: Path to the image file
            
        Returns:
            LayoutAnalysis with document structure
        """
        start_time = datetime.utcnow()
        
        if not image_path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        self.logger.info(f"Starting layout analysis for: {image_path.name}")
        
        try:
            # Load image
            if image_path.suffix.lower() in ['.pdf']:
                image = self._load_pdf_page(image_path, 0)
            else:
                image = cv2.imread(str(image_path))
                if image is None:
                    raise ValueError(f"Could not load image: {image_path}")
            
            # Initialize document structure
            structure = DocumentStructure()
            
            # Detect text lines and group into regions
            text_lines = self.text_line_detector.detect_text_lines(image)
            paragraphs = self.text_line_detector.group_lines_into_paragraphs(text_lines)
            
            # Convert paragraphs to text regions
            text_regions = []
            for i, paragraph_lines in enumerate(paragraphs):
                if paragraph_lines:
                    # Calculate bounding box for entire paragraph
                    min_x = min(line['bbox'].x for line in paragraph_lines)
                    min_y = min(line['bbox'].y for line in paragraph_lines)
                    max_x = max(line['bbox'].x + line['bbox'].width for line in paragraph_lines)
                    max_y = max(line['bbox'].y + line['bbox'].height for line in paragraph_lines)
                    
                    paragraph_bbox = BoundingBox(
                        x=min_x, y=min_y, 
                        width=max_x - min_x, 
                        height=max_y - min_y
                    )
                    
                    # Determine region type based on position and characteristics
                    region_type = self._classify_text_region(paragraph_bbox, image.shape, i, len(paragraphs))
                    
                    text_region = TextRegion(
                        region_type=region_type,
                        text="",  # Will be filled by OCR
                        confidence=0.8,  # Placeholder
                        bounding_box=paragraph_bbox,
                        reading_order=i,
                        paragraph_index=i
                    )
                    
                    text_regions.append(text_region)
            
            structure.text_regions = text_regions
            
            # Detect tables
            if self.config.detect_tables:
                tables = self.table_detector.detect_tables(image)
                structure.table_regions = tables
            
            # Detect image regions
            image_regions = self.image_detector.detect_image_regions(image)
            structure.image_regions = image_regions
            
            # Determine reading order
            if text_regions:
                reading_order_indices = self.reading_order_analyzer.determine_reading_order(text_regions)
                structure.reading_order = [text_regions[i].id for i in reading_order_indices]
            
            # Classify document type
            structure.document_type = self.document_classifier.classify_document_type(structure)
            
            # Calculate quality metrics
            layout_confidence = self._calculate_layout_confidence(structure, image.shape)
            text_line_quality = self._calculate_text_line_quality(text_lines)
            region_separation_quality = self._calculate_region_separation_quality(structure)
            reading_order_confidence = self._calculate_reading_order_confidence(structure)
            
            # Detect multi-column layout
            is_multi_column, column_count = self._detect_columns(text_regions)
            has_complex_layout = self._has_complex_layout(structure)
            
            # Calculate processing duration
            end_time = datetime.utcnow()
            duration = (end_time - start_time).total_seconds()
            
            # Create layout analysis result
            analysis = LayoutAnalysis(
                document_path=str(image_path),
                analysis_duration_seconds=duration,
                structure=structure,
                total_text_regions=len(text_regions),
                total_image_regions=len(image_regions),
                total_table_regions=len(structure.table_regions),
                total_handwritten_regions=len(structure.handwritten_regions),
                layout_confidence=layout_confidence,
                text_line_quality=text_line_quality,
                region_separation_quality=region_separation_quality,
                is_multi_column=is_multi_column,
                column_count=column_count,
                has_complex_layout=has_complex_layout,
                reading_order_confidence=reading_order_confidence
            )
            
            self.logger.info(f"Layout analysis completed: {len(text_regions)} text regions, "
                           f"{len(structure.table_regions)} tables, {len(image_regions)} images")
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Layout analysis failed: {str(e)}")
            # Return minimal result
            duration = (datetime.utcnow() - start_time).total_seconds()
            return LayoutAnalysis(
                document_path=str(image_path),
                analysis_duration_seconds=duration,
                structure=DocumentStructure(),
                total_text_regions=0,
                total_image_regions=0,
                total_table_regions=0,
                total_handwritten_regions=0,
                layout_confidence=0.0,
                text_line_quality=0.0,
                region_separation_quality=0.0,
                reading_order_confidence=0.0
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
    
    def _classify_text_region(self, bbox: BoundingBox, image_shape: Tuple[int, int], 
                             index: int, total_regions: int) -> RegionType:
        """Classify the type of a text region based on position and characteristics"""
        image_height, image_width = image_shape[:2]
        
        # Header detection: top 15% of document
        if bbox.y < image_height * 0.15:
            return RegionType.HEADER
        
        # Footer detection: bottom 10% of document
        if bbox.y + bbox.height > image_height * 0.9:
            return RegionType.FOOTER
        
        # Title detection: first region, centered, larger
        if index == 0 and bbox.x > image_width * 0.2 and bbox.x + bbox.width < image_width * 0.8:
            return RegionType.TITLE
        
        # Default to paragraph
        return RegionType.PARAGRAPH
    
    def _calculate_layout_confidence(self, structure: DocumentStructure, 
                                   image_shape: Tuple[int, int]) -> float:
        """Calculate confidence in layout detection"""
        confidence_factors = []
        
        # Factor 1: Text region coverage
        total_text_area = sum(region.bounding_box.area for region in structure.text_regions)
        image_area = image_shape[0] * image_shape[1]
        text_coverage = min(total_text_area / image_area, 1.0) if image_area > 0 else 0
        confidence_factors.append(text_coverage)
        
        # Factor 2: Region size consistency
        if structure.text_regions:
            region_areas = [region.bounding_box.area for region in structure.text_regions]
            area_std = np.std(region_areas)
            area_mean = np.mean(region_areas)
            consistency = 1.0 - min(area_std / area_mean, 1.0) if area_mean > 0 else 0
            confidence_factors.append(consistency)
        
        # Factor 3: Presence of expected elements
        has_text = len(structure.text_regions) > 0
        element_score = 0.5 if has_text else 0.0
        if structure.table_regions:
            element_score += 0.25
        if structure.image_regions:
            element_score += 0.25
        confidence_factors.append(element_score)
        
        return sum(confidence_factors) / len(confidence_factors) if confidence_factors else 0.0
    
    def _calculate_text_line_quality(self, text_lines: List[Dict[str, Any]]) -> float:
        """Calculate quality of text line detection"""
        if not text_lines:
            return 0.0
        
        # Check line alignment and spacing consistency
        y_positions = [line['center_y'] for line in text_lines]
        
        if len(y_positions) < 2:
            return 0.8  # Single line
        
        # Calculate spacing between consecutive lines
        spacings = [y_positions[i+1] - y_positions[i] for i in range(len(y_positions)-1)]
        
        # Good quality if spacings are consistent
        spacing_std = np.std(spacings)
        spacing_mean = np.mean(spacings)
        
        consistency = 1.0 - min(spacing_std / spacing_mean, 1.0) if spacing_mean > 0 else 0
        
        return max(consistency, 0.3)  # Minimum quality score
    
    def _calculate_region_separation_quality(self, structure: DocumentStructure) -> float:
        """Calculate quality of region separation"""
        if len(structure.text_regions) < 2:
            return 1.0  # Perfect if only one region
        
        # Check for overlapping regions
        overlaps = 0
        total_pairs = 0
        
        for i in range(len(structure.text_regions)):
            for j in range(i+1, len(structure.text_regions)):
                total_pairs += 1
                region1 = structure.text_regions[i]
                region2 = structure.text_regions[j]
                
                if region1.bounding_box.overlaps_with(region2.bounding_box):
                    overlaps += 1
        
        # Quality is inverse of overlap ratio
        overlap_ratio = overlaps / total_pairs if total_pairs > 0 else 0
        return 1.0 - overlap_ratio
    
    def _calculate_reading_order_confidence(self, structure: DocumentStructure) -> float:
        """Calculate confidence in reading order determination"""
        if len(structure.text_regions) <= 1:
            return 1.0
        
        # Simple heuristic: confidence decreases with layout complexity
        if structure.document_type in [DocumentType.LEGAL_DOCUMENT, DocumentType.CONTRACT]:
            return 0.9  # High confidence for standard documents
        elif len(structure.table_regions) > 0:
            return 0.7  # Medium confidence with tables
        elif len(structure.image_regions) > 0:
            return 0.6  # Lower confidence with images
        else:
            return 0.8  # Default confidence
    
    def _detect_columns(self, text_regions: List[TextRegion]) -> Tuple[bool, Optional[int]]:
        """Detect if document has multi-column layout"""
        if len(text_regions) < 4:
            return False, None
        
        # Group regions by approximate x-position
        x_positions = [region.bounding_box.x for region in text_regions]
        
        if SKLEARN_AVAILABLE:
            # Use clustering to detect columns
            try:
                positions = np.array(x_positions).reshape(-1, 1)
                scaler = StandardScaler()
                scaled_positions = scaler.fit_transform(positions)
                
                # Try different numbers of clusters
                for n_clusters in range(2, 5):
                    clustering = DBSCAN(eps=0.5, min_samples=2).fit(scaled_positions)
                    n_found_clusters = len(set(clustering.labels_)) - (1 if -1 in clustering.labels_ else 0)
                    
                    if n_found_clusters >= 2:
                        return True, n_found_clusters
                
            except Exception:
                pass
        
        # Fallback: simple threshold-based detection
        unique_x = sorted(set(x_positions))
        if len(unique_x) >= 2:
            # Check if there are distinct column positions
            gaps = [unique_x[i+1] - unique_x[i] for i in range(len(unique_x)-1)]
            max_gap = max(gaps) if gaps else 0
            avg_gap = sum(gaps) / len(gaps) if gaps else 0
            
            # If there's a significant gap, likely multi-column
            if max_gap > avg_gap * 2:
                return True, 2  # Assume 2 columns
        
        return False, None
    
    def _has_complex_layout(self, structure: DocumentStructure) -> bool:
        """Determine if document has complex layout"""
        complexity_factors = 0
        
        # Multiple tables
        if len(structure.table_regions) > 1:
            complexity_factors += 1
        
        # Images mixed with text
        if len(structure.image_regions) > 0 and len(structure.text_regions) > 5:
            complexity_factors += 1
        
        # Many text regions (suggesting complex structure)
        if len(structure.text_regions) > 20:
            complexity_factors += 1
        
        # Mixed content types
        if (len(structure.handwritten_regions) > 0 and 
            len(structure.text_regions) > 0 and 
            len(structure.image_regions) > 0):
            complexity_factors += 1
        
        return complexity_factors >= 2


# Convenience function for direct usage
def analyze_document_layout(image_path: Path, config: Optional[OCRConfig] = None) -> LayoutAnalysis:
    """
    Convenience function to analyze document layout
    
    Args:
        image_path: Path to image or PDF file
        config: Optional configuration settings
        
    Returns:
        LayoutAnalysis with document structure
    """
    if config is None:
        config = OCRConfig()
    
    analyzer = LayoutAnalyzer(config)
    return analyzer.analyze_layout(image_path)