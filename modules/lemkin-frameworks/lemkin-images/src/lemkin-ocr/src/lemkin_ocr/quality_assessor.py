"""
OCR Quality Assessment Module for Lemkin OCR Suite

This module provides comprehensive quality assessment for OCR results, including
accuracy evaluation, confidence scoring, image quality analysis, and improvement
recommendations. Designed specifically for legal document processing where
quality and reliability are paramount.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any, Union
from datetime import datetime
import logging
import re
import string
from collections import Counter
import statistics

# Text processing imports
try:
    import nltk
    from nltk.corpus import words
    from nltk.metrics import edit_distance
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    logging.warning("NLTK not available - some quality metrics will be limited")

try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    logging.warning("spaCy not available - advanced text analysis disabled")

# Statistical analysis
try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    logging.warning("SciPy not available - advanced statistical analysis disabled")

from .core import (
    OCRConfig, QualityAssessment, ProcessingResult, OCRResult,
    LayoutAnalysis, HandwritingResult
)

logger = logging.getLogger(__name__)


class ImageQualityAnalyzer:
    """Analyzes image quality factors that affect OCR performance"""
    
    def __init__(self, config: OCRConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.ImageQualityAnalyzer")
    
    def analyze_image_quality(self, image_path: Path) -> Dict[str, Any]:
        """
        Analyze various image quality factors
        
        Args:
            image_path: Path to the image
            
        Returns:
            Dictionary with quality metrics
        """
        try:
            # Load image
            if image_path.suffix.lower() in ['.pdf']:
                image = self._load_pdf_page(image_path, 0)
            else:
                image = cv2.imread(str(image_path))
                if image is None:
                    raise ValueError(f"Could not load image: {image_path}")
            
            # Convert to grayscale for analysis
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            quality_metrics = {}
            
            # Resolution analysis
            quality_metrics.update(self._analyze_resolution(gray))
            
            # Contrast analysis
            quality_metrics.update(self._analyze_contrast(gray))
            
            # Noise analysis
            quality_metrics.update(self._analyze_noise(gray))
            
            # Sharpness analysis
            quality_metrics.update(self._analyze_sharpness(gray))
            
            # Skew detection
            quality_metrics.update(self._analyze_skew(gray))
            
            # Lighting analysis
            quality_metrics.update(self._analyze_lighting(gray))
            
            return quality_metrics
            
        except Exception as e:
            self.logger.error(f"Image quality analysis failed: {str(e)}")
            return {
                'resolution_adequate': False,
                'contrast_adequate': False,
                'noise_level': 'high',
                'sharpness_score': 0.0,
                'skew_angle': 0.0,
                'lighting_uniformity': 0.0,
                'overall_image_quality': 0.0
            }
    
    def _analyze_resolution(self, image: np.ndarray) -> Dict[str, Any]:
        """Analyze image resolution"""
        height, width = image.shape
        total_pixels = height * width
        
        # Minimum resolution for good OCR (rule of thumb: ~300 DPI for 8.5x11 inch)
        min_pixels_for_good_ocr = 2550 * 3300  # Approximately 300 DPI for letter size
        
        resolution_adequate = total_pixels >= min_pixels_for_good_ocr * 0.5  # 50% threshold
        
        return {
            'image_width': width,
            'image_height': height,
            'total_pixels': total_pixels,
            'resolution_adequate': resolution_adequate,
            'estimated_dpi': self._estimate_dpi(width, height)
        }
    
    def _analyze_contrast(self, image: np.ndarray) -> Dict[str, Any]:
        """Analyze image contrast"""
        # Calculate standard deviation as contrast measure
        contrast_std = np.std(image)
        
        # Calculate dynamic range
        min_val = np.min(image)
        max_val = np.max(image)
        dynamic_range = max_val - min_val
        
        # Calculate histogram spread
        hist = cv2.calcHist([image], [0], None, [256], [0, 256])
        hist_std = np.std(hist)
        
        # Good contrast typically has std > 40 and dynamic range > 150
        contrast_adequate = contrast_std > 40 and dynamic_range > 150
        
        return {
            'contrast_std': float(contrast_std),
            'dynamic_range': int(dynamic_range),
            'histogram_spread': float(hist_std),
            'contrast_adequate': contrast_adequate,
            'contrast_score': min(contrast_std / 80.0, 1.0)  # Normalize to 0-1
        }
    
    def _analyze_noise(self, image: np.ndarray) -> Dict[str, Any]:
        """Analyze image noise levels"""
        # Estimate noise using Laplacian variance
        laplacian_var = cv2.Laplacian(image, cv2.CV_64F).var()
        
        # Use bilateral filter to reduce noise and compare
        filtered = cv2.bilateralFilter(image, 9, 75, 75)
        noise_estimate = np.mean(np.abs(image.astype(float) - filtered.astype(float)))
        
        # Categorize noise level
        if noise_estimate < 5:
            noise_level = "low"
        elif noise_estimate < 15:
            noise_level = "medium"
        else:
            noise_level = "high"
        
        return {
            'laplacian_variance': float(laplacian_var),
            'noise_estimate': float(noise_estimate),
            'noise_level': noise_level,
            'noise_score': max(0.0, 1.0 - noise_estimate / 20.0)  # Normalize
        }
    
    def _analyze_sharpness(self, image: np.ndarray) -> Dict[str, Any]:
        """Analyze image sharpness"""
        # Sobel edge detection
        sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        sobel_magnitude = np.sqrt(sobelx**2 + sobely**2)
        sharpness_sobel = np.mean(sobel_magnitude)
        
        # Laplacian sharpness
        laplacian = cv2.Laplacian(image, cv2.CV_64F)
        sharpness_laplacian = laplacian.var()
        
        # Combined sharpness score
        sharpness_score = min(sharpness_laplacian / 1000.0, 1.0)
        
        return {
            'sobel_sharpness': float(sharpness_sobel),
            'laplacian_sharpness': float(sharpness_laplacian),
            'sharpness_score': float(sharpness_score),
            'sharpness_adequate': sharpness_score > 0.3
        }
    
    def _analyze_skew(self, image: np.ndarray) -> Dict[str, Any]:
        """Analyze image skew"""
        # Simple skew detection using Hough line transform
        edges = cv2.Canny(image, 50, 150, apertureSize=3)
        lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
        
        angles = []
        if lines is not None:
            for line in lines:
                rho, theta = line[0]
                angle = theta * 180 / np.pi
                # Convert to -90 to 90 range
                if angle > 90:
                    angle = angle - 180
                angles.append(angle)
        
        if angles:
            # Find dominant angle (mode)
            angle_hist, bins = np.histogram(angles, bins=180, range=(-90, 90))
            dominant_angle_idx = np.argmax(angle_hist)
            dominant_angle = bins[dominant_angle_idx]
            
            # Skew is deviation from 0 or 90 degrees
            skew_angle = min(abs(dominant_angle), abs(dominant_angle - 90), abs(dominant_angle + 90))
        else:
            skew_angle = 0.0
        
        skew_acceptable = skew_angle < 2.0  # Less than 2 degrees
        
        return {
            'detected_angles': angles[:10],  # First 10 for brevity
            'dominant_angle': float(dominant_angle) if angles else 0.0,
            'skew_angle': float(skew_angle),
            'skew_acceptable': skew_acceptable
        }
    
    def _analyze_lighting(self, image: np.ndarray) -> Dict[str, Any]:
        """Analyze lighting uniformity"""
        # Divide image into blocks and analyze brightness variation
        block_size = 64
        height, width = image.shape
        brightness_values = []
        
        for y in range(0, height - block_size, block_size):
            for x in range(0, width - block_size, block_size):
                block = image[y:y+block_size, x:x+block_size]
                brightness_values.append(np.mean(block))
        
        if brightness_values:
            brightness_std = np.std(brightness_values)
            brightness_mean = np.mean(brightness_values)
            
            # Uniformity score (lower std = more uniform)
            uniformity = 1.0 - min(brightness_std / brightness_mean, 1.0) if brightness_mean > 0 else 0.0
        else:
            uniformity = 0.0
        
        return {
            'brightness_std': float(brightness_std) if brightness_values else 0.0,
            'brightness_mean': float(brightness_mean) if brightness_values else 0.0,
            'lighting_uniformity': float(uniformity),
            'lighting_adequate': uniformity > 0.7
        }
    
    def _estimate_dpi(self, width: int, height: int) -> int:
        """Estimate DPI assuming standard document sizes"""
        # Assume letter size (8.5 x 11 inches) for estimation
        letter_width_inches = 8.5
        letter_height_inches = 11.0
        
        dpi_from_width = width / letter_width_inches
        dpi_from_height = height / letter_height_inches
        
        # Use the smaller value (more conservative estimate)
        estimated_dpi = int(min(dpi_from_width, dpi_from_height))
        
        return estimated_dpi
    
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


class TextQualityAnalyzer:
    """Analyzes the quality of extracted text"""
    
    def __init__(self, config: OCRConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.TextQualityAnalyzer")
        
        # Initialize language models if available
        self._nlp = None
        self._word_set = None
        self._initialize_language_resources()
    
    def _initialize_language_resources(self):
        """Initialize language processing resources"""
        if NLTK_AVAILABLE:
            try:
                nltk.data.find('corpora/words')
                self._word_set = set(words.words())
            except LookupError:
                self.logger.warning("NLTK words corpus not found. Some quality metrics will be limited.")
        
        if SPACY_AVAILABLE:
            try:
                self._nlp = spacy.load("en_core_web_sm")
            except OSError:
                self.logger.warning("spaCy English model not found. Advanced text analysis disabled.")
    
    def analyze_text_quality(self, ocr_result: OCRResult) -> Dict[str, Any]:
        """
        Analyze the quality of OCR-extracted text
        
        Args:
            ocr_result: OCR result to analyze
            
        Returns:
            Dictionary with text quality metrics
        """
        text = ocr_result.text
        word_confidences = ocr_result.word_confidences
        
        quality_metrics = {}
        
        # Basic text statistics
        quality_metrics.update(self._analyze_basic_statistics(text))
        
        # Language and vocabulary analysis
        quality_metrics.update(self._analyze_vocabulary(text))
        
        # Confidence analysis
        quality_metrics.update(self._analyze_confidence_patterns(word_confidences))
        
        # Error pattern detection
        quality_metrics.update(self._detect_error_patterns(text))
        
        # Completeness analysis
        quality_metrics.update(self._analyze_completeness(text))
        
        # Character accuracy estimation
        quality_metrics.update(self._estimate_character_accuracy(text, word_confidences))
        
        return quality_metrics
    
    def _analyze_basic_statistics(self, text: str) -> Dict[str, Any]:
        """Analyze basic text statistics"""
        if not text:
            return {
                'text_length': 0,
                'word_count': 0,
                'sentence_count': 0,
                'avg_word_length': 0.0,
                'character_distribution': {}
            }
        
        words = text.split()
        sentences = re.split(r'[.!?]+', text)
        
        # Character distribution
        char_counts = Counter(text.lower())
        total_chars = sum(char_counts.values())
        char_distribution = {char: count/total_chars for char, count in char_counts.most_common(10)}
        
        return {
            'text_length': len(text),
            'word_count': len(words),
            'sentence_count': len([s for s in sentences if s.strip()]),
            'avg_word_length': sum(len(word) for word in words) / len(words) if words else 0.0,
            'character_distribution': char_distribution
        }
    
    def _analyze_vocabulary(self, text: str) -> Dict[str, Any]:
        """Analyze vocabulary and language characteristics"""
        if not text or not self._word_set:
            return {
                'dictionary_word_ratio': 0.0,
                'unknown_words': [],
                'vocabulary_richness': 0.0
            }
        
        words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
        
        if not words:
            return {
                'dictionary_word_ratio': 0.0,
                'unknown_words': [],
                'vocabulary_richness': 0.0
            }
        
        # Check against dictionary
        dictionary_words = [word for word in words if word in self._word_set]
        unknown_words = [word for word in set(words) if word not in self._word_set]
        
        dictionary_ratio = len(dictionary_words) / len(words)
        
        # Vocabulary richness (unique words / total words)
        unique_words = len(set(words))
        vocabulary_richness = unique_words / len(words)
        
        return {
            'dictionary_word_ratio': dictionary_ratio,
            'unknown_words': unknown_words[:20],  # First 20 for brevity
            'vocabulary_richness': vocabulary_richness,
            'total_unique_words': unique_words
        }
    
    def _analyze_confidence_patterns(self, word_confidences: List[float]) -> Dict[str, Any]:
        """Analyze patterns in confidence scores"""
        if not word_confidences:
            return {
                'confidence_mean': 0.0,
                'confidence_std': 0.0,
                'low_confidence_ratio': 1.0,
                'confidence_consistency': 0.0
            }
        
        mean_conf = statistics.mean(word_confidences)
        std_conf = statistics.stdev(word_confidences) if len(word_confidences) > 1 else 0.0
        
        # Ratio of low-confidence words (below 0.7)
        low_conf_count = sum(1 for conf in word_confidences if conf < 0.7)
        low_conf_ratio = low_conf_count / len(word_confidences)
        
        # Confidence consistency (inverse of coefficient of variation)
        consistency = 1.0 - (std_conf / mean_conf) if mean_conf > 0 else 0.0
        consistency = max(0.0, consistency)
        
        return {
            'confidence_mean': mean_conf,
            'confidence_std': std_conf,
            'low_confidence_ratio': low_conf_ratio,
            'confidence_consistency': consistency,
            'confidence_distribution': self._analyze_confidence_distribution(word_confidences)
        }
    
    def _analyze_confidence_distribution(self, confidences: List[float]) -> Dict[str, int]:
        """Analyze distribution of confidence scores"""
        bins = {'very_low': 0, 'low': 0, 'medium': 0, 'high': 0, 'very_high': 0}
        
        for conf in confidences:
            if conf < 0.3:
                bins['very_low'] += 1
            elif conf < 0.5:
                bins['low'] += 1
            elif conf < 0.7:
                bins['medium'] += 1
            elif conf < 0.9:
                bins['high'] += 1
            else:
                bins['very_high'] += 1
        
        return bins
    
    def _detect_error_patterns(self, text: str) -> Dict[str, Any]:
        """Detect common OCR error patterns"""
        error_patterns = {
            'common_substitutions': 0,
            'character_insertion_errors': 0,
            'character_deletion_errors': 0,
            'case_errors': 0,
            'punctuation_errors': 0,
            'number_letter_confusion': 0
        }
        
        # Common OCR substitutions
        substitution_patterns = [
            (r'\brn\b', 'm'),  # 'rn' -> 'm'
            (r'\b1\b', 'l'),   # '1' -> 'l'
            (r'\b0\b', 'O'),   # '0' -> 'O'
            (r'\bvv\b', 'w'),  # 'vv' -> 'w'
            (r'\bc1\b', 'd'),  # 'cl' -> 'd'
        ]
        
        for pattern, replacement in substitution_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            error_patterns['common_substitutions'] += len(matches)
        
        # Number-letter confusion
        number_letter_patterns = [
            r'[0-9]+[a-zA-Z]+[0-9]+',  # Mixed number-letter sequences
            r'[a-zA-Z]+[0-9]+[a-zA-Z]+',
        ]
        
        for pattern in number_letter_patterns:
            matches = re.findall(pattern, text)
            error_patterns['number_letter_confusion'] += len(matches)
        
        # Excessive punctuation
        excessive_punct = re.findall(r'[.,;:!?]{3,}', text)
        error_patterns['punctuation_errors'] = len(excessive_punct)
        
        return error_patterns
    
    def _analyze_completeness(self, text: str) -> Dict[str, Any]:
        """Analyze text completeness"""
        # Check for incomplete words (fragments)
        words = text.split()
        
        # Very short words that might be fragments
        fragments = [word for word in words if len(word) == 1 and word.isalpha()]
        fragment_ratio = len(fragments) / len(words) if words else 0.0
        
        # Check for incomplete sentences
        sentences = re.split(r'[.!?]+', text)
        complete_sentences = [s for s in sentences if s.strip() and len(s.strip()) > 10]
        sentence_completeness = len(complete_sentences) / len(sentences) if sentences else 0.0
        
        # Check for truncated text (common OCR issue)
        lines = text.split('\n')
        truncated_lines = [line for line in lines if line.endswith('-') or len(line) < 5]
        truncation_ratio = len(truncated_lines) / len(lines) if lines else 0.0
        
        return {
            'fragment_ratio': fragment_ratio,
            'sentence_completeness': sentence_completeness,
            'truncation_ratio': truncation_ratio,
            'text_completeness_score': 1.0 - (fragment_ratio + truncation_ratio) / 2
        }
    
    def _estimate_character_accuracy(self, text: str, confidences: List[float]) -> Dict[str, Any]:
        """Estimate character-level accuracy"""
        if not text or not confidences:
            return {
                'estimated_character_accuracy': 0.0,
                'estimated_word_accuracy': 0.0
            }
        
        # Simple heuristic: use confidence as proxy for accuracy
        # This would be more accurate with ground truth data
        
        # Character accuracy estimate
        char_accuracy = statistics.mean(confidences) if confidences else 0.0
        
        # Word accuracy estimate (stricter)
        high_conf_words = sum(1 for conf in confidences if conf > 0.8)
        word_accuracy = high_conf_words / len(confidences) if confidences else 0.0
        
        return {
            'estimated_character_accuracy': char_accuracy,
            'estimated_word_accuracy': word_accuracy
        }


class OCREngineComparator:
    """Compares results from multiple OCR engines"""
    
    def __init__(self, config: OCRConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.OCREngineComparator")
    
    def compare_engine_results(self, engine_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Compare results from multiple OCR engines
        
        Args:
            engine_results: Dictionary mapping engine names to their results
            
        Returns:
            Comparison metrics
        """
        if len(engine_results) < 2:
            return {'engine_agreement_score': 1.0, 'consensus_confidence': 1.0}
        
        # Extract texts from different engines
        texts = {engine: result.get('text', '') for engine, result in engine_results.items()}
        confidences = {engine: result.get('confidence', 0.0) for engine, result in engine_results.items()}
        
        # Calculate text similarity between engines
        similarities = []
        engine_pairs = []
        
        engine_names = list(texts.keys())
        for i in range(len(engine_names)):
            for j in range(i + 1, len(engine_names)):
                engine1, engine2 = engine_names[i], engine_names[j]
                similarity = self._calculate_text_similarity(texts[engine1], texts[engine2])
                similarities.append(similarity)
                engine_pairs.append((engine1, engine2))
        
        # Overall agreement score
        agreement_score = statistics.mean(similarities) if similarities else 0.0
        
        # Confidence consensus
        conf_values = list(confidences.values())
        consensus_confidence = statistics.mean(conf_values) if conf_values else 0.0
        
        # Identify most reliable engine
        best_engine = max(confidences.keys(), key=lambda k: confidences[k]) if confidences else None
        
        return {
            'engine_agreement_score': agreement_score,
            'consensus_confidence': consensus_confidence,
            'individual_similarities': dict(zip(engine_pairs, similarities)),
            'best_performing_engine': best_engine,
            'confidence_spread': max(conf_values) - min(conf_values) if conf_values else 0.0
        }
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts"""
        if not text1 and not text2:
            return 1.0
        if not text1 or not text2:
            return 0.0
        
        # Normalize texts
        text1_norm = re.sub(r'\s+', ' ', text1.lower().strip())
        text2_norm = re.sub(r'\s+', ' ', text2.lower().strip())
        
        # Calculate edit distance
        if NLTK_AVAILABLE:
            distance = edit_distance(text1_norm, text2_norm)
            max_len = max(len(text1_norm), len(text2_norm))
            similarity = 1.0 - (distance / max_len) if max_len > 0 else 1.0
        else:
            # Simple character-based similarity
            common_chars = sum(1 for c1, c2 in zip(text1_norm, text2_norm) if c1 == c2)
            max_len = max(len(text1_norm), len(text2_norm))
            similarity = common_chars / max_len if max_len > 0 else 1.0
        
        return max(0.0, similarity)


class QualityAssessor:
    """
    Main quality assessor that orchestrates comprehensive quality evaluation
    """
    
    def __init__(self, config: OCRConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.QualityAssessor")
        
        # Initialize components
        self.image_analyzer = ImageQualityAnalyzer(config)
        self.text_analyzer = TextQualityAnalyzer(config)
        self.engine_comparator = OCREngineComparator(config)
    
    def assess_quality(self, processing_result: ProcessingResult) -> QualityAssessment:
        """
        Perform comprehensive quality assessment of OCR processing results
        
        Args:
            processing_result: Complete processing result to assess
            
        Returns:
            QualityAssessment with detailed quality metrics
        """
        start_time = datetime.utcnow()
        
        document_path = processing_result.document_path
        self.logger.info(f"Starting quality assessment for: {Path(document_path).name}")
        
        try:
            # Analyze image quality
            image_quality = self.image_analyzer.analyze_image_quality(Path(document_path))
            
            # Analyze text quality
            text_quality = {}
            if processing_result.ocr_result:
                text_quality = self.text_analyzer.analyze_text_quality(processing_result.ocr_result)
            
            # Compare engine results if available
            engine_comparison = {}
            if (processing_result.ocr_result and 
                processing_result.ocr_result.engine_results and 
                len(processing_result.ocr_result.engine_results) > 1):
                engine_comparison = self.engine_comparator.compare_engine_results(
                    processing_result.ocr_result.engine_results
                )
            
            # Calculate overall quality scores
            overall_scores = self._calculate_overall_scores(
                image_quality, text_quality, engine_comparison, processing_result
            )
            
            # Identify quality issues and generate suggestions
            quality_issues, suggestions = self._identify_issues_and_suggestions(
                image_quality, text_quality, processing_result
            )
            
            # Assess legal compliance
            legal_assessment = self._assess_legal_compliance(overall_scores, processing_result)
            
            # Calculate processing duration
            end_time = datetime.utcnow()
            duration = (end_time - start_time).total_seconds()
            
            # Create quality assessment
            assessment = QualityAssessment(
                document_path=document_path,
                assessment_duration_seconds=duration,
                
                # Overall scores
                overall_quality_score=overall_scores['overall_quality'],
                text_accuracy_score=overall_scores['text_accuracy'],
                layout_accuracy_score=overall_scores['layout_accuracy'],
                confidence_reliability_score=overall_scores['confidence_reliability'],
                
                # Detailed metrics from image analysis
                image_resolution_adequate=image_quality.get('resolution_adequate', False),
                image_contrast_adequate=image_quality.get('contrast_adequate', False),
                image_skew_acceptable=image_quality.get('skew_acceptable', True),
                image_noise_level=image_quality.get('noise_level', 'unknown'),
                
                # Content quality
                text_completeness=text_quality.get('text_completeness_score', 0.0),
                formatting_preservation=self._assess_formatting_preservation(processing_result),
                special_characters_accuracy=self._assess_special_characters(text_quality),
                
                # Quality issues and suggestions
                quality_issues=quality_issues,
                improvement_suggestions=suggestions,
                
                # Reliability indicators
                consistent_confidence_scores=(
                    text_quality.get('confidence_consistency', 0.0) > 0.7
                ),
                engine_agreement_score=engine_comparison.get('engine_agreement_score'),
                
                # Legal compliance
                meets_legal_standards=legal_assessment['meets_standards'],
                admissible_quality=legal_assessment['admissible'],
                requires_expert_validation=legal_assessment['needs_expert_review']
            )
            
            self.logger.info(f"Quality assessment completed: {overall_scores['overall_quality']:.2%} overall quality")
            
            return assessment
            
        except Exception as e:
            self.logger.error(f"Quality assessment failed: {str(e)}")
            # Return minimal assessment
            duration = (datetime.utcnow() - start_time).total_seconds()
            return QualityAssessment(
                document_path=document_path,
                assessment_duration_seconds=duration,
                overall_quality_score=0.0,
                text_accuracy_score=0.0,
                layout_accuracy_score=0.0,
                confidence_reliability_score=0.0,
                image_resolution_adequate=False,
                image_contrast_adequate=False,
                image_skew_acceptable=False,
                image_noise_level="unknown",
                text_completeness=0.0,
                formatting_preservation=0.0,
                special_characters_accuracy=0.0,
                quality_issues=["Quality assessment failed"],
                improvement_suggestions=["Retry with different settings"],
                consistent_confidence_scores=False,
                meets_legal_standards=False,
                admissible_quality=False,
                requires_expert_validation=True
            )
    
    def _calculate_overall_scores(self, image_quality: Dict[str, Any], 
                                 text_quality: Dict[str, Any],
                                 engine_comparison: Dict[str, Any],
                                 processing_result: ProcessingResult) -> Dict[str, float]:
        """Calculate overall quality scores"""
        scores = []
        
        # Image quality contribution (30%)
        image_score = 0.0
        if image_quality.get('resolution_adequate', False):
            image_score += 0.25
        if image_quality.get('contrast_adequate', False):
            image_score += 0.25
        if image_quality.get('skew_acceptable', True):
            image_score += 0.25
        if image_quality.get('noise_level') == 'low':
            image_score += 0.25
        
        # Text quality contribution (40%)
        text_score = 0.0
        if text_quality:
            text_score += text_quality.get('dictionary_word_ratio', 0.0) * 0.4
            text_score += text_quality.get('confidence_consistency', 0.0) * 0.3
            text_score += text_quality.get('text_completeness_score', 0.0) * 0.3
        
        # Layout quality contribution (20%)
        layout_score = 0.0
        if processing_result.layout_analysis:
            layout_score = processing_result.layout_analysis.layout_confidence
        
        # Engine agreement contribution (10%)
        agreement_score = engine_comparison.get('engine_agreement_score', 1.0)
        
        # Weighted overall score
        overall_quality = (image_score * 0.3 + text_score * 0.4 + 
                          layout_score * 0.2 + agreement_score * 0.1)
        
        return {
            'overall_quality': overall_quality,
            'text_accuracy': text_score,
            'layout_accuracy': layout_score,
            'confidence_reliability': text_quality.get('confidence_consistency', 0.0)
        }
    
    def _identify_issues_and_suggestions(self, image_quality: Dict[str, Any],
                                       text_quality: Dict[str, Any],
                                       processing_result: ProcessingResult) -> Tuple[List[str], List[str]]:
        """Identify quality issues and generate improvement suggestions"""
        issues = []
        suggestions = []
        
        # Image quality issues
        if not image_quality.get('resolution_adequate', False):
            issues.append("Image resolution is too low for optimal OCR")
            suggestions.append("Rescan document at higher resolution (300+ DPI)")
        
        if not image_quality.get('contrast_adequate', False):
            issues.append("Poor image contrast affecting text recognition")
            suggestions.append("Enhance image contrast or use better lighting when scanning")
        
        if not image_quality.get('skew_acceptable', True):
            issues.append("Document is skewed, affecting text alignment")
            suggestions.append("Straighten document before scanning or apply deskewing")
        
        if image_quality.get('noise_level') == 'high':
            issues.append("High noise levels detected in image")
            suggestions.append("Clean document surface and scanner bed, use noise reduction")
        
        # Text quality issues
        if text_quality.get('dictionary_word_ratio', 1.0) < 0.7:
            issues.append("Many unrecognized words detected")
            suggestions.append("Review text for OCR errors, consider manual correction")
        
        if text_quality.get('confidence_consistency', 1.0) < 0.5:
            issues.append("Inconsistent confidence scores across text")
            suggestions.append("Review low-confidence regions, may need manual verification")
        
        if text_quality.get('low_confidence_ratio', 0.0) > 0.3:
            issues.append("High proportion of low-confidence words")
            suggestions.append("Consider re-processing with different OCR settings")
        
        # Processing-specific issues
        if processing_result.ocr_result and processing_result.ocr_result.overall_confidence < 0.7:
            issues.append("Overall OCR confidence is low")
            suggestions.append("Consider using multiple OCR engines and comparing results")
        
        if processing_result.handwriting_result and processing_result.handwriting_result.requires_manual_review:
            issues.append("Handwritten content requires manual review")
            suggestions.append("Verify handwriting recognition results manually")
        
        return issues, suggestions
    
    def _assess_formatting_preservation(self, processing_result: ProcessingResult) -> float:
        """Assess how well formatting was preserved"""
        # This is a simplified assessment
        # In practice, would compare with original layout
        
        score = 0.5  # Base score
        
        if processing_result.layout_analysis:
            layout = processing_result.layout_analysis
            
            # Bonus for detecting structure
            if layout.total_table_regions > 0:
                score += 0.2
            
            if layout.structure.reading_order:
                score += 0.2
            
            if layout.layout_confidence > 0.8:
                score += 0.1
        
        return min(score, 1.0)
    
    def _assess_special_characters(self, text_quality: Dict[str, Any]) -> float:
        """Assess accuracy of special characters and punctuation"""
        # Simplified assessment based on character distribution
        char_dist = text_quality.get('character_distribution', {})
        
        # Check for reasonable punctuation presence
        punctuation_chars = [char for char in char_dist.keys() if char in string.punctuation]
        
        if punctuation_chars:
            # If punctuation is present and reasonably distributed
            punct_ratio = sum(char_dist.get(char, 0) for char in punctuation_chars)
            if 0.01 < punct_ratio < 0.15:  # Reasonable punctuation ratio
                return 0.8
            else:
                return 0.6
        else:
            # No punctuation detected - could be an issue
            return 0.4
    
    def _assess_legal_compliance(self, overall_scores: Dict[str, float],
                               processing_result: ProcessingResult) -> Dict[str, bool]:
        """Assess whether results meet legal standards"""
        overall_quality = overall_scores['overall_quality']
        text_accuracy = overall_scores['text_accuracy']
        
        # Legal standards thresholds (configurable)
        quality_threshold = 0.85
        accuracy_threshold = 0.90
        
        meets_standards = (overall_quality >= quality_threshold and 
                          text_accuracy >= accuracy_threshold)
        
        # Admissible quality (slightly lower threshold)
        admissible = overall_quality >= 0.75 and text_accuracy >= 0.80
        
        # Expert review needed for borderline cases
        needs_expert_review = (not meets_standards or 
                              overall_quality < 0.70 or
                              (processing_result.handwriting_result and 
                               processing_result.handwriting_result.requires_manual_review))
        
        return {
            'meets_standards': meets_standards,
            'admissible': admissible,
            'needs_expert_review': needs_expert_review
        }


# Convenience function for direct usage
def assess_ocr_quality(processing_result: ProcessingResult, 
                      config: Optional[OCRConfig] = None) -> QualityAssessment:
    """
    Convenience function to assess OCR quality
    
    Args:
        processing_result: Processing result to assess
        config: Optional configuration settings
        
    Returns:
        QualityAssessment with detailed quality metrics
    """
    if config is None:
        config = OCRConfig()
    
    assessor = QualityAssessor(config)
    return assessor.assess_quality(processing_result)