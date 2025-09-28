"""
Multilingual OCR Module for Lemkin OCR Suite

This module provides comprehensive multilingual OCR capabilities using multiple
OCR engines (Tesseract, EasyOCR, PaddleOCR) with advanced preprocessing,
language detection, and result fusion for optimal accuracy.

Designed specifically for legal document processing with high accuracy requirements.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any, Union
from PIL import Image, ImageEnhance, ImageFilter
from datetime import datetime
import logging
import concurrent.futures
from dataclasses import dataclass
import hashlib
import json

# OCR Engine imports
try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False
    logging.warning("Tesseract not available - install pytesseract")

try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False
    logging.warning("EasyOCR not available - install easyocr")

try:
    import paddleocr
    PADDLEOCR_AVAILABLE = True
except ImportError:
    PADDLEOCR_AVAILABLE = False
    logging.warning("PaddleOCR not available - install paddleocr")

# Language detection
try:
    from langdetect import detect, detect_langs
    LANGDETECT_AVAILABLE = True
except ImportError:
    LANGDETECT_AVAILABLE = False
    logging.warning("Language detection not available - install langdetect")

from .core import (
    OCRConfig, OCRResult, OCREngine, WordResult, BoundingBox, 
    ConfidenceLevel, ProcessingStatus
)

logger = logging.getLogger(__name__)


@dataclass
class PreprocessingStep:
    """Represents an image preprocessing step"""
    name: str
    applied: bool
    parameters: Dict[str, Any]


class ImagePreprocessor:
    """Advanced image preprocessing for OCR optimization"""
    
    def __init__(self, config: OCRConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.ImagePreprocessor")
    
    def preprocess_image(self, image: np.ndarray) -> Tuple[np.ndarray, List[PreprocessingStep]]:
        """
        Apply comprehensive image preprocessing for OCR
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Tuple of (processed_image, applied_steps)
        """
        processed_image = image.copy()
        steps_applied = []
        
        if not self.config.enable_preprocessing:
            return processed_image, steps_applied
        
        # Convert to grayscale if needed
        if len(processed_image.shape) == 3:
            processed_image = cv2.cvtColor(processed_image, cv2.COLOR_BGR2GRAY)
            steps_applied.append(PreprocessingStep("grayscale_conversion", True, {}))
        
        # Upscale image if specified
        if self.config.upscale_factor > 1.0:
            new_width = int(processed_image.shape[1] * self.config.upscale_factor)
            new_height = int(processed_image.shape[0] * self.config.upscale_factor)
            processed_image = cv2.resize(processed_image, (new_width, new_height), 
                                       interpolation=cv2.INTER_CUBIC)
            steps_applied.append(PreprocessingStep("upscaling", True, 
                                                 {"factor": self.config.upscale_factor}))
        
        # Deskew image
        if self.config.deskew_image:
            processed_image, angle = self._deskew_image(processed_image)
            steps_applied.append(PreprocessingStep("deskewing", True, {"angle": angle}))
        
        # Denoise image
        if self.config.denoise_image:
            processed_image = self._denoise_image(processed_image)
            steps_applied.append(PreprocessingStep("denoising", True, {}))
        
        # Enhance contrast
        if self.config.enhance_contrast:
            processed_image = self._enhance_contrast(processed_image)
            steps_applied.append(PreprocessingStep("contrast_enhancement", True, {}))
        
        # Binarize image if specified
        if self.config.binarize_image:
            processed_image = self._binarize_image(processed_image)
            steps_applied.append(PreprocessingStep("binarization", True, {}))
        
        return processed_image, steps_applied
    
    def _deskew_image(self, image: np.ndarray) -> Tuple[np.ndarray, float]:
        """Detect and correct image skew"""
        # Create binary image for skew detection
        binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by area and find text lines
        text_contours = []
        min_area = image.shape[0] * image.shape[1] * 0.001  # Minimum 0.1% of image area
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > min_area:
                # Get bounding rectangle
                rect = cv2.minAreaRect(contour)
                width, height = rect[1]
                if width > height * 2:  # Likely a text line
                    text_contours.append(rect)
        
        if not text_contours:
            return image, 0.0
        
        # Calculate average angle of text lines
        angles = [rect[2] for rect in text_contours]
        
        # Adjust angles to be between -45 and 45 degrees
        adjusted_angles = []
        for angle in angles:
            if angle < -45:
                angle = 90 + angle
            elif angle > 45:
                angle = angle - 90
            adjusted_angles.append(angle)
        
        # Calculate median angle
        adjusted_angles.sort()
        n = len(adjusted_angles)
        if n % 2 == 0:
            median_angle = (adjusted_angles[n//2 - 1] + adjusted_angles[n//2]) / 2
        else:
            median_angle = adjusted_angles[n//2]
        
        # Only correct if skew is significant
        if abs(median_angle) < 0.5:
            return image, 0.0
        
        # Rotate image to correct skew
        center = (image.shape[1] // 2, image.shape[0] // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, median_angle, 1.0)
        
        # Calculate new image dimensions
        cos_angle = abs(rotation_matrix[0, 0])
        sin_angle = abs(rotation_matrix[0, 1])
        new_width = int(image.shape[1] * cos_angle + image.shape[0] * sin_angle)
        new_height = int(image.shape[1] * sin_angle + image.shape[0] * cos_angle)
        
        # Adjust rotation matrix for translation
        rotation_matrix[0, 2] += (new_width / 2) - center[0]
        rotation_matrix[1, 2] += (new_height / 2) - center[1]
        
        # Apply rotation
        deskewed = cv2.warpAffine(image, rotation_matrix, (new_width, new_height), 
                                 flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        
        return deskewed, median_angle
    
    def _denoise_image(self, image: np.ndarray) -> np.ndarray:
        """Apply noise reduction to image"""
        # Apply bilateral filter to reduce noise while preserving edges
        denoised = cv2.bilateralFilter(image, 9, 75, 75)
        
        # Apply morphological operations to clean up small noise
        kernel = np.ones((2, 2), np.uint8)
        denoised = cv2.morphologyEx(denoised, cv2.MORPH_CLOSE, kernel)
        
        return denoised
    
    def _enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        """Enhance image contrast using CLAHE"""
        # Create CLAHE object
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        
        # Apply CLAHE
        enhanced = clahe.apply(image)
        
        # Additional histogram equalization if needed
        # enhanced = cv2.equalizeHist(enhanced)
        
        return enhanced
    
    def _binarize_image(self, image: np.ndarray) -> np.ndarray:
        """Convert image to binary using adaptive thresholding"""
        # Apply Gaussian blur first
        blurred = cv2.GaussianBlur(image, (5, 5), 0)
        
        # Use adaptive threshold for better results on varying lighting
        binary = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        
        return binary


class TesseractEngine:
    """Tesseract OCR engine wrapper"""
    
    def __init__(self, config: OCRConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.TesseractEngine")
        
        if not TESSERACT_AVAILABLE:
            raise ImportError("Tesseract not available. Install pytesseract.")
    
    def perform_ocr(self, image: np.ndarray, language: str) -> Dict[str, Any]:
        """
        Perform OCR using Tesseract
        
        Args:
            image: Preprocessed image
            language: Language code
            
        Returns:
            Dictionary with OCR results
        """
        try:
            # Configure Tesseract parameters
            config_str = f'--oem {self.config.tesseract_oem} --psm {self.config.tesseract_psm}'
            if self.config.tesseract_config:
                config_str += f' {self.config.tesseract_config}'
            
            # Get detailed OCR data
            data = pytesseract.image_to_data(
                image, lang=language, config=config_str, output_type=pytesseract.Output.DICT
            )
            
            # Extract text
            text = pytesseract.image_to_string(image, lang=language, config=config_str)
            
            # Process word-level results
            words = []
            for i in range(len(data['text'])):
                if int(data['conf'][i]) > 0:  # Filter out low confidence detections
                    word_text = data['text'][i].strip()
                    if word_text:
                        bbox = BoundingBox(
                            x=data['left'][i],
                            y=data['top'][i],
                            width=data['width'][i],
                            height=data['height'][i]
                        )
                        
                        word = WordResult(
                            text=word_text,
                            confidence=float(data['conf'][i]) / 100.0,
                            bounding_box=bbox
                        )
                        words.append(word)
            
            # Calculate overall confidence
            confidences = [float(c) for c in data['conf'] if int(c) > 0]
            overall_confidence = sum(confidences) / len(confidences) / 100.0 if confidences else 0.0
            
            return {
                'engine': 'tesseract',
                'text': text.strip(),
                'words': words,
                'confidence': overall_confidence,
                'raw_data': data
            }
            
        except Exception as e:
            self.logger.error(f"Tesseract OCR failed: {str(e)}")
            return {
                'engine': 'tesseract',
                'text': '',
                'words': [],
                'confidence': 0.0,
                'error': str(e)
            }


class EasyOCREngine:
    """EasyOCR engine wrapper"""
    
    def __init__(self, config: OCRConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.EasyOCREngine")
        
        if not EASYOCR_AVAILABLE:
            raise ImportError("EasyOCR not available. Install easyocr.")
        
        # Initialize EasyOCR reader (lazy initialization)
        self._reader = None
    
    def _get_reader(self, languages: List[str]):
        """Get or create EasyOCR reader for specified languages"""
        if self._reader is None:
            try:
                self._reader = easyocr.Reader(languages, gpu=True)
            except:
                # Fallback to CPU if GPU not available
                self._reader = easyocr.Reader(languages, gpu=False)
        return self._reader
    
    def perform_ocr(self, image: np.ndarray, language: str) -> Dict[str, Any]:
        """
        Perform OCR using EasyOCR
        
        Args:
            image: Preprocessed image
            language: Language code
            
        Returns:
            Dictionary with OCR results
        """
        try:
            # Convert language code (EasyOCR uses different codes)
            easyocr_lang = self._convert_language_code(language)
            languages = [easyocr_lang]
            
            # Add secondary languages if specified
            for secondary_lang in self.config.secondary_languages:
                easyocr_secondary = self._convert_language_code(secondary_lang)
                if easyocr_secondary not in languages:
                    languages.append(easyocr_secondary)
            
            reader = self._get_reader(languages)
            
            # Perform OCR
            results = reader.readtext(image, detail=1, paragraph=False)
            
            # Process results
            words = []
            text_parts = []
            confidences = []
            
            for result in results:
                bbox_coords, text_content, confidence = result
                
                # Convert bounding box format
                # EasyOCR returns [[x1,y1], [x2,y1], [x2,y2], [x1,y2]]
                x_coords = [point[0] for point in bbox_coords]
                y_coords = [point[1] for point in bbox_coords]
                
                x = int(min(x_coords))
                y = int(min(y_coords))
                width = int(max(x_coords) - x)
                height = int(max(y_coords) - y)
                
                bbox = BoundingBox(x=x, y=y, width=width, height=height)
                
                word = WordResult(
                    text=text_content,
                    confidence=float(confidence),
                    bounding_box=bbox
                )
                
                words.append(word)
                text_parts.append(text_content)
                confidences.append(float(confidence))
            
            # Combine text and calculate overall confidence
            combined_text = ' '.join(text_parts)
            overall_confidence = sum(confidences) / len(confidences) if confidences else 0.0
            
            return {
                'engine': 'easyocr',
                'text': combined_text,
                'words': words,
                'confidence': overall_confidence,
                'raw_data': results
            }
            
        except Exception as e:
            self.logger.error(f"EasyOCR failed: {str(e)}")
            return {
                'engine': 'easyocr',
                'text': '',
                'words': [],
                'confidence': 0.0,
                'error': str(e)
            }
    
    def _convert_language_code(self, iso_code: str) -> str:
        """Convert ISO language code to EasyOCR format"""
        mapping = {
            'en': 'en',
            'es': 'es', 
            'fr': 'fr',
            'de': 'de',
            'it': 'it',
            'pt': 'pt',
            'ru': 'ru',
            'ja': 'ja',
            'ko': 'ko',
            'zh': 'ch_sim',
            'ar': 'ar',
            'hi': 'hi',
            'th': 'th',
            'vi': 'vi'
        }
        return mapping.get(iso_code, 'en')


class PaddleOCREngine:
    """PaddleOCR engine wrapper"""
    
    def __init__(self, config: OCRConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.PaddleOCREngine")
        
        if not PADDLEOCR_AVAILABLE:
            raise ImportError("PaddleOCR not available. Install paddleocr.")
        
        # Initialize PaddleOCR (lazy initialization)
        self._ocr = None
    
    def _get_ocr(self, language: str):
        """Get or create PaddleOCR instance for specified language"""
        if self._ocr is None:
            try:
                self._ocr = paddleocr.PaddleOCR(use_angle_cls=True, lang=language, show_log=False)
            except Exception as e:
                self.logger.error(f"Failed to initialize PaddleOCR: {str(e)}")
                raise
        return self._ocr
    
    def perform_ocr(self, image: np.ndarray, language: str) -> Dict[str, Any]:
        """
        Perform OCR using PaddleOCR
        
        Args:
            image: Preprocessed image
            language: Language code
            
        Returns:
            Dictionary with OCR results
        """
        try:
            # Convert language code
            paddle_lang = self._convert_language_code(language)
            
            ocr = self._get_ocr(paddle_lang)
            
            # Perform OCR
            results = ocr.ocr(image, cls=True)
            
            # Process results
            words = []
            text_parts = []
            confidences = []
            
            if results and results[0]:
                for line in results[0]:
                    bbox_coords, (text_content, confidence) = line
                    
                    # Convert bounding box format
                    # PaddleOCR returns [[x1,y1], [x2,y1], [x2,y2], [x1,y2]]
                    x_coords = [point[0] for point in bbox_coords]
                    y_coords = [point[1] for point in bbox_coords]
                    
                    x = int(min(x_coords))
                    y = int(min(y_coords))
                    width = int(max(x_coords) - x)
                    height = int(max(y_coords) - y)
                    
                    bbox = BoundingBox(x=x, y=y, width=width, height=height)
                    
                    word = WordResult(
                        text=text_content,
                        confidence=float(confidence),
                        bounding_box=bbox
                    )
                    
                    words.append(word)
                    text_parts.append(text_content)
                    confidences.append(float(confidence))
            
            # Combine text and calculate overall confidence
            combined_text = ' '.join(text_parts)
            overall_confidence = sum(confidences) / len(confidences) if confidences else 0.0
            
            return {
                'engine': 'paddleocr',
                'text': combined_text,
                'words': words,
                'confidence': overall_confidence,
                'raw_data': results
            }
            
        except Exception as e:
            self.logger.error(f"PaddleOCR failed: {str(e)}")
            return {
                'engine': 'paddleocr',
                'text': '',
                'words': [],
                'confidence': 0.0,
                'error': str(e)
            }
    
    def _convert_language_code(self, iso_code: str) -> str:
        """Convert ISO language code to PaddleOCR format"""
        mapping = {
            'en': 'en',
            'es': 'es',
            'fr': 'fr',
            'de': 'german',
            'it': 'it',
            'pt': 'pt',
            'ru': 'ru',
            'ja': 'japan',
            'ko': 'korean',
            'zh': 'ch',
            'ar': 'ar',
            'hi': 'hi',
            'th': 'th',
            'vi': 'vi'
        }
        return mapping.get(iso_code, 'en')


class LanguageDetector:
    """Language detection utility"""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.LanguageDetector")
    
    def detect_language(self, text: str) -> Tuple[Optional[str], Optional[float]]:
        """
        Detect primary language of text
        
        Args:
            text: Input text
            
        Returns:
            Tuple of (language_code, confidence)
        """
        if not LANGDETECT_AVAILABLE or not text.strip():
            return None, None
        
        try:
            # Detect language
            detected_lang = detect(text)
            
            # Get detailed language probabilities
            lang_probs = detect_langs(text)
            
            # Find confidence for detected language
            confidence = 0.0
            for lang_prob in lang_probs:
                if lang_prob.lang == detected_lang:
                    confidence = lang_prob.prob
                    break
            
            return detected_lang, confidence
            
        except Exception as e:
            self.logger.warning(f"Language detection failed: {str(e)}")
            return None, None
    
    def detect_multiple_languages(self, text: str) -> List[Tuple[str, float]]:
        """
        Detect multiple languages in text with probabilities
        
        Args:
            text: Input text
            
        Returns:
            List of (language_code, probability) tuples
        """
        if not LANGDETECT_AVAILABLE or not text.strip():
            return []
        
        try:
            lang_probs = detect_langs(text)
            return [(lang.lang, lang.prob) for lang in lang_probs]
        except Exception as e:
            self.logger.warning(f"Multiple language detection failed: {str(e)}")
            return []


class ResultFusion:
    """Fuses results from multiple OCR engines"""
    
    def __init__(self, config: OCRConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.ResultFusion")
    
    def fuse_results(self, engine_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Fuse results from multiple OCR engines
        
        Args:
            engine_results: List of results from different engines
            
        Returns:
            Fused OCR result
        """
        if not engine_results:
            return {
                'text': '',
                'words': [],
                'confidence': 0.0,
                'engines_used': []
            }
        
        # Filter out failed results
        valid_results = [r for r in engine_results if 'error' not in r]
        
        if not valid_results:
            return {
                'text': '',
                'words': [],
                'confidence': 0.0,
                'engines_used': [r.get('engine', 'unknown') for r in engine_results],
                'fusion_method': 'all_failed'
            }
        
        # If only one valid result, return it
        if len(valid_results) == 1:
            result = valid_results[0].copy()
            result['engines_used'] = [result.get('engine')]
            result['fusion_method'] = 'single_engine'
            return result
        
        # Multiple valid results - perform fusion
        return self._perform_fusion(valid_results)
    
    def _perform_fusion(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Perform intelligent fusion of multiple OCR results
        
        Args:
            results: List of valid OCR results
            
        Returns:
            Fused result
        """
        # Strategy: Choose best result based on confidence and text length
        # Could be enhanced with more sophisticated voting mechanisms
        
        # Weight results by confidence
        weighted_results = []
        for result in results:
            confidence = result.get('confidence', 0.0)
            text_length = len(result.get('text', ''))
            
            # Penalize very short results (likely incomplete)
            length_penalty = 1.0 if text_length > 10 else 0.5
            
            # Combined score
            score = confidence * length_penalty
            weighted_results.append((score, result))
        
        # Sort by score (highest first)
        weighted_results.sort(key=lambda x: x[0], reverse=True)
        
        # Use the highest scoring result as base
        best_score, best_result = weighted_results[0]
        
        fused_result = {
            'text': best_result.get('text', ''),
            'words': best_result.get('words', []),
            'confidence': best_result.get('confidence', 0.0),
            'engines_used': [r.get('engine') for _, r in weighted_results],
            'fusion_method': 'confidence_weighted',
            'fusion_scores': [score for score, _ in weighted_results]
        }
        
        # Could add character-level or word-level fusion here for even better results
        
        return fused_result


class MultilingualOCR:
    """
    Main multilingual OCR class that orchestrates multiple OCR engines
    """
    
    def __init__(self, config: OCRConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.MultilingualOCR")
        
        # Initialize components
        self.preprocessor = ImagePreprocessor(config)
        self.language_detector = LanguageDetector()
        self.result_fusion = ResultFusion(config)
        
        # Initialize OCR engines based on configuration
        self.engines = {}
        self._initialize_engines()
    
    def _initialize_engines(self):
        """Initialize available OCR engines"""
        for engine in self.config.ocr_engines:
            if engine == OCREngine.ALL:
                # Initialize all available engines
                if TESSERACT_AVAILABLE and OCREngine.TESSERACT not in self.engines:
                    try:
                        self.engines[OCREngine.TESSERACT] = TesseractEngine(self.config)
                        self.logger.info("Tesseract engine initialized")
                    except Exception as e:
                        self.logger.error(f"Failed to initialize Tesseract: {str(e)}")
                
                if EASYOCR_AVAILABLE and OCREngine.EASYOCR not in self.engines:
                    try:
                        self.engines[OCREngine.EASYOCR] = EasyOCREngine(self.config)
                        self.logger.info("EasyOCR engine initialized")
                    except Exception as e:
                        self.logger.error(f"Failed to initialize EasyOCR: {str(e)}")
                
                if PADDLEOCR_AVAILABLE and OCREngine.PADDLEOCR not in self.engines:
                    try:
                        self.engines[OCREngine.PADDLEOCR] = PaddleOCREngine(self.config)
                        self.logger.info("PaddleOCR engine initialized")
                    except Exception as e:
                        self.logger.error(f"Failed to initialize PaddleOCR: {str(e)}")
            
            else:
                # Initialize specific engine
                try:
                    if engine == OCREngine.TESSERACT and TESSERACT_AVAILABLE:
                        self.engines[engine] = TesseractEngine(self.config)
                        self.logger.info("Tesseract engine initialized")
                    elif engine == OCREngine.EASYOCR and EASYOCR_AVAILABLE:
                        self.engines[engine] = EasyOCREngine(self.config)
                        self.logger.info("EasyOCR engine initialized")
                    elif engine == OCREngine.PADDLEOCR and PADDLEOCR_AVAILABLE:
                        self.engines[engine] = PaddleOCREngine(self.config)
                        self.logger.info("PaddleOCR engine initialized")
                except Exception as e:
                    self.logger.error(f"Failed to initialize {engine}: {str(e)}")
        
        if not self.engines:
            raise RuntimeError("No OCR engines could be initialized")
        
        self.logger.info(f"Initialized {len(self.engines)} OCR engines: {list(self.engines.keys())}")
    
    def perform_ocr(self, image_path: Path, language: str) -> OCRResult:
        """
        Perform comprehensive multilingual OCR on an image
        
        Args:
            image_path: Path to image file
            language: Primary language code (ISO 639-1)
            
        Returns:
            OCRResult with comprehensive analysis
        """
        start_time = datetime.utcnow()
        
        if not image_path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        self.logger.info(f"Starting OCR processing for: {image_path.name}")
        
        try:
            # Load image
            if image_path.suffix.lower() in ['.pdf']:
                # Handle PDF files
                image = self._load_pdf_page(image_path, 0)
            else:
                # Handle image files
                image = cv2.imread(str(image_path))
                if image is None:
                    raise ValueError(f"Could not load image: {image_path}")
            
            # Preprocess image
            processed_image, preprocessing_steps = self.preprocessor.preprocess_image(image)
            
            # Run OCR with all configured engines
            engine_results = []
            
            if self.config.parallel_processing and len(self.engines) > 1:
                # Parallel processing
                with concurrent.futures.ThreadPoolExecutor(max_workers=len(self.engines)) as executor:
                    future_to_engine = {
                        executor.submit(engine.perform_ocr, processed_image, language): engine_name
                        for engine_name, engine in self.engines.items()
                    }
                    
                    for future in concurrent.futures.as_completed(future_to_engine):
                        result = future.result()
                        engine_results.append(result)
            else:
                # Sequential processing
                for engine_name, engine in self.engines.items():
                    result = engine.perform_ocr(processed_image, language)
                    engine_results.append(result)
            
            # Fuse results from multiple engines
            fused_result = self.result_fusion.fuse_results(engine_results)
            
            # Detect language if not specified or for validation
            detected_language = None
            language_confidence = None
            detected_languages = []
            
            if fused_result['text']:
                detected_language, language_confidence = self.language_detector.detect_language(
                    fused_result['text']
                )
                detected_languages = self.language_detector.detect_multiple_languages(
                    fused_result['text']
                )
            
            # Calculate processing duration
            end_time = datetime.utcnow()
            duration = (end_time - start_time).total_seconds()
            
            # Determine confidence level
            confidence = fused_result.get('confidence', 0.0)
            if confidence >= 0.95:
                confidence_level = ConfidenceLevel.VERY_HIGH
            elif confidence >= 0.85:
                confidence_level = ConfidenceLevel.HIGH
            elif confidence >= 0.70:
                confidence_level = ConfidenceLevel.MEDIUM
            elif confidence >= 0.50:
                confidence_level = ConfidenceLevel.LOW
            else:
                confidence_level = ConfidenceLevel.VERY_LOW
            
            # Extract word confidences
            word_confidences = [word.confidence for word in fused_result.get('words', [])]
            
            # Create OCR result
            ocr_result = OCRResult(
                document_path=str(image_path),
                processing_duration_seconds=duration,
                engines_used=fused_result.get('engines_used', []),
                text=fused_result.get('text', ''),
                word_count=len(fused_result.get('text', '').split()),
                character_count=len(fused_result.get('text', '')),
                overall_confidence=confidence,
                confidence_level=confidence_level,
                word_confidences=word_confidences,
                primary_language=detected_language or language,
                languages_detected=[lang for lang, _ in detected_languages[:5]],  # Top 5
                language_confidence=language_confidence,
                image_preprocessing_applied=[step.name for step in preprocessing_steps],
                ocr_parameters={
                    'tesseract_psm': self.config.tesseract_psm,
                    'tesseract_oem': self.config.tesseract_oem,
                    'upscale_factor': self.config.upscale_factor,
                    'preprocessing_enabled': self.config.enable_preprocessing
                },
                engine_results={
                    result.get('engine', 'unknown'): {
                        'text': result.get('text', ''),
                        'confidence': result.get('confidence', 0.0),
                        'word_count': len(result.get('words', []))
                    } for result in engine_results if 'error' not in result
                }
            )
            
            self.logger.info(f"OCR processing completed: {confidence:.2%} confidence, "
                           f"{ocr_result.word_count} words")
            
            return ocr_result
            
        except Exception as e:
            self.logger.error(f"OCR processing failed: {str(e)}")
            # Return failed result
            duration = (datetime.utcnow() - start_time).total_seconds()
            return OCRResult(
                document_path=str(image_path),
                processing_duration_seconds=duration,
                engines_used=list(self.engines.keys()),
                text="",
                word_count=0,
                character_count=0,
                overall_confidence=0.0,
                confidence_level=ConfidenceLevel.VERY_LOW,
                word_confidences=[],
                image_preprocessing_applied=[],
                ocr_parameters={},
                engine_results={}
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
                # Convert PIL image to OpenCV format
                pil_image = pages[0]
                return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            else:
                raise ValueError(f"Could not extract page {page_number} from PDF")
                
        except ImportError:
            raise ImportError("pdf2image not available. Install pdf2image to process PDFs.")


# Convenience function for direct usage
def ocr_document(image_path: Path, language: str = "en", config: Optional[OCRConfig] = None) -> OCRResult:
    """
    Convenience function to perform OCR on a document
    
    Args:
        image_path: Path to image or PDF file
        language: Primary language code (ISO 639-1)
        config: Optional configuration settings
        
    Returns:
        OCRResult with extracted text and metadata
    """
    if config is None:
        config = OCRConfig()
    
    ocr = MultilingualOCR(config)
    return ocr.perform_ocr(image_path, language)