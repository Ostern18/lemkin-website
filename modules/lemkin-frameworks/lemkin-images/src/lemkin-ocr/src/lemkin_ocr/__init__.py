"""
Lemkin OCR & Document Digitization Suite

This module provides comprehensive OCR and document digitization capabilities
designed specifically for legal document processing. It supports multi-language
OCR, document layout analysis, handwriting recognition, and quality assessment
with proper chain of custody for legal proceedings.

Key Features:
- Multi-engine OCR (Tesseract, EasyOCR, PaddleOCR) with 100+ language support
- Advanced document layout analysis and structure extraction
- Handwriting recognition for mixed-content documents
- OCR quality assessment and confidence scoring
- Document preprocessing and enhancement
- Searchable PDF generation with legal-grade metadata
- Batch processing capabilities for large document sets
- Chain of custody maintenance for legal compliance

Legal Compliance: Meets standards for digital evidence handling in legal proceedings
"""

from pathlib import Path
from typing import List, Optional

from .core import (
    DocumentDigitizer,
    OCRConfig,
    OCRResult,
    LayoutAnalysis,
    HandwritingResult,
    QualityAssessment,
    ProcessingResult,
    DocumentStructure,
    TextRegion,
    HandwrittenRegion,
    ImageRegion,
    DocumentMetadata,
)

from .multilingual_ocr import MultilingualOCR, ocr_document
from .layout_analyzer import LayoutAnalyzer, analyze_document_layout
from .handwriting_processor import HandwritingProcessor, process_handwriting
from .quality_assessor import QualityAssessor, assess_ocr_quality

# Version information
__version__ = "0.1.0"
__author__ = "Lemkin AI Contributors"
__email__ = "contributors@lemkin.org"

# Public API
__all__ = [
    # Core classes
    "DocumentDigitizer",
    "OCRConfig",
    
    # Main functions
    "ocr_document",
    "analyze_document_layout", 
    "process_handwriting",
    "assess_ocr_quality",
    
    # Data models
    "OCRResult",
    "LayoutAnalysis",
    "HandwritingResult",
    "QualityAssessment",
    "ProcessingResult",
    "DocumentStructure",
    "TextRegion",
    "HandwrittenRegion",
    "ImageRegion",
    "DocumentMetadata",
    
    # Component classes
    "MultilingualOCR",
    "LayoutAnalyzer",
    "HandwritingProcessor",
    "QualityAssessor",
]

# Convenience functions for quick access
def digitize_document(
    image_path: Path,
    language: str = "en",
    include_layout_analysis: bool = True,
    include_handwriting: bool = True,
    include_quality_assessment: bool = True,
    config: Optional[OCRConfig] = None
) -> ProcessingResult:
    """
    Convenience function to perform complete document digitization
    
    Args:
        image_path: Path to image or PDF file
        language: Primary language code (ISO 639-1)
        include_layout_analysis: Whether to analyze document layout
        include_handwriting: Whether to process handwritten content
        include_quality_assessment: Whether to assess OCR quality
        config: Optional configuration settings
        
    Returns:
        ProcessingResult with complete analysis
    """
    digitizer = DocumentDigitizer(config or OCRConfig())
    return digitizer.process_document(
        image_path=image_path,
        language=language,
        include_layout_analysis=include_layout_analysis,
        include_handwriting=include_handwriting,
        include_quality_assessment=include_quality_assessment
    )

def batch_digitize_documents(
    input_paths: List[Path],
    output_dir: Optional[Path] = None,
    language: str = "en",
    config: Optional[OCRConfig] = None
) -> List[ProcessingResult]:
    """
    Convenience function to batch process multiple documents
    
    Args:
        input_paths: List of paths to process
        output_dir: Optional output directory for results
        language: Primary language code
        config: Optional configuration settings
        
    Returns:
        List of ProcessingResult objects
    """
    digitizer = DocumentDigitizer(config or OCRConfig())
    return digitizer.batch_process_documents(
        input_paths=input_paths,
        output_dir=output_dir,
        language=language
    )