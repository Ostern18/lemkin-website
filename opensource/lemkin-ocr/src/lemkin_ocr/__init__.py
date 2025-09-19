"""
Lemkin Document Processing and OCR Toolkit

Comprehensive OCR and document processing capabilities for legal investigations.
"""

__version__ = "0.1.0"
__author__ = "Lemkin AI Contributors"
__email__ = "contributors@lemkin.org"

from .core import (
    DocumentProcessor,
    OCREngine,
    LayoutAnalyzer,
    TextExtractor,
    DocumentAnalysis,
    OCRResult,
    LayoutAnalysis,
    ExtractionResult,
)

__all__ = [
    "DocumentProcessor",
    "OCREngine",
    "LayoutAnalyzer",
    "TextExtractor",
    "DocumentAnalysis",
    "OCRResult",
    "LayoutAnalysis",
    "ExtractionResult",
]