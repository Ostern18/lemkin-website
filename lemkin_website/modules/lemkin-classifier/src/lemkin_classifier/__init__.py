"""
Lemkin Legal Document Classifier

This package provides automated classification of legal documents using fine-tuned BERT models
to assist in evidence triage, case organization, and legal document management.
"""

__version__ = "0.1.0"
__author__ = "Lemkin AI Contributors"
__email__ = "contributors@lemkin.org"

from .core import (
    DocumentClassifier,
    DocumentContent,
    DocumentClassification,
    ClassificationResult,
    ClassificationConfig,
    ModelMetrics,
    TrainingMetrics,
)

from .legal_taxonomy import (
    LegalDocumentCategory,
    DocumentType,
    LegalDomain,
    CategoryHierarchy,
    get_category_hierarchy,
    get_supported_categories,
    validate_category,
)

from .confidence_scorer import (
    ConfidenceScorer,
    ConfidenceAssessment,
    ReviewTrigger,
    ConfidenceLevel,
    ScoreThresholds,
)

from .batch_processor import (
    BatchProcessor,
    BatchProcessingResult,
    ProcessingConfig,
    BatchMetrics,
    DocumentBatch,
)

__all__ = [
    # Core classification
    "DocumentClassifier",
    "DocumentContent", 
    "DocumentClassification",
    "ClassificationResult",
    "ClassificationConfig",
    "ModelMetrics",
    "TrainingMetrics",
    
    # Legal taxonomy
    "LegalDocumentCategory",
    "DocumentType",
    "LegalDomain",
    "CategoryHierarchy",
    "get_category_hierarchy",
    "get_supported_categories", 
    "validate_category",
    
    # Confidence assessment
    "ConfidenceScorer",
    "ConfidenceAssessment",
    "ReviewTrigger",
    "ConfidenceLevel",
    "ScoreThresholds",
    
    # Batch processing
    "BatchProcessor",
    "BatchProcessingResult",
    "ProcessingConfig",
    "BatchMetrics",
    "DocumentBatch",
]