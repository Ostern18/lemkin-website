"""
Lemkin Multilingual Named Entity Recognition and Linking

This package provides advanced NER capabilities optimized for legal documents
with support for entity linking, validation, and cross-document analysis.
"""

__version__ = "0.1.0"
__author__ = "Lemkin AI Contributors"
__email__ = "contributors@lemkin.org"

from .core import (
    LegalNERProcessor,
    NERConfig,
    Entity,
    EntityType,
    EntityGraph,
    EntityLinkResult,
    ValidationResult,
    LanguageCode,
    create_default_config,
    validate_config,
)

from .legal_ner import LegalEntityRecognizer
from .entity_linking import EntityLinker
from .multilingual_processor import MultilingualProcessor
from .entity_validator import EntityValidator, ValidationStatus, QualityMetric

__all__ = [
    "LegalNERProcessor",
    "NERConfig",
    "Entity",
    "EntityType", 
    "EntityGraph",
    "EntityLinkResult",
    "ValidationResult",
    "LanguageCode",
    "create_default_config",
    "validate_config",
    "LegalEntityRecognizer",
    "EntityLinker",
    "MultilingualProcessor",
    "EntityValidator",
    "ValidationStatus",
    "QualityMetric",
]