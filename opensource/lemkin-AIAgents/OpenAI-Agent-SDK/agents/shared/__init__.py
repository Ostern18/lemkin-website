"""
Shared Infrastructure for LemkinAI Agents (OpenAI Agents SDK Implementation)

This module provides common infrastructure for all LemkinAI agents using
the OpenAI Agents SDK framework with evidentiary compliance features.
"""

from .base_agent import LemkinAgent, BaseAgent, VisionCapableAgent
from .audit_logger import AuditLogger, AuditEventType
from .evidence_handler import (
    EvidenceHandler,
    EvidenceType,
    EvidenceStatus,
    EvidenceMetadata
)
from .output_formatter import OutputFormatter
from .utils import (
    extract_dates,
    extract_names,
    extract_locations,
    calculate_confidence_score,
    format_timestamp,
    parse_iso_timestamp,
    sanitize_filename,
    chunk_text,
    merge_dictionaries,
    validate_evidence_id,
    classify_confidence_level,
    truncate_text
)

__all__ = [
    # Base classes
    'LemkinAgent',
    'BaseAgent',  # Backward compatibility
    'VisionCapableAgent',  # Backward compatibility

    # Audit logging
    'AuditLogger',
    'AuditEventType',

    # Evidence handling
    'EvidenceHandler',
    'EvidenceType',
    'EvidenceStatus',
    'EvidenceMetadata',

    # Output formatting
    'OutputFormatter',

    # Utilities
    'extract_dates',
    'extract_names',
    'extract_locations',
    'calculate_confidence_score',
    'format_timestamp',
    'parse_iso_timestamp',
    'sanitize_filename',
    'chunk_text',
    'merge_dictionaries',
    'validate_evidence_id',
    'classify_confidence_level',
    'truncate_text'
]
