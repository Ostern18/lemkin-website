"""
LemkinAI Shared Infrastructure

Core components for building evidentiary-compliant AI agents.
"""

from .audit_logger import AuditLogger, AuditEventType, AuditDecorator
from .base_agent import BaseAgent, VisionCapableAgent
from .evidence_handler import (
    EvidenceHandler,
    EvidenceType,
    EvidenceStatus,
    EvidenceMetadata
)
from .output_formatter import OutputFormatter
from . import utils

__all__ = [
    'AuditLogger',
    'AuditEventType',
    'AuditDecorator',
    'BaseAgent',
    'VisionCapableAgent',
    'EvidenceHandler',
    'EvidenceType',
    'EvidenceStatus',
    'EvidenceMetadata',
    'OutputFormatter',
    'utils'
]

__version__ = '0.1.0'
