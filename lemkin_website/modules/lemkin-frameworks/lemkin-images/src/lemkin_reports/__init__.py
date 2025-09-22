"""
Lemkin Report Generator Suite

Professional legal report generation and documentation tools for legal proceedings.
Provides standardized templates, evidence cataloging, and multi-format export capabilities.

Key Features:
- Standardized fact sheet generation
- Comprehensive evidence inventory
- Auto-populated legal brief templates
- Multi-format export (PDF, Word, LaTeX, HTML)
- Professional legal document formatting
- Citation management
- Template customization
- Batch report generation
"""

from .core import (
    ReportGenerator,
    ReportConfig,
    # Data Models
    FactSheet,
    EvidenceCatalog,
    LegalBrief,
    ExportedReport,
    CaseData,
    Evidence,
    LegalTemplate,
    ReportSection,
    ExportSettings,
    TemplateMetadata,
    # Enums
    ReportType,
    ExportFormat,
    EvidenceType,
    TemplateType,
    ReportStatus,
    CitationStyle,
    DocumentStandard
)

from .fact_sheet_generator import FactSheetGenerator
from .evidence_cataloger import EvidenceCataloger
from .legal_brief_formatter import LegalBriefFormatter
from .export_manager import ExportManager

# Convenience functions
def create_report_generator(config=None):
    """Create a new ReportGenerator instance"""
    return ReportGenerator(config)

def create_default_config():
    """Create default report configuration"""
    return ReportConfig()

# Version information
__version__ = "1.0.0"
__author__ = "Lemkin Legal Technologies"
__license__ = "MIT"

# Export public API
__all__ = [
    # Main classes
    'ReportGenerator',
    'ReportConfig',
    'FactSheetGenerator', 
    'EvidenceCataloger',
    'LegalBriefFormatter',
    'ExportManager',
    
    # Data models
    'FactSheet',
    'EvidenceCatalog', 
    'LegalBrief',
    'ExportedReport',
    'CaseData',
    'Evidence',
    'LegalTemplate',
    'ReportSection',
    'ExportSettings',
    'TemplateMetadata',
    
    # Enums
    'ReportType',
    'ExportFormat',
    'EvidenceType', 
    'TemplateType',
    'ReportStatus',
    'CitationStyle',
    'DocumentStandard',
    
    # Convenience functions
    'create_report_generator',
    'create_default_config',
]