"""
Lemkin Communication Analysis Suite

Comprehensive communication analysis toolkit for seized communications pattern analysis 
and evidence extraction for legal investigations.

This package provides:
- Multi-platform chat export analysis (WhatsApp, Telegram, Signal, etc.)
- Email thread reconstruction and relationship mapping
- Communication network visualization with interactive graphs
- Temporal pattern analysis and anomaly detection
- Contact relationship mapping and communication frequency analysis
- Sentiment and topic analysis for forensic investigations

Legal Compliance: Meets standards for digital evidence handling in legal proceedings
"""

__version__ = "0.1.0"
__author__ = "Lemkin Digital Forensics"
__email__ = "forensics@lemkin.com"

# Core classes and functions
from .core import (
    # Main analyzer class
    CommunicationAnalyzer,
    
    # Configuration
    CommsConfig,
    
    # Data models
    AnalysisResult,
    ChatAnalysis,
    EmailAnalysis,
    NetworkGraph,
    PatternAnalysis,
    
    # Communication models
    Communication,
    ChatMessage,
    EmailMessage,
    Contact,
    CommunicationNetwork,
    TemporalPattern,
    Anomaly,
    
    # Enums
    CommunicationType,
    PlatformType,
    AnalysisStatus,
    PatternType,
    AnomalyType,
    AnalysisLevel,
    
    # Results
    NetworkNode,
    NetworkEdge,
    PatternMatch,
    AnomalyIndicator,
)

# Module-specific functions
from .chat_processor import (
    process_chat_exports,
    ChatProcessor,
    WhatsAppProcessor,
    TelegramProcessor,
    SignalProcessor,
)

from .email_analyzer import (
    analyze_email_threads,
    EmailAnalyzer,
    EmailThreadReconstructor,
    EmailRelationshipMapper,
)

from .network_mapper import (
    map_communication_network,
    NetworkMapper,
    NetworkVisualizer,
    CommunicationNetworkBuilder,
)

from .pattern_detector import (
    detect_communication_patterns,
    PatternDetector,
    TemporalAnalyzer,
    AnomalyDetector,
    SentimentAnalyzer,
)

# Utility functions
from .core import (
    load_communications,
    export_analysis_results,
    generate_forensic_report,
)

# CLI interface
from .cli import app

# Version info
__all__ = [
    # Core
    "CommunicationAnalyzer",
    "CommsConfig",
    "AnalysisResult",
    
    # Analysis results
    "ChatAnalysis",
    "EmailAnalysis", 
    "NetworkGraph",
    "PatternAnalysis",
    
    # Data models
    "Communication",
    "ChatMessage",
    "EmailMessage",
    "Contact",
    "CommunicationNetwork",
    "TemporalPattern",
    "Anomaly",
    
    # Enums
    "CommunicationType",
    "PlatformType",
    "AnalysisStatus",
    "PatternType",
    "AnomalyType",
    "AnalysisLevel",
    
    # Results
    "NetworkNode",
    "NetworkEdge",
    "PatternMatch",
    "AnomalyIndicator",
    
    # Processors
    "ChatProcessor",
    "EmailAnalyzer", 
    "NetworkMapper",
    "PatternDetector",
    
    # Main functions
    "process_chat_exports",
    "analyze_email_threads",
    "map_communication_network",
    "detect_communication_patterns",
    
    # Utilities
    "load_communications",
    "export_analysis_results",
    "generate_forensic_report",
    
    # CLI
    "app",
]