"""
Core classes and data models for the Lemkin Legal Research Assistant.

This module provides the main LegalResearchAssistant class and comprehensive
data models for legal research operations including case law search,
precedent analysis, and citation processing.
"""

from datetime import datetime, date
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Set
from uuid import uuid4

from pydantic import BaseModel, Field, validator, ConfigDict
from loguru import logger


# Enums for type safety and standardization

class CitationType(str, Enum):
    """Legal citation types"""
    CASE = "case"
    STATUTE = "statute"
    REGULATION = "regulation"
    CONSTITUTIONAL = "constitutional"
    SECONDARY = "secondary"
    FOREIGN = "foreign"
    TREATY = "treaty"
    UNKNOWN = "unknown"


class JurisdictionType(str, Enum):
    """Legal jurisdiction types"""
    FEDERAL = "federal"
    STATE = "state"
    LOCAL = "local"
    INTERNATIONAL = "international"
    TRIBAL = "tribal"
    ADMINISTRATIVE = "administrative"


class DatabaseType(str, Enum):
    """Legal database types"""
    WESTLAW = "westlaw"
    LEXIS = "lexis"
    GOOGLE_SCHOLAR = "google_scholar"
    JUSTIA = "justia"
    COURTLISTENER = "courtlistener"
    CASELAW_ACCESS = "caselaw_access"
    FASTCASE = "fastcase"
    BLOOMBERG_LAW = "bloomberg_law"
    FREE_LAW = "free_law"
    CUSTOM = "custom"


class ResearchStatus(str, Enum):
    """Research operation status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIAL = "partial"
    CANCELLED = "cancelled"


class RelevanceScore(str, Enum):
    """Case relevance scoring levels"""
    HIGHLY_RELEVANT = "highly_relevant"
    RELEVANT = "relevant"
    SOMEWHAT_RELEVANT = "somewhat_relevant"
    MARGINALLY_RELEVANT = "marginally_relevant"
    NOT_RELEVANT = "not_relevant"


class CitationStyle(str, Enum):
    """Legal citation formatting styles"""
    BLUEBOOK = "bluebook"
    ALWD = "alwd"
    APA = "apa"
    MLA = "mla"
    CHICAGO = "chicago"
    CUSTOM = "custom"


# Base data models

class BaseResearchModel(BaseModel):
    """Base model for all research entities"""
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        use_enum_values=True
    )
    
    id: str = Field(default_factory=lambda: str(uuid4()))
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class LegalCitation(BaseResearchModel):
    """Legal citation with parsing and validation"""
    raw_citation: str = Field(..., description="Original citation text")
    parsed_citation: Optional[str] = Field(None, description="Standardized citation")
    citation_type: CitationType = Field(CitationType.UNKNOWN)
    case_name: Optional[str] = None
    reporter: Optional[str] = None
    volume: Optional[str] = None
    page: Optional[str] = None
    year: Optional[int] = None
    court: Optional[str] = None
    jurisdiction: Optional[JurisdictionType] = None
    citation_style: CitationStyle = Field(CitationStyle.BLUEBOOK)
    is_valid: bool = Field(False)
    validation_errors: List[str] = Field(default_factory=list)
    pin_cite: Optional[str] = None
    parallel_citations: List[str] = Field(default_factory=list)
    
    @validator('year')
    def validate_year(cls, v):
        if v is not None and (v < 1600 or v > datetime.now().year + 1):
            raise ValueError(f"Invalid year: {v}")
        return v


class CaseOpinion(BaseResearchModel):
    """Individual case opinion or decision"""
    case_name: str = Field(..., description="Full case name")
    citation: str = Field(..., description="Primary citation")
    court: str = Field(..., description="Court that decided the case")
    date_decided: Optional[date] = None
    judge: Optional[str] = None
    docket_number: Optional[str] = None
    opinion_type: str = Field("majority", description="Type of opinion")
    full_text: Optional[str] = None
    summary: Optional[str] = None
    key_facts: List[str] = Field(default_factory=list)
    legal_issues: List[str] = Field(default_factory=list)
    holdings: List[str] = Field(default_factory=list)
    jurisdiction: Optional[JurisdictionType] = None
    subject_areas: List[str] = Field(default_factory=list)
    cited_cases: List[str] = Field(default_factory=list)
    citing_cases: List[str] = Field(default_factory=list)
    overruled: bool = Field(False)
    reversed: bool = Field(False)
    url: Optional[str] = None


class SimilarCase(BaseResearchModel):
    """Case identified as similar to query case"""
    case_opinion: CaseOpinion
    similarity_score: float = Field(..., ge=0.0, le=1.0)
    relevance_score: RelevanceScore = Field(RelevanceScore.NOT_RELEVANT)
    similarity_reasons: List[str] = Field(default_factory=list)
    key_similarities: List[str] = Field(default_factory=list)
    key_differences: List[str] = Field(default_factory=list)
    procedural_posture: Optional[str] = None
    factual_similarity: float = Field(0.0, ge=0.0, le=1.0)
    legal_similarity: float = Field(0.0, ge=0.0, le=1.0)
    outcome_similarity: float = Field(0.0, ge=0.0, le=1.0)


class Precedent(BaseResearchModel):
    """Legal precedent with binding analysis"""
    case_opinion: CaseOpinion
    binding_strength: str = Field("persuasive", description="Binding authority level")
    precedential_value: float = Field(..., ge=0.0, le=1.0)
    distinguishable_factors: List[str] = Field(default_factory=list)
    supporting_rationale: List[str] = Field(default_factory=list)
    contradictory_precedents: List[str] = Field(default_factory=list)
    jurisdictional_weight: float = Field(0.0, ge=0.0, le=1.0)
    temporal_relevance: float = Field(0.0, ge=0.0, le=1.0)
    subject_matter_relevance: float = Field(0.0, ge=0.0, le=1.0)


class DatabaseResult(BaseResearchModel):
    """Result from a specific legal database"""
    database: DatabaseType
    query: str
    results_count: int = Field(0, ge=0)
    cases: List[CaseOpinion] = Field(default_factory=list)
    search_time: float = Field(0.0, ge=0.0)
    next_page_token: Optional[str] = None
    total_estimated: Optional[int] = None
    filters_applied: Dict[str, Any] = Field(default_factory=dict)
    api_response: Optional[Dict[str, Any]] = None


class CaseLawResults(BaseResearchModel):
    """Comprehensive case law search results"""
    query: str = Field(..., description="Original search query")
    total_results: int = Field(0, ge=0)
    database_results: List[DatabaseResult] = Field(default_factory=list)
    aggregated_cases: List[CaseOpinion] = Field(default_factory=list)
    similar_cases: List[SimilarCase] = Field(default_factory=list)
    precedents: List[Precedent] = Field(default_factory=list)
    search_duration: float = Field(0.0, ge=0.0)
    filters_applied: Dict[str, Any] = Field(default_factory=dict)
    jurisdiction_breakdown: Dict[str, int] = Field(default_factory=dict)
    date_range: Optional[Dict[str, date]] = None
    status: ResearchStatus = Field(ResearchStatus.PENDING)


class CitationMatch(BaseResearchModel):
    """Citation parsing and validation result"""
    original_text: str
    citations_found: List[LegalCitation] = Field(default_factory=list)
    parsing_confidence: float = Field(0.0, ge=0.0, le=1.0)
    extraction_method: str = Field("regex")
    validation_passed: bool = Field(False)
    standardized_citations: List[str] = Field(default_factory=list)
    formatting_suggestions: List[str] = Field(default_factory=list)


class PrecedentMatch(BaseResearchModel):
    """Precedent analysis match result"""
    query_case: str
    matched_precedent: Precedent
    match_confidence: float = Field(..., ge=0.0, le=1.0)
    analysis_method: str = Field("semantic_similarity")
    supporting_evidence: List[str] = Field(default_factory=list)
    distinguishing_factors: List[str] = Field(default_factory=list)
    recommendation: str = Field("review_manually")


class LegalDatabase(BaseResearchModel):
    """Legal database configuration"""
    name: str
    database_type: DatabaseType
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    rate_limit: int = Field(100, gt=0)  # requests per minute
    timeout: int = Field(30, gt=0)  # seconds
    enabled: bool = Field(True)
    subscription_level: str = Field("free")
    supported_jurisdictions: List[JurisdictionType] = Field(default_factory=list)
    supported_citation_styles: List[CitationStyle] = Field(default_factory=list)
    search_capabilities: Dict[str, bool] = Field(default_factory=dict)
    last_accessed: Optional[datetime] = None


class SearchQuery(BaseResearchModel):
    """Legal search query specification"""
    query_text: str = Field(..., description="Main search query")
    case_name: Optional[str] = None
    court: Optional[str] = None
    judge: Optional[str] = None
    date_range: Optional[Dict[str, date]] = None
    jurisdiction: Optional[JurisdictionType] = None
    subject_areas: List[str] = Field(default_factory=list)
    citation_requirements: List[str] = Field(default_factory=list)
    exclude_terms: List[str] = Field(default_factory=list)
    boolean_operators: bool = Field(True)
    fuzzy_matching: bool = Field(True)
    max_results: int = Field(100, gt=0, le=1000)
    sort_by: str = Field("relevance")
    include_related: bool = Field(True)


class ResearchSummary(BaseResearchModel):
    """Comprehensive legal research summary"""
    research_question: str
    key_findings: List[str] = Field(default_factory=list)
    primary_precedents: List[Precedent] = Field(default_factory=list)
    supporting_cases: List[SimilarCase] = Field(default_factory=list)
    contradictory_authority: List[CaseOpinion] = Field(default_factory=list)
    jurisdictional_analysis: Dict[str, str] = Field(default_factory=dict)
    temporal_analysis: Dict[str, str] = Field(default_factory=dict)
    confidence_level: float = Field(0.0, ge=0.0, le=1.0)
    research_gaps: List[str] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)
    citations_analyzed: int = Field(0, ge=0)
    databases_searched: List[DatabaseType] = Field(default_factory=list)


class LegalMemo(BaseResearchModel):
    """Legal research memorandum"""
    title: str
    client: Optional[str] = None
    attorney: Optional[str] = None
    date_prepared: date = Field(default_factory=date.today)
    research_question: str
    brief_answer: str
    executive_summary: str
    factual_background: str
    legal_analysis: str
    conclusion: str
    recommendations: List[str] = Field(default_factory=list)
    supporting_authority: List[str] = Field(default_factory=list)
    contrary_authority: List[str] = Field(default_factory=list)
    research_sources: List[str] = Field(default_factory=list)
    appendices: List[str] = Field(default_factory=list)
    word_count: int = Field(0, ge=0)
    formatting_style: CitationStyle = Field(CitationStyle.BLUEBOOK)


class CitationFormat(BaseResearchModel):
    """Citation formatting configuration"""
    style: CitationStyle
    case_format: str = Field("{case_name}, {citation} ({court} {year})")
    statute_format: str = Field("{title} {code} § {section} ({year})")
    regulation_format: str = Field("{title} C.F.R. § {section} ({year})")
    abbreviation_rules: Dict[str, str] = Field(default_factory=dict)
    punctuation_rules: Dict[str, str] = Field(default_factory=dict)
    spacing_rules: Dict[str, str] = Field(default_factory=dict)
    italics_rules: List[str] = Field(default_factory=list)
    short_citation_rules: Dict[str, str] = Field(default_factory=dict)


class ResearchConfig(BaseResearchModel):
    """Configuration for legal research operations"""
    # Database settings
    enabled_databases: List[DatabaseType] = Field(default_factory=lambda: [
        DatabaseType.GOOGLE_SCHOLAR, 
        DatabaseType.JUSTIA,
        DatabaseType.COURTLISTENER
    ])
    database_configs: Dict[str, LegalDatabase] = Field(default_factory=dict)
    
    # Search settings
    default_max_results: int = Field(50, gt=0, le=1000)
    search_timeout: int = Field(60, gt=0)
    rate_limit_delay: float = Field(1.0, ge=0.0)
    retry_attempts: int = Field(3, ge=0)
    
    # Analysis settings
    similarity_threshold: float = Field(0.7, ge=0.0, le=1.0)
    precedent_threshold: float = Field(0.8, ge=0.0, le=1.0)
    embedding_model: str = Field("sentence-transformers/legal-bert-base-uncased")
    semantic_search_enabled: bool = Field(True)
    
    # Citation settings
    default_citation_style: CitationStyle = Field(CitationStyle.BLUEBOOK)
    citation_formats: Dict[str, CitationFormat] = Field(default_factory=dict)
    validate_citations: bool = Field(True)
    
    # Output settings
    output_directory: Optional[Path] = None
    save_intermediate_results: bool = Field(True)
    export_formats: List[str] = Field(default_factory=lambda: ["json", "pdf", "docx"])
    
    # Logging settings
    log_level: str = Field("INFO")
    log_file: Optional[Path] = None
    enable_detailed_logging: bool = Field(True)


class LegalResearchAssistant:
    """
    Main class for legal research operations.
    
    Provides comprehensive legal research capabilities including:
    - Multi-database case law search
    - Precedent analysis and similarity matching
    - Legal citation parsing and validation
    - Research aggregation and synthesis
    - Legal memo generation
    """
    
    def __init__(self, config: Optional[ResearchConfig] = None):
        """Initialize the legal research assistant"""
        self.config = config or ResearchConfig()
        self.session_id = str(uuid4())
        
        # Initialize components (will be set by individual modules)
        self._case_searcher = None
        self._precedent_analyzer = None
        self._citation_processor = None
        self._research_aggregator = None
        
        # Research state
        self._current_research: Optional[ResearchSummary] = None
        self._research_history: List[ResearchSummary] = []
        
        logger.info(f"Legal Research Assistant initialized with session {self.session_id}")
    
    @property
    def case_searcher(self):
        """Lazy-loaded case law searcher"""
        if self._case_searcher is None:
            from .case_law_searcher import CaseLawSearcher
            self._case_searcher = CaseLawSearcher(self.config)
        return self._case_searcher
    
    @property
    def precedent_analyzer(self):
        """Lazy-loaded precedent analyzer"""
        if self._precedent_analyzer is None:
            from .precedent_analyzer import PrecedentAnalyzer
            self._precedent_analyzer = PrecedentAnalyzer(self.config)
        return self._precedent_analyzer
    
    @property
    def citation_processor(self):
        """Lazy-loaded citation processor"""
        if self._citation_processor is None:
            from .citation_processor import CitationProcessor
            self._citation_processor = CitationProcessor(self.config)
        return self._citation_processor
    
    @property
    def research_aggregator(self):
        """Lazy-loaded research aggregator"""
        if self._research_aggregator is None:
            from .research_aggregator import ResearchAggregator
            self._research_aggregator = ResearchAggregator(self.config)
        return self._research_aggregator
    
    def search_cases(
        self, 
        query: Union[str, SearchQuery],
        databases: Optional[List[DatabaseType]] = None
    ) -> CaseLawResults:
        """
        Search for case law across multiple databases
        
        Args:
            query: Search query string or SearchQuery object
            databases: Specific databases to search (default: all enabled)
            
        Returns:
            CaseLawResults with search results and metadata
        """
        return self.case_searcher.search(query, databases)
    
    def find_precedents(
        self,
        reference_case: Union[str, CaseOpinion],
        max_results: int = 10
    ) -> List[PrecedentMatch]:
        """
        Find similar legal precedents for a reference case
        
        Args:
            reference_case: Case name/citation or CaseOpinion object
            max_results: Maximum number of precedents to return
            
        Returns:
            List of PrecedentMatch objects with similarity analysis
        """
        return self.precedent_analyzer.find_similar(reference_case, max_results)
    
    def parse_citations(
        self,
        text: str,
        citation_style: Optional[CitationStyle] = None
    ) -> CitationMatch:
        """
        Parse and validate legal citations from text
        
        Args:
            text: Text containing legal citations
            citation_style: Target citation style for formatting
            
        Returns:
            CitationMatch with parsed citations and validation
        """
        return self.citation_processor.parse(text, citation_style)
    
    def aggregate_research(
        self,
        research_question: str,
        sources: Optional[List[str]] = None
    ) -> ResearchSummary:
        """
        Aggregate research from multiple sources and queries
        
        Args:
            research_question: Primary research question
            sources: Specific sources to include (default: all available)
            
        Returns:
            ResearchSummary with comprehensive analysis
        """
        return self.research_aggregator.aggregate(research_question, sources)
    
    def generate_memo(
        self,
        research_summary: ResearchSummary,
        memo_template: Optional[str] = None
    ) -> LegalMemo:
        """
        Generate a legal research memorandum
        
        Args:
            research_summary: Research summary to base memo on
            memo_template: Custom memo template (optional)
            
        Returns:
            LegalMemo formatted according to template
        """
        return self.research_aggregator.generate_memo(research_summary, memo_template)
    
    def shepardize_case(
        self,
        case_citation: str
    ) -> Dict[str, Any]:
        """
        Perform Shepardizing-equivalent analysis on a case
        
        Args:
            case_citation: Citation of case to analyze
            
        Returns:
            Dictionary with case treatment information
        """
        return self.precedent_analyzer.shepardize(case_citation)
    
    def get_research_history(self) -> List[ResearchSummary]:
        """Get history of research summaries for current session"""
        return self._research_history.copy()
    
    def save_research(
        self,
        research: ResearchSummary,
        filepath: Optional[Path] = None
    ) -> Path:
        """
        Save research summary to file
        
        Args:
            research: Research summary to save
            filepath: Target file path (default: auto-generated)
            
        Returns:
            Path to saved file
        """
        if filepath is None:
            if self.config.output_directory:
                filepath = self.config.output_directory / f"research_{research.id}.json"
            else:
                filepath = Path(f"research_{research.id}.json")
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(research.model_dump_json(indent=2))
        
        logger.info(f"Research saved to {filepath}")
        return filepath
    
    def load_research(self, filepath: Path) -> ResearchSummary:
        """
        Load research summary from file
        
        Args:
            filepath: Path to saved research file
            
        Returns:
            ResearchSummary object
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            data = f.read()
        
        research = ResearchSummary.model_validate_json(data)
        logger.info(f"Research loaded from {filepath}")
        return research
    
    def export_results(
        self,
        results: Union[CaseLawResults, ResearchSummary, LegalMemo],
        format_type: str = "json",
        filepath: Optional[Path] = None
    ) -> Path:
        """
        Export research results in various formats
        
        Args:
            results: Research results to export
            format_type: Export format (json, pdf, docx)
            filepath: Target file path (default: auto-generated)
            
        Returns:
            Path to exported file
        """
        if format_type not in self.config.export_formats:
            raise ValueError(f"Export format {format_type} not supported")
        
        # Implementation would depend on the specific format
        # For now, just export as JSON
        if filepath is None:
            suffix = "json" if format_type == "json" else format_type
            if self.config.output_directory:
                filepath = self.config.output_directory / f"export_{results.id}.{suffix}"
            else:
                filepath = Path(f"export_{results.id}.{suffix}")
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(results.model_dump_json(indent=2))
        
        logger.info(f"Results exported to {filepath}")
        return filepath


# Convenience functions for direct module usage

def create_research_assistant(
    config: Optional[ResearchConfig] = None
) -> LegalResearchAssistant:
    """Create a new LegalResearchAssistant instance"""
    return LegalResearchAssistant(config)


def create_default_config() -> ResearchConfig:
    """Create a default research configuration"""
    return ResearchConfig()


# Export all models and classes
__all__ = [
    # Main classes
    'LegalResearchAssistant',
    'ResearchConfig',
    
    # Data models
    'CaseLawResults',
    'SimilarCase', 
    'LegalCitation',
    'ResearchSummary',
    'CaseOpinion',
    'Precedent',
    'CitationFormat',
    'LegalDatabase',
    'SearchQuery',
    'LegalMemo',
    'DatabaseResult',
    'PrecedentMatch',
    'CitationMatch',
    
    # Enums
    'CitationType',
    'JurisdictionType',
    'DatabaseType',
    'ResearchStatus',
    'RelevanceScore', 
    'CitationStyle',
    
    # Base classes
    'BaseResearchModel',
    
    # Convenience functions
    'create_research_assistant',
    'create_default_config',
]