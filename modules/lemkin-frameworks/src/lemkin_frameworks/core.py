"""
Core classes and data models for the Legal Framework Mapper.

This module provides the main LegalFrameworkMapper class and all Pydantic data models
used throughout the legal analysis system.
"""

from datetime import datetime
from enum import Enum
from typing import List, Dict, Optional, Any, Union
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, validator
import pandas as pd
from loguru import logger


class EvidenceType(str, Enum):
    """Types of evidence that can be analyzed."""
    DOCUMENT = "document"
    TESTIMONY = "testimony"
    PHYSICAL = "physical"
    DIGITAL = "digital"
    PHOTO = "photo"
    VIDEO = "video"
    AUDIO = "audio"
    GEOSPATIAL = "geospatial"
    FORENSIC = "forensic"
    EXPERT_REPORT = "expert_report"


class ConfidenceLevel(str, Enum):
    """Confidence levels for legal assessments."""
    VERY_HIGH = "very_high"  # 90-100%
    HIGH = "high"           # 75-89%
    MODERATE = "moderate"   # 50-74%
    LOW = "low"            # 25-49%
    VERY_LOW = "very_low"  # 0-24%


class LegalFramework(str, Enum):
    """Supported legal frameworks."""
    ROME_STATUTE = "rome_statute"
    GENEVA_CONVENTIONS = "geneva_conventions"
    ICCPR = "iccpr"
    ECHR = "echr"
    ACHR = "achr"
    ACHPR = "achpr"
    UDHR = "udhr"


class ElementStatus(str, Enum):
    """Status of legal element satisfaction."""
    SATISFIED = "satisfied"
    PARTIALLY_SATISFIED = "partially_satisfied"
    NOT_SATISFIED = "not_satisfied"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


class ViolationType(str, Enum):
    """Types of legal violations."""
    GRAVE_BREACH = "grave_breach"
    SERIOUS_VIOLATION = "serious_violation"
    MINOR_VIOLATION = "minor_violation"
    TECHNICAL_VIOLATION = "technical_violation"


class SatisfactionLevel(str, Enum):
    """Levels of element satisfaction for scoring."""
    FULLY_SATISFIED = "fully_satisfied"
    SUBSTANTIALLY_SATISFIED = "substantially_satisfied"
    PARTIALLY_SATISFIED = "partially_satisfied"
    MINIMALLY_SATISFIED = "minimally_satisfied"
    NOT_SATISFIED = "not_satisfied"


class FrameworkConfig(BaseModel):
    """Configuration settings for framework analysis."""
    confidence_threshold: float = Field(0.6, ge=0.0, le=1.0, description="Minimum confidence for violations")
    include_weak_evidence: bool = Field(True, description="Include low-reliability evidence")
    require_corroboration: bool = Field(False, description="Require multiple sources for high-confidence assessments")
    enable_advanced_analytics: bool = Field(False, description="Enable machine learning features")
    temporal_weight: float = Field(0.1, ge=0.0, le=1.0, description="Weight for temporal relevance")
    reliability_weight: float = Field(0.2, ge=0.0, le=1.0, description="Weight for evidence reliability")
    keyword_weight: float = Field(0.25, ge=0.0, le=1.0, description="Weight for keyword matching")
    semantic_weight: float = Field(0.2, ge=0.0, le=1.0, description="Weight for semantic similarity")
    max_elements_analyzed: Optional[int] = Field(None, description="Maximum number of elements to analyze")
    output_detailed_reasoning: bool = Field(True, description="Include detailed reasoning in outputs")
    generate_visualizations: bool = Field(False, description="Generate analysis visualizations")


class Evidence(BaseModel):
    """
    Represents a piece of evidence that can be analyzed against legal frameworks.
    """
    id: UUID = Field(default_factory=uuid4)
    title: str = Field(..., description="Title or identifier of the evidence")
    content: str = Field(..., description="Main content of the evidence")
    evidence_type: EvidenceType = Field(..., description="Type of evidence")
    source: str = Field(..., description="Source of the evidence")
    date_collected: Optional[datetime] = Field(None, description="Date evidence was collected")
    incident_date: Optional[datetime] = Field(None, description="Date of the incident")
    location: Optional[str] = Field(None, description="Location where evidence originates")
    tags: List[str] = Field(default_factory=list, description="Tags for categorization")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    reliability_score: float = Field(0.0, ge=0.0, le=1.0, description="Reliability score (0-1)")

    @validator('reliability_score')
    def validate_reliability(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError('Reliability score must be between 0.0 and 1.0')
        return v


class LegalElement(BaseModel):
    """
    Represents a specific legal element within a framework.
    """
    id: str = Field(..., description="Unique identifier for the element")
    framework: LegalFramework = Field(..., description="Legal framework this element belongs to")
    article: Optional[str] = Field(None, description="Article number or reference")
    title: str = Field(..., description="Title of the legal element")
    description: str = Field(..., description="Detailed description of the element")
    requirements: List[str] = Field(default_factory=list, description="Specific requirements")
    keywords: List[str] = Field(default_factory=list, description="Keywords for matching")
    parent_element: Optional[str] = Field(None, description="Parent element ID if hierarchical")
    sub_elements: List[str] = Field(default_factory=list, description="Sub-element IDs")
    citation: str = Field(..., description="Full legal citation")
    precedents: List[str] = Field(default_factory=list, description="Relevant legal precedents")


class ElementSatisfaction(BaseModel):
    """
    Represents the satisfaction level of a legal element by evidence.
    """
    element_id: str = Field(..., description="ID of the legal element")
    status: ElementStatus = Field(..., description="Satisfaction status")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score (0-1)")
    supporting_evidence: List[UUID] = Field(default_factory=list, description="Supporting evidence IDs")
    reasoning: str = Field(..., description="Explanation of the assessment")
    gaps: List[str] = Field(default_factory=list, description="Identified evidence gaps")
    score_breakdown: Dict[str, float] = Field(default_factory=dict, description="Detailed scoring")


class GapAnalysis(BaseModel):
    """
    Analysis of evidence gaps for legal elements.
    """
    framework: LegalFramework = Field(..., description="Legal framework analyzed")
    missing_elements: List[str] = Field(default_factory=list, description="Completely missing elements")
    weak_elements: List[str] = Field(default_factory=list, description="Weakly supported elements")
    evidence_needs: Dict[str, List[str]] = Field(default_factory=dict, description="Specific evidence needed")
    recommendations: List[str] = Field(default_factory=list, description="Investigation recommendations")
    priority_score: float = Field(0.0, ge=0.0, le=1.0, description="Priority for addressing gaps")


class FrameworkAnalysis(BaseModel):
    """
    Complete analysis of evidence against a specific legal framework.
    """
    id: UUID = Field(default_factory=uuid4)
    framework: LegalFramework = Field(..., description="Legal framework analyzed")
    analysis_date: datetime = Field(default_factory=datetime.now)
    evidence_count: int = Field(..., description="Number of pieces of evidence analyzed")
    elements_analyzed: List[str] = Field(..., description="Legal elements that were analyzed")
    element_satisfactions: List[ElementSatisfaction] = Field(..., description="Element satisfaction results")
    overall_confidence: float = Field(..., ge=0.0, le=1.0, description="Overall confidence score")
    gap_analysis: GapAnalysis = Field(..., description="Analysis of evidence gaps")
    summary: str = Field(..., description="Executive summary of findings")
    violations_identified: List[str] = Field(default_factory=list, description="Potential violations found")
    recommendations: List[str] = Field(default_factory=list, description="Legal recommendations")


class LegalAssessment(BaseModel):
    """
    Comprehensive legal assessment across multiple frameworks.
    """
    id: UUID = Field(default_factory=uuid4)
    title: str = Field(..., description="Title of the assessment")
    description: str = Field(..., description="Description of the case or incident")
    assessment_date: datetime = Field(default_factory=datetime.now)
    frameworks_analyzed: List[LegalFramework] = Field(..., description="Frameworks included in assessment")
    framework_analyses: List[FrameworkAnalysis] = Field(..., description="Individual framework analyses")
    cross_framework_findings: Dict[str, Any] = Field(default_factory=dict, description="Cross-cutting findings")
    overall_assessment: str = Field(..., description="Overall legal assessment")
    jurisdiction_recommendations: List[str] = Field(default_factory=list, description="Jurisdiction recommendations")
    strength_of_case: ConfidenceLevel = Field(..., description="Overall strength assessment")
    next_steps: List[str] = Field(default_factory=list, description="Recommended next steps")
    generated_citations: List[str] = Field(default_factory=list, description="Generated legal citations")


class LegalFrameworkMapper:
    """
    Main class for mapping evidence to legal framework elements.
    
    This class coordinates the analysis of evidence against various international
    legal frameworks and generates comprehensive assessments.
    """
    
    def __init__(self, config: Optional[FrameworkConfig] = None):
        """
        Initialize the Legal Framework Mapper.
        
        Args:
            config: Configuration settings for analysis behavior
        """
        self.config = config or FrameworkConfig()
        self.frameworks_loaded = set()
        self.legal_elements = {}
        self.analyzers = {}
        logger.info("LegalFrameworkMapper initialized with configuration")
        logger.debug(f"Configuration: {self.config}")
    
    def update_configuration(self, config: FrameworkConfig):
        """
        Update the configuration settings.
        
        Args:
            config: New configuration settings
        """
        self.config = config
        logger.info("Configuration updated")
        
        # Update existing analyzers with new configuration
        for analyzer in self.analyzers.values():
            if hasattr(analyzer, 'update_config'):
                analyzer.update_config(config)
    
    def load_framework(self, framework: LegalFramework) -> None:
        """
        Load a specific legal framework for analysis.
        
        Args:
            framework: The legal framework to load
        """
        logger.info(f"Loading framework: {framework.value}")
        
        if framework in self.frameworks_loaded:
            logger.warning(f"Framework {framework.value} already loaded")
            return
            
        # Import and initialize framework-specific analyzers
        if framework == LegalFramework.ROME_STATUTE:
            from .rome_statute import RomeStatuteAnalyzer
            self.analyzers[framework] = RomeStatuteAnalyzer()
        elif framework == LegalFramework.GENEVA_CONVENTIONS:
            from .geneva_conventions import GenevaAnalyzer
            self.analyzers[framework] = GenevaAnalyzer()
        elif framework in [LegalFramework.ICCPR, LegalFramework.ECHR, 
                          LegalFramework.ACHR, LegalFramework.ACHPR, LegalFramework.UDHR]:
            from .human_rights_frameworks import HumanRightsAnalyzer
            self.analyzers[framework] = HumanRightsAnalyzer(framework)
        else:
            raise ValueError(f"Unsupported framework: {framework.value}")
            
        self.frameworks_loaded.add(framework)
        logger.info(f"Framework {framework.value} loaded successfully")
    
    def map_to_legal_framework(self, evidence: List[Evidence], 
                              framework: LegalFramework) -> FrameworkAnalysis:
        """
        Map evidence to a specific legal framework.
        
        Args:
            evidence: List of evidence to analyze
            framework: Target legal framework
            
        Returns:
            FrameworkAnalysis: Complete analysis results
        """
        logger.info(f"Mapping {len(evidence)} pieces of evidence to {framework.value}")
        
        if framework not in self.frameworks_loaded:
            self.load_framework(framework)
            
        analyzer = self.analyzers[framework]
        return analyzer.analyze(evidence)
    
    def generate_legal_assessment(self, evidence: List[Evidence],
                                 frameworks: List[LegalFramework],
                                 title: str = "Legal Assessment",
                                 description: str = "") -> LegalAssessment:
        """
        Generate a comprehensive legal assessment across multiple frameworks.
        
        Args:
            evidence: Evidence to analyze
            frameworks: Legal frameworks to include in assessment
            title: Title for the assessment
            description: Description of the case
            
        Returns:
            LegalAssessment: Comprehensive assessment results
        """
        logger.info(f"Generating legal assessment for {len(frameworks)} frameworks")
        
        # Analyze each framework
        framework_analyses = []
        for framework in frameworks:
            analysis = self.map_to_legal_framework(evidence, framework)
            framework_analyses.append(analysis)
        
        # Generate cross-framework findings
        cross_findings = self._analyze_cross_framework_patterns(framework_analyses)
        
        # Determine overall strength
        overall_confidence = sum(a.overall_confidence for a in framework_analyses) / len(framework_analyses)
        strength_of_case = self._confidence_to_level(overall_confidence)
        
        # Generate assessment
        assessment = LegalAssessment(
            title=title,
            description=description,
            frameworks_analyzed=frameworks,
            framework_analyses=framework_analyses,
            cross_framework_findings=cross_findings,
            overall_assessment=self._generate_overall_assessment(framework_analyses),
            jurisdiction_recommendations=self._generate_jurisdiction_recommendations(framework_analyses),
            strength_of_case=strength_of_case,
            next_steps=self._generate_next_steps(framework_analyses),
            generated_citations=self._generate_citations(framework_analyses)
        )
        
        logger.info(f"Legal assessment generated with {len(framework_analyses)} framework analyses")
        return assessment
    
    def _analyze_cross_framework_patterns(self, analyses: List[FrameworkAnalysis]) -> Dict[str, Any]:
        """Analyze patterns across multiple framework analyses."""
        patterns = {
            "common_violations": [],
            "overlapping_evidence": [],
            "jurisdictional_conflicts": [],
            "reinforcing_elements": []
        }
        
        # Find common violations across frameworks
        all_violations = [v for analysis in analyses for v in analysis.violations_identified]
        violation_counts = pd.Series(all_violations).value_counts()
        patterns["common_violations"] = violation_counts[violation_counts > 1].index.tolist()
        
        # Analyze evidence overlap
        evidence_usage = {}
        for analysis in analyses:
            for satisfaction in analysis.element_satisfactions:
                for evidence_id in satisfaction.supporting_evidence:
                    if evidence_id not in evidence_usage:
                        evidence_usage[evidence_id] = []
                    evidence_usage[evidence_id].append(analysis.framework.value)
        
        patterns["overlapping_evidence"] = [
            {"evidence_id": str(eid), "frameworks": frameworks}
            for eid, frameworks in evidence_usage.items()
            if len(frameworks) > 1
        ]
        
        return patterns
    
    def _confidence_to_level(self, confidence: float) -> ConfidenceLevel:
        """Convert numerical confidence to confidence level."""
        if confidence >= 0.9:
            return ConfidenceLevel.VERY_HIGH
        elif confidence >= 0.75:
            return ConfidenceLevel.HIGH
        elif confidence >= 0.5:
            return ConfidenceLevel.MODERATE
        elif confidence >= 0.25:
            return ConfidenceLevel.LOW
        else:
            return ConfidenceLevel.VERY_LOW
    
    def _generate_overall_assessment(self, analyses: List[FrameworkAnalysis]) -> str:
        """Generate overall assessment summary."""
        total_violations = sum(len(a.violations_identified) for a in analyses)
        avg_confidence = sum(a.overall_confidence for a in analyses) / len(analyses)
        
        assessment = f"Analysis of {len(analyses)} legal frameworks reveals "
        assessment += f"{total_violations} potential violations with an average "
        assessment += f"confidence of {avg_confidence:.1%}. "
        
        if avg_confidence >= 0.75:
            assessment += "The evidence strongly supports potential legal violations."
        elif avg_confidence >= 0.5:
            assessment += "The evidence provides moderate support for potential violations."
        else:
            assessment += "Additional evidence may be needed to strengthen the case."
        
        return assessment
    
    def _generate_jurisdiction_recommendations(self, analyses: List[FrameworkAnalysis]) -> List[str]:
        """Generate jurisdiction recommendations based on analyses."""
        recommendations = []
        
        # Check for ICC jurisdiction (Rome Statute)
        for analysis in analyses:
            if analysis.framework == LegalFramework.ROME_STATUTE and analysis.violations_identified:
                if analysis.overall_confidence >= 0.6:
                    recommendations.append("International Criminal Court (ICC) - Strong case for jurisdiction")
                else:
                    recommendations.append("International Criminal Court (ICC) - Consider strengthening evidence")
        
        # Check for regional courts
        human_rights_frameworks = [
            LegalFramework.ECHR, LegalFramework.ACHR, LegalFramework.ACHPR
        ]
        for analysis in analyses:
            if analysis.framework in human_rights_frameworks and analysis.violations_identified:
                court_name = {
                    LegalFramework.ECHR: "European Court of Human Rights",
                    LegalFramework.ACHR: "Inter-American Court of Human Rights",
                    LegalFramework.ACHPR: "African Court on Human and Peoples' Rights"
                }[analysis.framework]
                recommendations.append(f"{court_name} - Regional jurisdiction applicable")
        
        return recommendations
    
    def _generate_next_steps(self, analyses: List[FrameworkAnalysis]) -> List[str]:
        """Generate recommended next steps."""
        next_steps = []
        
        # Collect all recommendations from individual analyses
        all_recommendations = [rec for analysis in analyses for rec in analysis.recommendations]
        
        # Add assessment-specific recommendations
        next_steps.extend([
            "Review evidence gaps identified in individual framework analyses",
            "Consult with legal experts specializing in relevant jurisdictions",
            "Consider additional evidence collection based on gap analysis",
            "Prepare preliminary case documentation",
            "Assess victim and witness protection needs"
        ])
        
        # Add unique recommendations from framework analyses
        unique_recommendations = list(set(all_recommendations))
        next_steps.extend(unique_recommendations)
        
        return next_steps
    
    def _generate_citations(self, analyses: List[FrameworkAnalysis]) -> List[str]:
        """Generate legal citations from all analyses."""
        citations = []
        
        for analysis in analyses:
            framework_name = analysis.framework.value.replace("_", " ").title()
            for violation in analysis.violations_identified:
                citations.append(f"{framework_name}: {violation}")
        
        return citations


# Utility functions for the main API
def map_to_legal_framework(evidence: List[Evidence], framework: str) -> FrameworkAnalysis:
    """
    Convenience function to map evidence to a legal framework.
    
    Args:
        evidence: List of evidence to analyze
        framework: Framework name as string
        
    Returns:
        FrameworkAnalysis: Analysis results
    """
    mapper = LegalFrameworkMapper()
    framework_enum = LegalFramework(framework.lower())
    return mapper.map_to_legal_framework(evidence, framework_enum)


def generate_legal_assessment(analyses: List[FrameworkAnalysis]) -> LegalAssessment:
    """
    Generate a legal assessment from existing framework analyses.
    
    Args:
        analyses: List of framework analyses
        
    Returns:
        LegalAssessment: Comprehensive assessment
    """
    mapper = LegalFrameworkMapper()
    frameworks = [analysis.framework for analysis in analyses]
    
    # Create a basic assessment from existing analyses
    overall_confidence = sum(a.overall_confidence for a in analyses) / len(analyses)
    strength_of_case = mapper._confidence_to_level(overall_confidence)
    
    return LegalAssessment(
        title="Generated Legal Assessment",
        description="Assessment generated from framework analyses",
        frameworks_analyzed=frameworks,
        framework_analyses=analyses,
        cross_framework_findings=mapper._analyze_cross_framework_patterns(analyses),
        overall_assessment=mapper._generate_overall_assessment(analyses),
        jurisdiction_recommendations=mapper._generate_jurisdiction_recommendations(analyses),
        strength_of_case=strength_of_case,
        next_steps=mapper._generate_next_steps(analyses),
        generated_citations=mapper._generate_citations(analyses)
    )