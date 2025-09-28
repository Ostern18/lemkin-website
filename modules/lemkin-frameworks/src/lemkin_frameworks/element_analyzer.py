"""
Element Analyzer for Legal Framework Elements.

This module provides detailed analysis of evidence against specific legal elements,
including confidence scoring, gap analysis, and evidence-to-element mapping.
"""

import re
import math
from datetime import datetime
from enum import Enum
from typing import List, Dict, Optional, Set, Tuple, Any
from uuid import UUID

from pydantic import BaseModel, Field
import numpy as np
from loguru import logger
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .core import (
    Evidence, LegalElement, ElementSatisfaction, ElementStatus,
    EvidenceType
)


class ConfidenceScore(BaseModel):
    """Detailed confidence scoring breakdown."""
    overall_confidence: float = Field(..., ge=0.0, le=1.0)
    keyword_match_score: float = Field(..., ge=0.0, le=1.0)
    semantic_similarity_score: float = Field(..., ge=0.0, le=1.0)
    evidence_strength_score: float = Field(..., ge=0.0, le=1.0)
    evidence_quantity_score: float = Field(..., ge=0.0, le=1.0)
    evidence_diversity_score: float = Field(..., ge=0.0, le=1.0)
    temporal_relevance_score: float = Field(..., ge=0.0, le=1.0)
    reliability_score: float = Field(..., ge=0.0, le=1.0)
    completeness_score: float = Field(..., ge=0.0, le=1.0)


class EvidenceRelevance(BaseModel):
    """Evidence relevance scoring for a specific legal element."""
    evidence_id: UUID
    relevance_score: float = Field(..., ge=0.0, le=1.0)
    matching_keywords: List[str]
    matching_requirements: List[str]
    evidence_type_weight: float
    reliability_factor: float
    temporal_factor: float
    content_similarity: float


class ElementAnalyzer:
    """
    Advanced analyzer for legal element satisfaction.
    
    This class implements sophisticated algorithms for analyzing how well
    evidence satisfies specific legal elements, including:
    - Keyword matching and semantic similarity
    - Evidence strength and reliability assessment
    - Gap analysis and missing evidence identification
    - Confidence scoring with detailed breakdown
    """
    
    def __init__(self):
        """Initialize the Element Analyzer."""
        self.vectorizer = TfidfVectorizer(
            stop_words='english',
            max_features=1000,
            ngram_range=(1, 3),
            lowercase=True
        )
        logger.info("ElementAnalyzer initialized with TF-IDF vectorization")
    
    def analyze_element_satisfaction(self, evidence: List[Evidence], 
                                   legal_element: LegalElement) -> ElementSatisfaction:
        """
        Analyze how well evidence satisfies a specific legal element.
        
        Args:
            evidence: List of evidence to analyze
            legal_element: The legal element to evaluate against
            
        Returns:
            ElementSatisfaction: Detailed satisfaction analysis
        """
        logger.debug(f"Analyzing satisfaction for element: {legal_element.id}")
        
        if not evidence:
            return ElementSatisfaction(
                element_id=legal_element.id,
                status=ElementStatus.INSUFFICIENT_EVIDENCE,
                confidence=0.0,
                supporting_evidence=[],
                reasoning="No evidence provided for analysis",
                gaps=["No evidence available"],
                score_breakdown={}
            )
        
        # Calculate evidence relevance for each piece of evidence
        evidence_relevances = []
        for ev in evidence:
            relevance = self._calculate_evidence_relevance(ev, legal_element)
            evidence_relevances.append(relevance)
        
        # Filter relevant evidence (relevance > 0.3)
        relevant_evidence = [
            er for er in evidence_relevances if er.relevance_score > 0.3
        ]
        
        if not relevant_evidence:
            return ElementSatisfaction(
                element_id=legal_element.id,
                status=ElementStatus.NOT_SATISFIED,
                confidence=0.0,
                supporting_evidence=[],
                reasoning="No evidence found relevant to this legal element",
                gaps=self._identify_evidence_gaps(legal_element, []),
                score_breakdown={}
            )
        
        # Calculate overall confidence score
        confidence_score = self._calculate_confidence_score(
            relevant_evidence, legal_element
        )
        
        # Determine satisfaction status
        status = self._determine_satisfaction_status(
            confidence_score.overall_confidence, relevant_evidence, legal_element
        )
        
        # Generate reasoning
        reasoning = self._generate_reasoning(
            relevant_evidence, legal_element, confidence_score, status
        )
        
        # Identify evidence gaps
        gaps = self._identify_evidence_gaps(legal_element, relevant_evidence)
        
        # Create score breakdown dictionary
        score_breakdown = self._create_score_breakdown(confidence_score)
        
        satisfaction = ElementSatisfaction(
            element_id=legal_element.id,
            status=status,
            confidence=confidence_score.overall_confidence,
            supporting_evidence=[er.evidence_id for er in relevant_evidence],
            reasoning=reasoning,
            gaps=gaps,
            score_breakdown=score_breakdown
        )
        
        logger.debug(f"Element {legal_element.id} satisfaction: {status} ({confidence_score.overall_confidence:.2f})")
        return satisfaction
    
    def _calculate_evidence_relevance(self, evidence: Evidence, 
                                    legal_element: LegalElement) -> EvidenceRelevance:
        """Calculate how relevant a piece of evidence is to a legal element."""
        
        # Keyword matching
        matching_keywords = self._find_matching_keywords(
            evidence.content + " " + evidence.title, legal_element.keywords
        )
        keyword_score = len(matching_keywords) / max(len(legal_element.keywords), 1)
        keyword_score = min(keyword_score, 1.0)  # Cap at 1.0
        
        # Requirement matching
        matching_requirements = self._find_matching_requirements(
            evidence.content, legal_element.requirements
        )
        requirement_score = len(matching_requirements) / max(len(legal_element.requirements), 1)
        requirement_score = min(requirement_score, 1.0)
        
        # Content similarity using TF-IDF
        content_similarity = self._calculate_semantic_similarity(
            evidence.content, legal_element.description + " " + " ".join(legal_element.requirements)
        )
        
        # Evidence type weighting
        evidence_type_weight = self._get_evidence_type_weight(
            evidence.evidence_type, legal_element
        )
        
        # Reliability factor
        reliability_factor = evidence.reliability_score
        
        # Temporal relevance factor
        temporal_factor = self._calculate_temporal_relevance(evidence, legal_element)
        
        # Calculate overall relevance score
        relevance_score = (
            keyword_score * 0.25 +
            requirement_score * 0.25 +
            content_similarity * 0.20 +
            evidence_type_weight * 0.15 +
            reliability_factor * 0.10 +
            temporal_factor * 0.05
        )
        
        return EvidenceRelevance(
            evidence_id=evidence.id,
            relevance_score=relevance_score,
            matching_keywords=matching_keywords,
            matching_requirements=matching_requirements,
            evidence_type_weight=evidence_type_weight,
            reliability_factor=reliability_factor,
            temporal_factor=temporal_factor,
            content_similarity=content_similarity
        )
    
    def _find_matching_keywords(self, text: str, keywords: List[str]) -> List[str]:
        """Find keywords that match in the text."""
        text_lower = text.lower()
        matching = []
        
        for keyword in keywords:
            # Use word boundaries for exact matching
            pattern = r'\b' + re.escape(keyword.lower()) + r'\b'
            if re.search(pattern, text_lower):
                matching.append(keyword)
            # Also check for partial matches with high similarity
            elif keyword.lower() in text_lower:
                matching.append(keyword)
        
        return matching
    
    def _find_matching_requirements(self, text: str, requirements: List[str]) -> List[str]:
        """Find requirements that are addressed in the text."""
        text_lower = text.lower()
        matching = []
        
        for req in requirements:
            # Extract key concepts from requirement
            req_keywords = self._extract_key_concepts(req)
            
            # Check if most key concepts are present
            matches = sum(1 for kw in req_keywords if kw.lower() in text_lower)
            if matches >= len(req_keywords) * 0.5:  # At least 50% of concepts match
                matching.append(req)
        
        return matching
    
    def _extract_key_concepts(self, text: str) -> List[str]:
        """Extract key concepts from a text."""
        # Remove common legal words that aren't substantive
        stop_legal_words = {
            'shall', 'must', 'may', 'the', 'of', 'and', 'or', 'in', 'to', 'for',
            'with', 'by', 'any', 'such', 'person', 'persons', 'individual', 'one'
        }
        
        # Simple extraction: words longer than 3 characters, not in stop list
        words = re.findall(r'\b\w{4,}\b', text.lower())
        concepts = [w for w in words if w not in stop_legal_words]
        
        # Return unique concepts
        return list(set(concepts))
    
    def _calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity between two texts using TF-IDF."""
        try:
            # Combine texts for vectorization
            texts = [text1, text2]
            
            # Handle empty texts
            if not text1.strip() or not text2.strip():
                return 0.0
            
            # Vectorize texts
            tfidf_matrix = self.vectorizer.fit_transform(texts)
            
            # Calculate cosine similarity
            similarity_matrix = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
            similarity = similarity_matrix[0][0]
            
            return float(similarity)
        
        except Exception as e:
            logger.warning(f"Error calculating semantic similarity: {e}")
            return 0.0
    
    def _get_evidence_type_weight(self, evidence_type: EvidenceType, 
                                 legal_element: LegalElement) -> float:
        """Get weighting factor based on evidence type relevance to legal element."""
        
        # Define evidence type weights for different legal contexts
        weights = {
            EvidenceType.DOCUMENT: 0.9,  # High weight for official documents
            EvidenceType.TESTIMONY: 0.8,  # High weight for witness testimony
            EvidenceType.EXPERT_REPORT: 0.9,  # High weight for expert analysis
            EvidenceType.FORENSIC: 0.85,  # High weight for forensic evidence
            EvidenceType.PHOTO: 0.7,  # Medium-high weight for visual evidence
            EvidenceType.VIDEO: 0.8,  # High weight for video evidence
            EvidenceType.AUDIO: 0.7,  # Medium-high weight for audio
            EvidenceType.GEOSPATIAL: 0.7,  # Medium-high for location evidence
            EvidenceType.DIGITAL: 0.75,  # Medium-high for digital evidence
            EvidenceType.PHYSICAL: 0.8   # High weight for physical evidence
        }
        
        base_weight = weights.get(evidence_type, 0.5)
        
        # Adjust weight based on legal element context
        element_context = (legal_element.title + " " + legal_element.description).lower()
        
        # Boost certain evidence types for specific contexts
        if evidence_type == EvidenceType.TESTIMONY and any(word in element_context for word in ['witness', 'victim', 'testimony']):
            base_weight += 0.1
        elif evidence_type == EvidenceType.DOCUMENT and any(word in element_context for word in ['official', 'order', 'policy']):
            base_weight += 0.1
        elif evidence_type == EvidenceType.PHOTO and any(word in element_context for word in ['damage', 'destruction', 'injury']):
            base_weight += 0.1
        elif evidence_type == EvidenceType.GEOSPATIAL and any(word in element_context for word in ['location', 'area', 'territory']):
            base_weight += 0.1
        
        return min(base_weight, 1.0)  # Cap at 1.0
    
    def _calculate_temporal_relevance(self, evidence: Evidence, 
                                    legal_element: LegalElement) -> float:
        """Calculate temporal relevance of evidence."""
        # If no date information, assume moderate relevance
        if not evidence.incident_date:
            return 0.7
        
        # Evidence from incident is most relevant
        incident_date = evidence.incident_date
        current_date = datetime.now()
        
        # Calculate days since incident
        days_diff = (current_date - incident_date).days
        
        # Fresher evidence generally more relevant, but not too heavily weighted
        if days_diff <= 30:
            return 1.0
        elif days_diff <= 365:
            return 0.9
        elif days_diff <= 1825:  # 5 years
            return 0.8
        else:
            return 0.7
    
    def _calculate_confidence_score(self, relevant_evidence: List[EvidenceRelevance],
                                  legal_element: LegalElement) -> ConfidenceScore:
        """Calculate comprehensive confidence score."""
        
        if not relevant_evidence:
            return ConfidenceScore(
                overall_confidence=0.0,
                keyword_match_score=0.0,
                semantic_similarity_score=0.0,
                evidence_strength_score=0.0,
                evidence_quantity_score=0.0,
                evidence_diversity_score=0.0,
                temporal_relevance_score=0.0,
                reliability_score=0.0,
                completeness_score=0.0
            )
        
        # Keyword match score (average of matching keywords ratio)
        keyword_scores = [
            len(er.matching_keywords) / max(len(legal_element.keywords), 1)
            for er in relevant_evidence
        ]
        keyword_match_score = min(sum(keyword_scores) / len(keyword_scores), 1.0)
        
        # Semantic similarity score (average)
        semantic_similarity_score = sum(er.content_similarity for er in relevant_evidence) / len(relevant_evidence)
        
        # Evidence strength score (weighted by relevance)
        strength_scores = [er.relevance_score * er.reliability_factor for er in relevant_evidence]
        evidence_strength_score = sum(strength_scores) / len(strength_scores)
        
        # Evidence quantity score (logarithmic scaling)
        evidence_count = len(relevant_evidence)
        # Optimal around 5-10 pieces of evidence
        if evidence_count == 0:
            evidence_quantity_score = 0.0
        elif evidence_count <= 3:
            evidence_quantity_score = evidence_count / 3.0 * 0.7  # Up to 70% for 3 pieces
        elif evidence_count <= 8:
            evidence_quantity_score = 0.7 + (evidence_count - 3) / 5.0 * 0.3  # 70-100% for 4-8 pieces
        else:
            # Diminishing returns after 8 pieces, but still beneficial
            evidence_quantity_score = 1.0 - 0.1 * math.log10(evidence_count - 7)
            evidence_quantity_score = max(evidence_quantity_score, 0.9)
        
        # Evidence diversity score (different types of evidence)
        evidence_ids = [er.evidence_id for er in relevant_evidence]
        # This would require accessing the original evidence list - simplified for now
        evidence_diversity_score = min(evidence_count / 3.0, 1.0)  # Simplified
        
        # Temporal relevance score (average)
        temporal_relevance_score = sum(er.temporal_factor for er in relevant_evidence) / len(relevant_evidence)
        
        # Reliability score (average)
        reliability_score = sum(er.reliability_factor for er in relevant_evidence) / len(relevant_evidence)
        
        # Completeness score (how many requirements are addressed)
        requirement_coverage = len(set().union(*[er.matching_requirements for er in relevant_evidence]))
        completeness_score = min(requirement_coverage / max(len(legal_element.requirements), 1), 1.0)
        
        # Overall confidence (weighted combination)
        overall_confidence = (
            keyword_match_score * 0.15 +
            semantic_similarity_score * 0.15 +
            evidence_strength_score * 0.20 +
            evidence_quantity_score * 0.15 +
            evidence_diversity_score * 0.10 +
            temporal_relevance_score * 0.05 +
            reliability_score * 0.10 +
            completeness_score * 0.10
        )
        
        return ConfidenceScore(
            overall_confidence=overall_confidence,
            keyword_match_score=keyword_match_score,
            semantic_similarity_score=semantic_similarity_score,
            evidence_strength_score=evidence_strength_score,
            evidence_quantity_score=evidence_quantity_score,
            evidence_diversity_score=evidence_diversity_score,
            temporal_relevance_score=temporal_relevance_score,
            reliability_score=reliability_score,
            completeness_score=completeness_score
        )
    
    def _determine_satisfaction_status(self, confidence: float, 
                                     relevant_evidence: List[EvidenceRelevance],
                                     legal_element: LegalElement) -> ElementStatus:
        """Determine the satisfaction status based on confidence and evidence."""
        
        if confidence >= 0.8:
            return ElementStatus.SATISFIED
        elif confidence >= 0.5:
            return ElementStatus.PARTIALLY_SATISFIED
        elif confidence >= 0.2:
            return ElementStatus.INSUFFICIENT_EVIDENCE
        else:
            return ElementStatus.NOT_SATISFIED
    
    def _generate_reasoning(self, relevant_evidence: List[EvidenceRelevance],
                          legal_element: LegalElement, confidence_score: ConfidenceScore,
                          status: ElementStatus) -> str:
        """Generate human-readable reasoning for the satisfaction assessment."""
        
        reasoning_parts = []
        
        # Status explanation
        status_explanations = {
            ElementStatus.SATISFIED: "The available evidence strongly supports satisfaction of this legal element.",
            ElementStatus.PARTIALLY_SATISFIED: "The available evidence provides moderate support for this legal element.",
            ElementStatus.INSUFFICIENT_EVIDENCE: "There is some evidence relevant to this element, but additional evidence would strengthen the assessment.",
            ElementStatus.NOT_SATISFIED: "The available evidence does not adequately support this legal element."
        }
        reasoning_parts.append(status_explanations[status])
        
        # Evidence summary
        if relevant_evidence:
            reasoning_parts.append(f"Analysis based on {len(relevant_evidence)} relevant piece(s) of evidence.")
            
            # Highlight strongest evidence
            best_evidence = max(relevant_evidence, key=lambda x: x.relevance_score)
            reasoning_parts.append(f"Strongest evidence shows {best_evidence.relevance_score:.1%} relevance.")
            
            # Keyword matches
            all_keywords = set().union(*[er.matching_keywords for er in relevant_evidence])
            if all_keywords:
                reasoning_parts.append(f"Key matching terms: {', '.join(list(all_keywords)[:5])}.")
        
        # Confidence breakdown
        if confidence_score.overall_confidence >= 0.7:
            reasoning_parts.append("High confidence assessment.")
        elif confidence_score.overall_confidence >= 0.4:
            reasoning_parts.append("Moderate confidence assessment.")
        else:
            reasoning_parts.append("Low confidence assessment.")
        
        # Specific weaknesses
        if confidence_score.completeness_score < 0.5:
            reasoning_parts.append("Some legal requirements may not be fully addressed by available evidence.")
        
        if confidence_score.reliability_score < 0.6:
            reasoning_parts.append("Evidence reliability could be strengthened.")
        
        return " ".join(reasoning_parts)
    
    def _identify_evidence_gaps(self, legal_element: LegalElement, 
                               relevant_evidence: List[EvidenceRelevance]) -> List[str]:
        """Identify specific evidence gaps for the legal element."""
        gaps = []
        
        # Check requirement coverage
        covered_requirements = set().union(*[er.matching_requirements for er in relevant_evidence]) if relevant_evidence else set()
        uncovered_requirements = set(legal_element.requirements) - covered_requirements
        
        for req in uncovered_requirements:
            gaps.append(f"Evidence needed for requirement: {req[:100]}...")
        
        # Check keyword coverage
        covered_keywords = set().union(*[er.matching_keywords for er in relevant_evidence]) if relevant_evidence else set()
        uncovered_keywords = set(legal_element.keywords) - covered_keywords
        
        if uncovered_keywords:
            gaps.append(f"Evidence lacking key terms: {', '.join(list(uncovered_keywords)[:3])}")
        
        # Evidence type diversity
        if not relevant_evidence:
            gaps.append("No relevant evidence identified")
        elif len(relevant_evidence) < 3:
            gaps.append("Additional corroborating evidence would strengthen assessment")
        
        # Reliability concerns
        if relevant_evidence:
            low_reliability = [er for er in relevant_evidence if er.reliability_factor < 0.6]
            if low_reliability:
                gaps.append("Some evidence has reliability concerns - seek additional corroboration")
        
        # Suggest specific evidence types based on legal element
        element_context = (legal_element.title + " " + legal_element.description).lower()
        
        if "witness" in element_context or "testimony" in element_context:
            gaps.append("Witness testimony would strengthen this element")
        
        if "document" in element_context or "official" in element_context:
            gaps.append("Official documentation would support this element")
        
        if "physical" in element_context or "damage" in element_context:
            gaps.append("Physical evidence or documentation of damage would be valuable")
        
        return gaps[:10]  # Limit to most important gaps
    
    def _create_score_breakdown(self, confidence_score: ConfidenceScore) -> Dict[str, float]:
        """Create a dictionary breakdown of confidence scores."""
        return {
            "overall_confidence": confidence_score.overall_confidence,
            "keyword_match": confidence_score.keyword_match_score,
            "semantic_similarity": confidence_score.semantic_similarity_score,
            "evidence_strength": confidence_score.evidence_strength_score,
            "evidence_quantity": confidence_score.evidence_quantity_score,
            "evidence_diversity": confidence_score.evidence_diversity_score,
            "temporal_relevance": confidence_score.temporal_relevance_score,
            "reliability": confidence_score.reliability_score,
            "completeness": confidence_score.completeness_score
        }


class AdvancedElementAnalyzer(ElementAnalyzer):
    """
    Advanced version of ElementAnalyzer with machine learning capabilities.
    
    This class extends the base ElementAnalyzer with more sophisticated
    algorithms including pattern recognition and learning from past analyses.
    """
    
    def __init__(self):
        """Initialize the Advanced Element Analyzer."""
        super().__init__()
        self.analysis_history: List[Dict[str, Any]] = []
        logger.info("AdvancedElementAnalyzer initialized")
    
    def learn_from_analysis(self, evidence: List[Evidence], legal_element: LegalElement,
                          satisfaction: ElementSatisfaction, feedback_score: float):
        """
        Learn from analysis results to improve future assessments.
        
        Args:
            evidence: Original evidence analyzed
            legal_element: Legal element analyzed
            satisfaction: Analysis results
            feedback_score: Expert feedback score (0-1) on analysis quality
        """
        analysis_record = {
            "element_id": legal_element.id,
            "evidence_count": len(evidence),
            "confidence_predicted": satisfaction.confidence,
            "status_predicted": satisfaction.status,
            "feedback_score": feedback_score,
            "timestamp": datetime.now(),
            "keywords_used": legal_element.keywords,
            "requirements_count": len(legal_element.requirements)
        }
        
        self.analysis_history.append(analysis_record)
        logger.info(f"Recorded analysis feedback for element {legal_element.id}: {feedback_score}")
    
    def get_similar_element_patterns(self, legal_element: LegalElement) -> List[Dict[str, Any]]:
        """
        Find patterns from similar legal elements analyzed previously.
        
        Args:
            legal_element: Legal element to find patterns for
            
        Returns:
            List of similar analysis patterns
        """
        similar_patterns = []
        
        for record in self.analysis_history:
            if record["element_id"] == legal_element.id:
                similar_patterns.append(record)
        
        # Sort by feedback score (best performing analyses first)
        similar_patterns.sort(key=lambda x: x["feedback_score"], reverse=True)
        
        return similar_patterns[:5]  # Return top 5 similar patterns


def create_element_analyzer(advanced: bool = False) -> ElementAnalyzer:
    """
    Factory function to create appropriate element analyzer.
    
    Args:
        advanced: Whether to create advanced analyzer with ML capabilities
        
    Returns:
        ElementAnalyzer instance
    """
    if advanced:
        return AdvancedElementAnalyzer()
    else:
        return ElementAnalyzer()