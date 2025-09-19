"""
Confidence assessment and human review trigger system for legal document classification.

This module provides sophisticated confidence scoring and determines when human review
is required based on multiple factors including prediction confidence, document
characteristics, and legal requirements.
"""

import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
from dataclasses import dataclass
import numpy as np
from pydantic import BaseModel, Field, validator

from .legal_taxonomy import DocumentType, LegalDomain, LegalDocumentCategory

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ConfidenceLevel(str, Enum):
    """Confidence levels for classification results"""
    VERY_HIGH = "very_high"      # 0.9 - 1.0
    HIGH = "high"                # 0.8 - 0.9
    MEDIUM = "medium"            # 0.6 - 0.8
    LOW = "low"                  # 0.4 - 0.6
    VERY_LOW = "very_low"        # 0.0 - 0.4


class ReviewTrigger(str, Enum):
    """Reasons why human review might be triggered"""
    LOW_CONFIDENCE = "low_confidence"
    AMBIGUOUS_CLASSIFICATION = "ambiguous_classification"
    SENSITIVE_CONTENT = "sensitive_content"
    CRITICAL_DOCUMENT_TYPE = "critical_document_type"
    REGULATORY_REQUIREMENT = "regulatory_requirement"
    QUALITY_THRESHOLD = "quality_threshold"
    OUTLIER_DETECTION = "outlier_detection"
    MULTI_CLASS_UNCERTAINTY = "multi_class_uncertainty"
    DOMAIN_MISMATCH = "domain_mismatch"
    LENGTH_ANOMALY = "length_anomaly"


class ScoreThresholds(BaseModel):
    """Configurable thresholds for confidence assessment"""
    
    # Confidence score thresholds
    very_high_threshold: float = Field(default=0.9, ge=0.0, le=1.0)
    high_threshold: float = Field(default=0.8, ge=0.0, le=1.0)
    medium_threshold: float = Field(default=0.6, ge=0.0, le=1.0)
    low_threshold: float = Field(default=0.4, ge=0.0, le=1.0)
    
    # Review trigger thresholds
    review_confidence_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    ambiguity_threshold: float = Field(default=0.3, ge=0.0, le=1.0)  # Diff between top 2 predictions
    outlier_threshold: float = Field(default=2.0, ge=0.0)  # Standard deviations
    
    # Document characteristics thresholds
    min_document_length: int = Field(default=50, ge=1)
    max_document_length: int = Field(default=100000, ge=1)
    
    @validator('high_threshold', 'medium_threshold', 'low_threshold')
    def validate_threshold_order(cls, v, values):
        if 'very_high_threshold' in values and v >= values['very_high_threshold']:
            raise ValueError("Thresholds must be in descending order")
        return v


class ConfidenceAssessment(BaseModel):
    """Comprehensive confidence assessment for a classification result"""
    
    # Basic confidence metrics
    primary_confidence: float = Field(description="Primary prediction confidence score")
    confidence_level: ConfidenceLevel = Field(description="Categorical confidence level")
    
    # Detailed scoring
    prediction_entropy: float = Field(description="Entropy of prediction distribution")
    class_separation: float = Field(description="Difference between top 2 predictions")
    prediction_stability: float = Field(description="Stability across multiple runs")
    
    # Review assessment
    requires_review: bool = Field(description="Whether human review is recommended")
    review_priority: str = Field(description="Priority level: low, medium, high, critical")
    review_triggers: List[ReviewTrigger] = Field(description="Specific reasons for review")
    
    # Quality indicators
    quality_score: float = Field(description="Overall quality score (0-1)")
    reliability_indicators: Dict[str, float] = Field(description="Various reliability metrics")
    
    # Context-specific factors
    document_complexity: float = Field(description="Estimated document complexity")
    domain_confidence: float = Field(description="Confidence in legal domain assignment")
    linguistic_quality: float = Field(description="Quality of linguistic features")
    
    # Metadata
    assessment_timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    assessment_version: str = Field(default="1.0", description="Version of assessment algorithm")
    
    class Config:
        use_enum_values = True
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


@dataclass
class ClassificationContext:
    """Context information for confidence assessment"""
    document_length: int
    document_type_prediction: DocumentType
    legal_domain: LegalDomain
    probability_distribution: Dict[str, float]
    document_metadata: Dict[str, Any]
    processing_time: float


class ConfidenceScorer:
    """Advanced confidence scoring system for legal document classification"""
    
    def __init__(self, thresholds: Optional[ScoreThresholds] = None):
        """
        Initialize the confidence scorer
        
        Args:
            thresholds: Custom score thresholds, uses defaults if None
        """
        self.thresholds = thresholds or ScoreThresholds()
        self.historical_scores = []  # For outlier detection
        
        logger.info("ConfidenceScorer initialized")
    
    def assess_confidence(
        self,
        classification_context: ClassificationContext,
        legal_category: LegalDocumentCategory
    ) -> ConfidenceAssessment:
        """
        Perform comprehensive confidence assessment
        
        Args:
            classification_context: Context information about the classification
            legal_category: Legal category information
            
        Returns:
            ConfidenceAssessment with detailed scoring and recommendations
        """
        # Get primary confidence from top prediction
        max_prob = max(classification_context.probability_distribution.values())
        primary_confidence = max_prob
        
        # Calculate detailed metrics
        entropy = self._calculate_entropy(classification_context.probability_distribution)
        class_separation = self._calculate_class_separation(
            classification_context.probability_distribution
        )
        
        # Assess document characteristics
        complexity = self._assess_document_complexity(classification_context)
        linguistic_quality = self._assess_linguistic_quality(classification_context)
        
        # Calculate prediction stability (simplified - would need multiple runs in practice)
        prediction_stability = min(primary_confidence * 1.1, 1.0)
        
        # Domain confidence assessment
        domain_confidence = self._assess_domain_confidence(
            classification_context, legal_category
        )
        
        # Overall quality score
        quality_score = self._calculate_quality_score(
            primary_confidence, entropy, class_separation, 
            complexity, linguistic_quality, domain_confidence
        )
        
        # Determine confidence level
        confidence_level = self._determine_confidence_level(primary_confidence)
        
        # Assess review requirements
        requires_review, review_triggers, priority = self._assess_review_requirements(
            classification_context, legal_category, primary_confidence,
            entropy, class_separation, complexity
        )
        
        # Calculate reliability indicators
        reliability_indicators = {
            "prediction_consistency": prediction_stability,
            "class_distinctiveness": class_separation,
            "entropy_score": 1.0 - (entropy / np.log(len(classification_context.probability_distribution))),
            "length_appropriateness": self._assess_length_appropriateness(classification_context),
            "processing_speed": min(1.0, 10.0 / max(classification_context.processing_time, 0.1))
        }
        
        # Create assessment
        assessment = ConfidenceAssessment(
            primary_confidence=primary_confidence,
            confidence_level=confidence_level,
            prediction_entropy=entropy,
            class_separation=class_separation,
            prediction_stability=prediction_stability,
            requires_review=requires_review,
            review_priority=priority,
            review_triggers=review_triggers,
            quality_score=quality_score,
            reliability_indicators=reliability_indicators,
            document_complexity=complexity,
            domain_confidence=domain_confidence,
            linguistic_quality=linguistic_quality
        )
        
        # Store for historical analysis
        self.historical_scores.append(primary_confidence)
        
        logger.info(
            f"Confidence assessment completed: {confidence_level.value}, "
            f"score={primary_confidence:.3f}, review={requires_review}"
        )
        
        return assessment
    
    def _calculate_entropy(self, probability_distribution: Dict[str, float]) -> float:
        """Calculate Shannon entropy of prediction distribution"""
        probs = list(probability_distribution.values())
        probs = [p for p in probs if p > 0]  # Remove zero probabilities
        
        if not probs:
            return 0.0
        
        entropy = -sum(p * np.log(p) for p in probs)
        return float(entropy)
    
    def _calculate_class_separation(self, probability_distribution: Dict[str, float]) -> float:
        """Calculate separation between top two predictions"""
        sorted_probs = sorted(probability_distribution.values(), reverse=True)
        
        if len(sorted_probs) < 2:
            return 1.0  # Perfect separation if only one class
        
        return sorted_probs[0] - sorted_probs[1]
    
    def _assess_document_complexity(self, context: ClassificationContext) -> float:
        """Assess document complexity based on various factors"""
        complexity_factors = []
        
        # Length-based complexity
        length_complexity = min(1.0, context.document_length / 10000)
        complexity_factors.append(length_complexity)
        
        # Vocabulary diversity (simplified estimation)
        if 'unique_words' in context.document_metadata:
            vocab_diversity = min(1.0, context.document_metadata['unique_words'] / 1000)
            complexity_factors.append(vocab_diversity)
        
        # Processing time as complexity indicator
        time_complexity = min(1.0, context.processing_time / 5.0)
        complexity_factors.append(time_complexity)
        
        return np.mean(complexity_factors) if complexity_factors else 0.5
    
    def _assess_linguistic_quality(self, context: ClassificationContext) -> float:
        """Assess linguistic quality of the document"""
        quality_factors = []
        
        # Basic length assessment
        if context.document_length < 10:
            quality_factors.append(0.2)
        elif context.document_length < 100:
            quality_factors.append(0.5)
        else:
            quality_factors.append(0.8)
        
        # Metadata-based quality assessment
        if 'readability_score' in context.document_metadata:
            quality_factors.append(context.document_metadata['readability_score'])
        
        if 'spelling_errors' in context.document_metadata:
            error_ratio = context.document_metadata['spelling_errors'] / max(context.document_length / 100, 1)
            quality_factors.append(max(0.0, 1.0 - error_ratio))
        
        return np.mean(quality_factors) if quality_factors else 0.7
    
    def _assess_domain_confidence(
        self, 
        context: ClassificationContext, 
        legal_category: LegalDocumentCategory
    ) -> float:
        """Assess confidence in legal domain assignment"""
        # Simple heuristic based on document type consistency with domain
        domain_consistency = {
            (DocumentType.POLICE_REPORT, LegalDomain.CRIMINAL_LAW): 0.9,
            (DocumentType.WITNESS_STATEMENT, LegalDomain.CRIMINAL_LAW): 0.9,
            (DocumentType.MEDICAL_RECORD, LegalDomain.CIVIL_RIGHTS): 0.8,
            (DocumentType.MILITARY_REPORT, LegalDomain.INTERNATIONAL_HUMANITARIAN_LAW): 0.9,
            (DocumentType.GOVERNMENT_DOCUMENT, LegalDomain.ADMINISTRATIVE_LAW): 0.8,
        }
        
        key = (context.document_type_prediction, context.legal_domain)
        return domain_consistency.get(key, 0.6)  # Default moderate confidence
    
    def _calculate_quality_score(
        self,
        confidence: float,
        entropy: float,
        class_separation: float,
        complexity: float,
        linguistic_quality: float,
        domain_confidence: float
    ) -> float:
        """Calculate overall quality score"""
        # Weighted combination of quality factors
        weights = {
            'confidence': 0.3,
            'entropy': 0.15,
            'separation': 0.15,
            'complexity': 0.1,
            'linguistic': 0.15,
            'domain': 0.15
        }
        
        # Normalize entropy (lower entropy is better for classification)
        max_entropy = np.log(10)  # Assuming max 10 classes
        normalized_entropy = 1.0 - min(1.0, entropy / max_entropy)
        
        quality_score = (
            weights['confidence'] * confidence +
            weights['entropy'] * normalized_entropy +
            weights['separation'] * class_separation +
            weights['complexity'] * (1.0 - complexity * 0.5) +  # High complexity reduces quality
            weights['linguistic'] * linguistic_quality +
            weights['domain'] * domain_confidence
        )
        
        return max(0.0, min(1.0, quality_score))
    
    def _determine_confidence_level(self, confidence_score: float) -> ConfidenceLevel:
        """Determine categorical confidence level"""
        if confidence_score >= self.thresholds.very_high_threshold:
            return ConfidenceLevel.VERY_HIGH
        elif confidence_score >= self.thresholds.high_threshold:
            return ConfidenceLevel.HIGH
        elif confidence_score >= self.thresholds.medium_threshold:
            return ConfidenceLevel.MEDIUM
        elif confidence_score >= self.thresholds.low_threshold:
            return ConfidenceLevel.LOW
        else:
            return ConfidenceLevel.VERY_LOW
    
    def _assess_review_requirements(
        self,
        context: ClassificationContext,
        legal_category: LegalDocumentCategory,
        confidence: float,
        entropy: float,
        class_separation: float,
        complexity: float
    ) -> Tuple[bool, List[ReviewTrigger], str]:
        """Assess whether human review is required and determine priority"""
        
        requires_review = False
        triggers = []
        priority = "low"
        
        # Confidence-based triggers
        if confidence < self.thresholds.review_confidence_threshold:
            requires_review = True
            triggers.append(ReviewTrigger.LOW_CONFIDENCE)
            if confidence < 0.5:
                priority = "high"
            elif confidence < 0.6:
                priority = "medium"
        
        # Class separation (ambiguity) trigger
        if class_separation < self.thresholds.ambiguity_threshold:
            requires_review = True
            triggers.append(ReviewTrigger.AMBIGUOUS_CLASSIFICATION)
            if class_separation < 0.1:
                priority = "high"
        
        # Multi-class uncertainty
        top_3_probs = sorted(context.probability_distribution.values(), reverse=True)[:3]
        if len(top_3_probs) >= 3 and (top_3_probs[0] - top_3_probs[2]) < 0.3:
            requires_review = True
            triggers.append(ReviewTrigger.MULTI_CLASS_UNCERTAINTY)
        
        # Legal category-based triggers
        if legal_category.requires_human_review:
            requires_review = True
            triggers.append(ReviewTrigger.REGULATORY_REQUIREMENT)
            
        if legal_category.sensitivity_level in ["confidential", "restricted"]:
            requires_review = True
            triggers.append(ReviewTrigger.SENSITIVE_CONTENT)
            priority = "high"
            
        if legal_category.urgency_level == "critical":
            requires_review = True
            triggers.append(ReviewTrigger.CRITICAL_DOCUMENT_TYPE)
            priority = "critical"
        
        # Document characteristics triggers
        if (context.document_length < self.thresholds.min_document_length or 
            context.document_length > self.thresholds.max_document_length):
            requires_review = True
            triggers.append(ReviewTrigger.LENGTH_ANOMALY)
        
        # High complexity trigger
        if complexity > 0.8:
            requires_review = True
            triggers.append(ReviewTrigger.OUTLIER_DETECTION)
        
        # Quality threshold trigger
        if entropy > 2.0:  # High entropy indicates uncertainty
            requires_review = True
            triggers.append(ReviewTrigger.QUALITY_THRESHOLD)
        
        return requires_review, triggers, priority
    
    def _assess_length_appropriateness(self, context: ClassificationContext) -> float:
        """Assess if document length is appropriate for its type"""
        # Expected length ranges for different document types (in characters)
        expected_lengths = {
            DocumentType.WITNESS_STATEMENT: (500, 5000),
            DocumentType.POLICE_REPORT: (1000, 10000),
            DocumentType.MEDICAL_RECORD: (200, 3000),
            DocumentType.COURT_FILING: (1000, 20000),
            DocumentType.EMAIL: (50, 2000),
            DocumentType.FORENSIC_REPORT: (2000, 15000),
        }
        
        expected_range = expected_lengths.get(
            context.document_type_prediction, 
            (100, 10000)  # Default range
        )
        
        min_length, max_length = expected_range
        
        if min_length <= context.document_length <= max_length:
            return 1.0
        elif context.document_length < min_length:
            return max(0.3, context.document_length / min_length)
        else:  # length > max_length
            return max(0.5, max_length / context.document_length)
    
    def update_thresholds(self, new_thresholds: ScoreThresholds) -> None:
        """Update scoring thresholds"""
        self.thresholds = new_thresholds
        logger.info("Confidence scoring thresholds updated")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about historical confidence scores"""
        if not self.historical_scores:
            return {"message": "No historical data available"}
        
        scores = np.array(self.historical_scores)
        
        return {
            "total_assessments": len(scores),
            "mean_confidence": float(np.mean(scores)),
            "std_confidence": float(np.std(scores)),
            "median_confidence": float(np.median(scores)),
            "min_confidence": float(np.min(scores)),
            "max_confidence": float(np.max(scores)),
            "confidence_distribution": {
                "very_low": int(np.sum(scores < 0.4)),
                "low": int(np.sum((scores >= 0.4) & (scores < 0.6))),
                "medium": int(np.sum((scores >= 0.6) & (scores < 0.8))),
                "high": int(np.sum((scores >= 0.8) & (scores < 0.9))),
                "very_high": int(np.sum(scores >= 0.9)),
            }
        }
    
    def calibrate_thresholds(self, validation_data: List[Tuple[float, bool]]) -> ScoreThresholds:
        """
        Calibrate thresholds based on validation data
        
        Args:
            validation_data: List of (confidence_score, is_correct) tuples
            
        Returns:
            Optimized ScoreThresholds
        """
        if not validation_data:
            logger.warning("No validation data provided for threshold calibration")
            return self.thresholds
        
        scores, correctness = zip(*validation_data)
        scores = np.array(scores)
        correctness = np.array(correctness)
        
        # Find optimal threshold for review trigger
        # Goal: maximize accuracy while minimizing false reviews
        best_threshold = 0.7
        best_score = 0.0
        
        for threshold in np.arange(0.3, 0.95, 0.05):
            # Calculate metrics at this threshold
            true_positives = np.sum((scores >= threshold) & correctness)
            false_positives = np.sum((scores >= threshold) & ~correctness)
            true_negatives = np.sum((scores < threshold) & ~correctness)
            false_negatives = np.sum((scores < threshold) & correctness)
            
            if (true_positives + false_positives) > 0:
                precision = true_positives / (true_positives + false_positives)
                recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
                f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                
                # Weighted score favoring high precision (avoid unnecessary reviews)
                score = 0.7 * precision + 0.3 * recall
                
                if score > best_score:
                    best_score = score
                    best_threshold = threshold
        
        # Update thresholds
        optimized_thresholds = ScoreThresholds(
            review_confidence_threshold=best_threshold,
            very_high_threshold=max(best_threshold + 0.15, 0.9),
            high_threshold=max(best_threshold + 0.05, 0.8),
            medium_threshold=best_threshold - 0.1,
            low_threshold=best_threshold - 0.2
        )
        
        logger.info(f"Thresholds calibrated. Review threshold: {best_threshold:.3f}")
        return optimized_thresholds