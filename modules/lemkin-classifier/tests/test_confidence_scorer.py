"""
Tests for the confidence scorer module.
"""

import pytest
from unittest.mock import Mock
import numpy as np
from datetime import datetime

from lemkin_classifier.confidence_scorer import (
    ConfidenceScorer,
    ConfidenceAssessment,
    ConfidenceLevel,
    ReviewTrigger,
    ScoreThresholds,
    ClassificationContext
)
from lemkin_classifier.legal_taxonomy import (
    DocumentType,
    LegalDomain,
    LegalDocumentCategory,
    CATEGORY_DEFINITIONS
)


class TestScoreThresholds:
    """Test ScoreThresholds model"""
    
    def test_default_thresholds(self):
        """Test default threshold values"""
        thresholds = ScoreThresholds()
        
        assert thresholds.very_high_threshold == 0.9
        assert thresholds.high_threshold == 0.8
        assert thresholds.medium_threshold == 0.6
        assert thresholds.low_threshold == 0.4
        assert thresholds.review_confidence_threshold == 0.7
        assert thresholds.ambiguity_threshold == 0.3
        assert thresholds.min_document_length == 50
        assert thresholds.max_document_length == 100000
    
    def test_custom_thresholds(self):
        """Test custom threshold values"""
        thresholds = ScoreThresholds(
            very_high_threshold=0.95,
            high_threshold=0.85,
            medium_threshold=0.65,
            low_threshold=0.45,
            review_confidence_threshold=0.75
        )
        
        assert thresholds.very_high_threshold == 0.95
        assert thresholds.high_threshold == 0.85
        assert thresholds.medium_threshold == 0.65
        assert thresholds.low_threshold == 0.45
        assert thresholds.review_confidence_threshold == 0.75
    
    def test_threshold_validation(self):
        """Test threshold validation"""
        # Valid thresholds
        thresholds = ScoreThresholds(
            very_high_threshold=0.9,
            high_threshold=0.8,
            medium_threshold=0.6,
            low_threshold=0.4
        )
        assert thresholds.very_high_threshold == 0.9
        
        # Invalid threshold order should raise validation error
        with pytest.raises(ValueError):
            ScoreThresholds(
                very_high_threshold=0.8,
                high_threshold=0.9  # Higher than very_high_threshold
            )


class TestConfidenceAssessment:
    """Test ConfidenceAssessment model"""
    
    def test_assessment_creation(self):
        """Test creating a ConfidenceAssessment instance"""
        assessment = ConfidenceAssessment(
            primary_confidence=0.85,
            confidence_level=ConfidenceLevel.HIGH,
            prediction_entropy=0.5,
            class_separation=0.4,
            prediction_stability=0.9,
            requires_review=True,
            review_priority="high",
            review_triggers=[ReviewTrigger.LOW_CONFIDENCE, ReviewTrigger.SENSITIVE_CONTENT],
            quality_score=0.8,
            reliability_indicators={"consistency": 0.9, "distinctiveness": 0.7},
            document_complexity=0.6,
            domain_confidence=0.8,
            linguistic_quality=0.7
        )
        
        assert assessment.primary_confidence == 0.85
        assert assessment.confidence_level == ConfidenceLevel.HIGH
        assert assessment.prediction_entropy == 0.5
        assert assessment.class_separation == 0.4
        assert assessment.prediction_stability == 0.9
        assert assessment.requires_review is True
        assert assessment.review_priority == "high"
        assert len(assessment.review_triggers) == 2
        assert ReviewTrigger.LOW_CONFIDENCE in assessment.review_triggers
        assert assessment.quality_score == 0.8
        assert assessment.document_complexity == 0.6
        assert isinstance(assessment.assessment_timestamp, datetime)


class TestClassificationContext:
    """Test ClassificationContext dataclass"""
    
    def test_context_creation(self):
        """Test creating a ClassificationContext instance"""
        context = ClassificationContext(
            document_length=1000,
            document_type_prediction=DocumentType.WITNESS_STATEMENT,
            legal_domain=LegalDomain.CRIMINAL_LAW,
            probability_distribution={
                "witness_statement": 0.85,
                "police_report": 0.10,
                "medical_record": 0.05
            },
            document_metadata={"source": "police_station"},
            processing_time=2.5
        )
        
        assert context.document_length == 1000
        assert context.document_type_prediction == DocumentType.WITNESS_STATEMENT
        assert context.legal_domain == LegalDomain.CRIMINAL_LAW
        assert context.probability_distribution["witness_statement"] == 0.85
        assert context.document_metadata["source"] == "police_station"
        assert context.processing_time == 2.5


class TestConfidenceScorer:
    """Test ConfidenceScorer class"""
    
    @pytest.fixture
    def scorer(self):
        """Create a ConfidenceScorer instance for testing"""
        return ConfidenceScorer()
    
    @pytest.fixture
    def custom_scorer(self):
        """Create a ConfidenceScorer with custom thresholds"""
        thresholds = ScoreThresholds(
            review_confidence_threshold=0.8,
            ambiguity_threshold=0.2,
            min_document_length=100
        )
        return ConfidenceScorer(thresholds)
    
    @pytest.fixture
    def sample_context(self):
        """Sample classification context for testing"""
        return ClassificationContext(
            document_length=500,
            document_type_prediction=DocumentType.WITNESS_STATEMENT,
            legal_domain=LegalDomain.CRIMINAL_LAW,
            probability_distribution={
                "witness_statement": 0.75,
                "police_report": 0.20,
                "medical_record": 0.05
            },
            document_metadata={"readability_score": 0.8},
            processing_time=1.5
        )
    
    @pytest.fixture
    def sample_legal_category(self):
        """Sample legal category for testing"""
        return CATEGORY_DEFINITIONS.get(
            DocumentType.WITNESS_STATEMENT,
            LegalDocumentCategory(
                document_type=DocumentType.WITNESS_STATEMENT,
                legal_domain=LegalDomain.CRIMINAL_LAW,
                urgency_level="high",
                sensitivity_level="confidential",
                requires_human_review=True,
                redaction_required=True,
                chain_of_custody_critical=True
            )
        )
    
    def test_scorer_initialization(self, scorer):
        """Test ConfidenceScorer initialization"""
        assert isinstance(scorer.thresholds, ScoreThresholds)
        assert isinstance(scorer.historical_scores, list)
        assert len(scorer.historical_scores) == 0
    
    def test_custom_thresholds_initialization(self, custom_scorer):
        """Test ConfidenceScorer with custom thresholds"""
        assert custom_scorer.thresholds.review_confidence_threshold == 0.8
        assert custom_scorer.thresholds.ambiguity_threshold == 0.2
        assert custom_scorer.thresholds.min_document_length == 100
    
    def test_calculate_entropy(self, scorer):
        """Test entropy calculation"""
        # Uniform distribution should have higher entropy
        uniform_dist = {"class1": 0.25, "class2": 0.25, "class3": 0.25, "class4": 0.25}
        uniform_entropy = scorer._calculate_entropy(uniform_dist)
        
        # Skewed distribution should have lower entropy
        skewed_dist = {"class1": 0.9, "class2": 0.05, "class3": 0.03, "class4": 0.02}
        skewed_entropy = scorer._calculate_entropy(skewed_dist)
        
        assert uniform_entropy > skewed_entropy
        assert uniform_entropy > 0
        assert skewed_entropy >= 0
    
    def test_calculate_class_separation(self, scorer):
        """Test class separation calculation"""
        # High separation
        high_sep_dist = {"class1": 0.9, "class2": 0.1}
        high_separation = scorer._calculate_class_separation(high_sep_dist)
        
        # Low separation
        low_sep_dist = {"class1": 0.55, "class2": 0.45}
        low_separation = scorer._calculate_class_separation(low_sep_dist)
        
        assert high_separation > low_separation
        assert high_separation == 0.8
        assert low_separation == 0.1
        
        # Single class should return 1.0 (perfect separation)
        single_class = {"class1": 1.0}
        single_separation = scorer._calculate_class_separation(single_class)
        assert single_separation == 1.0
    
    def test_assess_document_complexity(self, scorer, sample_context):
        """Test document complexity assessment"""
        complexity = scorer._assess_document_complexity(sample_context)
        
        assert 0 <= complexity <= 1
        assert isinstance(complexity, float)
        
        # Test with different document lengths
        short_context = ClassificationContext(
            document_length=10,
            document_type_prediction=DocumentType.WITNESS_STATEMENT,
            legal_domain=LegalDomain.CRIMINAL_LAW,
            probability_distribution={"witness_statement": 0.8},
            document_metadata={},
            processing_time=0.5
        )
        
        long_context = ClassificationContext(
            document_length=50000,
            document_type_prediction=DocumentType.WITNESS_STATEMENT,
            legal_domain=LegalDomain.CRIMINAL_LAW,
            probability_distribution={"witness_statement": 0.8},
            document_metadata={},
            processing_time=5.0
        )
        
        short_complexity = scorer._assess_document_complexity(short_context)
        long_complexity = scorer._assess_document_complexity(long_context)
        
        assert long_complexity > short_complexity
    
    def test_assess_linguistic_quality(self, scorer, sample_context):
        """Test linguistic quality assessment"""
        quality = scorer._assess_linguistic_quality(sample_context)
        
        assert 0 <= quality <= 1
        assert isinstance(quality, float)
        
        # Test with very short document
        short_context = ClassificationContext(
            document_length=5,
            document_type_prediction=DocumentType.WITNESS_STATEMENT,
            legal_domain=LegalDomain.CRIMINAL_LAW,
            probability_distribution={"witness_statement": 0.8},
            document_metadata={},
            processing_time=0.5
        )
        
        short_quality = scorer._assess_linguistic_quality(short_context)
        assert short_quality < quality  # Should be lower quality for very short documents
    
    def test_determine_confidence_level(self, scorer):
        """Test confidence level determination"""
        assert scorer._determine_confidence_level(0.95) == ConfidenceLevel.VERY_HIGH
        assert scorer._determine_confidence_level(0.85) == ConfidenceLevel.HIGH
        assert scorer._determine_confidence_level(0.70) == ConfidenceLevel.MEDIUM
        assert scorer._determine_confidence_level(0.50) == ConfidenceLevel.LOW
        assert scorer._determine_confidence_level(0.30) == ConfidenceLevel.VERY_LOW
    
    def test_assess_confidence(self, scorer, sample_context, sample_legal_category):
        """Test comprehensive confidence assessment"""
        assessment = scorer.assess_confidence(sample_context, sample_legal_category)
        
        assert isinstance(assessment, ConfidenceAssessment)
        assert 0 <= assessment.primary_confidence <= 1
        assert isinstance(assessment.confidence_level, ConfidenceLevel)
        assert assessment.prediction_entropy >= 0
        assert 0 <= assessment.class_separation <= 1
        assert 0 <= assessment.prediction_stability <= 1
        assert isinstance(assessment.requires_review, bool)
        assert assessment.review_priority in ["low", "medium", "high", "critical"]
        assert isinstance(assessment.review_triggers, list)
        assert 0 <= assessment.quality_score <= 1
        assert isinstance(assessment.reliability_indicators, dict)
        assert 0 <= assessment.document_complexity <= 1
        assert 0 <= assessment.domain_confidence <= 1
        assert 0 <= assessment.linguistic_quality <= 1
    
    def test_review_requirements_low_confidence(self, scorer, sample_context, sample_legal_category):
        """Test review requirements for low confidence"""
        # Create low confidence context
        low_confidence_context = ClassificationContext(
            document_length=500,
            document_type_prediction=DocumentType.WITNESS_STATEMENT,
            legal_domain=LegalDomain.CRIMINAL_LAW,
            probability_distribution={
                "witness_statement": 0.5,  # Low confidence
                "police_report": 0.3,
                "medical_record": 0.2
            },
            document_metadata={},
            processing_time=1.5
        )
        
        requires_review, triggers, priority = scorer._assess_review_requirements(
            low_confidence_context, sample_legal_category, 0.5, 1.0, 0.2, 0.5
        )
        
        assert requires_review is True
        assert ReviewTrigger.LOW_CONFIDENCE in triggers
        assert priority in ["medium", "high"]
    
    def test_review_requirements_ambiguous_classification(self, scorer, sample_context, sample_legal_category):
        """Test review requirements for ambiguous classification"""
        # Create ambiguous context
        ambiguous_context = ClassificationContext(
            document_length=500,
            document_type_prediction=DocumentType.WITNESS_STATEMENT,
            legal_domain=LegalDomain.CRIMINAL_LAW,
            probability_distribution={
                "witness_statement": 0.51,  # Very close to other classes
                "police_report": 0.49
            },
            document_metadata={},
            processing_time=1.5
        )
        
        requires_review, triggers, priority = scorer._assess_review_requirements(
            ambiguous_context, sample_legal_category, 0.8, 0.5, 0.02, 0.5  # Low class separation
        )
        
        assert requires_review is True
        assert ReviewTrigger.AMBIGUOUS_CLASSIFICATION in triggers
    
    def test_review_requirements_sensitive_content(self, scorer, sample_context):
        """Test review requirements for sensitive content"""
        # Create high-sensitivity category
        sensitive_category = LegalDocumentCategory(
            document_type=DocumentType.WITNESS_STATEMENT,
            legal_domain=LegalDomain.CRIMINAL_LAW,
            sensitivity_level="restricted",
            requires_human_review=True,
            urgency_level="critical"
        )
        
        requires_review, triggers, priority = scorer._assess_review_requirements(
            sample_context, sensitive_category, 0.9, 0.5, 0.5, 0.5
        )
        
        assert requires_review is True
        assert ReviewTrigger.REGULATORY_REQUIREMENT in triggers
        assert ReviewTrigger.SENSITIVE_CONTENT in triggers
        assert ReviewTrigger.CRITICAL_DOCUMENT_TYPE in triggers
        assert priority == "critical"
    
    def test_length_appropriateness(self, scorer, sample_context):
        """Test document length appropriateness assessment"""
        appropriateness = scorer._assess_length_appropriateness(sample_context)
        
        assert 0 <= appropriateness <= 1
        
        # Test with very short document
        short_context = ClassificationContext(
            document_length=10,
            document_type_prediction=DocumentType.WITNESS_STATEMENT,
            legal_domain=LegalDomain.CRIMINAL_LAW,
            probability_distribution={"witness_statement": 0.8},
            document_metadata={},
            processing_time=0.5
        )
        
        short_appropriateness = scorer._assess_length_appropriateness(short_context)
        assert short_appropriateness < appropriateness
        
        # Test with very long document
        long_context = ClassificationContext(
            document_length=100000,
            document_type_prediction=DocumentType.WITNESS_STATEMENT,
            legal_domain=LegalDomain.CRIMINAL_LAW,
            probability_distribution={"witness_statement": 0.8},
            document_metadata={},
            processing_time=10.0
        )
        
        long_appropriateness = scorer._assess_length_appropriateness(long_context)
        assert long_appropriateness < 1.0  # Should be penalized for being too long
    
    def test_update_thresholds(self, scorer):
        """Test updating score thresholds"""
        new_thresholds = ScoreThresholds(
            review_confidence_threshold=0.8,
            ambiguity_threshold=0.2
        )
        
        scorer.update_thresholds(new_thresholds)
        
        assert scorer.thresholds.review_confidence_threshold == 0.8
        assert scorer.thresholds.ambiguity_threshold == 0.2
    
    def test_historical_scores_tracking(self, scorer, sample_context, sample_legal_category):
        """Test that historical scores are tracked"""
        initial_count = len(scorer.historical_scores)
        
        # Perform several assessments
        for _ in range(3):
            scorer.assess_confidence(sample_context, sample_legal_category)
        
        assert len(scorer.historical_scores) == initial_count + 3
    
    def test_get_statistics_empty(self, scorer):
        """Test statistics with no historical data"""
        stats = scorer.get_statistics()
        
        assert "message" in stats
        assert stats["message"] == "No historical data available"
    
    def test_get_statistics_with_data(self, scorer):
        """Test statistics with historical data"""
        # Add some historical scores
        scorer.historical_scores = [0.8, 0.9, 0.7, 0.6, 0.95]
        
        stats = scorer.get_statistics()
        
        assert stats["total_assessments"] == 5
        assert stats["mean_confidence"] == 0.8
        assert "std_confidence" in stats
        assert "median_confidence" in stats
        assert "min_confidence" in stats
        assert "max_confidence" in stats
        assert "confidence_distribution" in stats
        
        # Check confidence distribution
        dist = stats["confidence_distribution"]
        assert "very_low" in dist
        assert "low" in dist
        assert "medium" in dist
        assert "high" in dist
        assert "very_high" in dist
        assert sum(dist.values()) == 5
    
    def test_calibrate_thresholds(self, scorer):
        """Test threshold calibration"""
        # Sample validation data: (confidence_score, is_correct)
        validation_data = [
            (0.9, True), (0.8, True), (0.7, True), (0.6, False),
            (0.5, False), (0.4, False), (0.95, True), (0.85, True),
            (0.75, True), (0.65, False)
        ]
        
        optimized_thresholds = scorer.calibrate_thresholds(validation_data)
        
        assert isinstance(optimized_thresholds, ScoreThresholds)
        assert 0 <= optimized_thresholds.review_confidence_threshold <= 1
        assert optimized_thresholds.very_high_threshold >= optimized_thresholds.high_threshold
        assert optimized_thresholds.high_threshold >= optimized_thresholds.medium_threshold
        assert optimized_thresholds.medium_threshold >= optimized_thresholds.low_threshold
    
    def test_calibrate_thresholds_empty_data(self, scorer):
        """Test threshold calibration with empty data"""
        result = scorer.calibrate_thresholds([])
        
        # Should return current thresholds unchanged
        assert result == scorer.thresholds


class TestConfidenceScorerEdgeCases:
    """Test edge cases for ConfidenceScorer"""
    
    @pytest.fixture
    def scorer(self):
        return ConfidenceScorer()
    
    def test_zero_entropy_calculation(self, scorer):
        """Test entropy calculation with zero probabilities"""
        dist_with_zeros = {"class1": 0.8, "class2": 0.2, "class3": 0.0, "class4": 0.0}
        entropy = scorer._calculate_entropy(dist_with_zeros)
        
        assert entropy >= 0
        assert np.isfinite(entropy)
    
    def test_single_class_distribution(self, scorer):
        """Test with single class distribution"""
        single_class = {"class1": 1.0}
        
        entropy = scorer._calculate_entropy(single_class)
        separation = scorer._calculate_class_separation(single_class)
        
        assert entropy == 0.0  # No uncertainty
        assert separation == 1.0  # Perfect separation
    
    def test_empty_probability_distribution(self, scorer):
        """Test with empty probability distribution"""
        empty_dist = {}
        
        entropy = scorer._calculate_entropy(empty_dist)
        separation = scorer._calculate_class_separation(empty_dist)
        
        assert entropy == 0.0
        assert separation == 1.0
    
    def test_extreme_document_lengths(self, scorer):
        """Test with extreme document lengths"""
        # Very short document
        short_context = ClassificationContext(
            document_length=1,
            document_type_prediction=DocumentType.EMAIL,
            legal_domain=LegalDomain.GENERAL,
            probability_distribution={"email": 1.0},
            document_metadata={},
            processing_time=0.1
        )
        
        # Very long document
        long_context = ClassificationContext(
            document_length=1000000,
            document_type_prediction=DocumentType.GOVERNMENT_DOCUMENT,
            legal_domain=LegalDomain.ADMINISTRATIVE_LAW,
            probability_distribution={"government_document": 1.0},
            document_metadata={},
            processing_time=30.0
        )
        
        category = LegalDocumentCategory(
            document_type=DocumentType.EMAIL,
            legal_domain=LegalDomain.GENERAL
        )
        
        short_assessment = scorer.assess_confidence(short_context, category)
        long_assessment = scorer.assess_confidence(long_context, category)
        
        # Both should complete without errors
        assert isinstance(short_assessment, ConfidenceAssessment)
        assert isinstance(long_assessment, ConfidenceAssessment)
        
        # Short documents should likely trigger length anomaly
        assert ReviewTrigger.LENGTH_ANOMALY in short_assessment.review_triggers
        
        # Very long documents should also trigger length anomaly
        assert ReviewTrigger.LENGTH_ANOMALY in long_assessment.review_triggers