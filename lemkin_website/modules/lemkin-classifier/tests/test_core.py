"""
Tests for the core document classification functionality.
"""

import json
import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
import tempfile
import os

from lemkin_classifier.core import (
    DocumentClassifier,
    DocumentContent,
    DocumentClassification,
    ClassificationResult,
    ClassificationConfig,
    ModelMetrics,
    TrainingMetrics,
)
from lemkin_classifier.legal_taxonomy import DocumentType, LegalDomain


class TestDocumentContent:
    """Test DocumentContent model"""
    
    def test_document_content_creation(self):
        """Test creating a DocumentContent instance"""
        content = DocumentContent(
            text="This is a sample legal document.",
            metadata={"author": "John Doe"},
            file_path="/path/to/document.pdf",
            file_type="pdf",
            language="en"
        )
        
        assert content.text == "This is a sample legal document."
        assert content.metadata["author"] == "John Doe"
        assert content.file_path == "/path/to/document.pdf"
        assert content.file_type == "pdf"
        assert content.language == "en"
        assert content.length == len("This is a sample legal document.")
        assert content.word_count == 6
    
    def test_document_content_length_calculation(self):
        """Test automatic length calculation"""
        content = DocumentContent(text="Hello world")
        assert content.length == 11
        assert content.word_count == 2
    
    def test_document_content_empty_text(self):
        """Test DocumentContent with empty text"""
        content = DocumentContent(text="")
        assert content.length == 0
        assert content.word_count == 1  # split() on empty string returns [""]


class TestDocumentClassification:
    """Test DocumentClassification model"""
    
    def test_classification_creation(self):
        """Test creating a DocumentClassification instance"""
        classification = DocumentClassification(
            document_type=DocumentType.WITNESS_STATEMENT,
            legal_domain=LegalDomain.CRIMINAL_LAW,
            confidence_score=0.85,
            probability_distribution={"witness_statement": 0.85, "police_report": 0.15},
            model_version="1.0",
            model_name="distilbert-base-uncased"
        )
        
        assert classification.document_type == DocumentType.WITNESS_STATEMENT
        assert classification.legal_domain == LegalDomain.CRIMINAL_LAW
        assert classification.confidence_score == 0.85
        assert classification.probability_distribution["witness_statement"] == 0.85
        assert classification.model_version == "1.0"
        assert classification.model_name == "distilbert-base-uncased"
        assert isinstance(classification.timestamp, datetime)


class TestClassificationResult:
    """Test ClassificationResult model"""
    
    @pytest.fixture
    def sample_document_content(self):
        return DocumentContent(
            text="This is a witness statement about the incident.",
            metadata={"source": "police_station"},
            file_path="/path/to/witness.pdf",
            file_type="pdf"
        )
    
    @pytest.fixture
    def sample_classification(self):
        return DocumentClassification(
            document_type=DocumentType.WITNESS_STATEMENT,
            legal_domain=LegalDomain.CRIMINAL_LAW,
            confidence_score=0.85,
            probability_distribution={"witness_statement": 0.85},
            model_version="1.0",
            model_name="test-model"
        )
    
    def test_classification_result_creation(self, sample_document_content, sample_classification):
        """Test creating a ClassificationResult instance"""
        from lemkin_classifier.legal_taxonomy import CATEGORY_DEFINITIONS
        
        legal_category = CATEGORY_DEFINITIONS.get(
            DocumentType.WITNESS_STATEMENT,
            None
        )
        
        # Skip test if category not defined
        if legal_category is None:
            pytest.skip("Witness statement category not defined in taxonomy")
        
        result = ClassificationResult(
            document_content=sample_document_content,
            classification=sample_classification,
            legal_category=legal_category,
            processing_time=1.5,
            requires_review=True,
            review_reasons=["High sensitivity"],
            recommended_actions=["Apply PII redaction"],
            urgency_level="high",
            sensitivity_level="confidential"
        )
        
        assert result.processing_time == 1.5
        assert result.requires_review is True
        assert len(result.review_reasons) == 1
        assert len(result.recommended_actions) == 1
        assert result.urgency_level == "high"
        assert result.sensitivity_level == "confidential"


class TestClassificationConfig:
    """Test ClassificationConfig model"""
    
    def test_config_creation(self):
        """Test creating a ClassificationConfig instance"""
        config = ClassificationConfig(
            model_name="distilbert-base-uncased",
            model_path="/path/to/model",
            max_length=512,
            confidence_threshold=0.7,
            batch_size=16
        )
        
        assert config.model_name == "distilbert-base-uncased"
        assert config.model_path == "/path/to/model"
        assert config.max_length == 512
        assert config.confidence_threshold == 0.7
        assert config.batch_size == 16
    
    def test_config_defaults(self):
        """Test default configuration values"""
        config = ClassificationConfig()
        
        assert config.model_name == "distilbert-base-uncased"
        assert config.model_path is None
        assert config.max_length == 512
        assert config.confidence_threshold == 0.7
        assert config.batch_size == 16
        assert config.device in ["cpu", "cuda"]  # auto-detected
        assert config.supported_languages == ["en"]
        assert config.default_language == "en"
    
    def test_device_validation(self):
        """Test device validation"""
        config = ClassificationConfig(device="cpu")
        assert config.device == "cpu"
        
        config = ClassificationConfig(device="cuda")
        assert config.device == "cuda"
        
        config = ClassificationConfig(device="auto")
        assert config.device in ["cpu", "cuda"]


class TestModelMetrics:
    """Test ModelMetrics model"""
    
    def test_metrics_creation(self):
        """Test creating ModelMetrics instance"""
        metrics = ModelMetrics(
            accuracy=0.85,
            precision=0.80,
            recall=0.82,
            f1_score=0.81,
            class_metrics={"witness_statement": {"precision": 0.85, "recall": 0.80, "f1-score": 0.82}},
            confusion_matrix=[[10, 2], [1, 15]],
            model_version="1.0",
            test_set_size=100
        )
        
        assert metrics.accuracy == 0.85
        assert metrics.precision == 0.80
        assert metrics.recall == 0.82
        assert metrics.f1_score == 0.81
        assert len(metrics.class_metrics) == 1
        assert metrics.confusion_matrix == [[10, 2], [1, 15]]
        assert metrics.test_set_size == 100
        assert isinstance(metrics.evaluation_date, datetime)


class TestTrainingMetrics:
    """Test TrainingMetrics model"""
    
    def test_training_metrics_creation(self):
        """Test creating TrainingMetrics instance"""
        metrics = TrainingMetrics(
            epoch=1,
            train_loss=0.5,
            eval_loss=0.6,
            eval_accuracy=0.85,
            learning_rate=5e-5,
            total_steps=100
        )
        
        assert metrics.epoch == 1
        assert metrics.train_loss == 0.5
        assert metrics.eval_loss == 0.6
        assert metrics.eval_accuracy == 0.85
        assert metrics.learning_rate == 5e-5
        assert metrics.total_steps == 100
        assert isinstance(metrics.timestamp, datetime)


class TestDocumentClassifier:
    """Test DocumentClassifier class"""
    
    @pytest.fixture
    def mock_config(self):
        """Mock configuration for testing"""
        return ClassificationConfig(
            model_name="distilbert-base-uncased",
            max_length=512,
            confidence_threshold=0.7,
            batch_size=8
        )
    
    @patch('lemkin_classifier.core.AutoTokenizer')
    @patch('lemkin_classifier.core.AutoModelForSequenceClassification')
    @patch('lemkin_classifier.core.pipeline')
    def test_classifier_initialization(self, mock_pipeline, mock_model, mock_tokenizer, mock_config):
        """Test classifier initialization"""
        # Mock the tokenizer
        mock_tokenizer.from_pretrained.return_value = MagicMock()
        
        # Mock the model
        mock_model_instance = MagicMock()
        mock_model.from_pretrained.return_value = mock_model_instance
        
        # Mock the pipeline
        mock_pipeline_instance = MagicMock()
        mock_pipeline.return_value = mock_pipeline_instance
        
        classifier = DocumentClassifier(mock_config)
        
        assert classifier.config == mock_config
        assert classifier.tokenizer is not None
        assert classifier.model is not None
        assert classifier.classifier_pipeline is not None
        
        # Verify calls
        mock_tokenizer.from_pretrained.assert_called_once()
        mock_model.from_pretrained.assert_called_once()
        mock_pipeline.assert_called_once()
    
    @patch('lemkin_classifier.core.pdfplumber')
    def test_extract_text_from_pdf(self, mock_pdfplumber, mock_config):
        """Test text extraction from PDF"""
        # Create a temporary PDF file
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp_file:
            tmp_path = Path(tmp_file.name)
        
        try:
            # Mock pdfplumber
            mock_pdf = MagicMock()
            mock_page = MagicMock()
            mock_page.extract_text.return_value = "Sample PDF content"
            mock_pdf.pages = [mock_page]
            mock_pdfplumber.open.return_value.__enter__.return_value = mock_pdf
            
            # Mock classifier initialization to avoid loading actual models
            with patch('lemkin_classifier.core.AutoTokenizer'), \
                 patch('lemkin_classifier.core.AutoModelForSequenceClassification'), \
                 patch('lemkin_classifier.core.pipeline'):
                
                classifier = DocumentClassifier(mock_config)
                content = classifier.extract_text_from_file(tmp_path)
            
            assert content.text == "Sample PDF content"
            assert content.file_type == "pdf"
            assert content.file_path == str(tmp_path)
            assert content.metadata["filename"] == tmp_path.name
            assert content.metadata["num_pages"] == 1
        
        finally:
            # Clean up
            if tmp_path.exists():
                os.unlink(tmp_path)
    
    def test_extract_text_from_txt(self, mock_config):
        """Test text extraction from text file"""
        # Create a temporary text file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as tmp_file:
            tmp_file.write("Sample text content")
            tmp_path = Path(tmp_file.name)
        
        try:
            # Mock classifier initialization to avoid loading actual models
            with patch('lemkin_classifier.core.AutoTokenizer'), \
                 patch('lemkin_classifier.core.AutoModelForSequenceClassification'), \
                 patch('lemkin_classifier.core.pipeline'):
                
                classifier = DocumentClassifier(mock_config)
                content = classifier.extract_text_from_file(tmp_path)
            
            assert content.text == "Sample text content"
            assert content.file_type == "text"
            assert content.file_path == str(tmp_path)
            assert content.metadata["filename"] == tmp_path.name
        
        finally:
            # Clean up
            if tmp_path.exists():
                os.unlink(tmp_path)
    
    def test_extract_text_file_not_found(self, mock_config):
        """Test text extraction with non-existent file"""
        with patch('lemkin_classifier.core.AutoTokenizer'), \
             patch('lemkin_classifier.core.AutoModelForSequenceClassification'), \
             patch('lemkin_classifier.core.pipeline'):
            
            classifier = DocumentClassifier(mock_config)
            
            with pytest.raises(FileNotFoundError):
                classifier.extract_text_from_file(Path("/nonexistent/file.pdf"))
    
    def test_preprocess_text(self, mock_config):
        """Test text preprocessing"""
        with patch('lemkin_classifier.core.AutoTokenizer'), \
             patch('lemkin_classifier.core.AutoModelForSequenceClassification'), \
             patch('lemkin_classifier.core.pipeline'):
            
            classifier = DocumentClassifier(mock_config)
            
            # Test basic preprocessing
            text = "  This is   a    test   document  "
            processed = classifier.preprocess_text(text)
            assert processed == "This is a test document"
            
            # Test with preprocessing disabled
            classifier.config.enable_preprocessing = False
            processed = classifier.preprocess_text(text)
            assert processed == text.strip()
    
    @patch('lemkin_classifier.core.AutoTokenizer')
    @patch('lemkin_classifier.core.AutoModelForSequenceClassification')
    @patch('lemkin_classifier.core.pipeline')
    def test_classify_document(self, mock_pipeline, mock_model, mock_tokenizer, mock_config):
        """Test document classification"""
        # Mock the components
        mock_tokenizer.from_pretrained.return_value = MagicMock()
        mock_model.from_pretrained.return_value = MagicMock()
        
        # Mock classification pipeline
        mock_pipeline_instance = MagicMock()
        mock_pipeline_instance.return_value = [
            {'label': 'witness_statement', 'score': 0.85},
            {'label': 'police_report', 'score': 0.15}
        ]
        mock_pipeline.return_value = mock_pipeline_instance
        
        classifier = DocumentClassifier(mock_config)
        
        # Create test document
        document_content = DocumentContent(
            text="This is a witness statement about the incident.",
            metadata={"source": "test"}
        )
        
        # Classify document
        result = classifier.classify_document(document_content)
        
        assert isinstance(result, ClassificationResult)
        assert result.classification.document_type == DocumentType.WITNESS_STATEMENT
        assert result.classification.confidence_score == 0.85
        assert result.processing_time > 0
    
    def test_get_model_info(self, mock_config):
        """Test getting model information"""
        with patch('lemkin_classifier.core.AutoTokenizer'), \
             patch('lemkin_classifier.core.AutoModelForSequenceClassification'), \
             patch('lemkin_classifier.core.pipeline'):
            
            classifier = DocumentClassifier(mock_config)
            info = classifier.get_model_info()
            
            assert isinstance(info, dict)
            assert "model_name" in info
            assert "model_path" in info
            assert "num_labels" in info
            assert "supported_categories" in info
            assert "device" in info
            assert info["model_name"] == mock_config.model_name


class TestDocumentClassifierIntegration:
    """Integration tests for DocumentClassifier"""
    
    @pytest.fixture
    def sample_training_data(self):
        """Sample training data for testing"""
        return [
            ("This is a witness statement from John Doe about the incident on Main Street.", "witness_statement"),
            ("Police report filed by Officer Smith regarding the traffic violation.", "police_report"),
            ("Medical examination results for patient showing signs of injury.", "medical_record"),
            ("Court filing motion for summary judgment in case 2023-CV-001.", "court_filing"),
        ]
    
    @patch('lemkin_classifier.core.AutoTokenizer')
    @patch('lemkin_classifier.core.AutoModelForSequenceClassification')
    @patch('lemkin_classifier.core.pipeline')
    @patch('lemkin_classifier.core.Trainer')
    def test_evaluate_model(self, mock_trainer, mock_pipeline, mock_model, mock_tokenizer, mock_config, sample_training_data):
        """Test model evaluation"""
        # Mock components
        mock_tokenizer.from_pretrained.return_value = MagicMock()
        mock_model.from_pretrained.return_value = MagicMock()
        
        # Mock pipeline for classification
        def mock_classify(text):
            if "witness" in text.lower():
                return [{'label': 'witness_statement', 'score': 0.85}]
            elif "police" in text.lower():
                return [{'label': 'police_report', 'score': 0.90}]
            elif "medical" in text.lower():
                return [{'label': 'medical_record', 'score': 0.80}]
            elif "court" in text.lower():
                return [{'label': 'court_filing', 'score': 0.75}]
            else:
                return [{'label': 'unknown', 'score': 0.60}]
        
        mock_pipeline_instance = MagicMock()
        mock_pipeline_instance.side_effect = mock_classify
        mock_pipeline.return_value = mock_pipeline_instance
        
        classifier = DocumentClassifier(mock_config)
        
        # Mock the classify_document method to return predictable results
        def mock_classify_document(content):
            mock_result = MagicMock()
            mock_result.classification.document_type.value = mock_classify(content.text)[0]['label']
            return mock_result
        
        classifier.classify_document = mock_classify_document
        
        # Run evaluation
        metrics = classifier.evaluate_model(sample_training_data)
        
        assert isinstance(metrics, ModelMetrics)
        assert 0 <= metrics.accuracy <= 1
        assert 0 <= metrics.precision <= 1
        assert 0 <= metrics.recall <= 1
        assert 0 <= metrics.f1_score <= 1
        assert metrics.test_set_size == len(sample_training_data)


@pytest.fixture
def temp_pdf_file():
    """Create a temporary PDF file for testing"""
    with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp_file:
        tmp_path = Path(tmp_file.name)
    yield tmp_path
    if tmp_path.exists():
        os.unlink(tmp_path)


@pytest.fixture
def temp_text_file():
    """Create a temporary text file for testing"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as tmp_file:
        tmp_file.write("This is a sample legal document for testing purposes.")
        tmp_path = Path(tmp_file.name)
    yield tmp_path
    if tmp_path.exists():
        os.unlink(tmp_path)


class TestFileHandling:
    """Test file handling functionality"""
    
    def test_text_file_extraction(self, temp_text_file, mock_config):
        """Test extracting text from text file"""
        with patch('lemkin_classifier.core.AutoTokenizer'), \
             patch('lemkin_classifier.core.AutoModelForSequenceClassification'), \
             patch('lemkin_classifier.core.pipeline'):
            
            classifier = DocumentClassifier(mock_config)
            content = classifier.extract_text_from_file(temp_text_file)
            
            assert content.text == "This is a sample legal document for testing purposes."
            assert content.file_type == "text"
            assert content.file_path == str(temp_text_file)
            assert content.length > 0
            assert content.word_count > 0


@pytest.fixture
def mock_config():
    """Mock configuration for testing"""
    return ClassificationConfig(
        model_name="distilbert-base-uncased",
        max_length=512,
        confidence_threshold=0.7,
        batch_size=8
    )