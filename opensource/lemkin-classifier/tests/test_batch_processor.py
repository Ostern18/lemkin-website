"""
Tests for the batch processor module.
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
import tempfile
import os
from pathlib import Path
from datetime import datetime
import asyncio

from lemkin_classifier.batch_processor import (
    BatchProcessor,
    DocumentBatch,
    ProcessingConfig,
    ProcessingMode,
    ProcessingStatus,
    BatchMetrics,
    BatchProcessingResult,
    ProcessingTask
)
from lemkin_classifier.core import (
    DocumentClassifier,
    ClassificationResult,
    DocumentClassification,
    DocumentContent,
    ClassificationConfig
)
from lemkin_classifier.confidence_scorer import ConfidenceScorer
from lemkin_classifier.legal_taxonomy import DocumentType, LegalDomain


class TestDocumentBatch:
    """Test DocumentBatch model"""
    
    def test_batch_creation(self):
        """Test creating a DocumentBatch instance"""
        batch = DocumentBatch(
            batch_id="test_batch_001",
            documents=["/path/to/doc1.pdf", "/path/to/doc2.docx"],
            batch_name="Test Batch",
            metadata={"source": "test_data"},
            priority=1,
            max_retries=3,
            timeout_seconds=300
        )
        
        assert batch.batch_id == "test_batch_001"
        assert len(batch.documents) == 2
        assert batch.batch_name == "Test Batch"
        assert batch.metadata["source"] == "test_data"
        assert batch.priority == 1
        assert batch.max_retries == 3
        assert batch.timeout_seconds == 300
        assert isinstance(batch.created_at, datetime)
    
    def test_batch_length(self):
        """Test batch length calculation"""
        batch = DocumentBatch(
            batch_id="test",
            documents=["doc1", "doc2", "doc3"]
        )
        
        assert len(batch) == 3
    
    def test_batch_empty_documents_validation(self):
        """Test validation of empty documents list"""
        with pytest.raises(ValueError, match="Document list cannot be empty"):
            DocumentBatch(
                batch_id="test",
                documents=[]
            )


class TestProcessingConfig:
    """Test ProcessingConfig model"""
    
    def test_default_config(self):
        """Test default processing configuration"""
        config = ProcessingConfig()
        
        assert config.max_workers == 4
        assert config.batch_size == 32
        assert config.processing_mode == ProcessingMode.THREADED
        assert config.memory_limit_gb is None
        assert config.gpu_enabled is True
        assert config.cache_size == 1000
        assert config.fail_fast is False
        assert config.continue_on_error is True
        assert config.error_threshold == 0.1
        assert config.output_format == "json"
        assert config.enable_progress_bar is True
        assert config.log_interval == 100
        assert config.checkpoint_interval == 500
    
    def test_custom_config(self):
        """Test custom processing configuration"""
        config = ProcessingConfig(
            max_workers=8,
            batch_size=16,
            processing_mode=ProcessingMode.MULTIPROCESS,
            memory_limit_gb=8.0,
            gpu_enabled=False,
            fail_fast=True,
            error_threshold=0.05,
            output_format="csv"
        )
        
        assert config.max_workers == 8
        assert config.batch_size == 16
        assert config.processing_mode == ProcessingMode.MULTIPROCESS
        assert config.memory_limit_gb == 8.0
        assert config.gpu_enabled is False
        assert config.fail_fast is True
        assert config.error_threshold == 0.05
        assert config.output_format == "csv"


class TestBatchMetrics:
    """Test BatchMetrics model"""
    
    def test_metrics_creation(self):
        """Test creating BatchMetrics instance"""
        start_time = datetime.now()
        end_time = datetime.now()
        
        metrics = BatchMetrics(
            start_time=start_time,
            end_time=end_time,
            total_duration=300.5,
            average_time_per_document=2.5,
            total_documents=120,
            successful_documents=115,
            failed_documents=5,
            skipped_documents=0,
            average_confidence=0.82,
            confidence_distribution={"high": 80, "medium": 35, "low": 5},
            review_required_count=15,
            documents_per_second=0.4,
            error_rate=0.042,
            error_types={"FileNotFound": 3, "ProcessingError": 2},
            retry_count=8
        )
        
        assert metrics.start_time == start_time
        assert metrics.end_time == end_time
        assert metrics.total_duration == 300.5
        assert metrics.total_documents == 120
        assert metrics.successful_documents == 115
        assert metrics.failed_documents == 5
        assert metrics.error_rate == 0.042
        assert len(metrics.error_types) == 2


class TestBatchProcessingResult:
    """Test BatchProcessingResult model"""
    
    def test_result_creation(self):
        """Test creating BatchProcessingResult instance"""
        # Create mock components
        mock_results = [Mock(spec=ClassificationResult) for _ in range(3)]
        mock_metrics = Mock(spec=BatchMetrics)
        mock_processing_config = Mock(spec=ProcessingConfig)
        mock_classification_config = Mock(spec=ClassificationConfig)
        
        result = BatchProcessingResult(
            batch_id="test_batch",
            status=ProcessingStatus.COMPLETED,
            results=mock_results,
            failed_documents=[{"error": "test error"}],
            metrics=mock_metrics,
            processing_config=mock_processing_config,
            classification_config=mock_classification_config,
            output_files=["results.json", "metrics.json"],
            checkpoint_files=["checkpoint_1.json"]
        )
        
        assert result.batch_id == "test_batch"
        assert result.status == ProcessingStatus.COMPLETED
        assert len(result.results) == 3
        assert len(result.failed_documents) == 1
        assert len(result.output_files) == 2
        assert len(result.checkpoint_files) == 1


class TestProcessingTask:
    """Test ProcessingTask dataclass"""
    
    def test_task_creation(self):
        """Test creating ProcessingTask instance"""
        task = ProcessingTask(
            document_id="doc_001",
            document="/path/to/document.pdf",
            batch_id="batch_001",
            retry_count=0
        )
        
        assert task.document_id == "doc_001"
        assert task.document == "/path/to/document.pdf"
        assert task.batch_id == "batch_001"
        assert task.retry_count == 0
        assert task.start_time is None
        assert task.end_time is None
        assert task.result is None
        assert task.error is None


class TestBatchProcessor:
    """Test BatchProcessor class"""
    
    @pytest.fixture
    def mock_classifier(self):
        """Mock DocumentClassifier for testing"""
        classifier = Mock(spec=DocumentClassifier)
        classifier.config = Mock(spec=ClassificationConfig)
        return classifier
    
    @pytest.fixture
    def mock_confidence_scorer(self):
        """Mock ConfidenceScorer for testing"""
        return Mock(spec=ConfidenceScorer)
    
    @pytest.fixture
    def processor_config(self):
        """Test processing configuration"""
        return ProcessingConfig(
            max_workers=2,
            batch_size=4,
            processing_mode=ProcessingMode.SEQUENTIAL,
            enable_progress_bar=False
        )
    
    @pytest.fixture
    def batch_processor(self, mock_classifier, mock_confidence_scorer, processor_config):
        """Create BatchProcessor instance for testing"""
        return BatchProcessor(mock_classifier, mock_confidence_scorer, processor_config)
    
    def test_processor_initialization(self, batch_processor, mock_classifier, mock_confidence_scorer, processor_config):
        """Test BatchProcessor initialization"""
        assert batch_processor.classifier == mock_classifier
        assert batch_processor.confidence_scorer == mock_confidence_scorer
        assert batch_processor.config == processor_config
        assert batch_processor.processing_status == ProcessingStatus.PENDING
        assert batch_processor.current_batch is None
        assert isinstance(batch_processor.results_cache, dict)
        assert isinstance(batch_processor.error_counts, dict)
    
    def test_create_sample_batch(self):
        """Test creating a sample document batch"""
        documents = ["doc1.pdf", "doc2.docx", "doc3.txt"]
        
        batch = DocumentBatch(
            batch_id="test_batch",
            documents=documents,
            batch_name="Sample Batch"
        )
        
        assert batch.batch_id == "test_batch"
        assert len(batch.documents) == 3
        assert batch.batch_name == "Sample Batch"
    
    @patch('lemkin_classifier.batch_processor.tqdm')
    def test_process_sequential(self, mock_tqdm, batch_processor, mock_classifier):
        """Test sequential processing mode"""
        # Mock classification result
        mock_result = Mock(spec=ClassificationResult)
        mock_result.classification.confidence_score = 0.8
        mock_result.requires_review = False
        batch_processor._process_single_document = Mock(return_value=mock_result)
        
        # Create test tasks
        tasks = [
            ProcessingTask("doc1", "content1", "batch1"),
            ProcessingTask("doc2", "content2", "batch1"),
        ]
        
        # Set up progress bar mock
        mock_progress = Mock()
        mock_tqdm.return_value.__enter__.return_value = mock_progress
        
        results, failed = batch_processor._process_sequential(tasks)
        
        assert len(results) == 2
        assert len(failed) == 0
        assert batch_processor._process_single_document.call_count == 2
    
    @patch('lemkin_classifier.batch_processor.ThreadPoolExecutor')
    @patch('lemkin_classifier.batch_processor.tqdm')
    def test_process_threaded(self, mock_tqdm, mock_executor, batch_processor):
        """Test threaded processing mode"""
        # Mock future results
        mock_future1 = Mock()
        mock_future2 = Mock()
        
        mock_result1 = Mock(spec=ClassificationResult)
        mock_result1.classification.confidence_score = 0.8
        
        mock_result2 = Mock(spec=ClassificationResult)
        mock_result2.classification.confidence_score = 0.75
        
        mock_future1.result.return_value = mock_result1
        mock_future2.result.return_value = mock_result2
        
        # Mock executor
        mock_executor_instance = Mock()
        mock_executor_instance.__enter__.return_value = mock_executor_instance
        mock_executor.return_value = mock_executor_instance
        
        # Mock submit method
        mock_executor_instance.submit.side_effect = [mock_future1, mock_future2]
        
        # Mock as_completed
        with patch('lemkin_classifier.batch_processor.as_completed') as mock_as_completed:
            mock_as_completed.return_value = [mock_future1, mock_future2]
            
            # Mock progress bar
            mock_progress = Mock()
            mock_tqdm.return_value.__enter__.return_value = mock_progress
            
            # Create test tasks
            tasks = [
                ProcessingTask("doc1", "content1", "batch1"),
                ProcessingTask("doc2", "content2", "batch1"),
            ]
            
            batch_processor._process_single_document = Mock()
            
            results, failed = batch_processor._process_threaded(tasks)
            
            assert len(results) == 2
            assert len(failed) == 0
    
    def test_process_single_document_with_file(self, batch_processor, mock_classifier):
        """Test processing single document from file"""
        mock_result = Mock(spec=ClassificationResult)
        mock_classifier.classify_file.return_value = mock_result
        
        result = batch_processor._process_single_document("/path/to/document.pdf")
        
        assert result == mock_result
        mock_classifier.classify_file.assert_called_once_with("/path/to/document.pdf")
    
    def test_process_single_document_with_content(self, batch_processor, mock_classifier):
        """Test processing single document from DocumentContent"""
        document_content = DocumentContent(text="Sample text", metadata={})
        mock_result = Mock(spec=ClassificationResult)
        mock_classifier.classify_document.return_value = mock_result
        
        result = batch_processor._process_single_document(document_content)
        
        assert result == mock_result
        mock_classifier.classify_document.assert_called_once_with(document_content)
    
    def test_calculate_confidence_distribution(self, batch_processor):
        """Test confidence distribution calculation"""
        # Create mock results with different confidence scores
        mock_results = []
        confidence_scores = [0.95, 0.85, 0.75, 0.65, 0.35, 0.25]
        
        for score in confidence_scores:
            mock_result = Mock()
            mock_result.classification.confidence_score = score
            mock_results.append(mock_result)
        
        distribution = batch_processor._calculate_confidence_distribution(mock_results)
        
        expected_distribution = {
            'very_high': 1,  # 0.95
            'high': 1,       # 0.85
            'medium': 2,     # 0.75, 0.65
            'low': 1,        # 0.35
            'very_low': 1    # 0.25
        }
        
        assert distribution == expected_distribution
    
    def test_flatten_result(self, batch_processor):
        """Test flattening classification result for CSV"""
        # Create mock result
        mock_content = Mock()
        mock_content.file_path = "/path/to/document.pdf"
        mock_content.length = 1000
        mock_content.word_count = 200
        
        mock_classification = Mock()
        mock_classification.document_type.value = "witness_statement"
        mock_classification.legal_domain.value = "criminal_law"
        mock_classification.confidence_score = 0.85
        
        mock_result = Mock()
        mock_result.document_content = mock_content
        mock_result.classification = mock_classification
        mock_result.urgency_level = "high"
        mock_result.sensitivity_level = "confidential"
        mock_result.requires_review = True
        mock_result.review_reasons = ["Low confidence", "Sensitive content"]
        mock_result.processing_time = 2.5
        
        flattened = batch_processor._flatten_result(mock_result)
        
        assert flattened["file_path"] == "/path/to/document.pdf"
        assert flattened["document_type"] == "witness_statement"
        assert flattened["legal_domain"] == "criminal_law"
        assert flattened["confidence_score"] == 0.85
        assert flattened["urgency_level"] == "high"
        assert flattened["requires_review"] is True
        assert flattened["review_reasons"] == "Low confidence; Sensitive content"
        assert flattened["processing_time"] == 2.5
    
    def test_create_batch_from_directory_not_found(self, batch_processor):
        """Test creating batch from non-existent directory"""
        non_existent_dir = Path("/non/existent/directory")
        
        with pytest.raises(FileNotFoundError):
            batch_processor.create_batch_from_directory(non_existent_dir)
    
    def test_stop_processing(self, batch_processor):
        """Test stopping processing"""
        batch_processor.stop_processing()
        
        assert batch_processor._stop_event.is_set()
        assert batch_processor.processing_status == ProcessingStatus.CANCELLED
    
    def test_pause_and_resume_processing(self, batch_processor):
        """Test pausing and resuming processing"""
        # Test pause
        batch_processor.pause_processing()
        assert batch_processor._pause_event.is_set()
        assert batch_processor.processing_status == ProcessingStatus.PAUSED
        
        # Test resume
        batch_processor.resume_processing()
        assert not batch_processor._pause_event.is_set()
        assert batch_processor.processing_status == ProcessingStatus.RUNNING
    
    def test_get_processing_status(self, batch_processor):
        """Test getting processing status"""
        status = batch_processor.get_processing_status()
        
        assert isinstance(status, dict)
        assert "status" in status
        assert "current_batch" in status
        assert "cache_size" in status
        assert "error_counts" in status
        assert "is_stopped" in status
        assert "is_paused" in status
        
        assert status["status"] == ProcessingStatus.PENDING.value
        assert status["current_batch"] is None
        assert status["cache_size"] == 0
        assert status["is_stopped"] is False
        assert status["is_paused"] is False


class TestBatchProcessorIntegration:
    """Integration tests for BatchProcessor"""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory with test files"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            
            # Create test files
            (tmp_path / "doc1.txt").write_text("This is a witness statement.")
            (tmp_path / "doc2.txt").write_text("Police report filed by officer.")
            (tmp_path / "doc3.txt").write_text("Medical examination results.")
            
            yield tmp_path
    
    def test_create_batch_from_directory(self, temp_dir):
        """Test creating batch from directory with actual files"""
        # Mock the dependencies to avoid loading actual models
        with patch('lemkin_classifier.core.AutoTokenizer'), \
             patch('lemkin_classifier.core.AutoModelForSequenceClassification'), \
             patch('lemkin_classifier.core.pipeline'):
            
            mock_classifier = Mock()
            mock_confidence_scorer = Mock()
            config = ProcessingConfig()
            
            processor = BatchProcessor(mock_classifier, mock_confidence_scorer, config)
            
            batch = processor.create_batch_from_directory(
                temp_dir,
                batch_name="Test Directory Batch",
                file_patterns=["*.txt"],
                recursive=False
            )
            
            assert len(batch.documents) == 3
            assert batch.batch_name == "Test Directory Batch"
            assert batch.metadata["source_directory"] == str(temp_dir)
            assert batch.metadata["total_files"] == 3
            
            # Check that all files are found
            file_names = [Path(doc).name for doc in batch.documents]
            assert "doc1.txt" in file_names
            assert "doc2.txt" in file_names
            assert "doc3.txt" in file_names
    
    @patch('lemkin_classifier.batch_processor.tqdm')
    def test_full_batch_processing(self, mock_tqdm, temp_dir):
        """Test complete batch processing workflow"""
        # Mock progress bar
        mock_progress = Mock()
        mock_tqdm.return_value.__enter__.return_value = mock_progress
        
        # Mock classifier and results
        mock_classifier = Mock()
        mock_confidence_scorer = Mock()
        
        # Create mock classification results
        def create_mock_result(confidence):
            mock_result = Mock(spec=ClassificationResult)
            mock_result.classification.confidence_score = confidence
            mock_result.requires_review = confidence < 0.7
            mock_result.urgency_level = "medium"
            mock_result.sensitivity_level = "internal"
            mock_result.processing_time = 1.0
            mock_result.document_content.file_path = "test_path"
            mock_result.document_content.length = 500
            mock_result.document_content.word_count = 100
            mock_result.classification.document_type.value = "witness_statement"
            mock_result.classification.legal_domain.value = "criminal_law"
            mock_result.review_reasons = []
            return mock_result
        
        mock_results = [
            create_mock_result(0.8),
            create_mock_result(0.9),
            create_mock_result(0.6)
        ]
        
        mock_classifier.classify_file.side_effect = mock_results
        
        # Create processor and batch
        config = ProcessingConfig(
            processing_mode=ProcessingMode.SEQUENTIAL,
            enable_progress_bar=False
        )
        processor = BatchProcessor(mock_classifier, mock_confidence_scorer, config)
        
        batch = DocumentBatch(
            batch_id="test_batch",
            documents=list(temp_dir.glob("*.txt")),
            batch_name="Integration Test Batch"
        )
        
        # Process batch
        result = processor.process_batch(batch)
        
        # Verify results
        assert result.status == ProcessingStatus.COMPLETED
        assert len(result.results) == 3
        assert len(result.failed_documents) == 0
        assert result.metrics.successful_documents == 3
        assert result.metrics.failed_documents == 0
        assert result.metrics.error_rate == 0.0
        assert result.metrics.review_required_count == 1  # One result has confidence < 0.7


class TestAsyncProcessing:
    """Test async processing functionality"""
    
    @pytest.mark.asyncio
    async def test_process_async(self):
        """Test async processing mode"""
        # Mock classifier
        mock_classifier = Mock()
        mock_result = Mock(spec=ClassificationResult)
        mock_result.classification.confidence_score = 0.8
        mock_classifier.classify_file.return_value = mock_result
        
        # Mock confidence scorer
        mock_confidence_scorer = Mock()
        
        # Create processor
        config = ProcessingConfig(
            max_workers=2,
            processing_mode=ProcessingMode.ASYNC,
            enable_progress_bar=False
        )
        processor = BatchProcessor(mock_classifier, mock_confidence_scorer, config)
        
        # Create test tasks
        tasks = [
            ProcessingTask("doc1", "content1", "batch1"),
            ProcessingTask("doc2", "content2", "batch1"),
        ]
        
        # Run async processing
        results, failed = await processor._process_async(tasks)
        
        assert len(results) == 2
        assert len(failed) == 0


class TestErrorHandling:
    """Test error handling in batch processing"""
    
    def test_processing_with_errors(self):
        """Test batch processing with individual document errors"""
        # Mock classifier that raises exceptions for some documents
        mock_classifier = Mock()
        mock_confidence_scorer = Mock()
        
        def side_effect(document):
            if "error" in str(document):
                raise ValueError("Processing error")
            else:
                mock_result = Mock(spec=ClassificationResult)
                mock_result.classification.confidence_score = 0.8
                mock_result.requires_review = False
                return mock_result
        
        mock_classifier.classify_file.side_effect = side_effect
        
        # Create processor
        config = ProcessingConfig(
            processing_mode=ProcessingMode.SEQUENTIAL,
            continue_on_error=True,
            enable_progress_bar=False
        )
        processor = BatchProcessor(mock_classifier, mock_confidence_scorer, config)
        
        # Create batch with some problematic documents
        batch = DocumentBatch(
            batch_id="error_test",
            documents=["good_doc1.txt", "error_doc.txt", "good_doc2.txt"]
        )
        
        # Process batch
        with patch('lemkin_classifier.batch_processor.tqdm'):
            result = processor.process_batch(batch)
        
        # Should have 2 successful results and 1 failed
        assert result.metrics.successful_documents == 2
        assert result.metrics.failed_documents == 1
        assert len(result.failed_documents) == 1
        assert result.metrics.error_rate == 1/3


@pytest.fixture
def sample_documents():
    """Fixture providing sample document paths"""
    return [
        "/path/to/witness_statement.pdf",
        "/path/to/police_report.docx",
        "/path/to/medical_record.txt"
    ]


@pytest.fixture
def sample_batch(sample_documents):
    """Fixture providing sample document batch"""
    return DocumentBatch(
        batch_id="sample_batch_001",
        documents=sample_documents,
        batch_name="Sample Legal Documents",
        metadata={"case_id": "CASE-2024-001", "jurisdiction": "US"},
        priority=1
    )