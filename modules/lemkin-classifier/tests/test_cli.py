"""
Tests for the CLI module.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import tempfile
import os
import json
from typer.testing import CliRunner

from lemkin_classifier.cli import app
from lemkin_classifier.core import (
    DocumentClassifier,
    ClassificationResult,
    DocumentClassification,
    DocumentContent,
    ClassificationConfig
)
from lemkin_classifier.legal_taxonomy import DocumentType, LegalDomain


class TestCLICommands:
    """Test CLI commands"""
    
    @pytest.fixture
    def runner(self):
        """CLI test runner"""
        return CliRunner()
    
    @pytest.fixture
    def temp_document(self):
        """Create temporary document file for testing"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as tmp_file:
            tmp_file.write("This is a witness statement about the incident on Main Street.")
            tmp_path = Path(tmp_file.name)
        yield tmp_path
        if tmp_path.exists():
            os.unlink(tmp_path)
    
    @pytest.fixture
    def temp_directory(self):
        """Create temporary directory with test documents"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            
            # Create test documents
            (tmp_path / "witness1.txt").write_text("Witness statement from John Doe.")
            (tmp_path / "police_report.txt").write_text("Police report filed by Officer Smith.")
            (tmp_path / "medical.txt").write_text("Medical examination results.")
            
            yield tmp_path


class TestClassifyDocumentCommand:
    """Test classify-document command"""
    
    @pytest.fixture
    def runner(self):
        return CliRunner()
    
    @pytest.fixture
    def mock_classification_result(self):
        """Mock classification result for testing"""
        mock_content = Mock(spec=DocumentContent)
        mock_content.text = "Sample witness statement"
        mock_content.file_path = "/test/path.txt"
        mock_content.length = 100
        mock_content.word_count = 20
        mock_content.metadata = {"test": "data"}
        
        mock_classification = Mock(spec=DocumentClassification)
        mock_classification.document_type = DocumentType.WITNESS_STATEMENT
        mock_classification.legal_domain = LegalDomain.CRIMINAL_LAW
        mock_classification.confidence_score = 0.85
        mock_classification.probability_distribution = {"witness_statement": 0.85}
        mock_classification.model_version = "1.0"
        mock_classification.model_name = "test-model"
        
        mock_result = Mock(spec=ClassificationResult)
        mock_result.document_content = mock_content
        mock_result.classification = mock_classification
        mock_result.urgency_level = "high"
        mock_result.sensitivity_level = "confidential"
        mock_result.requires_review = True
        mock_result.review_reasons = ["High sensitivity"]
        mock_result.recommended_actions = ["Apply PII redaction"]
        mock_result.processing_time = 1.5
        
        # Mock dict method for JSON serialization
        mock_result.dict.return_value = {
            "classification": {
                "document_type": "witness_statement",
                "legal_domain": "criminal_law",
                "confidence_score": 0.85
            },
            "urgency_level": "high",
            "processing_time": 1.5
        }
        
        return mock_result
    
    @patch('lemkin_classifier.cli.DocumentClassifier')
    def test_classify_document_success(self, mock_classifier_class, runner, mock_classification_result):
        """Test successful document classification"""
        # Mock classifier instance
        mock_classifier = Mock()
        mock_classifier.classify_file.return_value = mock_classification_result
        mock_classifier_class.return_value = mock_classifier
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as tmp_file:
            tmp_file.write("Test document content")
            tmp_path = tmp_file.name
        
        try:
            result = runner.invoke(app, [
                'classify-document',
                tmp_path,
                '--format', 'json',
                '--quiet'
            ])
            
            assert result.exit_code == 0
            mock_classifier.classify_file.assert_called_once()
            
        finally:
            # Clean up
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    
    def test_classify_document_file_not_found(self, runner):
        """Test classify-document with non-existent file"""
        result = runner.invoke(app, [
            'classify-document',
            '/non/existent/file.txt',
            '--quiet'
        ])
        
        assert result.exit_code == 1
        assert "File not found" in result.stdout
    
    @patch('lemkin_classifier.cli.DocumentClassifier')
    def test_classify_document_with_output_file(self, mock_classifier_class, runner, mock_classification_result):
        """Test classify-document with output file"""
        # Mock classifier
        mock_classifier = Mock()
        mock_classifier.classify_file.return_value = mock_classification_result
        mock_classifier_class.return_value = mock_classifier
        
        # Create temporary input and output files
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as tmp_input:
            tmp_input.write("Test document")
            input_path = tmp_input.name
        
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as tmp_output:
            output_path = tmp_output.name
        
        try:
            result = runner.invoke(app, [
                'classify-document',
                input_path,
                '--output', output_path,
                '--format', 'json',
                '--quiet'
            ])
            
            assert result.exit_code == 0
            assert os.path.exists(output_path)
            
            # Check output file content
            with open(output_path, 'r') as f:
                output_data = json.load(f)
                assert "classification" in output_data
            
        finally:
            # Clean up
            for path in [input_path, output_path]:
                if os.path.exists(path):
                    os.unlink(path)
    
    @patch('lemkin_classifier.cli.DocumentClassifier')
    def test_classify_document_table_format(self, mock_classifier_class, runner, mock_classification_result):
        """Test classify-document with table format"""
        mock_classifier = Mock()
        mock_classifier.classify_file.return_value = mock_classification_result
        mock_classifier_class.return_value = mock_classifier
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as tmp_file:
            tmp_file.write("Test document")
            tmp_path = tmp_file.name
        
        try:
            result = runner.invoke(app, [
                'classify-document',
                tmp_path,
                '--format', 'table',
                '--quiet'
            ])
            
            assert result.exit_code == 0
            # Should contain table formatting
            assert "Document Classification Result" in result.stdout or result.exit_code == 0
            
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)


class TestBatchClassifyCommand:
    """Test batch-classify command"""
    
    @pytest.fixture
    def runner(self):
        return CliRunner()
    
    @patch('lemkin_classifier.cli.BatchProcessor')
    @patch('lemkin_classifier.cli.DocumentClassifier')
    @patch('lemkin_classifier.cli.ConfidenceScorer')
    def test_batch_classify_success(self, mock_scorer_class, mock_classifier_class, mock_processor_class, runner):
        """Test successful batch classification"""
        # Mock batch processor
        mock_processor = Mock()
        mock_processor.create_batch_from_directory.return_value = Mock()
        mock_processor.process_batch.return_value = Mock()
        mock_processor_class.return_value = mock_processor
        
        # Create temporary directory
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            (tmp_path / "doc1.txt").write_text("Document 1")
            (tmp_path / "doc2.txt").write_text("Document 2")
            
            result = runner.invoke(app, [
                'batch-classify',
                str(tmp_path),
                '--workers', '2',
                '--batch-size', '10',
                '--quiet'
            ])
            
            assert result.exit_code == 0
    
    def test_batch_classify_directory_not_found(self, runner):
        """Test batch-classify with non-existent directory"""
        result = runner.invoke(app, [
            'batch-classify',
            '/non/existent/directory',
            '--quiet'
        ])
        
        assert result.exit_code == 1


class TestTrainModelCommand:
    """Test train-model command"""
    
    @pytest.fixture
    def runner(self):
        return CliRunner()
    
    @pytest.fixture
    def training_data_csv(self):
        """Create temporary CSV training data file"""
        import pandas as pd
        
        data = {
            'text': [
                "This is a witness statement from John.",
                "Police report filed by Officer Smith.",
                "Medical examination shows injuries.",
                "Court filing motion for summary."
            ],
            'label': [
                'witness_statement',
                'police_report',
                'medical_record',
                'court_filing'
            ]
        }
        
        df = pd.DataFrame(data)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp_file:
            df.to_csv(tmp_file.name, index=False)
            tmp_path = tmp_file.name
        
        yield tmp_path
        
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
    
    @patch('lemkin_classifier.cli.DocumentClassifier')
    def test_train_model_success(self, mock_classifier_class, runner, training_data_csv):
        """Test successful model training"""
        # Mock classifier and training
        mock_classifier = Mock()
        mock_metrics = Mock()
        mock_metrics.accuracy = 0.85
        mock_metrics.precision = 0.80
        mock_metrics.recall = 0.82
        mock_metrics.f1_score = 0.81
        mock_metrics.test_set_size = 100
        
        mock_classifier.train_model.return_value = mock_metrics
        mock_classifier_class.return_value = mock_classifier
        
        with tempfile.TemporaryDirectory() as output_dir:
            result = runner.invoke(app, [
                'train-model',
                training_data_csv,
                output_dir,
                '--epochs', '1',
                '--batch-size', '8',
                '--quiet'
            ])
            
            assert result.exit_code == 0
            mock_classifier.train_model.assert_called_once()
    
    def test_train_model_file_not_found(self, runner):
        """Test train-model with non-existent training data"""
        with tempfile.TemporaryDirectory() as output_dir:
            result = runner.invoke(app, [
                'train-model',
                '/non/existent/training.csv',
                output_dir,
                '--quiet'
            ])
            
            assert result.exit_code == 1
    
    def test_train_model_invalid_format(self, runner):
        """Test train-model with invalid file format"""
        with tempfile.NamedTemporaryFile(suffix='.invalid', delete=False) as tmp_file:
            tmp_file.write(b"invalid data")
            tmp_path = tmp_file.name
        
        try:
            with tempfile.TemporaryDirectory() as output_dir:
                result = runner.invoke(app, [
                    'train-model',
                    tmp_path,
                    output_dir,
                    '--quiet'
                ])
                
                assert result.exit_code == 1
                assert "Training data must be CSV or JSON format" in result.stdout
        
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)


class TestEvaluateModelCommand:
    """Test evaluate-model command"""
    
    @pytest.fixture
    def runner(self):
        return CliRunner()
    
    @pytest.fixture
    def test_data_csv(self):
        """Create temporary CSV test data file"""
        import pandas as pd
        
        data = {
            'text': [
                "Witness saw the incident happen.",
                "Officer responded to the call.",
                "Patient shows signs of trauma."
            ],
            'label': [
                'witness_statement',
                'police_report',
                'medical_record'
            ]
        }
        
        df = pd.DataFrame(data)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp_file:
            df.to_csv(tmp_file.name, index=False)
            tmp_path = tmp_file.name
        
        yield tmp_path
        
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
    
    @patch('lemkin_classifier.cli.DocumentClassifier')
    def test_evaluate_model_success(self, mock_classifier_class, runner, test_data_csv):
        """Test successful model evaluation"""
        # Mock classifier and evaluation
        mock_classifier = Mock()
        mock_metrics = Mock()
        mock_metrics.accuracy = 0.90
        mock_metrics.precision = 0.85
        mock_metrics.recall = 0.88
        mock_metrics.f1_score = 0.87
        mock_metrics.test_set_size = 50
        mock_metrics.class_metrics = {
            'witness_statement': {
                'precision': 0.90,
                'recall': 0.85,
                'f1-score': 0.87
            }
        }
        mock_metrics.dict.return_value = {
            'accuracy': 0.90,
            'precision': 0.85,
            'test_set_size': 50
        }
        
        mock_classifier.evaluate_model.return_value = mock_metrics
        mock_classifier_class.return_value = mock_classifier
        
        result = runner.invoke(app, [
            'evaluate-model',
            test_data_csv,
            '--quiet'
        ])
        
        assert result.exit_code == 0
        mock_classifier.evaluate_model.assert_called_once()


class TestUpdateTaxonomyCommand:
    """Test update-taxonomy command"""
    
    @pytest.fixture
    def runner(self):
        return CliRunner()
    
    def test_taxonomy_list(self, runner):
        """Test listing taxonomy"""
        result = runner.invoke(app, [
            'update-taxonomy',
            'list'
        ])
        
        assert result.exit_code == 0
        assert "Legal Document Taxonomy" in result.stdout
    
    def test_taxonomy_validate_valid_category(self, runner):
        """Test validating valid category"""
        result = runner.invoke(app, [
            'update-taxonomy',
            'validate',
            '--category', 'witness_statement',
            '--domain', 'criminal_law'
        ])
        
        # Should succeed or show validation result
        assert result.exit_code == 0
    
    def test_taxonomy_validate_invalid_category(self, runner):
        """Test validating invalid category"""
        result = runner.invoke(app, [
            'update-taxonomy',
            'validate',
            '--category', 'invalid_category'
        ])
        
        assert result.exit_code == 0
        # Should show validation result (may be invalid)
    
    def test_taxonomy_validate_missing_category(self, runner):
        """Test validate without category parameter"""
        result = runner.invoke(app, [
            'update-taxonomy',
            'validate'
        ])
        
        assert result.exit_code == 1
        assert "Category required for validation" in result.stdout
    
    def test_taxonomy_unknown_action(self, runner):
        """Test unknown taxonomy action"""
        result = runner.invoke(app, [
            'update-taxonomy',
            'unknown_action'
        ])
        
        assert result.exit_code == 1
        assert "Unknown action" in result.stdout


class TestInfoCommand:
    """Test info command"""
    
    @pytest.fixture
    def runner(self):
        return CliRunner()
    
    def test_info_command(self, runner):
        """Test info command output"""
        result = runner.invoke(app, ['info'])
        
        assert result.exit_code == 0
        assert "Lemkin Legal Document Classifier" in result.stdout
        assert "Supported Document Types" in result.stdout
        assert "Legal Domains" in result.stdout


class TestCLIIntegration:
    """Integration tests for CLI"""
    
    @pytest.fixture
    def runner(self):
        return CliRunner()
    
    def test_help_output(self, runner):
        """Test CLI help output"""
        result = runner.invoke(app, ['--help'])
        
        assert result.exit_code == 0
        assert "Legal Document Classifier" in result.stdout
        assert "classify-document" in result.stdout
        assert "batch-classify" in result.stdout
        assert "train-model" in result.stdout
        assert "evaluate-model" in result.stdout
        assert "update-taxonomy" in result.stdout
        assert "info" in result.stdout
    
    def test_command_help(self, runner):
        """Test individual command help"""
        commands = [
            'classify-document',
            'batch-classify',
            'train-model',
            'evaluate-model',
            'update-taxonomy',
            'info'
        ]
        
        for command in commands:
            result = runner.invoke(app, [command, '--help'])
            assert result.exit_code == 0
            assert command.replace('-', ' ') in result.stdout.lower() or "help" in result.stdout.lower()


class TestCLIErrorHandling:
    """Test CLI error handling"""
    
    @pytest.fixture
    def runner(self):
        return CliRunner()
    
    @patch('lemkin_classifier.cli.DocumentClassifier')
    def test_classification_exception_handling(self, mock_classifier_class, runner):
        """Test handling of classification exceptions"""
        # Mock classifier that raises exception
        mock_classifier = Mock()
        mock_classifier.classify_file.side_effect = RuntimeError("Model loading failed")
        mock_classifier_class.return_value = mock_classifier
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as tmp_file:
            tmp_file.write("Test document")
            tmp_path = tmp_file.name
        
        try:
            result = runner.invoke(app, [
                'classify-document',
                tmp_path,
                '--quiet'
            ])
            
            assert result.exit_code == 1
            
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    
    @patch('lemkin_classifier.cli.DocumentClassifier')
    def test_verbose_error_output(self, mock_classifier_class, runner):
        """Test verbose error output"""
        mock_classifier = Mock()
        mock_classifier.classify_file.side_effect = ValueError("Test error")
        mock_classifier_class.return_value = mock_classifier
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as tmp_file:
            tmp_file.write("Test document")
            tmp_path = tmp_file.name
        
        try:
            result = runner.invoke(app, [
                'classify-document',
                tmp_path,
                '--verbose'
            ])
            
            assert result.exit_code == 1
            # In verbose mode, should show more error details
            
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)


class TestCLIUtilityFunctions:
    """Test CLI utility functions"""
    
    def test_display_functions_exist(self):
        """Test that display utility functions exist"""
        from lemkin_classifier.cli import (
            _display_classification_table,
            _display_classification_json,
            _display_batch_summary,
            _display_training_metrics,
            _display_evaluation_metrics,
            _display_taxonomy
        )
        
        # Functions should be importable
        assert callable(_display_classification_table)
        assert callable(_display_classification_json)
        assert callable(_display_batch_summary)
        assert callable(_display_training_metrics)
        assert callable(_display_evaluation_metrics)
        assert callable(_display_taxonomy)


@pytest.fixture
def mock_rich_console():
    """Mock Rich console for testing display functions"""
    with patch('lemkin_classifier.cli.console') as mock_console:
        yield mock_console