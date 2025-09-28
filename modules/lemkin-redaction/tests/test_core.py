import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import hashlib

from lemkin_redaction.core import (
    PIIRedactor,
    RedactionConfig,
    RedactionResult,
    PIIEntity,
    EntityType,
    RedactionType,
    ConfidenceLevel
)

@pytest.fixture
def sample_config():
    """Sample redaction configuration"""
    return RedactionConfig(
        entity_types=[EntityType.PERSON, EntityType.EMAIL, EntityType.PHONE],
        min_confidence=0.7,
        language="en",
        preserve_formatting=True,
        generate_report=True
    )

@pytest.fixture
def sample_entities():
    """Sample PII entities for testing"""
    return [
        PIIEntity(
            entity_type=EntityType.PERSON,
            text="John Smith",
            start_pos=0,
            end_pos=10,
            confidence=0.95,
            confidence_level=ConfidenceLevel.VERY_HIGH,
            replacement="[PERSON]"
        ),
        PIIEntity(
            entity_type=EntityType.EMAIL,
            text="john@example.com",
            start_pos=20,
            end_pos=36,
            confidence=0.99,
            confidence_level=ConfidenceLevel.VERY_HIGH,
            replacement="***@***.com"
        ),
        PIIEntity(
            entity_type=EntityType.PHONE,
            text="555-123-4567",
            start_pos=45,
            end_pos=57,
            confidence=0.88,
            confidence_level=ConfidenceLevel.HIGH,
            replacement="***-***-****"
        )
    ]

@pytest.fixture
def sample_redaction_result(sample_config, sample_entities):
    """Sample redaction result"""
    content = "John Smith email john@example.com phone 555-123-4567"
    content_hash = hashlib.sha256(content.encode()).hexdigest()
    
    return RedactionResult(
        original_content_hash=content_hash,
        content_type="text",
        entities_detected=sample_entities,
        entities_redacted=sample_entities,
        total_entities=len(sample_entities),
        redacted_count=len(sample_entities),
        confidence_scores={
            "PERSON": 0.95,
            "EMAIL": 0.99,
            "PHONE": 0.88
        },
        processing_time=0.25,
        config_used=sample_config
    )

@pytest.fixture
def temp_text_file():
    """Create temporary text file for testing"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write("John Smith contacted jane.doe@email.com at 555-0123 regarding the case.")
        file_path = Path(f.name)
    yield file_path
    file_path.unlink(missing_ok=True)

@pytest.fixture
def temp_image_file():
    """Create temporary image file for testing"""
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
        # Create a minimal JPEG file (just for testing file operations)
        f.write(b'\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x01\x00H\x00H\x00\x00\xff\xdb\x00C\x00')
        file_path = Path(f.name)
    yield file_path
    file_path.unlink(missing_ok=True)

class TestPIIRedactor:
    
    def test_initialization_default_config(self):
        """Test redactor initialization with default configuration"""
        redactor = PIIRedactor()
        
        assert redactor.config is not None
        assert isinstance(redactor.config, RedactionConfig)
        assert redactor.config.min_confidence == 0.7
        assert EntityType.PERSON in redactor.config.entity_types
    
    def test_initialization_custom_config(self, sample_config):
        """Test redactor initialization with custom configuration"""
        redactor = PIIRedactor(sample_config)
        
        assert redactor.config == sample_config
        assert redactor.config.min_confidence == 0.7
        assert redactor.config.language == "en"
    
    def test_lazy_loading_text_redactor(self, sample_config):
        """Test lazy loading of text redactor component"""
        redactor = PIIRedactor(sample_config)
        
        # Initially None
        assert redactor._text_redactor is None
        
        # Lazy loaded on first access
        with patch('lemkin_redaction.core.TextRedactor') as MockTextRedactor:
            mock_text_redactor = Mock()
            MockTextRedactor.return_value = mock_text_redactor
            
            text_redactor = redactor.text_redactor
            
            assert text_redactor is mock_text_redactor
            MockTextRedactor.assert_called_once_with(sample_config)
            
            # Subsequent calls return same instance
            assert redactor.text_redactor is text_redactor
    
    def test_lazy_loading_image_redactor(self, sample_config):
        """Test lazy loading of image redactor component"""
        redactor = PIIRedactor(sample_config)
        
        assert redactor._image_redactor is None
        
        with patch('lemkin_redaction.core.ImageRedactor') as MockImageRedactor:
            mock_image_redactor = Mock()
            MockImageRedactor.return_value = mock_image_redactor
            
            image_redactor = redactor.image_redactor
            
            assert image_redactor is mock_image_redactor
            MockImageRedactor.assert_called_once_with(sample_config)
    
    def test_lazy_loading_audio_redactor(self, sample_config):
        """Test lazy loading of audio redactor component"""
        redactor = PIIRedactor(sample_config)
        
        assert redactor._audio_redactor is None
        
        with patch('lemkin_redaction.core.AudioRedactor') as MockAudioRedactor:
            mock_audio_redactor = Mock()
            MockAudioRedactor.return_value = mock_audio_redactor
            
            audio_redactor = redactor.audio_redactor
            
            assert audio_redactor is mock_audio_redactor
            MockAudioRedactor.assert_called_once_with(sample_config)
    
    def test_lazy_loading_video_redactor(self, sample_config):
        """Test lazy loading of video redactor component"""
        redactor = PIIRedactor(sample_config)
        
        assert redactor._video_redactor is None
        
        with patch('lemkin_redaction.core.VideoRedactor') as MockVideoRedactor:
            mock_video_redactor = Mock()
            MockVideoRedactor.return_value = mock_video_redactor
            
            video_redactor = redactor.video_redactor
            
            assert video_redactor is mock_video_redactor
            MockVideoRedactor.assert_called_once_with(sample_config)
    
    def test_redact_text_success(self, sample_config, sample_redaction_result):
        """Test successful text redaction"""
        redactor = PIIRedactor(sample_config)
        test_text = "John Smith called 555-1234"
        
        with patch.object(redactor, 'text_redactor') as mock_text_redactor:
            mock_text_redactor.redact.return_value = sample_redaction_result
            
            result = redactor.redact_text(test_text)
            
            assert result == sample_redaction_result
            mock_text_redactor.redact.assert_called_once_with(test_text, None)
    
    def test_redact_text_with_output_path(self, sample_config, sample_redaction_result):
        """Test text redaction with output path"""
        redactor = PIIRedactor(sample_config)
        test_text = "John Smith called 555-1234"
        output_path = Path("/tmp/output.txt")
        
        with patch.object(redactor, 'text_redactor') as mock_text_redactor:
            mock_text_redactor.redact.return_value = sample_redaction_result
            
            result = redactor.redact_text(test_text, output_path)
            
            assert result == sample_redaction_result
            mock_text_redactor.redact.assert_called_once_with(test_text, output_path)
    
    def test_redact_text_exception_handling(self, sample_config):
        """Test text redaction exception handling"""
        redactor = PIIRedactor(sample_config)
        test_text = "John Smith called 555-1234"
        
        with patch.object(redactor, 'text_redactor') as mock_text_redactor:
            mock_text_redactor.redact.side_effect = RuntimeError("Processing failed")
            
            with pytest.raises(RuntimeError, match="Processing failed"):
                redactor.redact_text(test_text)
    
    def test_redact_image_success(self, sample_config, sample_redaction_result, temp_image_file):
        """Test successful image redaction"""
        redactor = PIIRedactor(sample_config)
        
        with patch.object(redactor, 'image_redactor') as mock_image_redactor:
            mock_image_redactor.redact.return_value = sample_redaction_result
            
            result = redactor.redact_image(temp_image_file)
            
            assert result == sample_redaction_result
            mock_image_redactor.redact.assert_called_once_with(temp_image_file, None)
    
    def test_redact_audio_success(self, sample_config, sample_redaction_result):
        """Test successful audio redaction"""
        redactor = PIIRedactor(sample_config)
        audio_path = Path("/tmp/audio.wav")
        
        with patch.object(redactor, 'audio_redactor') as mock_audio_redactor:
            mock_audio_redactor.redact.return_value = sample_redaction_result
            
            result = redactor.redact_audio(audio_path)
            
            assert result == sample_redaction_result
            mock_audio_redactor.redact.assert_called_once_with(audio_path, None)
    
    def test_redact_video_success(self, sample_config, sample_redaction_result):
        """Test successful video redaction"""
        redactor = PIIRedactor(sample_config)
        video_path = Path("/tmp/video.mp4")
        
        with patch.object(redactor, 'video_redactor') as mock_video_redactor:
            mock_video_redactor.redact.return_value = sample_redaction_result
            
            result = redactor.redact_video(video_path)
            
            assert result == sample_redaction_result
            mock_video_redactor.redact.assert_called_once_with(video_path, None)
    
    def test_redact_file_text_type(self, sample_config, sample_redaction_result, temp_text_file):
        """Test redact_file with text file type"""
        redactor = PIIRedactor(sample_config)
        
        with patch.object(redactor, 'redact_text') as mock_redact_text:
            mock_redact_text.return_value = sample_redaction_result
            
            result = redactor.redact_file(temp_text_file)
            
            assert result == sample_redaction_result
            # Should read file content and pass to redact_text
            mock_redact_text.assert_called_once()
            args, kwargs = mock_redact_text.call_args
            assert "John Smith contacted jane.doe@email.com" in args[0]
    
    def test_redact_file_image_type(self, sample_config, sample_redaction_result, temp_image_file):
        """Test redact_file with image file type"""
        redactor = PIIRedactor(sample_config)
        
        with patch.object(redactor, 'redact_image') as mock_redact_image:
            mock_redact_image.return_value = sample_redaction_result
            
            result = redactor.redact_file(temp_image_file)
            
            assert result == sample_redaction_result
            mock_redact_image.assert_called_once_with(temp_image_file, None)
    
    def test_redact_file_unsupported_type(self, sample_config):
        """Test redact_file with unsupported file type"""
        redactor = PIIRedactor(sample_config)
        
        with tempfile.NamedTemporaryFile(suffix='.xyz', delete=False) as f:
            file_path = Path(f.name)
        
        try:
            with pytest.raises(ValueError, match="Unsupported file type: .xyz"):
                redactor.redact_file(file_path)
        finally:
            file_path.unlink(missing_ok=True)
    
    def test_redact_file_not_found(self, sample_config):
        """Test redact_file with non-existent file"""
        redactor = PIIRedactor(sample_config)
        non_existent_file = Path("/non/existent/file.txt")
        
        with pytest.raises(FileNotFoundError):
            redactor.redact_file(non_existent_file)
    
    def test_batch_redact_success(self, sample_config, sample_redaction_result):
        """Test successful batch redaction"""
        redactor = PIIRedactor(sample_config)
        
        # Create temporary files
        temp_files = []
        for i in range(3):
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
                f.write(f"Content {i}")
                temp_files.append(Path(f.name))
        
        try:
            with tempfile.TemporaryDirectory() as output_dir:
                output_path = Path(output_dir)
                
                with patch.object(redactor, 'redact_file') as mock_redact_file:
                    mock_redact_file.return_value = sample_redaction_result
                    
                    results = redactor.batch_redact(temp_files, output_path)
                    
                    assert len(results) == 3
                    assert all(r == sample_redaction_result for r in results)
                    assert mock_redact_file.call_count == 3
        
        finally:
            for temp_file in temp_files:
                temp_file.unlink(missing_ok=True)
    
    def test_batch_redact_with_errors(self, sample_config, sample_redaction_result):
        """Test batch redaction with some failures"""
        redactor = PIIRedactor(sample_config)
        
        # Create temporary files
        temp_files = []
        for i in range(2):
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
                f.write(f"Content {i}")
                temp_files.append(Path(f.name))
        
        try:
            with tempfile.TemporaryDirectory() as output_dir:
                output_path = Path(output_dir)
                
                with patch.object(redactor, 'redact_file') as mock_redact_file:
                    # First call succeeds, second fails
                    mock_redact_file.side_effect = [
                        sample_redaction_result,
                        RuntimeError("Processing failed")
                    ]
                    
                    results = redactor.batch_redact(temp_files, output_path)
                    
                    assert len(results) == 2
                    assert results[0] == sample_redaction_result
                    assert results[1].errors == ["Processing failed"]
                    assert results[1].redacted_count == 0
        
        finally:
            for temp_file in temp_files:
                temp_file.unlink(missing_ok=True)
    
    def test_update_config(self, sample_config):
        """Test configuration update"""
        redactor = PIIRedactor()
        
        # Initialize components
        _ = redactor.text_redactor
        _ = redactor.image_redactor
        
        # Verify components are loaded
        assert redactor._text_redactor is not None
        assert redactor._image_redactor is not None
        
        # Update configuration
        redactor.update_config(sample_config)
        
        assert redactor.config == sample_config
        # Components should be reset
        assert redactor._text_redactor is None
        assert redactor._image_redactor is None
    
    def test_get_supported_formats(self):
        """Test get_supported_formats method"""
        redactor = PIIRedactor()
        formats = redactor.get_supported_formats()
        
        assert isinstance(formats, dict)
        assert "text" in formats
        assert "image" in formats
        assert "audio" in formats
        assert "video" in formats
        
        assert ".txt" in formats["text"]
        assert ".jpg" in formats["image"]
        assert ".wav" in formats["audio"]
        assert ".mp4" in formats["video"]

class TestRedactionConfig:
    
    def test_default_config(self):
        """Test default configuration values"""
        config = RedactionConfig()
        
        assert config.min_confidence == 0.7
        assert config.language == "en"
        assert config.preserve_formatting is True
        assert config.generate_report is True
        assert config.track_changes is True
        assert EntityType.PERSON in config.entity_types
        assert EntityType.EMAIL in config.entity_types
    
    def test_custom_config(self):
        """Test custom configuration values"""
        config = RedactionConfig(
            entity_types=[EntityType.PERSON],
            min_confidence=0.9,
            language="es",
            preserve_formatting=False,
            generate_report=False
        )
        
        assert config.entity_types == [EntityType.PERSON]
        assert config.min_confidence == 0.9
        assert config.language == "es"
        assert config.preserve_formatting is False
        assert config.generate_report is False
    
    def test_confidence_validation(self):
        """Test confidence threshold validation"""
        # Valid confidence values
        config1 = RedactionConfig(min_confidence=0.0)
        assert config1.min_confidence == 0.0
        
        config2 = RedactionConfig(min_confidence=1.0)
        assert config2.min_confidence == 1.0
        
        # Invalid confidence values should raise validation error
        with pytest.raises(ValueError):
            RedactionConfig(min_confidence=-0.1)
        
        with pytest.raises(ValueError):
            RedactionConfig(min_confidence=1.1)

class TestPIIEntity:
    
    def test_pii_entity_creation(self):
        """Test PIIEntity creation"""
        entity = PIIEntity(
            entity_type=EntityType.PERSON,
            text="John Smith",
            start_pos=0,
            end_pos=10,
            confidence=0.95,
            confidence_level=ConfidenceLevel.VERY_HIGH,
            replacement="[PERSON]"
        )
        
        assert entity.entity_type == EntityType.PERSON
        assert entity.text == "John Smith"
        assert entity.start_pos == 0
        assert entity.end_pos == 10
        assert entity.confidence == 0.95
        assert entity.confidence_level == ConfidenceLevel.VERY_HIGH
        assert entity.replacement == "[PERSON]"

class TestRedactionResult:
    
    def test_redaction_result_creation(self, sample_config, sample_entities):
        """Test RedactionResult creation"""
        result = RedactionResult(
            original_content_hash="abc123",
            content_type="text",
            entities_detected=sample_entities,
            entities_redacted=sample_entities[:2],  # Only first 2 redacted
            total_entities=len(sample_entities),
            redacted_count=2,
            confidence_scores={"PERSON": 0.95, "EMAIL": 0.99},
            processing_time=0.5,
            config_used=sample_config
        )
        
        assert result.original_content_hash == "abc123"
        assert result.content_type == "text"
        assert len(result.entities_detected) == 3
        assert len(result.entities_redacted) == 2
        assert result.total_entities == 3
        assert result.redacted_count == 2
        assert result.processing_time == 0.5
        assert result.operation_id is not None  # Auto-generated UUID
    
    def test_redaction_result_defaults(self, sample_config):
        """Test RedactionResult with default values"""
        result = RedactionResult(
            original_content_hash="abc123",
            content_type="text",
            entities_detected=[],
            entities_redacted=[],
            total_entities=0,
            redacted_count=0,
            confidence_scores={},
            processing_time=0.0,
            config_used=sample_config
        )
        
        assert result.warnings == []
        assert result.errors == []
        assert result.redacted_content_path is None
        assert result.report_path is None
        assert result.redaction_quality == {}
        assert result.timestamp is not None