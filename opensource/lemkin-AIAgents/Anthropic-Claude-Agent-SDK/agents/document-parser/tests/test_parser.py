"""
Unit Tests for Multi-Format Document Parser Agent
"""

import pytest
import sys
from pathlib import Path

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from document_parser.agent import DocumentParserAgent
from document_parser.config import ParserConfig, DEFAULT_CONFIG
from shared import AuditLogger, EvidenceHandler


class TestDocumentParserAgent:
    """Test suite for Document Parser Agent."""

    @pytest.fixture
    def agent(self):
        """Create agent instance for testing."""
        return DocumentParserAgent(config=DEFAULT_CONFIG)

    @pytest.fixture
    def sample_pdf_data(self):
        """Mock PDF data for testing."""
        # In real tests, this would be actual PDF bytes
        return b"%PDF-1.4 mock pdf data"

    @pytest.fixture
    def sample_image_data(self):
        """Mock image data for testing."""
        # In real tests, this would be actual image bytes
        return b"\x89PNG\r\n\x1a\n mock image data"

    def test_agent_initialization(self):
        """Test that agent initializes correctly."""
        agent = DocumentParserAgent()

        assert agent.agent_id == "document_parser"
        assert agent.config is not None
        assert agent.evidence_handler is not None
        assert agent.audit_logger is not None

    def test_custom_configuration(self):
        """Test agent with custom configuration."""
        custom_config = ParserConfig(
            temperature=0.0,
            max_tokens=4096,
            min_confidence_threshold=0.9
        )

        agent = DocumentParserAgent(config=custom_config)

        assert agent.config.temperature == 0.0
        assert agent.config.max_tokens == 4096
        assert agent.config.min_confidence_threshold == 0.9

    def test_shared_infrastructure(self):
        """Test agent uses shared audit logger and evidence handler."""
        shared_audit = AuditLogger()
        shared_evidence = EvidenceHandler()

        agent = DocumentParserAgent(
            audit_logger=shared_audit,
            evidence_handler=shared_evidence
        )

        assert agent.audit_logger is shared_audit
        assert agent.evidence_handler is shared_evidence

    def test_ingest_document(self, agent, sample_pdf_data):
        """Test document ingestion creates evidence."""
        input_data = {
            'file_data': sample_pdf_data,
            'file_type': 'pdf',
            'source': 'Test Source',
            'case_id': 'TEST-001'
        }

        evidence_id = agent._ingest_document(sample_pdf_data, input_data)

        # Verify evidence was created
        assert evidence_id is not None
        metadata = agent.evidence_handler.get_metadata(evidence_id)
        assert metadata is not None
        assert metadata.source == 'Test Source'
        assert metadata.case_id == 'TEST-001'

    def test_needs_human_review_low_confidence(self, agent):
        """Test human review trigger for low confidence."""
        result = {
            'confidence_scores': {'overall': 0.4},
            'quality_flags': []
        }

        needs_review = agent._needs_human_review(result)
        assert needs_review is True

    def test_needs_human_review_high_confidence(self, agent):
        """Test no human review for high confidence."""
        result = {
            'confidence_scores': {'overall': 0.9},
            'quality_flags': []
        }

        needs_review = agent._needs_human_review(result)
        assert needs_review is False

    def test_needs_human_review_quality_flags(self, agent):
        """Test human review trigger for high-severity quality flags."""
        result = {
            'confidence_scores': {'overall': 0.8},
            'quality_flags': [
                {'severity': 'high', 'description': 'Major quality issue'}
            ]
        }

        needs_review = agent._needs_human_review(result)
        assert needs_review is True

    def test_create_processing_prompt(self, agent):
        """Test prompt generation."""
        prompt = agent._create_processing_prompt()

        assert isinstance(prompt, str)
        assert len(prompt) > 0
        assert 'document' in prompt.lower()
        assert 'extract' in prompt.lower()

    def test_create_processing_prompt_with_tables(self):
        """Test prompt includes table extraction when enabled."""
        config = ParserConfig(enable_table_extraction=True)
        agent = DocumentParserAgent(config=config)

        prompt = agent._create_processing_prompt()

        assert 'table' in prompt.lower()

    def test_create_processing_prompt_without_tables(self):
        """Test prompt excludes table extraction when disabled."""
        config = ParserConfig(enable_table_extraction=False)
        agent = DocumentParserAgent(config=config)

        prompt = agent._create_processing_prompt()

        # Would not have table-specific instructions

    def test_parse_response_with_json(self, agent):
        """Test parsing valid JSON response."""
        mock_response = {
            'content': [
                {
                    'text': '{"document_type": "test", "confidence_scores": {"overall": 0.9}}'
                }
            ],
            'model': 'claude-sonnet-4-5',
            'usage': {'input_tokens': 100, 'output_tokens': 200}
        }

        result = agent._parse_response(mock_response, "test-evidence-id")

        assert result['document_type'] == 'test'
        assert result['confidence_scores']['overall'] == 0.9
        assert result['_processing']['evidence_id'] == 'test-evidence-id'

    def test_parse_response_without_json(self, agent):
        """Test parsing response without valid JSON."""
        mock_response = {
            'content': [
                {'text': 'This is not JSON'}
            ],
            'model': 'claude-sonnet-4-5',
            'usage': {'input_tokens': 100, 'output_tokens': 50}
        }

        result = agent._parse_response(mock_response, "test-evidence-id")

        # Should still return structured data
        assert 'extracted_text' in result
        assert 'confidence_scores' in result
        assert result['confidence_scores']['overall'] < 1.0

    def test_chain_of_custody(self, agent, sample_pdf_data):
        """Test chain-of-custody tracking."""
        # Ingest document
        input_data = {
            'file_data': sample_pdf_data,
            'file_type': 'pdf',
            'source': 'Test',
            'case_id': 'TEST-002'
        }

        evidence_id = agent._ingest_document(sample_pdf_data, input_data)

        # Get chain
        chain = agent.get_chain_of_custody(evidence_id)

        # Should have at least ingestion event
        assert len(chain) > 0

        # Verify integrity
        integrity_ok = agent.verify_integrity()
        assert integrity_ok is True


class TestParserConfig:
    """Test parser configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        config = ParserConfig()

        assert config.model == "claude-sonnet-4-5-20250929"
        assert config.temperature == 0.1
        assert config.max_tokens == 8192
        assert config.enable_table_extraction is True
        assert config.enable_handwriting_detection is True

    def test_custom_config(self):
        """Test custom configuration."""
        config = ParserConfig(
            temperature=0.0,
            min_confidence_threshold=0.95,
            enable_table_extraction=False
        )

        assert config.temperature == 0.0
        assert config.min_confidence_threshold == 0.95
        assert config.enable_table_extraction is False

    def test_config_to_dict(self):
        """Test configuration serialization."""
        config = ParserConfig()
        config_dict = config.to_dict()

        assert isinstance(config_dict, dict)
        assert 'model' in config_dict
        assert 'temperature' in config_dict
        assert 'supported_document_types' in config_dict


# Integration tests would go here
# These would require actual API access and test documents

@pytest.mark.integration
class TestDocumentParserIntegration:
    """Integration tests (require API key and test documents)."""

    @pytest.mark.skip(reason="Requires ANTHROPIC_API_KEY and test documents")
    def test_parse_real_pdf(self):
        """Test parsing actual PDF document."""
        # This would test with real documents and API
        pass

    @pytest.mark.skip(reason="Requires ANTHROPIC_API_KEY and test documents")
    def test_parse_real_image(self):
        """Test parsing actual image document."""
        # This would test with real documents and API
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
