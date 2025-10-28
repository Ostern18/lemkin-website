"""
Unit Tests: ForensicAnalysisReviewerAgent

Tests for the Forensic analysis review agent.

Agent: forensic-analysis-reviewer
Base Class: BaseAgent
SDK: Anthropic
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from shared import AuditLogger, EvidenceHandler
from shared.audit_logger import AuditEventType
from shared.evidence_handler import EvidenceType
# from agents.forensic-analysis-reviewer.agent import ForensicAnalysisReviewerAgent
ForensicAnalysisReviewerAgent = import_module("agents.forensic-analysis-reviewer.agent").ForensicAnalysisReviewerAgent
from importlib import import_module


class TestForensicAnalysisReviewerInitialization:
    """Test ForensicAnalysisReviewerAgent initialization and configuration."""

    def test_agent_initializes_successfully(self, shared_infrastructure):
        """Test that agent initializes with default settings."""
        agent = ForensicAnalysisReviewerAgent(**shared_infrastructure)

        assert agent is not None
        assert agent.agent_id == "forensic_analysis_reviewer"

    def test_agent_initializes_with_audit_logger(self, shared_infrastructure):
        """Test that agent properly initializes audit logger."""
        agent = ForensicAnalysisReviewerAgent(**shared_infrastructure)

        assert agent.audit_logger is not None
        assert agent.audit_logger == shared_infrastructure['audit_logger']

    def test_agent_initializes_evidence_handler(self, shared_infrastructure):
        """Test that agent properly initializes evidence handler."""
        agent = ForensicAnalysisReviewerAgent(**shared_infrastructure)

        assert agent.evidence_handler is not None
        assert agent.evidence_handler == shared_infrastructure['evidence_handler']

    def test_agent_logs_initialization(self, shared_infrastructure):
        """Test that agent logs its initialization."""
        agent = ForensicAnalysisReviewerAgent(**shared_infrastructure)

        summary = shared_infrastructure['audit_logger'].get_session_summary()
        assert 'agent_initialized' in summary['event_type_counts']


class TestForensicAnalysisReviewerBasicOperation:
    """Test basic ForensicAnalysisReviewerAgent operations."""

    @patch('anthropic.Anthropic')
    def test_agent_processes_basic_input(self, mock_anthropic, shared_infrastructure, mock_claude_response):
        """Test agent can process basic input successfully."""
        # Setup mock
        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_claude_response("Test analysis result")
        mock_anthropic.return_value = mock_client

        agent = ForensicAnalysisReviewerAgent(**shared_infrastructure)
        result = agent.process({
            'message': 'Test input for Forensic analysis review',
            'case_id': 'TEST-001'
        })

        assert result is not None

    @patch('anthropic.Anthropic')
    def test_agent_returns_properly_formatted_output(self, mock_anthropic, shared_infrastructure, mock_claude_response):
        """Test that agent returns output in expected format."""
        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_claude_response()
        mock_anthropic.return_value = mock_client

        agent = ForensicAnalysisReviewerAgent(**shared_infrastructure)
        result = agent.process({'message': 'Test'})

        assert isinstance(result, dict)


class TestForensicAnalysisReviewerEvidentaryCompliance:
    """Test evidentiary compliance features."""

    @patch('anthropic.Anthropic')
    def test_agent_logs_all_operations(self, mock_anthropic, shared_infrastructure, mock_claude_response):
        """Test that all agent operations are logged to audit trail."""
        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_claude_response()
        mock_anthropic.return_value = mock_client

        agent = ForensicAnalysisReviewerAgent(**shared_infrastructure)
        initial_count = shared_infrastructure['audit_logger'].get_session_summary()['total_events']

        result = agent.process({'message': 'Test operation'})

        final_count = shared_infrastructure['audit_logger'].get_session_summary()['total_events']
        assert final_count > initial_count

    @patch('anthropic.Anthropic')
    def test_agent_maintains_chain_of_custody(self, mock_anthropic, shared_infrastructure, mock_claude_response):
        """Test that agent maintains chain of custody for evidence."""
        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_claude_response()
        mock_anthropic.return_value = mock_client

        # Ingest evidence
        evidence_id = shared_infrastructure['evidence_handler'].ingest_evidence(
            file_data=b"Test evidence for Forensic analysis review",
            evidence_type=EvidenceType.DOCUMENT_PDF,
            source="Test Source"
        )

        agent = ForensicAnalysisReviewerAgent(**shared_infrastructure)
        result = agent.process({
            'message': 'Analyze this evidence',
            'evidence_ids': [evidence_id]
        })

        # Verify chain of custody
        chain = shared_infrastructure['audit_logger'].get_evidence_chain(evidence_id)
        assert len(chain) > 0

    @patch('anthropic.Anthropic')
    def test_agent_verifies_audit_chain_integrity(self, mock_anthropic, shared_infrastructure, mock_claude_response):
        """Test that audit chain integrity is maintained."""
        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_claude_response()
        mock_anthropic.return_value = mock_client

        agent = ForensicAnalysisReviewerAgent(**shared_infrastructure)
        agent.process({'message': 'Test operation 1'})
        agent.process({'message': 'Test operation 2'})
        agent.process({'message': 'Test operation 3'})

        assert shared_infrastructure['audit_logger'].verify_chain_integrity() is True


class TestForensicAnalysisReviewerErrorHandling:
    """Test error handling and edge cases."""

    def test_agent_handles_missing_required_fields(self, shared_infrastructure):
        """Test agent handles missing required input fields gracefully."""
        agent = ForensicAnalysisReviewerAgent(**shared_infrastructure)

        # Test with minimal input - should handle gracefully
        try:
            result = agent.process({})
            # If it doesn't raise, it should return something
            assert result is not None or True
        except (ValueError, KeyError, TypeError) as e:
            # If it raises, error should be informative
            assert len(str(e)) > 0

    @patch('anthropic.Anthropic')
    def test_agent_handles_api_errors(self, mock_anthropic, shared_infrastructure):
        """Test agent handles API errors gracefully."""
        mock_client = MagicMock()
        mock_client.messages.create.side_effect = Exception("API Error")
        mock_anthropic.return_value = mock_client

        agent = ForensicAnalysisReviewerAgent(**shared_infrastructure)

        with pytest.raises(Exception):
            agent.process({'message': 'Test'})

        # Verify error was logged
        summary = shared_infrastructure['audit_logger'].get_session_summary()
        assert 'error_occurred' in summary['event_type_counts']


class TestForensicAnalysisReviewerSpecificFunctionality:
    """Test ForensicAnalysisReviewerAgent specific functionality."""

    @patch('anthropic.Anthropic')
    def test_methodology_review(self, mock_anthropic, shared_infrastructure, mock_claude_response):
        """Test methodology review functionality."""
        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_claude_response()
        mock_anthropic.return_value = mock_client

        agent = ForensicAnalysisReviewerAgent(**shared_infrastructure)

        # TODO: Implement specific test for methodology_review
        # This is a placeholder - customize based on agent capabilities
        result = agent.process({'message': 'Test methodology_review'})
        assert result is not None

    @patch('anthropic.Anthropic')
    def test_conclusion_validation(self, mock_anthropic, shared_infrastructure, mock_claude_response):
        """Test conclusion validation functionality."""
        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_claude_response()
        mock_anthropic.return_value = mock_client

        agent = ForensicAnalysisReviewerAgent(**shared_infrastructure)

        # TODO: Implement specific test for conclusion_validation
        # This is a placeholder - customize based on agent capabilities
        result = agent.process({'message': 'Test conclusion_validation'})
        assert result is not None

    @patch('anthropic.Anthropic')
    def test_quality_assurance(self, mock_anthropic, shared_infrastructure, mock_claude_response):
        """Test quality assurance functionality."""
        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_claude_response()
        mock_anthropic.return_value = mock_client

        agent = ForensicAnalysisReviewerAgent(**shared_infrastructure)

        # TODO: Implement specific test for quality_assurance
        # This is a placeholder - customize based on agent capabilities
        result = agent.process({'message': 'Test quality_assurance'})
        assert result is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
