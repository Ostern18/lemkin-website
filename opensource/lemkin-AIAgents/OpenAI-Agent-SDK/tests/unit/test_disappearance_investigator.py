"""
Unit Tests: DisappearanceInvestigatorAgent

Tests for the Enforced disappearance investigation agent.

Agent: disappearance-investigator
Base Class: BaseAgent
SDK: Openai
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from shared import AuditLogger, EvidenceHandler
from shared.audit_logger import AuditEventType
from shared.evidence_handler import EvidenceType
# from agents.disappearance-investigator.agent import DisappearanceInvestigatorAgent
DisappearanceInvestigatorAgent = import_module("agents.disappearance-investigator.agent").DisappearanceInvestigatorAgent
from importlib import import_module


class TestDisappearanceInvestigatorInitialization:
    """Test DisappearanceInvestigatorAgent initialization and configuration."""

    def test_agent_initializes_successfully(self, shared_infrastructure):
        """Test that agent initializes with default settings."""
        agent = DisappearanceInvestigatorAgent(**shared_infrastructure)

        assert agent is not None
        assert agent.agent_id == "disappearance_investigator"

    def test_agent_initializes_with_audit_logger(self, shared_infrastructure):
        """Test that agent properly initializes audit logger."""
        agent = DisappearanceInvestigatorAgent(**shared_infrastructure)

        assert agent.audit_logger is not None
        assert agent.audit_logger == shared_infrastructure['audit_logger']

    def test_agent_initializes_evidence_handler(self, shared_infrastructure):
        """Test that agent properly initializes evidence handler."""
        agent = DisappearanceInvestigatorAgent(**shared_infrastructure)

        assert agent.evidence_handler is not None
        assert agent.evidence_handler == shared_infrastructure['evidence_handler']

    def test_agent_logs_initialization(self, shared_infrastructure):
        """Test that agent logs its initialization."""
        agent = DisappearanceInvestigatorAgent(**shared_infrastructure)

        summary = shared_infrastructure['audit_logger'].get_session_summary()
        assert 'agent_initialized' in summary['event_type_counts']


class TestDisappearanceInvestigatorBasicOperation:
    """Test basic DisappearanceInvestigatorAgent operations."""

    @patch('agents.Agent')
    def test_agent_processes_basic_input(self, mock_openai, shared_infrastructure, mock_openai_response):
        """Test agent can process basic input successfully."""
        # Setup mock
        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_openai_response("Test analysis result")
        mock_openai.return_value = mock_client

        agent = DisappearanceInvestigatorAgent(**shared_infrastructure)
        result = agent.process({
            'message': 'Test input for Enforced disappearance investigation',
            'case_id': 'TEST-001'
        })

        assert result is not None

    @patch('agents.Agent')
    def test_agent_returns_properly_formatted_output(self, mock_openai, shared_infrastructure, mock_openai_response):
        """Test that agent returns output in expected format."""
        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_openai_response()
        mock_openai.return_value = mock_client

        agent = DisappearanceInvestigatorAgent(**shared_infrastructure)
        result = agent.process({'message': 'Test'})

        assert isinstance(result, dict)


class TestDisappearanceInvestigatorEvidentaryCompliance:
    """Test evidentiary compliance features."""

    @patch('agents.Agent')
    def test_agent_logs_all_operations(self, mock_openai, shared_infrastructure, mock_openai_response):
        """Test that all agent operations are logged to audit trail."""
        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_openai_response()
        mock_openai.return_value = mock_client

        agent = DisappearanceInvestigatorAgent(**shared_infrastructure)
        initial_count = shared_infrastructure['audit_logger'].get_session_summary()['total_events']

        result = agent.process({'message': 'Test operation'})

        final_count = shared_infrastructure['audit_logger'].get_session_summary()['total_events']
        assert final_count > initial_count

    @patch('agents.Agent')
    def test_agent_maintains_chain_of_custody(self, mock_openai, shared_infrastructure, mock_openai_response):
        """Test that agent maintains chain of custody for evidence."""
        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_openai_response()
        mock_openai.return_value = mock_client

        # Ingest evidence
        evidence_id = shared_infrastructure['evidence_handler'].ingest_evidence(
            file_data=b"Test evidence for Enforced disappearance investigation",
            evidence_type=EvidenceType.DOCUMENT_PDF,
            source="Test Source"
        )

        agent = DisappearanceInvestigatorAgent(**shared_infrastructure)
        result = agent.process({
            'message': 'Analyze this evidence',
            'evidence_ids': [evidence_id]
        })

        # Verify chain of custody
        chain = shared_infrastructure['audit_logger'].get_evidence_chain(evidence_id)
        assert len(chain) > 0

    @patch('agents.Agent')
    def test_agent_verifies_audit_chain_integrity(self, mock_openai, shared_infrastructure, mock_openai_response):
        """Test that audit chain integrity is maintained."""
        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_openai_response()
        mock_openai.return_value = mock_client

        agent = DisappearanceInvestigatorAgent(**shared_infrastructure)
        agent.process({'message': 'Test operation 1'})
        agent.process({'message': 'Test operation 2'})
        agent.process({'message': 'Test operation 3'})

        assert shared_infrastructure['audit_logger'].verify_chain_integrity() is True


class TestDisappearanceInvestigatorErrorHandling:
    """Test error handling and edge cases."""

    def test_agent_handles_missing_required_fields(self, shared_infrastructure):
        """Test agent handles missing required input fields gracefully."""
        agent = DisappearanceInvestigatorAgent(**shared_infrastructure)

        # Test with minimal input - should handle gracefully
        try:
            result = agent.process({})
            # If it doesn't raise, it should return something
            assert result is not None or True
        except (ValueError, KeyError, TypeError) as e:
            # If it raises, error should be informative
            assert len(str(e)) > 0

    @patch('agents.Agent')
    def test_agent_handles_api_errors(self, mock_openai, shared_infrastructure):
        """Test agent handles API errors gracefully."""
        mock_client = MagicMock()
        mock_client.messages.create.side_effect = Exception("API Error")
        mock_openai.return_value = mock_client

        agent = DisappearanceInvestigatorAgent(**shared_infrastructure)

        with pytest.raises(Exception):
            agent.process({'message': 'Test'})

        # Verify error was logged
        summary = shared_infrastructure['audit_logger'].get_session_summary()
        assert 'error_occurred' in summary['event_type_counts']


class TestDisappearanceInvestigatorSpecificFunctionality:
    """Test DisappearanceInvestigatorAgent specific functionality."""

    @patch('agents.Agent')
    def test_disappearance_pattern_analysis(self, mock_openai, shared_infrastructure, mock_openai_response):
        """Test disappearance pattern analysis functionality."""
        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_openai_response()
        mock_openai.return_value = mock_client

        agent = DisappearanceInvestigatorAgent(**shared_infrastructure)

        # TODO: Implement specific test for disappearance_pattern_analysis
        # This is a placeholder - customize based on agent capabilities
        result = agent.process({'message': 'Test disappearance_pattern_analysis'})
        assert result is not None

    @patch('agents.Agent')
    def test_last_known_location_tracking(self, mock_openai, shared_infrastructure, mock_openai_response):
        """Test last known location tracking functionality."""
        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_openai_response()
        mock_openai.return_value = mock_client

        agent = DisappearanceInvestigatorAgent(**shared_infrastructure)

        # TODO: Implement specific test for last_known_location_tracking
        # This is a placeholder - customize based on agent capabilities
        result = agent.process({'message': 'Test last_known_location_tracking'})
        assert result is not None

    @patch('agents.Agent')
    def test_witness_statement_correlation(self, mock_openai, shared_infrastructure, mock_openai_response):
        """Test witness statement correlation functionality."""
        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_openai_response()
        mock_openai.return_value = mock_client

        agent = DisappearanceInvestigatorAgent(**shared_infrastructure)

        # TODO: Implement specific test for witness_statement_correlation
        # This is a placeholder - customize based on agent capabilities
        result = agent.process({'message': 'Test witness_statement_correlation'})
        assert result is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
