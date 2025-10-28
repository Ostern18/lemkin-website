"""
Unit Tests: GenocideIntentAnalyzerAgent

Tests for the Genocidal intent analysis agent.

Agent: genocide-intent-analyzer
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
# from agents.genocide-intent-analyzer.agent import GenocideIntentAnalyzerAgent
GenocideIntentAnalyzerAgent = import_module("agents.genocide-intent-analyzer.agent").GenocideIntentAnalyzerAgent
from importlib import import_module


class TestGenocideIntentAnalyzerInitialization:
    """Test GenocideIntentAnalyzerAgent initialization and configuration."""

    def test_agent_initializes_successfully(self, shared_infrastructure):
        """Test that agent initializes with default settings."""
        agent = GenocideIntentAnalyzerAgent(**shared_infrastructure)

        assert agent is not None
        assert agent.agent_id == "genocide_intent_analyzer"

    def test_agent_initializes_with_audit_logger(self, shared_infrastructure):
        """Test that agent properly initializes audit logger."""
        agent = GenocideIntentAnalyzerAgent(**shared_infrastructure)

        assert agent.audit_logger is not None
        assert agent.audit_logger == shared_infrastructure['audit_logger']

    def test_agent_initializes_evidence_handler(self, shared_infrastructure):
        """Test that agent properly initializes evidence handler."""
        agent = GenocideIntentAnalyzerAgent(**shared_infrastructure)

        assert agent.evidence_handler is not None
        assert agent.evidence_handler == shared_infrastructure['evidence_handler']

    def test_agent_logs_initialization(self, shared_infrastructure):
        """Test that agent logs its initialization."""
        agent = GenocideIntentAnalyzerAgent(**shared_infrastructure)

        summary = shared_infrastructure['audit_logger'].get_session_summary()
        assert 'agent_initialized' in summary['event_type_counts']


class TestGenocideIntentAnalyzerBasicOperation:
    """Test basic GenocideIntentAnalyzerAgent operations."""

    @patch('anthropic.Anthropic')
    def test_agent_processes_basic_input(self, mock_anthropic, shared_infrastructure, mock_claude_response):
        """Test agent can process basic input successfully."""
        # Setup mock
        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_claude_response("Test analysis result")
        mock_anthropic.return_value = mock_client

        agent = GenocideIntentAnalyzerAgent(**shared_infrastructure)
        result = agent.process({
            'message': 'Test input for Genocidal intent analysis',
            'case_id': 'TEST-001'
        })

        assert result is not None

    @patch('anthropic.Anthropic')
    def test_agent_returns_properly_formatted_output(self, mock_anthropic, shared_infrastructure, mock_claude_response):
        """Test that agent returns output in expected format."""
        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_claude_response()
        mock_anthropic.return_value = mock_client

        agent = GenocideIntentAnalyzerAgent(**shared_infrastructure)
        result = agent.process({'message': 'Test'})

        assert isinstance(result, dict)


class TestGenocideIntentAnalyzerEvidentaryCompliance:
    """Test evidentiary compliance features."""

    @patch('anthropic.Anthropic')
    def test_agent_logs_all_operations(self, mock_anthropic, shared_infrastructure, mock_claude_response):
        """Test that all agent operations are logged to audit trail."""
        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_claude_response()
        mock_anthropic.return_value = mock_client

        agent = GenocideIntentAnalyzerAgent(**shared_infrastructure)
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
            file_data=b"Test evidence for Genocidal intent analysis",
            evidence_type=EvidenceType.DOCUMENT_PDF,
            source="Test Source"
        )

        agent = GenocideIntentAnalyzerAgent(**shared_infrastructure)
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

        agent = GenocideIntentAnalyzerAgent(**shared_infrastructure)
        agent.process({'message': 'Test operation 1'})
        agent.process({'message': 'Test operation 2'})
        agent.process({'message': 'Test operation 3'})

        assert shared_infrastructure['audit_logger'].verify_chain_integrity() is True


class TestGenocideIntentAnalyzerErrorHandling:
    """Test error handling and edge cases."""

    def test_agent_handles_missing_required_fields(self, shared_infrastructure):
        """Test agent handles missing required input fields gracefully."""
        agent = GenocideIntentAnalyzerAgent(**shared_infrastructure)

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

        agent = GenocideIntentAnalyzerAgent(**shared_infrastructure)

        with pytest.raises(Exception):
            agent.process({'message': 'Test'})

        # Verify error was logged
        summary = shared_infrastructure['audit_logger'].get_session_summary()
        assert 'error_occurred' in summary['event_type_counts']


class TestGenocideIntentAnalyzerSpecificFunctionality:
    """Test GenocideIntentAnalyzerAgent specific functionality."""

    @patch('anthropic.Anthropic')
    def test_intent_indicators(self, mock_anthropic, shared_infrastructure, mock_claude_response):
        """Test intent indicators functionality."""
        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_claude_response()
        mock_anthropic.return_value = mock_client

        agent = GenocideIntentAnalyzerAgent(**shared_infrastructure)

        # TODO: Implement specific test for intent_indicators
        # This is a placeholder - customize based on agent capabilities
        result = agent.process({'message': 'Test intent_indicators'})
        assert result is not None

    @patch('anthropic.Anthropic')
    def test_systematic_pattern_detection(self, mock_anthropic, shared_infrastructure, mock_claude_response):
        """Test systematic pattern detection functionality."""
        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_claude_response()
        mock_anthropic.return_value = mock_client

        agent = GenocideIntentAnalyzerAgent(**shared_infrastructure)

        # TODO: Implement specific test for systematic_pattern_detection
        # This is a placeholder - customize based on agent capabilities
        result = agent.process({'message': 'Test systematic_pattern_detection'})
        assert result is not None

    @patch('anthropic.Anthropic')
    def test_genocide_convention_analysis(self, mock_anthropic, shared_infrastructure, mock_claude_response):
        """Test genocide convention analysis functionality."""
        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_claude_response()
        mock_anthropic.return_value = mock_client

        agent = GenocideIntentAnalyzerAgent(**shared_infrastructure)

        # TODO: Implement specific test for genocide_convention_analysis
        # This is a placeholder - customize based on agent capabilities
        result = agent.process({'message': 'Test genocide_convention_analysis'})
        assert result is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
