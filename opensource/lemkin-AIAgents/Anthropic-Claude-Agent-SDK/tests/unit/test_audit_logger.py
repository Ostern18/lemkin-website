"""
Unit Tests: AuditLogger

Tests the blockchain-style audit logging system for evidentiary compliance.
"""

import pytest
import json
import hashlib
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from shared import AuditLogger
from shared.audit_logger import AuditEventType


class TestAuditLoggerInitialization:
    """Test AuditLogger initialization and setup."""

    def test_creates_log_directory(self, temp_directory):
        """Test that logger creates log directory if it doesn't exist."""
        log_dir = temp_directory / "custom_logs"
        logger = AuditLogger(log_directory=log_dir)

        assert log_dir.exists()
        assert logger.log_directory == log_dir

    def test_generates_unique_session_id(self, temp_directory):
        """Test that each logger instance gets a unique session ID."""
        logger1 = AuditLogger(log_directory=temp_directory / "logs1")
        logger2 = AuditLogger(log_directory=temp_directory / "logs2")

        assert logger1.session_id != logger2.session_id

    def test_accepts_custom_session_id(self, temp_directory):
        """Test that custom session ID is respected."""
        custom_id = "CUSTOM-SESSION-123"
        logger = AuditLogger(
            log_directory=temp_directory,
            session_id=custom_id
        )

        assert logger.session_id == custom_id

    def test_creates_log_file(self, audit_logger):
        """Test that log file is created on initialization."""
        assert audit_logger.log_file.exists()

    def test_logs_session_start(self, audit_logger):
        """Test that session start is automatically logged."""
        summary = audit_logger.get_session_summary()

        assert summary['total_events'] >= 1
        assert 'agent_initialized' in summary['event_type_counts']


class TestAuditLoggerEventLogging:
    """Test event logging functionality."""

    def test_log_single_event(self, audit_logger):
        """Test logging a single event."""
        event_id = audit_logger.log_event(
            event_type=AuditEventType.EVIDENCE_INGESTED,
            agent_id="test_agent",
            evidence_ids=["evidence_123"],
            details={"source": "test"}
        )

        assert event_id is not None
        assert len(event_id) > 0

    def test_event_has_required_fields(self, audit_logger, temp_directory):
        """Test that logged events contain all required fields."""
        audit_logger.log_event(
            event_type=AuditEventType.ANALYSIS_PERFORMED,
            agent_id="analyzer",
            evidence_ids=["ev1", "ev2"],
            details={"analysis_type": "medical"}
        )

        # Read the log file
        with open(audit_logger.log_file, 'r') as f:
            lines = f.readlines()
            last_event = json.loads(lines[-1])

        required_fields = [
            'event_id', 'timestamp', 'event_type', 'agent_id',
            'evidence_ids', 'details', 'metadata', 'previous_event_hash',
            'event_hash'
        ]

        for field in required_fields:
            assert field in last_event

    def test_event_types_are_logged_correctly(self, audit_logger):
        """Test that different event types are logged correctly."""
        event_types = [
            AuditEventType.EVIDENCE_INGESTED,
            AuditEventType.EVIDENCE_PROCESSED,
            AuditEventType.ANALYSIS_PERFORMED,
            AuditEventType.OUTPUT_GENERATED,
            AuditEventType.HUMAN_REVIEW_REQUESTED
        ]

        for event_type in event_types:
            audit_logger.log_event(
                event_type=event_type,
                agent_id="test_agent",
                details={"test": event_type.value}
            )

        summary = audit_logger.get_session_summary()

        for event_type in event_types:
            assert event_type.value in summary['event_type_counts']

    def test_multiple_evidence_ids(self, audit_logger):
        """Test logging events with multiple evidence IDs."""
        evidence_ids = ["ev1", "ev2", "ev3", "ev4"]

        audit_logger.log_event(
            event_type=AuditEventType.ANALYSIS_PERFORMED,
            agent_id="test_agent",
            evidence_ids=evidence_ids
        )

        # Verify all evidence IDs are in the chain
        for evidence_id in evidence_ids:
            chain = audit_logger.get_evidence_chain(evidence_id)
            assert len(chain) > 0

    def test_optional_metadata(self, audit_logger):
        """Test that optional metadata is stored correctly."""
        metadata = {
            'user_id': 'investigator_123',
            'ip_address': '192.168.1.1',
            'permission_level': 'admin'
        }

        audit_logger.log_event(
            event_type=AuditEventType.EVIDENCE_PROCESSED,
            agent_id="test_agent",
            metadata=metadata
        )

        # Read and verify metadata
        with open(audit_logger.log_file, 'r') as f:
            lines = f.readlines()
            last_event = json.loads(lines[-1])

        assert last_event['metadata'] == metadata


class TestAuditLoggerChainIntegrity:
    """Test blockchain-style chain integrity."""

    def test_chain_links_events_correctly(self, audit_logger):
        """Test that events are properly linked in a chain."""
        # Log multiple events
        for i in range(5):
            audit_logger.log_event(
                event_type=AuditEventType.ANALYSIS_PERFORMED,
                agent_id=f"agent_{i}",
                details={"step": i}
            )

        # Read all events
        events = []
        with open(audit_logger.log_file, 'r') as f:
            for line in f:
                events.append(json.loads(line))

        # Verify chain linkage
        for i in range(1, len(events)):
            assert events[i]['previous_event_hash'] == events[i-1]['event_hash']

    def test_verify_chain_integrity_passes(self, audit_logger):
        """Test that chain integrity verification passes for valid chain."""
        # Log several events
        for i in range(10):
            audit_logger.log_event(
                event_type=AuditEventType.EVIDENCE_PROCESSED,
                agent_id="test_agent",
                evidence_ids=[f"evidence_{i}"]
            )

        assert audit_logger.verify_chain_integrity() is True

    def test_verify_chain_integrity_fails_on_tamper(self, audit_logger):
        """Test that chain integrity verification fails if log is tampered."""
        # Log events
        for i in range(5):
            audit_logger.log_event(
                event_type=AuditEventType.ANALYSIS_PERFORMED,
                agent_id="test_agent"
            )

        # Tamper with the log file
        with open(audit_logger.log_file, 'r') as f:
            lines = f.readlines()

        # Modify the middle event
        middle_event = json.loads(lines[len(lines) // 2])
        middle_event['details']['tampered'] = True

        with open(audit_logger.log_file, 'w') as f:
            for i, line in enumerate(lines):
                if i == len(lines) // 2:
                    f.write(json.dumps(middle_event) + '\n')
                else:
                    f.write(line)

        # Verification should fail
        assert audit_logger.verify_chain_integrity() is False

    def test_hash_calculation_is_deterministic(self, audit_logger):
        """Test that hash calculation is deterministic."""
        event = {
            'event_id': 'test_123',
            'timestamp': '2024-01-15T10:00:00Z',
            'event_type': 'test',
            'agent_id': 'test_agent',
            'evidence_ids': ['ev1'],
            'details': {'key': 'value'},
            'metadata': {},
            'previous_event_hash': None
        }

        hash1 = audit_logger._hash_event(event)
        hash2 = audit_logger._hash_event(event)

        assert hash1 == hash2

    def test_different_events_have_different_hashes(self, audit_logger):
        """Test that different events produce different hashes."""
        event1 = {
            'event_id': '1',
            'event_type': 'type1',
            'agent_id': 'agent1',
            'evidence_ids': [],
            'details': {},
            'metadata': {},
            'previous_event_hash': None
        }

        event2 = {
            'event_id': '2',
            'event_type': 'type2',
            'agent_id': 'agent1',
            'evidence_ids': [],
            'details': {},
            'metadata': {},
            'previous_event_hash': None
        }

        hash1 = audit_logger._hash_event(event1)
        hash2 = audit_logger._hash_event(event2)

        assert hash1 != hash2


class TestAuditLoggerEvidenceChain:
    """Test evidence chain tracking."""

    def test_get_evidence_chain_single_evidence(self, audit_logger):
        """Test retrieving chain for a single evidence item."""
        evidence_id = "evidence_123"

        # Log multiple events for this evidence
        for i in range(5):
            audit_logger.log_event(
                event_type=AuditEventType.EVIDENCE_PROCESSED,
                agent_id=f"agent_{i}",
                evidence_ids=[evidence_id],
                details={"step": i}
            )

        chain = audit_logger.get_evidence_chain(evidence_id)

        assert len(chain) == 5
        for event in chain:
            assert evidence_id in event['evidence_ids']

    def test_get_evidence_chain_multiple_evidence(self, audit_logger):
        """Test evidence chain with multiple evidence items."""
        evidence_ids = ["ev1", "ev2", "ev3"]

        # Log events with different evidence combinations
        audit_logger.log_event(
            event_type=AuditEventType.EVIDENCE_INGESTED,
            agent_id="ingestion_agent",
            evidence_ids=["ev1"]
        )

        audit_logger.log_event(
            event_type=AuditEventType.ANALYSIS_PERFORMED,
            agent_id="analyzer",
            evidence_ids=["ev1", "ev2"]
        )

        audit_logger.log_event(
            event_type=AuditEventType.OUTPUT_GENERATED,
            agent_id="reporter",
            evidence_ids=["ev1", "ev2", "ev3"]
        )

        # Check individual chains
        chain_ev1 = audit_logger.get_evidence_chain("ev1")
        chain_ev2 = audit_logger.get_evidence_chain("ev2")
        chain_ev3 = audit_logger.get_evidence_chain("ev3")

        assert len(chain_ev1) == 3
        assert len(chain_ev2) == 2
        assert len(chain_ev3) == 1

    def test_get_evidence_chain_nonexistent(self, audit_logger):
        """Test retrieving chain for evidence that doesn't exist."""
        chain = audit_logger.get_evidence_chain("nonexistent_evidence")

        assert chain == []


class TestAuditLoggerSessionSummary:
    """Test session summary generation."""

    def test_session_summary_basic_info(self, audit_logger):
        """Test that session summary contains basic information."""
        summary = audit_logger.get_session_summary()

        assert 'session_id' in summary
        assert 'total_events' in summary
        assert 'event_type_counts' in summary
        assert 'unique_evidence_items' in summary
        assert 'unique_agents' in summary
        assert 'chain_integrity_verified' in summary

    def test_session_summary_counts_events(self, audit_logger):
        """Test that session summary correctly counts events."""
        initial_count = audit_logger.get_session_summary()['total_events']

        # Log 10 more events
        for i in range(10):
            audit_logger.log_event(
                event_type=AuditEventType.ANALYSIS_PERFORMED,
                agent_id="test_agent"
            )

        final_summary = audit_logger.get_session_summary()

        assert final_summary['total_events'] == initial_count + 10

    def test_session_summary_tracks_unique_evidence(self, audit_logger):
        """Test that session summary tracks unique evidence items."""
        evidence_ids = ["ev1", "ev2", "ev3"]

        # Log events with overlapping evidence
        audit_logger.log_event(
            event_type=AuditEventType.EVIDENCE_INGESTED,
            agent_id="test_agent",
            evidence_ids=["ev1"]
        )

        audit_logger.log_event(
            event_type=AuditEventType.EVIDENCE_PROCESSED,
            agent_id="test_agent",
            evidence_ids=["ev1", "ev2"]  # ev1 repeated
        )

        audit_logger.log_event(
            event_type=AuditEventType.ANALYSIS_PERFORMED,
            agent_id="test_agent",
            evidence_ids=["ev2", "ev3"]  # ev2 repeated
        )

        summary = audit_logger.get_session_summary()

        # Should count 3 unique evidence items
        assert summary['unique_evidence_items'] == 3

    def test_session_summary_tracks_unique_agents(self, audit_logger):
        """Test that session summary tracks unique agents."""
        # Log events from different agents
        agents = ["parser", "analyzer", "reporter"]

        for agent in agents:
            audit_logger.log_event(
                event_type=AuditEventType.ANALYSIS_PERFORMED,
                agent_id=agent
            )

        # Log another event from existing agent
        audit_logger.log_event(
            event_type=AuditEventType.OUTPUT_GENERATED,
            agent_id="parser"
        )

        summary = audit_logger.get_session_summary()

        # Should count audit_logger (from init) + 3 agents = 4
        assert summary['unique_agents'] >= 3
        assert set(agents).issubset(set(summary['agents']))

    def test_session_summary_includes_timestamps(self, audit_logger):
        """Test that session summary includes start and end timestamps."""
        # Log several events
        for i in range(3):
            audit_logger.log_event(
                event_type=AuditEventType.ANALYSIS_PERFORMED,
                agent_id="test_agent"
            )

        summary = audit_logger.get_session_summary()

        assert 'start_time' in summary
        assert 'end_time' in summary
        assert summary['start_time'] is not None
        assert summary['end_time'] is not None


class TestAuditLoggerErrorHandling:
    """Test error handling and edge cases."""

    def test_handles_empty_evidence_ids(self, audit_logger):
        """Test logging event with empty evidence_ids list."""
        event_id = audit_logger.log_event(
            event_type=AuditEventType.ANALYSIS_PERFORMED,
            agent_id="test_agent",
            evidence_ids=[]
        )

        assert event_id is not None

    def test_handles_none_evidence_ids(self, audit_logger):
        """Test logging event with None evidence_ids."""
        event_id = audit_logger.log_event(
            event_type=AuditEventType.TOOL_EXECUTED,
            agent_id="test_agent",
            evidence_ids=None
        )

        assert event_id is not None

    def test_handles_empty_details(self, audit_logger):
        """Test logging event with empty details."""
        event_id = audit_logger.log_event(
            event_type=AuditEventType.ANALYSIS_PERFORMED,
            agent_id="test_agent",
            details={}
        )

        assert event_id is not None

    def test_handles_none_details(self, audit_logger):
        """Test logging event with None details."""
        event_id = audit_logger.log_event(
            event_type=AuditEventType.ERROR_OCCURRED,
            agent_id="test_agent",
            details=None
        )

        assert event_id is not None

    def test_verify_integrity_empty_log(self, temp_directory):
        """Test chain integrity verification on new logger."""
        logger = AuditLogger(log_directory=temp_directory)

        # Should be valid even with minimal events
        assert logger.verify_chain_integrity() is True


class TestAuditLoggerPerformance:
    """Test performance and scalability."""

    def test_handles_large_number_of_events(self, audit_logger):
        """Test logging a large number of events."""
        num_events = 1000

        for i in range(num_events):
            audit_logger.log_event(
                event_type=AuditEventType.ANALYSIS_PERFORMED,
                agent_id=f"agent_{i % 10}",
                evidence_ids=[f"evidence_{i}"]
            )

        summary = audit_logger.get_session_summary()

        assert summary['total_events'] >= num_events
        assert audit_logger.verify_chain_integrity() is True

    def test_handles_large_evidence_lists(self, audit_logger):
        """Test logging event with many evidence IDs."""
        evidence_ids = [f"evidence_{i}" for i in range(100)]

        event_id = audit_logger.log_event(
            event_type=AuditEventType.ANALYSIS_PERFORMED,
            agent_id="test_agent",
            evidence_ids=evidence_ids
        )

        assert event_id is not None

    def test_handles_large_details(self, audit_logger):
        """Test logging event with large details dictionary."""
        large_details = {
            f"key_{i}": f"value_{i}" * 100
            for i in range(50)
        }

        event_id = audit_logger.log_event(
            event_type=AuditEventType.OUTPUT_GENERATED,
            agent_id="test_agent",
            details=large_details
        )

        assert event_id is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
