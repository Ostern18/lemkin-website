"""
Evidentiary Compliance Test Suite

Tests that all agents meet LemkinAI's evidentiary standards:
- Chain-of-custody tracking
- Evidence integrity verification
- Audit trail completeness
- Human-in-the-loop gates
- Source provenance
- Data verification

These tests are critical for legal admissibility and compliance.
"""

import pytest
from pathlib import Path
import sys
import hashlib
import json

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from shared import AuditLogger, EvidenceHandler
from shared.audit_logger import AuditEventType
from shared.evidence_handler import EvidenceType, EvidenceStatus


class TestChainOfCustody:
    """Test chain-of-custody requirements."""

    def test_evidence_ingestion_creates_chain_entry(self, evidence_handler, audit_logger):
        """Test that evidence ingestion creates initial chain-of-custody entry."""
        evidence_id = evidence_handler.ingest_evidence(
            file_data=b"Test evidence",
            evidence_type=EvidenceType.DOCUMENT_PDF,
            source="Test Source"
        )

        audit_logger.log_event(
            event_type=AuditEventType.EVIDENCE_INGESTED,
            agent_id="ingestion_system",
            evidence_ids=[evidence_id],
            details={'source': 'Test Source'}
        )

        chain = audit_logger.get_evidence_chain(evidence_id)
        assert len(chain) > 0
        assert chain[0]['event_type'] == AuditEventType.EVIDENCE_INGESTED.value

    def test_evidence_processing_maintains_chain(self, evidence_handler, audit_logger):
        """Test that evidence processing maintains unbroken chain."""
        evidence_id = evidence_handler.ingest_evidence(
            file_data=b"Test evidence",
            evidence_type=EvidenceType.MEDICAL_RECORD,
            source="Hospital"
        )

        # Simulate processing pipeline
        processing_steps = [
            AuditEventType.EVIDENCE_INGESTED,
            AuditEventType.EVIDENCE_PROCESSED,
            AuditEventType.ANALYSIS_PERFORMED,
            AuditEventType.OUTPUT_GENERATED
        ]

        for step in processing_steps:
            audit_logger.log_event(
                event_type=step,
                agent_id="test_agent",
                evidence_ids=[evidence_id]
            )

        chain = audit_logger.get_evidence_chain(evidence_id)

        # Verify completeness
        assert len(chain) == len(processing_steps)

        # Verify chronological order and linkage
        for i in range(1, len(chain)):
            assert chain[i]['previous_event_hash'] == chain[i-1]['event_hash']

    def test_multi_agent_chain_preservation(self, evidence_handler, audit_logger):
        """Test that chain is preserved across multiple agents."""
        evidence_id = evidence_handler.ingest_evidence(
            file_data=b"Multi-agent evidence",
            evidence_type=EvidenceType.DOCUMENT_PDF,
            source="Source"
        )

        agents = ['parser', 'analyzer', 'reviewer', 'reporter']

        for agent_id in agents:
            audit_logger.log_event(
                event_type=AuditEventType.ANALYSIS_PERFORMED,
                agent_id=agent_id,
                evidence_ids=[evidence_id],
                details={'agent': agent_id}
            )

        chain = audit_logger.get_evidence_chain(evidence_id)

        # Verify all agents are in chain
        agent_ids_in_chain = [event['agent_id'] for event in chain]
        for agent_id in agents:
            assert agent_id in agent_ids_in_chain

        # Verify integrity
        assert audit_logger.verify_chain_integrity() is True

    def test_chain_includes_all_modifications(self, evidence_handler, audit_logger):
        """Test that all evidence modifications are recorded in chain."""
        evidence_id = evidence_handler.ingest_evidence(
            file_data=b"Evidence",
            evidence_type=EvidenceType.PHOTO,
            source="Camera"
        )

        # Record various modifications
        modifications = [
            {'action': 'uploaded', 'user': 'investigator_1'},
            {'action': 'analyzed', 'user': 'analyst_1'},
            {'action': 'reviewed', 'user': 'reviewer_1'},
            {'action': 'approved', 'user': 'supervisor_1'}
        ]

        for modification in modifications:
            audit_logger.log_event(
                event_type=AuditEventType.EVIDENCE_PROCESSED,
                agent_id="system",
                evidence_ids=[evidence_id],
                details=modification
            )

        chain = audit_logger.get_evidence_chain(evidence_id)

        # Verify all modifications are recorded
        for modification in modifications:
            matching_events = [
                event for event in chain
                if event.get('details', {}).get('action') == modification['action']
            ]
            assert len(matching_events) > 0


class TestEvidenceIntegrity:
    """Test evidence integrity verification."""

    def test_sha256_hash_calculated_on_ingestion(self, evidence_handler):
        """Test that SHA-256 hash is calculated and stored on ingestion."""
        file_data = b"Test evidence for hashing"

        evidence_id = evidence_handler.ingest_evidence(
            file_data=file_data,
            evidence_type=EvidenceType.VIDEO,
            source="Camera"
        )

        metadata = evidence_handler.get_metadata(evidence_id)
        expected_hash = hashlib.sha256(file_data).hexdigest()

        assert metadata.file_hash_sha256 == expected_hash

    def test_integrity_verification_succeeds_for_unmodified(self, evidence_handler):
        """Test that integrity verification passes for unmodified evidence."""
        file_data = b"Unmodified evidence"

        evidence_id = evidence_handler.ingest_evidence(
            file_data=file_data,
            evidence_type=EvidenceType.DOCUMENT_PDF,
            source="Source"
        )

        assert evidence_handler.verify_integrity(evidence_id) is True

    def test_integrity_verification_fails_for_tampered(self, evidence_handler):
        """Test that integrity verification fails for tampered evidence."""
        file_data = b"Original evidence"

        evidence_id = evidence_handler.ingest_evidence(
            file_data=file_data,
            evidence_type=EvidenceType.DOCUMENT_PDF,
            source="Source"
        )

        # Tamper with evidence file
        file_path = evidence_handler.files_directory / evidence_id
        with open(file_path, 'wb') as f:
            f.write(b"Tampered evidence")

        assert evidence_handler.verify_integrity(evidence_id) is False

    def test_hash_verification_across_pipeline(self, evidence_handler, audit_logger):
        """Test that hash can be verified at any point in pipeline."""
        file_data = b"Pipeline test evidence"

        evidence_id = evidence_handler.ingest_evidence(
            file_data=file_data,
            evidence_type=EvidenceType.MEDICAL_RECORD,
            source="Hospital"
        )

        initial_hash = evidence_handler.get_metadata(evidence_id).file_hash_sha256

        # Process through multiple steps
        for i in range(5):
            audit_logger.log_event(
                event_type=AuditEventType.ANALYSIS_PERFORMED,
                agent_id=f"agent_{i}",
                evidence_ids=[evidence_id]
            )

            # Verify hash unchanged
            current_metadata = evidence_handler.get_metadata(evidence_id)
            assert current_metadata.file_hash_sha256 == initial_hash

            # Verify integrity
            assert evidence_handler.verify_integrity(evidence_id) is True


class TestAuditTrailCompleteness:
    """Test that audit trails are complete and immutable."""

    def test_audit_trail_records_all_events(self, audit_logger):
        """Test that all events are recorded in audit trail."""
        event_types = [
            AuditEventType.AGENT_INITIALIZED,
            AuditEventType.EVIDENCE_INGESTED,
            AuditEventType.EVIDENCE_PROCESSED,
            AuditEventType.ANALYSIS_PERFORMED,
            AuditEventType.OUTPUT_GENERATED
        ]

        for event_type in event_types:
            audit_logger.log_event(
                event_type=event_type,
                agent_id="test_agent",
                details={'test': True}
            )

        summary = audit_logger.get_session_summary()

        for event_type in event_types:
            assert event_type.value in summary['event_type_counts']

    def test_audit_trail_is_append_only(self, audit_logger):
        """Test that audit trail is append-only (no deletions)."""
        # Log initial events
        for i in range(5):
            audit_logger.log_event(
                event_type=AuditEventType.ANALYSIS_PERFORMED,
                agent_id="test_agent",
                details={'step': i}
            )

        initial_count = audit_logger.get_session_summary()['total_events']

        # Log more events
        for i in range(5, 10):
            audit_logger.log_event(
                event_type=AuditEventType.ANALYSIS_PERFORMED,
                agent_id="test_agent",
                details={'step': i}
            )

        final_count = audit_logger.get_session_summary()['total_events']

        # Count should only increase
        assert final_count > initial_count
        assert final_count == initial_count + 5

    def test_audit_trail_includes_timestamps(self, audit_logger):
        """Test that all audit events include timestamps."""
        audit_logger.log_event(
            event_type=AuditEventType.ANALYSIS_PERFORMED,
            agent_id="test_agent"
        )

        with open(audit_logger.log_file, 'r') as f:
            lines = f.readlines()
            last_event = json.loads(lines[-1])

        assert 'timestamp' in last_event
        assert len(last_event['timestamp']) > 0

    def test_audit_trail_blockchain_integrity(self, audit_logger):
        """Test blockchain-style hash chain integrity."""
        # Log multiple events
        for i in range(20):
            audit_logger.log_event(
                event_type=AuditEventType.ANALYSIS_PERFORMED,
                agent_id=f"agent_{i % 3}",
                details={'iteration': i}
            )

        # Verify chain integrity
        assert audit_logger.verify_chain_integrity() is True

        # Read events manually and verify
        events = []
        with open(audit_logger.log_file, 'r') as f:
            for line in f:
                events.append(json.loads(line))

        # Verify each link
        for i in range(1, len(events)):
            prev_hash = events[i]['previous_event_hash']
            expected_prev_hash = events[i-1]['event_hash']
            assert prev_hash == expected_prev_hash


class TestHumanInTheLoop:
    """Test human-in-the-loop review gates."""

    def test_human_review_can_be_requested(self, audit_logger):
        """Test that human review can be requested and logged."""
        from shared.base_agent import BaseAgent

        # Mock agent for testing
        class TestAgent(BaseAgent):
            def process(self, input_data):
                return {}

        agent = TestAgent(
            agent_id="test_agent",
            system_prompt="Test",
            audit_logger=audit_logger
        )

        review_request = agent.request_human_review(
            item_for_review={'analysis': 'Test analysis'},
            review_type='evidence_verification',
            priority='high'
        )

        assert 'review_request_id' in review_request
        assert review_request['status'] == 'pending_review'

        # Verify logged
        summary = audit_logger.get_session_summary()
        assert 'human_review_requested' in summary['event_type_counts']

    def test_human_review_completion_is_logged(self, audit_logger):
        """Test that human review completion is properly logged."""
        from shared.base_agent import BaseAgent

        class TestAgent(BaseAgent):
            def process(self, input_data):
                return {}

        agent = TestAgent(
            agent_id="test_agent",
            system_prompt="Test",
            audit_logger=audit_logger
        )

        review_request = agent.request_human_review(
            item_for_review={'test': 'data'},
            review_type='analysis_verification'
        )

        agent.complete_human_review(
            review_request_id=review_request['review_request_id'],
            decision='approved',
            reviewer_notes='Looks good',
            reviewer_id='reviewer_123'
        )

        summary = audit_logger.get_session_summary()
        assert 'human_review_completed' in summary['event_type_counts']

    def test_review_rejection_is_traceable(self, audit_logger):
        """Test that review rejections are traceable in audit trail."""
        from shared.base_agent import BaseAgent

        class TestAgent(BaseAgent):
            def process(self, input_data):
                return {}

        agent = TestAgent(
            agent_id="test_agent",
            system_prompt="Test",
            audit_logger=audit_logger
        )

        review_request = agent.request_human_review(
            item_for_review={'analysis': 'Questionable conclusion'},
            review_type='legal_analysis',
            priority='critical'
        )

        agent.complete_human_review(
            review_request_id=review_request['review_request_id'],
            decision='rejected',
            reviewer_notes='Insufficient evidence',
            reviewer_id='supervisor_456'
        )

        # Read audit log to verify rejection details
        with open(audit_logger.log_file, 'r') as f:
            events = [json.loads(line) for line in f]

        review_events = [
            e for e in events
            if e['event_type'] == 'human_review_completed'
        ]

        assert len(review_events) > 0
        last_review = review_events[-1]
        assert last_review['details']['decision'] == 'rejected'


class TestSourceProvenance:
    """Test source provenance tracking."""

    def test_evidence_source_is_recorded(self, evidence_handler):
        """Test that evidence source is recorded on ingestion."""
        source = "Hospital ABC - Emergency Department"

        evidence_id = evidence_handler.ingest_evidence(
            file_data=b"Medical record",
            evidence_type=EvidenceType.MEDICAL_RECORD,
            source=source
        )

        metadata = evidence_handler.get_metadata(evidence_id)
        assert metadata.source == source

    def test_evidence_collection_metadata_is_stored(self, evidence_handler):
        """Test that collection metadata is properly stored."""
        evidence_id = evidence_handler.ingest_evidence(
            file_data=b"Collected evidence",
            evidence_type=EvidenceType.PHOTO,
            source="Crime Scene",
            collected_by="Investigator Jane Doe",
            collected_date="2024-01-15T14:30:00Z",
            location={'lat': 35.6892, 'lon': 51.3890, 'address': '123 Main St'}
        )

        metadata = evidence_handler.get_metadata(evidence_id)

        assert metadata.collected_by == "Investigator Jane Doe"
        assert metadata.collected_date == "2024-01-15T14:30:00Z"
        assert metadata.location['lat'] == 35.6892

    def test_evidence_custodian_is_tracked(self, evidence_handler):
        """Test that evidence custodian is tracked."""
        evidence_id = evidence_handler.ingest_evidence(
            file_data=b"Custodial evidence",
            evidence_type=EvidenceType.DOCUMENT_PDF,
            source="Legal Department",
            custodian="Legal Team Lead"
        )

        metadata = evidence_handler.get_metadata(evidence_id)
        assert metadata.custodian == "Legal Team Lead"

    def test_related_evidence_links_are_maintained(self, evidence_handler):
        """Test that links between related evidence are maintained."""
        # Create primary evidence
        primary_id = evidence_handler.ingest_evidence(
            file_data=b"Primary evidence",
            evidence_type=EvidenceType.WITNESS_STATEMENT,
            source="Witness Interview"
        )

        # Create supporting evidence
        supporting_ids = []
        for i in range(3):
            supporting_id = evidence_handler.ingest_evidence(
                file_data=f"Supporting evidence {i}".encode(),
                evidence_type=EvidenceType.PHOTO,
                source=f"Photo {i}"
            )
            supporting_ids.append(supporting_id)

            # Link to primary
            evidence_handler.link_evidence(primary_id, supporting_id)

        # Verify links
        related = evidence_handler.get_related_evidence(primary_id)

        assert len(related) == 3
        assert set(related) == set(supporting_ids)


class TestCrossAgentCompliance:
    """Test compliance when evidence flows across multiple agents."""

    def test_evidence_chain_preserved_across_agents(self, shared_infrastructure):
        """Test that evidence chain is preserved when passed between agents."""
        audit_logger = shared_infrastructure['audit_logger']
        evidence_handler = shared_infrastructure['evidence_handler']

        # Ingest evidence
        evidence_id = evidence_handler.ingest_evidence(
            file_data=b"Multi-agent evidence",
            evidence_type=EvidenceType.DOCUMENT_PDF,
            source="Source"
        )

        # Simulate multiple agents processing
        agents = ['parser', 'analyzer', 'reviewer', 'reporter']

        for agent_id in agents:
            audit_logger.log_event(
                event_type=AuditEventType.ANALYSIS_PERFORMED,
                agent_id=agent_id,
                evidence_ids=[evidence_id],
                details={'agent': agent_id}
            )

        # Verify complete chain
        chain = audit_logger.get_evidence_chain(evidence_id)
        assert len(chain) == len(agents)

        # Verify integrity
        assert audit_logger.verify_chain_integrity() is True

    def test_shared_audit_logger_across_agents(self, shared_infrastructure):
        """Test that multiple agents can share single audit logger."""
        audit_logger = shared_infrastructure['audit_logger']

        # Multiple "agents" logging to same audit logger
        for i in range(5):
            audit_logger.log_event(
                event_type=AuditEventType.ANALYSIS_PERFORMED,
                agent_id=f"agent_{i}",
                details={'agent_number': i}
            )

        summary = audit_logger.get_session_summary()

        assert summary['unique_agents'] >= 5
        assert summary['total_events'] >= 5

    def test_evidence_integrity_maintained_across_workflow(self, shared_infrastructure):
        """Test that evidence integrity is maintained through complete workflow."""
        evidence_handler = shared_infrastructure['evidence_handler']

        file_data = b"Workflow evidence"
        evidence_id = evidence_handler.ingest_evidence(
            file_data=file_data,
            evidence_type=EvidenceType.MEDICAL_RECORD,
            source="Hospital"
        )

        initial_hash = evidence_handler.get_metadata(evidence_id).file_hash_sha256

        # Simulate workflow steps
        for i in range(10):
            # Verify integrity at each step
            assert evidence_handler.verify_integrity(evidence_id) is True

            current_hash = evidence_handler.get_metadata(evidence_id).file_hash_sha256
            assert current_hash == initial_hash


class TestDataVerification:
    """Test data verification and validation."""

    def test_evidence_metadata_is_complete(self, evidence_handler):
        """Test that evidence metadata is complete and accurate."""
        evidence_id = evidence_handler.ingest_evidence(
            file_data=b"Complete metadata test",
            evidence_type=EvidenceType.FORENSIC_REPORT,
            source="Forensics Lab",
            original_filename="forensic_report_2024.pdf",
            collected_by="Forensic Analyst",
            collected_date="2024-01-15T09:00:00Z",
            case_id="CASE-001",
            tags=["forensics", "analysis", "final"],
            mime_type="application/pdf"
        )

        metadata = evidence_handler.get_metadata(evidence_id)

        # Verify all fields are present and correct
        assert metadata.evidence_id == evidence_id
        assert metadata.evidence_type == EvidenceType.FORENSIC_REPORT
        assert metadata.source == "Forensics Lab"
        assert metadata.original_filename == "forensic_report_2024.pdf"
        assert metadata.collected_by == "Forensic Analyst"
        assert metadata.case_id == "CASE-001"
        assert len(metadata.tags) == 3
        assert metadata.mime_type == "application/pdf"

    def test_evidence_summary_includes_verification_status(self, evidence_handler):
        """Test that evidence summary includes integrity verification status."""
        evidence_id = evidence_handler.ingest_evidence(
            file_data=b"Summary test",
            evidence_type=EvidenceType.DOCUMENT_PDF,
            source="Test"
        )

        summary = evidence_handler.generate_evidence_summary(evidence_id)

        assert 'integrity_verified' in summary
        assert summary['integrity_verified'] is True

    def test_audit_trail_includes_verification_events(self, audit_logger):
        """Test that audit trail includes verification events."""
        audit_logger.log_event(
            event_type=AuditEventType.CHAIN_VERIFIED,
            agent_id="verification_system",
            details={'verification_result': True}
        )

        summary = audit_logger.get_session_summary()
        assert 'chain_verified' in summary['event_type_counts']


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
