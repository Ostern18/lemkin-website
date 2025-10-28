"""
Unit Tests: EvidenceHandler

Tests evidence ingestion, tracking, and integrity verification.
"""

import pytest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from shared import EvidenceHandler
from shared.evidence_handler import EvidenceType, EvidenceStatus


class TestEvidenceHandlerInitialization:
    """Test EvidenceHandler initialization."""

    def test_creates_storage_directory(self, temp_directory):
        """Test that handler creates storage directory."""
        storage_dir = temp_directory / "evidence"
        handler = EvidenceHandler(storage_directory=storage_dir)

        assert storage_dir.exists()
        assert (storage_dir / "metadata").exists()
        assert (storage_dir / "files").exists()


class TestEvidenceIngestion:
    """Test evidence ingestion functionality."""

    def test_ingest_evidence_basic(self, evidence_handler):
        """Test basic evidence ingestion."""
        file_data = b"Test evidence content"

        evidence_id = evidence_handler.ingest_evidence(
            file_data=file_data,
            evidence_type=EvidenceType.DOCUMENT_PDF,
            source="Test Source"
        )

        assert evidence_id is not None
        assert len(evidence_id) > 0

    def test_ingest_evidence_stores_file(self, evidence_handler):
        """Test that evidence file is stored correctly."""
        file_data = b"Important evidence"

        evidence_id = evidence_handler.ingest_evidence(
            file_data=file_data,
            evidence_type=EvidenceType.PHOTO,
            source="Witness"
        )

        retrieved = evidence_handler.get_evidence(evidence_id)

        assert retrieved == file_data

    def test_ingest_evidence_calculates_hash(self, evidence_handler):
        """Test that SHA-256 hash is calculated on ingestion."""
        file_data = b"Evidence for hashing"

        evidence_id = evidence_handler.ingest_evidence(
            file_data=file_data,
            evidence_type=EvidenceType.VIDEO,
            source="Camera"
        )

        metadata = evidence_handler.get_metadata(evidence_id)

        import hashlib
        expected_hash = hashlib.sha256(file_data).hexdigest()

        assert metadata.file_hash_sha256 == expected_hash

    def test_ingest_with_all_metadata(self, evidence_handler):
        """Test ingestion with complete metadata."""
        file_data = b"Complete metadata test"

        evidence_id = evidence_handler.ingest_evidence(
            file_data=file_data,
            evidence_type=EvidenceType.MEDICAL_RECORD,
            source="Hospital XYZ",
            original_filename="medical_report.pdf",
            collected_by="Investigator Smith",
            collected_date="2024-01-15T10:00:00Z",
            case_id="CASE-001",
            tags=["medical", "torture"],
            location={'lat': 35.6892, 'lon': 51.3890},
            mime_type="application/pdf",
            custodian="Legal Team"
        )

        metadata = evidence_handler.get_metadata(evidence_id)

        assert metadata.source == "Hospital XYZ"
        assert metadata.original_filename == "medical_report.pdf"
        assert metadata.collected_by == "Investigator Smith"
        assert metadata.case_id == "CASE-001"
        assert "medical" in metadata.tags
        assert metadata.custodian == "Legal Team"


class TestEvidenceRetrieval:
    """Test evidence retrieval functionality."""

    def test_get_evidence_exists(self, evidence_handler):
        """Test retrieving evidence that exists."""
        file_data = b"Retrievable evidence"

        evidence_id = evidence_handler.ingest_evidence(
            file_data=file_data,
            evidence_type=EvidenceType.DOCUMENT_PDF,
            source="Test"
        )

        retrieved = evidence_handler.get_evidence(evidence_id)

        assert retrieved == file_data

    def test_get_evidence_nonexistent(self, evidence_handler):
        """Test retrieving evidence that doesn't exist."""
        retrieved = evidence_handler.get_evidence("nonexistent_id")

        assert retrieved is None

    def test_get_metadata_exists(self, evidence_handler):
        """Test retrieving metadata that exists."""
        evidence_id = evidence_handler.ingest_evidence(
            file_data=b"Test",
            evidence_type=EvidenceType.PHOTO,
            source="Camera",
            case_id="CASE-123"
        )

        metadata = evidence_handler.get_metadata(evidence_id)

        assert metadata is not None
        assert metadata.evidence_id == evidence_id
        assert metadata.case_id == "CASE-123"

    def test_get_metadata_nonexistent(self, evidence_handler):
        """Test retrieving metadata that doesn't exist."""
        metadata = evidence_handler.get_metadata("nonexistent_id")

        assert metadata is None


class TestIntegrityVerification:
    """Test evidence integrity verification."""

    def test_verify_integrity_valid(self, evidence_handler):
        """Test integrity verification passes for valid evidence."""
        file_data = b"Integrity test data"

        evidence_id = evidence_handler.ingest_evidence(
            file_data=file_data,
            evidence_type=EvidenceType.VIDEO,
            source="Test"
        )

        assert evidence_handler.verify_integrity(evidence_id) is True

    def test_verify_integrity_tampered(self, evidence_handler):
        """Test integrity verification fails if file is tampered."""
        file_data = b"Original data"

        evidence_id = evidence_handler.ingest_evidence(
            file_data=file_data,
            evidence_type=EvidenceType.DOCUMENT_PDF,
            source="Test"
        )

        # Tamper with the file
        file_path = evidence_handler.files_directory / evidence_id
        with open(file_path, 'wb') as f:
            f.write(b"Tampered data")

        assert evidence_handler.verify_integrity(evidence_id) is False

    def test_verify_integrity_nonexistent(self, evidence_handler):
        """Test integrity verification for nonexistent evidence."""
        assert evidence_handler.verify_integrity("nonexistent") is False


class TestEvidenceStatus:
    """Test evidence status management."""

    def test_initial_status_is_ingested(self, evidence_handler):
        """Test that new evidence has INGESTED status."""
        evidence_id = evidence_handler.ingest_evidence(
            file_data=b"Test",
            evidence_type=EvidenceType.PHOTO,
            source="Test"
        )

        metadata = evidence_handler.get_metadata(evidence_id)

        assert metadata.status == EvidenceStatus.INGESTED

    def test_update_status(self, evidence_handler):
        """Test updating evidence status."""
        evidence_id = evidence_handler.ingest_evidence(
            file_data=b"Test",
            evidence_type=EvidenceType.DOCUMENT_PDF,
            source="Test"
        )

        success = evidence_handler.update_status(
            evidence_id=evidence_id,
            status=EvidenceStatus.PROCESSED,
            notes="Analysis complete"
        )

        assert success is True

        metadata = evidence_handler.get_metadata(evidence_id)
        assert metadata.status == EvidenceStatus.PROCESSED
        assert metadata.verification_notes == "Analysis complete"

    def test_update_status_nonexistent(self, evidence_handler):
        """Test updating status for nonexistent evidence."""
        success = evidence_handler.update_status(
            evidence_id="nonexistent",
            status=EvidenceStatus.PROCESSED
        )

        assert success is False


class TestEvidenceLinking:
    """Test linking related evidence."""

    def test_link_evidence(self, evidence_handler):
        """Test linking two pieces of evidence."""
        ev1 = evidence_handler.ingest_evidence(
            file_data=b"Evidence 1",
            evidence_type=EvidenceType.PHOTO,
            source="Camera"
        )

        ev2 = evidence_handler.ingest_evidence(
            file_data=b"Evidence 2",
            evidence_type=EvidenceType.VIDEO,
            source="Camera"
        )

        success = evidence_handler.link_evidence(ev1, ev2)

        assert success is True

        metadata = evidence_handler.get_metadata(ev1)
        assert ev2 in metadata.related_evidence

    def test_get_related_evidence(self, evidence_handler):
        """Test retrieving related evidence."""
        ev1 = evidence_handler.ingest_evidence(
            file_data=b"Main evidence",
            evidence_type=EvidenceType.DOCUMENT_PDF,
            source="Test"
        )

        related_ids = []
        for i in range(3):
            related_id = evidence_handler.ingest_evidence(
                file_data=f"Related {i}".encode(),
                evidence_type=EvidenceType.PHOTO,
                source="Test"
            )
            related_ids.append(related_id)
            evidence_handler.link_evidence(ev1, related_id)

        retrieved_related = evidence_handler.get_related_evidence(ev1)

        assert len(retrieved_related) == 3
        assert set(retrieved_related) == set(related_ids)


class TestEvidenceSearch:
    """Test evidence search functionality."""

    def test_search_by_tags(self, evidence_handler):
        """Test searching evidence by tags."""
        # Create evidence with different tags
        ev1 = evidence_handler.ingest_evidence(
            file_data=b"Medical",
            evidence_type=EvidenceType.MEDICAL_RECORD,
            source="Hospital",
            tags=["medical", "torture"]
        )

        ev2 = evidence_handler.ingest_evidence(
            file_data=b"Photo",
            evidence_type=EvidenceType.PHOTO,
            source="Witness",
            tags=["photo", "detention"]
        )

        ev3 = evidence_handler.ingest_evidence(
            file_data=b"Medical photo",
            evidence_type=EvidenceType.PHOTO,
            source="Hospital",
            tags=["medical", "photo"]
        )

        results = evidence_handler.search_by_tags(["medical"])

        assert len(results) >= 2
        assert ev1 in results
        assert ev3 in results

    def test_search_by_case(self, evidence_handler):
        """Test searching evidence by case ID."""
        case_id = "CASE-TEST-123"

        # Create evidence for different cases
        ev1 = evidence_handler.ingest_evidence(
            file_data=b"Case 1",
            evidence_type=EvidenceType.DOCUMENT_PDF,
            source="Test",
            case_id=case_id
        )

        ev2 = evidence_handler.ingest_evidence(
            file_data=b"Case 2",
            evidence_type=EvidenceType.PHOTO,
            source="Test",
            case_id="OTHER-CASE"
        )

        ev3 = evidence_handler.ingest_evidence(
            file_data=b"Case 1 again",
            evidence_type=EvidenceType.VIDEO,
            source="Test",
            case_id=case_id
        )

        results = evidence_handler.search_by_case(case_id)

        assert len(results) == 2
        assert ev1 in results
        assert ev3 in results
        assert ev2 not in results


class TestEvidenceSummary:
    """Test evidence summary generation."""

    def test_generate_evidence_summary(self, evidence_handler):
        """Test generating comprehensive evidence summary."""
        evidence_id = evidence_handler.ingest_evidence(
            file_data=b"Summary test",
            evidence_type=EvidenceType.MEDICAL_RECORD,
            source="Hospital ABC",
            original_filename="report.pdf",
            collected_by="Dr. Smith",
            collected_date="2024-01-15T10:00:00Z",
            case_id="CASE-001",
            tags=["medical", "torture"],
            custodian="Legal Team"
        )

        summary = evidence_handler.generate_evidence_summary(evidence_id)

        assert summary['evidence_id'] == evidence_id
        assert summary['type'] == EvidenceType.MEDICAL_RECORD.value
        assert summary['source'] == "Hospital ABC"
        assert summary['case_id'] == "CASE-001"
        assert 'file_info' in summary
        assert 'integrity_verified' in summary
        assert summary['integrity_verified'] is True

    def test_summary_includes_related_evidence(self, evidence_handler):
        """Test that summary includes related evidence count."""
        ev1 = evidence_handler.ingest_evidence(
            file_data=b"Main",
            evidence_type=EvidenceType.DOCUMENT_PDF,
            source="Test"
        )

        ev2 = evidence_handler.ingest_evidence(
            file_data=b"Related",
            evidence_type=EvidenceType.PHOTO,
            source="Test"
        )

        evidence_handler.link_evidence(ev1, ev2)

        summary = evidence_handler.generate_evidence_summary(ev1)

        assert summary['related_evidence_count'] == 1
        assert ev2 in summary['related_evidence_ids']


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
