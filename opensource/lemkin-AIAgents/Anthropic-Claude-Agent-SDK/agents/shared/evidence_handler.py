"""
Evidence Handler for LemkinAI Agents
Manages evidence ingestion, tracking, and verification for legal compliance.
"""

import hashlib
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from enum import Enum
import json


class EvidenceType(Enum):
    """Types of evidence that can be processed."""
    DOCUMENT_PDF = "document_pdf"
    DOCUMENT_IMAGE = "document_image"
    PHOTO = "photo"
    VIDEO = "video"
    AUDIO = "audio"
    TEXT = "text"
    SOCIAL_MEDIA = "social_media"
    SATELLITE_IMAGERY = "satellite_imagery"
    AERIAL_IMAGERY = "aerial_imagery"
    OSINT_REPORT = "osint_report"
    INTELLIGENCE_BRIEF = "intelligence_brief"
    IMAGERY_ANALYSIS = "imagery_analysis"
    MEDICAL_RECORD = "medical_record"
    FORENSIC_REPORT = "forensic_report"
    WITNESS_STATEMENT = "witness_statement"
    LEGAL_RESEARCH = "legal_research"
    HISTORICAL_RESEARCH = "historical_research"
    OTHER = "other"


class EvidenceStatus(Enum):
    """Status of evidence in the processing pipeline."""
    INGESTED = "ingested"
    VERIFIED = "verified"
    PROCESSING = "processing"
    PROCESSED = "processed"
    FLAGGED = "flagged"
    REVIEWED = "reviewed"
    ARCHIVED = "archived"


@dataclass
class EvidenceMetadata:
    """
    Metadata for a piece of evidence.

    Includes provenance, technical details, and chain-of-custody information.
    """
    evidence_id: str
    evidence_type: EvidenceType
    source: str  # Origin of the evidence
    collected_by: Optional[str]
    collected_date: Optional[str]
    ingested_date: str
    original_filename: Optional[str]
    file_size_bytes: Optional[int]
    file_hash_sha256: str
    mime_type: Optional[str]
    tags: List[str]
    case_id: Optional[str]
    location: Optional[Dict[str, Any]]  # Geographic location if known
    related_evidence: List[str]  # IDs of related evidence
    status: EvidenceStatus
    verification_notes: Optional[str]
    custodian: Optional[str]  # Current custodian
    classification: Optional[str]  # Security classification if any


class EvidenceHandler:
    """
    Handles evidence ingestion, tracking, and verification.

    Provides:
    - Unique evidence identification
    - Hash-based integrity verification
    - Metadata management
    - Chain-of-custody tracking
    - Evidence retrieval and search
    """

    def __init__(self, storage_directory: Optional[Path] = None):
        """
        Initialize evidence handler.

        Args:
            storage_directory: Directory for evidence storage (defaults to ./evidence_store)
        """
        self.storage_directory = storage_directory or Path("./evidence_store")
        self.storage_directory.mkdir(parents=True, exist_ok=True)

        # Metadata directory
        self.metadata_directory = self.storage_directory / "metadata"
        self.metadata_directory.mkdir(parents=True, exist_ok=True)

        # Files directory
        self.files_directory = self.storage_directory / "files"
        self.files_directory.mkdir(parents=True, exist_ok=True)

    def ingest_evidence(
        self,
        file_data: bytes,
        evidence_type: EvidenceType,
        source: str,
        original_filename: Optional[str] = None,
        collected_by: Optional[str] = None,
        collected_date: Optional[str] = None,
        case_id: Optional[str] = None,
        tags: Optional[List[str]] = None,
        location: Optional[Dict[str, Any]] = None,
        mime_type: Optional[str] = None,
        custodian: Optional[str] = None
    ) -> str:
        """
        Ingest new evidence into the system.

        Args:
            file_data: Raw bytes of the evidence file
            evidence_type: Type of evidence
            source: Source/origin of evidence
            original_filename: Original filename if known
            collected_by: Who collected the evidence
            collected_date: When evidence was collected (ISO format)
            case_id: Associated case identifier
            tags: List of tags for categorization
            location: Geographic location data
            mime_type: MIME type of the file
            custodian: Current custodian

        Returns:
            Evidence ID
        """
        # Generate unique evidence ID
        evidence_id = str(uuid.uuid4())

        # Calculate file hash for integrity
        file_hash = hashlib.sha256(file_data).hexdigest()

        # Create metadata
        metadata = EvidenceMetadata(
            evidence_id=evidence_id,
            evidence_type=evidence_type,
            source=source,
            collected_by=collected_by,
            collected_date=collected_date,
            ingested_date=datetime.now(timezone.utc).isoformat(),
            original_filename=original_filename,
            file_size_bytes=len(file_data),
            file_hash_sha256=file_hash,
            mime_type=mime_type,
            tags=tags or [],
            case_id=case_id,
            location=location,
            related_evidence=[],
            status=EvidenceStatus.INGESTED,
            verification_notes=None,
            custodian=custodian
        )

        # Store file
        file_path = self.files_directory / evidence_id
        with open(file_path, 'wb') as f:
            f.write(file_data)

        # Store metadata
        self._save_metadata(evidence_id, metadata)

        return evidence_id

    def get_evidence(self, evidence_id: str) -> Optional[bytes]:
        """
        Retrieve evidence file by ID.

        Args:
            evidence_id: Evidence identifier

        Returns:
            Evidence file bytes, or None if not found
        """
        file_path = self.files_directory / evidence_id

        if not file_path.exists():
            return None

        with open(file_path, 'rb') as f:
            return f.read()

    def get_metadata(self, evidence_id: str) -> Optional[EvidenceMetadata]:
        """
        Retrieve evidence metadata by ID.

        Args:
            evidence_id: Evidence identifier

        Returns:
            EvidenceMetadata object, or None if not found
        """
        metadata_path = self.metadata_directory / f"{evidence_id}.json"

        if not metadata_path.exists():
            return None

        with open(metadata_path, 'r') as f:
            data = json.load(f)

        # Reconstruct EvidenceMetadata
        data['evidence_type'] = EvidenceType(data['evidence_type'])
        data['status'] = EvidenceStatus(data['status'])

        return EvidenceMetadata(**data)

    def verify_integrity(self, evidence_id: str) -> bool:
        """
        Verify integrity of evidence by comparing current hash to stored hash.

        Args:
            evidence_id: Evidence identifier

        Returns:
            True if integrity verified, False otherwise
        """
        metadata = self.get_metadata(evidence_id)
        if not metadata:
            return False

        file_data = self.get_evidence(evidence_id)
        if not file_data:
            return False

        # Calculate current hash
        current_hash = hashlib.sha256(file_data).hexdigest()

        return current_hash == metadata.file_hash_sha256

    def update_status(
        self,
        evidence_id: str,
        status: EvidenceStatus,
        notes: Optional[str] = None
    ) -> bool:
        """
        Update evidence status.

        Args:
            evidence_id: Evidence identifier
            status: New status
            notes: Optional notes about status change

        Returns:
            True if updated successfully, False otherwise
        """
        metadata = self.get_metadata(evidence_id)
        if not metadata:
            return False

        metadata.status = status
        if notes:
            metadata.verification_notes = notes

        self._save_metadata(evidence_id, metadata)
        return True

    def link_evidence(self, evidence_id: str, related_evidence_id: str) -> bool:
        """
        Link two pieces of evidence as related.

        Args:
            evidence_id: First evidence ID
            related_evidence_id: Related evidence ID

        Returns:
            True if linked successfully, False otherwise
        """
        metadata = self.get_metadata(evidence_id)
        if not metadata:
            return False

        if related_evidence_id not in metadata.related_evidence:
            metadata.related_evidence.append(related_evidence_id)
            self._save_metadata(evidence_id, metadata)

        return True

    def search_by_tags(self, tags: List[str]) -> List[str]:
        """
        Search for evidence by tags.

        Args:
            tags: List of tags to search for

        Returns:
            List of evidence IDs matching any of the tags
        """
        results = []

        for metadata_file in self.metadata_directory.glob("*.json"):
            evidence_id = metadata_file.stem
            metadata = self.get_metadata(evidence_id)

            if metadata and any(tag in metadata.tags for tag in tags):
                results.append(evidence_id)

        return results

    def search_by_case(self, case_id: str) -> List[str]:
        """
        Search for all evidence related to a case.

        Args:
            case_id: Case identifier

        Returns:
            List of evidence IDs for this case
        """
        results = []

        for metadata_file in self.metadata_directory.glob("*.json"):
            evidence_id = metadata_file.stem
            metadata = self.get_metadata(evidence_id)

            if metadata and metadata.case_id == case_id:
                results.append(evidence_id)

        return results

    def get_related_evidence(self, evidence_id: str) -> List[str]:
        """
        Get all evidence linked to a given piece of evidence.

        Args:
            evidence_id: Evidence identifier

        Returns:
            List of related evidence IDs
        """
        metadata = self.get_metadata(evidence_id)
        if not metadata:
            return []

        return metadata.related_evidence

    def _save_metadata(self, evidence_id: str, metadata: EvidenceMetadata):
        """
        Save metadata to file.

        Args:
            evidence_id: Evidence identifier
            metadata: Metadata to save
        """
        metadata_path = self.metadata_directory / f"{evidence_id}.json"

        # Convert to dict with proper enum serialization
        metadata_dict = asdict(metadata)
        metadata_dict['evidence_type'] = metadata.evidence_type.value
        metadata_dict['status'] = metadata.status.value

        with open(metadata_path, 'w') as f:
            json.dump(metadata_dict, f, indent=2)

    def generate_evidence_summary(self, evidence_id: str) -> Dict[str, Any]:
        """
        Generate comprehensive summary of evidence.

        Args:
            evidence_id: Evidence identifier

        Returns:
            Dictionary containing all evidence information
        """
        metadata = self.get_metadata(evidence_id)
        if not metadata:
            return {"error": "Evidence not found"}

        # Verify integrity
        integrity_verified = self.verify_integrity(evidence_id)

        # Get related evidence
        related = self.get_related_evidence(evidence_id)

        return {
            "evidence_id": evidence_id,
            "type": metadata.evidence_type.value,
            "status": metadata.status.value,
            "source": metadata.source,
            "collected_by": metadata.collected_by,
            "collected_date": metadata.collected_date,
            "ingested_date": metadata.ingested_date,
            "file_info": {
                "original_filename": metadata.original_filename,
                "size_bytes": metadata.file_size_bytes,
                "mime_type": metadata.mime_type,
                "hash_sha256": metadata.file_hash_sha256
            },
            "case_id": metadata.case_id,
            "tags": metadata.tags,
            "location": metadata.location,
            "related_evidence_count": len(related),
            "related_evidence_ids": related,
            "custodian": metadata.custodian,
            "integrity_verified": integrity_verified,
            "verification_notes": metadata.verification_notes
        }
