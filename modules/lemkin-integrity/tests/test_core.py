import pytest
import tempfile
import sqlite3
from pathlib import Path
from datetime import datetime, timezone

from lemkin_integrity.core import (
    EvidenceIntegrityManager,
    EvidenceMetadata,
    ActionType,
    IntegrityStatus
)

@pytest.fixture
def temp_db():
    """Create temporary database for testing"""
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        db_path = f.name
    yield db_path
    Path(db_path).unlink(missing_ok=True)

@pytest.fixture
def temp_file():
    """Create temporary file for testing"""
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
        f.write("This is test evidence content.")
        file_path = Path(f.name)
    yield file_path
    file_path.unlink(missing_ok=True)

@pytest.fixture
def sample_metadata():
    """Sample evidence metadata"""
    return EvidenceMetadata(
        filename="test_evidence.txt",
        file_size=1024,
        mime_type="text/plain",
        created_date=datetime.now(timezone.utc),
        source="Test Source",
        case_id="TEST-001",
        collector="Test Collector",
        location="Test Location",
        description="Test evidence for unit testing",
        tags=["test", "evidence"]
    )

class TestEvidenceIntegrityManager:
    
    def test_initialization(self, temp_db):
        """Test manager initialization"""
        manager = EvidenceIntegrityManager(temp_db)
        assert Path(temp_db).exists()
        
        # Check database tables
        conn = sqlite3.connect(temp_db)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
        conn.close()
        
        expected_tables = ['evidence', 'custody_chain', 'integrity_checks']
        for table in expected_tables:
            assert table in tables
    
    def test_generate_evidence_hash(self, temp_db, temp_file, sample_metadata):
        """Test evidence hash generation"""
        manager = EvidenceIntegrityManager(temp_db)
        evidence_hash = manager.generate_evidence_hash(temp_file, sample_metadata)
        
        assert evidence_hash.evidence_id is not None
        assert len(evidence_hash.sha256_hash) == 64  # SHA-256 hex length
        assert len(evidence_hash.sha512_hash) == 128  # SHA-512 hex length
        assert evidence_hash.metadata == sample_metadata
        assert evidence_hash.chain_start is True
    
    def test_create_custody_entry(self, temp_db, temp_file, sample_metadata):
        """Test custody entry creation"""
        manager = EvidenceIntegrityManager(temp_db)
        evidence_hash = manager.generate_evidence_hash(temp_file, sample_metadata)
        
        custody_entry = manager.create_custody_entry(
            evidence_id=evidence_hash.evidence_id,
            action=ActionType.ACCESSED,
            actor="Test Actor",
            location="Test Location",
            notes="Test access"
        )
        
        assert custody_entry.evidence_id == evidence_hash.evidence_id
        assert custody_entry.action == ActionType.ACCESSED
        assert custody_entry.actor == "Test Actor"
        assert custody_entry.signature is not None
    
    def test_verify_integrity_success(self, temp_db, temp_file, sample_metadata):
        """Test successful integrity verification"""
        manager = EvidenceIntegrityManager(temp_db)
        evidence_hash = manager.generate_evidence_hash(temp_file, sample_metadata)
        
        integrity_report = manager.verify_integrity(evidence_hash.evidence_id, temp_file)
        
        assert integrity_report.status == IntegrityStatus.VERIFIED
        assert integrity_report.hash_verified is True
        assert integrity_report.custody_verified is True
        assert integrity_report.admissible is True
        assert len(integrity_report.issues) == 0
    
    def test_verify_integrity_file_modified(self, temp_db, temp_file, sample_metadata):
        """Test integrity verification with modified file"""
        manager = EvidenceIntegrityManager(temp_db)
        evidence_hash = manager.generate_evidence_hash(temp_file, sample_metadata)
        
        # Modify the file
        with open(temp_file, 'a') as f:
            f.write("\nModified content")
        
        integrity_report = manager.verify_integrity(evidence_hash.evidence_id, temp_file)
        
        assert integrity_report.status == IntegrityStatus.COMPROMISED
        assert integrity_report.hash_verified is False
        assert integrity_report.admissible is False
        assert len(integrity_report.issues) > 0
    
    def test_get_custody_chain(self, temp_db, temp_file, sample_metadata):
        """Test custody chain retrieval"""
        manager = EvidenceIntegrityManager(temp_db)
        evidence_hash = manager.generate_evidence_hash(temp_file, sample_metadata)
        
        # Add additional custody entries
        manager.create_custody_entry(
            evidence_id=evidence_hash.evidence_id,
            action=ActionType.ACCESSED,
            actor="Actor 1"
        )
        manager.create_custody_entry(
            evidence_id=evidence_hash.evidence_id,
            action=ActionType.TRANSFERRED,
            actor="Actor 2"
        )
        
        custody_chain = manager.get_custody_chain(evidence_hash.evidence_id)
        
        assert len(custody_chain) == 3  # CREATED + 2 additional
        assert custody_chain[0].action == ActionType.CREATED
        assert custody_chain[1].action == ActionType.ACCESSED
        assert custody_chain[2].action == ActionType.TRANSFERRED
    
    def test_generate_court_manifest(self, temp_db, temp_file, sample_metadata):
        """Test court manifest generation"""
        manager = EvidenceIntegrityManager(temp_db)
        evidence_hash = manager.generate_evidence_hash(temp_file, sample_metadata)
        
        manifest = manager.generate_court_manifest("TEST-001")
        
        assert manifest.case_id == "TEST-001"
        assert manifest.evidence_count == 1
        assert len(manifest.evidence_items) == 1
        assert manifest.evidence_items[0]['evidence_id'] == evidence_hash.evidence_id
    
    def test_export_evidence_package(self, temp_db, temp_file, sample_metadata):
        """Test evidence package export"""
        manager = EvidenceIntegrityManager(temp_db)
        evidence_hash = manager.generate_evidence_hash(temp_file, sample_metadata)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            export_summary = manager.export_evidence_package("TEST-001", temp_dir)
            
            assert export_summary['case_id'] == "TEST-001"
            assert export_summary['evidence_count'] == 1
            
            # Check exported files
            output_dir = Path(temp_dir)
            assert (output_dir / f"manifest_TEST-001.json").exists()
            assert (output_dir / "custody_chains").exists()
            assert (output_dir / "integrity_reports").exists()