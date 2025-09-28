"""
Core functionality for evidence integrity verification and chain of custody management.
"""

import hashlib
import json
import sqlite3
import uuid
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.exceptions import InvalidSignature
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ActionType(Enum):
    """Types of actions that can be performed on evidence"""
    CREATED = "created"
    ACCESSED = "accessed"
    TRANSFERRED = "transferred"
    COPIED = "copied"
    MODIFIED = "modified"
    DESTROYED = "destroyed"
    ARCHIVED = "archived"
    RESTORED = "restored"


class IntegrityStatus(Enum):
    """Status of evidence integrity verification"""
    VERIFIED = "verified"
    COMPROMISED = "compromised"
    UNKNOWN = "unknown"
    PENDING = "pending"


@dataclass
class EvidenceMetadata:
    """Metadata associated with evidence"""
    filename: str
    file_size: int
    mime_type: str
    created_date: datetime
    source: str
    case_id: str
    collector: str
    location: Optional[str] = None
    description: Optional[str] = None
    tags: Optional[List[str]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        data = asdict(self)
        data['created_date'] = self.created_date.isoformat()
        return data


@dataclass
class EvidenceHash:
    """Cryptographic hash of evidence with metadata"""
    evidence_id: str
    sha256_hash: str
    sha512_hash: str
    md5_hash: str
    timestamp: datetime
    metadata: EvidenceMetadata
    signature: Optional[str] = None
    chain_start: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        data = {
            'evidence_id': self.evidence_id,
            'sha256_hash': self.sha256_hash,
            'sha512_hash': self.sha512_hash,
            'md5_hash': self.md5_hash,
            'timestamp': self.timestamp.isoformat(),
            'metadata': self.metadata.to_dict(),
            'signature': self.signature,
            'chain_start': self.chain_start
        }
        return data


@dataclass
class CustodyEntry:
    """Entry in the chain of custody"""
    entry_id: str
    evidence_id: str
    timestamp: datetime
    action: ActionType
    actor: str
    location: Optional[str] = None
    notes: Optional[str] = None
    signature: Optional[str] = None
    previous_hash: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'entry_id': self.entry_id,
            'evidence_id': self.evidence_id,
            'timestamp': self.timestamp.isoformat(),
            'action': self.action.value,
            'actor': self.actor,
            'location': self.location,
            'notes': self.notes,
            'signature': self.signature,
            'previous_hash': self.previous_hash
        }


@dataclass
class IntegrityReport:
    """Report on evidence integrity verification"""
    evidence_id: str
    timestamp: datetime
    status: IntegrityStatus
    hash_verified: bool
    custody_verified: bool
    admissible: bool
    issues: List[str]
    recommendations: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'evidence_id': self.evidence_id,
            'timestamp': self.timestamp.isoformat(),
            'status': self.status.value,
            'hash_verified': self.hash_verified,
            'custody_verified': self.custody_verified,
            'admissible': self.admissible,
            'issues': self.issues,
            'recommendations': self.recommendations
        }


@dataclass
class CourtManifest:
    """Manifest for court submission"""
    case_id: str
    generated_date: datetime
    evidence_count: int
    total_size: int
    evidence_items: List[Dict[str, Any]]
    integrity_summary: Dict[str, int]
    signature: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'case_id': self.case_id,
            'generated_date': self.generated_date.isoformat(),
            'evidence_count': self.evidence_count,
            'total_size': self.total_size,
            'evidence_items': self.evidence_items,
            'integrity_summary': self.integrity_summary,
            'signature': self.signature
        }


class EvidenceIntegrityManager:
    """Main class for managing evidence integrity and chain of custody"""
    
    def __init__(self, db_path: str = "evidence_integrity.db"):
        """
        Initialize the evidence integrity manager
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self._private_key = None
        self._public_key = None
        self._init_database()
        self._init_cryptography()
    
    def _init_database(self) -> None:
        """Initialize SQLite database with required tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Evidence table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS evidence (
                evidence_id TEXT PRIMARY KEY,
                filename TEXT NOT NULL,
                file_size INTEGER NOT NULL,
                mime_type TEXT NOT NULL,
                sha256_hash TEXT NOT NULL,
                sha512_hash TEXT NOT NULL,
                md5_hash TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                metadata TEXT NOT NULL,
                signature TEXT,
                case_id TEXT NOT NULL,
                collector TEXT NOT NULL,
                INDEX(case_id),
                INDEX(collector)
            )
        """)
        
        # Custody chain table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS custody_chain (
                entry_id TEXT PRIMARY KEY,
                evidence_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                action TEXT NOT NULL,
                actor TEXT NOT NULL,
                location TEXT,
                notes TEXT,
                signature TEXT,
                previous_hash TEXT,
                FOREIGN KEY (evidence_id) REFERENCES evidence (evidence_id),
                INDEX(evidence_id),
                INDEX(timestamp)
            )
        """)
        
        # Integrity checks table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS integrity_checks (
                check_id TEXT PRIMARY KEY,
                evidence_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                status TEXT NOT NULL,
                hash_verified BOOLEAN NOT NULL,
                custody_verified BOOLEAN NOT NULL,
                issues TEXT,
                FOREIGN KEY (evidence_id) REFERENCES evidence (evidence_id),
                INDEX(evidence_id),
                INDEX(timestamp)
            )
        """)
        
        conn.commit()
        conn.close()
        logger.info(f"Database initialized at {self.db_path}")
    
    def _init_cryptography(self) -> None:
        """Initialize cryptographic keys for digital signatures"""
        try:
            # Generate RSA key pair for digital signatures
            self._private_key = rsa.generate_private_key(
                public_exponent=65537,
                key_size=2048
            )
            self._public_key = self._private_key.public_key()
            logger.info("Cryptographic keys initialized")
        except Exception as e:
            logger.error(f"Failed to initialize cryptographic keys: {e}")
            raise
    
    def _calculate_file_hashes(self, file_path: Path) -> Dict[str, str]:
        """Calculate multiple hash types for a file"""
        hashes_obj = {
            'md5': hashlib.md5(),
            'sha256': hashlib.sha256(),
            'sha512': hashlib.sha512()
        }
        
        with open(file_path, 'rb') as f:
            while chunk := f.read(8192):
                for hash_obj in hashes_obj.values():
                    hash_obj.update(chunk)
        
        return {name: hash_obj.hexdigest() for name, hash_obj in hashes_obj.items()}
    
    def _sign_data(self, data: str) -> str:
        """Create digital signature for data"""
        if not self._private_key:
            raise RuntimeError("Private key not initialized")
            
        try:
            signature = self._private_key.sign(
                data.encode('utf-8'),
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                hashes.SHA256()
            )
            return signature.hex()
        except Exception as e:
            logger.error(f"Failed to sign data: {e}")
            raise
    
    def _verify_signature(self, data: str, signature: str) -> bool:
        """Verify digital signature"""
        if not self._public_key:
            return False
            
        try:
            self._public_key.verify(
                bytes.fromhex(signature),
                data.encode('utf-8'),
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                hashes.SHA256()
            )
            return True
        except InvalidSignature:
            return False
        except Exception as e:
            logger.error(f"Error verifying signature: {e}")
            return False
    
    def generate_evidence_hash(self, file_path: Path, metadata: EvidenceMetadata) -> EvidenceHash:
        """
        Generate cryptographic hash for evidence file
        
        Args:
            file_path: Path to evidence file
            metadata: Evidence metadata
            
        Returns:
            EvidenceHash object with calculated hashes
        """
        if not file_path.exists():
            raise FileNotFoundError(f"Evidence file not found: {file_path}")
        
        # Calculate file hashes
        file_hashes = self._calculate_file_hashes(file_path)
        
        # Generate unique evidence ID
        evidence_id = str(uuid.uuid4())
        timestamp = datetime.now(timezone.utc)
        
        # Create evidence hash object
        evidence_hash = EvidenceHash(
            evidence_id=evidence_id,
            sha256_hash=file_hashes['sha256'],
            sha512_hash=file_hashes['sha512'],
            md5_hash=file_hashes['md5'],
            timestamp=timestamp,
            metadata=metadata
        )
        
        # Create digital signature
        sign_data = f"{evidence_id}{file_hashes['sha256']}{timestamp.isoformat()}"
        evidence_hash.signature = self._sign_data(sign_data)
        
        # Store in database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO evidence (
                evidence_id, filename, file_size, mime_type,
                sha256_hash, sha512_hash, md5_hash, timestamp,
                metadata, signature, case_id, collector
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            evidence_id, metadata.filename, metadata.file_size, metadata.mime_type,
            file_hashes['sha256'], file_hashes['sha512'], file_hashes['md5'],
            timestamp.isoformat(), json.dumps(metadata.to_dict()),
            evidence_hash.signature, metadata.case_id, metadata.collector
        ))
        
        conn.commit()
        conn.close()
        
        # Create initial custody entry
        self.create_custody_entry(
            evidence_id=evidence_id,
            action=ActionType.CREATED,
            actor=metadata.collector,
            location=metadata.location,
            notes=f"Evidence created from {metadata.filename}"
        )
        
        logger.info(f"Evidence hash generated: {evidence_id}")
        return evidence_hash
    
    def create_custody_entry(self, evidence_id: str, action: ActionType, actor: str,
                           location: Optional[str] = None, notes: Optional[str] = None) -> CustodyEntry:
        """
        Create new entry in chain of custody
        
        Args:
            evidence_id: Evidence identifier
            action: Type of action performed
            actor: Person performing action
            location: Optional location of action
            notes: Optional additional notes
            
        Returns:
            CustodyEntry object
        """
        entry_id = str(uuid.uuid4())
        timestamp = datetime.now(timezone.utc)
        
        # Get previous custody entry hash for chaining
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT signature FROM custody_chain 
            WHERE evidence_id = ? 
            ORDER BY timestamp DESC LIMIT 1
        """, (evidence_id,))
        
        result = cursor.fetchone()
        previous_hash = result[0] if result else None
        
        # Create custody entry
        custody_entry = CustodyEntry(
            entry_id=entry_id,
            evidence_id=evidence_id,
            timestamp=timestamp,
            action=action,
            actor=actor,
            location=location,
            notes=notes,
            previous_hash=previous_hash
        )
        
        # Create digital signature
        sign_data = f"{entry_id}{evidence_id}{timestamp.isoformat()}{action.value}{actor}"
        custody_entry.signature = self._sign_data(sign_data)
        
        # Store in database
        cursor.execute("""
            INSERT INTO custody_chain (
                entry_id, evidence_id, timestamp, action, actor,
                location, notes, signature, previous_hash
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            entry_id, evidence_id, timestamp.isoformat(), action.value,
            actor, location, notes, custody_entry.signature, previous_hash
        ))
        
        conn.commit()
        conn.close()
        
        logger.info(f"Custody entry created: {entry_id}")
        return custody_entry
    
    def verify_integrity(self, evidence_id: str, current_file_path: Optional[Path] = None) -> IntegrityReport:
        """
        Verify integrity of evidence
        
        Args:
            evidence_id: Evidence identifier
            current_file_path: Optional current file path for hash verification
            
        Returns:
            IntegrityReport with verification results
        """
        timestamp = datetime.now(timezone.utc)
        issues = []
        recommendations = []
        hash_verified = True
        custody_verified = True
        
        # Get evidence record
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT sha256_hash, sha512_hash, md5_hash, signature, metadata
            FROM evidence WHERE evidence_id = ?
        """, (evidence_id,))
        
        evidence_record = cursor.fetchone()
        if not evidence_record:
            issues.append(f"Evidence record not found: {evidence_id}")
            hash_verified = False
            custody_verified = False
        else:
            stored_sha256, stored_sha512, stored_md5, signature, metadata_json = evidence_record
            
            # Verify hash if current file provided
            if current_file_path and current_file_path.exists():
                current_hashes = self._calculate_file_hashes(current_file_path)
                if current_hashes['sha256'] != stored_sha256:
                    hash_verified = False
                    issues.append("SHA-256 hash mismatch - file may have been modified")
                if current_hashes['sha512'] != stored_sha512:
                    hash_verified = False
                    issues.append("SHA-512 hash mismatch - file may have been modified")
                if current_hashes['md5'] != stored_md5:
                    hash_verified = False
                    issues.append("MD5 hash mismatch - file may have been modified")
            elif current_file_path:
                issues.append(f"Current file not found: {current_file_path}")
                hash_verified = False
            
            # Verify digital signature
            if signature:
                metadata_obj = json.loads(metadata_json)
                sign_data = f"{evidence_id}{stored_sha256}{metadata_obj.get('created_date', '')}"
                if not self._verify_signature(sign_data, signature):
                    issues.append("Digital signature verification failed")
                    custody_verified = False
        
        # Verify custody chain integrity
        cursor.execute("""
            SELECT entry_id, timestamp, signature, previous_hash
            FROM custody_chain WHERE evidence_id = ?
            ORDER BY timestamp ASC
        """, (evidence_id,))
        
        custody_entries = cursor.fetchall()
        if not custody_entries:
            issues.append("No custody chain entries found")
            custody_verified = False
        else:
            # Verify chain integrity
            previous_signature = None
            for entry_id, timestamp_str, signature, previous_hash in custody_entries:
                if previous_signature and previous_hash != previous_signature:
                    custody_verified = False
                    issues.append(f"Custody chain broken at entry {entry_id}")
                previous_signature = signature
        
        conn.close()
        
        # Determine overall status
        if hash_verified and custody_verified and not issues:
            status = IntegrityStatus.VERIFIED
        elif issues:
            status = IntegrityStatus.COMPROMISED
        else:
            status = IntegrityStatus.UNKNOWN
        
        # Generate recommendations
        if not hash_verified:
            recommendations.append("Re-obtain original evidence file for verification")
        if not custody_verified:
            recommendations.append("Review chain of custody procedures")
        if issues:
            recommendations.append("Investigate integrity issues before court submission")
        
        # Determine admissibility
        admissible = hash_verified and custody_verified and not issues
        
        # Create integrity report
        integrity_report = IntegrityReport(
            evidence_id=evidence_id,
            timestamp=timestamp,
            status=status,
            hash_verified=hash_verified,
            custody_verified=custody_verified,
            admissible=admissible,
            issues=issues,
            recommendations=recommendations
        )
        
        # Store integrity check in database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO integrity_checks (
                check_id, evidence_id, timestamp, status,
                hash_verified, custody_verified, issues
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            str(uuid.uuid4()), evidence_id, timestamp.isoformat(),
            status.value, hash_verified, custody_verified, json.dumps(issues)
        ))
        
        conn.commit()
        conn.close()
        
        logger.info(f"Integrity check completed: {evidence_id} - {status.value}")
        return integrity_report
    
    def get_custody_chain(self, evidence_id: str) -> List[CustodyEntry]:
        """
        Get complete chain of custody for evidence
        
        Args:
            evidence_id: Evidence identifier
            
        Returns:
            List of CustodyEntry objects in chronological order
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT entry_id, evidence_id, timestamp, action, actor,
                   location, notes, signature, previous_hash
            FROM custody_chain 
            WHERE evidence_id = ?
            ORDER BY timestamp ASC
        """, (evidence_id,))
        
        custody_entries = []
        for row in cursor.fetchall():
            entry_id, evidence_id, timestamp_str, action, actor, location, notes, signature, previous_hash = row
            
            custody_entry = CustodyEntry(
                entry_id=entry_id,
                evidence_id=evidence_id,
                timestamp=datetime.fromisoformat(timestamp_str),
                action=ActionType(action),
                actor=actor,
                location=location,
                notes=notes,
                signature=signature,
                previous_hash=previous_hash
            )
            custody_entries.append(custody_entry)
        
        conn.close()
        logger.info(f"Retrieved {len(custody_entries)} custody entries for {evidence_id}")
        return custody_entries
    
    def generate_court_manifest(self, case_id: str) -> CourtManifest:
        """
        Generate court manifest for case
        
        Args:
            case_id: Case identifier
            
        Returns:
            CourtManifest object
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get all evidence for case
        cursor.execute("""
            SELECT evidence_id, filename, file_size, sha256_hash, timestamp, metadata
            FROM evidence WHERE case_id = ?
        """, (case_id,))
        
        evidence_items = []
        total_size = 0
        integrity_summary = {'verified': 0, 'compromised': 0, 'unknown': 0}
        
        for row in cursor.fetchall():
            evidence_id, filename, file_size, sha256_hash, timestamp_str, metadata_json = row
            
            # Get latest integrity check
            cursor.execute("""
                SELECT status FROM integrity_checks 
                WHERE evidence_id = ? 
                ORDER BY timestamp DESC LIMIT 1
            """, (evidence_id,))
            
            status_result = cursor.fetchone()
            status = status_result[0] if status_result else 'unknown'
            
            evidence_item = {
                'evidence_id': evidence_id,
                'filename': filename,
                'file_size': file_size,
                'sha256_hash': sha256_hash,
                'timestamp': timestamp_str,
                'integrity_status': status,
                'metadata': json.loads(metadata_json)
            }
            
            evidence_items.append(evidence_item)
            total_size += file_size
            integrity_summary[status] = integrity_summary.get(status, 0) + 1
        
        conn.close()
        
        # Create court manifest
        manifest = CourtManifest(
            case_id=case_id,
            generated_date=datetime.now(timezone.utc),
            evidence_count=len(evidence_items),
            total_size=total_size,
            evidence_items=evidence_items,
            integrity_summary=integrity_summary
        )
        
        # Create digital signature for manifest
        sign_data = f"{case_id}{manifest.generated_date.isoformat()}{manifest.evidence_count}"
        manifest.signature = self._sign_data(sign_data)
        
        logger.info(f"Court manifest generated for case {case_id}: {len(evidence_items)} items")
        return manifest
    
    def export_evidence_package(self, case_id: str, output_dir: Path) -> Dict[str, Any]:
        """
        Export complete evidence package for case
        
        Args:
            case_id: Case identifier
            output_dir: Output directory path
            
        Returns:
            Export summary dictionary
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate court manifest
        manifest = self.generate_court_manifest(case_id)
        
        # Export manifest
        manifest_file = output_dir / f"manifest_{case_id}.json"
        with open(manifest_file, 'w') as f:
            json.dump(manifest.to_dict(), f, indent=2, default=str)
        
        # Export custody chains
        custody_dir = output_dir / "custody_chains"
        custody_dir.mkdir(exist_ok=True)
        
        # Export integrity reports
        reports_dir = output_dir / "integrity_reports"
        reports_dir.mkdir(exist_ok=True)
        
        files_created = [str(manifest_file)]
        
        for evidence_item in manifest.evidence_items:
            evidence_id = evidence_item['evidence_id']
            
            # Export custody chain
            custody_entries = self.get_custody_chain(evidence_id)
            custody_file = custody_dir / f"custody_{evidence_id}.json"
            with open(custody_file, 'w') as f:
                json.dump([entry.to_dict() for entry in custody_entries], f, indent=2, default=str)
            files_created.append(str(custody_file))
            
            # Export latest integrity report
            integrity_report = self.verify_integrity(evidence_id)
            report_file = reports_dir / f"integrity_{evidence_id}.json"
            with open(report_file, 'w') as f:
                json.dump(integrity_report.to_dict(), f, indent=2, default=str)
            files_created.append(str(report_file))
        
        export_summary = {
            'case_id': case_id,
            'evidence_count': manifest.evidence_count,
            'output_directory': str(output_dir),
            'files_created': files_created,
            'export_timestamp': datetime.now(timezone.utc).isoformat()
        }
        
        logger.info(f"Evidence package exported for case {case_id} to {output_dir}")
        return export_summary


def create_sample_evidence() -> None:
    """Create sample evidence for demonstration purposes"""
    # This function would create sample evidence files and demonstrate the toolkit
    # Implementation would go here for demo purposes
    print("Sample evidence creation functionality would be implemented here")