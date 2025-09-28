"""
Core digital forensics functionality for legal investigations.
"""

import hashlib
import json
import os
import re
import sqlite3
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Set
from enum import Enum

import magic
from pydantic import BaseModel, Field, field_validator
from loguru import logger
import pyzipper


class FileType(str, Enum):
    """Common file types for forensic analysis"""
    DOCUMENT = "document"
    IMAGE = "image"
    VIDEO = "video"
    AUDIO = "audio"
    EXECUTABLE = "executable"
    ARCHIVE = "archive"
    DATABASE = "database"
    LOG = "log"
    UNKNOWN = "unknown"


class AuthenticityStatus(str, Enum):
    """Digital evidence authenticity status"""
    AUTHENTIC = "authentic"
    MODIFIED = "modified"
    SUSPICIOUS = "suspicious"
    CORRUPTED = "corrupted"
    UNKNOWN = "unknown"


class FileMetadata(BaseModel):
    """File metadata for forensic analysis"""
    file_path: Path
    file_name: str
    file_size: int
    mime_type: str
    file_type: FileType
    created_date: Optional[datetime] = None
    modified_date: Optional[datetime] = None
    accessed_date: Optional[datetime] = None
    md5_hash: str
    sha256_hash: str
    file_permissions: Optional[str] = None
    is_hidden: bool = False
    is_system: bool = False
    is_encrypted: bool = False


class NetworkLogEntry(BaseModel):
    """Network log entry for analysis"""
    timestamp: datetime
    source_ip: str
    destination_ip: str
    source_port: Optional[int] = None
    destination_port: Optional[int] = None
    protocol: str
    payload_size: int
    flags: Optional[str] = None
    packet_data: Optional[str] = None


class DigitalEvidence(BaseModel):
    """Digital evidence container"""
    evidence_id: str
    evidence_type: str
    file_path: Optional[Path] = None
    data_content: Optional[str] = None
    collection_date: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    chain_of_custody: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    integrity_hash: str = ""

    def model_post_init(self, __context: Any) -> None:
        """Calculate integrity hash after initialization"""
        if self.file_path and self.file_path.exists():
            self.integrity_hash = self._calculate_file_hash(self.file_path)
        elif self.data_content:
            self.integrity_hash = hashlib.sha256(
                self.data_content.encode('utf-8')
            ).hexdigest()

    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA-256 hash of file"""
        sha256 = hashlib.sha256()
        with open(file_path, 'rb') as f:
            while chunk := f.read(8192):
                sha256.update(chunk)
        return sha256.hexdigest()


class FileSystemAnalysis(BaseModel):
    """File system analysis results"""
    analysis_id: str
    source_path: Path
    total_files: int
    total_size: int
    file_types: Dict[str, int]
    deleted_files: List[FileMetadata] = Field(default_factory=list)
    suspicious_files: List[FileMetadata] = Field(default_factory=list)
    hidden_files: List[FileMetadata] = Field(default_factory=list)
    recent_files: List[FileMetadata] = Field(default_factory=list)
    analysis_date: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class NetworkAnalysis(BaseModel):
    """Network log analysis results"""
    analysis_id: str
    log_file: Path
    total_entries: int
    date_range: Dict[str, datetime]
    top_sources: List[Dict[str, Any]] = Field(default_factory=list)
    top_destinations: List[Dict[str, Any]] = Field(default_factory=list)
    suspicious_activities: List[Dict[str, Any]] = Field(default_factory=list)
    communication_patterns: List[Dict[str, Any]] = Field(default_factory=list)
    analysis_date: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class MobileDataExtraction(BaseModel):
    """Mobile device data extraction results"""
    extraction_id: str
    device_info: Dict[str, Any]
    contacts: List[Dict[str, Any]] = Field(default_factory=list)
    messages: List[Dict[str, Any]] = Field(default_factory=list)
    call_logs: List[Dict[str, Any]] = Field(default_factory=list)
    app_data: List[Dict[str, Any]] = Field(default_factory=list)
    location_data: List[Dict[str, Any]] = Field(default_factory=list)
    extraction_date: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class AuthenticityReport(BaseModel):
    """Digital evidence authenticity report"""
    evidence_id: str
    status: AuthenticityStatus
    confidence_score: float = Field(ge=0.0, le=1.0)
    integrity_verified: bool
    timestamp_verified: bool
    metadata_consistent: bool
    findings: List[str] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)
    analysis_date: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class FileAnalyzer:
    """Analyze file systems and recover deleted files"""

    def __init__(self):
        """Initialize file analyzer"""
        self.magic = magic.Magic(mime=True)
        logger.info("Initialized file analyzer")

    def analyze_file_system(self, source_path: Path) -> FileSystemAnalysis:
        """
        Analyze file system for forensic evidence.

        Args:
            source_path: Path to analyze (directory or disk image)

        Returns:
            FileSystemAnalysis with findings
        """
        import uuid

        analysis_id = str(uuid.uuid4())

        if not source_path.exists():
            raise FileNotFoundError(f"Path not found: {source_path}")

        logger.info(f"Starting file system analysis of {source_path}")

        # Initialize counters
        total_files = 0
        total_size = 0
        file_types = {}
        deleted_files = []
        suspicious_files = []
        hidden_files = []
        recent_files = []

        # Analyze directory recursively
        if source_path.is_dir():
            for root, dirs, files in os.walk(source_path):
                root_path = Path(root)

                for file_name in files:
                    file_path = root_path / file_name
                    try:
                        metadata = self._analyze_file(file_path)
                        total_files += 1
                        total_size += metadata.file_size

                        # Count file types
                        file_type = metadata.file_type.value
                        file_types[file_type] = file_types.get(file_type, 0) + 1

                        # Categorize files
                        if metadata.is_hidden:
                            hidden_files.append(metadata)

                        if self._is_suspicious_file(metadata):
                            suspicious_files.append(metadata)

                        if self._is_recent_file(metadata):
                            recent_files.append(metadata)

                    except Exception as e:
                        logger.warning(f"Could not analyze {file_path}: {e}")
                        continue

        # Look for deleted files (simplified simulation)
        deleted_files = self._find_deleted_files(source_path)

        return FileSystemAnalysis(
            analysis_id=analysis_id,
            source_path=source_path,
            total_files=total_files,
            total_size=total_size,
            file_types=file_types,
            deleted_files=deleted_files,
            suspicious_files=suspicious_files[:20],  # Limit to top 20
            hidden_files=hidden_files[:20],
            recent_files=recent_files[:20]
        )

    def _analyze_file(self, file_path: Path) -> FileMetadata:
        """Analyze individual file metadata"""
        stat = file_path.stat()

        # Calculate hashes
        md5_hash = self._calculate_md5(file_path)
        sha256_hash = self._calculate_sha256(file_path)

        # Detect MIME type
        try:
            mime_type = self.magic.from_file(str(file_path))
        except:
            mime_type = "application/octet-stream"

        # Determine file type
        file_type = self._classify_file_type(mime_type, file_path.suffix)

        # Check timestamps
        created_date = datetime.fromtimestamp(stat.st_ctime, tz=timezone.utc)
        modified_date = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc)
        accessed_date = datetime.fromtimestamp(stat.st_atime, tz=timezone.utc)

        # Check file attributes
        is_hidden = file_path.name.startswith('.')
        is_system = False  # Platform-specific logic would go here
        is_encrypted = self._check_if_encrypted(file_path, mime_type)

        return FileMetadata(
            file_path=file_path,
            file_name=file_path.name,
            file_size=stat.st_size,
            mime_type=mime_type,
            file_type=file_type,
            created_date=created_date,
            modified_date=modified_date,
            accessed_date=accessed_date,
            md5_hash=md5_hash,
            sha256_hash=sha256_hash,
            file_permissions=oct(stat.st_mode),
            is_hidden=is_hidden,
            is_system=is_system,
            is_encrypted=is_encrypted
        )

    def _calculate_md5(self, file_path: Path) -> str:
        """Calculate MD5 hash"""
        md5 = hashlib.md5()
        with open(file_path, 'rb') as f:
            while chunk := f.read(8192):
                md5.update(chunk)
        return md5.hexdigest()

    def _calculate_sha256(self, file_path: Path) -> str:
        """Calculate SHA-256 hash"""
        sha256 = hashlib.sha256()
        with open(file_path, 'rb') as f:
            while chunk := f.read(8192):
                sha256.update(chunk)
        return sha256.hexdigest()

    def _classify_file_type(self, mime_type: str, extension: str) -> FileType:
        """Classify file type based on MIME type and extension"""
        if mime_type.startswith('image/'):
            return FileType.IMAGE
        elif mime_type.startswith('video/'):
            return FileType.VIDEO
        elif mime_type.startswith('audio/'):
            return FileType.AUDIO
        elif 'application/pdf' in mime_type or 'document' in mime_type:
            return FileType.DOCUMENT
        elif 'application/zip' in mime_type or extension.lower() in ['.zip', '.rar', '.7z']:
            return FileType.ARCHIVE
        elif 'application/x-executable' in mime_type or extension.lower() in ['.exe', '.dll']:
            return FileType.EXECUTABLE
        elif 'database' in mime_type or extension.lower() in ['.db', '.sqlite']:
            return FileType.DATABASE
        elif extension.lower() in ['.log', '.txt']:
            return FileType.LOG
        else:
            return FileType.UNKNOWN

    def _check_if_encrypted(self, file_path: Path, mime_type: str) -> bool:
        """Check if file is encrypted (simplified detection)"""
        try:
            # Check for common encrypted file signatures
            with open(file_path, 'rb') as f:
                header = f.read(16)

            # Check for encrypted ZIP
            if header.startswith(b'PK'):
                try:
                    with pyzipper.AESZipFile(file_path, 'r') as zip_file:
                        # If we can't list contents without password, it's encrypted
                        zip_file.namelist()
                        return False
                except (pyzipper.BadZipFile, RuntimeError):
                    return True

            # Check for other encryption indicators
            encrypted_signatures = [
                b'\x00\x00\x00\x00',  # Some encrypted files start with null bytes
            ]

            return any(header.startswith(sig) for sig in encrypted_signatures)

        except:
            return False

    def _is_suspicious_file(self, metadata: FileMetadata) -> bool:
        """Check if file is suspicious"""
        suspicious_indicators = [
            metadata.is_hidden and metadata.file_type == FileType.EXECUTABLE,
            metadata.file_size == 0 and metadata.file_type != FileType.UNKNOWN,
            metadata.file_name.endswith('.tmp') and metadata.file_size > 1024*1024,  # Large temp files
            'suspicious' in metadata.file_name.lower(),
        ]

        return any(suspicious_indicators)

    def _is_recent_file(self, metadata: FileMetadata, days: int = 7) -> bool:
        """Check if file was recently modified"""
        if not metadata.modified_date:
            return False

        cutoff = datetime.now(timezone.utc).replace(day=datetime.now().day - days)
        return metadata.modified_date > cutoff

    def _find_deleted_files(self, source_path: Path) -> List[FileMetadata]:
        """Find deleted files (simplified simulation)"""
        # In a real implementation, this would use forensic tools
        # to scan unallocated disk space
        deleted_files = []

        # Simulate finding some deleted files
        deleted_files.append(FileMetadata(
            file_path=source_path / "deleted_document.doc",
            file_name="deleted_document.doc",
            file_size=1024,
            mime_type="application/msword",
            file_type=FileType.DOCUMENT,
            md5_hash="simulated_md5_hash",
            sha256_hash="simulated_sha256_hash",
            is_hidden=False,
            is_system=False,
            is_encrypted=False
        ))

        return deleted_files


class NetworkProcessor:
    """Process and analyze network logs"""

    def __init__(self):
        """Initialize network processor"""
        logger.info("Initialized network processor")

    def process_network_logs(self, log_files: List[Path]) -> NetworkAnalysis:
        """
        Process network log files for forensic analysis.

        Args:
            log_files: List of log file paths

        Returns:
            NetworkAnalysis with findings
        """
        import uuid

        analysis_id = str(uuid.uuid4())

        all_entries = []
        total_entries = 0

        # Process each log file
        for log_file in log_files:
            if not log_file.exists():
                logger.warning(f"Log file not found: {log_file}")
                continue

            logger.info(f"Processing network log: {log_file}")
            entries = self._parse_log_file(log_file)
            all_entries.extend(entries)
            total_entries += len(entries)

        if not all_entries:
            raise ValueError("No valid log entries found")

        # Calculate date range
        timestamps = [entry.timestamp for entry in all_entries]
        date_range = {
            "start": min(timestamps),
            "end": max(timestamps)
        }

        # Analyze traffic patterns
        top_sources = self._get_top_sources(all_entries)
        top_destinations = self._get_top_destinations(all_entries)
        suspicious_activities = self._find_suspicious_activities(all_entries)
        communication_patterns = self._analyze_communication_patterns(all_entries)

        return NetworkAnalysis(
            analysis_id=analysis_id,
            log_file=log_files[0] if log_files else Path("unknown"),
            total_entries=total_entries,
            date_range=date_range,
            top_sources=top_sources[:10],
            top_destinations=top_destinations[:10],
            suspicious_activities=suspicious_activities[:20],
            communication_patterns=communication_patterns[:10]
        )

    def _parse_log_file(self, log_file: Path) -> List[NetworkLogEntry]:
        """Parse network log file"""
        entries = []

        try:
            with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                for line_num, line in enumerate(f):
                    try:
                        entry = self._parse_log_line(line.strip())
                        if entry:
                            entries.append(entry)
                    except Exception as e:
                        if line_num < 10:  # Only log first few parsing errors
                            logger.debug(f"Could not parse line {line_num}: {e}")
                        continue

        except Exception as e:
            logger.error(f"Error reading log file {log_file}: {e}")

        return entries

    def _parse_log_line(self, line: str) -> Optional[NetworkLogEntry]:
        """Parse individual log line (supports common formats)"""
        if not line or line.startswith('#'):
            return None

        # Try to parse common log formats
        # Example: "2024-01-15 10:30:00 192.168.1.1:80 -> 10.0.0.1:443 TCP 1024 [SYN]"

        # Simple regex for basic parsing
        pattern = r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\s+(\d+\.\d+\.\d+\.\d+):?(\d+)?\s+\S*\s+(\d+\.\d+\.\d+\.\d+):?(\d+)?\s+(\w+)\s+(\d+)'
        match = re.search(pattern, line)

        if match:
            timestamp = datetime.strptime(match.group(1), "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
            source_ip = match.group(2)
            source_port = int(match.group(3)) if match.group(3) else None
            dest_ip = match.group(4)
            dest_port = int(match.group(5)) if match.group(5) else None
            protocol = match.group(6)
            payload_size = int(match.group(7))

            return NetworkLogEntry(
                timestamp=timestamp,
                source_ip=source_ip,
                destination_ip=dest_ip,
                source_port=source_port,
                destination_port=dest_port,
                protocol=protocol,
                payload_size=payload_size
            )

        return None

    def _get_top_sources(self, entries: List[NetworkLogEntry]) -> List[Dict[str, Any]]:
        """Get top source IPs by traffic volume"""
        from collections import defaultdict

        source_stats = defaultdict(lambda: {"count": 0, "total_bytes": 0})

        for entry in entries:
            source_stats[entry.source_ip]["count"] += 1
            source_stats[entry.source_ip]["total_bytes"] += entry.payload_size

        # Sort by total bytes
        sorted_sources = sorted(
            source_stats.items(),
            key=lambda x: x[1]["total_bytes"],
            reverse=True
        )

        return [
            {
                "ip": ip,
                "connections": stats["count"],
                "total_bytes": stats["total_bytes"]
            }
            for ip, stats in sorted_sources
        ]

    def _get_top_destinations(self, entries: List[NetworkLogEntry]) -> List[Dict[str, Any]]:
        """Get top destination IPs by traffic volume"""
        from collections import defaultdict

        dest_stats = defaultdict(lambda: {"count": 0, "total_bytes": 0})

        for entry in entries:
            dest_stats[entry.destination_ip]["count"] += 1
            dest_stats[entry.destination_ip]["total_bytes"] += entry.payload_size

        # Sort by total bytes
        sorted_dests = sorted(
            dest_stats.items(),
            key=lambda x: x[1]["total_bytes"],
            reverse=True
        )

        return [
            {
                "ip": ip,
                "connections": stats["count"],
                "total_bytes": stats["total_bytes"]
            }
            for ip, stats in sorted_dests
        ]

    def _find_suspicious_activities(self, entries: List[NetworkLogEntry]) -> List[Dict[str, Any]]:
        """Identify suspicious network activities"""
        suspicious = []

        # Group by source IP
        from collections import defaultdict
        ip_activities = defaultdict(list)

        for entry in entries:
            ip_activities[entry.source_ip].append(entry)

        # Check for suspicious patterns
        for ip, ip_entries in ip_activities.items():
            # Port scanning detection
            unique_ports = set()
            for entry in ip_entries:
                if entry.destination_port:
                    unique_ports.add(entry.destination_port)

            if len(unique_ports) > 50:  # Threshold for port scanning
                suspicious.append({
                    "type": "port_scanning",
                    "source_ip": ip,
                    "unique_ports": len(unique_ports),
                    "description": f"Possible port scanning from {ip}"
                })

            # High volume traffic
            total_bytes = sum(entry.payload_size for entry in ip_entries)
            if total_bytes > 100_000_000:  # 100MB threshold
                suspicious.append({
                    "type": "high_volume_traffic",
                    "source_ip": ip,
                    "total_bytes": total_bytes,
                    "description": f"High volume traffic from {ip}"
                })

        return suspicious

    def _analyze_communication_patterns(self, entries: List[NetworkLogEntry]) -> List[Dict[str, Any]]:
        """Analyze communication patterns"""
        patterns = []

        # Group by hour of day
        from collections import defaultdict
        hourly_traffic = defaultdict(int)

        for entry in entries:
            hour = entry.timestamp.hour
            hourly_traffic[hour] += entry.payload_size

        # Find peak hours
        peak_hour = max(hourly_traffic.items(), key=lambda x: x[1])
        patterns.append({
            "type": "peak_traffic_hour",
            "hour": peak_hour[0],
            "bytes": peak_hour[1],
            "description": f"Peak traffic at hour {peak_hour[0]}"
        })

        return patterns


class MobileAnalyzer:
    """Analyze mobile device data extractions"""

    def __init__(self):
        """Initialize mobile analyzer"""
        logger.info("Initialized mobile analyzer")

    def extract_mobile_data(self, backup_path: Path) -> MobileDataExtraction:
        """
        Extract data from mobile device backup.

        Args:
            backup_path: Path to mobile backup

        Returns:
            MobileDataExtraction with findings
        """
        import uuid

        extraction_id = str(uuid.uuid4())

        if not backup_path.exists():
            raise FileNotFoundError(f"Backup path not found: {backup_path}")

        logger.info(f"Extracting mobile data from {backup_path}")

        # Device info (would be extracted from backup metadata)
        device_info = {
            "device_type": "smartphone",
            "os_version": "unknown",
            "model": "unknown",
            "backup_date": datetime.now(timezone.utc).isoformat()
        }

        # Extract different data types
        contacts = self._extract_contacts(backup_path)
        messages = self._extract_messages(backup_path)
        call_logs = self._extract_call_logs(backup_path)
        app_data = self._extract_app_data(backup_path)
        location_data = self._extract_location_data(backup_path)

        return MobileDataExtraction(
            extraction_id=extraction_id,
            device_info=device_info,
            contacts=contacts,
            messages=messages,
            call_logs=call_logs,
            app_data=app_data,
            location_data=location_data
        )

    def _extract_contacts(self, backup_path: Path) -> List[Dict[str, Any]]:
        """Extract contacts from backup"""
        contacts = []

        # Look for contacts database
        contacts_db = backup_path / "contacts.db"
        if contacts_db.exists():
            try:
                conn = sqlite3.connect(contacts_db)
                cursor = conn.cursor()

                # Try common contacts table structures
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
                tables = cursor.fetchall()

                for table in tables:
                    if 'contact' in table[0].lower():
                        # Simplified contact extraction
                        contacts.append({
                            "name": "Sample Contact",
                            "phone": "+1234567890",
                            "email": "contact@example.com"
                        })
                        break

                conn.close()

            except Exception as e:
                logger.warning(f"Could not extract contacts: {e}")

        return contacts

    def _extract_messages(self, backup_path: Path) -> List[Dict[str, Any]]:
        """Extract messages from backup"""
        messages = []

        # Look for messages database
        messages_db = backup_path / "messages.db"
        if messages_db.exists():
            # Simplified message extraction
            messages.append({
                "sender": "+1234567890",
                "recipient": "+0987654321",
                "content": "Sample message content",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "type": "SMS"
            })

        return messages

    def _extract_call_logs(self, backup_path: Path) -> List[Dict[str, Any]]:
        """Extract call logs from backup"""
        call_logs = []

        # Look for call log database
        calls_db = backup_path / "calls.db"
        if calls_db.exists():
            # Simplified call log extraction
            call_logs.append({
                "number": "+1234567890",
                "type": "outgoing",
                "duration": 120,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })

        return call_logs

    def _extract_app_data(self, backup_path: Path) -> List[Dict[str, Any]]:
        """Extract application data from backup"""
        app_data = []

        # Look for app data directories
        apps_dir = backup_path / "apps"
        if apps_dir.exists():
            for app_dir in apps_dir.iterdir():
                if app_dir.is_dir():
                    app_data.append({
                        "app_name": app_dir.name,
                        "package_id": f"com.example.{app_dir.name}",
                        "data_size": sum(f.stat().st_size for f in app_dir.rglob('*') if f.is_file()),
                        "last_used": datetime.now(timezone.utc).isoformat()
                    })

        return app_data

    def _extract_location_data(self, backup_path: Path) -> List[Dict[str, Any]]:
        """Extract location data from backup"""
        location_data = []

        # Look for location database
        location_db = backup_path / "location.db"
        if location_db.exists():
            # Simplified location extraction
            location_data.append({
                "latitude": 40.7128,
                "longitude": -74.0060,
                "accuracy": 10.0,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "source": "GPS"
            })

        return location_data


class AuthenticityVerifier:
    """Verify digital evidence authenticity"""

    def __init__(self):
        """Initialize authenticity verifier"""
        logger.info("Initialized authenticity verifier")

    def verify_digital_authenticity(self, evidence: DigitalEvidence) -> AuthenticityReport:
        """
        Verify authenticity of digital evidence.

        Args:
            evidence: Digital evidence to verify

        Returns:
            AuthenticityReport with findings
        """
        findings = []
        recommendations = []

        # Check integrity hash
        integrity_verified = self._verify_integrity(evidence)
        if not integrity_verified:
            findings.append("File integrity hash mismatch detected")
            recommendations.append("Re-collect evidence from original source")

        # Check timestamp consistency
        timestamp_verified = self._verify_timestamps(evidence)
        if not timestamp_verified:
            findings.append("Timestamp inconsistencies detected")
            recommendations.append("Verify system clock accuracy during collection")

        # Check metadata consistency
        metadata_consistent = self._verify_metadata(evidence)
        if not metadata_consistent:
            findings.append("Metadata inconsistencies found")
            recommendations.append("Cross-reference with additional metadata sources")

        # Calculate overall confidence score
        checks_passed = sum([
            integrity_verified,
            timestamp_verified,
            metadata_consistent
        ])
        confidence_score = checks_passed / 3.0

        # Determine overall status
        if confidence_score >= 0.8:
            status = AuthenticityStatus.AUTHENTIC
        elif confidence_score >= 0.6:
            status = AuthenticityStatus.SUSPICIOUS
        else:
            status = AuthenticityStatus.MODIFIED

        return AuthenticityReport(
            evidence_id=evidence.evidence_id,
            status=status,
            confidence_score=confidence_score,
            integrity_verified=integrity_verified,
            timestamp_verified=timestamp_verified,
            metadata_consistent=metadata_consistent,
            findings=findings,
            recommendations=recommendations
        )

    def _verify_integrity(self, evidence: DigitalEvidence) -> bool:
        """Verify file integrity"""
        if evidence.file_path and evidence.file_path.exists():
            current_hash = hashlib.sha256()
            with open(evidence.file_path, 'rb') as f:
                while chunk := f.read(8192):
                    current_hash.update(chunk)

            return current_hash.hexdigest() == evidence.integrity_hash
        elif evidence.data_content:
            current_hash = hashlib.sha256(evidence.data_content.encode('utf-8')).hexdigest()
            return current_hash == evidence.integrity_hash

        return False

    def _verify_timestamps(self, evidence: DigitalEvidence) -> bool:
        """Verify timestamp consistency"""
        if evidence.file_path and evidence.file_path.exists():
            stat = evidence.file_path.stat()
            file_mtime = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc)

            # Check if collection date is reasonable compared to file modification
            time_diff = abs((evidence.collection_date - file_mtime).total_seconds())

            # Allow reasonable time difference (e.g., 1 day)
            return time_diff < 86400  # 24 hours

        return True  # Cannot verify without file

    def _verify_metadata(self, evidence: DigitalEvidence) -> bool:
        """Verify metadata consistency"""
        # Check for required metadata fields
        required_fields = ['evidence_type', 'collection_date']

        for field in required_fields:
            if not hasattr(evidence, field) or getattr(evidence, field) is None:
                return False

        # Check metadata format consistency
        if evidence.metadata:
            for key, value in evidence.metadata.items():
                if not isinstance(key, str):
                    return False

        return True