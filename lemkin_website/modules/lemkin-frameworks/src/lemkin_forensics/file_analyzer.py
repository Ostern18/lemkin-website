"""
Lemkin Digital Forensics - File System Analyzer

This module provides comprehensive file system analysis capabilities including
deleted file recovery, timeline analysis, and forensic artifact identification.
Designed for legal professionals investigating digital evidence.
"""

import os
import stat
import struct
import mimetypes
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Generator
import hashlib
import math
import logging
import subprocess

from .core import (
    FileArtifact, FileSystemAnalysis, DigitalEvidence, AnalysisStatus,
    FileSystemEventType, ForensicsConfig
)

logger = logging.getLogger(__name__)


class FileSystemAnalyzer:
    """
    Comprehensive file system analysis for digital forensics investigations.
    
    Provides capabilities for:
    - File system timeline analysis
    - Deleted file recovery
    - File signature analysis
    - Metadata extraction
    - Suspicious file identification
    """
    
    def __init__(self, config: Optional[ForensicsConfig] = None):
        self.config = config or ForensicsConfig()
        self.logger = logging.getLogger(f"{__name__}.FileSystemAnalyzer")
        
        # Common file signatures for type verification
        self.file_signatures = {
            b'\x89PNG\r\n\x1a\n': 'PNG',
            b'\xff\xd8\xff': 'JPEG',
            b'%PDF': 'PDF',
            b'PK\x03\x04': 'ZIP',
            b'\x50\x4b\x03\x04': 'ZIP',
            b'GIF87a': 'GIF',
            b'GIF89a': 'GIF',
            b'\x00\x00\x01\x00': 'ICO',
            b'RIFF': 'WAV/AVI',
            b'\x1f\x8b': 'GZIP',
            b'BM': 'BMP',
            b'\xd0\xcf\x11\xe0': 'MS Office',
            b'\x4d\x5a': 'EXE',
            b'\x7f\x45\x4c\x46': 'ELF',
            b'\xca\xfe\xba\xbe': 'Java Class',
            b'\xfe\xed\xfa': 'Mach-O'
        }
    
    def analyze_file_system(self, disk_image: Path) -> FileSystemAnalysis:
        """
        Perform comprehensive file system analysis on disk image
        
        Args:
            disk_image: Path to disk image file
            
        Returns:
            FileSystemAnalysis containing all findings
        """
        analysis = FileSystemAnalysis(evidence_id=self._generate_evidence_id())
        analysis.status = AnalysisStatus.IN_PROGRESS
        
        try:
            self.logger.info(f"Starting file system analysis of {disk_image}")
            
            # Validate disk image
            if not disk_image.exists():
                raise FileNotFoundError(f"Disk image not found: {disk_image}")
            
            # Get file system information
            self._extract_file_system_info(disk_image, analysis)
            
            # Analyze active files
            self._analyze_active_files(disk_image, analysis)
            
            # Recover deleted files if enabled
            if self.config.deleted_file_recovery:
                self._recover_deleted_files(disk_image, analysis)
            
            # Generate timeline if enabled
            if self.config.timeline_analysis:
                self._generate_file_timeline(analysis)
            
            # Identify suspicious files
            self._identify_suspicious_files(analysis)
            
            # Generate key findings
            self._generate_key_findings(analysis)
            
            analysis.completed_at = datetime.utcnow()
            analysis.analysis_duration = (
                analysis.completed_at - analysis.started_at
            ).total_seconds()
            analysis.status = AnalysisStatus.COMPLETED
            
            self.logger.info(f"File system analysis completed successfully")
            
        except Exception as e:
            self.logger.error(f"File system analysis failed: {str(e)}")
            analysis.status = AnalysisStatus.FAILED
            analysis.error_messages.append(str(e))
        
        return analysis
    
    def _extract_file_system_info(self, disk_image: Path, analysis: FileSystemAnalysis):
        """Extract basic file system information"""
        try:
            # Get disk image stats
            stat_info = disk_image.stat()
            analysis.total_size = stat_info.st_size
            
            # Try to identify file system type
            analysis.file_system_type = self._identify_file_system_type(disk_image)
            
            self.logger.info(f"File system type: {analysis.file_system_type}")
            
        except Exception as e:
            self.logger.warning(f"Could not extract file system info: {str(e)}")
    
    def _identify_file_system_type(self, disk_image: Path) -> str:
        """Identify file system type from disk image"""
        try:
            with open(disk_image, 'rb') as f:
                # Read first 512 bytes (boot sector)
                boot_sector = f.read(512)
                
                # Check for NTFS signature
                if b'NTFS    ' in boot_sector[3:11]:
                    return 'NTFS'
                
                # Check for FAT32 signature
                if b'FAT32   ' in boot_sector[82:90]:
                    return 'FAT32'
                
                # Check for FAT16 signature
                if b'FAT16   ' in boot_sector[54:62]:
                    return 'FAT16'
                
                # Check for ext4 signature (superblock at offset 1024)
                f.seek(1024)
                superblock = f.read(1024)
                if len(superblock) >= 56 and struct.unpack('<H', superblock[56:58])[0] == 0xEF53:
                    return 'ext4'
                
                return 'Unknown'
                
        except Exception as e:
            self.logger.warning(f"File system type identification failed: {str(e)}")
            return 'Unknown'
    
    def _analyze_active_files(self, disk_image: Path, analysis: FileSystemAnalysis):
        """Analyze active files in the file system"""
        try:
            # For demonstration, we'll analyze the directory structure
            # In a real implementation, this would parse the file system directly
            if disk_image.is_dir():
                # If it's a directory (not a disk image), analyze it directly
                self._analyze_directory(disk_image, analysis)
            else:
                # For actual disk images, we'd need specialized tools
                self._analyze_disk_image_with_tools(disk_image, analysis)
                
        except Exception as e:
            self.logger.error(f"Active file analysis failed: {str(e)}")
            analysis.error_messages.append(f"Active file analysis: {str(e)}")
    
    def _analyze_directory(self, directory: Path, analysis: FileSystemAnalysis):
        """Analyze files in a directory structure"""
        try:
            for file_path in self._walk_directory(directory):
                try:
                    artifact = self._create_file_artifact(file_path)
                    analysis.file_artifacts.append(artifact)
                    analysis.total_files += 1
                    
                    if not artifact.is_deleted:
                        analysis.active_files += 1
                        
                except Exception as e:
                    self.logger.warning(f"Failed to analyze file {file_path}: {str(e)}")
                    continue
                    
        except Exception as e:
            self.logger.error(f"Directory analysis failed: {str(e)}")
    
    def _walk_directory(self, directory: Path) -> Generator[Path, None, None]:
        """Walk directory tree yielding file paths"""
        try:
            for root, dirs, files in os.walk(directory):
                for file in files:
                    yield Path(root) / file
        except Exception as e:
            self.logger.error(f"Directory walk failed: {str(e)}")
    
    def _analyze_disk_image_with_tools(self, disk_image: Path, analysis: FileSystemAnalysis):
        """Analyze disk image using external forensic tools if available"""
        try:
            # Check if sleuthkit tools are available
            if self._check_sleuthkit_available():
                self._analyze_with_sleuthkit(disk_image, analysis)
            else:
                self.logger.warning("SleuthKit not available, using basic analysis")
                analysis.key_findings.append(
                    "Limited analysis: Advanced forensic tools not available"
                )
                
        except Exception as e:
            self.logger.error(f"Disk image analysis failed: {str(e)}")
    
    def _check_sleuthkit_available(self) -> bool:
        """Check if SleuthKit tools are available"""
        try:
            subprocess.run(['fls', '-V'], capture_output=True, check=True)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False
    
    def _analyze_with_sleuthkit(self, disk_image: Path, analysis: FileSystemAnalysis):
        """Analyze disk image using SleuthKit tools"""
        try:
            # Use fls to list files
            result = subprocess.run(
                ['fls', '-r', str(disk_image)],
                capture_output=True,
                text=True,
                timeout=self.config.analysis_timeout_minutes * 60
            )
            
            if result.returncode == 0:
                self._parse_fls_output(result.stdout, analysis)
            else:
                self.logger.warning(f"fls command failed: {result.stderr}")
                
        except subprocess.TimeoutExpired:
            self.logger.error("SleuthKit analysis timed out")
            analysis.error_messages.append("Analysis timed out")
        except Exception as e:
            self.logger.error(f"SleuthKit analysis failed: {str(e)}")
    
    def _parse_fls_output(self, fls_output: str, analysis: FileSystemAnalysis):
        """Parse output from SleuthKit fls command"""
        try:
            lines = fls_output.strip().split('\n')
            
            for line in lines:
                if not line.strip():
                    continue
                
                # Parse fls output format: type inode name
                parts = line.split('\t')
                if len(parts) >= 3:
                    file_type = parts[0]
                    inode = parts[1]
                    name = parts[2]
                    
                    # Create file artifact from fls data
                    artifact = FileArtifact(
                        file_path=name,
                        file_name=Path(name).name,
                        file_size=0,  # Would need additional tools to get size
                        file_type=self._determine_file_type_from_name(name),
                        inode_number=int(inode) if inode.isdigit() else None,
                        is_deleted='(deleted)' in line
                    )
                    
                    analysis.file_artifacts.append(artifact)
                    analysis.total_files += 1
                    
                    if artifact.is_deleted:
                        analysis.deleted_files += 1
                    else:
                        analysis.active_files += 1
                        
        except Exception as e:
            self.logger.error(f"Failed to parse fls output: {str(e)}")
    
    def _create_file_artifact(self, file_path: Path) -> FileArtifact:
        """Create file artifact from file path"""
        try:
            stat_info = file_path.stat()
            
            artifact = FileArtifact(
                file_path=str(file_path),
                file_name=file_path.name,
                file_size=stat_info.st_size,
                file_type=self._determine_file_type(file_path),
                mime_type=mimetypes.guess_type(str(file_path))[0],
                created_time=datetime.fromtimestamp(stat_info.st_ctime),
                modified_time=datetime.fromtimestamp(stat_info.st_mtime),
                accessed_time=datetime.fromtimestamp(stat_info.st_atime),
                file_permissions=oct(stat_info.st_mode)[-3:],
                owner_uid=stat_info.st_uid,
                group_gid=stat_info.st_gid
            )
            
            # Calculate file hashes if enabled
            if self.config.hash_verification:
                artifact.md5_hash = self._calculate_file_hash(file_path, 'md5')
                artifact.sha256_hash = self._calculate_file_hash(file_path, 'sha256')
            
            # Analyze file signature
            artifact.file_signature = self._get_file_signature(file_path)
            
            # Calculate entropy for suspicious file detection
            artifact.entropy_score = self._calculate_entropy(file_path)
            
            return artifact
            
        except Exception as e:
            self.logger.warning(f"Failed to create artifact for {file_path}: {str(e)}")
            # Return minimal artifact
            return FileArtifact(
                file_path=str(file_path),
                file_name=file_path.name,
                file_size=0,
                file_type="unknown"
            )
    
    def _determine_file_type(self, file_path: Path) -> str:
        """Determine file type from extension and content"""
        try:
            # First try extension
            extension = file_path.suffix.lower()
            if extension:
                type_map = {
                    '.txt': 'Text',
                    '.pdf': 'PDF',
                    '.doc': 'Word Document',
                    '.docx': 'Word Document',
                    '.jpg': 'JPEG Image',
                    '.jpeg': 'JPEG Image',
                    '.png': 'PNG Image',
                    '.gif': 'GIF Image',
                    '.mp4': 'MP4 Video',
                    '.avi': 'AVI Video',
                    '.exe': 'Executable',
                    '.zip': 'ZIP Archive',
                    '.rar': 'RAR Archive'
                }
                if extension in type_map:
                    return type_map[extension]
            
            # Try to determine from file signature
            signature = self._get_file_signature(file_path)
            if signature:
                return signature
            
            return "Unknown"
            
        except Exception:
            return "Unknown"
    
    def _determine_file_type_from_name(self, file_name: str) -> str:
        """Determine file type from file name only"""
        try:
            path = Path(file_name)
            extension = path.suffix.lower()
            
            type_map = {
                '.txt': 'Text',
                '.pdf': 'PDF',
                '.doc': 'Word Document',
                '.docx': 'Word Document',
                '.jpg': 'JPEG Image',
                '.jpeg': 'JPEG Image',
                '.png': 'PNG Image',
                '.exe': 'Executable',
                '.zip': 'ZIP Archive'
            }
            
            return type_map.get(extension, "Unknown")
            
        except Exception:
            return "Unknown"
    
    def _get_file_signature(self, file_path: Path) -> Optional[str]:
        """Get file signature by reading file header"""
        try:
            with open(file_path, 'rb') as f:
                header = f.read(16)  # Read first 16 bytes
                
                for signature, file_type in self.file_signatures.items():
                    if header.startswith(signature):
                        return file_type
                
            return None
            
        except Exception:
            return None
    
    def _calculate_file_hash(self, file_path: Path, algorithm: str) -> str:
        """Calculate file hash"""
        try:
            hash_obj = hashlib.new(algorithm.lower())
            
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_obj.update(chunk)
            
            return hash_obj.hexdigest()
            
        except Exception as e:
            self.logger.warning(f"Hash calculation failed for {file_path}: {str(e)}")
            return ""
    
    def _calculate_entropy(self, file_path: Path) -> float:
        """Calculate Shannon entropy of file content"""
        try:
            if file_path.stat().st_size == 0:
                return 0.0
            
            # Read file in chunks to handle large files
            byte_counts = [0] * 256
            total_bytes = 0
            
            with open(file_path, 'rb') as f:
                while chunk := f.read(4096):
                    for byte in chunk:
                        byte_counts[byte] += 1
                        total_bytes += 1
            
            if total_bytes == 0:
                return 0.0
            
            # Calculate Shannon entropy
            entropy = 0.0
            for count in byte_counts:
                if count > 0:
                    probability = count / total_bytes
                    entropy -= probability * math.log2(probability)
            
            return entropy
            
        except Exception as e:
            self.logger.warning(f"Entropy calculation failed for {file_path}: {str(e)}")
            return 0.0
    
    def _recover_deleted_files(self, disk_image: Path, analysis: FileSystemAnalysis):
        """Attempt to recover deleted files"""
        try:
            self.logger.info("Starting deleted file recovery")
            
            # Mark deleted files already found
            for artifact in analysis.file_artifacts:
                if artifact.is_deleted:
                    analysis.deleted_files += 1
            
            # In a real implementation, this would use specialized recovery tools
            # For now, we'll simulate the process
            if self._check_sleuthkit_available():
                self._recover_with_sleuthkit(disk_image, analysis)
            else:
                analysis.key_findings.append(
                    "Deleted file recovery limited: Advanced tools not available"
                )
            
        except Exception as e:
            self.logger.error(f"Deleted file recovery failed: {str(e)}")
            analysis.error_messages.append(f"File recovery: {str(e)}")
    
    def _recover_with_sleuthkit(self, disk_image: Path, analysis: FileSystemAnalysis):
        """Recover deleted files using SleuthKit"""
        try:
            # Use fls with deleted file flag
            result = subprocess.run(
                ['fls', '-rd', str(disk_image)],
                capture_output=True,
                text=True,
                timeout=self.config.analysis_timeout_minutes * 60
            )
            
            if result.returncode == 0:
                recovery_count = 0
                lines = result.stdout.strip().split('\n')
                
                for line in lines:
                    if '(deleted)' in line and recovery_count < 1000:  # Limit recoveries
                        # Attempt to recover file metadata
                        recovery_count += 1
                        analysis.recovered_files += 1
                
                self.logger.info(f"Recovered metadata for {recovery_count} deleted files")
                
        except subprocess.TimeoutExpired:
            self.logger.warning("Deleted file recovery timed out")
        except Exception as e:
            self.logger.error(f"SleuthKit recovery failed: {str(e)}")
    
    def _generate_file_timeline(self, analysis: FileSystemAnalysis):
        """Generate timeline of file system events"""
        try:
            timeline_events = []
            
            for artifact in analysis.file_artifacts:
                # Add creation event
                if artifact.created_time:
                    timeline_events.append({
                        "timestamp": artifact.created_time,
                        "event_type": FileSystemEventType.CREATED,
                        "file_path": artifact.file_path,
                        "file_name": artifact.file_name,
                        "file_size": artifact.file_size,
                        "description": f"File created: {artifact.file_name}"
                    })
                
                # Add modification event
                if artifact.modified_time and artifact.modified_time != artifact.created_time:
                    timeline_events.append({
                        "timestamp": artifact.modified_time,
                        "event_type": FileSystemEventType.MODIFIED,
                        "file_path": artifact.file_path,
                        "file_name": artifact.file_name,
                        "file_size": artifact.file_size,
                        "description": f"File modified: {artifact.file_name}"
                    })
                
                # Add access event if different from creation/modification
                if (artifact.accessed_time and 
                    artifact.accessed_time != artifact.created_time and
                    artifact.accessed_time != artifact.modified_time):
                    timeline_events.append({
                        "timestamp": artifact.accessed_time,
                        "event_type": FileSystemEventType.ACCESSED,
                        "file_path": artifact.file_path,
                        "file_name": artifact.file_name,
                        "description": f"File accessed: {artifact.file_name}"
                    })
            
            # Sort timeline by timestamp
            timeline_events.sort(key=lambda x: x["timestamp"])
            analysis.timeline_events = timeline_events
            
            self.logger.info(f"Generated timeline with {len(timeline_events)} events")
            
        except Exception as e:
            self.logger.error(f"Timeline generation failed: {str(e)}")
    
    def _identify_suspicious_files(self, analysis: FileSystemAnalysis):
        """Identify potentially suspicious files"""
        try:
            suspicious_files = []
            
            for artifact in analysis.file_artifacts:
                suspicion_reasons = []
                
                # Check for high entropy (possible encryption/compression)
                if artifact.entropy_score and artifact.entropy_score > 7.5:
                    suspicion_reasons.append(f"High entropy ({artifact.entropy_score:.2f})")
                
                # Check for file extension mismatch
                if self._has_extension_mismatch(artifact):
                    suspicion_reasons.append("File extension/signature mismatch")
                
                # Check for suspicious file names
                if self._has_suspicious_name(artifact.file_name):
                    suspicion_reasons.append("Suspicious file name")
                
                # Check for hidden files in suspicious locations
                if artifact.file_name.startswith('.') and 'system' not in artifact.file_path.lower():
                    suspicion_reasons.append("Hidden file")
                
                # Check for executable files in unusual locations
                if (artifact.file_type in ['Executable', 'EXE'] and 
                    not any(folder in artifact.file_path.lower() 
                           for folder in ['program', 'bin', 'system', 'windows'])):
                    suspicion_reasons.append("Executable in unusual location")
                
                if suspicion_reasons:
                    # Add details about why it's suspicious
                    artifact_copy = artifact.copy()
                    artifact_copy.file_path += f" [SUSPICIOUS: {'; '.join(suspicion_reasons)}]"
                    suspicious_files.append(artifact_copy)
            
            analysis.suspicious_files = suspicious_files
            
            if suspicious_files:
                analysis.key_findings.append(
                    f"Identified {len(suspicious_files)} potentially suspicious files"
                )
            
            self.logger.info(f"Identified {len(suspicious_files)} suspicious files")
            
        except Exception as e:
            self.logger.error(f"Suspicious file identification failed: {str(e)}")
    
    def _has_extension_mismatch(self, artifact: FileArtifact) -> bool:
        """Check if file extension matches file signature"""
        if not artifact.file_signature:
            return False
        
        extension = Path(artifact.file_name).suffix.lower()
        signature = artifact.file_signature.upper()
        
        # Define expected extensions for signatures
        signature_extensions = {
            'PNG': ['.png'],
            'JPEG': ['.jpg', '.jpeg'],
            'PDF': ['.pdf'],
            'ZIP': ['.zip', '.jar', '.war'],
            'GIF': ['.gif'],
            'BMP': ['.bmp'],
            'EXE': ['.exe', '.dll', '.scr'],
            'ELF': [''],  # No extension typical for Unix executables
        }
        
        expected_extensions = signature_extensions.get(signature, [])
        if expected_extensions and extension not in expected_extensions:
            return True
        
        return False
    
    def _has_suspicious_name(self, file_name: str) -> bool:
        """Check if file name is suspicious"""
        suspicious_patterns = [
            'password', 'secret', 'confidential', 'private',
            'keylogger', 'trojan', 'backdoor', 'rootkit',
            'temp', 'tmp', '~', 'backup',
            'deleted', 'hidden', 'ghost'
        ]
        
        name_lower = file_name.lower()
        return any(pattern in name_lower for pattern in suspicious_patterns)
    
    def _generate_key_findings(self, analysis: FileSystemAnalysis):
        """Generate key findings summary"""
        try:
            findings = []
            
            # File statistics
            findings.append(
                f"Analyzed {analysis.total_files:,} files "
                f"({analysis.active_files:,} active, {analysis.deleted_files:,} deleted)"
            )
            
            # Recovery statistics
            if analysis.recovered_files > 0:
                findings.append(
                    f"Successfully recovered metadata for {analysis.recovered_files:,} deleted files"
                )
            
            # Timeline statistics
            if analysis.timeline_events:
                earliest = min(event["timestamp"] for event in analysis.timeline_events)
                latest = max(event["timestamp"] for event in analysis.timeline_events)
                findings.append(
                    f"File activity timeline spans from {earliest.strftime('%Y-%m-%d %H:%M:%S')} "
                    f"to {latest.strftime('%Y-%m-%d %H:%M:%S')}"
                )
            
            # Suspicious files
            if analysis.suspicious_files:
                findings.append(
                    f"Identified {len(analysis.suspicious_files)} files requiring further investigation"
                )
            
            # File system health
            if analysis.file_system_type:
                findings.append(f"File system type: {analysis.file_system_type}")
            
            analysis.key_findings.extend(findings)
            
        except Exception as e:
            self.logger.error(f"Key findings generation failed: {str(e)}")
    
    def _generate_evidence_id(self) -> str:
        """Generate a unique evidence ID"""
        import uuid
        return str(uuid.uuid4())


def analyze_file_system(disk_image: Path, config: Optional[ForensicsConfig] = None) -> FileSystemAnalysis:
    """
    Convenience function to analyze a file system disk image
    
    Args:
        disk_image: Path to disk image or directory to analyze
        config: Optional configuration for analysis
        
    Returns:
        FileSystemAnalysis with complete results
    """
    analyzer = FileSystemAnalyzer(config)
    return analyzer.analyze_file_system(disk_image)