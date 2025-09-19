"""
Lemkin Digital Forensics File System Analyzer

This module provides comprehensive file system analysis capabilities including:
- Disk image analysis using The Sleuth Kit (TSK)
- File system timeline generation
- Deleted file recovery and carving
- File metadata extraction and analysis
- File signature analysis and type identification

Supports common forensic image formats: E01, DD, AFF, VMDK
"""

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple
from uuid import UUID, uuid4

import os
import subprocess
import hashlib
import mimetypes
import magic
import logging
from dataclasses import dataclass

from .core import (
    DigitalEvidence,
    AnalysisResult,
    FileSystemArtifact,
    TimelineEvent,
    AnalysisStatus,
    EvidenceType,
    ForensicsConfig
)

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class FileSystemInfo:
    """Information about a file system"""
    filesystem_type: str
    sector_size: int
    cluster_size: int
    total_sectors: int
    image_type: str
    partition_table: str
    partitions: List[Dict[str, Any]]


@dataclass
class FileCarveResult:
    """Result from file carving operation"""
    file_signature: str
    file_type: str
    start_offset: int
    size: int
    carved_file_path: str
    confidence: float
    header_match: bool


class FileSystemAnalysis:
    """Container for file system analysis results"""
    
    def __init__(self):
        self.filesystem_info: Optional[FileSystemInfo] = None
        self.file_artifacts: List[FileSystemArtifact] = []
        self.timeline_events: List[TimelineEvent] = []
        self.carved_files: List[FileCarveResult] = []
        self.analysis_summary: Dict[str, Any] = {}
        self.deleted_files_recovered: int = 0
        self.total_files_analyzed: int = 0


class FileAnalyzer:
    """
    File system analyzer using The Sleuth Kit and custom analysis techniques
    
    Provides comprehensive analysis of disk images including:
    - File system structure analysis
    - Timeline generation (MACB - Modified, Accessed, Created, Birth)
    - Deleted file recovery using TSK tools
    - File carving for fragment recovery
    - Metadata extraction and analysis
    - File type verification and signature analysis
    """
    
    def __init__(self, config: Optional[ForensicsConfig] = None):
        """Initialize file analyzer with configuration"""
        self.config = config or ForensicsConfig()
        self.logger = logging.getLogger(f"{__name__}.FileAnalyzer")
        
        # Tool paths
        self.tsk_path = self.config.sleuthkit_path or "/usr/local/bin"
        self.tools = {
            'mmls': os.path.join(self.tsk_path, 'mmls'),
            'fsstat': os.path.join(self.tsk_path, 'fsstat'),
            'fls': os.path.join(self.tsk_path, 'fls'),
            'icat': os.path.join(self.tsk_path, 'icat'),
            'istat': os.path.join(self.tsk_path, 'istat'),
            'mactime': os.path.join(self.tsk_path, 'mactime'),
            'tsk_recover': os.path.join(self.tsk_path, 'tsk_recover')
        }
        
        # File signatures for carving
        self.file_signatures = {
            b'\xFF\xD8\xFF': {'ext': 'jpg', 'type': 'JPEG Image'},
            b'\x89\x50\x4E\x47': {'ext': 'png', 'type': 'PNG Image'},
            b'\x47\x49\x46\x38': {'ext': 'gif', 'type': 'GIF Image'},
            b'\x25\x50\x44\x46': {'ext': 'pdf', 'type': 'PDF Document'},
            b'\x50\x4B\x03\x04': {'ext': 'zip', 'type': 'ZIP Archive'},
            b'\x52\x61\x72\x21': {'ext': 'rar', 'type': 'RAR Archive'},
            b'\xD0\xCF\x11\xE0': {'ext': 'doc', 'type': 'Microsoft Office Document'},
            b'\x4D\x5A': {'ext': 'exe', 'type': 'Executable'},
            b'\x7F\x45\x4C\x46': {'ext': 'elf', 'type': 'ELF Executable'},
            b'\xCA\xFE\xBA\xBE': {'ext': 'class', 'type': 'Java Class'},
        }
        
        self.logger.info("File Analyzer initialized")
    
    def analyze_disk_image(
        self,
        evidence: DigitalEvidence,
        output_dir: Optional[str] = None
    ) -> FileSystemAnalysis:
        """
        Perform comprehensive analysis of disk image
        
        Args:
            evidence: Digital evidence containing disk image
            output_dir: Directory to store recovered files
            
        Returns:
            FileSystemAnalysis with complete results
        """
        self.logger.info(f"Starting disk image analysis: {evidence.name}")
        
        analysis = FileSystemAnalysis()
        image_path = evidence.file_path
        
        try:
            # Verify image integrity
            if not evidence.verify_integrity():
                raise ValueError("Image integrity verification failed")
            
            # Analyze file system structure
            analysis.filesystem_info = self._analyze_filesystem_structure(image_path)
            
            # Generate file listing
            file_artifacts = self._generate_file_listing(image_path)
            analysis.file_artifacts.extend(file_artifacts)
            analysis.total_files_analyzed = len(file_artifacts)
            
            # Recover deleted files if enabled
            if self.config.enable_deleted_file_recovery:
                deleted_files = self._recover_deleted_files(image_path, output_dir)
                analysis.file_artifacts.extend(deleted_files)
                analysis.deleted_files_recovered = len(deleted_files)
            
            # Generate timeline
            if self.config.enable_timeline_generation:
                timeline = self._generate_macb_timeline(image_path)
                analysis.timeline_events.extend(timeline)
            
            # Perform file carving for additional recovery
            if output_dir:
                carved_files = self._perform_file_carving(image_path, output_dir)
                analysis.carved_files.extend(carved_files)
            
            # Extract metadata from key files
            if self.config.enable_metadata_extraction:
                self._extract_file_metadata(analysis.file_artifacts)
            
            # Generate analysis summary
            analysis.analysis_summary = self._generate_analysis_summary(analysis)
            
            self.logger.info(f"Disk image analysis completed: {len(analysis.file_artifacts)} files analyzed")
            
        except Exception as e:
            self.logger.error(f"Disk image analysis failed: {str(e)}")
            raise
        
        return analysis
    
    def _analyze_filesystem_structure(self, image_path: str) -> FileSystemInfo:
        """Analyze file system structure using TSK tools"""
        try:
            # Get partition information
            mmls_output = self._run_tsk_command(['mmls', image_path])
            partitions = self._parse_mmls_output(mmls_output)
            
            # Analyze primary file system (usually partition 2 or first data partition)
            data_partition = None
            for partition in partitions:
                if partition.get('type', '').lower() not in ['unallocated', 'meta', 'primary table']:
                    data_partition = partition
                    break
            
            if not data_partition:
                raise ValueError("No data partition found in image")
            
            # Get file system statistics
            fsstat_cmd = ['fsstat', '-o', str(data_partition['start']), image_path]
            fsstat_output = self._run_tsk_command(fsstat_cmd)
            fs_info = self._parse_fsstat_output(fsstat_output)
            
            return FileSystemInfo(
                filesystem_type=fs_info.get('filesystem_type', 'unknown'),
                sector_size=fs_info.get('sector_size', 512),
                cluster_size=fs_info.get('cluster_size', 4096),
                total_sectors=fs_info.get('total_sectors', 0),
                image_type=fs_info.get('image_type', 'raw'),
                partition_table=fs_info.get('partition_table', 'unknown'),
                partitions=partitions
            )
            
        except Exception as e:
            self.logger.error(f"Filesystem structure analysis failed: {str(e)}")
            raise
    
    def _generate_file_listing(self, image_path: str) -> List[FileSystemArtifact]:
        """Generate comprehensive file listing using fls"""
        artifacts = []
        
        try:
            # Get file listing with full details
            fls_cmd = ['fls', '-r', '-p', '-m', '/', image_path]
            fls_output = self._run_tsk_command(fls_cmd)
            
            for line in fls_output.strip().split('\n'):
                if not line or line.startswith('#'):
                    continue
                
                artifact = self._parse_fls_line(line, image_path)
                if artifact:
                    artifacts.append(artifact)
            
            self.logger.info(f"Generated file listing: {len(artifacts)} files")
            
        except Exception as e:
            self.logger.error(f"File listing generation failed: {str(e)}")
            raise
        
        return artifacts
    
    def _parse_fls_line(self, line: str, image_path: str) -> Optional[FileSystemArtifact]:
        """Parse fls output line into FileSystemArtifact"""
        try:
            # fls output format: permissions|inode|filename|metadata
            parts = line.split('|')
            if len(parts) < 3:
                return None
            
            permissions = parts[0]
            inode_info = parts[1]
            filename = parts[2]
            
            # Extract inode number
            inode_number = None
            if inode_info and inode_info != '0':
                inode_number = int(inode_info.split('-')[0])
            
            # Determine if file is deleted
            is_deleted = '*' in permissions or '(deleted)' in filename
            
            artifact = FileSystemArtifact(
                file_path=filename,
                file_name=os.path.basename(filename),
                inode_number=inode_number,
                file_permissions=permissions,
                is_deleted=is_deleted
            )
            
            # Get additional metadata if inode exists
            if inode_number and not is_deleted:
                self._enhance_artifact_metadata(artifact, image_path, inode_number)
            
            return artifact
            
        except Exception as e:
            self.logger.warning(f"Failed to parse fls line: {line} - {str(e)}")
            return None
    
    def _enhance_artifact_metadata(
        self,
        artifact: FileSystemArtifact,
        image_path: str,
        inode: int
    ):
        """Enhance artifact with detailed metadata using istat"""
        try:
            istat_cmd = ['istat', image_path, str(inode)]
            istat_output = self._run_tsk_command(istat_cmd)
            
            # Parse istat output for timestamps and metadata
            metadata = self._parse_istat_output(istat_output)
            
            # Update artifact with metadata
            artifact.file_size = metadata.get('size', 0)
            artifact.created_time = metadata.get('created_time')
            artifact.modified_time = metadata.get('modified_time')
            artifact.accessed_time = metadata.get('accessed_time')
            artifact.file_owner = metadata.get('owner')
            artifact.file_group = metadata.get('group')
            
            # Determine file type
            if not artifact.is_deleted and artifact.file_size > 0:
                file_type = self._identify_file_type(image_path, inode)
                artifact.file_type = file_type.get('type')
                artifact.mime_type = file_type.get('mime')
                artifact.file_signature = file_type.get('signature')
            
        except Exception as e:
            self.logger.warning(f"Failed to enhance metadata for inode {inode}: {str(e)}")
    
    def _recover_deleted_files(
        self,
        image_path: str,
        output_dir: Optional[str]
    ) -> List[FileSystemArtifact]:
        """Recover deleted files using TSK tools"""
        if not output_dir:
            output_dir = "/tmp/lemkin_recovered_files"
        
        os.makedirs(output_dir, exist_ok=True)
        recovered_files = []
        
        try:
            # Use tsk_recover to extract deleted files
            recover_cmd = ['tsk_recover', '-e', image_path, output_dir]
            self._run_tsk_command(recover_cmd)
            
            # Analyze recovered files
            for root, dirs, files in os.walk(output_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    
                    try:
                        stat = os.stat(file_path)
                        
                        artifact = FileSystemArtifact(
                            file_path=file_path,
                            file_name=file,
                            file_size=stat.st_size,
                            is_deleted=True,
                            recovery_method="tsk_recover",
                            recovery_confidence=0.8,
                            created_time=datetime.fromtimestamp(stat.st_ctime),
                            modified_time=datetime.fromtimestamp(stat.st_mtime),
                            accessed_time=datetime.fromtimestamp(stat.st_atime)
                        )
                        
                        # Calculate hashes
                        if stat.st_size > 0:
                            hashes = self._calculate_file_hashes(file_path)
                            artifact.md5_hash = hashes['md5']
                            artifact.sha1_hash = hashes['sha1']
                            artifact.sha256_hash = hashes['sha256']
                        
                        # Identify file type
                        file_type = self._identify_recovered_file_type(file_path)
                        artifact.file_type = file_type.get('type')
                        artifact.mime_type = file_type.get('mime')
                        
                        recovered_files.append(artifact)
                        
                    except Exception as e:
                        self.logger.warning(f"Failed to analyze recovered file {file_path}: {str(e)}")
            
            self.logger.info(f"Recovered {len(recovered_files)} deleted files")
            
        except Exception as e:
            self.logger.error(f"Deleted file recovery failed: {str(e)}")
        
        return recovered_files
    
    def _generate_macb_timeline(self, image_path: str) -> List[TimelineEvent]:
        """Generate MACB timeline using mactime"""
        timeline_events = []
        
        try:
            # Generate body file for timeline
            fls_cmd = ['fls', '-r', '-m', '/', image_path]
            body_output = self._run_tsk_command(fls_cmd)
            
            # Create temporary body file
            body_file = "/tmp/timeline_body.txt"
            with open(body_file, 'w') as f:
                f.write(body_output)
            
            # Generate timeline with mactime
            mactime_cmd = ['mactime', '-b', body_file]
            timeline_output = self._run_tsk_command(mactime_cmd)
            
            # Parse timeline output
            for line in timeline_output.strip().split('\n'):
                event = self._parse_timeline_line(line)
                if event:
                    timeline_events.append(event)
            
            # Clean up
            if os.path.exists(body_file):
                os.remove(body_file)
            
            self.logger.info(f"Generated timeline with {len(timeline_events)} events")
            
        except Exception as e:
            self.logger.error(f"Timeline generation failed: {str(e)}")
        
        return timeline_events
    
    def _perform_file_carving(self, image_path: str, output_dir: str) -> List[FileCarveResult]:
        """Perform file carving to recover file fragments"""
        carved_files = []
        carve_dir = os.path.join(output_dir, "carved_files")
        os.makedirs(carve_dir, exist_ok=True)
        
        try:
            # Read image in chunks for carving
            with open(image_path, 'rb') as img:
                chunk_size = 64 * 1024  # 64KB chunks
                offset = 0
                
                while True:
                    chunk = img.read(chunk_size)
                    if not chunk:
                        break
                    
                    # Search for file signatures
                    for signature, sig_info in self.file_signatures.items():
                        sig_offset = chunk.find(signature)
                        if sig_offset != -1:
                            file_offset = offset + sig_offset
                            carved_file = self._carve_file(
                                img, file_offset, sig_info, carve_dir
                            )
                            if carved_file:
                                carved_files.append(carved_file)
                    
                    offset += chunk_size
            
            self.logger.info(f"File carving completed: {len(carved_files)} files carved")
            
        except Exception as e:
            self.logger.error(f"File carving failed: {str(e)}")
        
        return carved_files
    
    def _carve_file(
        self,
        image_file,
        offset: int,
        signature_info: Dict[str, str],
        output_dir: str
    ) -> Optional[FileCarveResult]:
        """Carve a single file from the image"""
        try:
            # Seek to file start
            image_file.seek(offset)
            
            # Estimate file size (simple heuristic)
            max_size = 50 * 1024 * 1024  # 50MB max
            data = image_file.read(max_size)
            
            # Find potential end of file (heuristic based on file type)
            actual_size = self._estimate_file_size(data, signature_info['type'])
            
            if actual_size > 0:
                # Create output file
                output_filename = f"carved_{offset}_{signature_info['ext']}"
                output_path = os.path.join(output_dir, output_filename)
                
                with open(output_path, 'wb') as out_file:
                    out_file.write(data[:actual_size])
                
                return FileCarveResult(
                    file_signature=signature_info['type'],
                    file_type=signature_info['ext'],
                    start_offset=offset,
                    size=actual_size,
                    carved_file_path=output_path,
                    confidence=0.7,  # Medium confidence for carved files
                    header_match=True
                )
            
        except Exception as e:
            self.logger.warning(f"File carving failed at offset {offset}: {str(e)}")
        
        return None
    
    def _estimate_file_size(self, data: bytes, file_type: str) -> int:
        """Estimate actual file size from carved data"""
        # Simple heuristics for common file types
        if file_type == 'JPEG Image':
            # Look for JPEG EOI marker
            eoi_marker = b'\xFF\xD9'
            eoi_pos = data.find(eoi_marker)
            return eoi_pos + 2 if eoi_pos != -1 else len(data)
        
        elif file_type == 'PNG Image':
            # Look for PNG IEND chunk
            iend_marker = b'IEND'
            iend_pos = data.find(iend_marker)
            return iend_pos + 8 if iend_pos != -1 else len(data)
        
        elif file_type == 'PDF Document':
            # Look for PDF EOF
            eof_marker = b'%%EOF'
            eof_pos = data.rfind(eof_marker)
            return eof_pos + 5 if eof_pos != -1 else len(data)
        
        else:
            # Default: return all data
            return len(data)
    
    def _identify_file_type(self, image_path: str, inode: int) -> Dict[str, str]:
        """Identify file type using file signature analysis"""
        try:
            # Extract file content using icat
            icat_cmd = ['icat', image_path, str(inode)]
            file_data = subprocess.run(
                icat_cmd,
                capture_output=True,
                timeout=30
            ).stdout
            
            if not file_data:
                return {'type': 'unknown', 'mime': 'application/octet-stream'}
            
            # Check file signature
            for signature, sig_info in self.file_signatures.items():
                if file_data.startswith(signature):
                    mime_type = mimetypes.guess_type(f"file.{sig_info['ext']}")[0]
                    return {
                        'type': sig_info['type'],
                        'mime': mime_type or 'application/octet-stream',
                        'signature': signature.hex()
                    }
            
            # Use libmagic if available
            try:
                import magic
                mime_type = magic.from_buffer(file_data, mime=True)
                file_type = magic.from_buffer(file_data)
                return {
                    'type': file_type,
                    'mime': mime_type,
                    'signature': file_data[:16].hex() if len(file_data) >= 16 else file_data.hex()
                }
            except ImportError:
                pass
            
            return {'type': 'unknown', 'mime': 'application/octet-stream'}
            
        except Exception as e:
            self.logger.warning(f"File type identification failed for inode {inode}: {str(e)}")
            return {'type': 'unknown', 'mime': 'application/octet-stream'}
    
    def _identify_recovered_file_type(self, file_path: str) -> Dict[str, str]:
        """Identify file type for recovered files"""
        try:
            # Read file header
            with open(file_path, 'rb') as f:
                header = f.read(16)
            
            # Check against known signatures
            for signature, sig_info in self.file_signatures.items():
                if header.startswith(signature):
                    mime_type = mimetypes.guess_type(file_path)[0]
                    return {
                        'type': sig_info['type'],
                        'mime': mime_type or 'application/octet-stream'
                    }
            
            # Use libmagic
            try:
                import magic
                mime_type = magic.from_file(file_path, mime=True)
                file_type = magic.from_file(file_path)
                return {'type': file_type, 'mime': mime_type}
            except ImportError:
                pass
            
            # Fallback to file extension
            ext = os.path.splitext(file_path)[1].lower()
            mime_type = mimetypes.guess_type(file_path)[0]
            return {
                'type': f'{ext[1:]} file' if ext else 'unknown',
                'mime': mime_type or 'application/octet-stream'
            }
            
        except Exception as e:
            self.logger.warning(f"File type identification failed for {file_path}: {str(e)}")
            return {'type': 'unknown', 'mime': 'application/octet-stream'}
    
    def _calculate_file_hashes(self, file_path: str) -> Dict[str, str]:
        """Calculate MD5, SHA1, and SHA256 hashes for a file"""
        hashes = {'md5': '', 'sha1': '', 'sha256': ''}
        
        try:
            md5_hash = hashlib.md5()
            sha1_hash = hashlib.sha1()
            sha256_hash = hashlib.sha256()
            
            with open(file_path, 'rb') as f:
                while chunk := f.read(8192):
                    md5_hash.update(chunk)
                    sha1_hash.update(chunk)
                    sha256_hash.update(chunk)
            
            hashes['md5'] = md5_hash.hexdigest()
            hashes['sha1'] = sha1_hash.hexdigest()
            hashes['sha256'] = sha256_hash.hexdigest()
            
        except Exception as e:
            self.logger.warning(f"Hash calculation failed for {file_path}: {str(e)}")
        
        return hashes
    
    def _extract_file_metadata(self, artifacts: List[FileSystemArtifact]):
        """Extract metadata from files that may contain PII or relevant content"""
        for artifact in artifacts:
            try:
                # Check for common file types that may contain metadata
                if artifact.file_type in ['JPEG Image', 'PDF Document', 'Microsoft Office Document']:
                    # This would integrate with metadata extraction tools
                    # For now, just flag for manual review
                    artifact.contains_pii = True
                    
            except Exception as e:
                self.logger.warning(f"Metadata extraction failed for {artifact.file_path}: {str(e)}")
    
    def _generate_analysis_summary(self, analysis: FileSystemAnalysis) -> Dict[str, Any]:
        """Generate summary of analysis results"""
        file_types = {}
        total_size = 0
        
        for artifact in analysis.file_artifacts:
            if artifact.file_type:
                file_types[artifact.file_type] = file_types.get(artifact.file_type, 0) + 1
            
            if artifact.file_size:
                total_size += artifact.file_size
        
        return {
            'total_files': len(analysis.file_artifacts),
            'deleted_files_recovered': analysis.deleted_files_recovered,
            'carved_files': len(analysis.carved_files),
            'timeline_events': len(analysis.timeline_events),
            'total_size_bytes': total_size,
            'file_types_found': file_types,
            'filesystem_type': analysis.filesystem_info.filesystem_type if analysis.filesystem_info else 'unknown'
        }
    
    def _run_tsk_command(self, cmd: List[str]) -> str:
        """Execute TSK command and return output"""
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout
                check=True
            )
            return result.stdout
            
        except subprocess.CalledProcessError as e:
            self.logger.error(f"TSK command failed: {' '.join(cmd)} - {e.stderr}")
            raise
        except subprocess.TimeoutExpired:
            self.logger.error(f"TSK command timed out: {' '.join(cmd)}")
            raise
    
    def _parse_mmls_output(self, output: str) -> List[Dict[str, Any]]:
        """Parse mmls output to extract partition information"""
        partitions = []
        
        for line in output.strip().split('\n'):
            if line.startswith('  ') and not line.startswith('  ---'):
                parts = line.split()
                if len(parts) >= 5:
                    try:
                        partitions.append({
                            'slot': parts[0],
                            'start': int(parts[1]),
                            'end': int(parts[2]),
                            'length': int(parts[3]),
                            'type': ' '.join(parts[4:])
                        })
                    except (ValueError, IndexError):
                        continue
        
        return partitions
    
    def _parse_fsstat_output(self, output: str) -> Dict[str, Any]:
        """Parse fsstat output to extract filesystem information"""
        info = {}
        
        for line in output.split('\n'):
            if 'File System Type:' in line:
                info['filesystem_type'] = line.split(':', 1)[1].strip()
            elif 'Sector Size:' in line:
                try:
                    info['sector_size'] = int(line.split(':', 1)[1].strip())
                except ValueError:
                    pass
            elif 'Cluster Size:' in line:
                try:
                    info['cluster_size'] = int(line.split(':', 1)[1].strip())
                except ValueError:
                    pass
            elif 'Total Sectors:' in line:
                try:
                    info['total_sectors'] = int(line.split(':', 1)[1].strip())
                except ValueError:
                    pass
        
        return info
    
    def _parse_istat_output(self, output: str) -> Dict[str, Any]:
        """Parse istat output to extract file metadata"""
        metadata = {}
        
        for line in output.split('\n'):
            if 'size:' in line.lower():
                try:
                    metadata['size'] = int(line.split(':', 1)[1].strip())
                except ValueError:
                    pass
            elif 'written:' in line.lower() or 'modified:' in line.lower():
                try:
                    timestamp_str = line.split(':', 1)[1].strip()
                    # Parse timestamp - TSK format varies
                    metadata['modified_time'] = self._parse_tsk_timestamp(timestamp_str)
                except Exception:
                    pass
            elif 'accessed:' in line.lower():
                try:
                    timestamp_str = line.split(':', 1)[1].strip()
                    metadata['accessed_time'] = self._parse_tsk_timestamp(timestamp_str)
                except Exception:
                    pass
            elif 'created:' in line.lower():
                try:
                    timestamp_str = line.split(':', 1)[1].strip()
                    metadata['created_time'] = self._parse_tsk_timestamp(timestamp_str)
                except Exception:
                    pass
        
        return metadata
    
    def _parse_tsk_timestamp(self, timestamp_str: str) -> Optional[datetime]:
        """Parse TSK timestamp string to datetime"""
        try:
            # TSK timestamp format varies, attempt common formats
            formats = [
                '%a %b %d %H:%M:%S %Y',
                '%Y-%m-%d %H:%M:%S',
                '%m/%d/%Y %H:%M:%S'
            ]
            
            for fmt in formats:
                try:
                    return datetime.strptime(timestamp_str.strip(), fmt)
                except ValueError:
                    continue
            
            return None
            
        except Exception:
            return None
    
    def _parse_timeline_line(self, line: str) -> Optional[TimelineEvent]:
        """Parse mactime timeline output line"""
        try:
            # mactime output format varies
            parts = line.split('\t')
            if len(parts) < 4:
                return None
            
            timestamp_str = parts[0]
            event_type = parts[1] if len(parts) > 1 else 'file_activity'
            filepath = parts[-1] if parts else 'unknown'
            
            # Parse timestamp
            timestamp = self._parse_tsk_timestamp(timestamp_str)
            if not timestamp:
                return None
            
            return TimelineEvent(
                timestamp=timestamp,
                event_type=event_type,
                description=f"File system activity: {filepath}",
                source_file=filepath,
                artifact_type="file_system",
                analysis_tool="mactime"
            )
            
        except Exception as e:
            self.logger.warning(f"Failed to parse timeline line: {line} - {str(e)}")
            return None


# Alias for backward compatibility
FileArtifact = FileSystemArtifact