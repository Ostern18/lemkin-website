"""
Lemkin Digital Forensics - Mobile Device Analyzer

This module provides comprehensive mobile device data extraction and analysis
capabilities including SMS, calls, photos, app data, and location history
analysis for legal investigations.
"""

import os
import json
import sqlite3
import plistlib
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import logging
import hashlib

from .core import (
    MobileArtifact, MobileDataExtraction, DigitalEvidence, AnalysisStatus,
    MobileArtifactType, ForensicsConfig
)

logger = logging.getLogger(__name__)


class MobileDeviceAnalyzer:
    """
    Comprehensive mobile device analysis for digital forensics investigations.
    
    Provides capabilities for:
    - iOS backup analysis
    - Android backup analysis
    - SMS/call log extraction
    - Photo/media analysis
    - App data extraction
    - Location history analysis
    """
    
    def __init__(self, config: Optional[ForensicsConfig] = None):
        self.config = config or ForensicsConfig()
        self.logger = logging.getLogger(f"{__name__}.MobileDeviceAnalyzer")
        
        # Common iOS database paths
        self.ios_databases = {
            'sms': 'Library/SMS/sms.db',
            'contacts': 'Library/AddressBook/AddressBook.sqlitedb',
            'call_history': 'Library/CallHistoryDB/CallHistory.storedata',
            'safari': 'Library/Safari/History.db',
            'notes': 'Library/Notes/notes.sqlite',
            'calendar': 'Library/Calendar/Calendar.sqlitedb',
            'photos': 'Media/PhotoData/Photos.sqlite'
        }
        
        # Common Android database paths
        self.android_databases = {
            'sms': 'data/com.android.providers.telephony/databases/mmssms.db',
            'contacts': 'data/com.android.providers.contacts/databases/contacts2.db',
            'call_log': 'data/com.android.providers.contacts/databases/calllog.db',
            'browser': 'data/com.android.browser/databases/browser2.db',
            'calendar': 'data/com.android.providers.calendar/databases/calendar.db'
        }
        
        # Common messaging app databases
        self.messaging_apps = {
            'whatsapp_ios': 'AppDomain-net.whatsapp.WhatsApp/Documents/ChatStorage.sqlite',
            'whatsapp_android': 'data/com.whatsapp/databases/msgstore.db',
            'telegram_ios': 'AppDomain-ph.telegra.Telegraph/Documents/Database',
            'telegram_android': 'data/org.telegram.messenger/files/cache4.db',
            'signal_ios': 'AppDomain-org.whispersystems.signal/Documents/Signal.sqlite',
            'signal_android': 'data/org.thoughtcrime.securesms/databases/signal.db'
        }
        
        # File type signatures for media analysis
        self.media_signatures = {
            b'\xff\xd8\xff': ('JPEG', '.jpg'),
            b'\x89PNG\r\n\x1a\n': ('PNG', '.png'),
            b'GIF87a': ('GIF', '.gif'),
            b'GIF89a': ('GIF', '.gif'),
            b'\x00\x00\x00\x18ftypmp4': ('MP4', '.mp4'),
            b'\x00\x00\x00\x20ftypM4V': ('M4V', '.m4v'),
            b'RIFF': ('AVI/WAV', '.avi')
        }
    
    def extract_mobile_data(self, backup_path: Path) -> MobileDataExtraction:
        """
        Extract and analyze mobile device data from backup
        
        Args:
            backup_path: Path to mobile device backup directory
            
        Returns:
            MobileDataExtraction containing all findings
        """
        extraction = MobileDataExtraction(evidence_id=self._generate_evidence_id())
        extraction.status = AnalysisStatus.IN_PROGRESS
        
        try:
            self.logger.info(f"Starting mobile data extraction from {backup_path}")
            
            # Validate backup path
            if not backup_path.exists():
                raise FileNotFoundError(f"Backup path not found: {backup_path}")
            
            # Determine device type and OS
            device_info = self._identify_device_type(backup_path)
            extraction.device_model = device_info.get('model')
            extraction.device_serial = device_info.get('serial')
            extraction.os_version = device_info.get('os_version')
            extraction.backup_date = device_info.get('backup_date')
            
            self.logger.info(f"Identified device: {extraction.device_model}")
            
            # Extract based on device type
            if device_info.get('type') == 'ios':
                self._extract_ios_data(backup_path, extraction)
            elif device_info.get('type') == 'android':
                self._extract_android_data(backup_path, extraction)
            else:
                self._extract_generic_mobile_data(backup_path, extraction)
            
            # Analyze communication patterns
            self._analyze_communication_patterns(extraction)
            
            # Analyze location data
            self._analyze_location_data(extraction)
            
            # Generate activity timeline
            self._generate_activity_timeline(extraction)
            
            # Generate key findings
            self._generate_key_findings(extraction)
            
            extraction.completed_at = datetime.utcnow()
            extraction.status = AnalysisStatus.COMPLETED
            
            self.logger.info("Mobile data extraction completed successfully")
            
        except Exception as e:
            self.logger.error(f"Mobile data extraction failed: {str(e)}")
            extraction.status = AnalysisStatus.FAILED
            extraction.error_messages.append(str(e))
        
        return extraction
    
    def _identify_device_type(self, backup_path: Path) -> Dict[str, Any]:
        """Identify the type of mobile device from backup structure"""
        device_info = {'type': 'unknown'}
        
        try:
            # Check for iOS backup structure
            info_plist = backup_path / 'Info.plist'
            manifest_plist = backup_path / 'Manifest.plist'
            
            if info_plist.exists() and manifest_plist.exists():
                device_info['type'] = 'ios'
                
                # Read device info from Info.plist
                try:
                    with open(info_plist, 'rb') as f:
                        plist_data = plistlib.load(f)
                        
                    device_info.update({
                        'model': plist_data.get('Product Name', 'Unknown iOS Device'),
                        'serial': plist_data.get('Serial Number'),
                        'os_version': plist_data.get('Product Version'),
                        'backup_date': plist_data.get('Date')
                    })
                except Exception as e:
                    self.logger.warning(f"Could not read iOS device info: {str(e)}")
                
                return device_info
            
            # Check for Android backup structure
            android_indicators = [
                'data/com.android.providers.telephony',
                'data/com.android.providers.contacts',
                'system/users',
                'apps'
            ]
            
            android_matches = sum(1 for indicator in android_indicators 
                                if (backup_path / indicator).exists())
            
            if android_matches >= 2:
                device_info['type'] = 'android'
                
                # Try to get Android device info
                build_prop = backup_path / 'system' / 'build.prop'
                if build_prop.exists():
                    try:
                        with open(build_prop, 'r', encoding='utf-8', errors='ignore') as f:
                            build_data = f.read()
                            
                        # Parse build properties
                        model_match = re.search(r'ro\.product\.model=(.+)', build_data)
                        version_match = re.search(r'ro\.build\.version\.release=(.+)', build_data)
                        
                        if model_match:
                            device_info['model'] = model_match.group(1).strip()
                        if version_match:
                            device_info['os_version'] = version_match.group(1).strip()
                            
                    except Exception as e:
                        self.logger.warning(f"Could not read Android device info: {str(e)}")
                
                return device_info
            
            # Check for generic mobile data
            mobile_indicators = [
                'sms', 'contacts', 'photos', 'media',
                'Messages', 'AddressBook', 'PhotoData'
            ]
            
            for indicator in mobile_indicators:
                if any(indicator.lower() in str(p).lower() 
                      for p in backup_path.rglob('*')):
                    device_info['type'] = 'mobile'
                    break
            
        except Exception as e:
            self.logger.warning(f"Device type identification failed: {str(e)}")
        
        return device_info
    
    def _extract_ios_data(self, backup_path: Path, extraction: MobileDataExtraction):
        """Extract data from iOS backup"""
        try:
            # Extract SMS messages
            sms_artifacts = self._extract_ios_sms(backup_path)
            extraction.mobile_artifacts.extend(sms_artifacts)
            extraction.sms_messages = len([a for a in sms_artifacts 
                                         if a.artifact_type == MobileArtifactType.SMS_MESSAGE])
            
            # Extract call logs
            call_artifacts = self._extract_ios_calls(backup_path)
            extraction.mobile_artifacts.extend(call_artifacts)
            extraction.call_logs = len([a for a in call_artifacts 
                                      if a.artifact_type == MobileArtifactType.CALL_LOG])
            
            # Extract contacts
            contact_artifacts = self._extract_ios_contacts(backup_path)
            extraction.mobile_artifacts.extend(contact_artifacts)
            extraction.contacts = len([a for a in contact_artifacts 
                                     if a.artifact_type == MobileArtifactType.CONTACT])
            
            # Extract photos and media
            photo_artifacts = self._extract_ios_photos(backup_path)
            extraction.mobile_artifacts.extend(photo_artifacts)
            extraction.photos = len([a for a in photo_artifacts 
                                   if a.artifact_type in [MobileArtifactType.PHOTO, MobileArtifactType.VIDEO]])
            
            # Extract location data
            location_artifacts = self._extract_ios_location(backup_path)
            extraction.mobile_artifacts.extend(location_artifacts)
            extraction.location_points = len([a for a in location_artifacts 
                                            if a.artifact_type == MobileArtifactType.LOCATION_HISTORY])
            
            # Extract app data
            app_artifacts = self._extract_ios_app_data(backup_path)
            extraction.mobile_artifacts.extend(app_artifacts)
            extraction.apps_analyzed = len(set(a.app_package for a in app_artifacts if a.app_package))
            
        except Exception as e:
            self.logger.error(f"iOS data extraction failed: {str(e)}")
            extraction.error_messages.append(f"iOS extraction: {str(e)}")
    
    def _extract_ios_sms(self, backup_path: Path) -> List[MobileArtifact]:
        """Extract SMS messages from iOS backup"""
        artifacts = []
        
        try:
            # Find SMS database
            sms_db_path = self._find_ios_database(backup_path, 'sms.db')
            if not sms_db_path:
                return artifacts
            
            conn = sqlite3.connect(sms_db_path)
            cursor = conn.cursor()
            
            # Query SMS messages
            cursor.execute("""
                SELECT 
                    message.ROWID,
                    message.text,
                    message.date,
                    message.is_from_me,
                    handle.id as phone_number,
                    message.service
                FROM message 
                LEFT JOIN handle ON message.handle_id = handle.ROWID
                WHERE message.text IS NOT NULL
                ORDER BY message.date
            """)
            
            for row in cursor.fetchall():
                try:
                    # Convert Apple timestamp (nanoseconds since 2001-01-01)
                    apple_timestamp = row[2]
                    if apple_timestamp:
                        timestamp = datetime(2001, 1, 1) + timedelta(seconds=apple_timestamp / 1000000000)
                    else:
                        timestamp = None
                    
                    artifact = MobileArtifact(
                        artifact_type=MobileArtifactType.SMS_MESSAGE,
                        content={
                            'message_id': row[0],
                            'text': row[1],
                            'direction': 'outgoing' if row[3] else 'incoming',
                            'phone_number': row[4] or 'Unknown',
                            'service': row[5] or 'SMS'
                        },
                        timestamp=timestamp,
                        database_path=str(sms_db_path)
                    )
                    
                    artifacts.append(artifact)
                    
                except Exception as e:
                    self.logger.warning(f"Failed to process SMS row: {str(e)}")
                    continue
            
            conn.close()
            self.logger.info(f"Extracted {len(artifacts)} SMS messages")
            
        except Exception as e:
            self.logger.error(f"iOS SMS extraction failed: {str(e)}")
        
        return artifacts
    
    def _extract_ios_calls(self, backup_path: Path) -> List[MobileArtifact]:
        """Extract call logs from iOS backup"""
        artifacts = []
        
        try:
            # Find call history database
            call_db_path = self._find_ios_database(backup_path, 'CallHistory.storedata')
            if not call_db_path:
                return artifacts
            
            conn = sqlite3.connect(call_db_path)
            cursor = conn.cursor()
            
            # Query call history
            cursor.execute("""
                SELECT 
                    ROWID,
                    address,
                    date,
                    duration,
                    flags,
                    name
                FROM ZCALLRECORD
                ORDER BY date
            """)
            
            for row in cursor.fetchall():
                try:
                    # Convert Apple timestamp
                    apple_timestamp = row[2]
                    if apple_timestamp:
                        timestamp = datetime(2001, 1, 1) + timedelta(seconds=apple_timestamp)
                    else:
                        timestamp = None
                    
                    # Determine call type from flags
                    flags = row[4] or 0
                    call_type = 'unknown'
                    if flags & 0x01:  # Outgoing
                        call_type = 'outgoing'
                    elif flags & 0x02:  # Missed
                        call_type = 'missed'
                    else:
                        call_type = 'incoming'
                    
                    artifact = MobileArtifact(
                        artifact_type=MobileArtifactType.CALL_LOG,
                        content={
                            'call_id': row[0],
                            'phone_number': row[1] or 'Unknown',
                            'duration': row[3] or 0,
                            'call_type': call_type,
                            'contact_name': row[5]
                        },
                        timestamp=timestamp,
                        database_path=str(call_db_path)
                    )
                    
                    artifacts.append(artifact)
                    
                except Exception as e:
                    self.logger.warning(f"Failed to process call log row: {str(e)}")
                    continue
            
            conn.close()
            self.logger.info(f"Extracted {len(artifacts)} call log entries")
            
        except Exception as e:
            self.logger.error(f"iOS call extraction failed: {str(e)}")
        
        return artifacts
    
    def _extract_ios_contacts(self, backup_path: Path) -> List[MobileArtifact]:
        """Extract contacts from iOS backup"""
        artifacts = []
        
        try:
            contact_db_path = self._find_ios_database(backup_path, 'AddressBook.sqlitedb')
            if not contact_db_path:
                return artifacts
            
            conn = sqlite3.connect(contact_db_path)
            cursor = conn.cursor()
            
            # Query contacts
            cursor.execute("""
                SELECT 
                    p.ROWID,
                    p.first,
                    p.last,
                    v.value as phone_number
                FROM ABPerson p
                LEFT JOIN ABMultiValue v ON p.ROWID = v.record_id
                WHERE v.property = 3  -- Phone number property
            """)
            
            for row in cursor.fetchall():
                try:
                    artifact = MobileArtifact(
                        artifact_type=MobileArtifactType.CONTACT,
                        content={
                            'contact_id': row[0],
                            'first_name': row[1] or '',
                            'last_name': row[2] or '',
                            'full_name': f"{row[1] or ''} {row[2] or ''}".strip(),
                            'phone_number': row[3] or ''
                        },
                        database_path=str(contact_db_path)
                    )
                    
                    artifacts.append(artifact)
                    
                except Exception as e:
                    self.logger.warning(f"Failed to process contact row: {str(e)}")
                    continue
            
            conn.close()
            self.logger.info(f"Extracted {len(artifacts)} contacts")
            
        except Exception as e:
            self.logger.error(f"iOS contacts extraction failed: {str(e)}")
        
        return artifacts
    
    def _extract_ios_photos(self, backup_path: Path) -> List[MobileArtifact]:
        """Extract photos and media from iOS backup"""
        artifacts = []
        
        try:
            # Find photo database
            photos_db_path = self._find_ios_database(backup_path, 'Photos.sqlite')
            if photos_db_path:
                artifacts.extend(self._extract_from_photos_db(photos_db_path))
            
            # Also look for media files directly
            media_artifacts = self._extract_media_files(backup_path)
            artifacts.extend(media_artifacts)
            
            self.logger.info(f"Extracted {len(artifacts)} photo/media artifacts")
            
        except Exception as e:
            self.logger.error(f"iOS photos extraction failed: {str(e)}")
        
        return artifacts
    
    def _extract_from_photos_db(self, photos_db_path: Path) -> List[MobileArtifact]:
        """Extract photo metadata from Photos database"""
        artifacts = []
        
        try:
            conn = sqlite3.connect(photos_db_path)
            cursor = conn.cursor()
            
            # Query photo metadata
            cursor.execute("""
                SELECT 
                    ROWID,
                    filename,
                    date,
                    latitude,
                    longitude,
                    directory
                FROM ZGENERICASSET
                WHERE filename IS NOT NULL
                ORDER BY date
            """)
            
            for row in cursor.fetchall():
                try:
                    # Convert Apple timestamp
                    apple_timestamp = row[2]
                    if apple_timestamp:
                        timestamp = datetime(2001, 1, 1) + timedelta(seconds=apple_timestamp)
                    else:
                        timestamp = None
                    
                    artifact_type = MobileArtifactType.PHOTO
                    if row[1] and any(ext in row[1].lower() for ext in ['.mov', '.mp4', '.avi']):
                        artifact_type = MobileArtifactType.VIDEO
                    
                    artifact = MobileArtifact(
                        artifact_type=artifact_type,
                        content={
                            'photo_id': row[0],
                            'filename': row[1],
                            'directory': row[5]
                        },
                        timestamp=timestamp,
                        latitude=row[3] if row[3] != 0 else None,
                        longitude=row[4] if row[4] != 0 else None,
                        database_path=str(photos_db_path)
                    )
                    
                    artifacts.append(artifact)
                    
                except Exception as e:
                    self.logger.warning(f"Failed to process photo row: {str(e)}")
                    continue
            
            conn.close()
            
        except Exception as e:
            self.logger.warning(f"Photos database extraction failed: {str(e)}")
        
        return artifacts
    
    def _extract_media_files(self, backup_path: Path) -> List[MobileArtifact]:
        """Extract media files directly from backup"""
        artifacts = []
        
        try:
            media_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff',
                              '.mp4', '.mov', '.avi', '.mp3', '.wav', '.m4a'}
            
            # Search for media files
            for file_path in backup_path.rglob('*'):
                if file_path.is_file() and file_path.suffix.lower() in media_extensions:
                    try:
                        stat_info = file_path.stat()
                        
                        # Determine artifact type
                        if file_path.suffix.lower() in {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff'}:
                            artifact_type = MobileArtifactType.PHOTO
                        elif file_path.suffix.lower() in {'.mp4', '.mov', '.avi'}:
                            artifact_type = MobileArtifactType.VIDEO
                        else:
                            continue  # Skip audio files for now
                        
                        artifact = MobileArtifact(
                            artifact_type=artifact_type,
                            content={
                                'filename': file_path.name,
                                'file_path': str(file_path),
                                'file_size': stat_info.st_size
                            },
                            timestamp=datetime.fromtimestamp(stat_info.st_mtime)
                        )
                        
                        artifacts.append(artifact)
                        
                    except Exception as e:
                        self.logger.warning(f"Failed to process media file {file_path}: {str(e)}")
                        continue
                        
        except Exception as e:
            self.logger.warning(f"Media file extraction failed: {str(e)}")
        
        return artifacts
    
    def _extract_ios_location(self, backup_path: Path) -> List[MobileArtifact]:
        """Extract location data from iOS backup"""
        artifacts = []
        
        try:
            # Look for location databases
            location_paths = [
                'Library/Caches/locationd/consolidated.db',
                'Library/Caches/locationd/cache_encryptedA.db',
                'Library/Caches/com.apple.locationd/consolidated.db'
            ]
            
            for path in location_paths:
                full_path = self._find_ios_file(backup_path, path)
                if full_path:
                    location_artifacts = self._extract_from_location_db(full_path)
                    artifacts.extend(location_artifacts)
            
            self.logger.info(f"Extracted {len(artifacts)} location data points")
            
        except Exception as e:
            self.logger.error(f"iOS location extraction failed: {str(e)}")
        
        return artifacts
    
    def _extract_from_location_db(self, db_path: Path) -> List[MobileArtifact]:
        """Extract location data from location database"""
        artifacts = []
        
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # Try different table structures
            tables_to_try = ['CdmaCellLocation', 'CellLocation', 'WifiLocation', 'LocationHarvest']
            
            for table in tables_to_try:
                try:
                    cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table}'")
                    if cursor.fetchone():
                        cursor.execute(f"""
                            SELECT timestamp, latitude, longitude, accuracy 
                            FROM {table} 
                            WHERE latitude != 0 AND longitude != 0
                            LIMIT 1000
                        """)
                        
                        for row in cursor.fetchall():
                            try:
                                timestamp = datetime.fromtimestamp(row[0]) if row[0] else None
                                
                                artifact = MobileArtifact(
                                    artifact_type=MobileArtifactType.LOCATION_HISTORY,
                                    content={
                                        'source_table': table,
                                        'accuracy': row[3] if len(row) > 3 else None
                                    },
                                    timestamp=timestamp,
                                    latitude=float(row[1]),
                                    longitude=float(row[2]),
                                    location_accuracy=float(row[3]) if len(row) > 3 and row[3] else None,
                                    database_path=str(db_path)
                                )
                                
                                artifacts.append(artifact)
                                
                            except Exception as e:
                                continue
                                
                except sqlite3.OperationalError:
                    continue
            
            conn.close()
            
        except Exception as e:
            self.logger.warning(f"Location database extraction failed: {str(e)}")
        
        return artifacts
    
    def _extract_ios_app_data(self, backup_path: Path) -> List[MobileArtifact]:
        """Extract data from various iOS apps"""
        artifacts = []
        
        try:
            # Extract messaging app data
            for app_name, db_path in self.messaging_apps.items():
                if 'ios' in app_name:
                    full_path = self._find_ios_file(backup_path, db_path)
                    if full_path:
                        app_artifacts = self._extract_messaging_app_data(full_path, app_name)
                        artifacts.extend(app_artifacts)
            
            # Extract browser history
            safari_db = self._find_ios_database(backup_path, 'History.db')
            if safari_db:
                browser_artifacts = self._extract_safari_history(safari_db)
                artifacts.extend(browser_artifacts)
            
            self.logger.info(f"Extracted {len(artifacts)} app data artifacts")
            
        except Exception as e:
            self.logger.error(f"iOS app data extraction failed: {str(e)}")
        
        return artifacts
    
    def _extract_android_data(self, backup_path: Path, extraction: MobileDataExtraction):
        """Extract data from Android backup"""
        try:
            # Implementation would be similar to iOS but adapted for Android structure
            # For brevity, implementing basic structure
            
            # Extract SMS messages
            sms_artifacts = self._extract_android_sms(backup_path)
            extraction.mobile_artifacts.extend(sms_artifacts)
            extraction.sms_messages = len([a for a in sms_artifacts 
                                         if a.artifact_type == MobileArtifactType.SMS_MESSAGE])
            
            # Extract call logs
            call_artifacts = self._extract_android_calls(backup_path)
            extraction.mobile_artifacts.extend(call_artifacts)
            extraction.call_logs = len([a for a in call_artifacts 
                                      if a.artifact_type == MobileArtifactType.CALL_LOG])
            
            # Extract contacts
            contact_artifacts = self._extract_android_contacts(backup_path)
            extraction.mobile_artifacts.extend(contact_artifacts)
            extraction.contacts = len([a for a in contact_artifacts 
                                     if a.artifact_type == MobileArtifactType.CONTACT])
            
        except Exception as e:
            self.logger.error(f"Android data extraction failed: {str(e)}")
            extraction.error_messages.append(f"Android extraction: {str(e)}")
    
    def _extract_android_sms(self, backup_path: Path) -> List[MobileArtifact]:
        """Extract SMS messages from Android backup"""
        artifacts = []
        
        try:
            sms_db_path = backup_path / self.android_databases['sms']
            if not sms_db_path.exists():
                return artifacts
            
            conn = sqlite3.connect(sms_db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT _id, address, body, date, type
                FROM sms
                WHERE body IS NOT NULL
                ORDER BY date
            """)
            
            for row in cursor.fetchall():
                try:
                    timestamp = datetime.fromtimestamp(row[3] / 1000) if row[3] else None
                    
                    artifact = MobileArtifact(
                        artifact_type=MobileArtifactType.SMS_MESSAGE,
                        content={
                            'message_id': row[0],
                            'phone_number': row[1] or 'Unknown',
                            'text': row[2],
                            'direction': 'outgoing' if row[4] == 2 else 'incoming'
                        },
                        timestamp=timestamp,
                        database_path=str(sms_db_path)
                    )
                    
                    artifacts.append(artifact)
                    
                except Exception as e:
                    continue
            
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Android SMS extraction failed: {str(e)}")
        
        return artifacts
    
    def _extract_android_calls(self, backup_path: Path) -> List[MobileArtifact]:
        """Extract call logs from Android backup"""
        # Similar implementation to iOS but adapted for Android database structure
        return []
    
    def _extract_android_contacts(self, backup_path: Path) -> List[MobileArtifact]:
        """Extract contacts from Android backup"""
        # Similar implementation to iOS but adapted for Android database structure
        return []
    
    def _extract_generic_mobile_data(self, backup_path: Path, extraction: MobileDataExtraction):
        """Extract data from generic mobile backup"""
        try:
            # Look for common database files
            database_files = []
            for ext in ['.db', '.sqlite', '.sqlite3']:
                database_files.extend(backup_path.rglob(f'*{ext}'))
            
            artifacts = []
            for db_file in database_files:
                try:
                    db_artifacts = self._analyze_generic_database(db_file)
                    artifacts.extend(db_artifacts)
                except Exception as e:
                    continue
            
            extraction.mobile_artifacts.extend(artifacts)
            
        except Exception as e:
            self.logger.error(f"Generic mobile data extraction failed: {str(e)}")
    
    def _analyze_generic_database(self, db_path: Path) -> List[MobileArtifact]:
        """Analyze a generic SQLite database for mobile artifacts"""
        artifacts = []
        
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # Get table names
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]
            
            for table in tables:
                if any(keyword in table.lower() for keyword in ['message', 'sms', 'chat']):
                    # Try to extract message-like data
                    message_artifacts = self._extract_generic_messages(cursor, table, db_path)
                    artifacts.extend(message_artifacts)
            
            conn.close()
            
        except Exception as e:
            self.logger.warning(f"Generic database analysis failed for {db_path}: {str(e)}")
        
        return artifacts
    
    def _extract_generic_messages(self, cursor, table_name: str, db_path: Path) -> List[MobileArtifact]:
        """Extract message-like data from generic table"""
        artifacts = []
        
        try:
            # Get column names
            cursor.execute(f"PRAGMA table_info({table_name})")
            columns = [row[1] for row in cursor.fetchall()]
            
            # Look for text/message columns
            text_columns = [col for col in columns if any(keyword in col.lower() 
                          for keyword in ['text', 'message', 'content', 'body'])]
            
            if text_columns:
                text_col = text_columns[0]
                cursor.execute(f"SELECT * FROM {table_name} WHERE {text_col} IS NOT NULL LIMIT 100")
                
                for row in cursor.fetchall():
                    try:
                        artifact = MobileArtifact(
                            artifact_type=MobileArtifactType.APP_DATA,
                            content={
                                'table': table_name,
                                'text': str(row[columns.index(text_col)]),
                                'raw_data': str(row)
                            },
                            database_path=str(db_path),
                            app_package=f"unknown.{table_name.lower()}"
                        )
                        
                        artifacts.append(artifact)
                        
                    except Exception as e:
                        continue
            
        except Exception as e:
            self.logger.warning(f"Generic message extraction failed: {str(e)}")
        
        return artifacts
    
    def _find_ios_database(self, backup_path: Path, db_name: str) -> Optional[Path]:
        """Find iOS database file in backup"""
        try:
            # iOS backups have hashed filenames, need to search
            for file_path in backup_path.rglob('*.db'):
                if db_name.lower() in file_path.name.lower():
                    return file_path
            
            for file_path in backup_path.rglob('*.sqlite*'):
                if db_name.lower() in file_path.name.lower():
                    return file_path
            
            return None
            
        except Exception as e:
            self.logger.warning(f"Database search failed: {str(e)}")
            return None
    
    def _find_ios_file(self, backup_path: Path, relative_path: str) -> Optional[Path]:
        """Find iOS file in backup by relative path"""
        try:
            # Try direct path
            full_path = backup_path / relative_path
            if full_path.exists():
                return full_path
            
            # Search by filename
            filename = Path(relative_path).name
            for file_path in backup_path.rglob(filename):
                return file_path
            
            return None
            
        except Exception as e:
            return None
    
    def _extract_messaging_app_data(self, db_path: Path, app_name: str) -> List[MobileArtifact]:
        """Extract data from messaging app database"""
        # Implementation would depend on specific app database structure
        return []
    
    def _extract_safari_history(self, db_path: Path) -> List[MobileArtifact]:
        """Extract Safari browsing history"""
        artifacts = []
        
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT url, title, visit_time
                FROM history_visits
                JOIN history_items ON history_visits.history_item = history_items.id
                ORDER BY visit_time DESC
                LIMIT 1000
            """)
            
            for row in cursor.fetchall():
                try:
                    # Convert Apple timestamp
                    timestamp = datetime(2001, 1, 1) + timedelta(seconds=row[2]) if row[2] else None
                    
                    artifact = MobileArtifact(
                        artifact_type=MobileArtifactType.BROWSER_HISTORY,
                        content={
                            'url': row[0],
                            'title': row[1] or 'Untitled',
                            'browser': 'Safari'
                        },
                        timestamp=timestamp,
                        database_path=str(db_path),
                        app_package='com.apple.mobilesafari'
                    )
                    
                    artifacts.append(artifact)
                    
                except Exception as e:
                    continue
            
            conn.close()
            
        except Exception as e:
            self.logger.warning(f"Safari history extraction failed: {str(e)}")
        
        return artifacts
    
    def _analyze_communication_patterns(self, extraction: MobileDataExtraction):
        """Analyze communication patterns in mobile data"""
        try:
            # Analyze SMS patterns
            sms_artifacts = [a for a in extraction.mobile_artifacts 
                           if a.artifact_type == MobileArtifactType.SMS_MESSAGE]
            
            # Count messages by contact
            contact_message_count = {}
            for artifact in sms_artifacts:
                phone = artifact.content.get('phone_number', 'Unknown')
                contact_message_count[phone] = contact_message_count.get(phone, 0) + 1
            
            # Analyze call patterns
            call_artifacts = [a for a in extraction.mobile_artifacts 
                            if a.artifact_type == MobileArtifactType.CALL_LOG]
            
            contact_call_count = {}
            for artifact in call_artifacts:
                phone = artifact.content.get('phone_number', 'Unknown')
                contact_call_count[phone] = contact_call_count.get(phone, 0) + 1
            
            # Create communication summary
            extraction.communication_summary = {
                'total_sms_messages': len(sms_artifacts),
                'total_calls': len(call_artifacts),
                'unique_sms_contacts': len(contact_message_count),
                'unique_call_contacts': len(contact_call_count),
                'top_sms_contacts': sorted(contact_message_count.items(), 
                                         key=lambda x: x[1], reverse=True)[:10],
                'top_call_contacts': sorted(contact_call_count.items(), 
                                          key=lambda x: x[1], reverse=True)[:10]
            }
            
        except Exception as e:
            self.logger.error(f"Communication pattern analysis failed: {str(e)}")
    
    def _analyze_location_data(self, extraction: MobileDataExtraction):
        """Analyze location data patterns"""
        try:
            location_artifacts = [a for a in extraction.mobile_artifacts 
                                if a.artifact_type == MobileArtifactType.LOCATION_HISTORY]
            
            if not location_artifacts:
                return
            
            # Calculate location statistics
            latitudes = [a.latitude for a in location_artifacts if a.latitude]
            longitudes = [a.longitude for a in location_artifacts if a.longitude]
            
            if latitudes and longitudes:
                extraction.location_summary = {
                    'total_location_points': len(location_artifacts),
                    'latitude_range': [min(latitudes), max(latitudes)],
                    'longitude_range': [min(longitudes), max(longitudes)],
                    'center_point': [
                        sum(latitudes) / len(latitudes),
                        sum(longitudes) / len(longitudes)
                    ]
                }
            
        except Exception as e:
            self.logger.error(f"Location data analysis failed: {str(e)}")
    
    def _generate_activity_timeline(self, extraction: MobileDataExtraction):
        """Generate timeline of mobile device activity"""
        try:
            timeline_events = []
            
            for artifact in extraction.mobile_artifacts:
                if artifact.timestamp:
                    event = {
                        'timestamp': artifact.timestamp,
                        'type': artifact.artifact_type,
                        'description': self._format_artifact_description(artifact),
                        'details': artifact.content
                    }
                    
                    if artifact.latitude and artifact.longitude:
                        event['location'] = {
                            'latitude': artifact.latitude,
                            'longitude': artifact.longitude
                        }
                    
                    timeline_events.append(event)
            
            # Sort by timestamp
            timeline_events.sort(key=lambda x: x['timestamp'])
            extraction.activity_timeline = timeline_events
            
        except Exception as e:
            self.logger.error(f"Activity timeline generation failed: {str(e)}")
    
    def _format_artifact_description(self, artifact: MobileArtifact) -> str:
        """Format artifact for timeline display"""
        if artifact.artifact_type == MobileArtifactType.SMS_MESSAGE:
            direction = artifact.content.get('direction', 'unknown')
            phone = artifact.content.get('phone_number', 'unknown')
            return f"SMS {direction}: {phone}"
        elif artifact.artifact_type == MobileArtifactType.CALL_LOG:
            call_type = artifact.content.get('call_type', 'unknown')
            phone = artifact.content.get('phone_number', 'unknown')
            return f"Call {call_type}: {phone}"
        elif artifact.artifact_type == MobileArtifactType.PHOTO:
            filename = artifact.content.get('filename', 'unknown')
            return f"Photo taken: {filename}"
        elif artifact.artifact_type == MobileArtifactType.LOCATION_HISTORY:
            return f"Location: {artifact.latitude:.6f}, {artifact.longitude:.6f}"
        else:
            return f"{artifact.artifact_type.replace('_', ' ').title()}"
    
    def _generate_key_findings(self, extraction: MobileDataExtraction):
        """Generate key findings summary"""
        try:
            findings = []
            
            # Device information
            if extraction.device_model:
                findings.append(f"Device: {extraction.device_model}")
            
            # Data extraction summary
            findings.append(
                f"Extracted {extraction.sms_messages:,} SMS messages, "
                f"{extraction.call_logs:,} call records, "
                f"{extraction.contacts:,} contacts"
            )
            
            if extraction.photos > 0:
                findings.append(f"Found {extraction.photos:,} photos and media files")
            
            if extraction.location_points > 0:
                findings.append(f"Extracted {extraction.location_points:,} location data points")
            
            if extraction.apps_analyzed > 0:
                findings.append(f"Analyzed data from {extraction.apps_analyzed} applications")
            
            # Timeline information
            if extraction.activity_timeline:
                earliest = min(event['timestamp'] for event in extraction.activity_timeline)
                latest = max(event['timestamp'] for event in extraction.activity_timeline)
                findings.append(
                    f"Activity timeline spans from {earliest.strftime('%Y-%m-%d')} "
                    f"to {latest.strftime('%Y-%m-%d')}"
                )
            
            # Communication patterns
            if extraction.communication_summary:
                comm = extraction.communication_summary
                findings.append(
                    f"Communication with {comm.get('unique_sms_contacts', 0)} SMS contacts "
                    f"and {comm.get('unique_call_contacts', 0)} call contacts"
                )
            
            extraction.key_findings.extend(findings)
            
        except Exception as e:
            self.logger.error(f"Key findings generation failed: {str(e)}")
    
    def _generate_evidence_id(self) -> str:
        """Generate a unique evidence ID"""
        import uuid
        return str(uuid.uuid4())


def extract_mobile_data(
    backup_path: Path, 
    config: Optional[ForensicsConfig] = None
) -> MobileDataExtraction:
    """
    Convenience function to extract mobile device data
    
    Args:
        backup_path: Path to mobile device backup directory
        config: Optional configuration for extraction
        
    Returns:
        MobileDataExtraction with complete results
    """
    analyzer = MobileDeviceAnalyzer(config)
    return analyzer.extract_mobile_data(backup_path)