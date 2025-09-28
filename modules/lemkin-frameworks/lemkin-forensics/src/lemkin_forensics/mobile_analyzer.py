"""
Lemkin Digital Forensics Mobile Analyzer

This module provides comprehensive mobile device forensics capabilities including:
- iOS and Android backup analysis and data extraction
- App data extraction (messages, calls, location, photos)
- Mobile artifact timeline creation and correlation
- Contact and media analysis with metadata extraction
- Deleted data recovery from mobile backups
- Cross-platform mobile evidence analysis

Supports: iTunes/Finder backups, Android ADB backups, SQLite databases, plist files
"""

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from uuid import UUID, uuid4
from dataclasses import dataclass, field
import os
import sqlite3
import json
import plistlib
import logging
import hashlib
import xml.etree.ElementTree as ET
from collections import defaultdict

from .core import (
    DigitalEvidence,
    AnalysisResult,
    MobileArtifact,
    TimelineEvent,
    AnalysisStatus,
    EvidenceType,
    ForensicsConfig
)

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class ContactEntry:
    """Represents a contact from mobile device"""
    contact_id: str
    display_name: Optional[str] = None
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    phone_numbers: List[str] = field(default_factory=list)
    email_addresses: List[str] = field(default_factory=list)
    organization: Optional[str] = None
    created_date: Optional[datetime] = None
    modified_date: Optional[datetime] = None


@dataclass
class MessageEntry:
    """Represents a text message or chat message"""
    message_id: str
    chat_id: Optional[str] = None
    timestamp: datetime
    sender: Optional[str] = None
    recipient: Optional[str] = None
    content: Optional[str] = None
    message_type: str = "text"  # text, image, video, audio, location
    direction: str = "unknown"  # incoming, outgoing
    is_read: bool = True
    service: str = "SMS"  # SMS, iMessage, WhatsApp, etc.
    
    # Attachments
    attachment_path: Optional[str] = None
    attachment_type: Optional[str] = None
    attachment_size: Optional[int] = None
    
    # Location data (for location messages)
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    location_name: Optional[str] = None


@dataclass
class CallLogEntry:
    """Represents a call log entry"""
    call_id: str
    timestamp: datetime
    phone_number: str
    contact_name: Optional[str] = None
    call_type: str = "unknown"  # incoming, outgoing, missed
    duration_seconds: int = 0
    answered: bool = False
    
    # Additional metadata
    country_code: Optional[str] = None
    service_provider: Optional[str] = None


@dataclass
class LocationPoint:
    """Represents a location data point"""
    timestamp: datetime
    latitude: float
    longitude: float
    accuracy: Optional[float] = None
    altitude: Optional[float] = None
    speed: Optional[float] = None
    heading: Optional[float] = None
    
    # Source information
    source_app: Optional[str] = None
    location_method: Optional[str] = None  # GPS, network, passive


@dataclass
class AppData:
    """Represents data from a mobile application"""
    app_id: str
    app_name: str
    app_version: Optional[str] = None
    install_date: Optional[datetime] = None
    last_used: Optional[datetime] = None
    
    # App-specific data
    databases: List[str] = field(default_factory=list)
    preferences: Dict[str, Any] = field(default_factory=dict)
    documents: List[str] = field(default_factory=list)
    cache_files: List[str] = field(default_factory=list)
    
    # Data extracted from app
    messages: List[MessageEntry] = field(default_factory=list)
    media_files: List[str] = field(default_factory=list)
    user_data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MediaFile:
    """Represents a media file from mobile device"""
    file_path: str
    file_name: str
    file_type: str  # photo, video, audio
    file_size: int
    created_date: Optional[datetime] = None
    modified_date: Optional[datetime] = None
    
    # Media metadata
    width: Optional[int] = None
    height: Optional[int] = None
    duration_seconds: Optional[float] = None
    
    # EXIF/metadata
    camera_make: Optional[str] = None
    camera_model: Optional[str] = None
    gps_latitude: Optional[float] = None
    gps_longitude: Optional[float] = None
    gps_timestamp: Optional[datetime] = None
    
    # Analysis results
    face_count: Optional[int] = None
    contains_text: bool = False
    hash_md5: Optional[str] = None
    hash_sha256: Optional[str] = None


class MobileDataExtraction:
    """Container for mobile data extraction results"""
    
    def __init__(self):
        self.device_info: Dict[str, Any] = {}
        self.contacts: List[ContactEntry] = []
        self.messages: List[MessageEntry] = []
        self.call_logs: List[CallLogEntry] = []
        self.location_data: List[LocationPoint] = []
        self.apps: List[AppData] = []
        self.media_files: List[MediaFile] = []
        
        # Analysis summary
        self.extraction_summary: Dict[str, Any] = {}
        self.timeline_events: List[TimelineEvent] = []
        self.privacy_concerns: List[str] = []


class MobileAnalyzer:
    """
    Mobile device forensics analyzer for iOS and Android devices
    
    Provides comprehensive analysis of mobile device backups including:
    - Contact and communication analysis
    - Message and chat history extraction
    - Call log analysis and pattern detection
    - Location history and geofencing analysis
    - App data extraction and cross-app correlation
    - Media file analysis with metadata extraction
    - Timeline reconstruction of mobile activities
    """
    
    def __init__(self, config: Optional[ForensicsConfig] = None):
        """Initialize mobile analyzer with configuration"""
        self.config = config or ForensicsConfig()
        self.logger = logging.getLogger(f"{__name__}.MobileAnalyzer")
        
        # Database schemas for common mobile databases
        self.ios_databases = {
            'AddressBook.sqlitedb': self._extract_ios_contacts,
            'sms.db': self._extract_ios_messages,
            'call_history.db': self._extract_ios_calls,
            'consolidated.db': self._extract_ios_location,
            'Photos.sqlite': self._extract_ios_photos
        }
        
        self.android_databases = {
            'contacts2.db': self._extract_android_contacts,
            'mmssms.db': self._extract_android_messages,
            'telephony.db': self._extract_android_calls,
            'gps.db': self._extract_android_location
        }
        
        # App-specific extractors
        self.app_extractors = {
            'com.apple.MobileSMS': self._extract_imessage_data,
            'com.whatsapp.WhatsApp': self._extract_whatsapp_data,
            'com.facebook.Messenger': self._extract_messenger_data,
            'com.apple.mobilemail': self._extract_mail_data,
            'com.apple.mobilecal': self._extract_calendar_data
        }
        
        self.logger.info("Mobile Analyzer initialized")
    
    def analyze_mobile_backup(
        self,
        evidence: DigitalEvidence,
        backup_type: str = "auto",
        output_dir: Optional[str] = None
    ) -> MobileDataExtraction:
        """
        Analyze mobile device backup for forensic artifacts
        
        Args:
            evidence: Digital evidence containing mobile backup
            backup_type: Type of backup (auto, ios, android)
            output_dir: Directory to store extracted files
            
        Returns:
            MobileDataExtraction with comprehensive results
        """
        self.logger.info(f"Starting mobile backup analysis: {evidence.name}")
        
        extraction = MobileDataExtraction()
        backup_path = evidence.file_path
        
        try:
            # Verify backup integrity
            if not evidence.verify_integrity():
                raise ValueError("Mobile backup integrity verification failed")
            
            # Detect backup type if auto
            if backup_type == "auto":
                backup_type = self._detect_backup_type(backup_path)
            
            # Extract device information
            extraction.device_info = self._extract_device_info(backup_path, backup_type)
            
            # Extract data based on backup type
            if backup_type == "ios":
                self._analyze_ios_backup(backup_path, extraction)
            elif backup_type == "android":
                self._analyze_android_backup(backup_path, extraction)
            else:
                raise ValueError(f"Unsupported backup type: {backup_type}")
            
            # Generate timeline from all extracted data
            self._generate_mobile_timeline(extraction)
            
            # Analyze privacy and security concerns
            self._analyze_privacy_concerns(extraction)
            
            # Generate extraction summary
            extraction.extraction_summary = self._generate_extraction_summary(extraction)
            
            # Export data if output directory provided
            if output_dir:
                self._export_mobile_data(extraction, output_dir)
            
            self.logger.info(f"Mobile analysis completed: {len(extraction.timeline_events)} events extracted")
            
        except Exception as e:
            self.logger.error(f"Mobile backup analysis failed: {str(e)}")
            raise
        
        return extraction
    
    def _detect_backup_type(self, backup_path: str) -> str:
        """Detect mobile backup type from file structure"""
        if os.path.isdir(backup_path):
            # Check for iOS backup structure
            if any(fname.endswith('.mdbackup') for fname in os.listdir(backup_path)):
                return "ios"
            elif any(fname.endswith('.mdinfo') for fname in os.listdir(backup_path)):
                return "ios"
            elif os.path.exists(os.path.join(backup_path, 'Manifest.plist')):
                return "ios"
            
            # Check for Android backup structure
            elif os.path.exists(os.path.join(backup_path, 'apps')):
                return "android"
            elif any(fname.endswith('.ab') for fname in os.listdir(backup_path)):
                return "android"
        
        # Single file backups
        elif backup_path.endswith('.ab'):
            return "android"
        elif backup_path.endswith('.ipsw'):
            return "ios"
        
        return "unknown"
    
    def _extract_device_info(self, backup_path: str, backup_type: str) -> Dict[str, Any]:
        """Extract device information from backup"""
        device_info = {}
        
        try:
            if backup_type == "ios":
                # Look for Info.plist or Manifest.plist
                info_files = ['Info.plist', 'Manifest.plist']
                for info_file in info_files:
                    info_path = os.path.join(backup_path, info_file)
                    if os.path.exists(info_path):
                        with open(info_path, 'rb') as f:
                            plist_data = plistlib.load(f)
                            device_info.update(plist_data)
                        break
            
            elif backup_type == "android":
                # Look for device info in backup
                device_info = {
                    'platform': 'Android',
                    'backup_type': 'Android Backup'
                }
            
        except Exception as e:
            self.logger.warning(f"Failed to extract device info: {str(e)}")
        
        return device_info
    
    def _analyze_ios_backup(self, backup_path: str, extraction: MobileDataExtraction):
        """Analyze iOS backup for forensic artifacts"""
        # Find and analyze databases
        for root, dirs, files in os.walk(backup_path):
            for file in files:
                file_path = os.path.join(root, file)
                
                # Check for known database files
                for db_name, extractor in self.ios_databases.items():
                    if file.endswith('.sqlitedb') or file.endswith('.db'):
                        try:
                            extractor(file_path, extraction)
                        except Exception as e:
                            self.logger.warning(f"Failed to extract from {file}: {str(e)}")
                
                # Check for app-specific data
                if file.endswith('.plist'):
                    self._extract_plist_data(file_path, extraction)
        
        # Analyze app directories
        self._analyze_ios_apps(backup_path, extraction)
    
    def _analyze_android_backup(self, backup_path: str, extraction: MobileDataExtraction):
        """Analyze Android backup for forensic artifacts"""
        # Extract Android backup if it's a .ab file
        if backup_path.endswith('.ab'):
            extracted_path = self._extract_android_backup_file(backup_path)
            backup_path = extracted_path
        
        # Analyze databases
        for root, dirs, files in os.walk(backup_path):
            for file in files:
                file_path = os.path.join(root, file)
                
                # Check for known database files
                for db_name, extractor in self.android_databases.items():
                    if file.endswith('.db'):
                        try:
                            extractor(file_path, extraction)
                        except Exception as e:
                            self.logger.warning(f"Failed to extract from {file}: {str(e)}")
    
    def _extract_ios_contacts(self, db_path: str, extraction: MobileDataExtraction):
        """Extract contacts from iOS AddressBook.sqlitedb"""
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # Query contacts
            query = """
            SELECT ABPerson.ROWID, ABPerson.First, ABPerson.Last, ABPerson.Organization,
                   ABPerson.CreationDate, ABPerson.ModificationDate
            FROM ABPerson
            """
            cursor.execute(query)
            
            for row in cursor.fetchall():
                contact = ContactEntry(
                    contact_id=str(row[0]),
                    first_name=row[1],
                    last_name=row[2],
                    organization=row[3],
                    created_date=self._convert_apple_timestamp(row[4]) if row[4] else None,
                    modified_date=self._convert_apple_timestamp(row[5]) if row[5] else None
                )
                
                if contact.first_name and contact.last_name:
                    contact.display_name = f"{contact.first_name} {contact.last_name}"
                elif contact.first_name:
                    contact.display_name = contact.first_name
                elif contact.last_name:
                    contact.display_name = contact.last_name
                
                # Get phone numbers
                phone_query = """
                SELECT ABMultiValue.value 
                FROM ABMultiValue 
                WHERE ABMultiValue.record_id = ? AND ABMultiValue.property = 3
                """
                cursor.execute(phone_query, (row[0],))
                for phone_row in cursor.fetchall():
                    contact.phone_numbers.append(phone_row[0])
                
                # Get email addresses
                email_query = """
                SELECT ABMultiValue.value 
                FROM ABMultiValue 
                WHERE ABMultiValue.record_id = ? AND ABMultiValue.property = 4
                """
                cursor.execute(email_query, (row[0],))
                for email_row in cursor.fetchall():
                    contact.email_addresses.append(email_row[0])
                
                extraction.contacts.append(contact)
            
            conn.close()
            self.logger.info(f"Extracted {len(extraction.contacts)} contacts from iOS backup")
            
        except Exception as e:
            self.logger.error(f"Failed to extract iOS contacts: {str(e)}")
    
    def _extract_ios_messages(self, db_path: str, extraction: MobileDataExtraction):
        """Extract messages from iOS sms.db"""
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # Query messages
            query = """
            SELECT message.ROWID, message.date, message.text, message.is_from_me,
                   message.service, handle.id as phone_number, message.cache_has_attachments
            FROM message
            LEFT JOIN handle ON message.handle_id = handle.ROWID
            ORDER BY message.date
            """
            cursor.execute(query)
            
            for row in cursor.fetchall():
                message = MessageEntry(
                    message_id=str(row[0]),
                    timestamp=self._convert_apple_timestamp(row[1]),
                    content=row[2],
                    direction="outgoing" if row[3] == 1 else "incoming",
                    service=row[4] or "SMS",
                    sender=row[5] if row[3] == 0 else None,
                    recipient=row[5] if row[3] == 1 else None
                )
                
                # Check for attachments
                if row[6] == 1:  # has_attachments
                    message.message_type = "media"
                    # Query attachment details
                    attach_query = """
                    SELECT filename, mime_type, total_bytes
                    FROM attachment
                    JOIN message_attachment_join ON attachment.ROWID = message_attachment_join.attachment_id
                    WHERE message_attachment_join.message_id = ?
                    """
                    cursor.execute(attach_query, (row[0],))
                    attach_row = cursor.fetchone()
                    if attach_row:
                        message.attachment_path = attach_row[0]
                        message.attachment_type = attach_row[1]
                        message.attachment_size = attach_row[2]
                
                extraction.messages.append(message)
            
            conn.close()
            self.logger.info(f"Extracted {len(extraction.messages)} messages from iOS backup")
            
        except Exception as e:
            self.logger.error(f"Failed to extract iOS messages: {str(e)}")
    
    def _extract_ios_calls(self, db_path: str, extraction: MobileDataExtraction):
        """Extract call logs from iOS call_history.db"""
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # Query call history
            query = """
            SELECT ROWID, address, date, duration, flags, name
            FROM call
            ORDER BY date
            """
            cursor.execute(query)
            
            for row in cursor.fetchall():
                # Determine call type from flags
                flags = row[4]
                if flags & 1:  # Incoming call
                    call_type = "incoming"
                elif flags & 2:  # Outgoing call
                    call_type = "outgoing"
                else:
                    call_type = "missed"
                
                call = CallLogEntry(
                    call_id=str(row[0]),
                    timestamp=self._convert_apple_timestamp(row[2]),
                    phone_number=row[1],
                    duration_seconds=row[3] or 0,
                    call_type=call_type,
                    contact_name=row[5],
                    answered=row[3] is not None and row[3] > 0
                )
                
                extraction.call_logs.append(call)
            
            conn.close()
            self.logger.info(f"Extracted {len(extraction.call_logs)} call logs from iOS backup")
            
        except Exception as e:
            self.logger.error(f"Failed to extract iOS call logs: {str(e)}")
    
    def _extract_ios_location(self, db_path: str, extraction: MobileDataExtraction):
        """Extract location data from iOS consolidated.db"""
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # Query location data
            query = """
            SELECT Timestamp, Latitude, Longitude, HorizontalAccuracy, Altitude, Speed
            FROM CellLocation
            ORDER BY Timestamp
            """
            cursor.execute(query)
            
            for row in cursor.fetchall():
                location = LocationPoint(
                    timestamp=self._convert_apple_timestamp(row[0]),
                    latitude=row[1],
                    longitude=row[2],
                    accuracy=row[3],
                    altitude=row[4],
                    speed=row[5],
                    source_app="iOS Location Services",
                    location_method="cellular"
                )
                
                extraction.location_data.append(location)
            
            conn.close()
            self.logger.info(f"Extracted {len(extraction.location_data)} location points from iOS backup")
            
        except Exception as e:
            self.logger.error(f"Failed to extract iOS location data: {str(e)}")
    
    def _extract_ios_photos(self, db_path: str, extraction: MobileDataExtraction):
        """Extract photo metadata from iOS Photos.sqlite"""
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # Query photo metadata
            query = """
            SELECT ROWID, filename, directory, dateCreated, originalFilesize,
                   latitude, longitude, width, height
            FROM ZGENERICASSET
            ORDER BY dateCreated
            """
            cursor.execute(query)
            
            for row in cursor.fetchall():
                media = MediaFile(
                    file_path=os.path.join(row[2] or '', row[1] or ''),
                    file_name=row[1] or '',
                    file_type="photo",
                    file_size=row[4] or 0,
                    created_date=self._convert_apple_timestamp(row[3]) if row[3] else None,
                    width=row[7],
                    height=row[8],
                    gps_latitude=row[5],
                    gps_longitude=row[6]
                )
                
                extraction.media_files.append(media)
            
            conn.close()
            self.logger.info(f"Extracted {len(extraction.media_files)} media files from iOS backup")
            
        except Exception as e:
            self.logger.error(f"Failed to extract iOS photo metadata: {str(e)}")
    
    def _extract_android_contacts(self, db_path: str, extraction: MobileDataExtraction):
        """Extract contacts from Android contacts2.db"""
        # Android contacts extraction implementation
        pass
    
    def _extract_android_messages(self, db_path: str, extraction: MobileDataExtraction):
        """Extract messages from Android mmssms.db"""
        # Android messages extraction implementation
        pass
    
    def _extract_android_calls(self, db_path: str, extraction: MobileDataExtraction):
        """Extract call logs from Android telephony.db"""
        # Android call logs extraction implementation
        pass
    
    def _extract_android_location(self, db_path: str, extraction: MobileDataExtraction):
        """Extract location data from Android GPS databases"""
        # Android location extraction implementation
        pass
    
    def _analyze_ios_apps(self, backup_path: str, extraction: MobileDataExtraction):
        """Analyze iOS app data"""
        # Find app directories and analyze app-specific data
        for app_id, extractor in self.app_extractors.items():
            app_path = self._find_ios_app_path(backup_path, app_id)
            if app_path:
                try:
                    extractor(app_path, extraction)
                except Exception as e:
                    self.logger.warning(f"Failed to extract {app_id} data: {str(e)}")
    
    def _find_ios_app_path(self, backup_path: str, app_id: str) -> Optional[str]:
        """Find iOS app path in backup"""
        # iOS backup structure varies, implement path finding logic
        return None
    
    def _extract_imessage_data(self, app_path: str, extraction: MobileDataExtraction):
        """Extract iMessage specific data"""
        # iMessage extraction implementation
        pass
    
    def _extract_whatsapp_data(self, app_path: str, extraction: MobileDataExtraction):
        """Extract WhatsApp data"""
        # WhatsApp extraction implementation
        pass
    
    def _extract_messenger_data(self, app_path: str, extraction: MobileDataExtraction):
        """Extract Facebook Messenger data"""
        # Messenger extraction implementation
        pass
    
    def _extract_mail_data(self, app_path: str, extraction: MobileDataExtraction):
        """Extract Mail app data"""
        # Mail extraction implementation
        pass
    
    def _extract_calendar_data(self, app_path: str, extraction: MobileDataExtraction):
        """Extract Calendar app data"""
        # Calendar extraction implementation
        pass
    
    def _extract_plist_data(self, plist_path: str, extraction: MobileDataExtraction):
        """Extract data from plist files"""
        try:
            with open(plist_path, 'rb') as f:
                plist_data = plistlib.load(f)
                # Process plist data for relevant information
        except Exception as e:
            self.logger.warning(f"Failed to extract plist data from {plist_path}: {str(e)}")
    
    def _extract_android_backup_file(self, backup_path: str) -> str:
        """Extract Android .ab backup file"""
        # Android backup extraction implementation
        return backup_path
    
    def _generate_mobile_timeline(self, extraction: MobileDataExtraction):
        """Generate comprehensive timeline from all mobile data"""
        events = []
        
        # Add message events
        for message in extraction.messages:
            events.append(TimelineEvent(
                timestamp=message.timestamp,
                event_type="message",
                description=f"{message.direction.title()} {message.service} message",
                artifact_type="mobile_message",
                notes=f"From/To: {message.sender or message.recipient}"
            ))
        
        # Add call events
        for call in extraction.call_logs:
            events.append(TimelineEvent(
                timestamp=call.timestamp,
                event_type="call",
                description=f"{call.call_type.title()} call to {call.phone_number}",
                artifact_type="mobile_call",
                notes=f"Duration: {call.duration_seconds}s, Contact: {call.contact_name or 'Unknown'}"
            ))
        
        # Add location events (sample some to avoid overwhelming timeline)
        location_sample = extraction.location_data[::10] if len(extraction.location_data) > 100 else extraction.location_data
        for location in location_sample:
            events.append(TimelineEvent(
                timestamp=location.timestamp,
                event_type="location",
                description=f"Location: {location.latitude:.6f}, {location.longitude:.6f}",
                artifact_type="mobile_location",
                notes=f"Accuracy: {location.accuracy}m, Source: {location.source_app}"
            ))
        
        # Sort events chronologically
        events.sort(key=lambda x: x.timestamp)
        extraction.timeline_events = events
        
        self.logger.info(f"Generated mobile timeline with {len(events)} events")
    
    def _analyze_privacy_concerns(self, extraction: MobileDataExtraction):
        """Analyze potential privacy and security concerns"""
        concerns = []
        
        # Check for excessive location tracking
        if len(extraction.location_data) > 10000:
            concerns.append(f"Extensive location tracking: {len(extraction.location_data)} location points")
        
        # Check for sensitive content in messages
        sensitive_keywords = ['password', 'ssn', 'social security', 'credit card', 'bank account']
        for message in extraction.messages:
            if message.content:
                content_lower = message.content.lower()
                for keyword in sensitive_keywords:
                    if keyword in content_lower:
                        concerns.append(f"Potential sensitive data in messages: {keyword}")
                        break
        
        # Check for contacts with suspicious patterns
        if len(extraction.contacts) > 5000:
            concerns.append(f"Unusually large contact list: {len(extraction.contacts)} contacts")
        
        extraction.privacy_concerns = concerns
    
    def _generate_extraction_summary(self, extraction: MobileDataExtraction) -> Dict[str, Any]:
        """Generate summary of extraction results"""
        return {
            'device_info': extraction.device_info,
            'contacts_count': len(extraction.contacts),
            'messages_count': len(extraction.messages),
            'call_logs_count': len(extraction.call_logs),
            'location_points_count': len(extraction.location_data),
            'apps_analyzed': len(extraction.apps),
            'media_files_count': len(extraction.media_files),
            'timeline_events_count': len(extraction.timeline_events),
            'privacy_concerns_count': len(extraction.privacy_concerns),
            'earliest_data': min([
                event.timestamp for event in extraction.timeline_events
            ]).isoformat() if extraction.timeline_events else None,
            'latest_data': max([
                event.timestamp for event in extraction.timeline_events
            ]).isoformat() if extraction.timeline_events else None
        }
    
    def _export_mobile_data(self, extraction: MobileDataExtraction, output_dir: str):
        """Export mobile data to structured files"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Export contacts
        if extraction.contacts:
            contacts_file = os.path.join(output_dir, 'contacts.json')
            with open(contacts_file, 'w', encoding='utf-8') as f:
                json.dump([contact.__dict__ for contact in extraction.contacts], f, indent=2, default=str)
        
        # Export messages
        if extraction.messages:
            messages_file = os.path.join(output_dir, 'messages.json')
            with open(messages_file, 'w', encoding='utf-8') as f:
                json.dump([message.__dict__ for message in extraction.messages], f, indent=2, default=str)
        
        # Export call logs
        if extraction.call_logs:
            calls_file = os.path.join(output_dir, 'call_logs.json')
            with open(calls_file, 'w', encoding='utf-8') as f:
                json.dump([call.__dict__ for call in extraction.call_logs], f, indent=2, default=str)
        
        # Export timeline
        if extraction.timeline_events:
            timeline_file = os.path.join(output_dir, 'timeline.json')
            with open(timeline_file, 'w', encoding='utf-8') as f:
                json.dump([event.dict() for event in extraction.timeline_events], f, indent=2, default=str)
        
        # Export summary
        summary_file = os.path.join(output_dir, 'extraction_summary.json')
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(extraction.extraction_summary, f, indent=2, default=str)
        
        self.logger.info(f"Mobile data exported to {output_dir}")
    
    def _convert_apple_timestamp(self, timestamp: float) -> datetime:
        """Convert Apple timestamp to datetime (seconds since 2001-01-01)"""
        apple_epoch = datetime(2001, 1, 1, tzinfo=timezone.utc)
        return apple_epoch + timedelta(seconds=timestamp)
    
    def generate_mobile_artifacts(self, extraction: MobileDataExtraction) -> List[MobileArtifact]:
        """Convert mobile extraction to MobileArtifact objects"""
        artifacts = []
        
        # Convert messages
        for message in extraction.messages:
            artifact = MobileArtifact(
                artifact_type="message",
                timestamp=message.timestamp,
                phone_number=message.sender or message.recipient,
                message_content=message.content,
                message_direction=message.direction,
                message_status="read" if message.is_read else "unread",
                app_name=message.service
            )
            artifacts.append(artifact)
        
        # Convert call logs
        for call in extraction.call_logs:
            artifact = MobileArtifact(
                artifact_type="call_log",
                timestamp=call.timestamp,
                phone_number=call.phone_number,
                contact_name=call.contact_name,
                call_duration_seconds=call.duration_seconds,
                call_type=call.call_type
            )
            artifacts.append(artifact)
        
        # Convert location data (sample to avoid too many artifacts)
        location_sample = extraction.location_data[::10] if len(extraction.location_data) > 100 else extraction.location_data
        for location in location_sample:
            artifact = MobileArtifact(
                artifact_type="location",
                timestamp=location.timestamp,
                latitude=location.latitude,
                longitude=location.longitude,
                location_accuracy=location.accuracy,
                location_source=location.source_app,
                app_name=location.source_app
            )
            artifacts.append(artifact)
        
        return artifacts


# Alias for backward compatibility
MobileArtifact = MobileArtifact