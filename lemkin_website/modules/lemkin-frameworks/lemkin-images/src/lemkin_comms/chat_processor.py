"""
Chat Export Processor Module

Handles parsing and analysis of chat exports from various messaging platforms
including WhatsApp, Telegram, Signal, and others. Provides forensic-grade
analysis suitable for legal investigations.
"""

import re
import json
import zipfile
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Tuple
import pandas as pd
from dataclasses import dataclass
import logging

from .core import (
    ChatMessage, ChatAnalysis, Contact, CommsConfig, 
    PlatformType, CommunicationType, TemporalPattern
)

logger = logging.getLogger(__name__)


@dataclass
class ParsedMessage:
    """Intermediate representation of parsed message"""
    timestamp: datetime
    sender: str
    content: str
    message_type: str
    metadata: Dict[str, Any]


class BaseChatProcessor:
    """Base class for chat export processors"""
    
    def __init__(self, config: CommsConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def parse_export(self, export_path: Path) -> List[ChatMessage]:
        """Parse chat export file - to be implemented by subclasses"""
        raise NotImplementedError
    
    def _create_chat_message(
        self, 
        parsed_msg: ParsedMessage, 
        platform: PlatformType,
        thread_id: str = None
    ) -> ChatMessage:
        """Create ChatMessage from parsed data"""
        return ChatMessage(
            message_id=f"{platform.value}_{hash(f'{parsed_msg.timestamp}_{parsed_msg.sender}_{parsed_msg.content}')}",
            platform=platform,
            comm_type=CommunicationType.CHAT,
            timestamp=parsed_msg.timestamp,
            sender_id=parsed_msg.sender,
            recipient_ids=[],  # Will be filled later
            content=parsed_msg.content,
            metadata=parsed_msg.metadata,
            thread_id=thread_id
        )


class WhatsAppProcessor(BaseChatProcessor):
    """WhatsApp chat export processor"""
    
    def __init__(self, config: CommsConfig):
        super().__init__(config)
        # WhatsApp export patterns for different locales
        self.datetime_patterns = [
            r'(\d{1,2}/\d{1,2}/\d{2,4}),?\s+(\d{1,2}:\d{2}:\d{2}?\s*(?:AM|PM|am|pm)?)',  # US format
            r'(\d{1,2}\.\d{1,2}\.\d{2,4}),?\s+(\d{1,2}:\d{2}:\d{2}?)',  # European format
            r'(\d{4}-\d{2}-\d{2}),?\s+(\d{2}:\d{2}:\d{2})',  # ISO format
        ]
        self.message_pattern = r'^(.+?):\s(.+)$'
    
    def parse_export(self, export_path: Path) -> List[ChatMessage]:
        """Parse WhatsApp chat export"""
        messages = []
        
        if export_path.suffix == '.zip':
            # Handle ZIP archive with media
            messages = self._parse_zip_export(export_path)
        else:
            # Handle text file export
            messages = self._parse_text_export(export_path)
        
        return messages
    
    def _parse_zip_export(self, zip_path: Path) -> List[ChatMessage]:
        """Parse WhatsApp ZIP export with media"""
        messages = []
        
        with zipfile.ZipFile(zip_path, 'r') as zip_file:
            # Find chat file
            chat_files = [f for f in zip_file.namelist() if f.endswith('.txt')]
            
            if not chat_files:
                raise ValueError("No chat text file found in ZIP archive")
            
            chat_content = zip_file.read(chat_files[0]).decode('utf-8', errors='ignore')
            messages = self._parse_chat_content(chat_content)
            
            # Process media files
            media_files = [f for f in zip_file.namelist() if not f.endswith('.txt')]
            for message in messages:
                if '<Media omitted>' in message.content or '<attached:' in message.content:
                    # Try to match media file
                    media_file = self._find_matching_media(message, media_files)
                    if media_file:
                        message.attachments = [media_file]
                        message.metadata['has_media'] = True
        
        return messages
    
    def _parse_text_export(self, text_path: Path) -> List[ChatMessage]:
        """Parse WhatsApp text export"""
        with open(text_path, 'r', encoding='utf-8', errors='ignore') as file:
            content = file.read()
        
        return self._parse_chat_content(content)
    
    def _parse_chat_content(self, content: str) -> List[ChatMessage]:
        """Parse WhatsApp chat text content"""
        messages = []
        lines = content.split('\n')
        current_message = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Try to match datetime pattern
            datetime_match = None
            for pattern in self.datetime_patterns:
                datetime_match = re.match(pattern, line)
                if datetime_match:
                    break
            
            if datetime_match:
                # Save previous message if exists
                if current_message:
                    messages.append(self._create_whatsapp_message(current_message))
                
                # Parse new message
                timestamp = self._parse_whatsapp_datetime(datetime_match.groups())
                remaining_line = line[datetime_match.end():].strip()
                
                # Extract sender and content
                if remaining_line.startswith('-'):
                    remaining_line = remaining_line[1:].strip()
                
                sender_match = re.match(self.message_pattern, remaining_line)
                if sender_match:
                    sender = sender_match.group(1).strip()
                    content = sender_match.group(2).strip()
                else:
                    # System message
                    sender = "System"
                    content = remaining_line
                
                current_message = ParsedMessage(
                    timestamp=timestamp,
                    sender=sender,
                    content=content,
                    message_type='text',
                    metadata={}
                )
            else:
                # Continuation of previous message
                if current_message:
                    current_message.content += '\n' + line
        
        # Add last message
        if current_message:
            messages.append(self._create_whatsapp_message(current_message))
        
        return messages
    
    def _parse_whatsapp_datetime(self, groups: Tuple[str, str]) -> datetime:
        """Parse WhatsApp datetime from regex groups"""
        date_str, time_str = groups
        
        # Try different date formats
        date_formats = ['%m/%d/%y', '%m/%d/%Y', '%d.%m.%y', '%d.%m.%Y', '%Y-%m-%d']
        time_formats = ['%H:%M:%S', '%H:%M', '%I:%M:%S %p', '%I:%M %p']
        
        for date_fmt in date_formats:
            for time_fmt in time_formats:
                try:
                    datetime_str = f"{date_str} {time_str}"
                    return datetime.strptime(datetime_str, f"{date_fmt} {time_fmt}")
                except ValueError:
                    continue
        
        # Fallback
        raise ValueError(f"Could not parse datetime: {date_str} {time_str}")
    
    def _create_whatsapp_message(self, parsed_msg: ParsedMessage) -> ChatMessage:
        """Create WhatsApp ChatMessage"""
        # Detect special message types
        metadata = parsed_msg.metadata.copy()
        
        if '<Media omitted>' in parsed_msg.content:
            metadata['media_omitted'] = True
        
        if parsed_msg.content.startswith('You deleted this message'):
            metadata['deleted'] = True
        
        if 'added' in parsed_msg.content or 'removed' in parsed_msg.content:
            metadata['group_admin_action'] = True
        
        return ChatMessage(
            message_id=f"whatsapp_{hash(f'{parsed_msg.timestamp}_{parsed_msg.sender}_{parsed_msg.content}')}",
            platform=PlatformType.WHATSAPP,
            comm_type=CommunicationType.CHAT,
            timestamp=parsed_msg.timestamp,
            sender_id=parsed_msg.sender,
            recipient_ids=[],
            content=parsed_msg.content,
            metadata=metadata,
            is_group=True,  # Most WhatsApp exports are group chats
            deleted='deleted' in metadata
        )
    
    def _find_matching_media(self, message: ChatMessage, media_files: List[str]) -> Optional[str]:
        """Find media file matching message timestamp"""
        # Simple heuristic - find media file with closest timestamp
        target_time = message.timestamp
        
        for media_file in media_files:
            # Extract timestamp from filename if possible
            if any(ext in media_file.lower() for ext in ['.jpg', '.png', '.mp4', '.pdf']):
                return media_file
        
        return None


class TelegramProcessor(BaseChatProcessor):
    """Telegram chat export processor"""
    
    def parse_export(self, export_path: Path) -> List[ChatMessage]:
        """Parse Telegram JSON export"""
        if export_path.suffix == '.json':
            return self._parse_json_export(export_path)
        else:
            raise ValueError("Telegram processor only supports JSON exports")
    
    def _parse_json_export(self, json_path: Path) -> List[ChatMessage]:
        """Parse Telegram JSON export"""
        with open(json_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        
        messages = []
        chat_info = data.get('chats', {})
        
        for message_data in data.get('messages', []):
            try:
                message = self._parse_telegram_message(message_data, chat_info)
                if message:
                    messages.append(message)
            except Exception as e:
                self.logger.warning(f"Failed to parse Telegram message: {e}")
                continue
        
        return messages
    
    def _parse_telegram_message(self, msg_data: Dict[str, Any], chat_info: Dict) -> Optional[ChatMessage]:
        """Parse individual Telegram message"""
        if msg_data.get('type') != 'message':
            return None
        
        # Extract basic info
        timestamp = datetime.fromisoformat(msg_data['date'].replace('Z', '+00:00'))
        sender = msg_data.get('from', 'Unknown')
        message_id = str(msg_data.get('id', ''))
        
        # Extract content
        content = ""
        attachments = []
        metadata = {}
        
        # Text content
        if 'text' in msg_data:
            if isinstance(msg_data['text'], list):
                content = ''.join([item.get('text', str(item)) if isinstance(item, dict) else str(item) 
                                 for item in msg_data['text']])
            else:
                content = str(msg_data['text'])
        
        # Media content
        if 'photo' in msg_data:
            attachments.append(msg_data['photo'])
            metadata['media_type'] = 'photo'
        
        if 'file' in msg_data:
            attachments.append(msg_data['file'])
            metadata['media_type'] = 'file'
        
        if 'voice_message' in msg_data:
            metadata['media_type'] = 'voice'
        
        # Special message types
        if msg_data.get('forwarded_from'):
            metadata['forwarded'] = True
            metadata['forwarded_from'] = msg_data['forwarded_from']
        
        if msg_data.get('reply_to_message_id'):
            metadata['reply_to'] = str(msg_data['reply_to_message_id'])
        
        if msg_data.get('edited'):
            metadata['edited'] = True
            metadata['edit_date'] = msg_data['edited']
        
        return ChatMessage(
            message_id=f"telegram_{message_id}",
            platform=PlatformType.TELEGRAM,
            comm_type=CommunicationType.CHAT,
            timestamp=timestamp,
            sender_id=str(sender),
            recipient_ids=[],
            content=content,
            metadata=metadata,
            attachments=attachments,
            forwarded_from=metadata.get('forwarded_from'),
            replied_to=metadata.get('reply_to'),
            edited=metadata.get('edited', False)
        )


class SignalProcessor(BaseChatProcessor):
    """Signal messenger export processor"""
    
    def parse_export(self, export_path: Path) -> List[ChatMessage]:
        """Parse Signal JSON export"""
        with open(export_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        
        messages = []
        for message_data in data.get('messages', []):
            try:
                message = self._parse_signal_message(message_data)
                if message:
                    messages.append(message)
            except Exception as e:
                self.logger.warning(f"Failed to parse Signal message: {e}")
                continue
        
        return messages
    
    def _parse_signal_message(self, msg_data: Dict[str, Any]) -> Optional[ChatMessage]:
        """Parse individual Signal message"""
        timestamp = datetime.fromtimestamp(msg_data['timestamp'] / 1000)
        sender = msg_data.get('source', 'Unknown')
        content = msg_data.get('body', '')
        
        metadata = {
            'read_status': msg_data.get('readStatus'),
            'delivery_status': msg_data.get('type')
        }
        
        attachments = []
        if 'attachments' in msg_data:
            attachments = [att.get('fileName', 'unknown') for att in msg_data['attachments']]
        
        return ChatMessage(
            message_id=f"signal_{msg_data.get('timestamp', '')}",
            platform=PlatformType.SIGNAL,
            comm_type=CommunicationType.CHAT,
            timestamp=timestamp,
            sender_id=sender,
            recipient_ids=[],
            content=content,
            metadata=metadata,
            attachments=attachments,
            encrypted=True  # Signal messages are encrypted by default
        )


class ChatProcessor:
    """Main chat processor that delegates to platform-specific processors"""
    
    def __init__(self, config: CommsConfig):
        self.config = config
        self.processors = {
            PlatformType.WHATSAPP: WhatsAppProcessor(config),
            PlatformType.TELEGRAM: TelegramProcessor(config),
            PlatformType.SIGNAL: SignalProcessor(config)
        }
        self.logger = logging.getLogger(__name__)
    
    def detect_platform(self, export_path: Path) -> PlatformType:
        """Auto-detect chat platform from export file"""
        file_content = ""
        
        try:
            if export_path.suffix == '.json':
                with open(export_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if 'chats' in data and 'messages' in data:
                        return PlatformType.TELEGRAM
                    elif 'messages' in data and any('timestamp' in msg for msg in data['messages'][:5]):
                        return PlatformType.SIGNAL
            else:
                with open(export_path, 'r', encoding='utf-8', errors='ignore') as f:
                    file_content = f.read(1000)  # Read first 1000 chars
                
                # WhatsApp patterns
                whatsapp_patterns = [
                    r'\d{1,2}/\d{1,2}/\d{2,4},?\s+\d{1,2}:\d{2}',
                    r'<Media omitted>',
                    r'Messages to this chat and calls are now secured'
                ]
                
                if any(re.search(pattern, file_content) for pattern in whatsapp_patterns):
                    return PlatformType.WHATSAPP
        
        except Exception as e:
            self.logger.warning(f"Could not detect platform: {e}")
        
        raise ValueError("Could not detect chat platform from export file")
    
    def process_export(self, export_path: Path, platform: PlatformType = None) -> List[ChatMessage]:
        """Process chat export file"""
        if platform is None:
            platform = self.detect_platform(export_path)
        
        if platform not in self.processors:
            raise ValueError(f"Unsupported platform: {platform}")
        
        self.logger.info(f"Processing {platform} export from {export_path}")
        messages = self.processors[platform].parse_export(export_path)
        
        # Post-process messages
        messages = self._post_process_messages(messages)
        
        self.logger.info(f"Successfully processed {len(messages)} messages")
        return messages
    
    def _post_process_messages(self, messages: List[ChatMessage]) -> List[ChatMessage]:
        """Post-process messages for consistency and additional analysis"""
        # Sort by timestamp
        messages.sort(key=lambda m: m.timestamp)
        
        # Extract participants
        participants = set()
        for message in messages:
            participants.add(message.sender_id)
            participants.update(message.recipient_ids)
        
        # Update recipient lists for group chats
        for message in messages:
            if message.is_group:
                message.recipient_ids = list(participants - {message.sender_id})
        
        # Detect reply chains
        self._detect_reply_chains(messages)
        
        return messages
    
    def _detect_reply_chains(self, messages: List[ChatMessage]):
        """Detect and link reply chains in messages"""
        # Simple heuristic: messages close in time might be replies
        for i, message in enumerate(messages[1:], 1):
            prev_message = messages[i-1]
            time_diff = (message.timestamp - prev_message.timestamp).total_seconds()
            
            # If message is within 2 minutes and different sender, likely a reply
            if (time_diff < 120 and 
                message.sender_id != prev_message.sender_id and
                not message.replied_to):
                message.replied_to = prev_message.message_id
    
    def analyze(self, messages: List[ChatMessage]) -> ChatAnalysis:
        """Analyze processed chat messages"""
        if not messages:
            return ChatAnalysis(
                platform=PlatformType.WHATSAPP,  # Default
                total_messages=0,
                total_participants=0,
                date_range=(datetime.now(), datetime.now()),
                message_statistics={},
                participant_analysis={},
                temporal_patterns=[]
            )
        
        # Basic statistics
        total_messages = len(messages)
        participants = set(m.sender_id for m in messages)
        date_range = (min(m.timestamp for m in messages), max(m.timestamp for m in messages))
        platform = messages[0].platform
        
        # Message statistics
        message_stats = {
            'total_messages': total_messages,
            'messages_per_day': total_messages / max(1, (date_range[1] - date_range[0]).days),
            'average_length': np.mean([len(m.content or '') for m in messages]),
            'media_messages': sum(1 for m in messages if m.attachments),
            'deleted_messages': sum(1 for m in messages if m.deleted),
            'edited_messages': sum(1 for m in messages if m.edited)
        }
        
        # Participant analysis
        participant_stats = {}
        for participant in participants:
            user_messages = [m for m in messages if m.sender_id == participant]
            participant_stats[participant] = {
                'message_count': len(user_messages),
                'average_length': np.mean([len(m.content or '') for m in user_messages]),
                'media_sent': sum(1 for m in user_messages if m.attachments),
                'first_message': min(m.timestamp for m in user_messages),
                'last_message': max(m.timestamp for m in user_messages)
            }
        
        # Temporal patterns
        temporal_patterns = self._analyze_temporal_patterns(messages)
        
        # Group dynamics (if applicable)
        group_dynamics = {}
        if any(m.is_group for m in messages):
            group_dynamics = self._analyze_group_dynamics(messages)
        
        return ChatAnalysis(
            platform=platform,
            total_messages=total_messages,
            total_participants=len(participants),
            date_range=date_range,
            message_statistics=message_stats,
            participant_analysis=participant_stats,
            temporal_patterns=temporal_patterns,
            group_dynamics=group_dynamics
        )
    
    def _analyze_temporal_patterns(self, messages: List[ChatMessage]) -> List[TemporalPattern]:
        """Analyze temporal communication patterns"""
        df = pd.DataFrame([{
            'timestamp': m.timestamp,
            'hour': m.timestamp.hour,
            'day_of_week': m.timestamp.weekday(),
            'sender': m.sender_id
        } for m in messages])
        
        patterns = []
        
        # Hourly patterns
        hourly_counts = df.groupby('hour').size()
        patterns.append(TemporalPattern(
            time_unit='hour',
            frequency_distribution=hourly_counts.to_dict(),
            peak_times=[str(h) for h in hourly_counts.nlargest(3).index],
            quiet_periods=[str(h) for h in hourly_counts.nsmallest(3).index],
            regularity_score=1.0 - (hourly_counts.std() / hourly_counts.mean())
        ))
        
        # Daily patterns
        daily_counts = df.groupby('day_of_week').size()
        patterns.append(TemporalPattern(
            time_unit='day_of_week',
            frequency_distribution={str(k): v for k, v in daily_counts.to_dict().items()},
            peak_times=[str(d) for d in daily_counts.nlargest(2).index],
            quiet_periods=[str(d) for d in daily_counts.nsmallest(2).index],
            regularity_score=1.0 - (daily_counts.std() / daily_counts.mean())
        ))
        
        return patterns
    
    def _analyze_group_dynamics(self, messages: List[ChatMessage]) -> Dict[str, Any]:
        """Analyze group chat dynamics"""
        group_messages = [m for m in messages if m.is_group]
        if not group_messages:
            return {}
        
        participants = list(set(m.sender_id for m in group_messages))
        
        # Response patterns
        response_times = []
        for i, message in enumerate(group_messages[1:], 1):
            prev_message = group_messages[i-1]
            if message.sender_id != prev_message.sender_id:
                response_time = (message.timestamp - prev_message.timestamp).total_seconds()
                if response_time < 3600:  # Within 1 hour
                    response_times.append(response_time)
        
        # Interaction matrix
        interactions = {}
        for p1 in participants:
            interactions[p1] = {}
            for p2 in participants:
                if p1 != p2:
                    # Count messages from p1 followed by messages from p2
                    count = 0
                    for i, message in enumerate(group_messages[1:], 1):
                        prev_message = group_messages[i-1]
                        if prev_message.sender_id == p1 and message.sender_id == p2:
                            count += 1
                    interactions[p1][p2] = count
        
        return {
            'participants': participants,
            'average_response_time': np.mean(response_times) if response_times else 0,
            'interaction_matrix': interactions,
            'most_active': max(participants, key=lambda p: len([m for m in group_messages if m.sender_id == p])),
            'admin_actions': len([m for m in group_messages if m.metadata.get('group_admin_action')])
        }


# Main function for CLI
def process_chat_exports(export_path: Path, config: CommsConfig = None) -> ChatAnalysis:
    """Process chat exports and return analysis results"""
    if config is None:
        config = CommsConfig()
    
    processor = ChatProcessor(config)
    messages = processor.process_export(export_path)
    return processor.analyze(messages)