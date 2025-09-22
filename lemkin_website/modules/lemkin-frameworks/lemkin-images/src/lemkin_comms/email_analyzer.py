"""
Email Analysis Module

Handles email thread reconstruction, relationship mapping, and forensic analysis
of email communications. Supports multiple email formats including PST, MBOX,
EML and provides comprehensive email investigation capabilities.
"""

import email
import mailbox
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Set, Tuple
import pandas as pd
import numpy as np
from email.header import decode_header
from email.utils import parseaddr, parsedate_to_datetime
import re
from collections import defaultdict, Counter
import logging

from .core import (
    EmailMessage, EmailAnalysis, Contact, CommsConfig,
    PlatformType, CommunicationType
)

logger = logging.getLogger(__name__)


class EmailThreadReconstructor:
    """Reconstructs email conversation threads from individual messages"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def reconstruct_threads(self, emails: List[EmailMessage]) -> Dict[str, List[EmailMessage]]:
        """Reconstruct email threads using subject lines and message IDs"""
        threads = defaultdict(list)
        subject_groups = defaultdict(list)
        
        # Group by normalized subject
        for email_msg in emails:
            normalized_subject = self._normalize_subject(email_msg.subject)
            subject_groups[normalized_subject].append(email_msg)
        
        # Build threads within each subject group
        thread_id = 0
        for subject, subject_emails in subject_groups.items():
            if len(subject_emails) == 1:
                # Single email thread
                threads[f"thread_{thread_id}"] = subject_emails
                thread_id += 1
            else:
                # Multiple emails - build thread tree
                thread_tree = self._build_thread_tree(subject_emails)
                for i, thread in enumerate(thread_tree):
                    threads[f"thread_{thread_id}_{i}"] = thread
                    thread_id += 1
        
        return dict(threads)
    
    def _normalize_subject(self, subject: str) -> str:
        """Normalize email subject for thread grouping"""
        if not subject:
            return ""
        
        # Remove Re:, Fwd:, etc.
        normalized = re.sub(r'^(Re:|RE:|Fwd:|FWD:|Fw:|FW:)\s*', '', subject, flags=re.IGNORECASE)
        normalized = re.sub(r'\s+', ' ', normalized).strip()
        return normalized.lower()
    
    def _build_thread_tree(self, emails: List[EmailMessage]) -> List[List[EmailMessage]]:
        """Build thread tree from emails with same subject"""
        # Sort by timestamp
        emails.sort(key=lambda e: e.timestamp)
        
        threads = []
        used_emails = set()
        
        for email_msg in emails:
            if email_msg.message_id in used_emails:
                continue
            
            # Start new thread with this email
            thread = [email_msg]
            used_emails.add(email_msg.message_id)
            
            # Find replies in this thread
            self._find_thread_replies(email_msg, emails, thread, used_emails)
            threads.append(thread)
        
        return threads
    
    def _find_thread_replies(
        self, 
        parent_email: EmailMessage, 
        all_emails: List[EmailMessage],
        thread: List[EmailMessage], 
        used_emails: Set[str]
    ):
        """Recursively find replies to parent email"""
        for email_msg in all_emails:
            if (email_msg.message_id in used_emails or 
                email_msg.timestamp <= parent_email.timestamp):
                continue
            
            # Check if this is a reply
            if self._is_reply(parent_email, email_msg):
                thread.append(email_msg)
                used_emails.add(email_msg.message_id)
                
                # Find replies to this email
                self._find_thread_replies(email_msg, all_emails, thread, used_emails)
    
    def _is_reply(self, parent: EmailMessage, candidate: EmailMessage) -> bool:
        """Determine if candidate is a reply to parent"""
        # Check In-Reply-To header
        if candidate.replied_to == parent.message_id:
            return True
        
        # Check if sender/recipient relationship suggests reply
        if (parent.sender_id in candidate.recipient_ids and 
            candidate.sender_id in parent.recipient_ids + [parent.sender_id]):
            return True
        
        # Check time proximity (within reasonable timeframe)
        time_diff = (candidate.timestamp - parent.timestamp).total_seconds()
        if time_diff < 0 or time_diff > 7 * 24 * 3600:  # Within 7 days
            return False
        
        return False


class EmailRelationshipMapper:
    """Maps relationships between email contacts"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def map_relationships(self, emails: List[EmailMessage]) -> Dict[str, Any]:
        """Create comprehensive relationship mapping"""
        relationships = defaultdict(lambda: defaultdict(int))
        contact_info = {}
        communication_patterns = defaultdict(lambda: defaultdict(list))
        
        # Build contact database
        for email_msg in emails:
            # Add sender
            if email_msg.sender_id not in contact_info:
                contact_info[email_msg.sender_id] = {
                    'email': email_msg.sender_id,
                    'first_seen': email_msg.timestamp,
                    'last_seen': email_msg.timestamp,
                    'sent_count': 0,
                    'received_count': 0
                }
            
            contact_info[email_msg.sender_id]['last_seen'] = max(
                contact_info[email_msg.sender_id]['last_seen'], 
                email_msg.timestamp
            )
            contact_info[email_msg.sender_id]['sent_count'] += 1
            
            # Add recipients
            all_recipients = email_msg.recipient_ids + email_msg.cc_recipients + email_msg.bcc_recipients
            
            for recipient in all_recipients:
                if recipient not in contact_info:
                    contact_info[recipient] = {
                        'email': recipient,
                        'first_seen': email_msg.timestamp,
                        'last_seen': email_msg.timestamp,
                        'sent_count': 0,
                        'received_count': 0
                    }
                
                contact_info[recipient]['received_count'] += 1
                
                # Build relationship matrix
                relationships[email_msg.sender_id][recipient] += 1
                
                # Track communication patterns
                communication_patterns[email_msg.sender_id][recipient].append({
                    'timestamp': email_msg.timestamp,
                    'subject': email_msg.subject,
                    'type': 'email'
                })
        
        # Calculate relationship strengths
        relationship_strengths = {}
        for sender, recipients in relationships.items():
            relationship_strengths[sender] = {}
            for recipient, count in recipients.items():
                # Calculate strength based on frequency and recency
                recent_contacts = [
                    c for c in communication_patterns[sender][recipient]
                    if (datetime.now() - c['timestamp']).days < 30
                ]
                
                strength = min(1.0, count / 10.0)  # Normalize to 0-1
                if recent_contacts:
                    strength *= 1.5  # Boost for recent activity
                
                relationship_strengths[sender][recipient] = min(1.0, strength)
        
        return {
            'contacts': contact_info,
            'relationships': dict(relationships),
            'relationship_strengths': relationship_strengths,
            'communication_patterns': dict(communication_patterns)
        }
    
    def identify_key_figures(self, relationship_map: Dict[str, Any]) -> Dict[str, List[str]]:
        """Identify key figures in email communications"""
        contacts = relationship_map['contacts']
        relationships = relationship_map['relationships']
        
        # Calculate centrality metrics
        contact_metrics = {}
        for contact_id, contact_info in contacts.items():
            sent = contact_info['sent_count']
            received = contact_info['received_count']
            unique_contacts = len(set(
                list(relationships.get(contact_id, {}).keys()) +
                [sender for sender, recipients in relationships.items() 
                 if contact_id in recipients]
            ))
            
            contact_metrics[contact_id] = {
                'total_activity': sent + received,
                'unique_contacts': unique_contacts,
                'activity_ratio': sent / max(1, received)
            }
        
        # Sort by different metrics
        most_active = sorted(
            contact_metrics.keys(),
            key=lambda c: contact_metrics[c]['total_activity'],
            reverse=True
        )[:10]
        
        most_connected = sorted(
            contact_metrics.keys(),
            key=lambda c: contact_metrics[c]['unique_contacts'],
            reverse=True
        )[:10]
        
        most_influential = sorted(
            contact_metrics.keys(),
            key=lambda c: contact_metrics[c]['activity_ratio'],
            reverse=True
        )[:10]
        
        return {
            'most_active': most_active,
            'most_connected': most_connected,
            'most_influential': most_influential,
            'metrics': contact_metrics
        }


class EmailFormatParser:
    """Parses various email formats (PST, MBOX, EML, etc.)"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def parse_mbox(self, mbox_path: Path) -> List[EmailMessage]:
        """Parse MBOX format mailbox"""
        messages = []
        
        try:
            mbox = mailbox.mbox(str(mbox_path))
            for message in mbox:
                parsed = self._parse_email_message(message)
                if parsed:
                    messages.append(parsed)
        
        except Exception as e:
            self.logger.error(f"Failed to parse MBOX file: {e}")
            raise
        
        return messages
    
    def parse_eml_directory(self, eml_dir: Path) -> List[EmailMessage]:
        """Parse directory containing EML files"""
        messages = []
        
        for eml_file in eml_dir.glob("*.eml"):
            try:
                with open(eml_file, 'rb') as f:
                    message = email.message_from_bytes(f.read())
                    parsed = self._parse_email_message(message)
                    if parsed:
                        messages.append(parsed)
            except Exception as e:
                self.logger.warning(f"Failed to parse {eml_file}: {e}")
                continue
        
        return messages
    
    def parse_outlook_pst(self, pst_path: Path) -> List[EmailMessage]:
        """Parse Outlook PST file (requires pypff)"""
        try:
            import pypff
        except ImportError:
            raise ImportError("pypff library required for PST parsing")
        
        messages = []
        
        try:
            pst_file = pypff.file()
            pst_file.open(str(pst_path))
            
            root = pst_file.get_root_folder()
            self._extract_pst_messages(root, messages)
            
            pst_file.close()
        
        except Exception as e:
            self.logger.error(f"Failed to parse PST file: {e}")
            raise
        
        return messages
    
    def _extract_pst_messages(self, folder, messages: List[EmailMessage]):
        """Recursively extract messages from PST folder"""
        try:
            import pypff
        except ImportError:
            return
        
        # Extract messages from current folder
        for message in folder.sub_messages:
            try:
                parsed = self._parse_pst_message(message)
                if parsed:
                    messages.append(parsed)
            except Exception as e:
                self.logger.warning(f"Failed to parse PST message: {e}")
                continue
        
        # Recursively process subfolders
        for subfolder in folder.sub_folders:
            self._extract_pst_messages(subfolder, messages)
    
    def _parse_pst_message(self, pst_message) -> Optional[EmailMessage]:
        """Parse individual PST message"""
        try:
            # Extract basic info
            subject = pst_message.subject or ""
            sender = pst_message.sender_email_address or "unknown"
            recipients = []
            
            if hasattr(pst_message, 'recipients'):
                for recipient in pst_message.recipients:
                    if hasattr(recipient, 'email_address'):
                        recipients.append(recipient.email_address)
            
            # Extract timestamp
            timestamp = datetime.now()
            if hasattr(pst_message, 'client_submit_time'):
                timestamp = pst_message.client_submit_time
            
            # Extract content
            content = ""
            html_content = None
            
            if hasattr(pst_message, 'plain_text_body'):
                content = pst_message.plain_text_body or ""
            
            if hasattr(pst_message, 'html_body'):
                html_content = pst_message.html_body
            
            # Extract headers
            headers = {}
            if hasattr(pst_message, 'transport_headers'):
                headers_str = pst_message.transport_headers or ""
                headers = self._parse_headers(headers_str)
            
            return EmailMessage(
                message_id=f"pst_{hash(f'{timestamp}_{sender}_{subject}')}",
                platform=PlatformType.EMAIL,
                comm_type=CommunicationType.EMAIL,
                timestamp=timestamp,
                sender_id=sender,
                recipient_ids=recipients,
                subject=subject,
                content=content,
                html_content=html_content,
                headers=headers,
                read=getattr(pst_message, 'is_read', False)
            )
        
        except Exception as e:
            self.logger.warning(f"Error parsing PST message: {e}")
            return None
    
    def _parse_email_message(self, message) -> Optional[EmailMessage]:
        """Parse standard email.message.Message object"""
        try:
            # Extract headers
            subject = self._decode_header(message.get('Subject', ''))
            sender = self._extract_email_address(message.get('From', ''))
            
            # Extract recipients
            recipients = []
            cc_recipients = []
            bcc_recipients = []
            
            to_header = message.get('To', '')
            if to_header:
                recipients.extend(self._extract_email_addresses(to_header))
            
            cc_header = message.get('Cc', '')
            if cc_header:
                cc_recipients.extend(self._extract_email_addresses(cc_header))
            
            bcc_header = message.get('Bcc', '')
            if bcc_header:
                bcc_recipients.extend(self._extract_email_addresses(bcc_header))
            
            # Extract timestamp
            date_header = message.get('Date')
            timestamp = datetime.now()
            if date_header:
                try:
                    timestamp = parsedate_to_datetime(date_header)
                except Exception:
                    pass
            
            # Extract message ID
            message_id = message.get('Message-ID', f"email_{hash(f'{timestamp}_{sender}_{subject}')}")
            
            # Extract replied-to
            replied_to = message.get('In-Reply-To')
            
            # Extract content
            content = ""
            html_content = None
            attachments = []
            
            if message.is_multipart():
                for part in message.walk():
                    content_type = part.get_content_type()
                    content_disposition = str(part.get('Content-Disposition', ''))
                    
                    if content_type == 'text/plain' and 'attachment' not in content_disposition:
                        content = part.get_payload(decode=True).decode('utf-8', errors='ignore')
                    elif content_type == 'text/html' and 'attachment' not in content_disposition:
                        html_content = part.get_payload(decode=True).decode('utf-8', errors='ignore')
                    elif 'attachment' in content_disposition:
                        filename = part.get_filename()
                        if filename:
                            attachments.append(filename)
            else:
                content = message.get_payload(decode=True).decode('utf-8', errors='ignore')
            
            # Extract all headers
            headers = dict(message.items())
            
            return EmailMessage(
                message_id=message_id,
                platform=PlatformType.EMAIL,
                comm_type=CommunicationType.EMAIL,
                timestamp=timestamp,
                sender_id=sender,
                recipient_ids=recipients,
                cc_recipients=cc_recipients,
                bcc_recipients=bcc_recipients,
                subject=subject,
                content=content,
                html_content=html_content,
                headers=headers,
                replied_to=replied_to,
                attachments=attachments
            )
        
        except Exception as e:
            self.logger.warning(f"Error parsing email message: {e}")
            return None
    
    def _decode_header(self, header_value: str) -> str:
        """Decode email header value"""
        if not header_value:
            return ""
        
        try:
            decoded_parts = decode_header(header_value)
            decoded_string = ""
            
            for part, encoding in decoded_parts:
                if isinstance(part, bytes):
                    if encoding:
                        decoded_string += part.decode(encoding, errors='ignore')
                    else:
                        decoded_string += part.decode('utf-8', errors='ignore')
                else:
                    decoded_string += str(part)
            
            return decoded_string.strip()
        except Exception:
            return str(header_value)
    
    def _extract_email_address(self, address_string: str) -> str:
        """Extract email address from address string"""
        if not address_string:
            return ""
        
        try:
            name, email_addr = parseaddr(address_string)
            return email_addr or address_string.strip()
        except Exception:
            return address_string.strip()
    
    def _extract_email_addresses(self, addresses_string: str) -> List[str]:
        """Extract multiple email addresses from string"""
        if not addresses_string:
            return []
        
        addresses = []
        # Split on common delimiters
        for addr in re.split(r'[,;]', addresses_string):
            email_addr = self._extract_email_address(addr.strip())
            if email_addr and '@' in email_addr:
                addresses.append(email_addr)
        
        return addresses
    
    def _parse_headers(self, headers_string: str) -> Dict[str, str]:
        """Parse headers from string"""
        headers = {}
        
        for line in headers_string.split('\n'):
            if ':' in line:
                key, value = line.split(':', 1)
                headers[key.strip()] = value.strip()
        
        return headers


class EmailAnalyzer:
    """Main email analysis class"""
    
    def __init__(self, config: CommsConfig):
        self.config = config
        self.parser = EmailFormatParser()
        self.thread_reconstructor = EmailThreadReconstructor()
        self.relationship_mapper = EmailRelationshipMapper()
        self.logger = logging.getLogger(__name__)
    
    def load_emails(self, source_path: Path) -> List[EmailMessage]:
        """Load emails from various sources"""
        if not source_path.exists():
            raise FileNotFoundError(f"Email source not found: {source_path}")
        
        messages = []
        
        if source_path.is_dir():
            # Directory of EML files
            messages = self.parser.parse_eml_directory(source_path)
        elif source_path.suffix.lower() == '.mbox':
            # MBOX format
            messages = self.parser.parse_mbox(source_path)
        elif source_path.suffix.lower() == '.pst':
            # Outlook PST format
            messages = self.parser.parse_outlook_pst(source_path)
        else:
            raise ValueError(f"Unsupported email format: {source_path.suffix}")
        
        self.logger.info(f"Loaded {len(messages)} email messages")
        return messages
    
    def analyze(self, emails: List[EmailMessage]) -> EmailAnalysis:
        """Perform comprehensive email analysis"""
        if not emails:
            return EmailAnalysis(
                total_emails=0,
                thread_count=0,
                participants=[],
                date_range=(datetime.now(), datetime.now()),
                thread_analysis={},
                relationship_mapping={},
                subject_analysis={},
                attachment_analysis={}
            )
        
        # Basic statistics
        total_emails = len(emails)
        date_range = (min(e.timestamp for e in emails), max(e.timestamp for e in emails))
        
        # Thread reconstruction
        threads = self.thread_reconstructor.reconstruct_threads(emails)
        thread_count = len(threads)
        
        # Relationship mapping
        relationship_data = self.relationship_mapper.map_relationships(emails)
        key_figures = self.relationship_mapper.identify_key_figures(relationship_data)
        
        # Extract participants
        participants = []
        for contact_id, contact_info in relationship_data['contacts'].items():
            participants.append(Contact(
                contact_id=contact_id,
                email_address=contact_info['email'],
                platforms=[PlatformType.EMAIL],
                first_seen=contact_info['first_seen'],
                last_seen=contact_info['last_seen'],
                total_messages=contact_info['sent_count'] + contact_info['received_count']
            ))
        
        # Thread analysis
        thread_analysis = self._analyze_threads(threads)
        
        # Subject analysis
        subject_analysis = self._analyze_subjects(emails)
        
        # Attachment analysis
        attachment_analysis = self._analyze_attachments(emails)
        
        # Spam analysis
        spam_analysis = self._analyze_spam(emails)
        
        return EmailAnalysis(
            total_emails=total_emails,
            thread_count=thread_count,
            participants=participants,
            date_range=date_range,
            thread_analysis=thread_analysis,
            relationship_mapping={
                **relationship_data,
                'key_figures': key_figures
            },
            subject_analysis=subject_analysis,
            attachment_analysis=attachment_analysis,
            spam_analysis=spam_analysis
        )
    
    def _analyze_threads(self, threads: Dict[str, List[EmailMessage]]) -> Dict[str, Any]:
        """Analyze email thread characteristics"""
        thread_lengths = [len(thread) for thread in threads.values()]
        
        # Calculate thread statistics
        avg_thread_length = np.mean(thread_lengths) if thread_lengths else 0
        max_thread_length = max(thread_lengths) if thread_lengths else 0
        
        # Identify long threads (potential investigations)
        long_threads = {
            thread_id: len(thread) 
            for thread_id, thread in threads.items() 
            if len(thread) > 5
        }
        
        # Calculate response times
        response_times = []
        for thread in threads.values():
            if len(thread) > 1:
                for i in range(1, len(thread)):
                    time_diff = (thread[i].timestamp - thread[i-1].timestamp).total_seconds()
                    if time_diff > 0:  # Ensure chronological order
                        response_times.append(time_diff / 3600)  # Convert to hours
        
        return {
            'total_threads': len(threads),
            'average_thread_length': avg_thread_length,
            'max_thread_length': max_thread_length,
            'long_threads': long_threads,
            'average_response_time_hours': np.mean(response_times) if response_times else 0,
            'single_message_threads': sum(1 for thread in threads.values() if len(thread) == 1)
        }
    
    def _analyze_subjects(self, emails: List[EmailMessage]) -> Dict[str, Any]:
        """Analyze email subject patterns"""
        subjects = [email.subject for email in emails if email.subject]
        
        # Subject frequency
        subject_counter = Counter(subjects)
        most_common_subjects = subject_counter.most_common(10)
        
        # Subject keywords
        all_words = []
        for subject in subjects:
            words = re.findall(r'\b\w+\b', subject.lower())
            all_words.extend(words)
        
        keyword_counter = Counter(all_words)
        top_keywords = keyword_counter.most_common(20)
        
        # Urgent/priority indicators
        urgent_indicators = ['urgent', 'asap', 'important', 'priority', 'emergency']
        urgent_count = sum(1 for subject in subjects 
                          if any(indicator in subject.lower() for indicator in urgent_indicators))
        
        return {
            'total_unique_subjects': len(set(subjects)),
            'most_common_subjects': most_common_subjects,
            'top_keywords': top_keywords,
            'urgent_emails': urgent_count,
            'average_subject_length': np.mean([len(s) for s in subjects]) if subjects else 0
        }
    
    def _analyze_attachments(self, emails: List[EmailMessage]) -> Dict[str, Any]:
        """Analyze email attachments"""
        emails_with_attachments = [e for e in emails if e.attachments]
        total_attachments = sum(len(e.attachments) for e in emails_with_attachments)
        
        # Attachment types
        attachment_types = Counter()
        for email in emails_with_attachments:
            for attachment in email.attachments:
                ext = Path(attachment).suffix.lower()
                attachment_types[ext] += 1
        
        # Suspicious extensions
        suspicious_extensions = ['.exe', '.bat', '.cmd', '.scr', '.zip', '.rar']
        suspicious_attachments = sum(
            attachment_types[ext] for ext in suspicious_extensions 
            if ext in attachment_types
        )
        
        return {
            'emails_with_attachments': len(emails_with_attachments),
            'total_attachments': total_attachments,
            'attachment_types': dict(attachment_types.most_common(10)),
            'suspicious_attachments': suspicious_attachments,
            'average_attachments_per_email': total_attachments / len(emails) if emails else 0
        }
    
    def _analyze_spam(self, emails: List[EmailMessage]) -> Dict[str, Any]:
        """Analyze spam characteristics"""
        spam_indicators = {
            'external_senders': 0,
            'bulk_emails': 0,
            'suspicious_subjects': 0,
            'no_recipients': 0
        }
        
        # Get internal domains (most common domains)
        sender_domains = [email.sender_id.split('@')[-1] for email in emails if '@' in email.sender_id]
        domain_counter = Counter(sender_domains)
        internal_domains = set(domain for domain, count in domain_counter.most_common(5))
        
        # Spam indicators
        spam_keywords = ['offer', 'free', 'click here', 'limited time', 'act now']
        
        for email in emails:
            sender_domain = email.sender_id.split('@')[-1] if '@' in email.sender_id else ''
            
            if sender_domain not in internal_domains:
                spam_indicators['external_senders'] += 1
            
            if len(email.recipient_ids) > 10:  # Bulk email indicator
                spam_indicators['bulk_emails'] += 1
            
            if any(keyword in email.subject.lower() for keyword in spam_keywords):
                spam_indicators['suspicious_subjects'] += 1
            
            if not email.recipient_ids:
                spam_indicators['no_recipients'] += 1
        
        return spam_indicators


# Main function for CLI
def analyze_email_threads(email_source: Path, config: CommsConfig = None) -> EmailAnalysis:
    """Analyze email threads and return comprehensive analysis"""
    if config is None:
        config = CommsConfig()
    
    analyzer = EmailAnalyzer(config)
    emails = analyzer.load_emails(email_source)
    return analyzer.analyze(emails)