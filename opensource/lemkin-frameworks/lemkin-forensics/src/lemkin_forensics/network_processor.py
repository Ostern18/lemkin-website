"""
Lemkin Digital Forensics Network Processor

This module provides comprehensive network traffic analysis capabilities including:
- PCAP file analysis and packet inspection
- Network flow reconstruction and analysis
- Communication pattern detection and visualization
- Protocol analysis (TCP, UDP, HTTP, DNS, TLS)
- Suspicious activity detection and threat hunting
- Geolocation and ASN resolution for IP addresses

Supports formats: PCAP, PCAPNG, network logs (Apache, IIS, firewall logs)
"""

from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from uuid import UUID, uuid4
from dataclasses import dataclass, field
from collections import defaultdict, Counter
from ipaddress import ip_address, ip_network
import re
import json
import logging

try:
    import scapy.all as scapy
    from scapy.layers.inet import IP, TCP, UDP, ICMP
    from scapy.layers.http import HTTP
    from scapy.layers.dns import DNS, DNSQR, DNSRR
    from scapy.layers.tls import TLS
    SCAPY_AVAILABLE = True
except ImportError:
    SCAPY_AVAILABLE = False
    scapy = None

from .core import (
    DigitalEvidence,
    AnalysisResult,
    NetworkArtifact,
    TimelineEvent,
    AnalysisStatus,
    EvidenceType,
    ForensicsConfig
)

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class NetworkFlow:
    """Represents a network communication flow"""
    flow_id: str
    source_ip: str
    destination_ip: str
    source_port: int
    destination_port: int
    protocol: str
    
    # Flow statistics
    packets_count: int = 0
    bytes_sent: int = 0
    bytes_received: int = 0
    duration_seconds: float = 0.0
    
    # Timing
    first_packet_time: Optional[datetime] = None
    last_packet_time: Optional[datetime] = None
    
    # Content analysis
    http_requests: List[Dict[str, Any]] = field(default_factory=list)
    dns_queries: List[Dict[str, Any]] = field(default_factory=list)
    tls_info: Optional[Dict[str, Any]] = None
    
    # Behavioral analysis
    is_encrypted: bool = False
    is_suspicious: bool = False
    threat_indicators: List[str] = field(default_factory=list)
    
    # Geolocation
    source_country: Optional[str] = None
    destination_country: Optional[str] = None
    source_asn: Optional[str] = None
    destination_asn: Optional[str] = None


@dataclass
class HTTPTransaction:
    """Represents an HTTP request/response transaction"""
    request_time: datetime
    method: str
    url: str
    host: str
    user_agent: Optional[str] = None
    
    # Request details
    request_headers: Dict[str, str] = field(default_factory=dict)
    request_body: Optional[str] = None
    
    # Response details
    response_code: Optional[int] = None
    response_headers: Dict[str, str] = field(default_factory=dict)
    response_body: Optional[str] = None
    response_time: Optional[datetime] = None
    
    # Analysis
    contains_credentials: bool = False
    contains_pii: bool = False
    is_file_download: bool = False
    file_type: Optional[str] = None


@dataclass
class DNSQuery:
    """Represents a DNS query and response"""
    query_time: datetime
    query_name: str
    query_type: str
    
    # Response details
    response_code: Optional[int] = None
    response_ips: List[str] = field(default_factory=list)
    response_cnames: List[str] = field(default_factory=list)
    
    # Analysis
    is_suspicious_domain: bool = False
    domain_reputation: Optional[float] = None
    is_dga_domain: bool = False  # Domain Generation Algorithm


@dataclass
class TLSSession:
    """Represents a TLS/SSL session"""
    session_start: datetime
    client_ip: str
    server_ip: str
    server_port: int
    
    # TLS details
    tls_version: Optional[str] = None
    cipher_suite: Optional[str] = None
    certificate_cn: Optional[str] = None
    certificate_issuer: Optional[str] = None
    certificate_valid_from: Optional[datetime] = None
    certificate_valid_to: Optional[datetime] = None
    
    # Analysis
    certificate_valid: bool = True
    is_self_signed: bool = False
    uses_weak_cipher: bool = False


class NetworkAnalysis:
    """Container for network analysis results"""
    
    def __init__(self):
        self.flows: List[NetworkFlow] = []
        self.http_transactions: List[HTTPTransaction] = []
        self.dns_queries: List[DNSQuery] = []
        self.tls_sessions: List[TLSSession] = []
        
        # Analysis summary
        self.total_packets: int = 0
        self.total_bytes: int = 0
        self.unique_ips: Set[str] = set()
        self.protocols_seen: Set[str] = set()
        self.suspicious_activities: List[str] = []
        
        # Communication patterns
        self.top_talkers: List[Tuple[str, int]] = []
        self.communication_graph: Dict[str, List[str]] = defaultdict(list)
        self.temporal_patterns: Dict[str, Any] = {}


class NetworkProcessor:
    """
    Network traffic analyzer for comprehensive forensic analysis
    
    Provides analysis capabilities for:
    - PCAP file processing and packet analysis
    - Network flow reconstruction and behavioral analysis
    - HTTP/HTTPS traffic analysis and content extraction
    - DNS analysis and domain reputation checking
    - Communication pattern detection and visualization
    - Suspicious activity detection and threat hunting
    """
    
    def __init__(self, config: Optional[ForensicsConfig] = None):
        """Initialize network processor with configuration"""
        self.config = config or ForensicsConfig()
        self.logger = logging.getLogger(f"{__name__}.NetworkProcessor")
        
        # Threat intelligence data
        self.malicious_ips: Set[str] = set()
        self.suspicious_domains: Set[str] = set()
        self.known_bad_user_agents: Set[str] = set()
        
        # Protocol parsers
        self.protocol_parsers = {
            'HTTP': self._parse_http_packet,
            'HTTPS': self._parse_https_packet,
            'DNS': self._parse_dns_packet,
            'TLS': self._parse_tls_packet
        }
        
        # Regular expressions for content analysis
        self.pii_patterns = {
            'credit_card': re.compile(r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b'),
            'ssn': re.compile(r'\b\d{3}-\d{2}-\d{4}\b'),
            'email': re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
            'phone': re.compile(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b')
        }
        
        self.credential_patterns = {
            'password': re.compile(r'password\s*[=:]\s*[\'"]?([^\'"&\s]+)', re.IGNORECASE),
            'username': re.compile(r'username\s*[=:]\s*[\'"]?([^\'"&\s]+)', re.IGNORECASE),
            'token': re.compile(r'token\s*[=:]\s*[\'"]?([^\'"&\s]+)', re.IGNORECASE)
        }
        
        if not SCAPY_AVAILABLE:
            self.logger.warning("Scapy not available - PCAP analysis will be limited")
        
        self.logger.info("Network Processor initialized")
    
    def analyze_pcap(
        self,
        evidence: DigitalEvidence,
        output_dir: Optional[str] = None
    ) -> NetworkAnalysis:
        """
        Analyze PCAP file for network communications
        
        Args:
            evidence: Digital evidence containing PCAP file
            output_dir: Directory to store extracted files
            
        Returns:
            NetworkAnalysis with comprehensive results
        """
        if not SCAPY_AVAILABLE:
            raise RuntimeError("Scapy is required for PCAP analysis")
        
        self.logger.info(f"Starting PCAP analysis: {evidence.name}")
        
        analysis = NetworkAnalysis()
        pcap_path = evidence.file_path
        
        try:
            # Verify file integrity
            if not evidence.verify_integrity():
                raise ValueError("PCAP file integrity verification failed")
            
            # Load and analyze packets
            packets = scapy.rdpcap(pcap_path)
            analysis.total_packets = len(packets)
            
            # Process packets
            flows = {}
            for packet in packets:
                self._process_packet(packet, flows, analysis)
            
            # Convert flows to NetworkFlow objects
            analysis.flows = list(flows.values())
            
            # Perform flow analysis
            self._analyze_flows(analysis)
            
            # Detect communication patterns
            self._detect_communication_patterns(analysis)
            
            # Identify suspicious activities
            self._detect_suspicious_activities(analysis)
            
            # Generate temporal analysis
            self._analyze_temporal_patterns(analysis)
            
            # Extract files if output directory provided
            if output_dir:
                self._extract_files_from_traffic(analysis, output_dir)
            
            self.logger.info(f"PCAP analysis completed: {len(analysis.flows)} flows analyzed")
            
        except Exception as e:
            self.logger.error(f"PCAP analysis failed: {str(e)}")
            raise
        
        return analysis
    
    def analyze_log_file(
        self,
        evidence: DigitalEvidence,
        log_format: str = "auto"
    ) -> NetworkAnalysis:
        """
        Analyze network log files (Apache, IIS, firewall logs)
        
        Args:
            evidence: Digital evidence containing log file
            log_format: Format of log file (auto, apache, iis, firewall)
            
        Returns:
            NetworkAnalysis with extracted information
        """
        self.logger.info(f"Starting log file analysis: {evidence.name}")
        
        analysis = NetworkAnalysis()
        log_path = evidence.file_path
        
        try:
            # Detect log format if auto
            if log_format == "auto":
                log_format = self._detect_log_format(log_path)
            
            # Parse log file based on format
            if log_format == "apache":
                self._parse_apache_logs(log_path, analysis)
            elif log_format == "iis":
                self._parse_iis_logs(log_path, analysis)
            elif log_format == "firewall":
                self._parse_firewall_logs(log_path, analysis)
            else:
                self._parse_generic_logs(log_path, analysis)
            
            # Analyze extracted data
            self._analyze_log_patterns(analysis)
            
            self.logger.info(f"Log analysis completed: {len(analysis.http_transactions)} transactions")
            
        except Exception as e:
            self.logger.error(f"Log file analysis failed: {str(e)}")
            raise
        
        return analysis
    
    def _process_packet(
        self,
        packet,
        flows: Dict[str, NetworkFlow],
        analysis: NetworkAnalysis
    ):
        """Process individual packet and update flow information"""
        try:
            if not packet.haslayer(IP):
                return
            
            ip_layer = packet[IP]
            src_ip = ip_layer.src
            dst_ip = ip_layer.dst
            protocol = ip_layer.proto
            
            # Add IPs to unique set
            analysis.unique_ips.add(src_ip)
            analysis.unique_ips.add(dst_ip)
            
            # Determine protocol name
            protocol_name = self._get_protocol_name(packet)
            analysis.protocols_seen.add(protocol_name)
            
            # Get ports for TCP/UDP
            src_port = dst_port = 0
            if packet.haslayer(TCP):
                src_port = packet[TCP].sport
                dst_port = packet[TCP].dport
            elif packet.haslayer(UDP):
                src_port = packet[UDP].sport
                dst_port = packet[UDP].dport
            
            # Create flow ID
            flow_id = self._create_flow_id(src_ip, dst_ip, src_port, dst_port, protocol_name)
            
            # Get or create flow
            if flow_id not in flows:
                flows[flow_id] = NetworkFlow(
                    flow_id=flow_id,
                    source_ip=src_ip,
                    destination_ip=dst_ip,
                    source_port=src_port,
                    destination_port=dst_port,
                    protocol=protocol_name,
                    first_packet_time=datetime.fromtimestamp(float(packet.time))
                )
            
            flow = flows[flow_id]
            
            # Update flow statistics
            flow.packets_count += 1
            flow.bytes_sent += len(packet)
            flow.last_packet_time = datetime.fromtimestamp(float(packet.time))
            
            if flow.first_packet_time and flow.last_packet_time:
                duration = flow.last_packet_time - flow.first_packet_time
                flow.duration_seconds = duration.total_seconds()
            
            # Parse protocol-specific data
            if protocol_name in self.protocol_parsers:
                self.protocol_parsers[protocol_name](packet, flow, analysis)
            
            # Update total bytes
            analysis.total_bytes += len(packet)
            
        except Exception as e:
            self.logger.warning(f"Failed to process packet: {str(e)}")
    
    def _parse_http_packet(self, packet, flow: NetworkFlow, analysis: NetworkAnalysis):
        """Parse HTTP packet and extract transaction data"""
        try:
            if not packet.haslayer(HTTP):
                return
            
            http_layer = packet[HTTP]
            
            # Check if this is an HTTP request
            if hasattr(http_layer, 'Method'):
                transaction = HTTPTransaction(
                    request_time=datetime.fromtimestamp(float(packet.time)),
                    method=http_layer.Method.decode() if http_layer.Method else 'UNKNOWN',
                    url=http_layer.Path.decode() if http_layer.Path else '/',
                    host=http_layer.Host.decode() if http_layer.Host else flow.destination_ip
                )
                
                # Extract headers
                if hasattr(http_layer, 'User_Agent'):
                    transaction.user_agent = http_layer.User_Agent.decode()
                
                # Check for credentials or PII in request
                if hasattr(http_layer, 'load'):
                    payload = http_layer.load.decode('utf-8', errors='ignore')
                    transaction.contains_credentials = self._check_for_credentials(payload)
                    transaction.contains_pii = self._check_for_pii(payload)
                    transaction.request_body = payload[:1000]  # Limit size
                
                flow.http_requests.append(transaction.dict())
                analysis.http_transactions.append(transaction)
            
            # Check if this is an HTTP response
            elif hasattr(http_layer, 'Status_Code'):
                # Find corresponding request and update it
                if flow.http_requests:
                    last_request = flow.http_requests[-1]
                    if isinstance(last_request, dict):
                        last_request['response_code'] = int(http_layer.Status_Code)
                        last_request['response_time'] = datetime.fromtimestamp(float(packet.time))
                        
                        if hasattr(http_layer, 'load'):
                            payload = http_layer.load.decode('utf-8', errors='ignore')
                            last_request['response_body'] = payload[:1000]
            
        except Exception as e:
            self.logger.warning(f"Failed to parse HTTP packet: {str(e)}")
    
    def _parse_https_packet(self, packet, flow: NetworkFlow, analysis: NetworkAnalysis):
        """Parse HTTPS/TLS packet for metadata"""
        try:
            if packet.haslayer(TLS):
                flow.is_encrypted = True
                # TLS analysis would be implemented here
                # For now, just mark as encrypted
        except Exception as e:
            self.logger.warning(f"Failed to parse HTTPS packet: {str(e)}")
    
    def _parse_dns_packet(self, packet, flow: NetworkFlow, analysis: NetworkAnalysis):
        """Parse DNS packet and extract query information"""
        try:
            if not packet.haslayer(DNS):
                return
            
            dns_layer = packet[DNS]
            
            # DNS Query
            if dns_layer.qr == 0 and dns_layer.qdcount > 0:
                query = DNSQuery(
                    query_time=datetime.fromtimestamp(float(packet.time)),
                    query_name=dns_layer.qd.qname.decode().rstrip('.'),
                    query_type=dns_layer.qd.qtype
                )
                
                # Check for suspicious domains
                query.is_suspicious_domain = self._is_suspicious_domain(query.query_name)
                query.is_dga_domain = self._is_dga_domain(query.query_name)
                
                flow.dns_queries.append(query.__dict__)
                analysis.dns_queries.append(query)
            
            # DNS Response
            elif dns_layer.qr == 1 and dns_layer.ancount > 0:
                # Find corresponding query and update it
                if flow.dns_queries:
                    last_query = flow.dns_queries[-1]
                    if isinstance(last_query, dict):
                        last_query['response_code'] = dns_layer.rcode
                        
                        # Extract response IPs
                        response_ips = []
                        for i in range(dns_layer.ancount):
                            if hasattr(dns_layer.an[i], 'rdata'):
                                response_ips.append(str(dns_layer.an[i].rdata))
                        
                        last_query['response_ips'] = response_ips
            
        except Exception as e:
            self.logger.warning(f"Failed to parse DNS packet: {str(e)}")
    
    def _parse_tls_packet(self, packet, flow: NetworkFlow, analysis: NetworkAnalysis):
        """Parse TLS packet for session information"""
        try:
            if packet.haslayer(TLS):
                flow.is_encrypted = True
                # Detailed TLS analysis would be implemented here
        except Exception as e:
            self.logger.warning(f"Failed to parse TLS packet: {str(e)}")
    
    def _analyze_flows(self, analysis: NetworkAnalysis):
        """Analyze network flows for patterns and anomalies"""
        # Calculate top talkers
        ip_bytes = defaultdict(int)
        for flow in analysis.flows:
            ip_bytes[flow.source_ip] += flow.bytes_sent
            ip_bytes[flow.destination_ip] += flow.bytes_sent
        
        analysis.top_talkers = sorted(ip_bytes.items(), key=lambda x: x[1], reverse=True)[:10]
        
        # Build communication graph
        for flow in analysis.flows:
            analysis.communication_graph[flow.source_ip].append(flow.destination_ip)
    
    def _detect_communication_patterns(self, analysis: NetworkAnalysis):
        """Detect communication patterns and behavioral indicators"""
        # Analyze HTTP patterns
        user_agents = Counter()
        request_methods = Counter()
        
        for transaction in analysis.http_transactions:
            if transaction.user_agent:
                user_agents[transaction.user_agent] += 1
            request_methods[transaction.method] += 1
        
        # Check for suspicious user agents
        for ua, count in user_agents.items():
            if ua in self.known_bad_user_agents or self._is_suspicious_user_agent(ua):
                analysis.suspicious_activities.append(f"Suspicious user agent detected: {ua}")
        
        # Analyze DNS patterns
        domain_requests = Counter()
        for query in analysis.dns_queries:
            domain_requests[query.query_name] += 1
        
        # Check for excessive DNS queries (potential DGA)
        for domain, count in domain_requests.items():
            if count > 100:  # Threshold for excessive queries
                analysis.suspicious_activities.append(f"Excessive DNS queries to {domain}: {count}")
    
    def _detect_suspicious_activities(self, analysis: NetworkAnalysis):
        """Detect suspicious network activities"""
        # Check for connections to known malicious IPs
        for flow in analysis.flows:
            if flow.destination_ip in self.malicious_ips:
                flow.is_suspicious = True
                flow.threat_indicators.append("Connection to known malicious IP")
                analysis.suspicious_activities.append(f"Connection to malicious IP: {flow.destination_ip}")
        
        # Check for large data transfers
        for flow in analysis.flows:
            if flow.bytes_sent > 100 * 1024 * 1024:  # 100MB threshold
                flow.threat_indicators.append("Large data transfer")
        
        # Check for port scanning patterns
        source_ports = defaultdict(set)
        for flow in analysis.flows:
            source_ports[flow.source_ip].add(flow.destination_port)
        
        for source_ip, ports in source_ports.items():
            if len(ports) > 100:  # Many different ports
                analysis.suspicious_activities.append(f"Potential port scan from {source_ip}: {len(ports)} ports")
    
    def _analyze_temporal_patterns(self, analysis: NetworkAnalysis):
        """Analyze temporal patterns in network traffic"""
        hourly_activity = defaultdict(int)
        daily_activity = defaultdict(int)
        
        for flow in analysis.flows:
            if flow.first_packet_time:
                hour = flow.first_packet_time.hour
                day = flow.first_packet_time.strftime('%Y-%m-%d')
                hourly_activity[hour] += flow.bytes_sent
                daily_activity[day] += flow.bytes_sent
        
        analysis.temporal_patterns = {
            'hourly_activity': dict(hourly_activity),
            'daily_activity': dict(daily_activity)
        }
    
    def _extract_files_from_traffic(self, analysis: NetworkAnalysis, output_dir: str):
        """Extract files from HTTP traffic"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        file_count = 0
        for transaction in analysis.http_transactions:
            if transaction.is_file_download and transaction.response_body:
                try:
                    filename = f"extracted_file_{file_count}_{transaction.file_type or 'unknown'}"
                    filepath = os.path.join(output_dir, filename)
                    
                    with open(filepath, 'wb') as f:
                        f.write(transaction.response_body.encode('utf-8', errors='ignore'))
                    
                    file_count += 1
                    
                except Exception as e:
                    self.logger.warning(f"Failed to extract file: {str(e)}")
    
    def _detect_log_format(self, log_path: str) -> str:
        """Detect log file format from content"""
        try:
            with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
                first_line = f.readline().strip()
            
            # Apache Common Log Format
            if re.match(r'^\d+\.\d+\.\d+\.\d+ - - \[', first_line):
                return "apache"
            
            # IIS Log Format
            if first_line.startswith('#Software:') or first_line.startswith('#Version:'):
                return "iis"
            
            # Generic format
            return "generic"
            
        except Exception as e:
            self.logger.warning(f"Failed to detect log format: {str(e)}")
            return "generic"
    
    def _parse_apache_logs(self, log_path: str, analysis: NetworkAnalysis):
        """Parse Apache access logs"""
        apache_pattern = re.compile(
            r'(?P<ip>\d+\.\d+\.\d+\.\d+) - - \[(?P<timestamp>[^\]]+)\] '
            r'"(?P<method>\w+) (?P<url>[^"]*) [^"]*" (?P<status>\d+) (?P<size>\d+|-)'
        )
        
        try:
            with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    match = apache_pattern.match(line.strip())
                    if match:
                        data = match.groupdict()
                        
                        # Parse timestamp
                        timestamp_str = data['timestamp']
                        timestamp = self._parse_apache_timestamp(timestamp_str)
                        
                        transaction = HTTPTransaction(
                            request_time=timestamp,
                            method=data['method'],
                            url=data['url'],
                            host='unknown',
                            response_code=int(data['status']) if data['status'].isdigit() else None
                        )
                        
                        analysis.http_transactions.append(transaction)
                        
        except Exception as e:
            self.logger.error(f"Failed to parse Apache logs: {str(e)}")
    
    def _parse_iis_logs(self, log_path: str, analysis: NetworkAnalysis):
        """Parse IIS logs"""
        # IIS log parsing implementation
        pass
    
    def _parse_firewall_logs(self, log_path: str, analysis: NetworkAnalysis):
        """Parse firewall logs"""
        # Firewall log parsing implementation
        pass
    
    def _parse_generic_logs(self, log_path: str, analysis: NetworkAnalysis):
        """Parse generic network logs"""
        # Generic log parsing implementation
        pass
    
    def _analyze_log_patterns(self, analysis: NetworkAnalysis):
        """Analyze patterns in log data"""
        # Log pattern analysis implementation
        pass
    
    def _get_protocol_name(self, packet) -> str:
        """Get protocol name from packet"""
        if packet.haslayer(TCP):
            if packet.haslayer(HTTP):
                return "HTTP"
            elif packet[TCP].dport == 443 or packet[TCP].sport == 443:
                return "HTTPS"
            elif packet[TCP].dport == 25 or packet[TCP].sport == 25:
                return "SMTP"
            elif packet[TCP].dport == 21 or packet[TCP].sport == 21:
                return "FTP"
            else:
                return "TCP"
        elif packet.haslayer(UDP):
            if packet.haslayer(DNS):
                return "DNS"
            elif packet[UDP].dport == 67 or packet[UDP].sport == 67:
                return "DHCP"
            else:
                return "UDP"
        elif packet.haslayer(ICMP):
            return "ICMP"
        else:
            return "OTHER"
    
    def _create_flow_id(
        self,
        src_ip: str,
        dst_ip: str,
        src_port: int,
        dst_port: int,
        protocol: str
    ) -> str:
        """Create unique flow identifier"""
        # Normalize flow direction
        if src_ip < dst_ip or (src_ip == dst_ip and src_port < dst_port):
            return f"{src_ip}:{src_port}-{dst_ip}:{dst_port}-{protocol}"
        else:
            return f"{dst_ip}:{dst_port}-{src_ip}:{src_port}-{protocol}"
    
    def _check_for_credentials(self, content: str) -> bool:
        """Check if content contains credentials"""
        for pattern in self.credential_patterns.values():
            if pattern.search(content):
                return True
        return False
    
    def _check_for_pii(self, content: str) -> bool:
        """Check if content contains PII"""
        for pattern in self.pii_patterns.values():
            if pattern.search(content):
                return True
        return False
    
    def _is_suspicious_domain(self, domain: str) -> bool:
        """Check if domain is known to be suspicious"""
        return domain in self.suspicious_domains
    
    def _is_dga_domain(self, domain: str) -> bool:
        """Check if domain appears to be generated by DGA"""
        # Simple heuristic for DGA detection
        if len(domain) > 20:  # Very long domains
            return True
        
        # Count consonant clusters
        consonants = 'bcdfghjklmnpqrstvwxyz'
        consecutive_consonants = 0
        max_consecutive = 0
        
        for char in domain.lower():
            if char in consonants:
                consecutive_consonants += 1
                max_consecutive = max(max_consecutive, consecutive_consonants)
            else:
                consecutive_consonants = 0
        
        return max_consecutive > 4  # Too many consecutive consonants
    
    def _is_suspicious_user_agent(self, user_agent: str) -> bool:
        """Check if user agent is suspicious"""
        suspicious_patterns = [
            'python-requests',
            'curl/',
            'wget/',
            'scanner',
            'bot',
            'crawler'
        ]
        
        ua_lower = user_agent.lower()
        return any(pattern in ua_lower for pattern in suspicious_patterns)
    
    def _parse_apache_timestamp(self, timestamp_str: str) -> datetime:
        """Parse Apache timestamp format"""
        try:
            # Format: 10/Oct/2000:13:55:36 -0700
            return datetime.strptime(timestamp_str.split()[0], '%d/%b/%Y:%H:%M:%S')
        except ValueError:
            return datetime.utcnow()
    
    def generate_network_artifacts(self, analysis: NetworkAnalysis) -> List[NetworkArtifact]:
        """Convert network analysis to NetworkArtifact objects"""
        artifacts = []
        
        for flow in analysis.flows:
            artifact = NetworkArtifact(
                timestamp=flow.first_packet_time or datetime.utcnow(),
                source_ip=flow.source_ip,
                destination_ip=flow.destination_ip,
                source_port=flow.source_port,
                destination_port=flow.destination_port,
                protocol=flow.protocol,
                bytes_sent=flow.bytes_sent,
                bytes_received=flow.bytes_received,
                duration_seconds=flow.duration_seconds,
                is_suspicious=flow.is_suspicious,
                threat_indicators=flow.threat_indicators,
                source_country=flow.source_country,
                destination_country=flow.destination_country,
                source_asn=flow.source_asn,
                destination_asn=flow.destination_asn
            )
            artifacts.append(artifact)
        
        return artifacts