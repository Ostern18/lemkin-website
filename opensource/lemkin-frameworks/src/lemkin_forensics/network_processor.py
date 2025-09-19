"""
Lemkin Digital Forensics - Network Processor

This module provides comprehensive network log analysis capabilities including
communication pattern detection, suspicious activity identification, and
forensic timeline generation for legal investigations.
"""

import re
import csv
import json
import ipaddress
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Set
from collections import defaultdict, Counter
import logging
import geoip2.database
import geoip2.errors

from .core import (
    NetworkFlow, NetworkAnalysis, DigitalEvidence, AnalysisStatus,
    NetworkEventType, ForensicsConfig
)

logger = logging.getLogger(__name__)


class NetworkLogProcessor:
    """
    Comprehensive network log analysis for digital forensics investigations.
    
    Provides capabilities for:
    - Network flow analysis
    - Communication pattern detection
    - Geolocation analysis
    - Suspicious activity identification
    - Protocol analysis
    """
    
    def __init__(self, config: Optional[ForensicsConfig] = None):
        self.config = config or ForensicsConfig()
        self.logger = logging.getLogger(f"{__name__}.NetworkLogProcessor")
        
        # Suspicious indicators
        self.suspicious_ports = {
            22: "SSH (potential unauthorized access)",
            23: "Telnet (unencrypted)",
            135: "RPC (potential malware)",
            139: "NetBIOS (file sharing)",
            445: "SMB (file sharing)",
            1433: "MSSQL (database access)",
            3389: "RDP (remote desktop)",
            4444: "Common backdoor port",
            5555: "Common backdoor port",
            6666: "Common backdoor port",
            7777: "Common backdoor port",
            8080: "HTTP alternate (proxy)",
            9999: "Common backdoor port"
        }
        
        # Known malicious IP ranges (example - would be updated from threat feeds)
        self.known_malicious_ranges = [
            ipaddress.IPv4Network('10.0.0.0/8'),    # Private (if unexpected)
            ipaddress.IPv4Network('172.16.0.0/12'), # Private (if unexpected)
            ipaddress.IPv4Network('192.168.0.0/16') # Private (if unexpected)
        ]
        
        # Protocol patterns
        self.protocol_patterns = {
            'HTTP': re.compile(r'GET|POST|PUT|DELETE|HEAD|OPTIONS'),
            'HTTPS': re.compile(r'Client Hello|Server Hello|Certificate'),
            'DNS': re.compile(r'A\?|AAAA\?|MX\?|NS\?|PTR\?'),
            'SMTP': re.compile(r'MAIL FROM|RCPT TO|DATA|QUIT'),
            'FTP': re.compile(r'USER|PASS|STOR|RETR|LIST'),
            'SSH': re.compile(r'SSH-\d+\.\d+'),
            'Telnet': re.compile(r'\xff[\xfb-\xfe]'),
        }
    
    def process_network_logs(self, log_files: List[Path]) -> NetworkAnalysis:
        """
        Process network log files and perform comprehensive analysis
        
        Args:
            log_files: List of network log file paths
            
        Returns:
            NetworkAnalysis containing all findings
        """
        analysis = NetworkAnalysis(evidence_id=self._generate_evidence_id())
        analysis.status = AnalysisStatus.IN_PROGRESS
        
        try:
            self.logger.info(f"Starting network analysis of {len(log_files)} log files")
            
            # Process each log file
            all_flows = []
            for log_file in log_files:
                if log_file.exists():
                    flows = self._process_log_file(log_file)
                    all_flows.extend(flows)
                    analysis.log_files_processed += 1
                else:
                    self.logger.warning(f"Log file not found: {log_file}")
            
            # Store flows in analysis
            analysis.network_flows = all_flows
            
            # Calculate basic statistics
            self._calculate_network_statistics(analysis)
            
            # Detect communication patterns
            self._detect_communication_patterns(analysis)
            
            # Identify suspicious connections
            self._identify_suspicious_connections(analysis)
            
            # Perform geolocation analysis
            self._perform_geolocation_analysis(analysis)
            
            # Analyze protocols
            self._analyze_protocols(analysis)
            
            # Generate timeline
            self._generate_network_timeline(analysis)
            
            # Generate key findings
            self._generate_key_findings(analysis)
            
            analysis.completed_at = datetime.utcnow()
            analysis.status = AnalysisStatus.COMPLETED
            
            self.logger.info("Network analysis completed successfully")
            
        except Exception as e:
            self.logger.error(f"Network analysis failed: {str(e)}")
            analysis.status = AnalysisStatus.FAILED
            analysis.error_messages.append(str(e))
        
        return analysis
    
    def _process_log_file(self, log_file: Path) -> List[NetworkFlow]:
        """Process a single network log file"""
        flows = []
        
        try:
            # Determine log file format
            log_format = self._detect_log_format(log_file)
            self.logger.info(f"Processing {log_file} as {log_format} format")
            
            if log_format == "csv":
                flows = self._process_csv_log(log_file)
            elif log_format == "json":
                flows = self._process_json_log(log_file)
            elif log_format == "apache":
                flows = self._process_apache_log(log_file)
            elif log_format == "firewall":
                flows = self._process_firewall_log(log_file)
            else:
                flows = self._process_generic_log(log_file)
            
            self.logger.info(f"Extracted {len(flows)} network flows from {log_file}")
            
        except Exception as e:
            self.logger.error(f"Failed to process log file {log_file}: {str(e)}")
        
        return flows
    
    def _detect_log_format(self, log_file: Path) -> str:
        """Detect the format of a log file"""
        try:
            with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                first_lines = [f.readline().strip() for _ in range(5)]
                sample_text = '\n'.join(first_lines)
            
            # Check for JSON format
            if any(line.strip().startswith('{') for line in first_lines):
                return "json"
            
            # Check for CSV format
            if ',' in sample_text and ('src' in sample_text.lower() or 'dst' in sample_text.lower()):
                return "csv"
            
            # Check for Apache/web server format
            if re.search(r'\d+\.\d+\.\d+\.\d+.*\[.*\].*"(GET|POST)', sample_text):
                return "apache"
            
            # Check for firewall log format
            if re.search(r'(ALLOW|DENY|BLOCK|DROP)', sample_text, re.IGNORECASE):
                return "firewall"
            
            return "generic"
            
        except Exception as e:
            self.logger.warning(f"Could not detect log format for {log_file}: {str(e)}")
            return "generic"
    
    def _process_csv_log(self, log_file: Path) -> List[NetworkFlow]:
        """Process CSV format network log"""
        flows = []
        
        try:
            with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                csv_reader = csv.DictReader(f)
                
                for row in csv_reader:
                    try:
                        flow = self._create_flow_from_csv_row(row)
                        if flow:
                            flows.append(flow)
                    except Exception as e:
                        self.logger.warning(f"Failed to process CSV row: {str(e)}")
                        continue
                        
        except Exception as e:
            self.logger.error(f"CSV processing failed: {str(e)}")
        
        return flows
    
    def _create_flow_from_csv_row(self, row: Dict[str, str]) -> Optional[NetworkFlow]:
        """Create NetworkFlow from CSV row"""
        try:
            # Common CSV column mappings
            mappings = {
                'source_ip': ['src_ip', 'source_ip', 'src', 'source', 'client_ip'],
                'dest_ip': ['dst_ip', 'dest_ip', 'dst', 'destination', 'server_ip'],
                'source_port': ['src_port', 'source_port', 'sport'],
                'dest_port': ['dst_port', 'dest_port', 'dport', 'port'],
                'protocol': ['protocol', 'proto'],
                'timestamp': ['timestamp', 'time', 'datetime'],
                'bytes_sent': ['bytes_sent', 'bytes_out', 'sent_bytes'],
                'bytes_received': ['bytes_received', 'bytes_in', 'received_bytes'],
            }
            
            # Extract values using mappings
            values = {}
            for field, possible_columns in mappings.items():
                for col in possible_columns:
                    if col in row and row[col]:
                        values[field] = row[col]
                        break
            
            if 'source_ip' not in values or 'dest_ip' not in values:
                return None
            
            # Parse timestamp
            timestamp = datetime.utcnow()
            if 'timestamp' in values:
                timestamp = self._parse_timestamp(values['timestamp'])
            
            flow = NetworkFlow(
                source_ip=values['source_ip'],
                destination_ip=values['dest_ip'],
                source_port=self._safe_int_convert(values.get('source_port')),
                destination_port=self._safe_int_convert(values.get('dest_port')),
                protocol=values.get('protocol', 'Unknown'),
                start_time=timestamp,
                bytes_sent=self._safe_int_convert(values.get('bytes_sent'), 0),
                bytes_received=self._safe_int_convert(values.get('bytes_received'), 0)
            )
            
            return flow
            
        except Exception as e:
            self.logger.warning(f"Failed to create flow from CSV row: {str(e)}")
            return None
    
    def _process_json_log(self, log_file: Path) -> List[NetworkFlow]:
        """Process JSON format network log"""
        flows = []
        
        try:
            with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        data = json.loads(line.strip())
                        flow = self._create_flow_from_json(data)
                        if flow:
                            flows.append(flow)
                    except json.JSONDecodeError:
                        if line_num <= 10:  # Only warn for first 10 lines
                            self.logger.warning(f"Invalid JSON on line {line_num}")
                        continue
                    except Exception as e:
                        self.logger.warning(f"Failed to process JSON line {line_num}: {str(e)}")
                        continue
                        
        except Exception as e:
            self.logger.error(f"JSON processing failed: {str(e)}")
        
        return flows
    
    def _create_flow_from_json(self, data: Dict[str, Any]) -> Optional[NetworkFlow]:
        """Create NetworkFlow from JSON data"""
        try:
            # Extract required fields
            source_ip = (data.get('src_ip') or data.get('source_ip') or 
                        data.get('client_ip') or data.get('src'))
            dest_ip = (data.get('dst_ip') or data.get('dest_ip') or 
                      data.get('destination_ip') or data.get('dst'))
            
            if not source_ip or not dest_ip:
                return None
            
            timestamp = datetime.utcnow()
            if 'timestamp' in data:
                timestamp = self._parse_timestamp(data['timestamp'])
            elif '@timestamp' in data:
                timestamp = self._parse_timestamp(data['@timestamp'])
            
            flow = NetworkFlow(
                source_ip=str(source_ip),
                destination_ip=str(dest_ip),
                source_port=self._safe_int_convert(data.get('src_port') or data.get('source_port')),
                destination_port=self._safe_int_convert(data.get('dst_port') or data.get('dest_port')),
                protocol=data.get('protocol', 'Unknown'),
                start_time=timestamp,
                bytes_sent=self._safe_int_convert(data.get('bytes_sent'), 0),
                bytes_received=self._safe_int_convert(data.get('bytes_received'), 0),
                application=data.get('application'),
                service=data.get('service'),
                user_agent=data.get('user_agent')
            )
            
            return flow
            
        except Exception as e:
            self.logger.warning(f"Failed to create flow from JSON: {str(e)}")
            return None
    
    def _process_apache_log(self, log_file: Path) -> List[NetworkFlow]:
        """Process Apache/web server log format"""
        flows = []
        
        # Apache combined log format pattern
        apache_pattern = re.compile(
            r'(?P<ip>\d+\.\d+\.\d+\.\d+).*'
            r'\[(?P<timestamp>[^\]]+)\].*'
            r'"(?P<method>\S+)\s+(?P<url>\S+)\s+(?P<protocol>[^"]+)".*'
            r'(?P<status>\d+)\s+(?P<size>\d+|-)'
        )
        
        try:
            with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        match = apache_pattern.match(line.strip())
                        if match:
                            flow = self._create_flow_from_apache_match(match)
                            if flow:
                                flows.append(flow)
                    except Exception as e:
                        if line_num <= 10:
                            self.logger.warning(f"Failed to process Apache log line {line_num}: {str(e)}")
                        continue
                        
        except Exception as e:
            self.logger.error(f"Apache log processing failed: {str(e)}")
        
        return flows
    
    def _create_flow_from_apache_match(self, match) -> Optional[NetworkFlow]:
        """Create NetworkFlow from Apache log regex match"""
        try:
            timestamp = self._parse_apache_timestamp(match.group('timestamp'))
            
            flow = NetworkFlow(
                source_ip=match.group('ip'),
                destination_ip='127.0.0.1',  # Assume local server
                destination_port=80,  # Default HTTP port
                protocol='HTTP',
                start_time=timestamp,
                application=match.group('method'),
                service=f"{match.group('method')} {match.group('url')}",
                bytes_received=self._safe_int_convert(match.group('size'), 0)
            )
            
            return flow
            
        except Exception as e:
            self.logger.warning(f"Failed to create flow from Apache match: {str(e)}")
            return None
    
    def _process_firewall_log(self, log_file: Path) -> List[NetworkFlow]:
        """Process firewall log format"""
        flows = []
        
        # Generic firewall pattern
        firewall_pattern = re.compile(
            r'(?P<timestamp>\w+\s+\d+\s+\d+:\d+:\d+).*'
            r'(?P<action>ALLOW|DENY|BLOCK|DROP).*'
            r'(?P<src_ip>\d+\.\d+\.\d+\.\d+):?(?P<src_port>\d+)?.*'
            r'(?P<dst_ip>\d+\.\d+\.\d+\.\d+):?(?P<dst_port>\d+)?'
        )
        
        try:
            with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        match = firewall_pattern.search(line)
                        if match:
                            flow = self._create_flow_from_firewall_match(match)
                            if flow:
                                flows.append(flow)
                    except Exception as e:
                        if line_num <= 10:
                            self.logger.warning(f"Failed to process firewall log line {line_num}: {str(e)}")
                        continue
                        
        except Exception as e:
            self.logger.error(f"Firewall log processing failed: {str(e)}")
        
        return flows
    
    def _create_flow_from_firewall_match(self, match) -> Optional[NetworkFlow]:
        """Create NetworkFlow from firewall log regex match"""
        try:
            timestamp = self._parse_syslog_timestamp(match.group('timestamp'))
            
            flow = NetworkFlow(
                source_ip=match.group('src_ip'),
                destination_ip=match.group('dst_ip'),
                source_port=self._safe_int_convert(match.group('src_port')),
                destination_port=self._safe_int_convert(match.group('dst_port')),
                protocol='Unknown',
                start_time=timestamp,
                application=match.group('action')
            )
            
            return flow
            
        except Exception as e:
            self.logger.warning(f"Failed to create flow from firewall match: {str(e)}")
            return None
    
    def _process_generic_log(self, log_file: Path) -> List[NetworkFlow]:
        """Process generic log file by looking for IP patterns"""
        flows = []
        
        ip_pattern = re.compile(r'(\d+\.\d+\.\d+\.\d+)')
        
        try:
            with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        ips = ip_pattern.findall(line)
                        if len(ips) >= 2:
                            # Create basic flow from IP addresses found
                            flow = NetworkFlow(
                                source_ip=ips[0],
                                destination_ip=ips[1],
                                protocol='Unknown',
                                start_time=datetime.utcnow()
                            )
                            flows.append(flow)
                    except Exception as e:
                        continue
                        
        except Exception as e:
            self.logger.error(f"Generic log processing failed: {str(e)}")
        
        return flows
    
    def _calculate_network_statistics(self, analysis: NetworkAnalysis):
        """Calculate basic network statistics"""
        try:
            flows = analysis.network_flows
            
            if not flows:
                return
            
            analysis.total_connections = len(flows)
            analysis.unique_sources = len(set(flow.source_ip for flow in flows))
            analysis.unique_destinations = len(set(flow.destination_ip for flow in flows))
            analysis.total_data_bytes = sum(
                (flow.bytes_sent or 0) + (flow.bytes_received or 0) for flow in flows
            )
            
            # Find time range
            timestamps = [flow.start_time for flow in flows if flow.start_time]
            if timestamps:
                analysis.earliest_activity = min(timestamps)
                analysis.latest_activity = max(timestamps)
            
            self.logger.info(f"Network statistics: {analysis.total_connections} connections, "
                           f"{analysis.unique_destinations} unique destinations")
            
        except Exception as e:
            self.logger.error(f"Network statistics calculation failed: {str(e)}")
    
    def _detect_communication_patterns(self, analysis: NetworkAnalysis):
        """Detect communication patterns in network flows"""
        try:
            flows = analysis.network_flows
            patterns = []
            
            # Analyze destination frequency
            dest_counter = Counter(flow.destination_ip for flow in flows)
            top_destinations = []
            
            for dest_ip, count in dest_counter.most_common(10):
                total_bytes = sum(
                    (flow.bytes_sent or 0) + (flow.bytes_received or 0)
                    for flow in flows if flow.destination_ip == dest_ip
                )
                
                top_destinations.append({
                    'ip': dest_ip,
                    'connections': count,
                    'total_bytes': total_bytes,
                    'percentage': (count / len(flows)) * 100
                })
            
            analysis.top_destinations = top_destinations
            
            # Detect beaconing behavior
            beaconing_patterns = self._detect_beaconing(flows)
            patterns.extend(beaconing_patterns)
            
            # Detect data exfiltration patterns
            exfiltration_patterns = self._detect_data_exfiltration(flows)
            patterns.extend(exfiltration_patterns)
            
            # Detect port scanning
            scanning_patterns = self._detect_port_scanning(flows)
            patterns.extend(scanning_patterns)
            
            analysis.communication_patterns = patterns
            
        except Exception as e:
            self.logger.error(f"Communication pattern detection failed: {str(e)}")
    
    def _detect_beaconing(self, flows: List[NetworkFlow]) -> List[Dict[str, Any]]:
        """Detect beaconing behavior (regular communication patterns)"""
        patterns = []
        
        try:
            # Group flows by destination
            dest_groups = defaultdict(list)
            for flow in flows:
                if flow.start_time:
                    dest_groups[flow.destination_ip].append(flow.start_time)
            
            for dest_ip, timestamps in dest_groups.items():
                if len(timestamps) < 5:  # Need at least 5 connections
                    continue
                
                # Sort timestamps
                timestamps.sort()
                
                # Calculate intervals between connections
                intervals = []
                for i in range(1, len(timestamps)):
                    interval = (timestamps[i] - timestamps[i-1]).total_seconds()
                    intervals.append(interval)
                
                if len(intervals) < 3:
                    continue
                
                # Check for regular intervals (beaconing)
                avg_interval = sum(intervals) / len(intervals)
                interval_variance = sum((x - avg_interval) ** 2 for x in intervals) / len(intervals)
                
                # If variance is low, it might be beaconing
                if interval_variance < (avg_interval * 0.2) ** 2 and avg_interval < 3600:
                    patterns.append({
                        'type': 'beaconing',
                        'destination_ip': dest_ip,
                        'average_interval_seconds': avg_interval,
                        'connection_count': len(timestamps),
                        'confidence': min(1.0, len(timestamps) / 20.0),
                        'description': f"Regular communication pattern every {avg_interval:.1f} seconds"
                    })
            
        except Exception as e:
            self.logger.warning(f"Beaconing detection failed: {str(e)}")
        
        return patterns
    
    def _detect_data_exfiltration(self, flows: List[NetworkFlow]) -> List[Dict[str, Any]]:
        """Detect potential data exfiltration patterns"""
        patterns = []
        
        try:
            # Look for large outbound transfers
            for flow in flows:
                bytes_sent = flow.bytes_sent or 0
                bytes_received = flow.bytes_received or 0
                
                # Check for unusually large outbound transfers
                if bytes_sent > 1024 * 1024 * 10:  # > 10MB
                    patterns.append({
                        'type': 'large_outbound_transfer',
                        'destination_ip': flow.destination_ip,
                        'bytes_sent': bytes_sent,
                        'timestamp': flow.start_time,
                        'confidence': min(1.0, bytes_sent / (1024 * 1024 * 100)),
                        'description': f"Large outbound transfer: {bytes_sent / (1024*1024):.1f} MB"
                    })
                
                # Check for unusual outbound/inbound ratio
                if bytes_received > 0:
                    ratio = bytes_sent / bytes_received
                    if ratio > 10 and bytes_sent > 1024 * 1024:  # 10:1 ratio and > 1MB
                        patterns.append({
                            'type': 'suspicious_transfer_ratio',
                            'destination_ip': flow.destination_ip,
                            'bytes_sent': bytes_sent,
                            'bytes_received': bytes_received,
                            'ratio': ratio,
                            'timestamp': flow.start_time,
                            'confidence': min(1.0, ratio / 50.0),
                            'description': f"Unusual transfer ratio {ratio:.1f}:1 (out:in)"
                        })
            
        except Exception as e:
            self.logger.warning(f"Data exfiltration detection failed: {str(e)}")
        
        return patterns
    
    def _detect_port_scanning(self, flows: List[NetworkFlow]) -> List[Dict[str, Any]]:
        """Detect port scanning behavior"""
        patterns = []
        
        try:
            # Group by source IP and destination IP
            source_dest_ports = defaultdict(set)
            
            for flow in flows:
                if flow.destination_port:
                    key = (flow.source_ip, flow.destination_ip)
                    source_dest_ports[key].add(flow.destination_port)
            
            # Look for connections to many ports on same destination
            for (source_ip, dest_ip), ports in source_dest_ports.items():
                if len(ports) > 10:  # Connected to more than 10 ports
                    patterns.append({
                        'type': 'port_scanning',
                        'source_ip': source_ip,
                        'destination_ip': dest_ip,
                        'ports_scanned': len(ports),
                        'port_list': sorted(list(ports)),
                        'confidence': min(1.0, len(ports) / 100.0),
                        'description': f"Scanned {len(ports)} ports on {dest_ip}"
                    })
            
        except Exception as e:
            self.logger.warning(f"Port scanning detection failed: {str(e)}")
        
        return patterns
    
    def _identify_suspicious_connections(self, analysis: NetworkAnalysis):
        """Identify potentially suspicious network connections"""
        try:
            suspicious = []
            
            for flow in analysis.network_flows:
                suspicion_reasons = []
                
                # Check for connections to suspicious ports
                if flow.destination_port in self.suspicious_ports:
                    suspicion_reasons.append(
                        self.suspicious_ports[flow.destination_port]
                    )
                
                # Check for connections to private IP ranges from public IPs
                if self._is_suspicious_ip_combination(flow.source_ip, flow.destination_ip):
                    suspicion_reasons.append("Suspicious IP address combination")
                
                # Check for large data transfers
                total_bytes = (flow.bytes_sent or 0) + (flow.bytes_received or 0)
                if total_bytes > 100 * 1024 * 1024:  # > 100MB
                    suspicion_reasons.append(f"Large data transfer: {total_bytes / (1024*1024):.1f} MB")
                
                # Check for unusual protocols
                if flow.protocol and flow.protocol.lower() in ['telnet', 'ftp', 'rsh']:
                    suspicion_reasons.append(f"Unencrypted protocol: {flow.protocol}")
                
                # Check for off-hours activity (outside 9-17)
                if flow.start_time:
                    hour = flow.start_time.hour
                    if hour < 9 or hour > 17:
                        suspicion_reasons.append("Off-hours network activity")
                
                if suspicion_reasons:
                    # Add suspicious indicators to the flow
                    flow.suspicious_indicators = suspicion_reasons
                    suspicious.append(flow)
            
            analysis.suspicious_connections = suspicious
            
            if suspicious:
                analysis.security_alerts.append(
                    f"Identified {len(suspicious)} suspicious network connections"
                )
            
        except Exception as e:
            self.logger.error(f"Suspicious connection identification failed: {str(e)}")
    
    def _is_suspicious_ip_combination(self, source_ip: str, dest_ip: str) -> bool:
        """Check if IP combination is suspicious"""
        try:
            src_addr = ipaddress.IPv4Address(source_ip)
            dst_addr = ipaddress.IPv4Address(dest_ip)
            
            # Check if one is private and one is public (might be suspicious)
            src_private = src_addr.is_private
            dst_private = dst_addr.is_private
            
            # Suspicious if mixing private/public in unexpected ways
            if src_private and not dst_private:
                return False  # Outbound from private to public is normal
            elif not src_private and dst_private:
                return True   # Inbound from public to private might be suspicious
            
            return False
            
        except ipaddress.AddressValueError:
            return False
    
    def _perform_geolocation_analysis(self, analysis: NetworkAnalysis):
        """Perform geolocation analysis on IP addresses"""
        try:
            # This would require a GeoIP database
            # For demonstration, we'll add placeholder geolocation data
            
            unique_ips = set()
            for flow in analysis.network_flows:
                unique_ips.add(flow.source_ip)
                unique_ips.add(flow.destination_ip)
            
            geo_summary = {
                'total_unique_ips': len(unique_ips),
                'external_ips': 0,
                'countries': set(),
                'suspicious_locations': []
            }
            
            for ip in unique_ips:
                try:
                    if not ipaddress.IPv4Address(ip).is_private:
                        geo_summary['external_ips'] += 1
                        # In real implementation, would lookup country
                        # geo_summary['countries'].add(country_code)
                except ipaddress.AddressValueError:
                    continue
            
            # Add geolocation data to flows
            for flow in analysis.network_flows:
                flow.geolocation_data = {
                    'source_country': 'Unknown',  # Would be looked up
                    'destination_country': 'Unknown'  # Would be looked up
                }
            
        except Exception as e:
            self.logger.warning(f"Geolocation analysis failed: {str(e)}")
    
    def _analyze_protocols(self, analysis: NetworkAnalysis):
        """Analyze network protocols used"""
        try:
            protocol_counter = Counter()
            
            for flow in analysis.network_flows:
                protocol = flow.protocol or 'Unknown'
                protocol_counter[protocol] += 1
            
            # Add protocol analysis to communication patterns
            protocol_analysis = {
                'type': 'protocol_distribution',
                'protocols': dict(protocol_counter.most_common()),
                'total_protocols': len(protocol_counter),
                'description': f"Traffic includes {len(protocol_counter)} different protocols"
            }
            
            analysis.communication_patterns.append(protocol_analysis)
            
        except Exception as e:
            self.logger.warning(f"Protocol analysis failed: {str(e)}")
    
    def _generate_network_timeline(self, analysis: NetworkAnalysis):
        """Generate timeline of network events"""
        try:
            # Timeline is implicitly created by the flow timestamps
            # Additional timeline events could be added here
            pass
            
        except Exception as e:
            self.logger.error(f"Network timeline generation failed: {str(e)}")
    
    def _generate_key_findings(self, analysis: NetworkAnalysis):
        """Generate key findings summary"""
        try:
            findings = []
            
            # Connection statistics
            findings.append(
                f"Analyzed {analysis.total_connections:,} network connections "
                f"involving {analysis.unique_destinations:,} unique destinations"
            )
            
            # Data transfer summary
            if analysis.total_data_bytes > 0:
                gb_transferred = analysis.total_data_bytes / (1024 * 1024 * 1024)
                findings.append(f"Total data transferred: {gb_transferred:.2f} GB")
            
            # Time span
            if analysis.earliest_activity and analysis.latest_activity:
                duration = analysis.latest_activity - analysis.earliest_activity
                findings.append(
                    f"Network activity span: {duration.days} days, "
                    f"{duration.seconds // 3600} hours"
                )
            
            # Communication patterns
            if analysis.communication_patterns:
                pattern_count = len([p for p in analysis.communication_patterns 
                                  if p.get('type') != 'protocol_distribution'])
                if pattern_count > 0:
                    findings.append(
                        f"Detected {pattern_count} communication patterns requiring investigation"
                    )
            
            # Suspicious connections
            if analysis.suspicious_connections:
                findings.append(
                    f"Identified {len(analysis.suspicious_connections)} suspicious connections"
                )
            
            # Top destinations
            if analysis.top_destinations:
                top_dest = analysis.top_destinations[0]
                findings.append(
                    f"Most frequent destination: {top_dest['ip']} "
                    f"({top_dest['connections']} connections, {top_dest['percentage']:.1f}%)"
                )
            
            analysis.key_findings.extend(findings)
            
        except Exception as e:
            self.logger.error(f"Key findings generation failed: {str(e)}")
    
    def _parse_timestamp(self, timestamp_str: str) -> datetime:
        """Parse various timestamp formats"""
        formats = [
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%dT%H:%M:%SZ",
            "%Y-%m-%dT%H:%M:%S.%fZ",
            "%d/%b/%Y:%H:%M:%S",
            "%b %d %H:%M:%S",
            "%Y-%m-%d %H:%M:%S.%f",
        ]
        
        for fmt in formats:
            try:
                return datetime.strptime(timestamp_str.strip(), fmt)
            except ValueError:
                continue
        
        # If all formats fail, return current time
        self.logger.warning(f"Could not parse timestamp: {timestamp_str}")
        return datetime.utcnow()
    
    def _parse_apache_timestamp(self, timestamp_str: str) -> datetime:
        """Parse Apache log timestamp format"""
        try:
            # Apache format: [25/Dec/2023:10:00:00 +0000]
            clean_timestamp = timestamp_str.split(' ')[0]  # Remove timezone
            return datetime.strptime(clean_timestamp, "%d/%b/%Y:%H:%M:%S")
        except ValueError:
            return datetime.utcnow()
    
    def _parse_syslog_timestamp(self, timestamp_str: str) -> datetime:
        """Parse syslog timestamp format"""
        try:
            # Syslog format: Dec 25 10:00:00
            current_year = datetime.now().year
            timestamp_with_year = f"{current_year} {timestamp_str}"
            return datetime.strptime(timestamp_with_year, "%Y %b %d %H:%M:%S")
        except ValueError:
            return datetime.utcnow()
    
    def _safe_int_convert(self, value: Any, default: int = None) -> Optional[int]:
        """Safely convert value to integer"""
        if value is None:
            return default
        
        try:
            if isinstance(value, str):
                value = value.strip()
                if value == '-' or value == '':
                    return default
            return int(float(value))  # Handle float strings
        except (ValueError, TypeError):
            return default
    
    def _generate_evidence_id(self) -> str:
        """Generate a unique evidence ID"""
        import uuid
        return str(uuid.uuid4())


def process_network_logs(
    log_files: List[Path], 
    config: Optional[ForensicsConfig] = None
) -> NetworkAnalysis:
    """
    Convenience function to process network log files
    
    Args:
        log_files: List of network log file paths
        config: Optional configuration for analysis
        
    Returns:
        NetworkAnalysis with complete results
    """
    processor = NetworkLogProcessor(config)
    return processor.process_network_logs(log_files)