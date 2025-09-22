"""
Core communication analysis functionality for legal investigations.

Provides analysis of communication patterns, network relationships,
and temporal correlations in various communication formats.
"""

import logging
import re
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
from uuid import uuid4

import networkx as nx
import pandas as pd
from pydantic import BaseModel, Field, validator

logger = logging.getLogger(__name__)


class CommunicationType(str, Enum):
    """Types of communication."""
    EMAIL = "email"
    SMS = "sms"
    PHONE_CALL = "phone_call"
    CHAT = "chat"
    SOCIAL_MEDIA = "social_media"
    MESSAGING_APP = "messaging_app"
    VOICE_MESSAGE = "voice_message"
    VIDEO_CALL = "video_call"


class CommunicationDirection(str, Enum):
    """Direction of communication."""
    INBOUND = "inbound"
    OUTBOUND = "outbound"
    INTERNAL = "internal"


class ContactType(str, Enum):
    """Type of contact entity."""
    INDIVIDUAL = "individual"
    ORGANIZATION = "organization"
    GROUP = "group"
    UNKNOWN = "unknown"


class Contact(BaseModel):
    """Represents a communication contact."""
    contact_id: str = Field(default_factory=lambda: str(uuid4()))
    name: Optional[str] = Field(default=None, description="Contact name")
    email: Optional[str] = Field(default=None, description="Email address")
    phone: Optional[str] = Field(default=None, description="Phone number")
    handle: Optional[str] = Field(default=None, description="Social media handle")
    contact_type: ContactType = Field(default=ContactType.UNKNOWN, description="Type of contact")
    aliases: List[str] = Field(default_factory=list, description="Known aliases")
    organization: Optional[str] = Field(default=None, description="Associated organization")
    role: Optional[str] = Field(default=None, description="Role or title")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class Communication(BaseModel):
    """Represents a single communication event."""
    communication_id: str = Field(default_factory=lambda: str(uuid4()))
    timestamp: datetime = Field(description="When the communication occurred")
    communication_type: CommunicationType = Field(description="Type of communication")
    direction: CommunicationDirection = Field(description="Communication direction")
    sender: Contact = Field(description="Sender of the communication")
    recipients: List[Contact] = Field(description="Recipients of the communication")
    subject: Optional[str] = Field(default=None, description="Subject line or title")
    content: Optional[str] = Field(default=None, description="Communication content")
    attachments: List[str] = Field(default_factory=list, description="Attachment file names")
    duration: Optional[float] = Field(default=None, description="Duration for calls (seconds)")
    thread_id: Optional[str] = Field(default=None, description="Conversation thread ID")
    priority: Optional[str] = Field(default=None, description="Communication priority")
    location: Optional[str] = Field(default=None, description="Location if available")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class NetworkAnalysis(BaseModel):
    """Analysis of communication networks."""
    analysis_id: str = Field(default_factory=lambda: str(uuid4()))
    network_metrics: Dict[str, Any] = Field(description="Network-wide metrics")
    contact_metrics: Dict[str, Dict[str, Any]] = Field(description="Per-contact metrics")
    communities: List[List[str]] = Field(description="Detected communities")
    key_players: List[Dict[str, Any]] = Field(description="Key network players")
    communication_flow: Dict[str, Any] = Field(description="Communication flow analysis")
    centrality_measures: Dict[str, Dict[str, float]] = Field(description="Centrality measures")
    analysis_timestamp: datetime = Field(default_factory=datetime.utcnow)


class PatternResult(BaseModel):
    """Result of pattern detection analysis."""
    pattern_id: str = Field(default_factory=lambda: str(uuid4()))
    pattern_type: str = Field(description="Type of pattern detected")
    description: str = Field(description="Pattern description")
    confidence: float = Field(ge=0.0, le=1.0, description="Detection confidence")
    frequency: int = Field(ge=0, description="Pattern frequency")
    time_period: Tuple[datetime, datetime] = Field(description="Time period of pattern")
    involved_contacts: List[str] = Field(description="Contacts involved in pattern")
    communications: List[str] = Field(description="Communication IDs in pattern")
    significance: str = Field(description="Legal or investigative significance")
    evidence_strength: str = Field(description="Strength as evidence")


class TimelineCorrelation(BaseModel):
    """Timeline correlation analysis result."""
    correlation_id: str = Field(default_factory=lambda: str(uuid4()))
    event_correlations: List[Dict[str, Any]] = Field(description="Correlated events")
    temporal_clusters: List[Dict[str, Any]] = Field(description="Temporal clustering")
    communication_bursts: List[Dict[str, Any]] = Field(description="Communication burst periods")
    quiet_periods: List[Dict[str, Any]] = Field(description="Unusually quiet periods")
    periodic_patterns: List[Dict[str, Any]] = Field(description="Recurring patterns")
    anomalous_times: List[Dict[str, Any]] = Field(description="Anomalous timing events")
    correlation_timestamp: datetime = Field(default_factory=datetime.utcnow)


class CommunicationAnalysis(BaseModel):
    """Comprehensive communication analysis result."""
    analysis_id: str = Field(default_factory=lambda: str(uuid4()))
    data_source: str = Field(description="Source of communication data")
    total_communications: int = Field(ge=0, description="Total communications analyzed")
    unique_contacts: int = Field(ge=0, description="Number of unique contacts")
    time_range: Tuple[datetime, datetime] = Field(description="Analysis time range")
    network_analysis: Optional[NetworkAnalysis] = Field(default=None)
    patterns: List[PatternResult] = Field(default_factory=list)
    timeline_correlation: Optional[TimelineCorrelation] = Field(default=None)
    summary_statistics: Dict[str, Any] = Field(description="Summary statistics")
    legal_insights: List[str] = Field(default_factory=list, description="Legal insights")
    investigation_leads: List[str] = Field(default_factory=list, description="Investigation leads")
    analysis_timestamp: datetime = Field(default_factory=datetime.utcnow)


class CommunicationAnalyzer:
    """Main communication analysis coordinator."""

    def __init__(self):
        """Initialize communication analyzer."""
        self.network_mapper = NetworkMapper()
        self.pattern_detector = PatternDetector()
        self.timeline_correlator = TimelineCorrelator()
        logger.info("Initialized CommunicationAnalyzer")

    def analyze_communications(
        self,
        communications: List[Communication],
        include_network: bool = True,
        include_patterns: bool = True,
        include_timeline: bool = True,
        focus_contacts: Optional[List[str]] = None
    ) -> CommunicationAnalysis:
        """Perform comprehensive communication analysis.

        Args:
            communications: List of communications to analyze
            include_network: Whether to perform network analysis
            include_patterns: Whether to detect patterns
            include_timeline: Whether to perform timeline correlation
            focus_contacts: Specific contacts to focus analysis on

        Returns:
            CommunicationAnalysis with comprehensive results
        """
        try:
            logger.info(f"Analyzing {len(communications)} communications")

            if not communications:
                raise ValueError("No communications provided for analysis")

            # Extract unique contacts
            all_contacts = self._extract_unique_contacts(communications)

            # Calculate time range
            timestamps = [comm.timestamp for comm in communications]
            time_range = (min(timestamps), max(timestamps))

            # Initialize analysis result
            analysis = CommunicationAnalysis(
                data_source="provided_data",
                total_communications=len(communications),
                unique_contacts=len(all_contacts),
                time_range=time_range,
                summary_statistics=self._calculate_summary_stats(communications)
            )

            # Network analysis
            if include_network:
                analysis.network_analysis = self.network_mapper.analyze_network(communications)

            # Pattern detection
            if include_patterns:
                analysis.patterns = self.pattern_detector.detect_patterns(communications)

            # Timeline correlation
            if include_timeline:
                analysis.timeline_correlation = self.timeline_correlator.correlate_timeline(communications)

            # Generate insights
            analysis.legal_insights = self._generate_legal_insights(analysis)
            analysis.investigation_leads = self._generate_investigation_leads(analysis)

            logger.info("Communication analysis completed successfully")
            return analysis

        except Exception as e:
            logger.error(f"Communication analysis failed: {e}")
            raise

    def _extract_unique_contacts(self, communications: List[Communication]) -> Set[Contact]:
        """Extract unique contacts from communications."""
        contacts = set()

        for comm in communications:
            contacts.add(comm.sender)
            contacts.update(comm.recipients)

        return contacts

    def _calculate_summary_stats(self, communications: List[Communication]) -> Dict[str, Any]:
        """Calculate summary statistics for communications."""
        stats = {}

        # Communication type distribution
        type_counts = {}
        for comm in communications:
            comm_type = comm.communication_type
            type_counts[comm_type] = type_counts.get(comm_type, 0) + 1
        stats["communication_types"] = type_counts

        # Direction distribution
        direction_counts = {}
        for comm in communications:
            direction = comm.direction
            direction_counts[direction] = direction_counts.get(direction, 0) + 1
        stats["direction_distribution"] = direction_counts

        # Temporal distribution (by hour of day)
        hourly_counts = [0] * 24
        for comm in communications:
            hour = comm.timestamp.hour
            hourly_counts[hour] += 1
        stats["hourly_distribution"] = hourly_counts

        # Average communications per day
        if communications:
            time_span = (max(comm.timestamp for comm in communications) -
                        min(comm.timestamp for comm in communications)).days
            if time_span > 0:
                stats["avg_communications_per_day"] = len(communications) / time_span
            else:
                stats["avg_communications_per_day"] = len(communications)

        return stats

    def _generate_legal_insights(self, analysis: CommunicationAnalysis) -> List[str]:
        """Generate legal insights from analysis."""
        insights = []

        # High communication volume insights
        if analysis.total_communications > 1000:
            insights.append("High volume of communications may indicate ongoing relationship or coordination")

        # Network centrality insights
        if analysis.network_analysis and analysis.network_analysis.key_players:
            key_player = analysis.network_analysis.key_players[0]
            insights.append(f"Contact '{key_player['contact']}' appears to be central to communication network")

        # Pattern insights
        for pattern in analysis.patterns:
            if pattern.confidence > 0.8:
                insights.append(f"Strong {pattern.pattern_type} pattern detected with {pattern.confidence:.0%} confidence")

        # Timeline insights
        if analysis.timeline_correlation and analysis.timeline_correlation.communication_bursts:
            insights.append("Communication bursts detected - may correlate with significant events")

        return insights

    def _generate_investigation_leads(self, analysis: CommunicationAnalysis) -> List[str]:
        """Generate investigation leads from analysis."""
        leads = []

        # Investigate key network players
        if analysis.network_analysis and analysis.network_analysis.key_players:
            for player in analysis.network_analysis.key_players[:3]:
                leads.append(f"Investigate role and background of '{player['contact']}'")

        # Follow up on suspicious patterns
        suspicious_patterns = [p for p in analysis.patterns if "suspicious" in p.pattern_type.lower()]
        for pattern in suspicious_patterns:
            leads.append(f"Investigate {pattern.pattern_type} involving {len(pattern.involved_contacts)} contacts")

        # Investigate communication gaps
        if analysis.timeline_correlation and analysis.timeline_correlation.quiet_periods:
            leads.append("Investigate reasons for communication quiet periods")

        return leads


class NetworkMapper:
    """Communication network analysis and visualization."""

    def __init__(self):
        """Initialize network mapper."""
        logger.info("Initialized NetworkMapper")

    def analyze_network(self, communications: List[Communication]) -> NetworkAnalysis:
        """Analyze communication network structure.

        Args:
            communications: Communications to analyze

        Returns:
            NetworkAnalysis with network metrics and insights
        """
        try:
            logger.info("Analyzing communication network")

            # Build network graph
            graph = self._build_network_graph(communications)

            # Calculate network metrics
            network_metrics = self._calculate_network_metrics(graph)

            # Calculate per-contact metrics
            contact_metrics = self._calculate_contact_metrics(graph)

            # Detect communities
            communities = self._detect_communities(graph)

            # Identify key players
            key_players = self._identify_key_players(graph, contact_metrics)

            # Analyze communication flow
            communication_flow = self._analyze_communication_flow(communications)

            # Calculate centrality measures
            centrality_measures = self._calculate_centrality_measures(graph)

            return NetworkAnalysis(
                network_metrics=network_metrics,
                contact_metrics=contact_metrics,
                communities=communities,
                key_players=key_players,
                communication_flow=communication_flow,
                centrality_measures=centrality_measures
            )

        except Exception as e:
            logger.error(f"Network analysis failed: {e}")
            raise

    def _build_network_graph(self, communications: List[Communication]) -> nx.Graph:
        """Build NetworkX graph from communications."""
        graph = nx.Graph()

        for comm in communications:
            sender_id = comm.sender.contact_id

            # Add sender node
            if not graph.has_node(sender_id):
                graph.add_node(sender_id,
                             name=comm.sender.name or "Unknown",
                             type=comm.sender.contact_type)

            # Add recipient nodes and edges
            for recipient in comm.recipients:
                recipient_id = recipient.contact_id

                # Add recipient node
                if not graph.has_node(recipient_id):
                    graph.add_node(recipient_id,
                                 name=recipient.name or "Unknown",
                                 type=recipient.contact_type)

                # Add edge (increment weight if exists)
                if graph.has_edge(sender_id, recipient_id):
                    graph[sender_id][recipient_id]['weight'] += 1
                else:
                    graph.add_edge(sender_id, recipient_id, weight=1)

        return graph

    def _calculate_network_metrics(self, graph: nx.Graph) -> Dict[str, Any]:
        """Calculate overall network metrics."""
        metrics = {}

        if graph.number_of_nodes() > 0:
            metrics["node_count"] = graph.number_of_nodes()
            metrics["edge_count"] = graph.number_of_edges()
            metrics["density"] = nx.density(graph)

            if nx.is_connected(graph):
                metrics["diameter"] = nx.diameter(graph)
                metrics["average_path_length"] = nx.average_shortest_path_length(graph)
            else:
                metrics["connected_components"] = nx.number_connected_components(graph)

            metrics["clustering_coefficient"] = nx.average_clustering(graph)

        return metrics

    def _calculate_contact_metrics(self, graph: nx.Graph) -> Dict[str, Dict[str, Any]]:
        """Calculate per-contact network metrics."""
        metrics = {}

        if graph.number_of_nodes() > 0:
            degree_centrality = nx.degree_centrality(graph)
            betweenness_centrality = nx.betweenness_centrality(graph)
            closeness_centrality = nx.closeness_centrality(graph)

            for node in graph.nodes():
                metrics[node] = {
                    "degree": graph.degree(node),
                    "degree_centrality": degree_centrality[node],
                    "betweenness_centrality": betweenness_centrality[node],
                    "closeness_centrality": closeness_centrality[node],
                    "clustering": nx.clustering(graph, node)
                }

        return metrics

    def _detect_communities(self, graph: nx.Graph) -> List[List[str]]:
        """Detect communities in the network."""
        try:
            if graph.number_of_nodes() > 2:
                # Use simple greedy modularity maximization
                communities = nx.community.greedy_modularity_communities(graph)
                return [list(community) for community in communities]
            else:
                return [list(graph.nodes())]
        except Exception as e:
            logger.warning(f"Community detection failed: {e}")
            return []

    def _identify_key_players(self, graph: nx.Graph, contact_metrics: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Identify key players in the network."""
        key_players = []

        if contact_metrics:
            # Sort by degree centrality
            sorted_contacts = sorted(
                contact_metrics.items(),
                key=lambda x: x[1]["degree_centrality"],
                reverse=True
            )

            for contact_id, metrics in sorted_contacts[:10]:
                node_data = graph.nodes[contact_id]
                key_players.append({
                    "contact": node_data.get("name", contact_id),
                    "contact_id": contact_id,
                    "degree": metrics["degree"],
                    "degree_centrality": metrics["degree_centrality"],
                    "betweenness_centrality": metrics["betweenness_centrality"],
                    "role": "hub" if metrics["degree_centrality"] > 0.1 else "connector" if metrics["betweenness_centrality"] > 0.1 else "regular"
                })

        return key_players

    def _analyze_communication_flow(self, communications: List[Communication]) -> Dict[str, Any]:
        """Analyze communication flow patterns."""
        flow_analysis = {}

        # Direction analysis
        direction_counts = {}
        for comm in communications:
            direction = comm.direction
            direction_counts[direction] = direction_counts.get(direction, 0) + 1

        flow_analysis["direction_distribution"] = direction_counts

        # Type flow analysis
        type_flow = {}
        for comm in communications:
            comm_type = comm.communication_type
            direction = comm.direction
            key = f"{comm_type}_{direction}"
            type_flow[key] = type_flow.get(key, 0) + 1

        flow_analysis["type_direction_flow"] = type_flow

        return flow_analysis

    def _calculate_centrality_measures(self, graph: nx.Graph) -> Dict[str, Dict[str, float]]:
        """Calculate various centrality measures."""
        centrality = {}

        if graph.number_of_nodes() > 0:
            centrality["degree"] = nx.degree_centrality(graph)
            centrality["betweenness"] = nx.betweenness_centrality(graph)
            centrality["closeness"] = nx.closeness_centrality(graph)

            try:
                centrality["eigenvector"] = nx.eigenvector_centrality(graph, max_iter=1000)
            except:
                centrality["eigenvector"] = {}

        return centrality


class PatternDetector:
    """Communication pattern detection."""

    def __init__(self):
        """Initialize pattern detector."""
        logger.info("Initialized PatternDetector")

    def detect_patterns(self, communications: List[Communication]) -> List[PatternResult]:
        """Detect patterns in communications.

        Args:
            communications: Communications to analyze

        Returns:
            List of detected patterns
        """
        try:
            logger.info("Detecting communication patterns")

            patterns = []

            # Detect various pattern types
            patterns.extend(self._detect_burst_patterns(communications))
            patterns.extend(self._detect_coordination_patterns(communications))
            patterns.extend(self._detect_escalation_patterns(communications))
            patterns.extend(self._detect_regular_meeting_patterns(communications))
            patterns.extend(self._detect_suspicious_timing_patterns(communications))

            logger.info(f"Detected {len(patterns)} communication patterns")
            return patterns

        except Exception as e:
            logger.error(f"Pattern detection failed: {e}")
            return []

    def _detect_burst_patterns(self, communications: List[Communication]) -> List[PatternResult]:
        """Detect communication burst patterns."""
        patterns = []

        if len(communications) < 10:
            return patterns

        # Sort by timestamp
        sorted_comms = sorted(communications, key=lambda x: x.timestamp)

        # Look for periods with unusually high communication
        burst_threshold = len(communications) / 30  # Expected per day
        current_burst = []
        burst_start = None

        for i, comm in enumerate(sorted_comms):
            if i == 0:
                current_burst = [comm]
                burst_start = comm.timestamp
                continue

            time_diff = (comm.timestamp - sorted_comms[i-1].timestamp).total_seconds()

            if time_diff <= 3600:  # Within 1 hour
                current_burst.append(comm)
            else:
                # Check if current burst is significant
                if len(current_burst) > burst_threshold:
                    pattern = PatternResult(
                        pattern_type="communication_burst",
                        description=f"Communication burst of {len(current_burst)} messages in short period",
                        confidence=0.8,
                        frequency=len(current_burst),
                        time_period=(burst_start, current_burst[-1].timestamp),
                        involved_contacts=[comm.sender.contact_id for comm in current_burst],
                        communications=[comm.communication_id for comm in current_burst],
                        significance="May indicate coordinated activity or crisis response",
                        evidence_strength="moderate"
                    )
                    patterns.append(pattern)

                # Start new burst
                current_burst = [comm]
                burst_start = comm.timestamp

        return patterns

    def _detect_coordination_patterns(self, communications: List[Communication]) -> List[PatternResult]:
        """Detect coordination patterns between multiple parties."""
        patterns = []

        # Group communications by time windows
        time_windows = {}
        window_size = timedelta(hours=2)

        for comm in communications:
            window_start = comm.timestamp.replace(minute=0, second=0, microsecond=0)
            window_key = window_start.isoformat()

            if window_key not in time_windows:
                time_windows[window_key] = []
            time_windows[window_key].append(comm)

        # Look for windows with multiple parties communicating
        for window_key, window_comms in time_windows.items():
            if len(window_comms) >= 3:
                unique_contacts = set()
                for comm in window_comms:
                    unique_contacts.add(comm.sender.contact_id)
                    unique_contacts.update(r.contact_id for r in comm.recipients)

                if len(unique_contacts) >= 3:
                    pattern = PatternResult(
                        pattern_type="coordination_pattern",
                        description=f"Coordinated communication involving {len(unique_contacts)} parties",
                        confidence=0.7,
                        frequency=len(window_comms),
                        time_period=(min(c.timestamp for c in window_comms),
                                   max(c.timestamp for c in window_comms)),
                        involved_contacts=list(unique_contacts),
                        communications=[c.communication_id for c in window_comms],
                        significance="May indicate planning or coordination activity",
                        evidence_strength="moderate"
                    )
                    patterns.append(pattern)

        return patterns

    def _detect_escalation_patterns(self, communications: List[Communication]) -> List[PatternResult]:
        """Detect escalation patterns in communication frequency."""
        patterns = []

        # Group by day and analyze frequency trends
        daily_counts = {}
        for comm in communications:
            day_key = comm.timestamp.date().isoformat()
            daily_counts[day_key] = daily_counts.get(day_key, 0) + 1

        # Look for increasing trends
        sorted_days = sorted(daily_counts.items())
        if len(sorted_days) >= 7:  # Need at least a week of data
            # Simple escalation detection
            recent_avg = sum(count for _, count in sorted_days[-3:]) / 3
            earlier_avg = sum(count for _, count in sorted_days[:3]) / 3

            if recent_avg > earlier_avg * 2:  # 2x increase
                pattern = PatternResult(
                    pattern_type="escalation_pattern",
                    description="Communication frequency escalation detected",
                    confidence=0.75,
                    frequency=int(recent_avg - earlier_avg),
                    time_period=(datetime.fromisoformat(sorted_days[0][0]),
                               datetime.fromisoformat(sorted_days[-1][0])),
                    involved_contacts=[],  # Would need more detailed analysis
                    communications=[],
                    significance="May indicate escalating situation or crisis",
                    evidence_strength="moderate"
                )
                patterns.append(pattern)

        return patterns

    def _detect_regular_meeting_patterns(self, communications: List[Communication]) -> List[PatternResult]:
        """Detect regular meeting or contact patterns."""
        patterns = []

        # Group by day of week and hour
        weekly_patterns = {}
        for comm in communications:
            if comm.communication_type in [CommunicationType.PHONE_CALL, CommunicationType.VIDEO_CALL]:
                day_hour = f"{comm.timestamp.weekday()}_{comm.timestamp.hour}"
                if day_hour not in weekly_patterns:
                    weekly_patterns[day_hour] = []
                weekly_patterns[day_hour].append(comm)

        # Look for regular patterns
        for day_hour, comms in weekly_patterns.items():
            if len(comms) >= 3:  # At least 3 occurrences
                day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
                day_idx, hour = map(int, day_hour.split("_"))

                pattern = PatternResult(
                    pattern_type="regular_meeting_pattern",
                    description=f"Regular {day_names[day_idx]} meetings at {hour:02d}:00",
                    confidence=0.8,
                    frequency=len(comms),
                    time_period=(min(c.timestamp for c in comms),
                               max(c.timestamp for c in comms)),
                    involved_contacts=list(set(c.sender.contact_id for c in comms)),
                    communications=[c.communication_id for c in comms],
                    significance="Indicates regular contact or scheduled meetings",
                    evidence_strength="strong"
                )
                patterns.append(pattern)

        return patterns

    def _detect_suspicious_timing_patterns(self, communications: List[Communication]) -> List[PatternResult]:
        """Detect suspicious timing patterns (e.g., unusual hours)."""
        patterns = []

        # Count communications by hour
        hourly_counts = [0] * 24
        unusual_hour_comms = []

        for comm in communications:
            hour = comm.timestamp.hour
            hourly_counts[hour] += 1

            # Flag communications in unusual hours (late night/early morning)
            if hour <= 5 or hour >= 23:
                unusual_hour_comms.append(comm)

        # If more than 10% of communications are at unusual hours
        if len(unusual_hour_comms) > len(communications) * 0.1:
            pattern = PatternResult(
                pattern_type="suspicious_timing_pattern",
                description="High frequency of communications during unusual hours",
                confidence=0.6,
                frequency=len(unusual_hour_comms),
                time_period=(min(c.timestamp for c in unusual_hour_comms),
                           max(c.timestamp for c in unusual_hour_comms)),
                involved_contacts=list(set(c.sender.contact_id for c in unusual_hour_comms)),
                communications=[c.communication_id for c in unusual_hour_comms],
                significance="May indicate covert activity or different time zones",
                evidence_strength="weak"
            )
            patterns.append(pattern)

        return patterns


class TimelineCorrelator:
    """Timeline correlation analysis."""

    def __init__(self):
        """Initialize timeline correlator."""
        logger.info("Initialized TimelineCorrelator")

    def correlate_timeline(self, communications: List[Communication]) -> TimelineCorrelation:
        """Correlate communications with timeline events.

        Args:
            communications: Communications to correlate

        Returns:
            TimelineCorrelation with temporal analysis
        """
        try:
            logger.info("Correlating communication timeline")

            # Analyze temporal patterns
            temporal_clusters = self._find_temporal_clusters(communications)
            communication_bursts = self._identify_communication_bursts(communications)
            quiet_periods = self._identify_quiet_periods(communications)
            periodic_patterns = self._find_periodic_patterns(communications)
            anomalous_times = self._detect_anomalous_timing(communications)

            return TimelineCorrelation(
                event_correlations=[],  # Would correlate with external events
                temporal_clusters=temporal_clusters,
                communication_bursts=communication_bursts,
                quiet_periods=quiet_periods,
                periodic_patterns=periodic_patterns,
                anomalous_times=anomalous_times
            )

        except Exception as e:
            logger.error(f"Timeline correlation failed: {e}")
            raise

    def _find_temporal_clusters(self, communications: List[Communication]) -> List[Dict[str, Any]]:
        """Find temporal clusters in communications."""
        if not communications:
            return []

        # Sort by timestamp
        sorted_comms = sorted(communications, key=lambda x: x.timestamp)
        clusters = []
        current_cluster = [sorted_comms[0]]

        for i in range(1, len(sorted_comms)):
            time_diff = (sorted_comms[i].timestamp - sorted_comms[i-1].timestamp).total_seconds()

            if time_diff <= 3600:  # Within 1 hour
                current_cluster.append(sorted_comms[i])
            else:
                if len(current_cluster) >= 3:
                    clusters.append({
                        "start_time": current_cluster[0].timestamp,
                        "end_time": current_cluster[-1].timestamp,
                        "communication_count": len(current_cluster),
                        "duration_minutes": (current_cluster[-1].timestamp - current_cluster[0].timestamp).total_seconds() / 60
                    })
                current_cluster = [sorted_comms[i]]

        # Don't forget the last cluster
        if len(current_cluster) >= 3:
            clusters.append({
                "start_time": current_cluster[0].timestamp,
                "end_time": current_cluster[-1].timestamp,
                "communication_count": len(current_cluster),
                "duration_minutes": (current_cluster[-1].timestamp - current_cluster[0].timestamp).total_seconds() / 60
            })

        return clusters

    def _identify_communication_bursts(self, communications: List[Communication]) -> List[Dict[str, Any]]:
        """Identify communication burst periods."""
        # Implementation similar to pattern detection but focused on timeline
        return []

    def _identify_quiet_periods(self, communications: List[Communication]) -> List[Dict[str, Any]]:
        """Identify unusually quiet periods."""
        if len(communications) < 10:
            return []

        sorted_comms = sorted(communications, key=lambda x: x.timestamp)
        quiet_periods = []

        # Calculate average time between communications
        time_diffs = []
        for i in range(1, len(sorted_comms)):
            diff = (sorted_comms[i].timestamp - sorted_comms[i-1].timestamp).total_seconds()
            time_diffs.append(diff)

        if not time_diffs:
            return []

        avg_diff = sum(time_diffs) / len(time_diffs)
        threshold = avg_diff * 5  # 5x average is "quiet"

        for i in range(len(time_diffs)):
            if time_diffs[i] > threshold:
                quiet_periods.append({
                    "start_time": sorted_comms[i].timestamp,
                    "end_time": sorted_comms[i+1].timestamp,
                    "duration_hours": time_diffs[i] / 3600,
                    "significance": "unusual_quiet_period"
                })

        return quiet_periods

    def _find_periodic_patterns(self, communications: List[Communication]) -> List[Dict[str, Any]]:
        """Find recurring periodic patterns."""
        # Analyze weekly and daily patterns
        return []

    def _detect_anomalous_timing(self, communications: List[Communication]) -> List[Dict[str, Any]]:
        """Detect anomalous timing in communications."""
        anomalies = []

        # Weekend communications
        weekend_comms = [c for c in communications if c.timestamp.weekday() >= 5]
        if len(weekend_comms) > len(communications) * 0.3:
            anomalies.append({
                "anomaly_type": "high_weekend_activity",
                "description": f"{len(weekend_comms)} weekend communications ({len(weekend_comms)/len(communications)*100:.1f}%)",
                "significance": "unusual_schedule"
            })

        return anomalies