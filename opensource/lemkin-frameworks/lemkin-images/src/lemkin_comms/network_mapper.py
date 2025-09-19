"""
Communication Network Mapper

Creates and visualizes communication networks from various communication data.
Provides network analysis, community detection, and suspicious pattern identification
for forensic investigations.
"""

import networkx as nx
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Set
from collections import defaultdict, Counter
import json
import logging

# Visualization imports
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import seaborn as sns

from .core import (
    Communication, NetworkGraph, NetworkNode, NetworkEdge, 
    CommunicationNetwork, Contact, CommsConfig, PlatformType
)

logger = logging.getLogger(__name__)


class CommunicationNetworkBuilder:
    """Builds communication networks from raw communication data"""
    
    def __init__(self, config: CommsConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def build_network(self, communications: List[Communication]) -> CommunicationNetwork:
        """Build network structure from communications"""
        # Extract all participants
        participants = set()
        for comm in communications:
            participants.add(comm.sender_id)
            participants.update(comm.recipient_ids)
        
        # Build nodes
        nodes = []
        participant_stats = self._calculate_participant_stats(communications, participants)
        
        for participant_id in participants:
            stats = participant_stats[participant_id]
            
            node = NetworkNode(
                node_id=participant_id,
                contact=Contact(
                    contact_id=participant_id,
                    platforms=list(set(stats['platforms'])),
                    first_seen=stats['first_seen'],
                    last_seen=stats['last_seen'],
                    total_messages=stats['message_count']
                ),
                message_count=stats['message_count'],
                connection_count=stats['connection_count'],
                activity_level=self._classify_activity_level(stats['message_count']),
                centrality_score=0.0,  # Will be calculated later
                suspicious_score=0.0   # Will be calculated later
            )
            nodes.append(node)
        
        # Build edges
        edges = self._build_edges(communications, participants)
        
        # Calculate network metrics
        total_messages = len(communications)
        date_range = (
            min(c.timestamp for c in communications),
            max(c.timestamp for c in communications)
        )
        platforms = list(set(c.platform for c in communications))
        
        # Create network
        network = CommunicationNetwork(
            nodes=nodes,
            edges=edges,
            total_messages=total_messages,
            date_range=date_range,
            platforms=platforms,
            density=0.0,  # Will be calculated later
            clustering_coefficient=0.0  # Will be calculated later
        )
        
        # Update with calculated metrics
        self._calculate_network_metrics(network)
        
        return network
    
    def _calculate_participant_stats(
        self, 
        communications: List[Communication], 
        participants: Set[str]
    ) -> Dict[str, Dict[str, Any]]:
        """Calculate statistics for each participant"""
        stats = {}
        
        for participant in participants:
            # Find all messages involving this participant
            participant_comms = [
                c for c in communications 
                if c.sender_id == participant or participant in c.recipient_ids
            ]
            
            # Calculate stats
            sent_count = len([c for c in participant_comms if c.sender_id == participant])
            received_count = len([c for c in participant_comms if participant in c.recipient_ids])
            
            # Find unique contacts
            contacts = set()
            for comm in participant_comms:
                if comm.sender_id == participant:
                    contacts.update(comm.recipient_ids)
                else:
                    contacts.add(comm.sender_id)
            
            platforms = list(set(c.platform for c in participant_comms))
            
            stats[participant] = {
                'message_count': sent_count + received_count,
                'sent_count': sent_count,
                'received_count': received_count,
                'connection_count': len(contacts),
                'platforms': platforms,
                'first_seen': min(c.timestamp for c in participant_comms),
                'last_seen': max(c.timestamp for c in participant_comms),
                'contacts': contacts
            }
        
        return stats
    
    def _classify_activity_level(self, message_count: int) -> str:
        """Classify participant activity level"""
        if message_count < 10:
            return "low"
        elif message_count < 100:
            return "medium"
        elif message_count < 1000:
            return "high"
        else:
            return "very_high"
    
    def _build_edges(
        self, 
        communications: List[Communication], 
        participants: Set[str]
    ) -> List[NetworkEdge]:
        """Build network edges from communications"""
        edges = {}  # (source, target) -> edge data
        
        for comm in communications:
            sender = comm.sender_id
            
            for recipient in comm.recipient_ids:
                # Create edge key (always source -> target alphabetically for consistency)
                edge_key = tuple(sorted([sender, recipient]))
                
                if edge_key not in edges:
                    edges[edge_key] = {
                        'message_count': 0,
                        'platforms': set(),
                        'first_contact': comm.timestamp,
                        'last_contact': comm.timestamp,
                        'sender_to_recipient': 0,
                        'recipient_to_sender': 0
                    }
                
                # Update edge data
                edge_data = edges[edge_key]
                edge_data['message_count'] += 1
                edge_data['platforms'].add(comm.platform)
                edge_data['first_contact'] = min(edge_data['first_contact'], comm.timestamp)
                edge_data['last_contact'] = max(edge_data['last_contact'], comm.timestamp)
                
                # Track direction
                if sender == edge_key[0]:
                    edge_data['sender_to_recipient'] += 1
                else:
                    edge_data['recipient_to_sender'] += 1
        
        # Convert to NetworkEdge objects
        network_edges = []
        for (node1, node2), data in edges.items():
            # Calculate relationship strength
            strength = min(1.0, data['message_count'] / 50.0)  # Normalize to 0-1
            
            # Check if bidirectional
            is_bidirectional = (data['sender_to_recipient'] > 0 and 
                              data['recipient_to_sender'] > 0)
            
            edge = NetworkEdge(
                source_id=node1,
                target_id=node2,
                message_count=data['message_count'],
                first_contact=data['first_contact'],
                last_contact=data['last_contact'],
                platforms=list(data['platforms']),
                relationship_strength=strength,
                is_bidirectional=is_bidirectional
            )
            network_edges.append(edge)
        
        return network_edges
    
    def _calculate_network_metrics(self, network: CommunicationNetwork):
        """Calculate network-level metrics"""
        # Create NetworkX graph for calculations
        G = nx.Graph()
        
        # Add nodes
        for node in network.nodes:
            G.add_node(node.node_id, **{
                'message_count': node.message_count,
                'activity_level': node.activity_level
            })
        
        # Add edges
        for edge in network.edges:
            G.add_edge(
                edge.source_id, 
                edge.target_id, 
                weight=edge.relationship_strength,
                message_count=edge.message_count
            )
        
        # Calculate network metrics
        if len(G.nodes) > 1:
            network.density = nx.density(G)
            network.clustering_coefficient = nx.average_clustering(G)
            
            # Calculate centrality scores
            centrality = nx.degree_centrality(G)
            betweenness = nx.betweenness_centrality(G)
            
            # Update node centrality scores
            for node in network.nodes:
                if node.node_id in centrality:
                    node.centrality_score = (
                        centrality[node.node_id] * 0.7 + 
                        betweenness[node.node_id] * 0.3
                    )


class NetworkVisualizer:
    """Creates visualizations of communication networks"""
    
    def __init__(self, config: CommsConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def create_interactive_network(
        self, 
        network: CommunicationNetwork,
        output_path: Optional[Path] = None
    ) -> go.Figure:
        """Create interactive network visualization using Plotly"""
        
        # Create NetworkX graph for layout
        G = nx.Graph()
        
        # Add nodes and edges
        for node in network.nodes:
            G.add_node(node.node_id, 
                      message_count=node.message_count,
                      activity_level=node.activity_level)
        
        for edge in network.edges:
            G.add_edge(edge.source_id, edge.target_id, 
                      weight=edge.relationship_strength)
        
        # Calculate layout
        pos = nx.spring_layout(G, k=1, iterations=50)
        
        # Prepare node traces
        node_trace = self._create_node_trace(network.nodes, pos)
        edge_traces = self._create_edge_traces(network.edges, pos)
        
        # Create figure
        fig = go.Figure(
            data=edge_traces + [node_trace],
            layout=go.Layout(
                title=f'Communication Network ({len(network.nodes)} nodes, {len(network.edges)} edges)',
                titlefont_size=16,
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20, l=5, r=5, t=40),
                annotations=[
                    dict(
                        text=f"Network Density: {network.density:.3f}<br>" +
                             f"Clustering: {network.clustering_coefficient:.3f}<br>" +
                             f"Platforms: {', '.join(network.platforms)}",
                        showarrow=False,
                        xref="paper", yref="paper",
                        x=0.005, y=-0.002,
                        xanchor='left', yanchor='bottom',
                        font=dict(size=10)
                    )
                ],
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
            )
        )
        
        if output_path:
            fig.write_html(str(output_path))
            self.logger.info(f"Interactive network saved to {output_path}")
        
        return fig
    
    def _create_node_trace(self, nodes: List[NetworkNode], pos: Dict) -> go.Scatter:
        """Create node trace for Plotly visualization"""
        x_coords = []
        y_coords = []
        node_info = []
        node_colors = []
        node_sizes = []
        
        activity_colors = {
            'low': 'lightblue',
            'medium': 'orange', 
            'high': 'red',
            'very_high': 'darkred'
        }
        
        for node in nodes:
            if node.node_id in pos:
                x, y = pos[node.node_id]
                x_coords.append(x)
                y_coords.append(y)
                
                # Node info for hover
                info = (f"Contact: {node.node_id}<br>"
                       f"Messages: {node.message_count}<br>"
                       f"Connections: {node.connection_count}<br>"
                       f"Activity: {node.activity_level}<br>"
                       f"Centrality: {node.centrality_score:.3f}")
                node_info.append(info)
                
                # Node styling
                node_colors.append(activity_colors.get(node.activity_level, 'gray'))
                node_sizes.append(max(10, min(50, node.message_count / 10)))
        
        return go.Scatter(
            x=x_coords, y=y_coords,
            mode='markers',
            hoverinfo='text',
            text=node_info,
            marker=dict(
                size=node_sizes,
                color=node_colors,
                line=dict(width=2, color='black')
            )
        )
    
    def _create_edge_traces(self, edges: List[NetworkEdge], pos: Dict) -> List[go.Scatter]:
        """Create edge traces for Plotly visualization"""
        edge_traces = []
        
        for edge in edges:
            if edge.source_id in pos and edge.target_id in pos:
                x0, y0 = pos[edge.source_id]
                x1, y1 = pos[edge.target_id]
                
                # Line width based on relationship strength
                line_width = max(1, edge.relationship_strength * 10)
                
                # Line color based on bidirectionality
                line_color = 'blue' if edge.is_bidirectional else 'gray'
                
                edge_trace = go.Scatter(
                    x=[x0, x1, None], y=[y0, y1, None],
                    mode='lines',
                    line=dict(width=line_width, color=line_color),
                    hoverinfo='none'
                )
                edge_traces.append(edge_trace)
        
        return edge_traces
    
    def create_network_metrics_dashboard(
        self, 
        network: CommunicationNetwork,
        output_path: Optional[Path] = None
    ) -> go.Figure:
        """Create dashboard with network metrics"""
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Node Activity Distribution', 'Platform Usage',
                          'Centrality vs Activity', 'Connection Distribution'),
            specs=[[{"type": "bar"}, {"type": "pie"}],
                   [{"type": "scatter"}, {"type": "histogram"}]]
        )
        
        # Activity distribution
        activity_counts = Counter(node.activity_level for node in network.nodes)
        fig.add_trace(
            go.Bar(x=list(activity_counts.keys()), y=list(activity_counts.values()),
                   name="Activity Levels"),
            row=1, col=1
        )
        
        # Platform usage
        platform_counts = Counter(platform for platform in network.platforms)
        fig.add_trace(
            go.Pie(labels=list(platform_counts.keys()), 
                   values=list(platform_counts.values()),
                   name="Platforms"),
            row=1, col=2
        )
        
        # Centrality vs Activity
        centralities = [node.centrality_score for node in network.nodes]
        message_counts = [node.message_count for node in network.nodes]
        fig.add_trace(
            go.Scatter(x=centralities, y=message_counts, mode='markers',
                      name="Centrality vs Messages"),
            row=2, col=1
        )
        
        # Connection distribution
        connection_counts = [node.connection_count for node in network.nodes]
        fig.add_trace(
            go.Histogram(x=connection_counts, name="Connections"),
            row=2, col=2
        )
        
        fig.update_layout(
            height=800,
            title_text="Communication Network Analysis Dashboard",
            showlegend=False
        )
        
        if output_path:
            fig.write_html(str(output_path))
            self.logger.info(f"Network dashboard saved to {output_path}")
        
        return fig
    
    def create_temporal_network_animation(
        self,
        communications: List[Communication],
        time_window_days: int = 7,
        output_path: Optional[Path] = None
    ) -> go.Figure:
        """Create animated visualization showing network evolution over time"""
        
        # Sort communications by timestamp
        communications.sort(key=lambda c: c.timestamp)
        
        if not communications:
            return go.Figure()
        
        # Create time windows
        start_date = communications[0].timestamp
        end_date = communications[-1].timestamp
        total_days = (end_date - start_date).days
        
        if total_days < time_window_days:
            time_window_days = max(1, total_days)
        
        frames = []
        current_date = start_date
        
        while current_date <= end_date:
            window_end = current_date + timedelta(days=time_window_days)
            
            # Get communications in this window
            window_comms = [
                c for c in communications 
                if current_date <= c.timestamp <= window_end
            ]
            
            if window_comms:
                # Build network for this window
                builder = CommunicationNetworkBuilder(self.config)
                window_network = builder.build_network(window_comms)
                
                # Create frame
                frame_data = self._create_network_frame(window_network)
                frames.append(go.Frame(
                    data=frame_data,
                    name=current_date.strftime('%Y-%m-%d')
                ))
            
            current_date += timedelta(days=time_window_days)
        
        # Create initial figure
        if frames:
            fig = go.Figure(
                data=frames[0].data,
                frames=frames
            )
            
            # Add animation controls
            fig.update_layout(
                title="Communication Network Evolution Over Time",
                updatemenus=[{
                    "buttons": [
                        {
                            "args": [None, {"frame": {"duration": 500, "redraw": True},
                                          "fromcurrent": True}],
                            "label": "Play",
                            "method": "animate"
                        },
                        {
                            "args": [[None], {"frame": {"duration": 0, "redraw": True},
                                            "mode": "immediate",
                                            "transition": {"duration": 0}}],
                            "label": "Pause",
                            "method": "animate"
                        }
                    ],
                    "direction": "left",
                    "pad": {"r": 10, "t": 87},
                    "showactive": False,
                    "type": "buttons",
                    "x": 0.1,
                    "xanchor": "right",
                    "y": 0,
                    "yanchor": "top"
                }]
            )
            
            if output_path:
                fig.write_html(str(output_path))
                self.logger.info(f"Temporal network animation saved to {output_path}")
        
        else:
            fig = go.Figure()
        
        return fig
    
    def _create_network_frame(self, network: CommunicationNetwork) -> List[go.Scatter]:
        """Create frame data for animated network"""
        # Simplified version - create basic node/edge traces
        G = nx.Graph()
        
        for node in network.nodes:
            G.add_node(node.node_id)
        
        for edge in network.edges:
            G.add_edge(edge.source_id, edge.target_id)
        
        pos = nx.spring_layout(G, k=1, iterations=30)
        
        node_trace = self._create_node_trace(network.nodes, pos)
        edge_traces = self._create_edge_traces(network.edges, pos)
        
        return edge_traces + [node_trace]


class NetworkAnalyzer:
    """Analyzes network structure for suspicious patterns and communities"""
    
    def __init__(self, config: CommsConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def detect_communities(self, network: CommunicationNetwork) -> List[List[str]]:
        """Detect communities in the communication network"""
        # Create NetworkX graph
        G = nx.Graph()
        
        for node in network.nodes:
            G.add_node(node.node_id)
        
        for edge in network.edges:
            G.add_edge(edge.source_id, edge.target_id, 
                      weight=edge.relationship_strength)
        
        communities = []
        
        try:
            if self.config.community_detection_algorithm == "louvain":
                import community as community_louvain
                partition = community_louvain.best_partition(G)
                
                # Group nodes by community
                community_dict = defaultdict(list)
                for node, comm_id in partition.items():
                    community_dict[comm_id].append(node)
                
                communities = list(community_dict.values())
            
            else:
                # Fallback to simple connected components
                components = nx.connected_components(G)
                communities = [list(comp) for comp in components if len(comp) > 1]
        
        except ImportError:
            self.logger.warning("python-louvain not available, using connected components")
            components = nx.connected_components(G)
            communities = [list(comp) for comp in components if len(comp) > 1]
        
        return communities
    
    def identify_central_figures(self, network: CommunicationNetwork) -> List[str]:
        """Identify most central/influential figures in network"""
        # Sort nodes by centrality score
        central_nodes = sorted(
            network.nodes,
            key=lambda n: n.centrality_score,
            reverse=True
        )
        
        # Return top 10 or 20% of nodes, whichever is smaller
        max_count = min(10, max(1, len(network.nodes) // 5))
        return [node.node_id for node in central_nodes[:max_count]]
    
    def find_suspicious_clusters(self, network: CommunicationNetwork) -> List[List[str]]:
        """Identify potentially suspicious communication clusters"""
        suspicious_clusters = []
        
        # Create NetworkX graph
        G = nx.Graph()
        
        node_suspicion = {}
        for node in network.nodes:
            G.add_node(node.node_id)
            
            # Calculate suspicion score based on various factors
            suspicion = 0.0
            
            # High activity in short time span
            if node.contact.first_seen and node.contact.last_seen:
                time_span = (node.contact.last_seen - node.contact.first_seen).days
                if time_span < 7 and node.message_count > 50:  # High activity in < 1 week
                    suspicion += 0.3
            
            # Unusual activity patterns
            if node.activity_level in ['high', 'very_high'] and node.connection_count < 3:
                suspicion += 0.2  # High activity but few connections
            
            # Platform switching
            if len(node.contact.platforms) > 2:
                suspicion += 0.1
            
            node_suspicion[node.node_id] = suspicion
        
        for edge in network.edges:
            G.add_edge(edge.source_id, edge.target_id)
        
        # Find clusters of suspicious nodes
        communities = self.detect_communities(network)
        
        for community in communities:
            if len(community) >= 3:  # Minimum cluster size
                avg_suspicion = np.mean([node_suspicion.get(node, 0) for node in community])
                
                if avg_suspicion > 0.3:  # Suspicion threshold
                    suspicious_clusters.append(community)
        
        return suspicious_clusters
    
    def find_isolated_nodes(self, network: CommunicationNetwork) -> List[str]:
        """Find nodes with very few connections (potential outliers)"""
        isolated = []
        
        # Define isolation threshold (less than 2 connections or very low activity)
        for node in network.nodes:
            if (node.connection_count <= 1 or 
                (node.connection_count <= 2 and node.message_count < 5)):
                isolated.append(node.node_id)
        
        return isolated


class NetworkMapper:
    """Main network mapping class that orchestrates network analysis"""
    
    def __init__(self, config: CommsConfig):
        self.config = config
        self.network_builder = CommunicationNetworkBuilder(config)
        self.visualizer = NetworkVisualizer(config)
        self.analyzer = NetworkAnalyzer(config)
        self.logger = logging.getLogger(__name__)
    
    def analyze(self, communications: List[Communication]) -> NetworkGraph:
        """Perform complete network analysis"""
        
        if not communications:
            return NetworkGraph(
                network=CommunicationNetwork(
                    nodes=[], edges=[], total_messages=0,
                    date_range=(datetime.now(), datetime.now()),
                    platforms=[], density=0.0, clustering_coefficient=0.0
                ),
                layout_data={},
                visualization_config={},
                metrics={},
                communities=[],
                central_figures=[],
                isolated_nodes=[],
                suspicious_clusters=[]
            )
        
        # Build network structure
        self.logger.info("Building communication network...")
        network = self.network_builder.build_network(communications)
        
        # Analyze network
        self.logger.info("Analyzing network structure...")
        communities = self.analyzer.detect_communities(network)
        central_figures = self.analyzer.identify_central_figures(network)
        isolated_nodes = self.analyzer.find_isolated_nodes(network)
        suspicious_clusters = self.analyzer.find_suspicious_clusters(network)
        
        # Calculate additional metrics
        metrics = self._calculate_additional_metrics(network, communications)
        
        # Prepare visualization data
        layout_data = self._prepare_layout_data(network)
        visualization_config = self._prepare_visualization_config()
        
        return NetworkGraph(
            network=network,
            layout_data=layout_data,
            visualization_config=visualization_config,
            metrics=metrics,
            communities=communities,
            central_figures=central_figures,
            isolated_nodes=isolated_nodes,
            suspicious_clusters=suspicious_clusters
        )
    
    def _calculate_additional_metrics(
        self, 
        network: CommunicationNetwork, 
        communications: List[Communication]
    ) -> Dict[str, float]:
        """Calculate additional network metrics"""
        
        metrics = {
            'density': network.density,
            'clustering_coefficient': network.clustering_coefficient,
            'total_nodes': len(network.nodes),
            'total_edges': len(network.edges),
            'average_degree': 2 * len(network.edges) / max(1, len(network.nodes))
        }
        
        # Communication frequency metrics
        if communications:
            time_span = (network.date_range[1] - network.date_range[0]).days
            metrics['messages_per_day'] = network.total_messages / max(1, time_span)
            
            # Platform diversity
            metrics['platform_diversity'] = len(network.platforms)
            
            # Activity concentration (Gini coefficient for message distribution)
            message_counts = [node.message_count for node in network.nodes]
            if message_counts:
                metrics['activity_concentration'] = self._gini_coefficient(message_counts)
        
        return metrics
    
    def _gini_coefficient(self, values: List[float]) -> float:
        """Calculate Gini coefficient for inequality measurement"""
        if not values:
            return 0.0
        
        values = sorted(values)
        n = len(values)
        cumsum = np.cumsum(values)
        
        return (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n
    
    def _prepare_layout_data(self, network: CommunicationNetwork) -> Dict[str, Any]:
        """Prepare layout data for visualization"""
        # Create NetworkX graph for layout calculation
        G = nx.Graph()
        
        for node in network.nodes:
            G.add_node(node.node_id)
        
        for edge in network.edges:
            G.add_edge(edge.source_id, edge.target_id, 
                      weight=edge.relationship_strength)
        
        # Calculate different layouts
        layouts = {}
        
        try:
            layouts['spring'] = nx.spring_layout(G, k=1, iterations=50)
            layouts['circular'] = nx.circular_layout(G)
            
            if len(G.nodes) > 3:
                layouts['kamada_kawai'] = nx.kamada_kawai_layout(G)
        
        except Exception as e:
            self.logger.warning(f"Layout calculation failed: {e}")
            # Fallback to simple positions
            layouts['spring'] = {node: (i, 0) for i, node in enumerate(G.nodes)}
        
        return layouts
    
    def _prepare_visualization_config(self) -> Dict[str, Any]:
        """Prepare visualization configuration"""
        return {
            'node_size_range': (10, 50),
            'edge_width_range': (1, 10),
            'color_scheme': 'activity_based',
            'layout_algorithm': 'spring',
            'show_labels': True,
            'interactive': True
        }
    
    def create_visualizations(
        self, 
        network_graph: NetworkGraph, 
        output_dir: Path
    ) -> Dict[str, Path]:
        """Create all network visualizations"""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        created_files = {}
        
        try:
            # Interactive network
            interactive_path = output_dir / "interactive_network.html"
            self.visualizer.create_interactive_network(
                network_graph.network, 
                interactive_path
            )
            created_files['interactive_network'] = interactive_path
            
            # Metrics dashboard
            dashboard_path = output_dir / "network_dashboard.html"
            self.visualizer.create_network_metrics_dashboard(
                network_graph.network,
                dashboard_path
            )
            created_files['dashboard'] = dashboard_path
            
        except Exception as e:
            self.logger.error(f"Visualization creation failed: {e}")
        
        return created_files


# Main function for CLI
def map_communication_network(
    communications: List[Communication], 
    config: CommsConfig = None
) -> NetworkGraph:
    """Map communication network and return analysis results"""
    if config is None:
        config = CommsConfig()
    
    mapper = NetworkMapper(config)
    return mapper.analyze(communications)