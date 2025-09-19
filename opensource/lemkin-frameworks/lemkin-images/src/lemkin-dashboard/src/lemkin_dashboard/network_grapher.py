"""
Lemkin Network Grapher Module

Interactive network graph visualization using NetworkX, Plotly, and Bokeh for
entity relationship mapping in legal investigations. Provides comprehensive
network analysis, centrality measures, and community detection.
"""

import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
from bokeh.plotting import figure, save, output_file
from bokeh.models import HoverTool, GraphRenderer, StaticLayoutProvider, Circle, MultiLine
from bokeh.palettes import Category20, Spectral8
from bokeh.layouts import column, row
from bokeh.io import curdoc
import numpy as np
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple, Set
import logging
from pathlib import Path
import json
import colorsys
from collections import defaultdict

from .core import (
    NetworkGraph, Entity, Relationship, EntityType, RelationshipType,
    DashboardConfig, ExportOptions, ExportFormat, VisualizationSettings
)

# Configure logging
logger = logging.getLogger(__name__)


class NetworkAnalyzer:
    """
    Advanced network analysis system for entity relationship mapping.
    
    Provides comprehensive network analysis including:
    - Centrality measures (degree, betweenness, closeness, eigenvector)
    - Community detection and clustering
    - Path analysis and shortest paths
    - Network topology analysis
    - Influence propagation modeling
    """
    
    def __init__(self, config: Optional[DashboardConfig] = None):
        """Initialize network analyzer with configuration"""
        self.config = config or DashboardConfig()
        self.logger = logging.getLogger(f"{__name__}.NetworkAnalyzer")
        
        # Color schemes for different entity types
        self.entity_colors = {
            EntityType.PERSON: "#3b82f6",
            EntityType.ORGANIZATION: "#10b981", 
            EntityType.LOCATION: "#f59e0b",
            EntityType.EVIDENCE: "#ef4444",
            EntityType.DOCUMENT: "#8b5cf6",
            EntityType.DEVICE: "#06b6d4",
            EntityType.ACCOUNT: "#ec4899",
            EntityType.VEHICLE: "#84cc16",
            EntityType.EVENT: "#f97316",
            EntityType.CONCEPT: "#6b7280"
        }
        
        # Layout algorithms
        self.layout_algorithms = {
            "spring": nx.spring_layout,
            "circular": nx.circular_layout,
            "kamada_kawai": nx.kamada_kawai_layout,
            "shell": nx.shell_layout,
            "spectral": nx.spectral_layout,
            "random": nx.random_layout
        }
        
        self.logger.info("Network Analyzer initialized")
    
    def create_network_graph(self, entities: List[Entity], relationships: List[Relationship],
                           case_id: str, title: Optional[str] = None) -> NetworkGraph:
        """
        Create comprehensive network graph from entities and relationships
        
        Args:
            entities: List of entities
            relationships: List of relationships
            case_id: Case identifier
            title: Graph title
            
        Returns:
            NetworkGraph: Complete network graph object
        """
        if not entities:
            raise ValueError("Cannot create network graph with empty entities list")
        
        # Determine clustering based on network size
        clustering_enabled = len(entities) > 20
        filter_by_relationship = len(relationships) > 10
        
        # Create network graph object
        network_graph = NetworkGraph(
            case_id=case_id,
            title=title or f"Entity Network: {case_id}",
            entities=entities,
            relationships=relationships,
            layout_algorithm="spring",
            node_sizing="importance",
            edge_weighting="strength",
            show_labels=True,
            clustering_enabled=clustering_enabled,
            filter_by_relationship_type=filter_by_relationship,
            highlight_paths=True
        )
        
        self.logger.info(f"Created network graph with {len(entities)} entities and {len(relationships)} relationships")
        return network_graph
    
    def build_networkx_graph(self, network_graph: NetworkGraph) -> nx.Graph:
        """
        Build NetworkX graph from network graph object
        
        Args:
            network_graph: NetworkGraph object
            
        Returns:
            nx.Graph: NetworkX graph object
        """
        G = nx.Graph()
        
        # Add nodes (entities)
        for entity in network_graph.entities:
            G.add_node(
                str(entity.id),
                name=entity.name,
                entity_type=entity.entity_type.value,
                importance=entity.importance,
                description=entity.description or "",
                role=entity.role_in_case or "",
                subject_of_investigation=entity.subject_of_investigation,
                witness=entity.witness,
                primary_location=entity.primary_location or "",
                aliases=entity.aliases
            )
        
        # Add edges (relationships)
        for relationship in network_graph.relationships:
            source_id = str(relationship.source_entity_id)
            target_id = str(relationship.target_entity_id)
            
            # Only add edge if both nodes exist
            if G.has_node(source_id) and G.has_node(target_id):
                G.add_edge(
                    source_id,
                    target_id,
                    relationship_type=relationship.relationship_type.value,
                    strength=relationship.strength,
                    confidence=relationship.confidence,
                    description=relationship.description or "",
                    context=relationship.context or "",
                    relevance=relationship.relevance,
                    verified=relationship.verified,
                    direction=relationship.direction,
                    start_date=relationship.start_date,
                    end_date=relationship.end_date
                )
        
        return G
    
    def analyze_network_metrics(self, G: nx.Graph) -> Dict[str, Any]:
        """
        Analyze comprehensive network metrics
        
        Args:
            G: NetworkX graph
            
        Returns:
            Dict containing network analysis metrics
        """
        if G.number_of_nodes() == 0:
            return {"error": "Empty graph"}
        
        metrics = {
            "basic_metrics": {
                "nodes": G.number_of_nodes(),
                "edges": G.number_of_edges(),
                "density": nx.density(G),
                "is_connected": nx.is_connected(G),
                "number_of_components": nx.number_connected_components(G)
            }
        }
        
        # Only calculate advanced metrics for connected graphs
        if G.number_of_nodes() > 1 and G.number_of_edges() > 0:
            # Centrality measures
            try:
                metrics["centrality"] = {
                    "degree_centrality": nx.degree_centrality(G),
                    "closeness_centrality": nx.closeness_centrality(G),
                    "betweenness_centrality": nx.betweenness_centrality(G),
                    "eigenvector_centrality": nx.eigenvector_centrality(G, max_iter=1000)
                }
            except Exception as e:
                self.logger.warning(f"Could not calculate centrality measures: {e}")
                metrics["centrality"] = {}
            
            # Path analysis
            if nx.is_connected(G):
                metrics["paths"] = {
                    "average_shortest_path_length": nx.average_shortest_path_length(G),
                    "diameter": nx.diameter(G),
                    "radius": nx.radius(G)
                }
            else:
                # For disconnected graphs, analyze largest component
                largest_cc = max(nx.connected_components(G), key=len)
                subgraph = G.subgraph(largest_cc)
                metrics["paths"] = {
                    "average_shortest_path_length": nx.average_shortest_path_length(subgraph),
                    "diameter": nx.diameter(subgraph),
                    "radius": nx.radius(subgraph),
                    "largest_component_size": len(largest_cc)
                }
            
            # Clustering
            metrics["clustering"] = {
                "average_clustering": nx.average_clustering(G),
                "clustering_coefficients": nx.clustering(G)
            }
            
            # Community detection
            try:
                communities = nx.community.greedy_modularity_communities(G)
                metrics["communities"] = {
                    "number_of_communities": len(communities),
                    "modularity": nx.community.modularity(G, communities),
                    "community_sizes": [len(community) for community in communities]
                }
            except Exception as e:
                self.logger.warning(f"Could not perform community detection: {e}")
                metrics["communities"] = {}
        
        return metrics
    
    def generate_plotly_network(self, network_graph: NetworkGraph, 
                              layout_algorithm: str = "spring") -> go.Figure:
        """
        Generate interactive Plotly network visualization
        
        Args:
            network_graph: Network graph object
            layout_algorithm: Layout algorithm to use
            
        Returns:
            go.Figure: Plotly figure object
        """
        G = self.build_networkx_graph(network_graph)
        
        if G.number_of_nodes() == 0:
            return go.Figure().add_annotation(text="No network data available")
        
        # Calculate layout
        if layout_algorithm in self.layout_algorithms:
            pos = self.layout_algorithms[layout_algorithm](G, k=3, iterations=50)
        else:
            pos = nx.spring_layout(G, k=3, iterations=50)
        
        # Prepare node data
        node_x = []
        node_y = []
        node_text = []
        node_colors = []
        node_sizes = []
        node_hover = []
        
        for node_id in G.nodes():
            x, y = pos[node_id]
            node_x.append(x)
            node_y.append(y)
            
            node_data = G.nodes[node_id]
            node_text.append(node_data.get('name', node_id))
            
            # Color by entity type
            entity_type = EntityType(node_data.get('entity_type', 'concept'))
            node_colors.append(self.entity_colors.get(entity_type, "#6b7280"))
            
            # Size by importance
            importance = node_data.get('importance', 'medium')
            size_map = {"low": 20, "medium": 30, "high": 40, "critical": 50}
            node_sizes.append(size_map.get(importance, 30))
            
            # Hover information
            hover_text = f"<b>{node_data.get('name', node_id)}</b><br>"
            hover_text += f"Type: {entity_type.value.title()}<br>"
            hover_text += f"Importance: {importance.title()}<br>"
            if node_data.get('role'):
                hover_text += f"Role: {node_data['role']}<br>"
            if node_data.get('description'):
                hover_text += f"Description: {node_data['description']}<br>"
            node_hover.append(hover_text)
        
        # Prepare edge data
        edge_x = []
        edge_y = []
        edge_info = []
        
        for edge in G.edges(data=True):
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            
            edge_data = edge[2]
            edge_info.append({
                'source': G.nodes[edge[0]].get('name', edge[0]),
                'target': G.nodes[edge[1]].get('name', edge[1]),
                'relationship': edge_data.get('relationship_type', 'unknown'),
                'strength': edge_data.get('strength', 0.5),
                'confidence': edge_data.get('confidence', 1.0)
            })
        
        # Create figure
        fig = go.Figure()
        
        # Add edges
        fig.add_trace(go.Scatter(
            x=edge_x, y=edge_y,
            mode='lines',
            line=dict(width=2, color='rgba(125, 125, 125, 0.5)'),
            hoverinfo='none',
            showlegend=False
        ))
        
        # Add nodes
        fig.add_trace(go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            marker=dict(
                size=node_sizes,
                color=node_colors,
                opacity=0.8,
                line=dict(width=2, color='white')
            ),
            text=node_text,
            textposition="middle center",
            textfont=dict(size=10, color='white'),
            hovertemplate="%{customdata}<extra></extra>",
            customdata=node_hover,
            name="Entities"
        ))
        
        # Update layout
        fig.update_layout(
            title={
                'text': network_graph.title,
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 20}
            },
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20, l=5, r=5, t=40),
            annotations=[
                dict(
                    text="Network analysis of entities and relationships",
                    showarrow=False,
                    xref="paper", yref="paper",
                    x=0.005, y=-0.002,
                    xanchor='left', yanchor='bottom',
                    font=dict(color="gray", size=12)
                )
            ],
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor='white',
            height=700
        )
        
        return fig
    
    def generate_bokeh_network(self, network_graph: NetworkGraph,
                             output_path: Optional[Path] = None) -> Any:
        """
        Generate Bokeh network visualization
        
        Args:
            network_graph: Network graph object
            output_path: Optional path to save HTML output
            
        Returns:
            Bokeh figure object
        """
        G = self.build_networkx_graph(network_graph)
        
        if G.number_of_nodes() == 0:
            return figure(title="No network data available")
        
        # Calculate layout
        pos = nx.spring_layout(G, k=3, iterations=50)
        
        # Prepare node data
        node_data = {
            'index': [],
            'x': [],
            'y': [],
            'name': [],
            'entity_type': [],
            'importance': [],
            'color': [],
            'size': []
        }
        
        node_id_to_index = {}
        for i, node_id in enumerate(G.nodes()):
            node_id_to_index[node_id] = i
            x, y = pos[node_id]
            
            node_attrs = G.nodes[node_id]
            entity_type = EntityType(node_attrs.get('entity_type', 'concept'))
            importance = node_attrs.get('importance', 'medium')
            
            node_data['index'].append(i)
            node_data['x'].append(x)
            node_data['y'].append(y)
            node_data['name'].append(node_attrs.get('name', node_id))
            node_data['entity_type'].append(entity_type.value)
            node_data['importance'].append(importance)
            node_data['color'].append(self.entity_colors.get(entity_type, "#6b7280"))
            
            size_map = {"low": 20, "medium": 30, "high": 40, "critical": 50}
            node_data['size'].append(size_map.get(importance, 30))
        
        # Prepare edge data
        edge_data = {
            'start': [],
            'end': [],
            'relationship_type': [],
            'strength': []
        }
        
        for edge in G.edges(data=True):
            edge_data['start'].append(node_id_to_index[edge[0]])
            edge_data['end'].append(node_id_to_index[edge[1]])
            edge_attrs = edge[2]
            edge_data['relationship_type'].append(edge_attrs.get('relationship_type', 'unknown'))
            edge_data['strength'].append(edge_attrs.get('strength', 0.5))
        
        # Create figure
        plot = figure(
            title=network_graph.title,
            width=1000,
            height=700,
            tools="pan,wheel_zoom,box_zoom,reset,save",
            toolbar_location="above"
        )
        
        # Create graph renderer
        graph = GraphRenderer()
        
        # Node renderer
        graph.node_renderer.data_source.data = node_data
        graph.node_renderer.glyph = Circle(size='size', fill_color='color', line_color='white', line_width=2)
        
        # Edge renderer  
        graph.edge_renderer.data_source.data = edge_data
        graph.edge_renderer.glyph = MultiLine(line_color="gray", line_alpha=0.8, line_width=2)
        
        # Layout provider
        graph_layout = dict(zip(range(len(node_data['index'])), zip(node_data['x'], node_data['y'])))
        graph.layout_provider = StaticLayoutProvider(graph_layout=graph_layout)
        
        # Add hover tool
        hover = HoverTool(
            tooltips=[
                ("Name", "@name"),
                ("Type", "@entity_type"),
                ("Importance", "@importance")
            ]
        )
        plot.add_tools(hover)
        
        # Add graph to plot
        plot.renderers.append(graph)
        
        # Styling
        plot.title.text_font_size = "16pt"
        plot.title.align = "center"
        
        # Save if output path provided
        if output_path:
            output_file(str(output_path))
            save(plot)
            self.logger.info(f"Bokeh network saved to: {output_path}")
        
        return plot
    
    def find_shortest_paths(self, G: nx.Graph, source_entity: str, 
                          target_entity: str) -> Dict[str, Any]:
        """
        Find shortest paths between entities
        
        Args:
            G: NetworkX graph
            source_entity: Source entity name
            target_entity: Target entity name
            
        Returns:
            Dict containing path analysis
        """
        # Find node IDs by entity names
        source_node = None
        target_node = None
        
        for node_id, node_data in G.nodes(data=True):
            if node_data.get('name') == source_entity:
                source_node = node_id
            elif node_data.get('name') == target_entity:
                target_node = node_id
        
        if not source_node or not target_node:
            return {"error": "Entity not found in network"}
        
        if not nx.has_path(G, source_node, target_node):
            return {"error": "No path exists between entities"}
        
        try:
            # Find shortest path
            shortest_path = nx.shortest_path(G, source_node, target_node)
            path_length = nx.shortest_path_length(G, source_node, target_node)
            
            # Get all simple paths (limited to reasonable number)
            all_paths = list(nx.all_simple_paths(G, source_node, target_node, cutoff=5))
            all_paths = all_paths[:10]  # Limit to 10 paths
            
            # Build path details
            path_details = []
            for path in all_paths:
                path_info = {
                    "length": len(path) - 1,
                    "entities": [G.nodes[node]['name'] for node in path],
                    "relationships": []
                }
                
                for i in range(len(path) - 1):
                    edge_data = G.edges[path[i], path[i+1]]
                    path_info["relationships"].append({
                        "type": edge_data.get('relationship_type', 'unknown'),
                        "strength": edge_data.get('strength', 0.5),
                        "confidence": edge_data.get('confidence', 1.0)
                    })
                
                path_details.append(path_info)
            
            return {
                "source_entity": source_entity,
                "target_entity": target_entity,
                "shortest_path_length": path_length,
                "shortest_path_entities": [G.nodes[node]['name'] for node in shortest_path],
                "total_paths_found": len(all_paths),
                "path_details": path_details
            }
            
        except Exception as e:
            return {"error": f"Path analysis failed: {str(e)}"}
    
    def detect_communities(self, G: nx.Graph) -> Dict[str, Any]:
        """
        Detect communities in the network using multiple algorithms
        
        Args:
            G: NetworkX graph
            
        Returns:
            Dict containing community analysis
        """
        if G.number_of_nodes() < 3:
            return {"error": "Network too small for community detection"}
        
        communities_analysis = {}
        
        try:
            # Greedy modularity communities
            greedy_communities = list(nx.community.greedy_modularity_communities(G))
            greedy_modularity = nx.community.modularity(G, greedy_communities)
            
            communities_analysis["greedy_modularity"] = {
                "number_of_communities": len(greedy_communities),
                "modularity": greedy_modularity,
                "communities": []
            }
            
            for i, community in enumerate(greedy_communities):
                community_info = {
                    "id": i,
                    "size": len(community),
                    "entities": [G.nodes[node]['name'] for node in community],
                    "entity_types": [G.nodes[node]['entity_type'] for node in community]
                }
                communities_analysis["greedy_modularity"]["communities"].append(community_info)
                
        except Exception as e:
            self.logger.warning(f"Greedy modularity community detection failed: {e}")
        
        try:
            # Label propagation communities
            label_communities = list(nx.community.label_propagation_communities(G))
            label_modularity = nx.community.modularity(G, label_communities)
            
            communities_analysis["label_propagation"] = {
                "number_of_communities": len(label_communities),
                "modularity": label_modularity,
                "communities": []
            }
            
            for i, community in enumerate(label_communities):
                community_info = {
                    "id": i,
                    "size": len(community),
                    "entities": [G.nodes[node]['name'] for node in community]
                }
                communities_analysis["label_propagation"]["communities"].append(community_info)
                
        except Exception as e:
            self.logger.warning(f"Label propagation community detection failed: {e}")
        
        return communities_analysis
    
    def analyze_key_players(self, G: nx.Graph, top_n: int = 10) -> Dict[str, Any]:
        """
        Identify key players in the network using various centrality measures
        
        Args:
            G: NetworkX graph
            top_n: Number of top entities to return
            
        Returns:
            Dict containing key player analysis
        """
        if G.number_of_nodes() == 0:
            return {"error": "Empty network"}
        
        key_players = {}
        
        try:
            # Degree centrality - most connected entities
            degree_centrality = nx.degree_centrality(G)
            top_degree = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:top_n]
            key_players["most_connected"] = [
                {
                    "entity": G.nodes[node_id]['name'],
                    "centrality_score": score,
                    "connections": G.degree[node_id]
                }
                for node_id, score in top_degree
            ]
            
            # Betweenness centrality - entities that bridge different parts of network
            betweenness_centrality = nx.betweenness_centrality(G)
            top_betweenness = sorted(betweenness_centrality.items(), key=lambda x: x[1], reverse=True)[:top_n]
            key_players["bridge_entities"] = [
                {
                    "entity": G.nodes[node_id]['name'],
                    "centrality_score": score,
                    "bridge_importance": "High" if score > 0.1 else "Medium" if score > 0.05 else "Low"
                }
                for node_id, score in top_betweenness
            ]
            
            # Closeness centrality - entities with shortest paths to others
            closeness_centrality = nx.closeness_centrality(G)
            top_closeness = sorted(closeness_centrality.items(), key=lambda x: x[1], reverse=True)[:top_n]
            key_players["central_entities"] = [
                {
                    "entity": G.nodes[node_id]['name'],
                    "centrality_score": score,
                    "accessibility": "High" if score > 0.5 else "Medium" if score > 0.3 else "Low"
                }
                for node_id, score in top_closeness
            ]
            
        except Exception as e:
            self.logger.warning(f"Key player analysis failed: {e}")
            key_players["error"] = str(e)
        
        return key_players
    
    def export_network_analysis(self, network_graph: NetworkGraph, 
                               output_dir: Path) -> Dict[str, Path]:
        """
        Export comprehensive network analysis report
        
        Args:
            network_graph: Network graph object
            output_dir: Directory to save analysis files
            
        Returns:
            Dict mapping analysis types to file paths
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        
        analysis_paths = {}
        G = self.build_networkx_graph(network_graph)
        
        try:
            # Generate main network visualization
            fig = self.generate_plotly_network(network_graph)
            html_path = output_dir / f"network_{network_graph.case_id}.html"
            fig.write_html(str(html_path))
            analysis_paths['network_html'] = html_path
            
            # Generate network metrics
            metrics = self.analyze_network_metrics(G)
            metrics_path = output_dir / f"network_metrics_{network_graph.case_id}.json"
            with open(metrics_path, 'w') as f:
                json.dump(metrics, f, indent=2, default=str)
            analysis_paths['metrics'] = metrics_path
            
            # Generate community analysis
            communities = self.detect_communities(G)
            communities_path = output_dir / f"network_communities_{network_graph.case_id}.json"
            with open(communities_path, 'w') as f:
                json.dump(communities, f, indent=2, default=str)
            analysis_paths['communities'] = communities_path
            
            # Generate key players analysis
            key_players = self.analyze_key_players(G)
            key_players_path = output_dir / f"network_key_players_{network_graph.case_id}.json"
            with open(key_players_path, 'w') as f:
                json.dump(key_players, f, indent=2, default=str)
            analysis_paths['key_players'] = key_players_path
            
            self.logger.info(f"Network analysis exported to: {output_dir}")
            
        except Exception as e:
            self.logger.error(f"Failed to export network analysis: {str(e)}")
        
        return analysis_paths


def visualize_entity_network(entities: List[Entity], relationships: List[Relationship],
                           case_id: str, title: Optional[str] = None,
                           config: Optional[DashboardConfig] = None) -> NetworkGraph:
    """
    Convenience function to create entity network visualization
    
    Args:
        entities: List of entities
        relationships: List of relationships  
        case_id: Case identifier
        title: Optional network title
        config: Optional dashboard configuration
        
    Returns:
        NetworkGraph: Complete network graph
    """
    analyzer = NetworkAnalyzer(config)
    return analyzer.create_network_graph(entities, relationships, case_id, title)


def generate_network_report(entities: List[Entity], relationships: List[Relationship],
                          case_id: str, output_dir: Path) -> Dict[str, Path]:
    """
    Generate comprehensive network analysis report
    
    Args:
        entities: List of entities
        relationships: List of relationships
        case_id: Case identifier
        output_dir: Directory to save report files
        
    Returns:
        Dict mapping report types to file paths
    """
    analyzer = NetworkAnalyzer()
    network_graph = analyzer.create_network_graph(entities, relationships, case_id)
    return analyzer.export_network_analysis(network_graph, output_dir)