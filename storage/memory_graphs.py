#!/usr/bin/env python3
"""
🧠 Memory Graph Visualization for MeRNSTA

Visualize semantic triplets as a graph with volatility edges and contradiction arcs,
using vector similarity to weight edges. Creates interactive knowledge graphs that
show belief structures, contradictions, and memory patterns.
"""

import logging
import time
import json
from typing import List, Dict, Set, Tuple, Optional, Any
from collections import defaultdict
from dataclasses import dataclass, asdict
import math

try:
    import networkx as nx
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib.colors import LinearSegmentedColormap
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False
    logging.warning("NetworkX/Matplotlib not available for memory graph visualization")

try:
    import numpy as np
    from scipy.spatial.distance import cosine
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

from .enhanced_triplet_extractor import EnhancedTripletFact


@dataclass
class GraphNode:
    """Represents a node in the memory graph."""
    node_id: str
    label: str
    node_type: str  # "subject", "predicate", "object", "concept"
    semantic_embedding: Optional[List[float]] = None
    volatility_score: float = 0.0
    consolidation_level: str = "emerging"  # "emerging", "stable", "consolidated"
    fact_count: int = 0
    last_updated: float = 0.0


@dataclass
class GraphEdge:
    """Represents an edge in the memory graph."""
    source: str
    target: str
    edge_type: str  # "semantic", "contradiction", "volatility", "reinforcement"
    weight: float
    confidence: float
    metadata: Dict[str, Any]
    created_time: float


class MemoryGraphSystem:
    """
    Creates and manages visual graphs of memory structures.
    Shows semantic relationships, contradictions, and belief patterns.
    """
    
    def __init__(self):
        self.graph = nx.MultiDiGraph() if VISUALIZATION_AVAILABLE else None
        self.nodes: Dict[str, GraphNode] = {}
        self.edges: List[GraphEdge] = []
        self.similarity_threshold = 0.4
        self.volatility_threshold = 0.6
        self.consolidation_threshold = 0.8
        
    def build_memory_graph(self, facts: List[EnhancedTripletFact], 
                          contradiction_clusters=None, 
                          belief_patterns=None) -> Optional[nx.MultiDiGraph]:
        """
        Build a comprehensive memory graph from facts and analysis results.
        """
        if not VISUALIZATION_AVAILABLE:
            print("[MemoryGraph] Visualization libraries not available")
            return None
            
        print(f"[MemoryGraph] Building graph from {len(facts)} facts")
        
        # Clear existing graph
        self.graph.clear()
        self.nodes.clear()
        self.edges.clear()
        
        # Add nodes for all entities
        self._add_entity_nodes(facts)
        
        # Add semantic relationship edges
        self._add_semantic_edges(facts)
        
        # Add contradiction edges if available
        if contradiction_clusters:
            self._add_contradiction_edges(contradiction_clusters)
        
        # Add belief consolidation information if available
        if belief_patterns:
            self._add_consolidation_edges(belief_patterns)
        
        # Add volatility indicators
        self._add_volatility_indicators(facts)
        
        # Calculate graph metrics
        self._calculate_graph_metrics()
        
        print(f"[MemoryGraph] Graph built with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges")
        
        return self.graph
    
    def _add_entity_nodes(self, facts: List[EnhancedTripletFact]):
        """Add nodes for all subjects, predicates, and objects."""
        entity_counts = defaultdict(int)
        entity_types = {}
        entity_volatilities = defaultdict(float)
        entity_last_seen = defaultdict(float)
        
        # Collect entity statistics
        for fact in facts:
            # Subjects
            subj_id = f"subj_{fact.subject.lower().replace(' ', '_')}"
            entity_counts[subj_id] += 1
            entity_types[subj_id] = "subject"
            entity_volatilities[subj_id] += getattr(fact, 'volatility_score', 0.0)
            entity_last_seen[subj_id] = max(entity_last_seen[subj_id], 
                                          getattr(fact, 'timestamp', time.time()))
            
            # Predicates
            pred_id = f"pred_{fact.predicate.lower().replace(' ', '_')}"
            entity_counts[pred_id] += 1
            entity_types[pred_id] = "predicate"
            entity_volatilities[pred_id] += getattr(fact, 'volatility_score', 0.0)
            entity_last_seen[pred_id] = max(entity_last_seen[pred_id],
                                          getattr(fact, 'timestamp', time.time()))
            
            # Objects
            obj_id = f"obj_{fact.object.lower().replace(' ', '_')}"
            entity_counts[obj_id] += 1
            entity_types[obj_id] = "object"
            entity_volatilities[obj_id] += getattr(fact, 'volatility_score', 0.0)
            entity_last_seen[obj_id] = max(entity_last_seen[obj_id],
                                         getattr(fact, 'timestamp', time.time()))
        
        # Create nodes
        for entity_id, count in entity_counts.items():
            avg_volatility = entity_volatilities[entity_id] / count
            
            # Determine consolidation level based on count and volatility
            if count >= 3 and avg_volatility < 0.3:
                consolidation_level = "consolidated"
            elif count >= 2 and avg_volatility < 0.6:
                consolidation_level = "stable"
            else:
                consolidation_level = "emerging"
            
            # Create node
            label = entity_id.split('_', 1)[1].replace('_', ' ').title()
            
            node = GraphNode(
                node_id=entity_id,
                label=label,
                node_type=entity_types[entity_id],
                volatility_score=avg_volatility,
                consolidation_level=consolidation_level,
                fact_count=count,
                last_updated=entity_last_seen[entity_id]
            )
            
            self.nodes[entity_id] = node
            
            # Add to NetworkX graph
            self.graph.add_node(entity_id, **asdict(node))
    
    def _add_semantic_edges(self, facts: List[EnhancedTripletFact]):
        """Add edges representing semantic relationships between facts."""
        for fact in facts:
            subj_id = f"subj_{fact.subject.lower().replace(' ', '_')}"
            pred_id = f"pred_{fact.predicate.lower().replace(' ', '_')}"
            obj_id = f"obj_{fact.object.lower().replace(' ', '_')}"
            
            # Subject -> Predicate edge
            edge1 = GraphEdge(
                source=subj_id,
                target=pred_id,
                edge_type="semantic",
                weight=getattr(fact, 'confidence', 0.5),
                confidence=getattr(fact, 'confidence', 0.5),
                metadata={
                    'fact_id': fact.id,
                    'relationship': 'subject_predicate',
                    'original_text': f"{fact.subject} {fact.predicate}"
                },
                created_time=getattr(fact, 'timestamp', time.time())
            )
            
            # Predicate -> Object edge
            edge2 = GraphEdge(
                source=pred_id,
                target=obj_id,
                edge_type="semantic",
                weight=getattr(fact, 'confidence', 0.5),
                confidence=getattr(fact, 'confidence', 0.5),
                metadata={
                    'fact_id': fact.id,
                    'relationship': 'predicate_object',
                    'original_text': f"{fact.predicate} {fact.object}"
                },
                created_time=getattr(fact, 'timestamp', time.time())
            )
            
            self.edges.extend([edge1, edge2])
            
            # Add to NetworkX graph
            self.graph.add_edge(subj_id, pred_id, 
                              edge_type="semantic",
                              weight=edge1.weight,
                              confidence=edge1.confidence,
                              **edge1.metadata)
            
            self.graph.add_edge(pred_id, obj_id,
                              edge_type="semantic", 
                              weight=edge2.weight,
                              confidence=edge2.confidence,
                              **edge2.metadata)
    
    def _add_contradiction_edges(self, contradiction_clusters):
        """Add edges showing contradiction relationships."""
        for cluster_id, cluster in contradiction_clusters.items():
            # Add contradiction edges between facts in the same cluster
            facts = cluster.contradictory_facts
            
            for i, fact1 in enumerate(facts):
                for fact2 in facts[i+1:]:
                    # Create contradiction edge between the objects of contradictory facts
                    obj1_id = f"obj_{fact1.object.lower().replace(' ', '_')}"
                    obj2_id = f"obj_{fact2.object.lower().replace(' ', '_')}"
                    
                    if obj1_id != obj2_id:  # Don't create self-loops
                        contradiction_edge = GraphEdge(
                            source=obj1_id,
                            target=obj2_id,
                            edge_type="contradiction",
                            weight=cluster.volatility_score,
                            confidence=0.8,
                            metadata={
                                'cluster_id': cluster_id,
                                'concept_theme': cluster.concept_theme,
                                'volatility_score': cluster.volatility_score
                            },
                            created_time=time.time()
                        )
                        
                        self.edges.append(contradiction_edge)
                        
                        # Add to NetworkX graph with special styling
                        self.graph.add_edge(obj1_id, obj2_id,
                                          edge_type="contradiction",
                                          weight=contradiction_edge.weight,
                                          confidence=contradiction_edge.confidence,
                                          color='red',
                                          style='dashed',
                                          **contradiction_edge.metadata)
    
    def _add_consolidation_edges(self, belief_patterns):
        """Add edges showing belief consolidation strength."""
        for pattern_id, belief in belief_patterns.items():
            if belief.consolidation_level in ["stable", "consolidated"]:
                # Add reinforcement edges between elements of consolidated beliefs
                subj_id = f"subj_{belief.subject.lower().replace(' ', '_')}"
                pred_id = f"pred_{belief.predicate.lower().replace(' ', '_')}"
                obj_id = f"obj_{belief.object.lower().replace(' ', '_')}"
                
                # Strong consolidation edge
                consolidation_edge = GraphEdge(
                    source=subj_id,
                    target=obj_id,
                    edge_type="consolidation",
                    weight=belief.stability_score,
                    confidence=belief.stability_score,
                    metadata={
                        'pattern_id': pattern_id,
                        'consolidation_level': belief.consolidation_level,
                        'support_count': len(belief.support_facts),
                        'reinforcements': len(belief.reinforcement_events)
                    },
                    created_time=time.time()
                )
                
                self.edges.append(consolidation_edge)
                
                # Add to NetworkX graph with special styling
                color = 'green' if belief.consolidation_level == "consolidated" else 'orange'
                width = 3.0 if belief.consolidation_level == "consolidated" else 2.0
                
                self.graph.add_edge(subj_id, obj_id,
                                  edge_type="consolidation",
                                  weight=consolidation_edge.weight,
                                  confidence=consolidation_edge.confidence,
                                  color=color,
                                  width=width,
                                  **consolidation_edge.metadata)
    
    def _add_volatility_indicators(self, facts: List[EnhancedTripletFact]):
        """Add visual indicators for volatile concepts."""
        for fact in facts:
            volatility = getattr(fact, 'volatility_score', 0.0)
            
            if volatility > self.volatility_threshold:
                # Add volatility markers to relevant nodes
                for entity_type, entity_value in [('subj', fact.subject), 
                                                 ('pred', fact.predicate), 
                                                 ('obj', fact.object)]:
                    entity_id = f"{entity_type}_{entity_value.lower().replace(' ', '_')}"
                    
                    if entity_id in self.graph.nodes:
                        # Update node with volatility information
                        self.graph.nodes[entity_id]['volatility_marker'] = True
                        self.graph.nodes[entity_id]['volatility_score'] = max(
                            self.graph.nodes[entity_id].get('volatility_score', 0.0),
                            volatility
                        )
    
    def _calculate_graph_metrics(self):
        """Calculate various graph metrics for analysis."""
        if not self.graph or self.graph.number_of_nodes() == 0:
            return
        
        # Basic metrics
        metrics = {
            'total_nodes': self.graph.number_of_nodes(),
            'total_edges': self.graph.number_of_edges(),
            'density': nx.density(self.graph),
            'avg_degree': sum(dict(self.graph.degree()).values()) / self.graph.number_of_nodes()
        }
        
        # Add metrics as graph attributes
        self.graph.graph['metrics'] = metrics
        
        print(f"[MemoryGraph] Graph metrics: {metrics}")
    
    def visualize_graph(self, output_file: str = None, show_labels: bool = True, 
                       highlight_contradictions: bool = True) -> bool:
        """
        Create a visual representation of the memory graph.
        """
        if not VISUALIZATION_AVAILABLE or not self.graph:
            print("[MemoryGraph] Cannot visualize: libraries not available or no graph")
            return False
        
        plt.figure(figsize=(16, 12))
        
        # Use spring layout for better visualization
        pos = nx.spring_layout(self.graph, k=2, iterations=50)
        
        # Separate nodes by type for different styling
        subject_nodes = [n for n, d in self.graph.nodes(data=True) if d.get('node_type') == 'subject']
        predicate_nodes = [n for n, d in self.graph.nodes(data=True) if d.get('node_type') == 'predicate']
        object_nodes = [n for n, d in self.graph.nodes(data=True) if d.get('node_type') == 'object']
        
        # Draw nodes with different colors and sizes based on properties
        for nodes, color, shape in [(subject_nodes, 'lightblue', 'o'),
                                   (predicate_nodes, 'lightgreen', 's'),
                                   (object_nodes, 'lightcoral', '^')]:
            
            if nodes:
                # Size based on fact count
                sizes = [self.graph.nodes[n].get('fact_count', 1) * 300 for n in nodes]
                
                # Color intensity based on volatility
                volatilities = [self.graph.nodes[n].get('volatility_score', 0.0) for n in nodes]
                
                nx.draw_networkx_nodes(self.graph, pos, nodelist=nodes,
                                     node_color=volatilities, 
                                     node_size=sizes,
                                     node_shape=shape,
                                     cmap=plt.cm.Reds,
                                     alpha=0.8)
        
        # Draw different types of edges with different styles
        semantic_edges = [(u, v) for u, v, d in self.graph.edges(data=True) 
                         if d.get('edge_type') == 'semantic']
        contradiction_edges = [(u, v) for u, v, d in self.graph.edges(data=True) 
                              if d.get('edge_type') == 'contradiction']
        consolidation_edges = [(u, v) for u, v, d in self.graph.edges(data=True) 
                              if d.get('edge_type') == 'consolidation']
        
        # Draw semantic edges (thin, gray)
        if semantic_edges:
            nx.draw_networkx_edges(self.graph, pos, edgelist=semantic_edges,
                                 edge_color='gray', width=1, alpha=0.6)
        
        # Draw contradiction edges (red, dashed)
        if contradiction_edges and highlight_contradictions:
            nx.draw_networkx_edges(self.graph, pos, edgelist=contradiction_edges,
                                 edge_color='red', width=2, style='dashed', alpha=0.8)
        
        # Draw consolidation edges (green, thick)
        if consolidation_edges:
            nx.draw_networkx_edges(self.graph, pos, edgelist=consolidation_edges,
                                 edge_color='green', width=3, alpha=0.9)
        
        # Add labels if requested
        if show_labels:
            labels = {n: d.get('label', n) for n, d in self.graph.nodes(data=True)}
            nx.draw_networkx_labels(self.graph, pos, labels, font_size=8)
        
        # Add title and legend
        plt.title("MeRNSTA Memory Graph\nBlue=Subjects, Green=Predicates, Red=Objects\nNode size=Fact count, Red intensity=Volatility", 
                 fontsize=14, pad=20)
        
        # Create legend
        legend_elements = [
            plt.Line2D([0], [0], color='gray', lw=2, label='Semantic Relations'),
            plt.Line2D([0], [0], color='red', lw=2, linestyle='--', label='Contradictions'),
            plt.Line2D([0], [0], color='green', lw=3, label='Consolidated Beliefs'),
            plt.scatter([], [], c='lightblue', s=100, marker='o', label='Subjects'),
            plt.scatter([], [], c='lightgreen', s=100, marker='s', label='Predicates'),
            plt.scatter([], [], c='lightcoral', s=100, marker='^', label='Objects')
        ]
        plt.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1, 1))
        
        plt.tight_layout()
        
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"[MemoryGraph] Graph saved to {output_file}")
        else:
            plt.show()
        
        return True
    
    def export_graph_data(self, output_file: str) -> bool:
        """Export graph data in JSON format for external visualization tools."""
        if not self.graph:
            return False
        
        # Convert to JSON-serializable format
        graph_data = {
            'nodes': [
                {
                    'id': node_id,
                    'label': data.get('label', node_id),
                    'type': data.get('node_type', 'unknown'),
                    'volatility_score': data.get('volatility_score', 0.0),
                    'consolidation_level': data.get('consolidation_level', 'emerging'),
                    'fact_count': data.get('fact_count', 0)
                }
                for node_id, data in self.graph.nodes(data=True)
            ],
            'edges': [
                {
                    'source': source,
                    'target': target,
                    'type': data.get('edge_type', 'semantic'),
                    'weight': data.get('weight', 1.0),
                    'confidence': data.get('confidence', 0.5),
                    'metadata': {k: v for k, v in data.items() 
                               if k not in ['edge_type', 'weight', 'confidence']}
                }
                for source, target, data in self.graph.edges(data=True)
            ],
            'metrics': self.graph.graph.get('metrics', {}),
            'generated_at': time.time()
        }
        
        try:
            with open(output_file, 'w') as f:
                json.dump(graph_data, f, indent=2, default=str)
            print(f"[MemoryGraph] Graph data exported to {output_file}")
            return True
        except Exception as e:
            print(f"[MemoryGraph] Failed to export graph data: {e}")
            return False
    
    def get_graph_summary(self) -> Dict:
        """Get a summary of the memory graph structure."""
        if not self.graph:
            return {'error': 'No graph available'}
        
        # Node type distribution
        node_types = defaultdict(int)
        consolidation_levels = defaultdict(int)
        total_volatility = 0.0
        
        for node_id, data in self.graph.nodes(data=True):
            node_types[data.get('node_type', 'unknown')] += 1
            consolidation_levels[data.get('consolidation_level', 'emerging')] += 1
            total_volatility += data.get('volatility_score', 0.0)
        
        # Edge type distribution
        edge_types = defaultdict(int)
        for source, target, data in self.graph.edges(data=True):
            edge_types[data.get('edge_type', 'semantic')] += 1
        
        return {
            'total_nodes': self.graph.number_of_nodes(),
            'total_edges': self.graph.number_of_edges(),
            'node_types': dict(node_types),
            'edge_types': dict(edge_types),
            'consolidation_levels': dict(consolidation_levels),
            'average_volatility': total_volatility / max(1, self.graph.number_of_nodes()),
            'graph_density': nx.density(self.graph) if self.graph.number_of_nodes() > 0 else 0.0,
            'connected_components': nx.number_weakly_connected_components(self.graph)
        } 