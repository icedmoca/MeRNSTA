#!/usr/bin/env python3
"""
📊 Causal Graph Visualizer for MeRNSTA

Creates visual representations of causal relationships using:
- NetworkX for graph structure
- Matplotlib for basic visualization
- Text-based representations for CLI
- DOT format for advanced visualization
"""

import logging
from typing import List, Dict, Optional, Tuple, Any
from collections import defaultdict
import textwrap

from .enhanced_memory_model import EnhancedTripletFact
from .memory_log import MemoryLog


class CausalGraphVisualizer:
    """
    Creates visual representations of causal graphs from memory facts.
    """
    
    def __init__(self, memory_log: MemoryLog = None):
        self.memory_log = memory_log or MemoryLog()
        
        # Try to import optional visualization libraries
        self.networkx_available = False
        self.matplotlib_available = False
        
        try:
            import networkx as nx
            self.nx = nx
            self.networkx_available = True
        except ImportError:
            print("[CausalViz] NetworkX not available - using text-based visualization")
        
        try:
            import matplotlib.pyplot as plt
            self.plt = plt
            self.matplotlib_available = True
        except ImportError:
            print("[CausalViz] Matplotlib not available - using text-based visualization")
    
    def generate_text_graph(self, user_profile_id: str = None, 
                          min_strength: float = 0.3) -> List[str]:
        """Generate a text-based representation of the causal graph."""
        facts = self._get_causal_facts(user_profile_id, min_strength)
        
        if not facts:
            return ["No causal relationships found with minimum strength {:.2f}".format(min_strength)]
        
        # Group facts by strength
        strong_facts = [f for f in facts if f.causal_strength > 0.7]
        medium_facts = [f for f in facts if 0.4 <= f.causal_strength <= 0.7]
        weak_facts = [f for f in facts if f.causal_strength < 0.4]
        
        lines = [
            "🔗 CAUSAL RELATIONSHIP GRAPH",
            "=" * 40,
            f"Total Links: {len(facts)}",
            f"Strong (>0.7): {len(strong_facts)}",
            f"Medium (0.4-0.7): {len(medium_facts)}",
            f"Weak (<0.4): {len(weak_facts)}",
            ""
        ]
        
        # Display strong relationships first
        if strong_facts:
            lines.extend([
                "⚡ STRONG CAUSAL LINKS",
                "-" * 25
            ])
            for fact in strong_facts:
                cause_text = fact.cause if hasattr(fact, 'cause') and fact.cause else "unknown"
                effect_text = f"{fact.subject} {fact.predicate} {fact.object}"
                strength_bar = "█" * int(fact.causal_strength * 10)
                lines.append(f"{cause_text} → {effect_text}")
                lines.append(f"   Strength: {strength_bar} ({fact.causal_strength:.3f})")
                lines.append("")
        
        # Display medium relationships
        if medium_facts:
            lines.extend([
                "🔸 MEDIUM CAUSAL LINKS",
                "-" * 25
            ])
            for fact in medium_facts[:5]:  # Limit to 5 for readability
                cause_text = fact.cause if hasattr(fact, 'cause') and fact.cause else "unknown"
                effect_text = f"{fact.subject} {fact.predicate} {fact.object}"
                strength_bar = "▓" * int(fact.causal_strength * 10)
                lines.append(f"{cause_text} → {effect_text}")
                lines.append(f"   Strength: {strength_bar} ({fact.causal_strength:.3f})")
                lines.append("")
        
        # Display weak relationships (limited)
        if weak_facts:
            lines.extend([
                f"⚪ WEAK CAUSAL LINKS ({len(weak_facts)} total, showing first 3)",
                "-" * 30
            ])
            for fact in weak_facts[:3]:
                cause_text = fact.cause if hasattr(fact, 'cause') and fact.cause else "unknown"
                effect_text = f"{fact.subject} {fact.predicate} {fact.object}"
                strength_bar = "░" * int(fact.causal_strength * 10)
                lines.append(f"{cause_text} → {effect_text}")
                lines.append(f"   Strength: {strength_bar} ({fact.causal_strength:.3f})")
                lines.append("")
        
        return lines

    def generate_network_summary(self, user_profile_id: str = None,
                                min_strength: float = 0.3) -> Dict[str, Any]:
        """Generate a network analysis summary."""
        facts = self._get_causal_facts(user_profile_id, min_strength)
        
        if not facts:
            return {"status": "no_data", "total_facts": 0}
        
        # Build network statistics
        nodes = set()
        edges = []
        node_degrees = defaultdict(int)
        
        for fact in facts:
            cause = fact.cause if hasattr(fact, 'cause') and fact.cause else f"unknown_cause_{fact.id}"
            effect = f"{fact.subject}_{fact.predicate}"
            
            nodes.add(cause)
            nodes.add(effect)
            edges.append((cause, effect, fact.causal_strength))
            
            node_degrees[cause] += 1  # Out-degree
            node_degrees[effect] += 1  # In-degree
        
        # Find most connected nodes
        top_nodes = sorted(node_degrees.items(), key=lambda x: x[1], reverse=True)[:5]
        
        # Calculate strength statistics
        strengths = [fact.causal_strength for fact in facts]
        avg_strength = sum(strengths) / len(strengths)
        
        return {
            "status": "success",
            "total_facts": len(facts),
            "total_nodes": len(nodes),
            "total_edges": len(edges),
            "average_strength": avg_strength,
            "strongest_link": max(strengths),
            "weakest_link": min(strengths),
            "most_connected_nodes": top_nodes,
            "network_density": len(edges) / (len(nodes) * (len(nodes) - 1)) if len(nodes) > 1 else 0
        }

    def generate_dot_graph(self, user_profile_id: str = None, 
                          min_strength: float = 0.3) -> str:
        """Generate a DOT format graph for external visualization."""
        facts = self._get_causal_facts(user_profile_id, min_strength)
        
        if not facts:
            return "digraph CausalGraph {\n  label=\"No causal relationships found\";\n}"
        
        dot_lines = [
            "digraph CausalGraph {",
            "  rankdir=LR;",
            "  node [shape=ellipse, style=filled];",
            "  edge [fontsize=10];",
            ""
        ]
        
        # Add nodes with different colors based on type
        nodes = set()
        for fact in facts:
            cause = fact.cause if hasattr(fact, 'cause') and fact.cause else f"unknown_{fact.id}"
            effect = f"{fact.subject}_{fact.predicate}"
            nodes.add(cause)
            nodes.add(effect)
        
        # Color nodes by type
        for node in nodes:
            if "user_feel" in node:
                color = "lightcoral"
            elif "user_work" in node or "user_study" in node:
                color = "lightblue"
            elif "user_be" in node:
                color = "lightgreen"
            else:
                color = "lightyellow"
            
            clean_node = node.replace("_", " ").title()
            dot_lines.append(f'  "{node}" [label="{clean_node}", fillcolor={color}];')
        
        dot_lines.append("")
        
        # Add edges with weights
        for fact in facts:
            cause = fact.cause if hasattr(fact, 'cause') and fact.cause else f"unknown_{fact.id}"
            effect = f"{fact.subject}_{fact.predicate}"
            
            # Edge thickness based on strength
            weight = max(1, int(fact.causal_strength * 5))
            color = "red" if fact.causal_strength > 0.7 else "orange" if fact.causal_strength > 0.4 else "gray"
            
            dot_lines.append(f'  "{cause}" -> "{effect}" [label="{fact.causal_strength:.2f}", penwidth={weight}, color={color}];')
        
        dot_lines.extend([
            "",
            "}",
            "",
            "// To visualize this graph:",
            "// 1. Save to file.dot",
            "// 2. Run: dot -Tpng file.dot -o graph.png",
            "// 3. Or use online viewers like http://magjac.com/graphviz-visual-editor/"
        ])
        
        return "\n".join(dot_lines)

    def generate_matplotlib_graph(self, user_profile_id: str = None, 
                                min_strength: float = 0.3, 
                                save_path: str = None) -> bool:
        """Generate a matplotlib visualization (if available)."""
        if not self.networkx_available or not self.matplotlib_available:
            print("[CausalViz] NetworkX or Matplotlib not available for graph generation")
            return False
        
        facts = self._get_causal_facts(user_profile_id, min_strength)
        
        if not facts:
            print("[CausalViz] No causal facts to visualize")
            return False
        
        # Create NetworkX graph
        G = self.nx.DiGraph()
        
        for fact in facts:
            cause = fact.cause if hasattr(fact, 'cause') and fact.cause else f"unknown_{fact.id}"
            effect = f"{fact.subject} {fact.predicate}"
            G.add_edge(cause, effect, weight=fact.causal_strength)
        
        # Create matplotlib plot
        self.plt.figure(figsize=(12, 8))
        pos = self.nx.spring_layout(G, k=2, iterations=50)
        
        # Draw nodes
        node_colors = []
        for node in G.nodes():
            if "feel" in node:
                node_colors.append("lightcoral")
            elif "work" in node or "study" in node:
                node_colors.append("lightblue")
            elif "be" in node:
                node_colors.append("lightgreen")
            else:
                node_colors.append("lightyellow")
        
        self.nx.draw_networkx_nodes(G, pos, node_color=node_colors, 
                                   node_size=1000, alpha=0.8)
        
        # Draw edges with varying thickness
        edges = G.edges(data=True)
        for (u, v, d) in edges:
            weight = d['weight']
            color = 'red' if weight > 0.7 else 'orange' if weight > 0.4 else 'gray'
            width = weight * 3
            self.nx.draw_networkx_edges(G, pos, [(u, v)], edge_color=color, 
                                       width=width, alpha=0.7, arrows=True)
        
        # Draw labels
        labels = {node: node.replace("_", "\n") for node in G.nodes()}
        self.nx.draw_networkx_labels(G, pos, labels, font_size=8)
        
        # Add edge labels
        edge_labels = {(u, v): f"{d['weight']:.2f}" for u, v, d in edges}
        self.nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=6)
        
        self.plt.title("Causal Relationship Graph", fontsize=16, fontweight='bold')
        self.plt.axis('off')
        self.plt.tight_layout()
        
        if save_path:
            self.plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"[CausalViz] Graph saved to {save_path}")
        else:
            self.plt.show()
        
        return True

    def generate_causal_chains(self, user_profile_id: str = None,
                             min_strength: float = 0.3) -> List[str]:
        """Generate causal chain analysis."""
        facts = self._get_causal_facts(user_profile_id, min_strength)
        
        if not facts:
            return ["No causal chains found"]
        
        # Build causal chains
        chains = []
        
        # Group facts by subjects to find chains
        subject_effects = defaultdict(list)
        for fact in facts:
            subject_effects[fact.subject].append(fact)
        
        lines = [
            "🔗 CAUSAL CHAIN ANALYSIS",
            "=" * 30,
            ""
        ]
        
        for subject, subject_facts in subject_effects.items():
            if len(subject_facts) > 1:
                lines.append(f"📊 {subject.title()} Causal Chain:")
                
                # Sort by causal strength
                sorted_facts = sorted(subject_facts, key=lambda x: x.causal_strength, reverse=True)
                
                for i, fact in enumerate(sorted_facts, 1):
                    cause = fact.cause if hasattr(fact, 'cause') and fact.cause else "unknown cause"
                    effect = f"{fact.predicate} {fact.object}"
                    strength_indicator = "⚡" if fact.causal_strength > 0.7 else "🔸" if fact.causal_strength > 0.4 else "⚪"
                    
                    lines.append(f"  {i}. {strength_indicator} {cause} → {effect}")
                    lines.append(f"     Strength: {fact.causal_strength:.3f}")
                
                lines.append("")
        
        return lines

    def _get_causal_facts(self, user_profile_id: str = None, 
                         min_strength: float = 0.3) -> List[EnhancedTripletFact]:
        """Get facts with causal information."""
        try:
            all_facts = self.memory_log.get_all_facts()
            
            # Filter for facts with causal information
            causal_facts = [
                f for f in all_facts 
                if hasattr(f, 'causal_strength') and f.causal_strength >= min_strength
            ]
            
            # Filter by user if specified
            if user_profile_id:
                causal_facts = [f for f in causal_facts if f.user_profile_id == user_profile_id]
            
            return causal_facts
            
        except Exception as e:
            print(f"[CausalViz] Error getting causal facts: {e}")
            return []

    def export_analysis_report(self, user_profile_id: str = None,
                             min_strength: float = 0.3) -> Dict[str, Any]:
        """Export a comprehensive causal analysis report."""
        return {
            "text_graph": self.generate_text_graph(user_profile_id, min_strength),
            "network_summary": self.generate_network_summary(user_profile_id, min_strength),
            "causal_chains": self.generate_causal_chains(user_profile_id, min_strength),
            "dot_graph": self.generate_dot_graph(user_profile_id, min_strength),
            "visualization_available": {
                "networkx": self.networkx_available,
                "matplotlib": self.matplotlib_available
            }
        } 