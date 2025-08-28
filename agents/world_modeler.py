#!/usr/bin/env python3
"""
World Modeler for MeRNSTA Phase 18: Internal World Modeling

Implements causal representation, belief management, and predictive simulation
to enable the system to model, reason about, and predict events in the world.
"""

import logging
import time
import json
import sqlite3
from typing import Dict, List, Tuple, Optional, Set, Any, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import defaultdict, deque
import math
import random

try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False
    logging.warning("NetworkX not available, using custom graph implementation")

from .base import BaseAgent
from config.settings import get_config
from .timeline_engine import get_timeline_engine, EventType


@dataclass
class CausalEdge:
    """Represents a causal relationship between concepts"""
    source: str
    target: str
    strength: float = 0.5  # 0.0 to 1.0
    confidence: float = 0.5  # 0.0 to 1.0
    edge_type: str = "causal"  # causal, temporal, correlational
    evidence_count: int = 1
    last_observed: float = field(default_factory=time.time)
    decay_rate: float = 0.98
    
    def decay(self, current_time: float = None) -> float:
        """Apply temporal decay to edge strength"""
        if current_time is None:
            current_time = time.time()
        
        time_diff = current_time - self.last_observed
        hours_passed = time_diff / 3600
        
        # Exponential decay
        new_strength = self.strength * (self.decay_rate ** hours_passed)
        self.strength = max(0.0, new_strength)
        return self.strength


@dataclass
class BeliefNode:
    """Represents a belief about the world"""
    concept: str
    truth_value: float = 0.5  # 0.0 (false) to 1.0 (true)
    confidence: float = 0.5  # How certain we are about this belief
    source: str = "observation"  # observation, inference, user_input
    created_at: float = field(default_factory=time.time)
    last_updated: float = field(default_factory=time.time)
    update_count: int = 1
    evidence: List[str] = field(default_factory=list)
    
    def get_recency_score(self, current_time: float = None) -> float:
        """Calculate recency-based score"""
        if current_time is None:
            current_time = time.time()
        
        hours_since_update = (current_time - self.last_updated) / 3600
        # Exponential decay with configurable half-life
        config = get_config().get('world_modeling', {})
        half_life_days = config.get('decay_half_life_days', 7)
        half_life_hours = half_life_days * 24
        
        decay_factor = 0.5 ** (hours_since_update / half_life_hours)
        return self.confidence * decay_factor
    
    def apply_truth_decay(self, decay_rate: float = 0.99):
        """Apply gradual truth decay if enabled"""
        config = get_config().get('world_modeling', {})
        if config.get('enable_truth_decay', True):
            self.confidence *= decay_rate


class CausalGraph:
    """
    Directed graph structure for causal relationships.
    Uses NetworkX if available, otherwise falls back to custom implementation.
    """
    
    def __init__(self):
        self.config = get_config().get('world_modeling', {})
        
        if NETWORKX_AVAILABLE:
            self.graph = nx.DiGraph()
            self.use_networkx = True
        else:
            # Custom graph implementation
            self.nodes: Set[str] = set()
            self.edges: Dict[Tuple[str, str], CausalEdge] = {}
            self.adjacency: Dict[str, Set[str]] = defaultdict(set)
            self.reverse_adjacency: Dict[str, Set[str]] = defaultdict(set)
            self.use_networkx = False
        
        logging.info(f"[CausalGraph] Initialized with {'NetworkX' if self.use_networkx else 'custom'} backend")
    
    def add_node(self, concept: str, **attributes):
        """Add a concept node to the graph"""
        if self.use_networkx:
            self.graph.add_node(concept, **attributes)
        else:
            self.nodes.add(concept)
    
    def add_edge(self, source: str, target: str, strength: float = 0.5, 
                 confidence: float = 0.5, edge_type: str = "causal", 
                 evidence: str = None) -> bool:
        """Add a causal edge between concepts"""
        # Validate parameters
        strength = max(0.0, min(1.0, strength))
        confidence = max(0.0, min(1.0, confidence))
        
        # Add nodes if they don't exist
        self.add_node(source)
        self.add_node(target)
        
        if self.use_networkx:
            if self.graph.has_edge(source, target):
                # Update existing edge
                edge_data = self.graph[source][target]
                edge_data['strength'] = (edge_data.get('strength', 0.5) + strength) / 2
                edge_data['confidence'] = max(edge_data.get('confidence', 0.5), confidence)
                edge_data['evidence_count'] = edge_data.get('evidence_count', 0) + 1
                edge_data['last_observed'] = time.time()
            else:
                # Add new edge
                self.graph.add_edge(source, target,
                                  strength=strength,
                                  confidence=confidence,
                                  edge_type=edge_type,
                                  evidence_count=1,
                                  last_observed=time.time())
            return True
        else:
            # Custom implementation
            edge_key = (source, target)
            if edge_key in self.edges:
                # Update existing edge
                existing_edge = self.edges[edge_key]
                existing_edge.strength = (existing_edge.strength + strength) / 2
                existing_edge.confidence = max(existing_edge.confidence, confidence)
                existing_edge.evidence_count += 1
                existing_edge.last_observed = time.time()
            else:
                # Add new edge
                self.edges[edge_key] = CausalEdge(
                    source=source, target=target, 
                    strength=strength, confidence=confidence, 
                    edge_type=edge_type
                )
                self.adjacency[source].add(target)
                self.reverse_adjacency[target].add(source)
            return True
    
    def get_edge_strength(self, source: str, target: str) -> float:
        """Get the strength of a causal edge"""
        if self.use_networkx:
            if self.graph.has_edge(source, target):
                return self.graph[source][target].get('strength', 0.0)
            return 0.0
        else:
            edge_key = (source, target)
            if edge_key in self.edges:
                return self.edges[edge_key].strength
            return 0.0
    
    def get_successors(self, concept: str) -> List[str]:
        """Get concepts that this concept causally influences"""
        if self.use_networkx:
            if concept in self.graph:
                return list(self.graph.successors(concept))
            return []
        else:
            return list(self.adjacency.get(concept, set()))
    
    def get_predecessors(self, concept: str) -> List[str]:
        """Get concepts that causally influence this concept"""
        if self.use_networkx:
            if concept in self.graph:
                return list(self.graph.predecessors(concept))
            return []
        else:
            return list(self.reverse_adjacency.get(concept, set()))
    
    def find_causal_path(self, source: str, target: str, max_depth: int = 5) -> List[List[str]]:
        """Find causal paths between two concepts"""
        if self.use_networkx:
            try:
                paths = list(nx.all_simple_paths(self.graph, source, target, cutoff=max_depth))
                return paths[:10]  # Limit to top 10 paths
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                return []
        else:
            # Custom BFS implementation
            paths = []
            queue = deque([(source, [source])])
            visited = set()
            
            while queue and len(paths) < 10:
                current, path = queue.popleft()
                
                if len(path) > max_depth:
                    continue
                
                if current == target and len(path) > 1:
                    paths.append(path)
                    continue
                
                if current in visited:
                    continue
                visited.add(current)
                
                for successor in self.get_successors(current):
                    if successor not in path:  # Avoid cycles
                        queue.append((successor, path + [successor]))
            
            return paths
    
    def get_strongest_edges(self, limit: int = 10) -> List[Tuple[str, str, float]]:
        """Get the strongest causal edges in the graph"""
        edges = []
        
        if self.use_networkx:
            for source, target, data in self.graph.edges(data=True):
                strength = data.get('strength', 0.0)
                confidence = data.get('confidence', 0.0)
                combined_score = strength * confidence
                edges.append((source, target, combined_score))
        else:
            for (source, target), edge in self.edges.items():
                combined_score = edge.strength * edge.confidence
                edges.append((source, target, combined_score))
        
        edges.sort(key=lambda x: x[2], reverse=True)
        return edges[:limit]
    
    def apply_decay(self):
        """Apply temporal decay to all edges"""
        current_time = time.time()
        
        if self.use_networkx:
            for source, target, data in self.graph.edges(data=True):
                last_observed = data.get('last_observed', current_time)
                decay_rate = data.get('decay_rate', 0.98)
                time_diff = current_time - last_observed
                hours_passed = time_diff / 3600
                
                new_strength = data.get('strength', 0.5) * (decay_rate ** hours_passed)
                data['strength'] = max(0.0, new_strength)
        else:
            edges_to_remove = []
            for edge_key, edge in self.edges.items():
                edge.decay(current_time)
                if edge.strength < 0.01:  # Remove very weak edges
                    edges_to_remove.append(edge_key)
            
            for edge_key in edges_to_remove:
                source, target = edge_key
                del self.edges[edge_key]
                self.adjacency[source].discard(target)
                self.reverse_adjacency[target].discard(source)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get graph statistics"""
        if self.use_networkx:
            return {
                'nodes': self.graph.number_of_nodes(),
                'edges': self.graph.number_of_edges(),
                'avg_degree': sum(dict(self.graph.degree()).values()) / max(1, self.graph.number_of_nodes()),
                'connected_components': nx.number_weakly_connected_components(self.graph)
            }
        else:
            total_degree = sum(len(successors) for successors in self.adjacency.values())
            return {
                'nodes': len(self.nodes),
                'edges': len(self.edges),
                'avg_degree': total_degree / max(1, len(self.nodes)),
                'connected_components': 1  # Simplified for custom implementation
            }


class BeliefState:
    """
    Manages current beliefs with truth decay and recency scoring.
    Supports querying current state and causal reasoning.
    """
    
    def __init__(self, db_path: str = "world_beliefs.db"):
        self.db_path = db_path
        self.config = get_config().get('world_modeling', {})
        self.max_beliefs = self.config.get('max_beliefs', 5000)
        self.beliefs: Dict[str, BeliefNode] = {}
        
        # Initialize database
        self._init_db()
        self._load_beliefs()
        
        logging.info(f"[BeliefState] Initialized with {len(self.beliefs)} beliefs")
    
    def _init_db(self):
        """Initialize SQLite database for persistent belief storage"""
        conn = sqlite3.connect(self.db_path)
        conn.execute('''
            CREATE TABLE IF NOT EXISTS beliefs (
                concept TEXT PRIMARY KEY,
                truth_value REAL,
                confidence REAL,
                source TEXT,
                created_at REAL,
                last_updated REAL,
                update_count INTEGER,
                evidence TEXT
            )
        ''')
        conn.commit()
        conn.close()
    
    def _load_beliefs(self):
        """Load beliefs from database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.execute('SELECT * FROM beliefs')
        
        for row in cursor:
            concept, truth_value, confidence, source, created_at, last_updated, update_count, evidence_json = row
            evidence = json.loads(evidence_json) if evidence_json else []
            
            self.beliefs[concept] = BeliefNode(
                concept=concept,
                truth_value=truth_value,
                confidence=confidence,
                source=source,
                created_at=created_at,
                last_updated=last_updated,
                update_count=update_count,
                evidence=evidence
            )
        
        conn.close()
    
    def _save_belief(self, belief: BeliefNode):
        """Save belief to database"""
        conn = sqlite3.connect(self.db_path)
        evidence_json = json.dumps(belief.evidence)
        
        conn.execute('''
            INSERT OR REPLACE INTO beliefs 
            (concept, truth_value, confidence, source, created_at, last_updated, update_count, evidence)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (belief.concept, belief.truth_value, belief.confidence, belief.source,
              belief.created_at, belief.last_updated, belief.update_count, evidence_json))
        
        conn.commit()
        conn.close()
    
    def update_belief(self, concept: str, truth_value: float, confidence: float, 
                     source: str = "observation", evidence: str = None) -> BeliefNode:
        """Update or create a belief"""
        truth_value = max(0.0, min(1.0, truth_value))
        confidence = max(0.0, min(1.0, confidence))
        
        current_time = time.time()
        
        if concept in self.beliefs:
            # Update existing belief
            belief = self.beliefs[concept]
            
            # Weighted average based on confidence
            old_weight = belief.confidence
            new_weight = confidence
            total_weight = old_weight + new_weight
            
            if total_weight > 0:
                belief.truth_value = (belief.truth_value * old_weight + truth_value * new_weight) / total_weight
                belief.confidence = max(belief.confidence, confidence)
            
            belief.last_updated = current_time
            belief.update_count += 1
            
            if evidence:
                belief.evidence.append(evidence)
                # Keep only recent evidence
                belief.evidence = belief.evidence[-10:]
        else:
            # Create new belief
            belief = BeliefNode(
                concept=concept,
                truth_value=truth_value,
                confidence=confidence,
                source=source,
                created_at=current_time,
                last_updated=current_time,
                evidence=[evidence] if evidence else []
            )
            self.beliefs[concept] = belief
        
        self._save_belief(belief)
        self._manage_belief_capacity()
        
        return belief
    
    def get_belief(self, concept: str) -> Optional[BeliefNode]:
        """Get belief about a concept"""
        return self.beliefs.get(concept)
    
    def query_beliefs(self, pattern: str = None, min_confidence: float = 0.0, 
                     min_truth: float = 0.0) -> List[BeliefNode]:
        """Query beliefs with optional filtering"""
        results = []
        
        for belief in self.beliefs.values():
            # Apply filters
            if belief.confidence < min_confidence:
                continue
            if belief.truth_value < min_truth:
                continue
            if pattern and pattern.lower() not in belief.concept.lower():
                continue
            
            results.append(belief)
        
        # Sort by recency score
        results.sort(key=lambda b: b.get_recency_score(), reverse=True)
        return results
    
    def get_current_beliefs(self, limit: int = 20) -> List[BeliefNode]:
        """Get currently held beliefs ranked by recency and confidence"""
        beliefs = list(self.beliefs.values())
        beliefs.sort(key=lambda b: b.get_recency_score(), reverse=True)
        return beliefs[:limit]
    
    def answer_query(self, query: str) -> Dict[str, Any]:
        """Answer queries about current beliefs"""
        query_lower = query.lower()
        
        # Simple keyword matching for now
        relevant_beliefs = []
        for belief in self.beliefs.values():
            if any(keyword in belief.concept.lower() for keyword in query_lower.split()):
                relevant_beliefs.append(belief)
        
        relevant_beliefs.sort(key=lambda b: b.get_recency_score(), reverse=True)
        
        return {
            'query': query,
            'relevant_beliefs': relevant_beliefs[:5],
            'confidence': max([b.confidence for b in relevant_beliefs[:3]], default=0.0),
            'evidence_count': sum(len(b.evidence) for b in relevant_beliefs[:3])
        }
    
    def apply_truth_decay(self):
        """Apply truth decay to all beliefs"""
        if not self.config.get('enable_truth_decay', True):
            return
        
        decay_rate = 0.995  # Slow decay
        beliefs_to_remove = []
        
        for concept, belief in self.beliefs.items():
            belief.apply_truth_decay(decay_rate)
            
            # Remove beliefs that have become too uncertain
            if belief.confidence < 0.1:
                beliefs_to_remove.append(concept)
        
        # Remove low-confidence beliefs
        for concept in beliefs_to_remove:
            del self.beliefs[concept]
            # Also remove from database
            conn = sqlite3.connect(self.db_path)
            conn.execute('DELETE FROM beliefs WHERE concept = ?', (concept,))
            conn.commit()
            conn.close()
    
    def _manage_belief_capacity(self):
        """Manage belief capacity by removing oldest/weakest beliefs"""
        if len(self.beliefs) <= self.max_beliefs:
            return
        
        # Sort by recency score (ascending, so weakest first)
        beliefs_by_score = sorted(self.beliefs.items(), 
                                key=lambda x: x[1].get_recency_score())
        
        # Remove weakest beliefs
        num_to_remove = len(self.beliefs) - self.max_beliefs
        for i in range(num_to_remove):
            concept, belief = beliefs_by_score[i]
            del self.beliefs[concept]
            
            # Remove from database
            conn = sqlite3.connect(self.db_path)
            conn.execute('DELETE FROM beliefs WHERE concept = ?', (concept,))
            conn.commit()
            conn.close()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get belief state statistics"""
        if not self.beliefs:
            return {'total_beliefs': 0}
        
        confidences = [b.confidence for b in self.beliefs.values()]
        truth_values = [b.truth_value for b in self.beliefs.values()]
        update_counts = [b.update_count for b in self.beliefs.values()]
        
        return {
            'total_beliefs': len(self.beliefs),
            'avg_confidence': sum(confidences) / len(confidences),
            'avg_truth_value': sum(truth_values) / len(truth_values),
            'avg_updates': sum(update_counts) / len(update_counts),
            'sources': list(set(b.source for b in self.beliefs.values()))
        }


@dataclass 
class PredictionResult:
    """Result of a predictive simulation"""
    predicted_states: List[Dict[str, Any]]
    confidence: float
    reasoning: str
    causal_chain: List[str]
    probability: float = 0.5
    time_horizon: float = 24.0  # hours


class PredictiveSimulator:
    """
    Simulates likely next states and performs causal reasoning.
    Supports forward prediction and backward reasoning.
    """
    
    def __init__(self, causal_graph: CausalGraph, belief_state: BeliefState):
        self.causal_graph = causal_graph
        self.belief_state = belief_state
        self.config = get_config().get('world_modeling', {})
        self.prediction_horizon = self.config.get('prediction_horizon', 5)
        
        logging.info(f"[PredictiveSimulator] Initialized with horizon={self.prediction_horizon}")
    
    def predict_next_state(self, current_state: Dict[str, Any], 
                          action: str = None) -> PredictionResult:
        """Predict likely next states given current state and optional action"""
        
        # Extract active concepts from current state
        active_concepts = []
        for key, value in current_state.items():
            if isinstance(value, (int, float)) and value > 0.5:
                active_concepts.append(key)
            elif isinstance(value, str) and value.lower() in ['true', 'active', 'high']:
                active_concepts.append(key)
        
        if action:
            active_concepts.append(action)
        
        # Find causal effects
        predicted_effects = set()
        causal_chains = []
        total_confidence = 0.0
        
        for concept in active_concepts:
            successors = self.causal_graph.get_successors(concept)
            for successor in successors:
                strength = self.causal_graph.get_edge_strength(concept, successor)
                if strength > 0.3:  # Only consider strong relationships
                    predicted_effects.add(successor)
                    causal_chains.append([concept, successor])
                    total_confidence += strength
        
        # Build predicted state
        predicted_states = []
        for effect in predicted_effects:
            belief = self.belief_state.get_belief(effect)
            effect_probability = 0.5
            
            if belief:
                effect_probability = belief.truth_value * belief.confidence
            
            predicted_states.append({
                'concept': effect,
                'probability': effect_probability,
                'activation_strength': min(1.0, total_confidence / len(predicted_effects) if predicted_effects else 0.0)
            })
        
        # Sort by probability
        predicted_states.sort(key=lambda x: x['probability'], reverse=True)
        
        # Calculate overall confidence
        overall_confidence = min(1.0, total_confidence / max(1, len(active_concepts)))
        
        reasoning = f"Based on {len(active_concepts)} active concepts, "
        reasoning += f"predicted {len(predicted_effects)} potential effects through "
        reasoning += f"{len(causal_chains)} causal relationships."
        
        return PredictionResult(
            predicted_states=predicted_states[:self.prediction_horizon],
            confidence=overall_confidence,
            reasoning=reasoning,
            causal_chain=[chain for chain in causal_chains],
            probability=overall_confidence,
            time_horizon=24.0
        )
    
    def find_possible_causes(self, target_effect: str) -> PredictionResult:
        """Find possible causes for a given effect (backward reasoning)"""
        
        # Get direct predecessors
        predecessors = self.causal_graph.get_predecessors(target_effect)
        
        possible_causes = []
        causal_chains = []
        total_strength = 0.0
        
        for predecessor in predecessors:
            strength = self.causal_graph.get_edge_strength(predecessor, target_effect)
            belief = self.belief_state.get_belief(predecessor)
            
            # Calculate likelihood that this cause is currently active
            cause_likelihood = 0.5
            if belief:
                cause_likelihood = belief.truth_value * belief.confidence
            
            possible_causes.append({
                'concept': predecessor,
                'causal_strength': strength,
                'current_likelihood': cause_likelihood,
                'combined_score': strength * cause_likelihood
            })
            
            causal_chains.append([predecessor, target_effect])
            total_strength += strength
        
        # Sort by combined score
        possible_causes.sort(key=lambda x: x['combined_score'], reverse=True)
        
        # Also look for indirect causes (2-step chains)
        for predecessor in predecessors:
            second_order_predecessors = self.causal_graph.get_predecessors(predecessor)
            for second_pred in second_order_predecessors:
                strength1 = self.causal_graph.get_edge_strength(second_pred, predecessor)
                strength2 = self.causal_graph.get_edge_strength(predecessor, target_effect)
                combined_strength = strength1 * strength2
                
                if combined_strength > 0.2:  # Only consider reasonably strong chains
                    belief = self.belief_state.get_belief(second_pred)
                    cause_likelihood = belief.truth_value * belief.confidence if belief else 0.5
                    
                    possible_causes.append({
                        'concept': second_pred,
                        'causal_strength': combined_strength,
                        'current_likelihood': cause_likelihood,
                        'combined_score': combined_strength * cause_likelihood,
                        'indirect': True,
                        'through': predecessor
                    })
                    
                    causal_chains.append([second_pred, predecessor, target_effect])
        
        # Re-sort including indirect causes
        possible_causes.sort(key=lambda x: x['combined_score'], reverse=True)
        
        overall_confidence = min(1.0, total_strength / max(1, len(predecessors)))
        
        reasoning = f"Found {len(possible_causes)} potential causes for '{target_effect}'. "
        reasoning += f"Analysis includes {len(predecessors)} direct and "
        reasoning += f"{len(possible_causes) - len(predecessors)} indirect causal relationships."
        
        return PredictionResult(
            predicted_states=possible_causes[:self.prediction_horizon],
            confidence=overall_confidence,
            reasoning=reasoning,
            causal_chain=causal_chains,
            probability=overall_confidence
        )
    
    def simulate_scenario(self, initial_state: Dict[str, Any], 
                         steps: int = 3) -> List[PredictionResult]:
        """Simulate multiple steps forward"""
        results = []
        current_state = initial_state.copy()
        
        for step in range(steps):
            prediction = self.predict_next_state(current_state)
            results.append(prediction)
            
            # Update current state based on prediction
            for predicted_state in prediction.predicted_states:
                if predicted_state['probability'] > 0.6:
                    current_state[predicted_state['concept']] = predicted_state['probability']
        
        return results
    
    def explain_causal_chain(self, chain: List[str]) -> str:
        """Generate natural language explanation of a causal chain"""
        if len(chain) < 2:
            return "No causal relationship to explain."
        
        explanation = ""
        for i in range(len(chain) - 1):
            cause = chain[i]
            effect = chain[i + 1]
            strength = self.causal_graph.get_edge_strength(cause, effect)
            
            strength_desc = "strongly" if strength > 0.7 else "moderately" if strength > 0.4 else "weakly"
            explanation += f"'{cause}' {strength_desc} influences '{effect}'"
            
            if i < len(chain) - 2:
                explanation += ", which in turn "
            else:
                explanation += "."
        
        return explanation


class WorldModeler(BaseAgent):
    """
    Main World Modeler agent that integrates all components.
    Provides high-level interface for world modeling capabilities.
    """
    
    def __init__(self):
        super().__init__("world_modeler")
        
        self.causal_graph = CausalGraph()
        self.belief_state = BeliefState()
        self.simulator = PredictiveSimulator(self.causal_graph, self.belief_state)
        
        # Timeline engine for temporal tracking
        self.timeline = get_timeline_engine()
        
        # Background maintenance
        self.last_decay_time = time.time()
        self.decay_interval = 3600  # 1 hour
        
        logging.info("[WorldModeler] Initialized with full world modeling capabilities and timeline engine")
    
    def get_agent_instructions(self) -> str:
        """Get instructions for this agent's role and capabilities"""
        return """You are the World Modeler agent for MeRNSTA's Phase 18 Internal World Modeling system.

Your primary responsibilities are:
1. Maintain an internal causal graph representing relationships between concepts
2. Manage beliefs about the world with truth decay and recency scoring
3. Perform predictive simulation for forward and backward reasoning
4. Process observations and experiences to update the world model
5. Answer queries about current beliefs and causal relationships

Key capabilities:
- Causal graph construction and maintenance with temporal decay
- Belief state management with confidence tracking
- Predictive simulation for event outcomes and cause analysis
- Natural language explanation of causal chains
- Integration with memory system for automatic world model updates

Use your world modeling capabilities to help understand relationships between facts,
predict likely outcomes of events, and explain causal reasoning behind beliefs."""
    
    def respond(self, message: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Process a message and respond using world modeling capabilities"""
        
        # Perform maintenance if needed
        self.maintenance_cycle()
        
        message_lower = message.lower()
        
        try:
            # Handle different types of queries
            if any(word in message_lower for word in ['predict', 'outcome', 'what will happen']):
                # Extract event from message
                event = message.replace('predict ', '').replace('what will happen if ', '').strip()
                prediction = self.predict_event_outcomes(event)
                
                return {
                    'response': f"Based on my world model, here are the predicted outcomes for '{event}':",
                    'prediction_data': prediction,
                    'confidence': prediction['prediction_confidence'],
                    'agent': 'world_modeler'
                }
            
            elif any(word in message_lower for word in ['believe', 'belief', 'think', 'true']):
                # Extract fact/belief from message
                fact = message.replace('do you believe ', '').replace('is it true that ', '').strip()
                analysis = self.get_belief_analysis(fact)
                
                if analysis.get('status') == 'unknown':
                    response = f"I don't have any established beliefs about '{fact}' yet."
                else:
                    belief = analysis['belief']
                    confidence_desc = "high" if belief['confidence'] >= 0.8 else "medium" if belief['confidence'] >= 0.6 else "low"
                    truth_desc = "likely true" if belief['truth_value'] >= 0.7 else "uncertain" if belief['truth_value'] >= 0.3 else "likely false"
                    
                    response = f"My belief about '{fact}': {truth_desc} (confidence: {confidence_desc})"
                
                return {
                    'response': response,
                    'belief_analysis': analysis,
                    'agent': 'world_modeler'
                }
            
            elif any(word in message_lower for word in ['cause', 'why', 'because', 'reason']):
                # Extract effect from message
                effect = message.replace('why ', '').replace('what causes ', '').strip()
                causes_result = self.simulator.find_possible_causes(effect)
                
                if causes_result.predicted_states:
                    top_cause = causes_result.predicted_states[0]
                    response = f"The most likely cause of '{effect}' is '{top_cause['concept']}' with {top_cause.get('causal_strength', 0.0):.2f} causal strength."
                else:
                    response = f"I don't have enough causal information to determine what causes '{effect}'."
                
                return {
                    'response': response,
                    'causal_analysis': causes_result,
                    'agent': 'world_modeler'
                }
            
            elif 'observe' in message_lower or 'learn' in message_lower:
                # Extract fact to observe
                fact = message.replace('observe that ', '').replace('learn that ', '').strip()
                self.observe_fact(fact, confidence=0.8, source="user_input")
                
                return {
                    'response': f"I've observed and integrated the fact: '{fact}' into my world model.",
                    'agent': 'world_modeler'
                }
            
            else:
                # General world state query
                world_state = self.get_world_state_summary(limit=5)
                
                beliefs_count = len(world_state['current_beliefs'])
                chains_count = len(world_state['causal_chains'])
                
                response = f"My current world model contains {beliefs_count} active beliefs and {chains_count} causal relationships. "
                response += f"Use specific queries like 'predict X', 'do you believe Y', or 'what causes Z' for detailed analysis."
                
                return {
                    'response': response,
                    'world_state_summary': world_state,
                    'agent': 'world_modeler'
                }
        
        except Exception as e:
            logging.error(f"[WorldModeler] Error processing message: {e}")
            return {
                'response': f"I encountered an error while processing your request: {str(e)}",
                'error': str(e),
                'agent': 'world_modeler'
            }
    
    def observe_fact(self, fact: str, confidence: float = 0.8, source: str = "observation"):
        """Process an observed fact and update world model"""
        
        # Add to timeline first
        timeline_event = self.timeline.add_event(
            fact=fact,
            confidence=confidence,
            source=source,
            event_type=EventType.OBSERVATION,
            reasoning_agent="world_modeler",
            metadata={"method": "observe_fact"}
        )
        
        # Update belief with timeline tracking
        self.update_belief_with_timeline(fact, truth_value=1.0, confidence=confidence, 
                                        source=source, evidence=f"Observed: {fact}",
                                        event_type=EventType.OBSERVATION)
        
        # Extract potential causal relationships (simplified)
        # In a more sophisticated system, this would use NLP
        words = fact.lower().split()
        causal_indicators = ['because', 'due to', 'caused by', 'leads to', 'results in']
        
        for indicator in causal_indicators:
            if indicator in fact.lower():
                # Try to extract cause-effect relationship
                parts = fact.lower().split(indicator)
                if len(parts) == 2:
                    cause = parts[0].strip()
                    effect = parts[1].strip()
                    self.causal_graph.add_edge(cause, effect, strength=confidence, confidence=confidence)
                    
                    # Add causal relationship to timeline
                    self.timeline.add_event(
                        fact=f"Causal relationship detected: {cause} -> {effect}",
                        confidence=confidence,
                        source=source,
                        event_type=EventType.INFERRED,
                        reasoning_agent="world_modeler",
                        metadata={"method": "observe_fact", "causal_link": True, "cause": cause, "effect": effect}
                    )
                break
    
    def update_belief_with_timeline(self, concept: str, truth_value: float, confidence: float,
                                   source: str = "observation", evidence: str = None,
                                   event_type: EventType = EventType.UPDATE) -> 'BeliefNode':
        """
        Update a belief and record the change in the timeline.
        This is the main method that should be used instead of calling belief_state.update_belief directly.
        """
        # Check if this is an update or new belief
        existing_belief = self.belief_state.get_belief(concept)
        is_new_belief = existing_belief is None
        
        # Update the belief
        belief = self.belief_state.update_belief(concept, truth_value, confidence, source, evidence)
        
        # Add to timeline
        if is_new_belief:
            timeline_fact = f"New belief: {concept} (truth: {truth_value:.2f}, confidence: {confidence:.2f})"
            timeline_event_type = EventType.BELIEF_CHANGE
        else:
            # Check if this is a significant change
            old_truth = existing_belief.truth_value
            old_confidence = existing_belief.confidence
            truth_change = abs(truth_value - old_truth)
            confidence_change = abs(confidence - old_confidence)
            
            if truth_change > 0.1 or confidence_change > 0.1:
                timeline_fact = f"Belief updated: {concept} (truth: {old_truth:.2f}→{truth_value:.2f}, confidence: {old_confidence:.2f}→{confidence:.2f})"
                timeline_event_type = EventType.BELIEF_CHANGE
            else:
                timeline_fact = f"Belief reinforced: {concept} (truth: {truth_value:.2f}, confidence: {confidence:.2f})"
                timeline_event_type = event_type
        
        # Add timeline event
        self.timeline.add_event(
            fact=timeline_fact,
            confidence=confidence,
            source=source,
            event_type=timeline_event_type,
            reasoning_agent="world_modeler",
            metadata={
                "method": "update_belief_with_timeline",
                "concept": concept,
                "truth_value": truth_value,
                "is_new_belief": is_new_belief,
                "evidence": evidence
            }
        )
        
        return belief
    
    def process_experience(self, experience: Dict[str, Any]):
        """Process a complex experience with multiple facts and relationships"""
        
        # Extract facts from experience
        for key, value in experience.items():
            if isinstance(value, bool):
                confidence = 0.9 if value else 0.1
                self.belief_state.update_belief(key, truth_value=1.0 if value else 0.0, 
                                              confidence=confidence, source="experience")
            elif isinstance(value, (int, float)):
                # Normalize to 0-1 range and use as truth value
                truth_value = max(0.0, min(1.0, value))
                self.belief_state.update_belief(key, truth_value=truth_value, 
                                              confidence=0.8, source="experience")
        
        # Look for temporal patterns to create causal edges
        experience_keys = list(experience.keys())
        for i in range(len(experience_keys) - 1):
            for j in range(i + 1, len(experience_keys)):
                key1, key2 = experience_keys[i], experience_keys[j]
                
                # Simple heuristic: if both are active/high, create weak causal edge
                val1 = experience[key1]
                val2 = experience[key2]
                
                if (isinstance(val1, (int, float)) and val1 > 0.5 and 
                    isinstance(val2, (int, float)) and val2 > 0.5):
                    self.causal_graph.add_edge(key1, key2, strength=0.3, confidence=0.4)
    
    def get_world_state_summary(self, limit: int = 20) -> Dict[str, Any]:
        """Get summary of current world state and top causal chains"""
        
        current_beliefs = self.belief_state.get_current_beliefs(limit)
        strongest_edges = self.causal_graph.get_strongest_edges(limit)
        
        # Format causal chains
        causal_chains = []
        for source, target, strength in strongest_edges:
            explanation = self.simulator.explain_causal_chain([source, target])
            causal_chains.append({
                'chain': [source, target],
                'strength': strength,
                'explanation': explanation
            })
        
        # Get statistics
        belief_stats = self.belief_state.get_stats()
        graph_stats = self.causal_graph.get_stats()
        
        return {
            'current_beliefs': [
                {
                    'concept': b.concept,
                    'truth_value': b.truth_value,
                    'confidence': b.confidence,
                    'recency_score': b.get_recency_score(),
                    'source': b.source
                } for b in current_beliefs
            ],
            'causal_chains': causal_chains,
            'belief_statistics': belief_stats,
            'graph_statistics': graph_stats,
            'last_updated': datetime.now().isoformat()
        }
    
    def predict_event_outcomes(self, event: str) -> Dict[str, Any]:
        """Predict likely outcomes of an event"""
        
        # Create a simple state representation
        current_state = {event: 1.0}
        
        # Get prediction
        prediction = self.simulator.predict_next_state(current_state)
        
        # Also find possible causes
        causes = self.simulator.find_possible_causes(event)
        
        return {
            'event': event,
            'predicted_outcomes': prediction.predicted_states,
            'prediction_confidence': prediction.confidence,
            'reasoning': prediction.reasoning,
            'possible_causes': causes.predicted_states,
            'causal_reasoning': causes.reasoning,
            'time_horizon_hours': prediction.time_horizon
        }
    
    def get_belief_analysis(self, fact: str) -> Dict[str, Any]:
        """Get detailed analysis of a belief"""
        
        belief = self.belief_state.get_belief(fact)
        if not belief:
            return {
                'fact': fact,
                'status': 'unknown',
                'message': 'No belief found for this fact'
            }
        
        # Find related beliefs
        related_query = self.belief_state.answer_query(fact)
        
        # Find causal relationships
        successors = self.causal_graph.get_successors(fact)
        predecessors = self.causal_graph.get_predecessors(fact)
        
        return {
            'fact': fact,
            'belief': {
                'truth_value': belief.truth_value,
                'confidence': belief.confidence,
                'recency_score': belief.get_recency_score(),
                'source': belief.source,
                'update_count': belief.update_count,
                'evidence': belief.evidence
            },
            'related_beliefs': [
                {
                    'concept': b.concept,
                    'truth_value': b.truth_value,
                    'confidence': b.confidence
                } for b in related_query['relevant_beliefs']
            ],
            'causal_influences': {
                'causes': predecessors,
                'effects': successors
            },
            'causal_explanations': {
                'incoming': [self.simulator.explain_causal_chain([pred, fact]) for pred in predecessors],
                'outgoing': [self.simulator.explain_causal_chain([fact, succ]) for succ in successors]
            }
        }
    
    def maintenance_cycle(self):
        """Perform background maintenance"""
        current_time = time.time()
        
        if current_time - self.last_decay_time > self.decay_interval:
            logging.info("[WorldModeler] Running maintenance cycle")
            
            # Apply decay
            self.causal_graph.apply_decay()
            self.belief_state.apply_truth_decay()
            
            self.last_decay_time = current_time
            
            # Log statistics
            belief_stats = self.belief_state.get_stats()
            graph_stats = self.causal_graph.get_stats()
            
            logging.info(f"[WorldModeler] Maintenance complete. "
                        f"Beliefs: {belief_stats.get('total_beliefs', 0)}, "
                        f"Causal edges: {graph_stats.get('edges', 0)}")


# Global instance for easy access
_world_modeler_instance = None

def get_world_modeler() -> WorldModeler:
    """Get global world modeler instance"""
    global _world_modeler_instance
    if _world_modeler_instance is None:
        _world_modeler_instance = WorldModeler()
    return _world_modeler_instance