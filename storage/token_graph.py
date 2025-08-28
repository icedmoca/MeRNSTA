#!/usr/bin/env python3
"""
Token-Causal Graph System for MeRNSTA

Tracks token propagation across beliefs, detects causal token clusters,
builds influence heatmaps, and supports token drift detection.
"""

import logging
import time
import json
from typing import Dict, Set, List, Optional, Tuple, Any
from collections import defaultdict, Counter
from dataclasses import dataclass, field
import math
import numpy as np

from .enhanced_memory_model import EnhancedTripletFact, compute_entropy


@dataclass
class TokenNode:
    """Represents a token in the propagation graph."""
    token_id: int
    fact_ids: Set[str] = field(default_factory=set)
    first_seen: float = 0.0
    last_seen: float = 0.0
    usage_count: int = 0
    entropy_score: float = 0.0
    drift_score: float = 0.0
    influence_score: float = 0.0
    semantic_cluster: Optional[str] = None


@dataclass
class TokenCluster:
    """Represents a cluster of semantically related tokens."""
    cluster_id: str
    token_ids: Set[int] = field(default_factory=set)
    fact_ids: Set[str] = field(default_factory=set)
    semantic_center: Optional[int] = None
    coherence_score: float = 0.0
    created_time: float = 0.0
    last_updated: float = 0.0


@dataclass
class TokenDriftEvent:
    """Represents a detected token drift event."""
    token_id: int
    old_semantic_cluster: str
    new_semantic_cluster: str
    drift_score: float
    timestamp: float
    affected_facts: List[str]
    drift_type: str  # "semantic_shift", "usage_increase", "usage_decrease"


class TokenPropagationGraph:
    """
    Tracks token propagation across beliefs and detects causal patterns.
    
    Features:
    - Token→Fact mapping
    - Causal token clusters
    - Influence heatmaps
    - Token drift detection
    - Semantic clustering
    """
    
    def __init__(self, db_path: str = "token_graph.db"):
        self.db_path = db_path
        self.graph: Dict[int, TokenNode] = {}  # token_id → TokenNode
        self.fact_to_tokens: Dict[str, Set[int]] = defaultdict(set)  # fact_id → set(token_ids)
        self.clusters: Dict[str, TokenCluster] = {}  # cluster_id → TokenCluster
        self.drift_events: List[TokenDriftEvent] = []
        
        # Configuration
        self.cluster_threshold = 0.7  # Similarity threshold for clustering
        self.drift_threshold = 0.3  # Threshold for drift detection
        self.max_cluster_size = 50  # Maximum tokens per cluster
        
        # Initialize database
        self._init_database()
        
        print(f"[TokenGraph] Initialized token propagation graph")
    
    def _init_database(self):
        """Initialize database for persistent storage."""
        import sqlite3
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Token nodes table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS token_nodes (
                token_id INTEGER PRIMARY KEY,
                fact_ids TEXT,
                first_seen REAL,
                last_seen REAL,
                usage_count INTEGER DEFAULT 0,
                entropy_score REAL DEFAULT 0.0,
                drift_score REAL DEFAULT 0.0,
                influence_score REAL DEFAULT 0.0,
                semantic_cluster TEXT
            )
        """)
        
        # Token clusters table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS token_clusters (
                cluster_id TEXT PRIMARY KEY,
                token_ids TEXT,
                fact_ids TEXT,
                semantic_center INTEGER,
                coherence_score REAL DEFAULT 0.0,
                created_time REAL,
                last_updated REAL
            )
        """)
        
        # Drift events table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS drift_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                token_id INTEGER,
                old_cluster TEXT,
                new_cluster TEXT,
                drift_score REAL,
                timestamp REAL,
                affected_facts TEXT,
                drift_type TEXT
            )
        """)
        
        # Load existing data
        self._load_from_database()
        
        conn.commit()
        conn.close()
    
    def _load_from_database(self):
        """Load existing token graph data from database."""
        import sqlite3
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Load token nodes
            cursor.execute("SELECT * FROM token_nodes")
            for row in cursor.fetchall():
                token_id, fact_ids_str, first_seen, last_seen, usage_count, entropy_score, drift_score, influence_score, semantic_cluster = row
                
                fact_ids = set(json.loads(fact_ids_str)) if fact_ids_str else set()
                
                self.graph[token_id] = TokenNode(
                    token_id=token_id,
                    fact_ids=fact_ids,
                    first_seen=first_seen,
                    last_seen=last_seen,
                    usage_count=usage_count,
                    entropy_score=entropy_score,
                    drift_score=drift_score,
                    influence_score=influence_score,
                    semantic_cluster=semantic_cluster
                )
                
                # Update fact_to_tokens mapping
                for fact_id in fact_ids:
                    self.fact_to_tokens[fact_id].add(token_id)
            
            # Load clusters
            cursor.execute("SELECT * FROM token_clusters")
            for row in cursor.fetchall():
                cluster_id, token_ids_str, fact_ids_str, semantic_center, coherence_score, created_time, last_updated = row
                
                token_ids = set(json.loads(token_ids_str)) if token_ids_str else set()
                fact_ids = set(json.loads(fact_ids_str)) if fact_ids_str else set()
                
                self.clusters[cluster_id] = TokenCluster(
                    cluster_id=cluster_id,
                    token_ids=token_ids,
                    fact_ids=fact_ids,
                    semantic_center=semantic_center,
                    coherence_score=coherence_score,
                    created_time=created_time,
                    last_updated=last_updated
                )
            
            # Load drift events
            cursor.execute("SELECT * FROM drift_events ORDER BY timestamp DESC LIMIT 100")
            for row in cursor.fetchall():
                _, token_id, old_cluster, new_cluster, drift_score, timestamp, affected_facts_str, drift_type = row
                
                affected_facts = json.loads(affected_facts_str) if affected_facts_str else []
                
                self.drift_events.append(TokenDriftEvent(
                    token_id=token_id,
                    old_semantic_cluster=old_cluster,
                    new_semantic_cluster=new_cluster,
                    drift_score=drift_score,
                    timestamp=timestamp,
                    affected_facts=affected_facts,
                    drift_type=drift_type
                ))
            
            conn.close()
            print(f"[TokenGraph] Loaded {len(self.graph)} tokens, {len(self.clusters)} clusters, {len(self.drift_events)} drift events")
            
        except Exception as e:
            print(f"[TokenGraph] Error loading from database: {e}")
    
    def _save_to_database(self):
        """Save current state to database."""
        import sqlite3
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Clear existing data
            cursor.execute("DELETE FROM token_nodes")
            cursor.execute("DELETE FROM token_clusters")
            cursor.execute("DELETE FROM drift_events")
            
            # Save token nodes
            for token_id, node in self.graph.items():
                cursor.execute("""
                    INSERT INTO token_nodes 
                    (token_id, fact_ids, first_seen, last_seen, usage_count, entropy_score, drift_score, influence_score, semantic_cluster)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    token_id,
                    json.dumps(list(node.fact_ids)),
                    node.first_seen,
                    node.last_seen,
                    node.usage_count,
                    node.entropy_score,
                    node.drift_score,
                    node.influence_score,
                    node.semantic_cluster
                ))
            
            # Save clusters
            for cluster_id, cluster in self.clusters.items():
                cursor.execute("""
                    INSERT INTO token_clusters 
                    (cluster_id, token_ids, fact_ids, semantic_center, coherence_score, created_time, last_updated)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    cluster_id,
                    json.dumps(list(cluster.token_ids)),
                    json.dumps(list(cluster.fact_ids)),
                    cluster.semantic_center,
                    cluster.coherence_score,
                    cluster.created_time,
                    cluster.last_updated
                ))
            
            # Save drift events
            for event in self.drift_events[-100:]:  # Keep last 100 events
                cursor.execute("""
                    INSERT INTO drift_events 
                    (token_id, old_cluster, new_cluster, drift_score, timestamp, affected_facts, drift_type)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    event.token_id,
                    event.old_semantic_cluster,
                    event.new_semantic_cluster,
                    event.drift_score,
                    event.timestamp,
                    json.dumps(event.affected_facts),
                    event.drift_type
                ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            print(f"[TokenGraph] Error saving to database: {e}")
    
    def add_propagation(self, token_id: int, fact_id: str, timestamp: Optional[float] = None):
        """
        Add a token→fact propagation link.
        
        Args:
            token_id: The token ID
            fact_id: The fact ID
            timestamp: When this propagation occurred
        """
        if timestamp is None:
            timestamp = time.time()
        
        # Create or update token node
        if token_id not in self.graph:
            self.graph[token_id] = TokenNode(
                token_id=token_id,
                first_seen=timestamp,
                last_seen=timestamp,
                usage_count=1
            )
        else:
            node = self.graph[token_id]
            node.fact_ids.add(fact_id)
            node.last_seen = timestamp
            node.usage_count += 1
        
        # Update fact→tokens mapping
        self.fact_to_tokens[fact_id].add(token_id)
        
        # Update influence score
        self._update_influence_score(token_id)
        
        # Check for drift
        self._check_token_drift(token_id)
        
        # Periodically save to database
        if len(self.graph) % 100 == 0:
            self._save_to_database()
    
    def get_token_influencers(self, token_id: int) -> Set[str]:
        """Get all facts influenced by a token."""
        if token_id in self.graph:
            return self.graph[token_id].fact_ids.copy()
        return set()
    
    def get_facts_by_token_overlap(self, token_ids: List[int]) -> Dict[int, Set[str]]:
        """Get facts for each token in the list."""
        return {tid: self.get_token_influencers(tid) for tid in token_ids}
    
    def get_common_facts(self, token_ids: List[int]) -> Set[str]:
        """Get facts that contain all specified tokens."""
        if not token_ids:
            return set()
        
        fact_sets = [self.get_token_influencers(tid) for tid in token_ids]
        return set.intersection(*fact_sets) if fact_sets else set()
    
    def get_token_cluster(self, token_id: int) -> Optional[TokenCluster]:
        """Get the semantic cluster containing a token."""
        if token_id in self.graph:
            cluster_id = self.graph[token_id].semantic_cluster
            if cluster_id and cluster_id in self.clusters:
                return self.clusters[cluster_id]
        return None
    
    def get_top_influencers(self, limit: int = 10) -> List[Tuple[int, float]]:
        """Get tokens with highest influence scores."""
        influencers = [(tid, node.influence_score) for tid, node in self.graph.items()]
        return sorted(influencers, key=lambda x: x[1], reverse=True)[:limit]
    
    def get_token_drift_events(self, token_id: Optional[int] = None, limit: int = 20) -> List[TokenDriftEvent]:
        """Get recent drift events, optionally filtered by token."""
        events = self.drift_events
        if token_id is not None:
            events = [e for e in events if e.token_id == token_id]
        return events[-limit:]
    
    def build_influence_heatmap(self, fact_ids: Optional[List[str]] = None) -> Dict[int, float]:
        """
        Build influence heatmap showing which tokens drive the most beliefs.
        
        Args:
            fact_ids: Optional list of fact IDs to focus on
            
        Returns:
            Dictionary mapping token_id to influence score
        """
        heatmap = {}
        
        if fact_ids:
            # Focus on specific facts
            relevant_tokens = set()
            for fact_id in fact_ids:
                relevant_tokens.update(self.fact_to_tokens.get(fact_id, set()))
            
            for token_id in relevant_tokens:
                if token_id in self.graph:
                    heatmap[token_id] = self.graph[token_id].influence_score
        else:
            # Global heatmap
            for token_id, node in self.graph.items():
                heatmap[token_id] = node.influence_score
        
        return heatmap
    
    def detect_causal_clusters(self, min_cluster_size: int = 3) -> List[TokenCluster]:
        """
        Detect clusters of tokens that frequently co-occur in facts.
        
        Args:
            min_cluster_size: Minimum number of tokens in a cluster
            
        Returns:
            List of detected token clusters
        """
        # Build co-occurrence matrix
        co_occurrence = defaultdict(int)
        
        for fact_id, token_set in self.fact_to_tokens.items():
            tokens = list(token_set)
            for i in range(len(tokens)):
                for j in range(i + 1, len(tokens)):
                    pair = tuple(sorted([tokens[i], tokens[j]]))
                    co_occurrence[pair] += 1
        
        # Find clusters using co-occurrence
        clusters = []
        used_tokens = set()
        
        # Sort pairs by co-occurrence frequency
        sorted_pairs = sorted(co_occurrence.items(), key=lambda x: x[1], reverse=True)
        
        for (token1, token2), frequency in sorted_pairs:
            if frequency < 2:  # Minimum co-occurrence threshold
                break
            
            if token1 in used_tokens or token2 in used_tokens:
                continue
            
            # Start a new cluster
            cluster_tokens = {token1, token2}
            cluster_facts = self.get_common_facts([token1, token2])
            
            # Expand cluster
            for other_token in self.graph.keys():
                if other_token in used_tokens:
                    continue
                
                # Check if other_token co-occurs with cluster tokens
                cluster_token_list = list(cluster_tokens)
                common_facts = self.get_common_facts(cluster_token_list + [other_token])
                
                if len(common_facts) >= 2:  # Minimum fact overlap
                    cluster_tokens.add(other_token)
                    cluster_facts.update(common_facts)
            
            if len(cluster_tokens) >= min_cluster_size:
                cluster_id = f"cluster_{len(clusters)}"
                cluster = TokenCluster(
                    cluster_id=cluster_id,
                    token_ids=cluster_tokens,
                    fact_ids=cluster_facts,
                    created_time=time.time(),
                    last_updated=time.time()
                )
                
                # Calculate coherence score
                cluster.coherence_score = self._calculate_cluster_coherence(cluster)
                
                clusters.append(cluster)
                self.clusters[cluster_id] = cluster
                
                # Mark tokens as used
                used_tokens.update(cluster_tokens)
                
                # Update semantic cluster for tokens
                for token_id in cluster_tokens:
                    if token_id in self.graph:
                        self.graph[token_id].semantic_cluster = cluster_id
        
        return clusters
    
    def _update_influence_score(self, token_id: int):
        """Update the influence score for a token."""
        if token_id not in self.graph:
            return
        
        node = self.graph[token_id]
        
        # Calculate influence based on:
        # 1. Number of facts influenced
        # 2. Recency of usage
        # 3. Entropy of usage patterns
        # 4. Cluster coherence
        
        fact_count = len(node.fact_ids)
        recency_factor = 1.0 / (1.0 + (time.time() - node.last_seen) / 86400)  # Decay over days
        
        # Calculate entropy of usage patterns
        if fact_count > 1:
            node.entropy_score = compute_entropy(list(node.fact_ids))
        else:
            node.entropy_score = 0.0
        
        # Cluster coherence factor
        cluster_coherence = 0.0
        if node.semantic_cluster and node.semantic_cluster in self.clusters:
            cluster_coherence = self.clusters[node.semantic_cluster].coherence_score
        
        # Combined influence score
        node.influence_score = (
            fact_count * 0.4 +
            recency_factor * 0.3 +
            node.entropy_score * 0.2 +
            cluster_coherence * 0.1
        )
    
    def _calculate_cluster_coherence(self, cluster: TokenCluster) -> float:
        """Calculate coherence score for a token cluster."""
        if len(cluster.token_ids) < 2:
            return 1.0
        
        # Calculate average co-occurrence within cluster
        total_co_occurrence = 0
        pair_count = 0
        
        token_list = list(cluster.token_ids)
        for i in range(len(token_list)):
            for j in range(i + 1, len(token_list)):
                token1, token2 = token_list[i], token_list[j]
                common_facts = self.get_common_facts([token1, token2])
                total_co_occurrence += len(common_facts)
                pair_count += 1
        
        if pair_count == 0:
            return 0.0
        
        avg_co_occurrence = total_co_occurrence / pair_count
        return min(1.0, avg_co_occurrence / 5.0)  # Normalize to 0-1
    
    def _check_token_drift(self, token_id: int):
        """Check for token drift based on usage patterns."""
        if token_id not in self.graph:
            return
        
        node = self.graph[token_id]
        
        # Get current semantic cluster
        current_cluster = node.semantic_cluster
        
        # Detect clusters
        clusters = self.detect_causal_clusters()
        
        # Find which cluster this token belongs to now
        new_cluster = None
        for cluster in clusters:
            if token_id in cluster.token_ids:
                new_cluster = cluster.cluster_id
                break
        
        # Check for drift
        if current_cluster and new_cluster and current_cluster != new_cluster:
            # Calculate drift score
            old_cluster_facts = self.clusters[current_cluster].fact_ids if current_cluster in self.clusters else set()
            new_cluster_facts = self.clusters[new_cluster].fact_ids if new_cluster in self.clusters else set()
            
            drift_score = 1.0 - len(old_cluster_facts.intersection(new_cluster_facts)) / max(len(old_cluster_facts.union(new_cluster_facts)), 1)
            
            if drift_score > self.drift_threshold:
                # Record drift event
                affected_facts = list(old_cluster_facts.union(new_cluster_facts))
                
                drift_event = TokenDriftEvent(
                    token_id=token_id,
                    old_semantic_cluster=current_cluster,
                    new_semantic_cluster=new_cluster,
                    drift_score=drift_score,
                    timestamp=time.time(),
                    affected_facts=affected_facts,
                    drift_type="semantic_shift"
                )
                
                self.drift_events.append(drift_event)
                node.drift_score = drift_score
                
                print(f"[TokenGraph] Token drift detected: {token_id} moved from {current_cluster} to {new_cluster} (score: {drift_score:.3f})")
    
    def get_graph_summary(self) -> Dict[str, Any]:
        """Get a summary of the token graph."""
        total_tokens = len(self.graph)
        total_facts = len(self.fact_to_tokens)
        total_clusters = len(self.clusters)
        total_drift_events = len(self.drift_events)
        
        # Top influencers
        top_influencers = self.get_top_influencers(5)
        
        # Recent drift events
        recent_drift = self.get_token_drift_events(limit=5)
        
        return {
            "total_tokens": total_tokens,
            "total_facts": total_facts,
            "total_clusters": total_clusters,
            "total_drift_events": total_drift_events,
            "top_influencers": top_influencers,
            "recent_drift_events": [
                {
                    "token_id": event.token_id,
                    "old_cluster": event.old_semantic_cluster,
                    "new_cluster": event.new_semantic_cluster,
                    "drift_score": event.drift_score,
                    "drift_type": event.drift_type
                }
                for event in recent_drift
            ]
        }
    
    def shutdown(self):
        """Save data and cleanup."""
        self._save_to_database()
        print(f"[TokenGraph] Shutdown complete - saved {len(self.graph)} tokens")


# Global token graph instance
_token_graph_instance = None


def get_token_graph() -> TokenPropagationGraph:
    """Get or create the global token graph instance."""
    global _token_graph_instance
    
    if _token_graph_instance is None:
        _token_graph_instance = TokenPropagationGraph()
    
    return _token_graph_instance


def add_token_propagation(token_id: int, fact_id: str, timestamp: Optional[float] = None):
    """Convenience function to add token propagation."""
    get_token_graph().add_propagation(token_id, fact_id, timestamp)


def get_token_influencers(token_id: int) -> Set[str]:
    """Convenience function to get token influencers."""
    return get_token_graph().get_token_influencers(token_id) 