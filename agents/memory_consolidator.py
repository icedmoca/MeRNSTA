#!/usr/bin/env python3
"""
MemoryConsolidator - Memory consolidation and belief abstraction

Periodically analyzes stable clusters and consolidates them into
abstract belief facts using LLM summarization.

Features:
- Cluster stability analysis
- LLM-based belief summarization
- Abstract belief fact generation
- Periodic consolidation scheduling
- Belief abstraction tracking
"""

import time
import logging
import threading
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime

from storage.enhanced_memory_system import EnhancedMemorySystem
from storage.enhanced_memory_model import EnhancedTripletFact
from storage.token_graph import get_token_graph
from config.settings import get_config


@dataclass
class BeliefFact:
    """Represents an abstract belief fact created through consolidation."""
    belief_id: str
    cluster_id: str
    abstract_belief: str
    source_facts: List[str]
    confidence: float
    created_time: float
    stability_score: float
    consolidation_notes: str = ""


class MemoryConsolidator:
    """
    Consolidates stable clusters into abstract belief facts.
    
    Features:
    - Cluster stability analysis
    - LLM-based belief summarization
    - Abstract belief fact generation
    - Periodic consolidation scheduling
    - Belief abstraction tracking
    """
    
    def __init__(self, memory_system: EnhancedMemorySystem = None):
        self.memory_system = memory_system or EnhancedMemorySystem()
        self.token_graph = get_token_graph()
        
        # Configuration
        config = get_config()
        self.consolidation_interval = config.get('memory_consolidation_interval', 3600)  # 1 hour
        self.stability_threshold = config.get('cluster_stability_threshold', 0.8)
        self.min_facts_threshold = config.get('min_facts_for_consolidation', 3)
        self.max_volatility_threshold = config.get('max_volatility_for_consolidation', 0.2)
        
        # Consolidation state
        self.running = False
        self.consolidation_thread = None
        self.last_consolidation_time = time.time()
        self.consolidated_clusters: List[str] = []
        self.belief_facts: Dict[str, BeliefFact] = {}
        
        # Performance tracking
        self.consolidation_count = 0
        self.belief_facts_created = 0
        
        print(f"[MemoryConsolidator] Initialized with interval={self.consolidation_interval}s, "
              f"stability_threshold={self.stability_threshold}")
    
    def start_consolidation(self):
        """Start the background consolidation daemon."""
        if self.running:
            return
        
        self.running = True
        self.consolidation_thread = threading.Thread(target=self._consolidation_loop, daemon=True)
        self.consolidation_thread.start()
        print(f"[MemoryConsolidator] Started background consolidation daemon")
    
    def stop_consolidation(self):
        """Stop the background consolidation daemon."""
        self.running = False
        if self.consolidation_thread:
            self.consolidation_thread.join(timeout=5)
        print(f"[MemoryConsolidator] Stopped background consolidation daemon")
    
    def _consolidation_loop(self):
        """Background consolidation loop."""
        while self.running:
            try:
                self._consolidate_stable_clusters()
                time.sleep(self.consolidation_interval)
            except Exception as e:
                logging.error(f"[MemoryConsolidator] Error in consolidation loop: {e}")
                time.sleep(300)  # Wait 5 minutes before retrying
    
    def _consolidate_stable_clusters(self):
        """Consolidate stable clusters into abstract beliefs."""
        try:
            print(f"[MemoryConsolidator] Checking for stable clusters to consolidate...")
            
            # Get all clusters
            clusters_to_check = self._get_clusters_to_check()
            
            consolidated_count = 0
            
            for cluster_id in clusters_to_check:
                if self._is_cluster_stable(cluster_id):
                    if self._consolidate_cluster(cluster_id):
                        consolidated_count += 1
            
            self.last_consolidation_time = time.time()
            print(f"[MemoryConsolidator] Consolidated {consolidated_count} clusters")
            
        except Exception as e:
            logging.error(f"[MemoryConsolidator] Error consolidating clusters: {e}")
    
    def _get_clusters_to_check(self) -> List[str]:
        """Get list of clusters to check for consolidation."""
        try:
            clusters = []
            
            # Get clusters from token graph if available
            if self.token_graph and hasattr(self.token_graph, 'clusters'):
                clusters.extend(self.token_graph.clusters.keys())
            
            # Get clusters from memory system
            facts = self.memory_system.get_facts(limit=1000)
            for fact in facts:
                if hasattr(fact, 'cluster_id') and fact.cluster_id:
                    if fact.cluster_id not in clusters:
                        clusters.append(fact.cluster_id)
            
            return clusters[:50]  # Limit to 50 clusters for performance
            
        except Exception as e:
            logging.error(f"[MemoryConsolidator] Error getting clusters to check: {e}")
            return []
    
    def _is_cluster_stable(self, cluster_id: str) -> bool:
        """Check if a cluster is stable enough for consolidation."""
        try:
            # Get facts for this cluster
            facts = self.memory_system.get_facts(limit=100)
            cluster_facts = [f for f in facts if hasattr(f, 'cluster_id') and f.cluster_id == cluster_id]
            
            if len(cluster_facts) < self.min_facts_threshold:
                return False
            
            # Check for contradictions
            contradictions = sum(1 for f in cluster_facts if hasattr(f, 'contradiction') and f.contradiction)
            if contradictions > 0:
                return False
            
            # Check average confidence
            confidences = [f.confidence for f in cluster_facts if hasattr(f, 'confidence')]
            if confidences:
                avg_confidence = sum(confidences) / len(confidences)
                if avg_confidence < self.stability_threshold:
                    return False
            
            # Check volatility (if available)
            if self.token_graph:
                cluster = self.token_graph.clusters.get(cluster_id)
                if cluster and hasattr(cluster, 'volatility_score'):
                    if cluster.volatility_score > self.max_volatility_threshold:
                        return False
            
            return True
            
        except Exception as e:
            logging.error(f"[MemoryConsolidator] Error checking cluster stability: {e}")
            return False
    
    def _consolidate_cluster(self, cluster_id: str) -> bool:
        """Consolidate a stable cluster into an abstract belief."""
        try:
            # Get facts for this cluster
            facts = self.memory_system.get_facts(limit=100)
            cluster_facts = [f for f in facts if hasattr(f, 'cluster_id') and f.cluster_id == cluster_id]
            
            if not cluster_facts:
                return False
            
            # Skip if already consolidated
            if cluster_id in self.consolidated_clusters:
                return False
            
            # Generate abstract belief using LLM
            abstract_belief = self._generate_abstract_belief(cluster_facts)
            
            if not abstract_belief:
                return False
            
            # Create belief fact
            belief_id = f"belief_{cluster_id}_{int(time.time())}"
            belief_fact = BeliefFact(
                belief_id=belief_id,
                cluster_id=cluster_id,
                abstract_belief=abstract_belief,
                source_facts=[f.fact_id for f in cluster_facts if hasattr(f, 'fact_id')],
                confidence=0.9,  # High confidence for consolidated beliefs
                created_time=time.time(),
                stability_score=self._calculate_stability_score(cluster_facts),
                consolidation_notes=f"Consolidated from {len(cluster_facts)} stable facts"
            )
            
            # Store belief fact in memory
            self._store_belief_fact(belief_fact)
            
            # Mark cluster as consolidated
            self.consolidated_clusters.append(cluster_id)
            self.belief_facts[belief_id] = belief_fact
            
            self.consolidation_count += 1
            self.belief_facts_created += 1
            
            print(f"[MemoryConsolidator] Consolidated cluster {cluster_id} into belief: {abstract_belief[:100]}...")
            
            return True
            
        except Exception as e:
            logging.error(f"[MemoryConsolidator] Error consolidating cluster {cluster_id}: {e}")
            return False
    
    def _generate_abstract_belief(self, facts: List[EnhancedTripletFact]) -> Optional[str]:
        """Generate an abstract belief from a list of facts using LLM."""
        try:
            # Create a summary of the facts
            fact_summaries = []
            for fact in facts[:10]:  # Limit to 10 facts for performance
                summary = f"{fact.subject} {fact.predicate} {fact.object}"
                fact_summaries.append(summary)
            
            if not fact_summaries:
                return None
            
            # Create prompt for LLM
            prompt = f"""Based on the following facts about a cluster, generate a concise abstract belief that captures the core concept:

Facts:
{chr(10).join(fact_summaries)}

Abstract belief:"""
            
            # Use LLM to generate abstract belief
            # For now, we'll create a simple summary
            # In a full implementation, this would call an LLM API
            subjects = [f.subject for f in facts if hasattr(f, 'subject')]
            predicates = [f.predicate for f in facts if hasattr(f, 'predicate')]
            
            if subjects and predicates:
                # Create a simple abstract belief
                unique_subjects = list(set(subjects))
                unique_predicates = list(set(predicates))
                
                if len(unique_subjects) == 1:
                    abstract_belief = f"Belief about {unique_subjects[0]}: {', '.join(unique_predicates[:3])}"
                else:
                    abstract_belief = f"Belief about {len(unique_subjects)} entities: {', '.join(unique_predicates[:3])}"
                
                return abstract_belief
            
            return None
            
        except Exception as e:
            logging.error(f"[MemoryConsolidator] Error generating abstract belief: {e}")
            return None
    
    def _calculate_stability_score(self, facts: List[EnhancedTripletFact]) -> float:
        """Calculate stability score for a cluster of facts."""
        try:
            if not facts:
                return 0.0
            
            # Calculate average confidence
            confidences = [f.confidence for f in facts if hasattr(f, 'confidence')]
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
            
            # Calculate contradiction rate
            contradictions = sum(1 for f in facts if hasattr(f, 'contradiction') and f.contradiction)
            contradiction_rate = contradictions / len(facts)
            
            # Calculate stability score
            stability_score = avg_confidence * (1 - contradiction_rate)
            
            return stability_score
            
        except Exception as e:
            logging.error(f"[MemoryConsolidator] Error calculating stability score: {e}")
            return 0.0
    
    def _store_belief_fact(self, belief_fact: BeliefFact):
        """Store a belief fact in the memory system."""
        try:
            # Create memory fact for the belief
            fact = EnhancedTripletFact(
                subject="system",
                predicate="consolidated_belief",
                object=belief_fact.abstract_belief,
                confidence=belief_fact.confidence,
                source="MemoryConsolidator",
                metadata={
                    "belief_id": belief_fact.belief_id,
                    "cluster_id": belief_fact.cluster_id,
                    "source_facts": belief_fact.source_facts,
                    "stability_score": belief_fact.stability_score,
                    "consolidation_notes": belief_fact.consolidation_notes,
                    "created_time": belief_fact.created_time
                }
            )
            
            self.memory_system.store_fact(fact)
            
        except Exception as e:
            logging.error(f"[MemoryConsolidator] Error storing belief fact: {e}")
    
    def get_consolidation_status(self) -> Dict[str, Any]:
        """Get consolidation status and statistics."""
        return {
            "running": self.running,
            "last_consolidation_time": self.last_consolidation_time,
            "consolidation_count": self.consolidation_count,
            "belief_facts_created": self.belief_facts_created,
            "consolidated_clusters_count": len(self.consolidated_clusters),
            "belief_facts_count": len(self.belief_facts),
            "consolidation_interval": self.consolidation_interval
        }
    
    def get_belief_facts(self) -> List[Dict[str, Any]]:
        """Get list of created belief facts."""
        beliefs = []
        for belief_id, belief_fact in self.belief_facts.items():
            beliefs.append({
                "belief_id": belief_id,
                "cluster_id": belief_fact.cluster_id,
                "abstract_belief": belief_fact.abstract_belief,
                "confidence": belief_fact.confidence,
                "stability_score": belief_fact.stability_score,
                "source_facts_count": len(belief_fact.source_facts),
                "created_time": belief_fact.created_time
            })
        return beliefs


# Global memory consolidator instance
_memory_consolidator_instance = None


def get_memory_consolidator(memory_system: EnhancedMemorySystem = None) -> MemoryConsolidator:
    """Get or create the global memory consolidator instance."""
    global _memory_consolidator_instance
    
    if _memory_consolidator_instance is None:
        _memory_consolidator_instance = MemoryConsolidator(memory_system)
    
    return _memory_consolidator_instance


def start_memory_consolidation(memory_system: EnhancedMemorySystem = None):
    """Start the memory consolidation daemon."""
    consolidator = get_memory_consolidator(memory_system)
    consolidator.start_consolidation()


def stop_memory_consolidation():
    """Stop the memory consolidation daemon."""
    global _memory_consolidator_instance
    if _memory_consolidator_instance:
        _memory_consolidator_instance.stop_consolidation()


def get_consolidation_status() -> Dict[str, Any]:
    """Get memory consolidation status."""
    global _memory_consolidator_instance
    if _memory_consolidator_instance:
        return _memory_consolidator_instance.get_consolidation_status()
    return {"running": False} 