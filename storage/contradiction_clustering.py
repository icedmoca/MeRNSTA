#!/usr/bin/env python3
"""
🪢 Contradiction Clustering System for MeRNSTA

Groups volatile contradictions into semantic concept clusters and monitors which clusters
are unstable over time. Uses dynamic semantic similarity to identify related contradictions
and track belief drift patterns.
"""

import logging
import time
from typing import List, Dict, Set, Tuple, Optional
from collections import defaultdict
from dataclasses import dataclass
from scipy.spatial.distance import cosine
import numpy as np

try:
    import spacy
    from config.settings import get_config
    config = get_config()
    spacy_model = config.get('spacy', {}).get('model', 'en_core_web_sm')
    nlp = spacy.load(spacy_model)
except (ImportError, OSError):
    nlp = None
    logging.warning("spaCy not available for contradiction clustering")

from .enhanced_triplet_extractor import EnhancedTripletFact


@dataclass
class ContradictionCluster:
    """Represents a cluster of semantically related contradictions."""
    cluster_id: str
    concept_theme: str  # e.g., "food_preferences", "technology_choices"
    contradictory_facts: List[EnhancedTripletFact]
    volatility_score: float
    instability_events: List[Dict]  # Track when cluster became unstable
    last_updated: float
    semantic_center: Optional[List[float]] = None  # Centroid embedding


class ContradictionClusteringSystem:
    """
    Dynamically clusters contradictions into semantic concept groups.
    Monitors cluster instability over time for cognitive insights.
    """
    
    def __init__(self):
        self.clusters: Dict[str, ContradictionCluster] = {}
        self.cluster_counter = 0
        self.similarity_threshold = 0.4  # Dynamic threshold for clustering
        self.instability_threshold = 0.7  # When to flag a cluster as unstable
        
    def analyze_contradictions(self, facts: List[EnhancedTripletFact]) -> Dict[str, ContradictionCluster]:
        """
        Analyze all facts to identify and cluster contradictions dynamically.
        """
        print(f"[ContradictionClustering] Analyzing {len(facts)} facts for clustering")
        
        # Find contradictory facts
        contradictory_facts = [f for f in facts if getattr(f, 'contradiction', False)]
        
        if not contradictory_facts:
            print("[ContradictionClustering] No contradictory facts found")
            return self.clusters
            
        print(f"[ContradictionClustering] Found {len(contradictory_facts)} contradictory facts")
        
        # Group facts by semantic similarity
        unassigned_facts = contradictory_facts.copy()
        
        for fact in contradictory_facts:
            if not unassigned_facts or fact not in unassigned_facts:
                continue
                
            # Try to find an existing cluster for this fact
            assigned_cluster = self._find_matching_cluster(fact)
            
            if assigned_cluster:
                self._add_fact_to_cluster(assigned_cluster, fact)
                unassigned_facts.remove(fact)
            else:
                # Create new cluster
                cluster = self._create_new_cluster(fact)
                unassigned_facts.remove(fact)
                
                # Find other facts that belong to this cluster
                for other_fact in unassigned_facts.copy():
                    if self._facts_are_semantically_similar(fact, other_fact):
                        self._add_fact_to_cluster(cluster.cluster_id, other_fact)
                        unassigned_facts.remove(other_fact)
        
        # Update cluster metrics
        self._update_cluster_metrics()
        
        print(f"[ContradictionClustering] Created/updated {len(self.clusters)} clusters")
        return self.clusters
    
    def _find_matching_cluster(self, fact: EnhancedTripletFact) -> Optional[str]:
        """Find existing cluster that semantically matches this fact."""
        if not self.clusters:
            return None
            
        best_cluster = None
        best_similarity = 0.0
        
        for cluster_id, cluster in self.clusters.items():
            similarity = self._calculate_fact_cluster_similarity(fact, cluster)
            if similarity > self.similarity_threshold and similarity > best_similarity:
                best_similarity = similarity
                best_cluster = cluster_id
        
        return best_cluster
    
    def _create_new_cluster(self, seed_fact: EnhancedTripletFact) -> ContradictionCluster:
        """Create a new contradiction cluster around a seed fact."""
        self.cluster_counter += 1
        cluster_id = f"cluster_{self.cluster_counter}"
        
        # Determine concept theme dynamically
        concept_theme = self._extract_concept_theme(seed_fact)
        
        cluster = ContradictionCluster(
            cluster_id=cluster_id,
            concept_theme=concept_theme,
            contradictory_facts=[seed_fact],
            volatility_score=0.5,  # Initial volatility
            instability_events=[],
            last_updated=time.time()
        )
        
        self.clusters[cluster_id] = cluster
        print(f"[ContradictionClustering] Created cluster '{cluster_id}' for theme '{concept_theme}'")
        
        return cluster
    
    def _extract_concept_theme(self, fact: EnhancedTripletFact) -> str:
        """Dynamically extract the conceptual theme from a fact."""
        # Use semantic analysis to determine theme
        predicate = fact.predicate.lower()
        obj = fact.object.lower()
        
        # Dynamic theme detection based on semantic content
        if any(word in predicate for word in ['prefer', 'like', 'love', 'enjoy']):
            if any(food_word in obj for food_word in ['food', 'eat', 'taste', 'cuisine', 'pizza', 'pasta']):
                return "food_preferences"
            elif any(drink_word in obj for drink_word in ['coffee', 'tea', 'water', 'beverage', 'drink']):
                return "beverage_preferences"
            elif any(tech_word in obj for tech_word in ['python', 'javascript', 'code', 'programming', 'tech']):
                return "technology_preferences"
            elif any(activity_word in obj for activity_word in ['exercise', 'sport', 'hobby', 'activity']):
                return "activity_preferences"
            else:
                return "general_preferences"
        elif any(word in predicate for word in ['believe', 'think', 'feel']):
            return "beliefs_and_opinions"
        elif any(word in predicate for word in ['exercise', 'workout', 'practice']):
            return "physical_activities"
        else:
            # Fallback: use object as theme
            return f"{obj.replace(' ', '_')}_related"
    
    def _facts_are_semantically_similar(self, fact1: EnhancedTripletFact, fact2: EnhancedTripletFact) -> bool:
        """Check if two facts are semantically similar enough to cluster together."""
        # Same subject requirement
        if fact1.subject.lower() != fact2.subject.lower():
            return False
        
        # Semantic similarity check
        if nlp:
            try:
                # Combine predicate and object for semantic comparison
                text1 = f"{fact1.predicate} {fact1.object}".lower()
                text2 = f"{fact2.predicate} {fact2.object}".lower()
                
                doc1 = nlp(text1)
                doc2 = nlp(text2)
                
                similarity = doc1.similarity(doc2)
                
                # Dynamic threshold based on context
                threshold = self.similarity_threshold
                if fact1.predicate == fact2.predicate:
                    threshold *= 0.8  # Lower threshold for same predicate
                
                return similarity >= threshold
            except Exception as e:
                print(f"[ContradictionClustering] spaCy similarity error: {e}")
        
        # Fallback: predicate matching
        return fact1.predicate.lower() == fact2.predicate.lower()
    
    def _calculate_fact_cluster_similarity(self, fact: EnhancedTripletFact, cluster: ContradictionCluster) -> float:
        """Calculate semantic similarity between a fact and an entire cluster."""
        if not cluster.contradictory_facts:
            return 0.0
        
        similarities = []
        for cluster_fact in cluster.contradictory_facts:
            if nlp:
                try:
                    text1 = f"{fact.predicate} {fact.object}".lower()
                    text2 = f"{cluster_fact.predicate} {cluster_fact.object}".lower()
                    
                    doc1 = nlp(text1)
                    doc2 = nlp(text2)
                    
                    similarity = doc1.similarity(doc2)
                    similarities.append(similarity)
                except Exception:
                    # Fallback similarity
                    if fact.predicate.lower() == cluster_fact.predicate.lower():
                        similarities.append(0.6)
                    else:
                        similarities.append(0.1)
            else:
                # Simple text-based similarity
                if fact.predicate.lower() == cluster_fact.predicate.lower():
                    similarities.append(0.6)
                else:
                    similarities.append(0.1)
        
        # Return average similarity
        return sum(similarities) / len(similarities) if similarities else 0.0
    
    def _add_fact_to_cluster(self, cluster_id: str, fact: EnhancedTripletFact):
        """Add a fact to an existing cluster and update metrics."""
        if cluster_id not in self.clusters:
            return
            
        cluster = self.clusters[cluster_id]
        
        # Avoid duplicates
        if fact not in cluster.contradictory_facts:
            cluster.contradictory_facts.append(fact)
            cluster.last_updated = time.time()
            
            print(f"[ContradictionClustering] Added fact to cluster '{cluster_id}': {fact.predicate} {fact.object}")
    
    def _update_cluster_metrics(self):
        """Update volatility scores and detect instability events."""
        for cluster_id, cluster in self.clusters.items():
            # Calculate dynamic volatility based on cluster characteristics
            fact_count = len(cluster.contradictory_facts)
            
            # More facts = higher base volatility
            base_volatility = min(1.0, fact_count / 5.0)
            
            # Check for recent contradictions (time-based volatility)
            recent_facts = [f for f in cluster.contradictory_facts 
                          if hasattr(f, 'timestamp') and f.timestamp and 
                          time.time() - f.timestamp < 86400]  # Last 24 hours
            
            temporal_volatility = len(recent_facts) / max(1, fact_count) * 0.5
            
            # Predicate diversity adds volatility
            predicates = set(f.predicate for f in cluster.contradictory_facts)
            predicate_diversity = len(predicates) / max(1, fact_count) * 0.3
            
            # Final volatility score
            cluster.volatility_score = min(1.0, base_volatility + temporal_volatility + predicate_diversity)
            
            # Check for instability
            if cluster.volatility_score >= self.instability_threshold:
                instability_event = {
                    'timestamp': time.time(),
                    'volatility_score': cluster.volatility_score,
                    'fact_count': fact_count,
                    'trigger': 'high_volatility'
                }
                cluster.instability_events.append(instability_event)
                
                print(f"[ContradictionClustering] Cluster '{cluster_id}' flagged as UNSTABLE (volatility: {cluster.volatility_score:.2f})")
    
    def get_unstable_clusters(self) -> List[ContradictionCluster]:
        """Get clusters that are currently unstable."""
        return [cluster for cluster in self.clusters.values() 
                if cluster.volatility_score >= self.instability_threshold]
    
    def get_cluster_summary(self) -> Dict:
        """Get a summary of all clusters for analysis."""
        summary = {
            'total_clusters': len(self.clusters),
            'unstable_clusters': len(self.get_unstable_clusters()),
            'clusters': {}
        }
        
        for cluster_id, cluster in self.clusters.items():
            summary['clusters'][cluster_id] = {
                'concept_theme': cluster.concept_theme,
                'fact_count': len(cluster.contradictory_facts),
                'volatility_score': cluster.volatility_score,
                'is_unstable': cluster.volatility_score >= self.instability_threshold,
                'instability_events': len(cluster.instability_events),
                'last_updated': cluster.last_updated
            }
        
        return summary
    
    def suggest_cluster_insights(self) -> List[str]:
        """Generate insights about contradiction patterns."""
        insights = []
        
        if not self.clusters:
            return ["No contradiction clusters found. Beliefs appear stable."]
        
        unstable_clusters = self.get_unstable_clusters()
        
        if unstable_clusters:
            for cluster in unstable_clusters:
                insights.append(f"🔥 Unstable belief cluster detected: {cluster.concept_theme}")
                insights.append(f"   → {len(cluster.contradictory_facts)} conflicting beliefs")
                insights.append(f"   → Volatility score: {cluster.volatility_score:.2f}")
        
        # Identify trending themes
        theme_counts = defaultdict(int)
        for cluster in self.clusters.values():
            theme_counts[cluster.concept_theme] += len(cluster.contradictory_facts)
        
        if theme_counts:
            top_theme = max(theme_counts.items(), key=lambda x: x[1])
            insights.append(f"📊 Most contradictory theme: {top_theme[0]} ({top_theme[1]} conflicts)")
        
        return insights if insights else ["All belief clusters appear stable."] 