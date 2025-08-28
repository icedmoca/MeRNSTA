#!/usr/bin/env python3
"""
🧠 Autonomous Memory Consolidation System for MeRNSTA Phase 22

Comprehensive memory management system that orchestrates:
- Automatic pruning of low-priority memories
- Clustering of similar facts
- Causal and temporal relevance scoring
- Consolidation of repeated or overlapping facts
- Permanent memory tagging and protection
- Timeline-aware reordering and indexing

Integrates all existing memory subsystems into a unified consolidation engine.
"""

import logging
import uuid
import time
import math
import json
from typing import Dict, List, Any, Optional, Tuple, Set
from datetime import datetime, timedelta
from collections import defaultdict, Counter
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
from scipy.spatial.distance import cosine
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics.pairwise import cosine_similarity

from config.settings import get_config
from storage.memory_log import MemoryLog
from storage.memory_utils import TripletFact, get_sentiment_score, get_volatility_score
from storage.contradiction_clustering import ContradictionClusteringSystem
from storage.belief_consolidation import BeliefConsolidationSystem
from storage.db_utils import get_conn
try:
    from storage.errors import safe_db_operation
except ImportError:
    # Fallback if safe_db_operation not available
    def safe_db_operation(func):
        return func


class ConsolidationPriority(Enum):
    """Priority levels for consolidation operations."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class MemoryProtectionLevel(Enum):
    """Protection levels for memory facts."""
    NONE = "none"
    SOFT = "soft"          # Can be pruned under extreme conditions
    PROTECTED = "protected" # Cannot be pruned unless corrupted
    PERMANENT = "permanent" # Never pruned, highest priority


@dataclass
class ConsolidationRule:
    """Rules for memory consolidation behavior."""
    rule_id: str
    name: str
    enabled: bool = True
    priority: ConsolidationPriority = ConsolidationPriority.MEDIUM
    conditions: Dict[str, Any] = field(default_factory=dict)
    actions: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MemoryCluster:
    """Cluster of semantically similar memory facts."""
    cluster_id: str
    cluster_type: str  # "semantic", "temporal", "causal", "duplicate"
    facts: List[TripletFact]
    centroid_embedding: Optional[np.ndarray] = None
    similarity_threshold: float = 0.8
    created_at: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)
    consolidation_score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConsolidationResult:
    """Result of a memory consolidation operation."""
    operation_id: str
    operation_type: str
    started_at: datetime
    completed_at: Optional[datetime] = None
    facts_processed: int = 0
    facts_pruned: int = 0
    facts_consolidated: int = 0
    clusters_created: int = 0
    clusters_merged: int = 0
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    statistics: Dict[str, Any] = field(default_factory=dict)
    success: bool = False


class MemoryConsolidator:
    """
    Autonomous Memory Consolidation System.
    
    Orchestrates comprehensive memory management including pruning, clustering,
    consolidation, and timeline organization of facts.
    """
    
    def __init__(self, memory_log: MemoryLog = None, config: Dict[str, Any] = None):
        """Initialize the memory consolidator."""
        self.memory_log = memory_log or MemoryLog()
        self.config = config or self._load_consolidation_config()
        
        # Initialize subsystems
        self.contradiction_clusterer = ContradictionClusteringSystem()
        self.belief_consolidator = BeliefConsolidationSystem()
        
        # Consolidation state
        self.active_clusters: Dict[str, MemoryCluster] = {}
        self.consolidation_rules: Dict[str, ConsolidationRule] = {}
        self.consolidation_history: List[ConsolidationResult] = []
        self.protected_fact_ids: Set[int] = set()
        
        # Configuration parameters
        self.similarity_threshold = self.config.get('similarity_threshold', 0.8)
        self.min_cluster_size = self.config.get('min_cluster_size', 3)
        self.max_cluster_size = self.config.get('max_cluster_size', 50)
        self.temporal_window_days = self.config.get('temporal_window_days', 30)
        self.causal_relevance_threshold = self.config.get('causal_relevance_threshold', 0.6)
        
        # Load default consolidation rules
        self._initialize_default_rules()
        
        logging.info(f"[MemoryConsolidator] Initialized with {len(self.consolidation_rules)} rules")
    
    def _load_consolidation_config(self) -> Dict[str, Any]:
        """Load consolidation configuration."""
        try:
            config = get_config()
            return config.get('memory_consolidation', {
                'enabled': True,
                'auto_consolidation_interval': 3600,  # 1 hour
                'similarity_threshold': 0.8,
                'min_cluster_size': 3,
                'max_cluster_size': 50,
                'temporal_window_days': 30,
                'causal_relevance_threshold': 0.6,
                'pruning': {
                    'enabled': True,
                    'confidence_threshold': 0.3,
                    'age_threshold_days': 90,
                    'protect_recent_days': 7
                },
                'clustering': {
                    'enabled': True,
                    'algorithm': 'dbscan',  # 'dbscan', 'kmeans', 'hierarchical'
                    'eps': 0.2,
                    'min_samples': 3
                },
                'consolidation': {
                    'enabled': True,
                    'merge_duplicates': True,
                    'strengthen_patterns': True,
                    'temporal_ordering': True
                }
            })
        except Exception as e:
            logging.warning(f"[MemoryConsolidator] Config loading failed: {e}")
            return {'enabled': True}
    
    def _initialize_default_rules(self):
        """Initialize default consolidation rules."""
        default_rules = [
            ConsolidationRule(
                rule_id="protect_recent",
                name="Protect Recent Memories",
                priority=ConsolidationPriority.HIGH,
                conditions={"max_age_days": 7},
                actions=["protect"],
                metadata={"description": "Protect memories from the last 7 days"}
            ),
            ConsolidationRule(
                rule_id="prune_low_confidence",
                name="Prune Low Confidence Facts",
                priority=ConsolidationPriority.MEDIUM,
                conditions={"max_confidence": 0.3, "min_age_days": 30},
                actions=["prune"],
                metadata={"description": "Remove facts with very low confidence after 30 days"}
            ),
            ConsolidationRule(
                rule_id="consolidate_duplicates",
                name="Consolidate Duplicate Facts",
                priority=ConsolidationPriority.HIGH,
                conditions={"similarity_threshold": 0.95},
                actions=["merge", "strengthen"],
                metadata={"description": "Merge nearly identical facts"}
            ),
            ConsolidationRule(
                rule_id="cluster_similar",
                name="Cluster Similar Facts",
                priority=ConsolidationPriority.MEDIUM,
                conditions={"similarity_threshold": 0.8, "min_cluster_size": 3},
                actions=["cluster"],
                metadata={"description": "Group semantically similar facts"}
            ),
            ConsolidationRule(
                rule_id="strengthen_patterns",
                name="Strengthen Consistent Patterns",
                priority=ConsolidationPriority.HIGH,
                conditions={"pattern_support": 3, "stability_threshold": 0.7},
                actions=["strengthen", "promote"],
                metadata={"description": "Reinforce facts that show consistent patterns"}
            )
        ]
        
        for rule in default_rules:
            self.consolidation_rules[rule.rule_id] = rule
    
    def consolidate_memory(self, full_consolidation: bool = False) -> ConsolidationResult:
        """
        Perform comprehensive memory consolidation.
        
        Args:
            full_consolidation: If True, perform deep consolidation including clustering
            
        Returns:
            ConsolidationResult with operation details
        """
        operation_id = str(uuid.uuid4())
        result = ConsolidationResult(
            operation_id=operation_id,
            operation_type="full" if full_consolidation else "incremental",
            started_at=datetime.now()
        )
        
        logging.info(f"[MemoryConsolidator] Starting {'full' if full_consolidation else 'incremental'} consolidation")
        
        try:
            # Step 1: Load all facts
            facts = self._load_facts_for_consolidation()
            result.facts_processed = len(facts)
            
            if not facts:
                result.warnings.append("No facts found for consolidation")
                result.success = True
                result.completed_at = datetime.now()
                return result
            
            # Step 2: Apply protection rules
            self._apply_protection_rules(facts)
            
            # Step 3: Prune low-priority memories
            if self.config.get('pruning', {}).get('enabled', True):
                pruned_count = self._prune_memories(facts)
                result.facts_pruned = pruned_count
            
            # Step 4: Cluster similar facts
            if full_consolidation and self.config.get('clustering', {}).get('enabled', True):
                clusters = self._cluster_similar_facts(facts)
                result.clusters_created = len(clusters)
                
                # Step 5: Consolidate clusters
                if self.config.get('consolidation', {}).get('enabled', True):
                    consolidated_count = self._consolidate_clusters(clusters)
                    result.facts_consolidated = consolidated_count
            
            # Step 6: Apply temporal and causal ordering
            if self.config.get('consolidation', {}).get('temporal_ordering', True):
                self._apply_temporal_ordering(facts)
            
            # Step 7: Update consolidation statistics
            result.statistics = self._calculate_consolidation_statistics(facts)
            
            result.success = True
            result.completed_at = datetime.now()
            
            # Store result in history
            self.consolidation_history.append(result)
            
            logging.info(f"[MemoryConsolidator] Consolidation completed: "
                        f"{result.facts_processed} processed, "
                        f"{result.facts_pruned} pruned, "
                        f"{result.facts_consolidated} consolidated")
            
        except Exception as e:
            logging.error(f"[MemoryConsolidator] Consolidation failed: {e}")
            result.errors.append(str(e))
            result.success = False
            result.completed_at = datetime.now()
        
        return result
    
    def _load_facts_for_consolidation(self) -> List[TripletFact]:
        """Load facts from memory for consolidation processing."""
        try:
            # Get facts from memory log
            facts = self.memory_log.list_facts()
            
            # Enhance with embeddings and metadata
            enhanced_facts = []
            for fact in facts:
                # Add consolidation metadata
                if not hasattr(fact, 'consolidation_metadata'):
                    fact.consolidation_metadata = {
                        'last_accessed': datetime.now(),
                        'access_count': 0,
                        'protection_level': MemoryProtectionLevel.NONE,
                        'cluster_assignments': [],
                        'causal_connections': [],
                        'temporal_neighbors': []
                    }
                enhanced_facts.append(fact)
            
            return enhanced_facts
            
        except Exception as e:
            logging.error(f"[MemoryConsolidator] Failed to load facts: {e}")
            return []
    
    def _apply_protection_rules(self, facts: List[TripletFact]):
        """Apply protection rules to prevent important facts from being pruned."""
        for fact in facts:
            protection_level = MemoryProtectionLevel.NONE
            
            # Apply protection rules
            for rule in self.consolidation_rules.values():
                if not rule.enabled or "protect" not in rule.actions:
                    continue
                
                if self._fact_matches_rule_conditions(fact, rule):
                    if rule.priority == ConsolidationPriority.CRITICAL:
                        protection_level = MemoryProtectionLevel.PERMANENT
                    elif rule.priority == ConsolidationPriority.HIGH:
                        protection_level = MemoryProtectionLevel.PROTECTED
                    elif protection_level == MemoryProtectionLevel.NONE:
                        protection_level = MemoryProtectionLevel.SOFT
            
            # Update fact protection
            if hasattr(fact, 'consolidation_metadata'):
                fact.consolidation_metadata['protection_level'] = protection_level
                if protection_level != MemoryProtectionLevel.NONE:
                    self.protected_fact_ids.add(fact.id)
    
    def _fact_matches_rule_conditions(self, fact: TripletFact, rule: ConsolidationRule) -> bool:
        """Check if a fact matches the conditions of a consolidation rule."""
        try:
            conditions = rule.conditions
            
            # Check age conditions
            if 'max_age_days' in conditions:
                fact_age_days = (datetime.now() - datetime.fromisoformat(fact.timestamp)).days
                if fact_age_days > conditions['max_age_days']:
                    return False
            
            if 'min_age_days' in conditions:
                fact_age_days = (datetime.now() - datetime.fromisoformat(fact.timestamp)).days
                if fact_age_days < conditions['min_age_days']:
                    return False
            
            # Check confidence conditions
            if 'max_confidence' in conditions:
                if fact.confidence > conditions['max_confidence']:
                    return False
            
            if 'min_confidence' in conditions:
                if fact.confidence < conditions['min_confidence']:
                    return False
            
            # Check content conditions
            if 'contains_keywords' in conditions:
                keywords = conditions['contains_keywords']
                fact_text = f"{fact.subject} {fact.predicate} {fact.object}".lower()
                if not any(keyword.lower() in fact_text for keyword in keywords):
                    return False
            
            return True
            
        except Exception as e:
            logging.warning(f"[MemoryConsolidator] Error checking rule conditions: {e}")
            return False
    
    def _prune_memories(self, facts: List[TripletFact]) -> int:
        """Prune low-priority memories according to consolidation rules."""
        pruning_config = self.config.get('pruning', {})
        if not pruning_config.get('enabled', True):
            return 0
        
        confidence_threshold = pruning_config.get('confidence_threshold', 0.3)
        age_threshold_days = pruning_config.get('age_threshold_days', 90)
        protect_recent_days = pruning_config.get('protect_recent_days', 7)
        
        pruned_count = 0
        current_time = datetime.now()
        
        for fact in facts:
            # Skip protected facts
            if fact.id in self.protected_fact_ids:
                continue
            
            # Skip recent facts
            fact_age_days = (current_time - datetime.fromisoformat(fact.timestamp)).days
            if fact_age_days < protect_recent_days:
                continue
            
            # Check pruning conditions
            should_prune = False
            
            # Low confidence and old
            if (fact.confidence < confidence_threshold and 
                fact_age_days > age_threshold_days):
                should_prune = True
            
            # Very low confidence regardless of age
            elif fact.confidence < 0.1 and fact_age_days > 7:
                should_prune = True
            
            # Apply consolidation rule checks
            for rule in self.consolidation_rules.values():
                if (rule.enabled and "prune" in rule.actions and 
                    self._fact_matches_rule_conditions(fact, rule)):
                    should_prune = True
                    break
            
            if should_prune:
                try:
                    self.memory_log.delete_fact(fact.id)
                    pruned_count += 1
                    logging.debug(f"[MemoryConsolidator] Pruned fact {fact.id}: "
                                f"confidence={fact.confidence:.3f}, age={fact_age_days}d")
                except Exception as e:
                    logging.warning(f"[MemoryConsolidator] Failed to prune fact {fact.id}: {e}")
        
        return pruned_count
    
    def _cluster_similar_facts(self, facts: List[TripletFact]) -> List[MemoryCluster]:
        """Cluster semantically similar facts using configurable algorithms."""
        clustering_config = self.config.get('clustering', {})
        algorithm = clustering_config.get('algorithm', 'dbscan')
        
        if not facts:
            return []
        
        try:
            # Generate embeddings for facts
            embeddings = []
            valid_facts = []
            
            for fact in facts:
                try:
                    # Create fact text for embedding
                    fact_text = f"{fact.subject} {fact.predicate} {fact.object}"
                    
                    # Get embedding using memory log's vectorizer
                    embedding = self.memory_log.vectorizer(fact_text)
                    if isinstance(embedding, list):
                        embedding = np.array(embedding)
                    
                    embeddings.append(embedding)
                    valid_facts.append(fact)
                    
                except Exception as e:
                    logging.warning(f"[MemoryConsolidator] Failed to embed fact {fact.id}: {e}")
                    continue
            
            if len(embeddings) < self.min_cluster_size:
                return []
            
            embeddings_matrix = np.array(embeddings)
            
            # Apply clustering algorithm
            if algorithm == 'dbscan':
                eps = clustering_config.get('eps', 0.2)
                min_samples = clustering_config.get('min_samples', 3)
                clusterer = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine')
                
            elif algorithm == 'kmeans':
                n_clusters = min(clustering_config.get('n_clusters', 10), len(valid_facts) // 3)
                if n_clusters < 2:
                    return []
                clusterer = KMeans(n_clusters=n_clusters, random_state=42)
                
            else:
                logging.warning(f"[MemoryConsolidator] Unknown clustering algorithm: {algorithm}")
                return []
            
            # Perform clustering
            cluster_labels = clusterer.fit_predict(embeddings_matrix)
            
            # Group facts by cluster
            clusters_dict = defaultdict(list)
            for i, label in enumerate(cluster_labels):
                if label != -1:  # Skip noise points in DBSCAN
                    clusters_dict[label].append((valid_facts[i], embeddings[i]))
            
            # Create MemoryCluster objects
            clusters = []
            for cluster_id, fact_embedding_pairs in clusters_dict.items():
                if len(fact_embedding_pairs) >= self.min_cluster_size:
                    cluster_facts = [pair[0] for pair in fact_embedding_pairs]
                    cluster_embeddings = [pair[1] for pair in fact_embedding_pairs]
                    
                    # Calculate centroid
                    centroid = np.mean(cluster_embeddings, axis=0)
                    
                    # Create cluster
                    cluster = MemoryCluster(
                        cluster_id=f"semantic_{cluster_id}_{int(time.time())}",
                        cluster_type="semantic",
                        facts=cluster_facts,
                        centroid_embedding=centroid,
                        similarity_threshold=self.similarity_threshold,
                        consolidation_score=self._calculate_cluster_consolidation_score(cluster_facts)
                    )
                    
                    clusters.append(cluster)
                    self.active_clusters[cluster.cluster_id] = cluster
            
            logging.info(f"[MemoryConsolidator] Created {len(clusters)} semantic clusters")
            return clusters
            
        except Exception as e:
            logging.error(f"[MemoryConsolidator] Clustering failed: {e}")
            return []
    
    def _calculate_cluster_consolidation_score(self, facts: List[TripletFact]) -> float:
        """Calculate a consolidation score for a cluster of facts."""
        if not facts:
            return 0.0
        
        try:
            # Factors contributing to consolidation score
            scores = []
            
            # 1. Confidence average
            confidences = [fact.confidence for fact in facts if fact.confidence is not None]
            if confidences:
                avg_confidence = sum(confidences) / len(confidences)
                scores.append(avg_confidence)
            
            # 2. Temporal coherence (facts closer in time score higher)
            timestamps = []
            for fact in facts:
                try:
                    timestamp = datetime.fromisoformat(fact.timestamp)
                    timestamps.append(timestamp)
                except:
                    continue
            
            if len(timestamps) > 1:
                timestamps.sort()
                time_spans = [(timestamps[i+1] - timestamps[i]).total_seconds() 
                             for i in range(len(timestamps)-1)]
                avg_time_span = sum(time_spans) / len(time_spans)
                # Normalize: closer facts get higher scores
                temporal_score = max(0, 1 - (avg_time_span / (30 * 24 * 3600)))  # 30 days normalization
                scores.append(temporal_score)
            
            # 3. Semantic consistency (similar predicates get higher scores)
            predicates = [fact.predicate for fact in facts]
            predicate_consistency = len(set(predicates)) / len(predicates)
            semantic_score = 1 - predicate_consistency  # More consistent = higher score
            scores.append(semantic_score)
            
            # 4. Size factor (larger clusters within limits score higher)
            size_score = min(1.0, len(facts) / self.max_cluster_size)
            scores.append(size_score)
            
            # Calculate weighted average
            if scores:
                return sum(scores) / len(scores)
            else:
                return 0.5
                
        except Exception as e:
            logging.warning(f"[MemoryConsolidator] Error calculating cluster score: {e}")
            return 0.5
    
    def _consolidate_clusters(self, clusters: List[MemoryCluster]) -> int:
        """Consolidate facts within clusters by merging duplicates and strengthening patterns."""
        consolidated_count = 0
        
        for cluster in clusters:
            try:
                # Find duplicate/near-duplicate facts within cluster
                duplicates = self._find_duplicate_facts(cluster.facts)
                
                # Merge duplicates
                for duplicate_group in duplicates:
                    if len(duplicate_group) > 1:
                        merged_fact = self._merge_duplicate_facts(duplicate_group)
                        if merged_fact:
                            consolidated_count += len(duplicate_group) - 1
                
                # Strengthen consistent patterns
                patterns = self._identify_consistent_patterns(cluster.facts)
                for pattern in patterns:
                    self._strengthen_pattern(pattern)
                    consolidated_count += 1
                
            except Exception as e:
                logging.warning(f"[MemoryConsolidator] Error consolidating cluster {cluster.cluster_id}: {e}")
        
        return consolidated_count
    
    def _find_duplicate_facts(self, facts: List[TripletFact]) -> List[List[TripletFact]]:
        """Find groups of duplicate or near-duplicate facts."""
        duplicate_groups = []
        processed_ids = set()
        
        for i, fact1 in enumerate(facts):
            if fact1.id in processed_ids:
                continue
            
            duplicates = [fact1]
            processed_ids.add(fact1.id)
            
            for j, fact2 in enumerate(facts[i+1:], i+1):
                if fact2.id in processed_ids:
                    continue
                
                # Check for duplication
                if self._are_facts_duplicates(fact1, fact2):
                    duplicates.append(fact2)
                    processed_ids.add(fact2.id)
            
            if len(duplicates) > 1:
                duplicate_groups.append(duplicates)
        
        return duplicate_groups
    
    def _are_facts_duplicates(self, fact1: TripletFact, fact2: TripletFact) -> bool:
        """Check if two facts are duplicates or near-duplicates."""
        try:
            # Exact match
            if (fact1.subject.lower() == fact2.subject.lower() and
                fact1.predicate.lower() == fact2.predicate.lower() and
                fact1.object.lower() == fact2.object.lower()):
                return True
            
            # Semantic similarity check
            text1 = f"{fact1.subject} {fact1.predicate} {fact1.object}".lower()
            text2 = f"{fact2.subject} {fact2.predicate} {fact2.object}".lower()
            
            # Simple word overlap check
            words1 = set(text1.split())
            words2 = set(text2.split())
            
            if len(words1) == 0 or len(words2) == 0:
                return False
            
            overlap = len(words1.intersection(words2))
            union = len(words1.union(words2))
            
            jaccard_similarity = overlap / union if union > 0 else 0
            
            return jaccard_similarity > 0.8
            
        except Exception as e:
            logging.warning(f"[MemoryConsolidator] Error checking fact duplication: {e}")
            return False
    
    def _merge_duplicate_facts(self, duplicate_facts: List[TripletFact]) -> Optional[TripletFact]:
        """Merge a group of duplicate facts into a single consolidated fact."""
        if not duplicate_facts:
            return None
        
        try:
            # Choose the fact with highest confidence as base
            base_fact = max(duplicate_facts, key=lambda f: f.confidence or 0)
            
            # Calculate merged confidence (average with recency weight)
            confidences = [f.confidence or 0 for f in duplicate_facts]
            timestamps = []
            for f in duplicate_facts:
                try:
                    timestamps.append(datetime.fromisoformat(f.timestamp))
                except:
                    timestamps.append(datetime.now())
            
            # Weight by recency
            now = datetime.now()
            weights = []
            for ts in timestamps:
                days_old = (now - ts).days
                weight = max(0.1, 1.0 / (1 + days_old * 0.1))  # Decay weight
                weights.append(weight)
            
            total_weight = sum(weights)
            weighted_confidence = sum(c * w for c, w in zip(confidences, weights)) / total_weight
            
            # Update base fact
            base_fact.confidence = min(1.0, weighted_confidence)
            
            # Add consolidation metadata
            if hasattr(base_fact, 'consolidation_metadata'):
                base_fact.consolidation_metadata['merged_from'] = [f.id for f in duplicate_facts if f.id != base_fact.id]
                base_fact.consolidation_metadata['merge_count'] = len(duplicate_facts)
                base_fact.consolidation_metadata['last_consolidated'] = datetime.now()
            
            # Remove other duplicate facts
            for fact in duplicate_facts:
                if fact.id != base_fact.id:
                    try:
                        self.memory_log.delete_fact(fact.id)
                    except Exception as e:
                        logging.warning(f"[MemoryConsolidator] Failed to delete duplicate fact {fact.id}: {e}")
            
            # Update base fact in memory
            try:
                self.memory_log.update_fact_confidence(base_fact.id, base_fact.confidence)
                logging.debug(f"[MemoryConsolidator] Merged {len(duplicate_facts)} facts into fact {base_fact.id}")
                return base_fact
            except Exception as e:
                logging.warning(f"[MemoryConsolidator] Failed to update merged fact: {e}")
                return None
                
        except Exception as e:
            logging.error(f"[MemoryConsolidator] Error merging duplicate facts: {e}")
            return None
    
    def _identify_consistent_patterns(self, facts: List[TripletFact]) -> List[Dict[str, Any]]:
        """Identify consistent patterns within a group of facts."""
        patterns = []
        
        try:
            # Group facts by subject-predicate pairs
            subject_predicate_groups = defaultdict(list)
            for fact in facts:
                key = (fact.subject.lower(), fact.predicate.lower())
                subject_predicate_groups[key].append(fact)
            
            # Look for patterns with multiple supporting facts
            for (subject, predicate), group_facts in subject_predicate_groups.items():
                if len(group_facts) >= 2:
                    # Check if facts have consistent objects/sentiments
                    objects = [f.object for f in group_facts]
                    sentiments = [get_sentiment_score(f.predicate) for f in group_facts]
                    
                    # Calculate pattern strength
                    object_consistency = len(set(objects)) / len(objects) if objects else 0
                    sentiment_variance = np.var(sentiments) if sentiments else 1.0
                    
                    pattern_strength = (1 - object_consistency) * (1 - min(1.0, sentiment_variance))
                    
                    if pattern_strength > 0.6:  # Threshold for consistent pattern
                        patterns.append({
                            'type': 'subject_predicate_consistency',
                            'subject': subject,
                            'predicate': predicate,
                            'supporting_facts': group_facts,
                            'strength': pattern_strength,
                            'objects': objects,
                            'consistency_score': 1 - object_consistency
                        })
            
            return patterns
            
        except Exception as e:
            logging.warning(f"[MemoryConsolidator] Error identifying patterns: {e}")
            return []
    
    def _strengthen_pattern(self, pattern: Dict[str, Any]):
        """Strengthen a consistent pattern by boosting confidence of supporting facts."""
        try:
            supporting_facts = pattern.get('supporting_facts', [])
            strength_boost = pattern.get('strength', 0.0) * 0.2  # Max 20% boost
            
            for fact in supporting_facts:
                if fact.id not in self.protected_fact_ids:
                    # Boost confidence
                    new_confidence = min(1.0, fact.confidence + strength_boost)
                    
                    try:
                        self.memory_log.update_fact_confidence(fact.id, new_confidence)
                        logging.debug(f"[MemoryConsolidator] Strengthened fact {fact.id}: "
                                    f"{fact.confidence:.3f} -> {new_confidence:.3f}")
                    except Exception as e:
                        logging.warning(f"[MemoryConsolidator] Failed to strengthen fact {fact.id}: {e}")
                    
        except Exception as e:
            logging.warning(f"[MemoryConsolidator] Error strengthening pattern: {e}")
    
    def _apply_temporal_ordering(self, facts: List[TripletFact]):
        """Apply timeline-aware reordering and indexing to facts."""
        try:
            # Sort facts by timestamp
            facts_with_timestamps = []
            for fact in facts:
                try:
                    timestamp = datetime.fromisoformat(fact.timestamp)
                    facts_with_timestamps.append((fact, timestamp))
                except:
                    # Use current time for facts without valid timestamps
                    facts_with_timestamps.append((fact, datetime.now()))
            
            facts_with_timestamps.sort(key=lambda x: x[1])
            
            # Create temporal neighborhoods
            window_size = timedelta(days=self.temporal_window_days)
            
            for i, (fact, timestamp) in enumerate(facts_with_timestamps):
                # Find temporal neighbors (facts within time window)
                neighbors = []
                
                for j, (other_fact, other_timestamp) in enumerate(facts_with_timestamps):
                    if i != j and abs((timestamp - other_timestamp).total_seconds()) <= window_size.total_seconds():
                        neighbors.append(other_fact.id)
                
                # Update fact metadata
                if hasattr(fact, 'consolidation_metadata'):
                    fact.consolidation_metadata['temporal_neighbors'] = neighbors
                    fact.consolidation_metadata['temporal_position'] = i
                    fact.consolidation_metadata['temporal_window'] = self.temporal_window_days
            
            logging.info(f"[MemoryConsolidator] Applied temporal ordering to {len(facts)} facts")
            
        except Exception as e:
            logging.error(f"[MemoryConsolidator] Error applying temporal ordering: {e}")
    
    def _calculate_consolidation_statistics(self, facts: List[TripletFact]) -> Dict[str, Any]:
        """Calculate statistics about the consolidation process."""
        try:
            stats = {
                'total_facts': len(facts),
                'protected_facts': len(self.protected_fact_ids),
                'active_clusters': len(self.active_clusters),
                'avg_confidence': 0.0,
                'confidence_distribution': {},
                'temporal_span_days': 0,
                'consolidation_rules_applied': len([r for r in self.consolidation_rules.values() if r.enabled])
            }
            
            if facts:
                # Calculate confidence statistics
                confidences = [f.confidence for f in facts if f.confidence is not None]
                if confidences:
                    stats['avg_confidence'] = sum(confidences) / len(confidences)
                    stats['confidence_distribution'] = {
                        'high': len([c for c in confidences if c > 0.7]),
                        'medium': len([c for c in confidences if 0.3 <= c <= 0.7]),
                        'low': len([c for c in confidences if c < 0.3])
                    }
                
                # Calculate temporal span
                timestamps = []
                for fact in facts:
                    try:
                        timestamps.append(datetime.fromisoformat(fact.timestamp))
                    except:
                        continue
                
                if len(timestamps) > 1:
                    timestamps.sort()
                    span = (timestamps[-1] - timestamps[0]).days
                    stats['temporal_span_days'] = span
            
            return stats
            
        except Exception as e:
            logging.warning(f"[MemoryConsolidator] Error calculating statistics: {e}")
            return {}
    
    # Public API methods
    
    def prune(self, confidence_threshold: float = None, age_threshold_days: int = None) -> Dict[str, Any]:
        """
        Prune low-priority memories.
        
        Args:
            confidence_threshold: Minimum confidence to keep (uses config default if None)
            age_threshold_days: Minimum age before pruning (uses config default if None)
            
        Returns:
            Dictionary with pruning results
        """
        # Load current configuration
        pruning_config = self.config.get('pruning', {})
        confidence_threshold = confidence_threshold or pruning_config.get('confidence_threshold', 0.3)
        age_threshold_days = age_threshold_days or pruning_config.get('age_threshold_days', 90)
        
        # Use existing memory log prune_memory method
        return self.memory_log.prune_memory(confidence_threshold, age_threshold_days)
    
    def cluster(self, algorithm: str = None, **kwargs) -> List[MemoryCluster]:
        """
        Cluster similar facts.
        
        Args:
            algorithm: Clustering algorithm ('dbscan', 'kmeans', 'hierarchical')
            **kwargs: Algorithm-specific parameters
            
        Returns:
            List of created clusters
        """
        # Override algorithm if specified
        if algorithm:
            self.config.setdefault('clustering', {})['algorithm'] = algorithm
            self.config['clustering'].update(kwargs)
        
        facts = self._load_facts_for_consolidation()
        return self._cluster_similar_facts(facts)
    
    def prioritize(self, facts: List[TripletFact] = None) -> List[TripletFact]:
        """
        Prioritize memories based on various scoring factors.
        
        Args:
            facts: List of facts to prioritize (loads all if None)
            
        Returns:
            Sorted list of facts by priority
        """
        if facts is None:
            facts = self._load_facts_for_consolidation()
        
        # Calculate priority scores
        for fact in facts:
            priority_score = self._calculate_priority_score(fact)
            if hasattr(fact, 'consolidation_metadata'):
                fact.consolidation_metadata['priority_score'] = priority_score
        
        # Sort by priority (highest first)
        facts.sort(key=lambda f: getattr(f, 'consolidation_metadata', {}).get('priority_score', 0), reverse=True)
        
        return facts
    
    def _calculate_priority_score(self, fact: TripletFact) -> float:
        """Calculate priority score for a fact based on multiple factors."""
        try:
            score = 0.0
            
            # 1. Confidence factor (0-1)
            confidence_factor = fact.confidence or 0.0
            score += confidence_factor * 0.3
            
            # 2. Recency factor (newer = higher priority)
            try:
                fact_age_days = (datetime.now() - datetime.fromisoformat(fact.timestamp)).days
                recency_factor = max(0, 1 - (fact_age_days / 365))  # Decay over a year
                score += recency_factor * 0.2
            except:
                score += 0.1  # Default for facts without valid timestamps
            
            # 3. Access frequency factor
            if hasattr(fact, 'consolidation_metadata'):
                access_count = fact.consolidation_metadata.get('access_count', 0)
                access_factor = min(1.0, access_count / 10)  # Normalize to 0-1
                score += access_factor * 0.2
            
            # 4. Protection level factor
            if hasattr(fact, 'consolidation_metadata'):
                protection_level = fact.consolidation_metadata.get('protection_level', MemoryProtectionLevel.NONE)
                if protection_level == MemoryProtectionLevel.PERMANENT:
                    score += 0.3
                elif protection_level == MemoryProtectionLevel.PROTECTED:
                    score += 0.2
                elif protection_level == MemoryProtectionLevel.SOFT:
                    score += 0.1
            
            # 5. Semantic richness factor
            fact_text = f"{fact.subject} {fact.predicate} {fact.object}"
            word_count = len(fact_text.split())
            richness_factor = min(1.0, word_count / 20)  # Normalize to 0-1
            score += richness_factor * 0.1
            
            return min(1.0, score)
            
        except Exception as e:
            logging.warning(f"[MemoryConsolidator] Error calculating priority score: {e}")
            return 0.5
    
    def consolidate(self, cluster_id: str = None) -> Dict[str, Any]:
        """
        Consolidate memories in a specific cluster or all clusters.
        
        Args:
            cluster_id: Specific cluster to consolidate (consolidates all if None)
            
        Returns:
            Dictionary with consolidation results
        """
        if cluster_id and cluster_id in self.active_clusters:
            clusters = [self.active_clusters[cluster_id]]
        else:
            clusters = list(self.active_clusters.values())
        
        consolidated_count = self._consolidate_clusters(clusters)
        
        return {
            'clusters_processed': len(clusters),
            'facts_consolidated': consolidated_count,
            'timestamp': datetime.now().isoformat()
        }
    
    def get_consolidation_statistics(self) -> Dict[str, Any]:
        """Get current consolidation statistics."""
        facts = self._load_facts_for_consolidation()
        return self._calculate_consolidation_statistics(facts)
    
    def add_permanent_tag(self, fact_id: int, tag: str) -> bool:
        """
        Add a permanent protection tag to a fact.
        
        Args:
            fact_id: ID of fact to protect
            tag: Protection tag to add
            
        Returns:
            True if successful
        """
        try:
            # Add to protected set
            self.protected_fact_ids.add(fact_id)
            
            # Update fact tags in database
            with get_conn(self.memory_log.db_path) as conn:
                # Get current tags
                cursor = conn.execute("SELECT tags FROM facts WHERE id = ?", (fact_id,))
                row = cursor.fetchone()
                
                if row:
                    current_tags = json.loads(row[0]) if row[0] else []
                    if tag not in current_tags:
                        current_tags.append(tag)
                    
                    # Add permanent protection tag
                    if "PERMANENT" not in current_tags:
                        current_tags.append("PERMANENT")
                    
                    # Update database
                    conn.execute(
                        "UPDATE facts SET tags = ? WHERE id = ?",
                        (json.dumps(current_tags), fact_id)
                    )
                    conn.commit()
                    
                    logging.info(f"[MemoryConsolidator] Added permanent tag '{tag}' to fact {fact_id}")
                    return True
                    
            return False
            
        except Exception as e:
            logging.error(f"[MemoryConsolidator] Failed to add permanent tag: {e}")
            return False
    
    def list_clusters(self, cluster_type: str = None) -> List[Dict[str, Any]]:
        """
        List all memory clusters with their details.
        
        Args:
            cluster_type: Filter by cluster type (returns all if None)
            
        Returns:
            List of cluster information
        """
        cluster_info = []
        
        for cluster in self.active_clusters.values():
            if cluster_type and cluster.cluster_type != cluster_type:
                continue
            
            info = {
                'cluster_id': cluster.cluster_id,
                'cluster_type': cluster.cluster_type,
                'fact_count': len(cluster.facts),
                'consolidation_score': cluster.consolidation_score,
                'created_at': cluster.created_at.isoformat(),
                'last_updated': cluster.last_updated.isoformat(),
                'similarity_threshold': cluster.similarity_threshold,
                'sample_facts': [
                    f"{fact.subject} {fact.predicate} {fact.object}"
                    for fact in cluster.facts[:3]
                ]
            }
            cluster_info.append(info)
        
        # Sort by consolidation score (highest first)
        cluster_info.sort(key=lambda x: x['consolidation_score'], reverse=True)
        
        return cluster_info