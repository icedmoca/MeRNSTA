#!/usr/bin/env python3
"""
Enhanced Memory System with Belief Abstraction Layer

This module implements the Belief Abstraction Layer that creates abstract beliefs
from consistent fact clusters, enabling higher-level reasoning and pattern recognition.
"""

import sqlite3
import logging
import json
import time
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
import numpy as np
from collections import defaultdict

from .enhanced_memory_model import EnhancedTripletFact
from .enhanced_semantic_search import SemanticMemorySearchEngine


@dataclass
class BeliefFact:
    """
    Represents an abstracted belief derived from consistent fact clusters.
    
    Beliefs are higher-level abstractions that emerge from patterns of consistent
    facts over time, enabling more sophisticated reasoning and decision-making.
    """
    belief_id: str
    cluster_id: str
    abstract_statement: str
    supporting_facts: List[str]  # List of fact IDs that support this belief
    vector: Optional[List[float]] = None
    created_at: float = 0.0
    confidence: float = 0.0
    volatility_score: float = 0.0
    last_updated: float = 0.0
    usage_count: int = 0
    coherence_score: float = 0.0
    
    def __post_init__(self):
        if self.created_at == 0.0:
            self.created_at = time.time()
        if self.last_updated == 0.0:
            self.last_updated = time.time()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        if self.vector is not None:
            data['vector'] = json.dumps(self.vector)
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BeliefFact':
        """Create from dictionary."""
        if 'vector' in data and isinstance(data['vector'], str):
            data['vector'] = json.loads(data['vector'])
        return cls(**data)


class BeliefAbstractionLayer:
    """
    Belief Abstraction Layer that creates abstract beliefs from consistent fact clusters.
    
    Features:
    - Scans clusters for consistent patterns (≥3 facts, 0 contradictions, low volatility)
    - Uses LLM to summarize facts into abstract beliefs
    - Indexes beliefs by cluster_id for future abstract reasoning
    - Tracks belief confidence and usage patterns
    """
    
    def __init__(self, db_path: str = "enhanced_memory.db", 
                 ollama_host: Optional[str] = None):
        self.db_path = db_path
        self.search_engine = SemanticMemorySearchEngine(
            ollama_host=ollama_host,
            ollama_model=None
        )
        self.goal_counter = 0
        self.belief_scan_interval = 10  # Scan every 10 goals
        self.min_facts_for_belief = 3
        self.max_contradictions = 0
        self.max_volatility = 0.3
        
        # Initialize database
        self._init_database()
    
    def _init_database(self):
        """Initialize belief facts database schema."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS belief_facts (
                belief_id TEXT PRIMARY KEY,
                cluster_id TEXT NOT NULL,
                abstract_statement TEXT NOT NULL,
                supporting_facts TEXT NOT NULL,
                vector TEXT,
                created_at REAL NOT NULL,
                confidence REAL DEFAULT 0.0,
                volatility_score REAL DEFAULT 0.0,
                last_updated REAL NOT NULL,
                usage_count INTEGER DEFAULT 0,
                coherence_score REAL DEFAULT 0.0,
                INDEX(cluster_id),
                INDEX(confidence),
                INDEX(coherence_score)
            )
        """)
        
        conn.commit()
        conn.close()
    
    def increment_goal_counter(self):
        """Increment goal counter and trigger belief scan if needed."""
        self.goal_counter += 1
        if self.goal_counter % self.belief_scan_interval == 0:
            self.scan_for_beliefs()
    
    def scan_for_beliefs(self) -> List[BeliefFact]:
        """
        Scan clusters for consistent patterns and create abstract beliefs.
        
        Returns:
            List of newly created or updated BeliefFact objects
        """
        logging.info(f"[BeliefAbstraction] Scanning for beliefs (goal #{self.goal_counter})")
        
        # Get all facts from database
        facts = self._get_all_facts()
        if not facts:
            return []
        
        # Group facts by clusters
        cluster_facts = self._group_facts_by_clusters(facts)
        
        # Analyze each cluster for belief potential
        new_beliefs = []
        for cluster_id, cluster_facts_list in cluster_facts.items():
            belief = self._analyze_cluster_for_belief(cluster_id, cluster_facts_list)
            if belief:
                new_beliefs.append(belief)
        
        logging.info(f"[BeliefAbstraction] Created {len(new_beliefs)} new beliefs")
        return new_beliefs
    
    def _get_all_facts(self) -> List[EnhancedTripletFact]:
        """Get all facts from the enhanced memory database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT id, subject, predicate, object, confidence, contradiction, 
                   volatility_score, embedding, token_ids
            FROM enhanced_facts 
            WHERE active = TRUE
        """)
        
        facts = []
        for row in cursor.fetchall():
            fact = EnhancedTripletFact(
                id=row[0],
                subject=row[1],
                predicate=row[2],
                object=row[3],
                confidence=row[4],
                contradiction=bool(row[5]),
                volatility_score=row[6],
                embedding=row[7],
                token_ids=row[8]
            )
            facts.append(fact)
        
        conn.close()
        return facts
    
    def _group_facts_by_clusters(self, facts: List[EnhancedTripletFact]) -> Dict[str, List[EnhancedTripletFact]]:
        """Group facts by their semantic clusters."""
        clusters = defaultdict(list)
        
        for fact in facts:
            # Use token_ids to determine cluster
            if fact.token_ids:
                try:
                    token_data = json.loads(fact.token_ids)
                    cluster_id = token_data.get('cluster_id', 'default')
                except (json.JSONDecodeError, KeyError):
                    cluster_id = 'default'
            else:
                cluster_id = 'default'
            
            clusters[cluster_id].append(fact)
        
        return dict(clusters)
    
    def _analyze_cluster_for_belief(self, cluster_id: str, 
                                   cluster_facts: List[EnhancedTripletFact]) -> Optional[BeliefFact]:
        """
        Analyze a cluster to determine if it should become a belief.
        
        Criteria:
        - ≥3 consistent facts
        - 0 contradictions
        - Low volatility
        """
        if len(cluster_facts) < self.min_facts_for_belief:
            return None
        
        # Check for contradictions
        contradictions = [f for f in cluster_facts if f.contradiction]
        if len(contradictions) > self.max_contradictions:
            return None
        
        # Check volatility
        avg_volatility = np.mean([f.volatility_score for f in cluster_facts])
        if avg_volatility > self.max_volatility:
            return None
        
        # Check if belief already exists
        existing_belief = self._get_belief_by_cluster(cluster_id)
        if existing_belief:
            # Update existing belief
            return self._update_existing_belief(existing_belief, cluster_facts)
        
        # Create new belief
        return self._create_new_belief(cluster_id, cluster_facts)
    
    def _get_belief_by_cluster(self, cluster_id: str) -> Optional[BeliefFact]:
        """Get existing belief for a cluster."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT belief_id, cluster_id, abstract_statement, supporting_facts,
                   vector, created_at, confidence, volatility_score, last_updated,
                   usage_count, coherence_score
            FROM belief_facts 
            WHERE cluster_id = ?
        """, (cluster_id,))
        
        row = cursor.fetchone()
        conn.close()
        
        if row:
            data = {
                'belief_id': row[0],
                'cluster_id': row[1],
                'abstract_statement': row[2],
                'supporting_facts': json.loads(row[3]),
                'vector': json.loads(row[4]) if row[4] else None,
                'created_at': row[5],
                'confidence': row[6],
                'volatility_score': row[7],
                'last_updated': row[8],
                'usage_count': row[9],
                'coherence_score': row[10]
            }
            return BeliefFact.from_dict(data)
        
        return None
    
    def _create_new_belief(self, cluster_id: str, 
                          cluster_facts: List[EnhancedTripletFact]) -> BeliefFact:
        """Create a new belief from cluster facts."""
        belief_id = f"belief_{cluster_id}_{int(time.time())}"
        
        # Generate abstract statement using LLM or pattern matching
        abstract_statement = self._generate_abstract_statement(cluster_facts)
        
        # Calculate confidence based on fact consistency
        confidence = self._calculate_belief_confidence(cluster_facts)
        
        # Calculate coherence score
        coherence_score = self._calculate_cluster_coherence(cluster_facts)
        
        # Create belief
        belief = BeliefFact(
            belief_id=belief_id,
            cluster_id=cluster_id,
            abstract_statement=abstract_statement,
            supporting_facts=[str(f.id) for f in cluster_facts],
            confidence=confidence,
            coherence_score=coherence_score,
            volatility_score=np.mean([f.volatility_score for f in cluster_facts])
        )
        
        # Store in database
        self._store_belief(belief)
        
        return belief
    
    def _update_existing_belief(self, belief: BeliefFact, 
                               cluster_facts: List[EnhancedTripletFact]) -> BeliefFact:
        """Update existing belief with new facts."""
        # Update supporting facts
        new_fact_ids = [str(f.id) for f in cluster_facts]
        belief.supporting_facts = list(set(belief.supporting_facts + new_fact_ids))
        
        # Recalculate metrics
        belief.confidence = self._calculate_belief_confidence(cluster_facts)
        belief.coherence_score = self._calculate_cluster_coherence(cluster_facts)
        belief.volatility_score = np.mean([f.volatility_score for f in cluster_facts])
        belief.last_updated = time.time()
        
        # Update in database
        self._update_belief(belief)
        
        return belief
    
    def _generate_abstract_statement(self, facts: List[EnhancedTripletFact]) -> str:
        """Generate abstract statement from facts using LLM or pattern matching."""
        # Simple pattern-based abstraction for now
        if not facts:
            return "No facts available"
        
        # Group by subject
        subjects = defaultdict(list)
        for fact in facts:
            subjects[fact.subject].append(fact)
        
        # Find most common subject
        if subjects:
            main_subject = max(subjects.keys(), key=lambda s: len(subjects[s]))
            subject_facts = subjects[main_subject]
            
            # Create simple abstraction
            predicates = [f.predicate for f in subject_facts]
            objects = [f.object for f in subject_facts]
            
            if len(set(predicates)) == 1:
                # Same predicate, different objects
                return f"{main_subject} {predicates[0]} multiple things: {', '.join(set(objects))}"
            else:
                # Different predicates
                return f"{main_subject} has various characteristics: {', '.join(set(predicates))}"
        
        return "Complex belief pattern detected"
    
    def _calculate_belief_confidence(self, facts: List[EnhancedTripletFact]) -> float:
        """Calculate confidence score for a belief based on fact consistency."""
        if not facts:
            return 0.0
        
        # Average confidence of supporting facts
        avg_confidence = np.mean([f.confidence for f in facts])
        
        # Penalty for contradictions
        contradiction_penalty = len([f for f in facts if f.contradiction]) * 0.1
        
        # Bonus for consistency
        consistency_bonus = 0.1 if len(facts) >= 5 else 0.0
        
        confidence = avg_confidence - contradiction_penalty + consistency_bonus
        return max(0.0, min(1.0, confidence))
    
    def _calculate_cluster_coherence(self, facts: List[EnhancedTripletFact]) -> float:
        """Calculate coherence score for a cluster of facts."""
        if len(facts) < 2:
            return 1.0
        
        # Calculate semantic similarity between facts
        similarities = []
        for i in range(len(facts)):
            for j in range(i + 1, len(facts)):
                if facts[i].embedding and facts[j].embedding:
                    try:
                        vec1 = json.loads(facts[i].embedding)
                        vec2 = json.loads(facts[j].embedding)
                        similarity = self._cosine_similarity(vec1, vec2)
                        similarities.append(similarity)
                    except (json.JSONDecodeError, ValueError):
                        continue
        
        if similarities:
            return np.mean(similarities)
        
        return 0.5  # Default coherence
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        if len(vec1) != len(vec2):
            return 0.0
        
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = sum(a * a for a in vec1) ** 0.5
        norm2 = sum(b * b for b in vec2) ** 0.5
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def _store_belief(self, belief: BeliefFact):
        """Store belief in database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO belief_facts 
            (belief_id, cluster_id, abstract_statement, supporting_facts, vector,
             created_at, confidence, volatility_score, last_updated, usage_count, coherence_score)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            belief.belief_id, belief.cluster_id, belief.abstract_statement,
            json.dumps(belief.supporting_facts), json.dumps(belief.vector) if belief.vector else None,
            belief.created_at, belief.confidence, belief.volatility_score,
            belief.last_updated, belief.usage_count, belief.coherence_score
        ))
        
        conn.commit()
        conn.close()
    
    def _update_belief(self, belief: BeliefFact):
        """Update belief in database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            UPDATE belief_facts 
            SET abstract_statement = ?, supporting_facts = ?, vector = ?,
                confidence = ?, volatility_score = ?, last_updated = ?, 
                usage_count = ?, coherence_score = ?
            WHERE belief_id = ?
        """, (
            belief.abstract_statement, json.dumps(belief.supporting_facts),
            json.dumps(belief.vector) if belief.vector else None,
            belief.confidence, belief.volatility_score, belief.last_updated,
            belief.usage_count, belief.coherence_score, belief.belief_id
        ))
        
        conn.commit()
        conn.close()
    
    def get_beliefs_by_cluster(self, cluster_id: str) -> List[BeliefFact]:
        """Get all beliefs for a specific cluster."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT belief_id, cluster_id, abstract_statement, supporting_facts,
                   vector, created_at, confidence, volatility_score, last_updated,
                   usage_count, coherence_score
            FROM belief_facts 
            WHERE cluster_id = ?
            ORDER BY confidence DESC
        """, (cluster_id,))
        
        beliefs = []
        for row in cursor.fetchall():
            data = {
                'belief_id': row[0],
                'cluster_id': row[1],
                'abstract_statement': row[2],
                'supporting_facts': json.loads(row[3]),
                'vector': json.loads(row[4]) if row[4] else None,
                'created_at': row[5],
                'confidence': row[6],
                'volatility_score': row[7],
                'last_updated': row[8],
                'usage_count': row[9],
                'coherence_score': row[10]
            }
            beliefs.append(BeliefFact.from_dict(data))
        
        conn.close()
        return beliefs
    
    def get_all_beliefs(self, limit: Optional[int] = None) -> List[BeliefFact]:
        """Get all beliefs, optionally limited."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        query = """
            SELECT belief_id, cluster_id, abstract_statement, supporting_facts,
                   vector, created_at, confidence, volatility_score, last_updated,
                   usage_count, coherence_score
            FROM belief_facts 
            ORDER BY confidence DESC, coherence_score DESC
        """
        
        if limit:
            query += f" LIMIT {limit}"
        
        cursor.execute(query)
        
        beliefs = []
        for row in cursor.fetchall():
            data = {
                'belief_id': row[0],
                'cluster_id': row[1],
                'abstract_statement': row[2],
                'supporting_facts': json.loads(row[3]),
                'vector': json.loads(row[4]) if row[4] else None,
                'created_at': row[5],
                'confidence': row[6],
                'volatility_score': row[7],
                'last_updated': row[8],
                'usage_count': row[9],
                'coherence_score': row[10]
            }
            beliefs.append(BeliefFact.from_dict(data))
        
        conn.close()
        return beliefs
    
    def increment_usage(self, belief_id: str):
        """Increment usage count for a belief."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            UPDATE belief_facts 
            SET usage_count = usage_count + 1, last_updated = ?
            WHERE belief_id = ?
        """, (time.time(), belief_id))
        
        conn.commit()
        conn.close()
    
    def get_belief_statistics(self) -> Dict[str, Any]:
        """Get statistics about beliefs."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Total beliefs
        cursor.execute("SELECT COUNT(*) FROM belief_facts")
        total_beliefs = cursor.fetchone()[0]
        
        # Average confidence
        cursor.execute("SELECT AVG(confidence) FROM belief_facts")
        avg_confidence = cursor.fetchone()[0] or 0.0
        
        # Average coherence
        cursor.execute("SELECT AVG(coherence_score) FROM belief_facts")
        avg_coherence = cursor.fetchone()[0] or 0.0
        
        # Most used beliefs
        cursor.execute("""
            SELECT belief_id, abstract_statement, usage_count 
            FROM belief_facts 
            ORDER BY usage_count DESC 
            LIMIT 5
        """)
        top_used = cursor.fetchall()
        
        # Beliefs by cluster
        cursor.execute("""
            SELECT cluster_id, COUNT(*) 
            FROM belief_facts 
            GROUP BY cluster_id 
            ORDER BY COUNT(*) DESC
        """)
        by_cluster = cursor.fetchall()
        
        conn.close()
        
        return {
            'total_beliefs': total_beliefs,
            'average_confidence': avg_confidence,
            'average_coherence': avg_coherence,
            'top_used_beliefs': [
                {'belief_id': row[0], 'statement': row[1], 'usage': row[2]}
                for row in top_used
            ],
            'beliefs_by_cluster': [
                {'cluster_id': row[0], 'count': row[1]}
                for row in by_cluster
            ]
        } 