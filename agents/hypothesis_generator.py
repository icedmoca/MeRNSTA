#!/usr/bin/env python3
"""
Hypothesis Generator Agent for MeRNSTA

This agent generates hypotheses about the causes of drift and contradictions,
enabling deeper understanding of cognitive instability patterns.
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
import hashlib

from storage.enhanced_memory_model import EnhancedTripletFact


@dataclass
class Hypothesis:
    """Represents a hypothesis about the cause of drift or contradiction."""
    hypothesis_id: str
    cause_token: str
    predicted_outcome: str
    supporting_evidence: List[str]
    probability: float
    status: str  # open, confirmed, rejected
    created_at: float = 0.0
    confirmed_at: Optional[float] = None
    rejected_at: Optional[float] = None
    reflex_cycle_id: Optional[str] = None
    drift_goal_id: Optional[str] = None
    confidence_score: float = 0.0
    hypothesis_type: str = 'drift_cause'  # drift_cause, contradiction_cause, semantic_decay
    
    def __post_init__(self):
        if self.created_at == 0.0:
            self.created_at = time.time()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Hypothesis':
        """Create from dictionary."""
        return cls(**data)


class HypothesisGeneratorAgent:
    """
    Hypothesis Generator Agent that creates hypotheses about drift causes.
    
    Features:
    - Generates hypotheses when contradictions or semantic decay is detected
    - Links hypotheses to reflex cycles and drift goals
    - Tracks hypothesis confirmation/rejection
    - Learns from hypothesis outcomes
    """
    
    def __init__(self, db_path: str = "causal_predictions.db"):
        self.db_path = db_path
        
        # Hypothesis generation parameters
        self.min_evidence_count = 2  # Minimum evidence pieces for hypothesis
        self.confidence_threshold = 0.6  # Minimum confidence for hypothesis
        self.max_hypotheses_per_event = 3  # Maximum hypotheses per drift event
        
        # Hypothesis templates
        self.hypothesis_templates = {
            'contradiction': [
                "A likely cause of this contradiction is {cause_token}",
                "This contradiction may stem from {cause_token}",
                "The conflicting beliefs could be due to {cause_token}",
                "This contradiction suggests {cause_token} is unstable"
            ],
            'drift': [
                "A likely cause of this drift is {cause_token}",
                "This semantic drift may stem from {cause_token}",
                "The belief shift could be due to {cause_token}",
                "This drift suggests {cause_token} is evolving"
            ],
            'semantic_decay': [
                "A likely cause of this semantic decay is {cause_token}",
                "This coherence loss may stem from {cause_token}",
                "The concept fragmentation could be due to {cause_token}",
                "This decay suggests {cause_token} is becoming unstable"
            ]
        }
        
        # Initialize database
        self._init_database()
    
    def _init_database(self):
        """Initialize hypothesis database schema."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS hypotheses (
                hypothesis_id TEXT PRIMARY KEY,
                cause_token TEXT NOT NULL,
                predicted_outcome TEXT NOT NULL,
                supporting_evidence TEXT NOT NULL,
                probability REAL NOT NULL,
                status TEXT DEFAULT 'open',
                created_at REAL NOT NULL,
                confirmed_at REAL,
                rejected_at REAL,
                reflex_cycle_id TEXT,
                drift_goal_id TEXT,
                confidence_score REAL DEFAULT 0.0,
                hypothesis_type TEXT DEFAULT 'drift_cause',
                INDEX(status),
                INDEX(hypothesis_type),
                INDEX(created_at)
            )
        """)
        
        conn.commit()
        conn.close()
    
    def generate_hypotheses_for_contradiction(self, contradiction_data: Dict[str, Any], 
                                            reflex_cycle_id: Optional[str] = None) -> List[Hypothesis]:
        """
        Generate hypotheses for a detected contradiction.
        
        Args:
            contradiction_data: Data about the contradiction
            reflex_cycle_id: Optional reflex cycle ID
            
        Returns:
            List of generated hypotheses
        """
        logging.info(f"[HypothesisGenerator] Generating hypotheses for contradiction")
        
        # Extract contradiction information
        fact_a = contradiction_data.get('fact_a', {})
        fact_b = contradiction_data.get('fact_b', {})
        contradiction_score = contradiction_data.get('contradiction_score', 0.0)
        
        # Find potential causes
        potential_causes = self._identify_contradiction_causes(fact_a, fact_b)
        
        # Generate hypotheses
        hypotheses = []
        for cause in potential_causes[:self.max_hypotheses_per_event]:
            hypothesis = self._create_hypothesis(
                cause_token=cause['token'],
                hypothesis_type='contradiction_cause',
                supporting_evidence=cause['evidence'],
                probability=cause['probability'],
                reflex_cycle_id=reflex_cycle_id
            )
            if hypothesis:
                hypotheses.append(hypothesis)
        
        # Store hypotheses
        for hypothesis in hypotheses:
            self._store_hypothesis(hypothesis)
        
        logging.info(f"[HypothesisGenerator] Generated {len(hypotheses)} hypotheses")
        return hypotheses
    
    def generate_hypotheses_for_drift(self, drift_data: Dict[str, Any],
                                     reflex_cycle_id: Optional[str] = None) -> List[Hypothesis]:
        """
        Generate hypotheses for detected semantic drift.
        
        Args:
            drift_data: Data about the drift
            reflex_cycle_id: Optional reflex cycle ID
            
        Returns:
            List of generated hypotheses
        """
        logging.info(f"[HypothesisGenerator] Generating hypotheses for drift")
        
        # Extract drift information
        token_id = drift_data.get('token_id')
        drift_score = drift_data.get('drift_score', 0.0)
        cluster_id = drift_data.get('cluster_id')
        
        # Find potential causes
        potential_causes = self._identify_drift_causes(token_id, cluster_id, drift_score)
        
        # Generate hypotheses
        hypotheses = []
        for cause in potential_causes[:self.max_hypotheses_per_event]:
            hypothesis = self._create_hypothesis(
                cause_token=cause['token'],
                hypothesis_type='drift_cause',
                supporting_evidence=cause['evidence'],
                probability=cause['probability'],
                reflex_cycle_id=reflex_cycle_id
            )
            if hypothesis:
                hypotheses.append(hypothesis)
        
        # Store hypotheses
        for hypothesis in hypotheses:
            self._store_hypothesis(hypothesis)
        
        logging.info(f"[HypothesisGenerator] Generated {len(hypotheses)} hypotheses")
        return hypotheses
    
    def generate_hypotheses_for_semantic_decay(self, decay_data: Dict[str, Any],
                                              reflex_cycle_id: Optional[str] = None) -> List[Hypothesis]:
        """
        Generate hypotheses for semantic decay.
        
        Args:
            decay_data: Data about the semantic decay
            reflex_cycle_id: Optional reflex cycle ID
            
        Returns:
            List of generated hypotheses
        """
        logging.info(f"[HypothesisGenerator] Generating hypotheses for semantic decay")
        
        # Extract decay information
        cluster_id = decay_data.get('cluster_id')
        coherence_score = decay_data.get('coherence_score', 0.0)
        volatility_score = decay_data.get('volatility_score', 0.0)
        
        # Find potential causes
        potential_causes = self._identify_decay_causes(cluster_id, coherence_score, volatility_score)
        
        # Generate hypotheses
        hypotheses = []
        for cause in potential_causes[:self.max_hypotheses_per_event]:
            hypothesis = self._create_hypothesis(
                cause_token=cause['token'],
                hypothesis_type='semantic_decay',
                supporting_evidence=cause['evidence'],
                probability=cause['probability'],
                reflex_cycle_id=reflex_cycle_id
            )
            if hypothesis:
                hypotheses.append(hypothesis)
        
        # Store hypotheses
        for hypothesis in hypotheses:
            self._store_hypothesis(hypothesis)
        
        logging.info(f"[HypothesisGenerator] Generated {len(hypotheses)} hypotheses")
        return hypotheses
    
    def _identify_contradiction_causes(self, fact_a: Dict[str, Any], 
                                      fact_b: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify potential causes of a contradiction."""
        causes = []
        
        # Analyze subject differences
        if fact_a.get('subject') != fact_b.get('subject'):
            causes.append({
                'token': f"subject_mismatch_{fact_a.get('subject')}_{fact_b.get('subject')}",
                'evidence': [f"Different subjects: {fact_a.get('subject')} vs {fact_b.get('subject')}"],
                'probability': 0.7
            })
        
        # Analyze predicate conflicts
        if fact_a.get('predicate') != fact_b.get('predicate'):
            causes.append({
                'token': f"predicate_conflict_{fact_a.get('predicate')}_{fact_b.get('predicate')}",
                'evidence': [f"Conflicting predicates: {fact_a.get('predicate')} vs {fact_b.get('predicate')}"],
                'probability': 0.8
            })
        
        # Analyze object contradictions
        if fact_a.get('object') != fact_b.get('object'):
            causes.append({
                'token': f"object_contradiction_{fact_a.get('object')}_{fact_b.get('object')}",
                'evidence': [f"Contradictory objects: {fact_a.get('object')} vs {fact_b.get('object')}"],
                'probability': 0.9
            })
        
        # Analyze confidence differences
        confidence_diff = abs(fact_a.get('confidence', 0.5) - fact_b.get('confidence', 0.5))
        if confidence_diff > 0.3:
            causes.append({
                'token': f"confidence_imbalance_{confidence_diff:.2f}",
                'evidence': [f"High confidence difference: {confidence_diff:.2f}"],
                'probability': 0.6
            })
        
        # Analyze temporal factors
        if fact_a.get('timestamp') and fact_b.get('timestamp'):
            time_diff = abs(fact_a['timestamp'] - fact_b['timestamp'])
            if time_diff > 24 * 3600:  # More than 24 hours
                causes.append({
                    'token': f"temporal_drift_{time_diff//3600}h",
                    'evidence': [f"Large time gap: {time_diff//3600} hours"],
                    'probability': 0.5
                })
        
        return sorted(causes, key=lambda x: x['probability'], reverse=True)
    
    def _identify_drift_causes(self, token_id: Optional[int], 
                               cluster_id: Optional[str], 
                               drift_score: float) -> List[Dict[str, Any]]:
        """Identify potential causes of semantic drift."""
        causes = []
        
        # High drift score
        if drift_score > 0.7:
            causes.append({
                'token': f"high_drift_score_{drift_score:.2f}",
                'evidence': [f"High drift score: {drift_score:.2f}"],
                'probability': 0.8
            })
        
        # Cluster instability
        if cluster_id:
            causes.append({
                'token': f"cluster_instability_{cluster_id}",
                'evidence': [f"Cluster {cluster_id} showing instability"],
                'probability': 0.6
            })
        
        # Token-specific factors
        if token_id:
            causes.append({
                'token': f"token_evolution_{token_id}",
                'evidence': [f"Token {token_id} evolving over time"],
                'probability': 0.7
            })
        
        # Semantic context changes
        causes.append({
            'token': "semantic_context_shift",
            'evidence': ["Semantic context has shifted"],
            'probability': 0.5
        })
        
        return sorted(causes, key=lambda x: x['probability'], reverse=True)
    
    def _identify_decay_causes(self, cluster_id: Optional[str], 
                               coherence_score: float, 
                               volatility_score: float) -> List[Dict[str, Any]]:
        """Identify potential causes of semantic decay."""
        causes = []
        
        # Low coherence
        if coherence_score < 0.5:
            causes.append({
                'token': f"low_coherence_{coherence_score:.2f}",
                'evidence': [f"Low coherence score: {coherence_score:.2f}"],
                'probability': 0.8
            })
        
        # High volatility
        if volatility_score > 0.6:
            causes.append({
                'token': f"high_volatility_{volatility_score:.2f}",
                'evidence': [f"High volatility score: {volatility_score:.2f}"],
                'probability': 0.7
            })
        
        # Cluster fragmentation
        if cluster_id:
            causes.append({
                'token': f"cluster_fragmentation_{cluster_id}",
                'evidence': [f"Cluster {cluster_id} is fragmenting"],
                'probability': 0.6
            })
        
        # Concept drift
        causes.append({
            'token': "concept_drift",
            'evidence': ["Underlying concept is drifting"],
            'probability': 0.5
        })
        
        return sorted(causes, key=lambda x: x['probability'], reverse=True)
    
    def _create_hypothesis(self, cause_token: str, hypothesis_type: str,
                          supporting_evidence: List[str], probability: float,
                          reflex_cycle_id: Optional[str] = None) -> Optional[Hypothesis]:
        """Create a hypothesis object."""
        if len(supporting_evidence) < self.min_evidence_count:
            return None
        
        if probability < self.confidence_threshold:
            return None
        
        hypothesis_id = f"hypothesis_{hypothesis_type}_{int(time.time())}"
        
        # Select template based on type
        templates = self.hypothesis_templates.get(hypothesis_type, self.hypothesis_templates['drift'])
        template = np.random.choice(templates)
        
        # Generate predicted outcome
        predicted_outcome = template.format(cause_token=cause_token)
        
        # Calculate confidence score
        confidence_score = probability * (1.0 + len(supporting_evidence) * 0.1)
        confidence_score = min(1.0, confidence_score)
        
        return Hypothesis(
            hypothesis_id=hypothesis_id,
            cause_token=cause_token,
            predicted_outcome=predicted_outcome,
            supporting_evidence=supporting_evidence,
            probability=probability,
            status='open',
            reflex_cycle_id=reflex_cycle_id,
            confidence_score=confidence_score,
            hypothesis_type=hypothesis_type
        )
    
    def _store_hypothesis(self, hypothesis: Hypothesis):
        """Store hypothesis in database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO hypotheses 
            (hypothesis_id, cause_token, predicted_outcome, supporting_evidence,
             probability, status, created_at, reflex_cycle_id, confidence_score, hypothesis_type)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            hypothesis.hypothesis_id, hypothesis.cause_token, hypothesis.predicted_outcome,
            json.dumps(hypothesis.supporting_evidence), hypothesis.probability,
            hypothesis.status, hypothesis.created_at, hypothesis.reflex_cycle_id,
            hypothesis.confidence_score, hypothesis.hypothesis_type
        ))
        
        conn.commit()
        conn.close()
    
    def confirm_hypothesis(self, hypothesis_id: str):
        """Mark a hypothesis as confirmed."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            UPDATE hypotheses 
            SET status = 'confirmed', confirmed_at = ?
            WHERE hypothesis_id = ?
        """, (time.time(), hypothesis_id))
        
        conn.commit()
        conn.close()
        
        logging.info(f"[HypothesisGenerator] Confirmed hypothesis {hypothesis_id}")
    
    def reject_hypothesis(self, hypothesis_id: str):
        """Mark a hypothesis as rejected."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            UPDATE hypotheses 
            SET status = 'rejected', rejected_at = ?
            WHERE hypothesis_id = ?
        """, (time.time(), hypothesis_id))
        
        conn.commit()
        conn.close()
        
        logging.info(f"[HypothesisGenerator] Rejected hypothesis {hypothesis_id}")
    
    def get_open_hypotheses(self, limit: Optional[int] = None) -> List[Hypothesis]:
        """Get open hypotheses."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        query = """
            SELECT hypothesis_id, cause_token, predicted_outcome, supporting_evidence,
                   probability, status, created_at, confirmed_at, rejected_at,
                   reflex_cycle_id, drift_goal_id, confidence_score, hypothesis_type
            FROM hypotheses 
            WHERE status = 'open'
            ORDER BY confidence_score DESC, created_at DESC
        """
        
        if limit:
            query += f" LIMIT {limit}"
        
        cursor.execute(query)
        
        hypotheses = []
        for row in cursor.fetchall():
            data = {
                'hypothesis_id': row[0],
                'cause_token': row[1],
                'predicted_outcome': row[2],
                'supporting_evidence': json.loads(row[3]),
                'probability': row[4],
                'status': row[5],
                'created_at': row[6],
                'confirmed_at': row[7],
                'rejected_at': row[8],
                'reflex_cycle_id': row[9],
                'drift_goal_id': row[10],
                'confidence_score': row[11],
                'hypothesis_type': row[12]
            }
            hypotheses.append(Hypothesis.from_dict(data))
        
        conn.close()
        return hypotheses
    
    def get_hypotheses_by_type(self, hypothesis_type: str, limit: Optional[int] = None) -> List[Hypothesis]:
        """Get hypotheses by type."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        query = """
            SELECT hypothesis_id, cause_token, predicted_outcome, supporting_evidence,
                   probability, status, created_at, confirmed_at, rejected_at,
                   reflex_cycle_id, drift_goal_id, confidence_score, hypothesis_type
            FROM hypotheses 
            WHERE hypothesis_type = ?
            ORDER BY confidence_score DESC, created_at DESC
        """
        
        if limit:
            query += f" LIMIT {limit}"
        
        cursor.execute(query, (hypothesis_type,))
        
        hypotheses = []
        for row in cursor.fetchall():
            data = {
                'hypothesis_id': row[0],
                'cause_token': row[1],
                'predicted_outcome': row[2],
                'supporting_evidence': json.loads(row[3]),
                'probability': row[4],
                'status': row[5],
                'created_at': row[6],
                'confirmed_at': row[7],
                'rejected_at': row[8],
                'reflex_cycle_id': row[9],
                'drift_goal_id': row[10],
                'confidence_score': row[11],
                'hypothesis_type': row[12]
            }
            hypotheses.append(Hypothesis.from_dict(data))
        
        conn.close()
        return hypotheses
    
    def get_hypothesis_statistics(self) -> Dict[str, Any]:
        """Get statistics about hypotheses."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Total hypotheses
        cursor.execute("SELECT COUNT(*) FROM hypotheses")
        total_hypotheses = cursor.fetchone()[0]
        
        # Hypotheses by status
        cursor.execute("""
            SELECT status, COUNT(*) 
            FROM hypotheses 
            GROUP BY status
        """)
        by_status = dict(cursor.fetchall())
        
        # Hypotheses by type
        cursor.execute("""
            SELECT hypothesis_type, COUNT(*) 
            FROM hypotheses 
            GROUP BY hypothesis_type
        """)
        by_type = dict(cursor.fetchall())
        
        # Average confidence
        cursor.execute("SELECT AVG(confidence_score) FROM hypotheses")
        avg_confidence = cursor.fetchone()[0] or 0.0
        
        # Recent hypotheses (last 24 hours)
        cursor.execute("""
            SELECT COUNT(*) FROM hypotheses 
            WHERE created_at > ?
        """, (time.time() - 24 * 3600,))
        recent_hypotheses = cursor.fetchone()[0]
        
        # Confirmation rate
        confirmed = by_status.get('confirmed', 0)
        rejected = by_status.get('rejected', 0)
        total_resolved = confirmed + rejected
        confirmation_rate = confirmed / total_resolved if total_resolved > 0 else 0.0
        
        conn.close()
        
        return {
            'total_hypotheses': total_hypotheses,
            'by_status': by_status,
            'by_type': by_type,
            'average_confidence': avg_confidence,
            'recent_hypotheses_24h': recent_hypotheses,
            'confirmation_rate': confirmation_rate
        }
    
    def link_hypothesis_to_goal(self, hypothesis_id: str, drift_goal_id: str):
        """Link a hypothesis to a drift goal."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            UPDATE hypotheses 
            SET drift_goal_id = ?
            WHERE hypothesis_id = ?
        """, (drift_goal_id, hypothesis_id))
        
        conn.commit()
        conn.close() 