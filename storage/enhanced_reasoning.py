#!/usr/bin/env python3
"""
Enhanced Reasoning from Abstract Beliefs for MeRNSTA

This module implements enhanced reasoning that leverages abstract beliefs
to inform strategy routing and execution planning for new goals.
"""

import sqlite3
import logging
import json
import time
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
import numpy as np
from collections import defaultdict

from .enhanced_memory import BeliefFact, BeliefAbstractionLayer
from .reflex_compression import ReflexTemplate, ReflexCompressor
from .enhanced_memory_model import EnhancedTripletFact


@dataclass
class BeliefTrace:
    """
    Represents a trace of beliefs that informed a decision.
    
    Tracks which abstract beliefs were considered during goal processing
    and how they influenced the final decision.
    """
    trace_id: str
    goal_id: str
    token_id: Optional[int]
    cluster_id: Optional[str]
    considered_beliefs: List[str]  # List of belief IDs
    belief_influences: Dict[str, float]  # Belief ID -> influence score
    final_strategy: str
    strategy_confidence: float
    timestamp: float = 0.0
    reasoning_notes: str = ""
    
    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BeliefTrace':
        """Create from dictionary."""
        return cls(**data)


class EnhancedReasoningEngine:
    """
    Enhanced reasoning engine that leverages abstract beliefs for better decision making.
    
    Features:
    - Retrieves relevant beliefs for new goals
    - Uses beliefs to inform strategy routing
    - Suggests reflex templates based on belief patterns
    - Tracks belief influence on decisions
    - Integrates with existing drift execution engine
    """
    
    def __init__(self, db_path: str = "enhanced_memory.db"):
        self.db_path = db_path
        self.belief_abstraction = BeliefAbstractionLayer(db_path)
        self.reflex_compressor = ReflexCompressor()
        
        # Reasoning parameters
        self.belief_relevance_threshold = 0.6
        self.max_beliefs_per_goal = 5
        self.strategy_confidence_threshold = 0.7
        
        # Initialize database
        self._init_database()
    
    def _init_database(self):
        """Initialize belief trace database schema."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS belief_traces (
                trace_id TEXT PRIMARY KEY,
                goal_id TEXT NOT NULL,
                token_id INTEGER,
                cluster_id TEXT,
                considered_beliefs TEXT NOT NULL,
                belief_influences TEXT NOT NULL,
                final_strategy TEXT NOT NULL,
                strategy_confidence REAL NOT NULL,
                timestamp REAL NOT NULL,
                reasoning_notes TEXT,
                INDEX(goal_id),
                INDEX(token_id),
                INDEX(cluster_id),
                INDEX(timestamp)
            )
        """)
        
        conn.commit()
        conn.close()
    
    def process_goal_with_beliefs(self, goal_id: str, drift_score: float,
                                 token_id: Optional[int] = None,
                                 cluster_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Process a goal using enhanced reasoning with abstract beliefs.
        
        Args:
            goal_id: The goal identifier
            drift_score: The drift score of the goal
            token_id: Optional token ID for context
            cluster_id: Optional cluster ID for context
            
        Returns:
            Dictionary with strategy recommendation and belief trace
        """
        logging.info(f"[EnhancedReasoning] Processing goal {goal_id} with beliefs")
        
        # Retrieve relevant beliefs
        relevant_beliefs = self._retrieve_relevant_beliefs(
            drift_score, token_id, cluster_id
        )
        
        # Analyze belief influences
        belief_influences = self._analyze_belief_influences(
            relevant_beliefs, drift_score, token_id, cluster_id
        )
        
        # Determine strategy based on beliefs
        strategy_result = self._determine_strategy_from_beliefs(
            relevant_beliefs, belief_influences, drift_score, token_id, cluster_id
        )
        
        # Get reflex template suggestions
        template_suggestions = self._get_template_suggestions(
            drift_score, token_id, cluster_id, relevant_beliefs
        )
        
        # Create belief trace
        belief_trace = self._create_belief_trace(
            goal_id, token_id, cluster_id, relevant_beliefs, 
            belief_influences, strategy_result, template_suggestions
        )
        
        # Store trace
        self._store_belief_trace(belief_trace)
        
        return {
            'strategy': strategy_result['strategy'],
            'confidence': strategy_result['confidence'],
            'reasoning': strategy_result['reasoning'],
            'relevant_beliefs': [b.belief_id for b in relevant_beliefs],
            'belief_influences': belief_influences,
            'template_suggestions': [t.template_id for t in template_suggestions],
            'trace_id': belief_trace.trace_id
        }
    
    def _retrieve_relevant_beliefs(self, drift_score: float,
                                  token_id: Optional[int] = None,
                                  cluster_id: Optional[str] = None) -> List[BeliefFact]:
        """Retrieve beliefs relevant to the current goal context."""
        all_beliefs = self.belief_abstraction.get_all_beliefs()
        
        if not all_beliefs:
            return []
        
        # Score beliefs for relevance
        scored_beliefs = []
        for belief in all_beliefs:
            relevance_score = self._calculate_belief_relevance(
                belief, drift_score, token_id, cluster_id
            )
            scored_beliefs.append((belief, relevance_score))
        
        # Sort by relevance and return top beliefs
        scored_beliefs.sort(key=lambda x: x[1], reverse=True)
        relevant_beliefs = [
            belief for belief, score in scored_beliefs 
            if score >= self.belief_relevance_threshold
        ]
        
        return relevant_beliefs[:self.max_beliefs_per_goal]
    
    def _calculate_belief_relevance(self, belief: BeliefFact, drift_score: float,
                                   token_id: Optional[int] = None,
                                   cluster_id: Optional[str] = None) -> float:
        """Calculate relevance score for a belief given the current context."""
        scores = []
        
        # Confidence weight
        scores.append(belief.confidence * 0.3)
        
        # Coherence weight
        scores.append(belief.coherence_score * 0.2)
        
        # Usage weight (normalized)
        usage_score = min(belief.usage_count / 10.0, 1.0) * 0.1
        scores.append(usage_score)
        
        # Recency weight
        days_since_update = (time.time() - belief.last_updated) / (24 * 3600)
        recency_score = max(0.0, 1.0 - (days_since_update / 30.0)) * 0.1
        scores.append(recency_score)
        
        # Context matching weight
        context_score = self._calculate_context_match_score(
            belief, drift_score, token_id, cluster_id
        )
        scores.append(context_score * 0.3)
        
        return sum(scores)
    
    def _calculate_context_match_score(self, belief: BeliefFact, drift_score: float,
                                      token_id: Optional[int] = None,
                                      cluster_id: Optional[str] = None) -> float:
        """Calculate how well a belief matches the current context."""
        # Cluster matching
        if cluster_id and belief.cluster_id == cluster_id:
            return 1.0
        
        # Volatility matching (beliefs with similar volatility patterns)
        volatility_diff = abs(belief.volatility_score - (1.0 - drift_score))
        volatility_score = max(0.0, 1.0 - volatility_diff)
        
        # Semantic matching (simplified)
        semantic_score = 0.5  # Placeholder for more sophisticated semantic matching
        
        return (volatility_score + semantic_score) / 2.0
    
    def _analyze_belief_influences(self, beliefs: List[BeliefFact], drift_score: float,
                                  token_id: Optional[int] = None,
                                  cluster_id: Optional[str] = None) -> Dict[str, float]:
        """Analyze how each belief influences the decision."""
        influences = {}
        
        for belief in beliefs:
            # Base influence from confidence and coherence
            base_influence = (belief.confidence + belief.coherence_score) / 2.0
            
            # Context adjustment
            context_adjustment = self._calculate_context_match_score(
                belief, drift_score, token_id, cluster_id
            )
            
            # Usage adjustment
            usage_adjustment = min(belief.usage_count / 5.0, 1.0) * 0.2
            
            # Calculate final influence
            influence = base_influence * context_adjustment + usage_adjustment
            influences[belief.belief_id] = min(1.0, influence)
        
        return influences
    
    def _determine_strategy_from_beliefs(self, beliefs: List[BeliefFact],
                                        belief_influences: Dict[str, float],
                                        drift_score: float,
                                        token_id: Optional[int] = None,
                                        cluster_id: Optional[str] = None) -> Dict[str, Any]:
        """Determine the best strategy based on belief analysis."""
        if not beliefs:
            # Fallback to default strategy
            return {
                'strategy': 'belief_clarification',
                'confidence': 0.5,
                'reasoning': 'No relevant beliefs found, using default strategy'
            }
        
        # Analyze belief patterns to suggest strategy
        strategy_scores = defaultdict(float)
        reasoning_notes = []
        
        for belief in beliefs:
            influence = belief_influences.get(belief.belief_id, 0.0)
            
            # Analyze belief content for strategy hints
            strategy_hint = self._extract_strategy_hint(belief)
            strategy_scores[strategy_hint] += influence
            
            reasoning_notes.append(f"Belief {belief.belief_id}: {strategy_hint} (influence: {influence:.2f})")
        
        # Determine best strategy
        if strategy_scores:
            best_strategy = max(strategy_scores.keys(), key=lambda s: strategy_scores[s])
            confidence = min(1.0, strategy_scores[best_strategy] / len(beliefs))
        else:
            best_strategy = 'belief_clarification'
            confidence = 0.5
        
        # Adjust confidence based on drift score
        if drift_score > 0.8:
            confidence *= 1.2  # High drift increases confidence
        elif drift_score < 0.3:
            confidence *= 0.8  # Low drift decreases confidence
        
        confidence = min(1.0, confidence)
        
        return {
            'strategy': best_strategy,
            'confidence': confidence,
            'reasoning': '; '.join(reasoning_notes)
        }
    
    def _extract_strategy_hint(self, belief: BeliefFact) -> str:
        """Extract strategy hint from belief content."""
        # Analyze belief abstract statement for strategy hints
        statement = belief.abstract_statement.lower()
        
        if any(word in statement for word in ['contradiction', 'conflict', 'disagree']):
            return 'belief_clarification'
        elif any(word in statement for word in ['cluster', 'group', 'pattern']):
            return 'cluster_reassessment'
        elif any(word in statement for word in ['consolidate', 'merge', 'combine']):
            return 'fact_consolidation'
        else:
            return 'belief_clarification'  # Default
    
    def _get_template_suggestions(self, drift_score: float,
                                 token_id: Optional[int] = None,
                                 cluster_id: Optional[str] = None,
                                 beliefs: List[BeliefFact] = None) -> List[ReflexTemplate]:
        """Get reflex template suggestions based on beliefs and context."""
        # Get template suggestions from reflex compressor
        templates = self.reflex_compressor.suggest_templates_for_drift(
            drift_score, token_id, cluster_id
        )
        
        # Filter templates based on belief compatibility
        if beliefs:
            compatible_templates = []
            for template in templates:
                compatibility = self._calculate_template_belief_compatibility(
                    template, beliefs
                )
                if compatibility > 0.5:  # Threshold for compatibility
                    compatible_templates.append(template)
            return compatible_templates[:3]  # Return top 3
        
        return templates[:3]
    
    def _calculate_template_belief_compatibility(self, template: ReflexTemplate,
                                                beliefs: List[BeliefFact]) -> float:
        """Calculate compatibility between a template and beliefs."""
        if not beliefs:
            return 0.5
        
        # Simple compatibility based on strategy match
        strategy_matches = sum(1 for belief in beliefs 
                             if self._extract_strategy_hint(belief) == template.strategy)
        
        return strategy_matches / len(beliefs)
    
    def _create_belief_trace(self, goal_id: str, token_id: Optional[int],
                            cluster_id: Optional[str], beliefs: List[BeliefFact],
                            belief_influences: Dict[str, float],
                            strategy_result: Dict[str, Any],
                            template_suggestions: List[ReflexTemplate]) -> BeliefTrace:
        """Create a belief trace for the decision."""
        trace_id = f"trace_{goal_id}_{int(time.time())}"
        
        reasoning_notes = f"Strategy: {strategy_result['strategy']} "
        reasoning_notes += f"(confidence: {strategy_result['confidence']:.2f}); "
        reasoning_notes += f"Templates suggested: {len(template_suggestions)}; "
        reasoning_notes += f"Beliefs considered: {len(beliefs)}"
        
        return BeliefTrace(
            trace_id=trace_id,
            goal_id=goal_id,
            token_id=token_id,
            cluster_id=cluster_id,
            considered_beliefs=[b.belief_id for b in beliefs],
            belief_influences=belief_influences,
            final_strategy=strategy_result['strategy'],
            strategy_confidence=strategy_result['confidence'],
            reasoning_notes=reasoning_notes
        )
    
    def _store_belief_trace(self, trace: BeliefTrace):
        """Store belief trace in database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO belief_traces 
            (trace_id, goal_id, token_id, cluster_id, considered_beliefs,
             belief_influences, final_strategy, strategy_confidence, timestamp, reasoning_notes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            trace.trace_id, trace.goal_id, trace.token_id, trace.cluster_id,
            json.dumps(trace.considered_beliefs), json.dumps(trace.belief_influences),
            trace.final_strategy, trace.strategy_confidence, trace.timestamp,
            trace.reasoning_notes
        ))
        
        conn.commit()
        conn.close()
    
    def get_belief_trace(self, goal_id: str) -> Optional[BeliefTrace]:
        """Get belief trace for a specific goal."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT trace_id, goal_id, token_id, cluster_id, considered_beliefs,
                   belief_influences, final_strategy, strategy_confidence, timestamp, reasoning_notes
            FROM belief_traces 
            WHERE goal_id = ?
        """, (goal_id,))
        
        row = cursor.fetchone()
        conn.close()
        
        if row:
            data = {
                'trace_id': row[0],
                'goal_id': row[1],
                'token_id': row[2],
                'cluster_id': row[3],
                'considered_beliefs': json.loads(row[4]),
                'belief_influences': json.loads(row[5]),
                'final_strategy': row[6],
                'strategy_confidence': row[7],
                'timestamp': row[8],
                'reasoning_notes': row[9]
            }
            return BeliefTrace.from_dict(data)
        
        return None
    
    def get_belief_traces_by_token(self, token_id: int, limit: Optional[int] = None) -> List[BeliefTrace]:
        """Get belief traces for a specific token."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        query = """
            SELECT trace_id, goal_id, token_id, cluster_id, considered_beliefs,
                   belief_influences, final_strategy, strategy_confidence, timestamp, reasoning_notes
            FROM belief_traces 
            WHERE token_id = ?
            ORDER BY timestamp DESC
        """
        
        if limit:
            query += f" LIMIT {limit}"
        
        cursor.execute(query, (token_id,))
        
        traces = []
        for row in cursor.fetchall():
            data = {
                'trace_id': row[0],
                'goal_id': row[1],
                'token_id': row[2],
                'cluster_id': row[3],
                'considered_beliefs': json.loads(row[4]),
                'belief_influences': json.loads(row[5]),
                'final_strategy': row[6],
                'strategy_confidence': row[7],
                'timestamp': row[8],
                'reasoning_notes': row[9]
            }
            traces.append(BeliefTrace.from_dict(data))
        
        conn.close()
        return traces
    
    def get_recent_belief_traces(self, limit: int = 20) -> List[BeliefTrace]:
        """Get recent belief traces."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT trace_id, goal_id, token_id, cluster_id, considered_beliefs,
                   belief_influences, final_strategy, strategy_confidence, timestamp, reasoning_notes
            FROM belief_traces 
            ORDER BY timestamp DESC
            LIMIT ?
        """, (limit,))
        
        traces = []
        for row in cursor.fetchall():
            data = {
                'trace_id': row[0],
                'goal_id': row[1],
                'token_id': row[2],
                'cluster_id': row[3],
                'considered_beliefs': json.loads(row[4]),
                'belief_influences': json.loads(row[5]),
                'final_strategy': row[6],
                'strategy_confidence': row[7],
                'timestamp': row[8],
                'reasoning_notes': row[9]
            }
            traces.append(BeliefTrace.from_dict(data))
        
        conn.close()
        return traces
    
    def get_belief_reasoning_statistics(self) -> Dict[str, Any]:
        """Get statistics about belief reasoning."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Total traces
        cursor.execute("SELECT COUNT(*) FROM belief_traces")
        total_traces = cursor.fetchone()[0]
        
        # Average strategy confidence
        cursor.execute("SELECT AVG(strategy_confidence) FROM belief_traces")
        avg_confidence = cursor.fetchone()[0] or 0.0
        
        # Strategies used
        cursor.execute("""
            SELECT final_strategy, COUNT(*) 
            FROM belief_traces 
            GROUP BY final_strategy 
            ORDER BY COUNT(*) DESC
        """)
        strategies_used = cursor.fetchall()
        
        # Recent activity (last 24 hours)
        cursor.execute("""
            SELECT COUNT(*) FROM belief_traces 
            WHERE timestamp > ?
        """, (time.time() - 24 * 3600,))
        recent_traces = cursor.fetchone()[0]
        
        # Beliefs most frequently considered
        cursor.execute("""
            SELECT considered_beliefs 
            FROM belief_traces 
            WHERE considered_beliefs != '[]'
        """)
        
        belief_counts = defaultdict(int)
        for row in cursor.fetchall():
            beliefs = json.loads(row[0])
            for belief_id in beliefs:
                belief_counts[belief_id] += 1
        
        top_beliefs = sorted(belief_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        
        conn.close()
        
        return {
            'total_traces': total_traces,
            'average_confidence': avg_confidence,
            'strategies_used': [
                {'strategy': row[0], 'count': row[1]}
                for row in strategies_used
            ],
            'recent_traces_24h': recent_traces,
            'top_considered_beliefs': [
                {'belief_id': belief_id, 'count': count}
                for belief_id, count in top_beliefs
            ]
        }
    
    def increment_belief_usage(self, belief_id: str):
        """Increment usage count for a belief."""
        self.belief_abstraction.increment_usage(belief_id)
    
    def trigger_belief_scan(self):
        """Trigger a belief abstraction scan."""
        return self.belief_abstraction.scan_for_beliefs()
    
    def get_beliefs_for_cluster(self, cluster_id: str) -> List[BeliefFact]:
        """Get beliefs for a specific cluster."""
        return self.belief_abstraction.get_beliefs_by_cluster(cluster_id) 