#!/usr/bin/env python3
"""
Self-Aware Symbolic Control Layer - Cognitive Self Model

This module implements symbolic self-modeling that enables MeRNSTA to reflect on
its own cognitive processes using logical predicates and symbolic reasoning.
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
import re

from .reflex_log import ReflexCycle, ReflexScore
from .drive_system import get_drive_system


@dataclass
class SymbolicRule:
    """Represents a symbolic rule in the cognitive self-model."""
    rule_id: str
    antecedent: str  # Logical condition (e.g., "Belief(cluster_x) AND Contradiction(fact_y)")
    consequent: str  # Logical conclusion (e.g., "Strategy(belief_clarification)")
    confidence: float  # Rule confidence (0.0 to 1.0)
    support_count: int  # Number of supporting examples
    contradiction_count: int  # Number of contradicting examples
    created_at: float = 0.0
    last_used: float = 0.0
    rule_type: str = "implication"  # implication, preference, constraint
    
    def __post_init__(self):
        if self.created_at == 0.0:
            self.created_at = time.time()
        if self.last_used == 0.0:
            self.last_used = time.time()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SymbolicRule':
        """Create from dictionary."""
        return cls(**data)


@dataclass
class CognitiveState:
    """Represents a snapshot of cognitive state."""
    state_id: str
    timestamp: float
    beliefs: Dict[str, float]  # cluster_id -> confidence
    contradictions: List[str]  # List of contradiction IDs
    active_strategies: List[str]  # Currently active strategies
    reflex_chain: List[str]  # Recent reflex cycle IDs
    success_patterns: Dict[str, float]  # strategy -> success_rate
    volatility_scores: Dict[str, float]  # cluster_id -> volatility
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CognitiveState':
        """Create from dictionary."""
        return cls(**data)


class CognitiveSelfModel:
    """
    Cognitive Self Model that maintains symbolic representations of system behavior.
    
    Features:
    - Stores abstract symbolic representations of behavior
    - Tracks belief dynamics, contradiction events, reflex chains
    - Uses logical predicates for reasoning
    - Evolves rules based on experience
    """
    
    def __init__(self, db_path: str = "enhanced_memory.db"):
        self.db_path = db_path
        
        # Symbolic reasoning parameters
        self.min_confidence_threshold = 0.6
        self.min_support_count = 3
        self.max_rules_per_type = 100
        
        # Phase 7: Internal motive tracking
        self.active_drives = {
            "curiosity": 0.0,
            "stability": 0.0, 
            "coherence": 0.0,
            "novelty": 0.0,
            "conflict": 0.0
        }
        self.drive_history: List[Dict[str, Any]] = []
        self.last_drive_update = 0.0
        
        # Initialize database
        self._init_database()
        
        # Rule templates for common patterns
        self._init_rule_templates()
    
    def _init_database(self):
        """Initialize cognitive self-model database schema."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Symbolic rules
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS symbolic_rules (
                rule_id TEXT PRIMARY KEY,
                antecedent TEXT NOT NULL,
                consequent TEXT NOT NULL,
                confidence REAL NOT NULL,
                support_count INTEGER DEFAULT 0,
                contradiction_count INTEGER DEFAULT 0,
                created_at REAL NOT NULL,
                last_used REAL NOT NULL,
                rule_type TEXT DEFAULT 'implication'
            )
        """)
        
        # Create indexes for symbolic_rules
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_symbolic_rules_type ON symbolic_rules(rule_type)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_symbolic_rules_confidence ON symbolic_rules(confidence)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_symbolic_rules_support ON symbolic_rules(support_count)")
        
        # Cognitive states
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS cognitive_states (
                state_id TEXT PRIMARY KEY,
                timestamp REAL NOT NULL,
                beliefs TEXT NOT NULL,
                contradictions TEXT NOT NULL,
                active_strategies TEXT NOT NULL,
                reflex_chain TEXT NOT NULL,
                success_patterns TEXT NOT NULL,
                volatility_scores TEXT NOT NULL
            )
        """)
        
        # Create indexes for cognitive_states
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_cognitive_states_timestamp ON cognitive_states(timestamp)")
        
        # Belief dynamics
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS belief_dynamics (
                dynamic_id TEXT PRIMARY KEY,
                cluster_id TEXT NOT NULL,
                belief_change REAL NOT NULL,
                cause_strategy TEXT,
                reflex_cycle_id TEXT,
                timestamp REAL NOT NULL
            )
        """)
        
        # Create indexes for belief_dynamics
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_belief_dynamics_cluster ON belief_dynamics(cluster_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_belief_dynamics_timestamp ON belief_dynamics(timestamp)")
        
        # Strategy effectiveness
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS strategy_effectiveness (
                strategy_id TEXT PRIMARY KEY,
                strategy_name TEXT NOT NULL,
                success_rate REAL DEFAULT 0.0,
                avg_score REAL DEFAULT 0.0,
                usage_count INTEGER DEFAULT 0,
                last_used REAL NOT NULL
            )
        """)
        
        # Create indexes for strategy_effectiveness
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_strategy_effectiveness_name ON strategy_effectiveness(strategy_name)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_strategy_effectiveness_rate ON strategy_effectiveness(success_rate)")
        
        conn.commit()
        conn.close()
    
    def _init_rule_templates(self):
        """Initialize common rule templates."""
        self.rule_templates = {
            'belief_contradiction': [
                "Belief({cluster}) AND Contradiction({fact}) IMPLIES Strategy(belief_clarification)",
                "Belief({cluster}) AND HighVolatility({cluster}) IMPLIES Strategy(cluster_reassessment)",
                "Contradiction({fact}) AND LowConfidence({fact}) IMPLIES Strategy(fact_consolidation)"
            ],
            'strategy_preference': [
                "DriftType(contradiction) IMPLIES Strategy(belief_clarification) PREFERRED",
                "DriftType(semantic_decay) IMPLIES Strategy(cluster_reassessment) PREFERRED",
                "HighVolatility({cluster}) IMPLIES Strategy(fact_consolidation) PREFERRED"
            ],
            'success_patterns': [
                "Strategy({strategy}) AND Context({context}) IMPLIES SuccessRate({rate})",
                "ReflexChain({chain}) IMPLIES Outcome({outcome})",
                "BeliefDynamics({dynamics}) IMPLIES Stability({stability})"
            ]
        }
    
    def record_cognitive_state(self, beliefs: Dict[str, float], 
                              contradictions: List[str],
                              active_strategies: List[str],
                              reflex_chain: List[str],
                              success_patterns: Dict[str, float],
                              volatility_scores: Dict[str, float]) -> str:
        """
        Record a snapshot of the current cognitive state.
        
        Args:
            beliefs: Current belief confidences by cluster
            contradictions: List of active contradiction IDs
            active_strategies: Currently active strategies
            reflex_chain: Recent reflex cycle IDs
            success_patterns: Strategy success rates
            volatility_scores: Cluster volatility scores
            
        Returns:
            State ID of the recorded state
        """
        state_id = f"state_{int(time.time())}"
        
        cognitive_state = CognitiveState(
            state_id=state_id,
            timestamp=time.time(),
            beliefs=beliefs,
            contradictions=contradictions,
            active_strategies=active_strategies,
            reflex_chain=reflex_chain,
            success_patterns=success_patterns,
            volatility_scores=volatility_scores
        )
        
        self._store_cognitive_state(cognitive_state)
        
        # Extract and learn rules from this state
        self._extract_rules_from_state(cognitive_state)
        
        return state_id
    
    def _store_cognitive_state(self, state: CognitiveState):
        """Store cognitive state in database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO cognitive_states 
            (state_id, timestamp, beliefs, contradictions, active_strategies,
             reflex_chain, success_patterns, volatility_scores)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            state.state_id, state.timestamp, json.dumps(state.beliefs),
            json.dumps(state.contradictions), json.dumps(state.active_strategies),
            json.dumps(state.reflex_chain), json.dumps(state.success_patterns),
            json.dumps(state.volatility_scores)
        ))
        
        conn.commit()
        conn.close()
    
    def _extract_rules_from_state(self, state: CognitiveState):
        """Extract symbolic rules from cognitive state."""
        # Extract belief-contradiction rules
        for cluster_id, belief_confidence in state.beliefs.items():
            for contradiction_id in state.contradictions:
                if belief_confidence < 0.7:  # Low confidence belief with contradiction
                    rule = self._create_rule(
                        antecedent=f"Belief({cluster_id}) AND Contradiction({contradiction_id})",
                        consequent="Strategy(belief_clarification)",
                        rule_type="implication"
                    )
                    self._add_or_update_rule(rule)
        
        # Extract volatility-strategy rules
        for cluster_id, volatility in state.volatility_scores.items():
            if volatility > 0.6:  # High volatility
                rule = self._create_rule(
                    antecedent=f"HighVolatility({cluster_id})",
                    consequent="Strategy(cluster_reassessment)",
                    rule_type="implication"
                )
                self._add_or_update_rule(rule)
        
        # Extract strategy effectiveness rules
        for strategy, success_rate in state.success_patterns.items():
            if success_rate > 0.7:  # Successful strategy
                rule = self._create_rule(
                    antecedent=f"Strategy({strategy})",
                    consequent=f"SuccessRate({success_rate:.2f})",
                    rule_type="success_patterns"
                )
                self._add_or_update_rule(rule)
    
    def _create_rule(self, antecedent: str, consequent: str, rule_type: str) -> SymbolicRule:
        """Create a new symbolic rule."""
        rule_id = f"rule_{rule_type}_{int(time.time())}"
        
        return SymbolicRule(
            rule_id=rule_id,
            antecedent=antecedent,
            consequent=consequent,
            confidence=0.5,  # Initial confidence
            support_count=1,
            contradiction_count=0,
            rule_type=rule_type
        )
    
    def _add_or_update_rule(self, new_rule: SymbolicRule):
        """Add new rule or update existing similar rule."""
        # Check if similar rule exists
        existing_rule = self._find_similar_rule(new_rule)
        
        if existing_rule:
            # Update existing rule
            existing_rule.support_count += 1
            existing_rule.confidence = min(1.0, existing_rule.confidence + 0.1)
            existing_rule.last_used = time.time()
            self._update_rule(existing_rule)
        else:
            # Add new rule
            self._store_rule(new_rule)
    
    def _find_similar_rule(self, rule: SymbolicRule) -> Optional[SymbolicRule]:
        """Find a similar rule in the database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT rule_id, antecedent, consequent, confidence, support_count,
                   contradiction_count, created_at, last_used, rule_type
            FROM symbolic_rules 
            WHERE antecedent = ? AND consequent = ? AND rule_type = ?
        """, (rule.antecedent, rule.consequent, rule.rule_type))
        
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return SymbolicRule(
                rule_id=row[0],
                antecedent=row[1],
                consequent=row[2],
                confidence=row[3],
                support_count=row[4],
                contradiction_count=row[5],
                created_at=row[6],
                last_used=row[7],
                rule_type=row[8]
            )
        
        return None
    
    def _store_rule(self, rule: SymbolicRule):
        """Store a new rule in database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO symbolic_rules 
            (rule_id, antecedent, consequent, confidence, support_count,
             contradiction_count, created_at, last_used, rule_type)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            rule.rule_id, rule.antecedent, rule.consequent, rule.confidence,
            rule.support_count, rule.contradiction_count, rule.created_at,
            rule.last_used, rule.rule_type
        ))
        
        conn.commit()
        conn.close()
    
    def _update_rule(self, rule: SymbolicRule):
        """Update an existing rule in database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            UPDATE symbolic_rules 
            SET confidence = ?, support_count = ?, contradiction_count = ?, last_used = ?
            WHERE rule_id = ?
        """, (
            rule.confidence, rule.support_count, rule.contradiction_count,
            rule.last_used, rule.rule_id
        ))
        
        conn.commit()
        conn.close()
    
    def query_rules(self, query: str) -> List[SymbolicRule]:
        """
        Query symbolic rules using logical expressions.
        
        Args:
            query: Logical query (e.g., "Strategy(belief_clarification)", "HighVolatility(cluster_x)")
            
        Returns:
            List of matching rules
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Simple pattern matching for now
        cursor.execute("""
            SELECT rule_id, antecedent, consequent, confidence, support_count,
                   contradiction_count, created_at, last_used, rule_type
            FROM symbolic_rules 
            WHERE antecedent LIKE ? OR consequent LIKE ?
            ORDER BY confidence DESC, support_count DESC
        """, (f"%{query}%", f"%{query}%"))
        
        rules = []
        for row in cursor.fetchall():
            rule = SymbolicRule(
                rule_id=row[0],
                antecedent=row[1],
                consequent=row[2],
                confidence=row[3],
                support_count=row[4],
                contradiction_count=row[5],
                created_at=row[6],
                last_used=row[7],
                rule_type=row[8]
            )
            rules.append(rule)
        
        conn.close()
        return rules
    
    def get_strategy_preferences(self, context: Dict[str, Any]) -> List[Tuple[str, float]]:
        """
        Get strategy preferences based on current context.
        
        Args:
            context: Current context (drift_type, cluster_id, etc.)
            
        Returns:
            List of (strategy, preference_score) tuples
        """
        preferences = []
        
        # Query relevant rules
        if 'drift_type' in context:
            rules = self.query_rules(f"DriftType({context['drift_type']})")
            for rule in rules:
                if "PREFERRED" in rule.consequent:
                    strategy = self._extract_strategy_from_rule(rule.consequent)
                    if strategy:
                        preferences.append((strategy, rule.confidence))
        
        if 'cluster_id' in context:
            rules = self.query_rules(f"HighVolatility({context['cluster_id']})")
            for rule in rules:
                strategy = self._extract_strategy_from_rule(rule.consequent)
                if strategy:
                    preferences.append((strategy, rule.confidence))
        
        # Sort by preference score
        preferences.sort(key=lambda x: x[1], reverse=True)
        return preferences
    
    def _extract_strategy_from_rule(self, consequent: str) -> Optional[str]:
        """Extract strategy name from rule consequent."""
        # Match patterns like "Strategy(belief_clarification)" or "Strategy(belief_clarification) PREFERRED"
        match = re.search(r'Strategy\(([^)]+)\)', consequent)
        if match:
            return match.group(1)
        return None
    
    def record_belief_dynamics(self, cluster_id: str, belief_change: float,
                              cause_strategy: Optional[str] = None,
                              reflex_cycle_id: Optional[str] = None):
        """Record belief dynamics for a cluster."""
        dynamic_id = f"dynamic_{cluster_id}_{int(time.time())}"
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO belief_dynamics 
            (dynamic_id, cluster_id, belief_change, cause_strategy, reflex_cycle_id, timestamp)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            dynamic_id, cluster_id, belief_change, cause_strategy,
            reflex_cycle_id, time.time()
        ))
        
        conn.commit()
        conn.close()
    
    def record_strategy_effectiveness(self, strategy_name: str, success: bool, score: float):
        """Record strategy effectiveness."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Check if strategy exists
        cursor.execute("""
            SELECT strategy_id, success_rate, avg_score, usage_count
            FROM strategy_effectiveness 
            WHERE strategy_name = ?
        """, (strategy_name,))
        
        row = cursor.fetchone()
        
        if row:
            # Update existing strategy
            strategy_id, old_success_rate, old_avg_score, old_usage_count = row
            new_usage_count = old_usage_count + 1
            new_success_rate = ((old_success_rate * old_usage_count) + (1.0 if success else 0.0)) / new_usage_count
            new_avg_score = ((old_avg_score * old_usage_count) + score) / new_usage_count
            
            cursor.execute("""
                UPDATE strategy_effectiveness 
                SET success_rate = ?, avg_score = ?, usage_count = ?, last_used = ?
                WHERE strategy_id = ?
            """, (new_success_rate, new_avg_score, new_usage_count, time.time(), strategy_id))
        else:
            # Add new strategy
            strategy_id = f"strategy_{strategy_name}_{int(time.time())}"
            cursor.execute("""
                INSERT INTO strategy_effectiveness 
                (strategy_id, strategy_name, success_rate, avg_score, usage_count, last_used)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                strategy_id, strategy_name, 1.0 if success else 0.0, score, 1, time.time()
            ))
        
        conn.commit()
        conn.close()
    
    def get_cognitive_statistics(self) -> Dict[str, Any]:
        """Get comprehensive cognitive statistics."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Total rules
        cursor.execute("SELECT COUNT(*) FROM symbolic_rules")
        total_rules = cursor.fetchone()[0]
        
        # Rules by type
        cursor.execute("""
            SELECT rule_type, COUNT(*) 
            FROM symbolic_rules 
            GROUP BY rule_type
        """)
        rules_by_type = dict(cursor.fetchall())
        
        # Average confidence
        cursor.execute("SELECT AVG(confidence) FROM symbolic_rules")
        avg_confidence = cursor.fetchone()[0] or 0.0
        
        # Most supported rules
        cursor.execute("""
            SELECT antecedent, consequent, support_count, confidence
            FROM symbolic_rules 
            ORDER BY support_count DESC 
            LIMIT 5
        """)
        top_rules = cursor.fetchall()
        
        # Strategy effectiveness
        cursor.execute("""
            SELECT strategy_name, success_rate, avg_score, usage_count
            FROM strategy_effectiveness 
            ORDER BY success_rate DESC 
            LIMIT 5
        """)
        top_strategies = cursor.fetchall()
        
        # Recent cognitive states
        cursor.execute("SELECT COUNT(*) FROM cognitive_states WHERE timestamp > ?", (time.time() - 24 * 3600,))
        recent_states = cursor.fetchone()[0]
        
        conn.close()
        
        return {
            'total_rules': total_rules,
            'rules_by_type': rules_by_type,
            'average_confidence': avg_confidence,
            'top_rules': [
                {
                    'antecedent': row[0],
                    'consequent': row[1],
                    'support_count': row[2],
                    'confidence': row[3]
                }
                for row in top_rules
            ],
            'top_strategies': [
                {
                    'strategy_name': row[0],
                    'success_rate': row[1],
                    'avg_score': row[2],
                    'usage_count': row[3]
                }
                for row in top_strategies
            ],
            'recent_states_24h': recent_states
        }
    
    def get_symbolic_self_representation(self) -> Dict[str, Any]:
        """Get symbolic self-representation for display."""
        stats = self.get_cognitive_statistics()
        
        # Get recent rules
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT antecedent, consequent, confidence, support_count, rule_type
            FROM symbolic_rules 
            ORDER BY last_used DESC 
            LIMIT 10
        """)
        
        recent_rules = []
        for row in cursor.fetchall():
            recent_rules.append({
                'antecedent': row[0],
                'consequent': row[1],
                'confidence': row[2],
                'support_count': row[3],
                'rule_type': row[4]
            })
        
        conn.close()
        
        return {
            'statistics': stats,
            'recent_rules': recent_rules,
            'self_model': {
                'total_beliefs_tracked': len(stats.get('rules_by_type', {})),
                'learning_rate': stats.get('recent_states_24h', 0) / 24.0,
                'rule_confidence': stats.get('average_confidence', 0.0),
                'strategy_effectiveness': len(stats.get('top_strategies', []))
            }
        }
    
    def get_dominant_motives(self) -> Dict[str, float]:
        """
        Get current dominant motives driving cognition.
        
        Integrates with the drive system to get real-time motivational state.
        
        Returns:
            Dict mapping drive names to current strength values (0.0-1.0)
        """
        try:
            # Update drives from drive system
            self.update_active_drives()
            
            # Return copy of current active drives
            return self.active_drives.copy()
            
        except Exception as e:
            logger.error(f"Failed to get dominant motives: {e}")
            return self.active_drives.copy()
    
    def update_active_drives(self):
        """Update active drives from the drive system."""
        try:
            # Get drive system instance
            drive_system = get_drive_system(self.db_path)
            
            # Get current dominant drives
            current_drives = drive_system.get_current_dominant_drives()
            
            if current_drives:
                # Update active drives
                for drive, strength in current_drives.items():
                    if drive in self.active_drives:
                        self.active_drives[drive] = strength
                
                # Record in drive history
                drive_snapshot = {
                    "timestamp": time.time(),
                    "drives": current_drives.copy(),
                    "update_reason": "routine_update"
                }
                self.drive_history.append(drive_snapshot)
                
                # Keep only recent history (last 24 hours)
                cutoff_time = time.time() - 86400
                self.drive_history = [
                    snapshot for snapshot in self.drive_history
                    if snapshot["timestamp"] > cutoff_time
                ]
                
                self.last_drive_update = time.time()
                
                logger.debug(f"[SelfAwareModel] Updated active drives: {current_drives}")
            
        except Exception as e:
            logger.error(f"Failed to update active drives: {e}")
    
    def record_drive_influenced_decision(self, decision_type: str, dominant_drive: str, 
                                       outcome: str, details: Dict[str, Any] = None):
        """
        Record a decision that was influenced by drive signals.
        
        Args:
            decision_type: Type of decision (strategy_selection, goal_prioritization, etc.)
            dominant_drive: The drive that most influenced the decision
            outcome: The result of the decision
            details: Additional context about the decision
        """
        try:
            decision_record = {
                "timestamp": time.time(),
                "decision_type": decision_type,
                "dominant_drive": dominant_drive,
                "outcome": outcome,
                "drive_state": self.active_drives.copy(),
                "details": details or {}
            }
            
            # Store decision in drive history
            self.drive_history.append(decision_record)
            
            # Create symbolic rule from drive-decision pattern
            antecedent = f"DominantDrive({dominant_drive}) AND Context({decision_type})"
            consequent = f"Decision({outcome})"
            
            rule = self._create_rule(
                antecedent=antecedent,
                consequent=consequent,
                rule_type="preference"
            )
            
            self._add_or_update_rule(rule)
            
            logger.info(f"[SelfAwareModel] Recorded drive-influenced decision: {decision_type} -> {outcome} "
                       f"(driven by {dominant_drive})")
            
        except Exception as e:
            logger.error(f"Failed to record drive-influenced decision: {e}")


@dataclass
class IdentityTrait:
    """Represents a personality/identity trait derived from behavioral patterns."""
    trait_name: str  # e.g., "curious", "analytical", "empathetic"  
    strength: float  # 0.0 to 1.0 - how strongly this trait is expressed
    confidence: float  # 0.0 to 1.0 - confidence in this trait assessment
    evidence_count: int  # Number of supporting behavioral instances
    last_updated: float  # Timestamp of last update
    supporting_patterns: List[str]  # Behavioral patterns that support this trait
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod 
    def from_dict(cls, data: Dict[str, Any]) -> 'IdentityTrait':
        return cls(**data)


class SelfAwareModel(CognitiveSelfModel):
    """
    Extended cognitive self-model with emotional state and identity tracking.
    
    Phase 8 Features:
    - Emotional state integration with decision-making
    - Dynamic identity signature derived from behavioral patterns  
    - Mood-based strategy and goal modulation
    - Emotional tagging of memories and experiences
    """
    
    def __init__(self, db_path: str = "enhanced_memory.db"):
        super().__init__(db_path)
        
        # Import emotion model (lazy import to avoid circular dependencies)
        try:
            from .emotion_model import get_emotion_model
            self.emotion_model = get_emotion_model(db_path)
        except ImportError:
            logger.warning("EmotionModel not available, emotional features disabled")
            self.emotion_model = None
        
        # Identity tracking
        self.identity_traits: Dict[str, IdentityTrait] = {}
        self.identity_update_interval = 3600  # Update identity every hour
        self.last_identity_update = 0.0
        
        # Trait discovery configuration
        self.trait_confidence_threshold = 0.6
        self.min_evidence_count = 5
        self.trait_decay_rate = 0.98  # Slow decay per day
        
        # Behavioral pattern mappings for trait derivation
        self.trait_patterns = {
            "curious": {
                "drive_patterns": ["curiosity", "novelty"],
                "strategy_patterns": ["exploration_goal", "deep_exploration"],
                "emotion_patterns": ["curiosity", "excitement"]
            },
            "analytical": {
                "drive_patterns": ["coherence", "stability"],
                "strategy_patterns": ["belief_clarification", "fact_consolidation"],
                "emotion_patterns": ["concentration", "satisfaction"]
            },
            "empathetic": {
                "drive_patterns": ["conflict"],
                "strategy_patterns": ["conflict_resolution"],
                "emotion_patterns": ["compassion", "concern"]
            },
            "skeptical": {
                "strategy_patterns": ["belief_clarification", "contradiction_analysis"],
                "emotion_patterns": ["doubt", "caution"]
            },
            "optimistic": {
                "emotion_patterns": ["contentment", "satisfaction", "excitement"],
                "valence_bias": 0.3  # Tends toward positive emotions
            },
            "resilient": {
                "drive_patterns": ["stability", "coherence"],
                "recovery_patterns": ["bounce_back_from_contradiction"],
                "emotion_patterns": ["calm", "determination"]
            }
        }
        
        # Initialize identity database tables
        self._init_identity_tables()
        
        # Load existing identity traits
        self._load_identity_traits()
        
        logger.info("[SelfAwareModel] Initialized with emotional and identity tracking")
    
    def _init_identity_tables(self):
        """Initialize identity tracking database tables."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Identity traits table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS identity_traits (
                    trait_name TEXT PRIMARY KEY,
                    strength REAL NOT NULL,
                    confidence REAL NOT NULL,
                    evidence_count INTEGER NOT NULL,
                    last_updated REAL NOT NULL,
                    supporting_patterns TEXT,  -- JSON array
                    created_at REAL DEFAULT (strftime('%s', 'now'))
                )
            """)
            
            # Identity evolution history
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS identity_evolution (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    trait_name TEXT NOT NULL,
                    old_strength REAL,
                    new_strength REAL,
                    change_reason TEXT,
                    timestamp REAL NOT NULL,
                    supporting_evidence TEXT  -- JSON
                )
            """)
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Failed to initialize identity tables: {e}")
    
    def _load_identity_traits(self):
        """Load existing identity traits from database."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("SELECT * FROM identity_traits")
            rows = cursor.fetchall()
            conn.close()
            
            for row in rows:
                trait_name, strength, confidence, evidence_count, last_updated, supporting_patterns_json, _ = row
                
                supporting_patterns = json.loads(supporting_patterns_json) if supporting_patterns_json else []
                
                trait = IdentityTrait(
                    trait_name=trait_name,
                    strength=strength,
                    confidence=confidence,
                    evidence_count=evidence_count,
                    last_updated=last_updated,
                    supporting_patterns=supporting_patterns
                )
                
                self.identity_traits[trait_name] = trait
            
            logger.info(f"[SelfAwareModel] Loaded {len(self.identity_traits)} identity traits")
            
        except Exception as e:
            logger.error(f"Failed to load identity traits: {e}")
    
    def update_identity_from_behavior(self, behavior_type: str, behavior_data: Dict[str, Any]):
        """
        Update identity traits based on observed behavior patterns.
        
        Args:
            behavior_type: Type of behavior (drive_activation, strategy_selection, emotion_event, etc.)
            behavior_data: Data about the behavior instance
        """
        try:
            current_time = time.time()
            
            # Skip if updated recently (avoid over-updating)
            if current_time - self.last_identity_update < self.identity_update_interval:
                return
            
            # Analyze behavior for trait evidence
            trait_evidence = self._extract_trait_evidence(behavior_type, behavior_data)
            
            # Update traits based on evidence
            for trait_name, evidence_strength in trait_evidence.items():
                self._update_identity_trait(trait_name, evidence_strength, behavior_type)
            
            # Decay traits that haven't been reinforced
            self._decay_unreinforced_traits()
            
            # Store updated traits
            self._save_identity_traits()
            
            self.last_identity_update = current_time
            
        except Exception as e:
            logger.error(f"Failed to update identity from behavior: {e}")
    
    def _extract_trait_evidence(self, behavior_type: str, behavior_data: Dict[str, Any]) -> Dict[str, float]:
        """Extract evidence for identity traits from behavior data."""
        trait_evidence = {}
        
        try:
            # Analyze different types of behavioral evidence
            if behavior_type == "drive_activation":
                drive_name = behavior_data.get("drive_name", "")
                drive_strength = behavior_data.get("strength", 0.0)
                
                for trait_name, trait_config in self.trait_patterns.items():
                    if drive_name in trait_config.get("drive_patterns", []):
                        trait_evidence[trait_name] = drive_strength * 0.6
            
            elif behavior_type == "strategy_selection":
                strategy = behavior_data.get("strategy", "")
                success_rate = behavior_data.get("success_rate", 0.5)
                
                for trait_name, trait_config in self.trait_patterns.items():
                    if strategy in trait_config.get("strategy_patterns", []):
                        trait_evidence[trait_name] = success_rate * 0.8
            
            elif behavior_type == "emotion_event":
                emotion_type = behavior_data.get("emotion_type", "")
                emotion_strength = behavior_data.get("strength", 0.0)
                
                for trait_name, trait_config in self.trait_patterns.items():
                    if emotion_type in trait_config.get("emotion_patterns", []):
                        trait_evidence[trait_name] = emotion_strength * 0.5
            
            elif behavior_type == "decision_outcome":
                decision_type = behavior_data.get("decision_type", "")
                outcome_valence = behavior_data.get("outcome_valence", 0.0)
                
                # Positive outcomes reinforce associated traits
                if outcome_valence > 0.3:
                    dominant_drive = behavior_data.get("dominant_drive", "")
                    for trait_name, trait_config in self.trait_patterns.items():
                        if dominant_drive in trait_config.get("drive_patterns", []):
                            trait_evidence[trait_name] = outcome_valence * 0.7
            
            # Add emotional valence bias for relevant traits
            if self.emotion_model:
                current_mood = self.emotion_model.get_current_mood()
                valence = current_mood.get("valence", 0.0)
                
                for trait_name, trait_config in self.trait_patterns.items():
                    valence_bias = trait_config.get("valence_bias", 0.0)
                    if abs(valence - valence_bias) < 0.3:  # Similar valence
                        trait_evidence[trait_name] = trait_evidence.get(trait_name, 0.0) + 0.2
            
        except Exception as e:
            logger.error(f"Failed to extract trait evidence: {e}")
        
        return trait_evidence
    
    def _update_identity_trait(self, trait_name: str, evidence_strength: float, source: str):
        """Update a specific identity trait with new evidence."""
        try:
            current_time = time.time()
            
            if trait_name in self.identity_traits:
                trait = self.identity_traits[trait_name]
                
                # Update strength with moving average
                learning_rate = 0.1
                old_strength = trait.strength
                trait.strength = trait.strength * (1 - learning_rate) + evidence_strength * learning_rate
                trait.strength = max(0.0, min(1.0, trait.strength))  # Clamp to [0,1]
                
                # Update confidence based on evidence consistency  
                trait.evidence_count += 1
                trait.confidence = min(1.0, trait.evidence_count / 20.0)  # Asymptotic confidence
                
                # Add supporting pattern
                if source not in trait.supporting_patterns:
                    trait.supporting_patterns.append(source)
                    if len(trait.supporting_patterns) > 10:  # Limit list size
                        trait.supporting_patterns = trait.supporting_patterns[-10:]
                
                trait.last_updated = current_time
                
                # Log significant changes
                if abs(trait.strength - old_strength) > 0.1:
                    self._log_identity_evolution(trait_name, old_strength, trait.strength, source)
                
            else:
                # Create new trait
                if evidence_strength > 0.3:  # Only create if strong initial evidence
                    trait = IdentityTrait(
                        trait_name=trait_name,
                        strength=evidence_strength,
                        confidence=0.3,  # Low initial confidence
                        evidence_count=1,
                        last_updated=current_time,
                        supporting_patterns=[source]
                    )
                    self.identity_traits[trait_name] = trait
                    
                    logger.info(f"[SelfAwareModel] Discovered new identity trait: {trait_name} "
                               f"(strength={evidence_strength:.2f})")
            
        except Exception as e:
            logger.error(f"Failed to update identity trait {trait_name}: {e}")
    
    def _decay_unreinforced_traits(self):
        """Apply decay to traits that haven't been reinforced recently."""
        try:
            current_time = time.time()
            decay_threshold = 86400  # 24 hours
            
            traits_to_remove = []
            
            for trait_name, trait in self.identity_traits.items():
                time_since_update = current_time - trait.last_updated
                
                if time_since_update > decay_threshold:
                    # Apply exponential decay
                    days_since_update = time_since_update / 86400
                    decay_factor = self.trait_decay_rate ** days_since_update
                    
                    old_strength = trait.strength
                    trait.strength *= decay_factor
                    
                    # Remove traits that become too weak
                    if trait.strength < 0.1:
                        traits_to_remove.append(trait_name)
                    else:
                        # Log decay if significant
                        if old_strength - trait.strength > 0.1:
                            self._log_identity_evolution(
                                trait_name, old_strength, trait.strength, "trait_decay"
                            )
            
            # Remove weak traits
            for trait_name in traits_to_remove:
                del self.identity_traits[trait_name]
                logger.info(f"[SelfAwareModel] Removed weak identity trait: {trait_name}")
            
        except Exception as e:
            logger.error(f"Failed to decay unreinforced traits: {e}")
    
    def _log_identity_evolution(self, trait_name: str, old_strength: float, 
                              new_strength: float, reason: str):
        """Log identity trait evolution for analysis."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO identity_evolution 
                (trait_name, old_strength, new_strength, change_reason, timestamp)
                VALUES (?, ?, ?, ?, ?)
            """, (trait_name, old_strength, new_strength, reason, time.time()))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Failed to log identity evolution: {e}")
    
    def _save_identity_traits(self):
        """Save current identity traits to database."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            for trait_name, trait in self.identity_traits.items():
                cursor.execute("""
                    INSERT OR REPLACE INTO identity_traits
                    (trait_name, strength, confidence, evidence_count, last_updated, supporting_patterns)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    trait.trait_name,
                    trait.strength,
                    trait.confidence,
                    trait.evidence_count,
                    trait.last_updated,
                    json.dumps(trait.supporting_patterns)
                ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Failed to save identity traits: {e}")
    
    def get_identity_signature(self) -> str:
        """
        Generate a human-readable identity signature from current traits.
        
        Returns:
            String describing the system's identity/personality
        """
        try:
            # Filter traits by confidence and strength
            significant_traits = [
                (name, trait) for name, trait in self.identity_traits.items()
                if trait.confidence >= self.trait_confidence_threshold and trait.strength >= 0.4
            ]
            
            if not significant_traits:
                return "developing identity (insufficient data)"
            
            # Sort by combined strength and confidence
            significant_traits.sort(
                key=lambda x: x[1].strength * x[1].confidence, 
                reverse=True
            )
            
            # Take top traits
            top_traits = significant_traits[:4]
            
            # Format signature
            trait_descriptions = []
            for name, trait in top_traits:
                strength_desc = ""
                if trait.strength > 0.8:
                    strength_desc = "very "
                elif trait.strength > 0.6:
                    strength_desc = "quite "
                elif trait.strength > 0.4:
                    strength_desc = "somewhat "
                
                trait_descriptions.append(f"{strength_desc}{name}")
            
            # Add emotional context if available
            emotional_context = ""
            if self.emotion_model:
                mood_sig = self.emotion_model.get_mood_signature()
                if mood_sig and mood_sig != "neutral":
                    emotional_context = f", currently {mood_sig}"
            
            signature = ", ".join(trait_descriptions) + emotional_context
            
            return signature
            
        except Exception as e:
            logger.error(f"Failed to generate identity signature: {e}")
            return "unknown identity"
    
    def get_emotional_influence_on_decision(self, decision_context: Dict[str, Any]) -> Dict[str, float]:
        """
        Calculate how current emotional state should influence a decision.
        
        Args:
            decision_context: Context about the decision being made
            
        Returns:
            Dictionary with influence factors for different aspects
        """
        if not self.emotion_model:
            return {"risk_tolerance": 0.5, "exploration_bias": 0.5, "patience": 0.5}
        
        try:
            mood_info = self.emotion_model.get_current_mood()
            valence = mood_info["valence"]
            arousal = mood_info["arousal"]
            mood_label = mood_info["mood_label"]
            
            # Calculate influence factors based on emotional state
            influence = {}
            
            # Risk tolerance: Higher valence = more risk tolerance
            # Lower arousal = more calculated risk assessment
            influence["risk_tolerance"] = max(0.1, min(0.9, 
                0.5 + (valence * 0.3) - (arousal * 0.2)
            ))
            
            # Exploration bias: Curiosity and excitement increase exploration
            exploration_boost = 0.0
            if mood_label in ["curious", "excited"]:
                exploration_boost = 0.3
            elif mood_label in ["frustrated", "tense"]:
                exploration_boost = -0.2
            
            influence["exploration_bias"] = max(0.1, min(0.9, 
                0.5 + exploration_boost + (arousal * 0.1)
            ))
            
            # Patience: Calm and content moods increase patience
            patience_bonus = 0.0
            if mood_label in ["calm", "content"]:
                patience_bonus = 0.3
            elif mood_label in ["angry", "frustrated", "tense"]:
                patience_bonus = -0.4
            
            influence["patience"] = max(0.1, min(0.9, 
                0.5 + patience_bonus - (arousal * 0.2)
            ))
            
            # Novelty seeking: Curiosity and openness increase novelty seeking
            if mood_label in ["curious", "excited"]:
                influence["novelty_seeking"] = min(0.9, 0.7 + (arousal * 0.2))
            else:
                influence["novelty_seeking"] = max(0.1, 0.4 + (valence * 0.2))
            
            return influence
            
        except Exception as e:
            logger.error(f"Failed to calculate emotional influence: {e}")
            return {"risk_tolerance": 0.5, "exploration_bias": 0.5, "patience": 0.5}


def get_self_aware_model(db_path: str = "enhanced_memory.db") -> SelfAwareModel:
    """Get or create the global self-aware model instance."""
    if not hasattr(get_self_aware_model, '_instance'):
        get_self_aware_model._instance = SelfAwareModel(db_path)
    return get_self_aware_model._instance 