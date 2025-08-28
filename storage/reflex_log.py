#!/usr/bin/env python3
"""
ReflexCycle Logging System for MeRNSTA

Tracks drift-triggered self-repair chains as cohesive objects:
drift → goal → execution → result

Provides full observability of the autonomic cognitive repair process.
Now includes reflex scoring to evaluate cognitive effectiveness.
"""

import json
import time
import sqlite3
import logging
from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path


@dataclass
class ReflexScore:
    """
    Represents the cognitive effectiveness score of a reflex cycle.
    
    Evaluates how well a repair cycle improved the system's cognitive state.
    """
    cycle_id: str
    success: bool
    strategy: str
    token_id: Optional[int]
    
    # Cognitive state deltas
    coherence_delta: float = 0.0  # Change in token cluster coherence
    volatility_delta: float = 0.0  # Change in token volatility
    belief_consistency_delta: float = 0.0  # Change in contradiction rate
    
    # Computed score
    score: float = 0.0  # Normalized score from 0-1
    timestamp: float = 0.0
    
    # Metadata
    cluster_id: Optional[str] = None
    affected_facts: List[str] = None
    scoring_notes: str = ""
    
    def __post_init__(self):
        if self.affected_facts is None:
            self.affected_facts = []
        if self.timestamp == 0.0:
            self.timestamp = time.time()
    
    @property
    def score_icon(self) -> str:
        """Get score icon for display."""
        if self.score >= 0.8:
            return "🟢"
        elif self.score >= 0.6:
            return "🟡"
        elif self.score >= 0.4:
            return "🟠"
        else:
            return "🔴"
    
    @property
    def effectiveness_description(self) -> str:
        """Get human-readable effectiveness description."""
        improvements = []
        if self.coherence_delta > 0:
            improvements.append("📈 improved coherence")
        if self.volatility_delta < 0:
            improvements.append("📉 reduced volatility")
        if self.belief_consistency_delta > 0:
            improvements.append("📈 improved consistency")
        
        if not improvements:
            return "no significant improvements"
        
        return ", ".join(improvements)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ReflexScore':
        """Create from dictionary."""
        return cls(**data)


@dataclass
class ReflexCycle:
    """
    Represents a complete reflex cycle: drift detection → goal generation → execution → result.
    
    A reflex cycle tracks the full autonomic cognitive repair process from initial
    drift detection through goal execution and result logging.
    """
    # Core identifiers
    cycle_id: str
    token_id: Optional[int]
    drift_score: float
    strategy: str  # belief_clarification, cluster_reassessment, fact_consolidation
    goal_id: str
    agent_used: str
    
    # Timing
    start_time: float
    end_time: Optional[float] = None
    
    # Execution results
    success: bool = False
    actions_taken: List[str] = None
    memory_refs: List[str] = None  # References to memory entries created
    error_message: Optional[str] = None
    completion_notes: str = ""
    
    # Metadata
    priority: float = 0.0
    affected_facts: List[str] = None
    cluster_id: Optional[str] = None
    
    # Reflex scoring
    reflex_score: Optional[ReflexScore] = None
    
    def __post_init__(self):
        if self.actions_taken is None:
            self.actions_taken = []
        if self.memory_refs is None:
            self.memory_refs = []
        if self.affected_facts is None:
            self.affected_facts = []
    
    @property
    def duration(self) -> float:
        """Get execution duration in seconds."""
        if self.end_time and self.start_time:
            return self.end_time - self.start_time
        return 0.0
    
    @property
    def status_icon(self) -> str:
        """Get status icon for display."""
        return "✅" if self.success else "❌"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        if self.reflex_score:
            data['reflex_score'] = self.reflex_score.to_dict()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ReflexCycle':
        """Create from dictionary."""
        reflex_score_data = data.pop('reflex_score', None)
        cycle = cls(**data)
        if reflex_score_data:
            cycle.reflex_score = ReflexScore.from_dict(reflex_score_data)
        return cycle


class ReflexScorer:
    """
    Evaluates the cognitive effectiveness of reflex cycles.
    
    Calculates scores based on changes in:
    - Token cluster coherence
    - Token volatility
    - Belief consistency (contradiction rate)
    """
    
    def __init__(self, memory_system=None, token_graph=None):
        self.memory_system = memory_system
        self.token_graph = token_graph
        
        # Scoring weights
        self.coherence_weight = 0.4
        self.volatility_weight = 0.3
        self.consistency_weight = 0.3
        
        print(f"[ReflexScorer] Initialized with weights: coherence={self.coherence_weight}, "
              f"volatility={self.volatility_weight}, consistency={self.consistency_weight}")
    
    def calculate_reflex_score(self, cycle: ReflexCycle, 
                             before_state: Dict[str, Any], 
                             after_state: Dict[str, Any]) -> ReflexScore:
        """
        Calculate the cognitive effectiveness score for a reflex cycle.
        
        Args:
            cycle: The reflex cycle to score
            before_state: Cognitive state before the repair
            after_state: Cognitive state after the repair
            
        Returns:
            ReflexScore with calculated effectiveness metrics
        """
        try:
            # Calculate deltas
            coherence_delta = self._calculate_coherence_delta(before_state, after_state)
            volatility_delta = self._calculate_volatility_delta(before_state, after_state)
            belief_consistency_delta = self._calculate_consistency_delta(before_state, after_state)
            
            # Normalize deltas to 0-1 range
            normalized_coherence = self._normalize_delta(coherence_delta, -1.0, 1.0)
            normalized_volatility = self._normalize_delta(-volatility_delta, -1.0, 1.0)  # Negative because lower volatility is better
            normalized_consistency = self._normalize_delta(belief_consistency_delta, -1.0, 1.0)
            
            # Calculate weighted score
            score = (
                normalized_coherence * self.coherence_weight +
                normalized_volatility * self.volatility_weight +
                normalized_consistency * self.consistency_weight
            )
            
            # Ensure score is in 0-1 range
            score = max(0.0, min(1.0, score))
            
            # Create scoring notes
            scoring_notes = self._generate_scoring_notes(
                coherence_delta, volatility_delta, belief_consistency_delta, score
            )
            
            return ReflexScore(
                cycle_id=cycle.cycle_id,
                success=cycle.success,
                strategy=cycle.strategy,
                token_id=cycle.token_id,
                coherence_delta=coherence_delta,
                volatility_delta=volatility_delta,
                belief_consistency_delta=belief_consistency_delta,
                score=score,
                timestamp=time.time(),
                cluster_id=cycle.cluster_id,
                affected_facts=cycle.affected_facts,
                scoring_notes=scoring_notes
            )
            
        except Exception as e:
            logging.error(f"[ReflexScorer] Error calculating reflex score: {e}")
            # Return default score on error
            return ReflexScore(
                cycle_id=cycle.cycle_id,
                success=cycle.success,
                strategy=cycle.strategy,
                token_id=cycle.token_id,
                score=0.5,  # Neutral score
                scoring_notes=f"Scoring error: {str(e)}"
            )
    
    def _calculate_coherence_delta(self, before_state: Dict[str, Any], 
                                 after_state: Dict[str, Any]) -> float:
        """Calculate change in token cluster coherence."""
        try:
            before_coherence = before_state.get('cluster_coherence', 0.5)
            after_coherence = after_state.get('cluster_coherence', 0.5)
            return after_coherence - before_coherence
        except Exception:
            return 0.0
    
    def _calculate_volatility_delta(self, before_state: Dict[str, Any], 
                                  after_state: Dict[str, Any]) -> float:
        """Calculate change in token volatility."""
        try:
            before_volatility = before_state.get('token_volatility', 0.5)
            after_volatility = after_state.get('token_volatility', 0.5)
            return after_volatility - before_volatility
        except Exception:
            return 0.0
    
    def _calculate_consistency_delta(self, before_state: Dict[str, Any], 
                                   after_state: Dict[str, Any]) -> float:
        """Calculate change in belief consistency (contradiction rate)."""
        try:
            before_contradictions = before_state.get('contradiction_rate', 0.5)
            after_contradictions = after_state.get('contradiction_rate', 0.5)
            # Lower contradiction rate is better, so we invert the delta
            return before_contradictions - after_contradictions
        except Exception:
            return 0.0
    
    def _normalize_delta(self, delta: float, min_val: float, max_val: float) -> float:
        """Normalize a delta value to 0-1 range."""
        if max_val == min_val:
            return 0.5
        normalized = (delta - min_val) / (max_val - min_val)
        return max(0.0, min(1.0, normalized))
    
    def _generate_scoring_notes(self, coherence_delta: float, volatility_delta: float, 
                              consistency_delta: float, score: float) -> str:
        """Generate human-readable scoring notes."""
        notes = []
        
        if coherence_delta > 0.1:
            notes.append(f"Coherence improved by {coherence_delta:.2f}")
        elif coherence_delta < -0.1:
            notes.append(f"Coherence decreased by {abs(coherence_delta):.2f}")
        
        if volatility_delta < -0.1:
            notes.append(f"Volatility reduced by {abs(volatility_delta):.2f}")
        elif volatility_delta > 0.1:
            notes.append(f"Volatility increased by {volatility_delta:.2f}")
        
        if consistency_delta > 0.1:
            notes.append(f"Consistency improved by {consistency_delta:.2f}")
        elif consistency_delta < -0.1:
            notes.append(f"Consistency decreased by {abs(consistency_delta):.2f}")
        
        if not notes:
            notes.append("Minimal cognitive state changes")
        
        notes.append(f"Overall score: {score:.2f}")
        return "; ".join(notes)
    
    def capture_cognitive_state(self, token_id: Optional[int] = None, 
                              cluster_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Capture the current cognitive state for scoring.
        
        Args:
            token_id: Specific token to focus on
            cluster_id: Specific cluster to focus on
            
        Returns:
            Dictionary with cognitive state metrics
        """
        try:
            state = {
                'cluster_coherence': 0.5,
                'token_volatility': 0.5,
                'contradiction_rate': 0.5
            }
            
            # Get cluster coherence if token_graph available
            if self.token_graph and token_id:
                try:
                    cluster = self.token_graph.get_cluster_by_token(token_id)
                    if cluster:
                        state['cluster_coherence'] = cluster.coherence_score
                except Exception:
                    pass
            
            # Get token volatility
            if self.token_graph and token_id:
                try:
                    token_node = self.token_graph.graph.get(token_id)
                    if token_node:
                        state['token_volatility'] = token_node.volatility_score
                except Exception:
                    pass
            
            # Get contradiction rate
            if self.memory_system:
                try:
                    facts = self.memory_system.get_facts(limit=100)
                    if facts:
                        contradictions = sum(1 for f in facts if f.contradiction)
                        state['contradiction_rate'] = contradictions / len(facts)
                except Exception:
                    pass
            
            return state
            
        except Exception as e:
            logging.error(f"[ReflexScorer] Error capturing cognitive state: {e}")
            return {
                'cluster_coherence': 0.5,
                'token_volatility': 0.5,
                'contradiction_rate': 0.5
            }


class ReflexLogger:
    """
    Manages reflex cycle logging and retrieval.
    
    Features:
    - Persist reflex cycles to database
    - Index by token for quick lookup
    - Provide formatted output for CLI
    - Track execution statistics
    - Score reflex effectiveness
    """
    
    def __init__(self, db_path: str = "reflex_cycles.db"):
        self.db_path = db_path
        self._init_database()
        self.scorer = ReflexScorer()
        print(f"[ReflexLogger] Initialized with database: {db_path}")
    
    def _init_database(self):
        """Initialize the reflex cycles database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create reflex cycles table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS reflex_cycles (
                cycle_id TEXT PRIMARY KEY,
                token_id INTEGER,
                drift_score REAL,
                strategy TEXT,
                goal_id TEXT,
                agent_used TEXT,
                start_time REAL,
                end_time REAL,
                success BOOLEAN,
                actions_taken TEXT,
                memory_refs TEXT,
                error_message TEXT,
                completion_notes TEXT,
                priority REAL,
                affected_facts TEXT,
                cluster_id TEXT,
                created_at REAL DEFAULT (strftime('%s', 'now'))
            )
        """)
        
        # Create reflex scores table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS reflex_scores (
                cycle_id TEXT PRIMARY KEY,
                success BOOLEAN,
                strategy TEXT,
                token_id INTEGER,
                coherence_delta REAL,
                volatility_delta REAL,
                belief_consistency_delta REAL,
                score REAL,
                timestamp REAL,
                cluster_id TEXT,
                affected_facts TEXT,
                scoring_notes TEXT,
                created_at REAL DEFAULT (strftime('%s', 'now'))
            )
        """)
        
        # Create indexes for fast lookup
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_token_id ON reflex_cycles(token_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_strategy ON reflex_cycles(strategy)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_success ON reflex_cycles(success)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_start_time ON reflex_cycles(start_time)")
        
        # Indexes for reflex scores
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_score_cycle_id ON reflex_scores(cycle_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_score_strategy ON reflex_scores(strategy)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_score_score ON reflex_scores(score)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_score_timestamp ON reflex_scores(timestamp)")
        
        conn.commit()
        conn.close()
    
    def log_reflex_cycle(self, cycle: ReflexCycle, 
                        before_state: Dict[str, Any] = None,
                        after_state: Dict[str, Any] = None) -> bool:
        """
        Log a reflex cycle to the database and calculate its score.
        
        Args:
            cycle: The reflex cycle to log
            before_state: Cognitive state before the repair
            after_state: Cognitive state after the repair
            
        Returns:
            True if successfully logged, False otherwise
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Set end time if not set
            if not cycle.end_time:
                cycle.end_time = time.time()
            
            # Log the reflex cycle
            cursor.execute("""
                INSERT OR REPLACE INTO reflex_cycles 
                (cycle_id, token_id, drift_score, strategy, goal_id, agent_used,
                 start_time, end_time, success, actions_taken, memory_refs,
                 error_message, completion_notes, priority, affected_facts, cluster_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                cycle.cycle_id,
                cycle.token_id,
                cycle.drift_score,
                cycle.strategy,
                cycle.goal_id,
                cycle.agent_used,
                cycle.start_time,
                cycle.end_time,
                cycle.success,
                json.dumps(cycle.actions_taken),
                json.dumps(cycle.memory_refs),
                cycle.error_message,
                cycle.completion_notes,
                cycle.priority,
                json.dumps(cycle.affected_facts),
                cycle.cluster_id
            ))
            
            # Calculate and log reflex score
            if before_state and after_state:
                reflex_score = self.scorer.calculate_reflex_score(cycle, before_state, after_state)
                cycle.reflex_score = reflex_score
                
                cursor.execute("""
                    INSERT OR REPLACE INTO reflex_scores 
                    (cycle_id, success, strategy, token_id, coherence_delta, volatility_delta,
                     belief_consistency_delta, score, timestamp, cluster_id, affected_facts, scoring_notes)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    reflex_score.cycle_id,
                    reflex_score.success,
                    reflex_score.strategy,
                    reflex_score.token_id,
                    reflex_score.coherence_delta,
                    reflex_score.volatility_delta,
                    reflex_score.belief_consistency_delta,
                    reflex_score.score,
                    reflex_score.timestamp,
                    reflex_score.cluster_id,
                    json.dumps(reflex_score.affected_facts),
                    reflex_score.scoring_notes
                ))
                
                print(f"[ReflexLogger] Logged reflex cycle: {cycle.cycle_id} ({cycle.status_icon}) Score: {reflex_score.score:.2f}")
            else:
                print(f"[ReflexLogger] Logged reflex cycle: {cycle.cycle_id} ({cycle.status_icon}) - No scoring data")
            
            conn.commit()
            conn.close()
            return True
            
        except Exception as e:
            logging.error(f"[ReflexLogger] Error logging reflex cycle: {e}")
            return False
    
    def get_reflex_scores(self, limit: int = 20) -> List[ReflexScore]:
        """
        Get recent reflex scores.
        
        Args:
            limit: Maximum number of scores to return
            
        Returns:
            List of recent reflex scores
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT * FROM reflex_scores 
                ORDER BY timestamp DESC 
                LIMIT ?
            """, (limit,))
            
            scores = []
            for row in cursor.fetchall():
                score = self._row_to_score(row)
                scores.append(score)
            
            conn.close()
            return scores
            
        except Exception as e:
            logging.error(f"[ReflexLogger] Error getting reflex scores: {e}")
            return []
    
    def get_scores_by_strategy(self, strategy: str, limit: int = 10) -> List[ReflexScore]:
        """
        Get reflex scores by strategy.
        
        Args:
            strategy: Strategy to filter by
            limit: Maximum number of scores to return
            
        Returns:
            List of reflex scores for the strategy
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT * FROM reflex_scores 
                WHERE strategy = ? 
                ORDER BY timestamp DESC 
                LIMIT ?
            """, (strategy, limit))
            
            scores = []
            for row in cursor.fetchall():
                score = self._row_to_score(row)
                scores.append(score)
            
            conn.close()
            return scores
            
        except Exception as e:
            logging.error(f"[ReflexLogger] Error getting scores by strategy: {e}")
            return []
    
    def get_scores_by_token(self, token_id: int, limit: int = 10) -> List[ReflexScore]:
        """
        Get reflex scores for a specific token.
        
        Args:
            token_id: Token ID to search for
            limit: Maximum number of scores to return
            
        Returns:
            List of reflex scores for the token
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT * FROM reflex_scores 
                WHERE token_id = ? 
                ORDER BY timestamp DESC 
                LIMIT ?
            """, (token_id, limit))
            
            scores = []
            for row in cursor.fetchall():
                score = self._row_to_score(row)
                scores.append(score)
            
            conn.close()
            return scores
            
        except Exception as e:
            logging.error(f"[ReflexLogger] Error getting scores by token: {e}")
            return []
    
    def get_score_statistics(self) -> Dict[str, Any]:
        """
        Get statistics for reflex scores.
        
        Returns:
            Dictionary with score statistics
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Total scores
            cursor.execute("SELECT COUNT(*) FROM reflex_scores")
            total_scores = cursor.fetchone()[0]
            
            # Average score
            cursor.execute("SELECT AVG(score) FROM reflex_scores")
            avg_score = cursor.fetchone()[0] or 0.0
            
            # Best and worst scores
            cursor.execute("SELECT MAX(score), MIN(score) FROM reflex_scores")
            max_score, min_score = cursor.fetchone()
            max_score = max_score or 0.0
            min_score = min_score or 0.0
            
            # Strategy breakdown
            cursor.execute("""
                SELECT strategy, 
                       COUNT(*) as total,
                       AVG(score) as avg_score,
                       MAX(score) as max_score,
                       MIN(score) as min_score
                FROM reflex_scores 
                GROUP BY strategy
            """)
            strategy_stats = {}
            for row in cursor.fetchall():
                strategy, total, avg, max_val, min_val = row
                strategy_stats[strategy] = {
                    'total': total,
                    'avg_score': avg or 0.0,
                    'max_score': max_val or 0.0,
                    'min_score': min_val or 0.0
                }
            
            # Rolling average (last 10 scores)
            cursor.execute("""
                SELECT AVG(score) FROM (
                    SELECT score FROM reflex_scores 
                    ORDER BY timestamp DESC 
                    LIMIT 10
                )
            """)
            rolling_avg = cursor.fetchone()[0] or 0.0
            
            conn.close()
            
            return {
                'total_scores': total_scores,
                'average_score': avg_score,
                'max_score': max_score,
                'min_score': min_score,
                'rolling_average': rolling_avg,
                'strategy_statistics': strategy_stats
            }
            
        except Exception as e:
            logging.error(f"[ReflexLogger] Error getting score statistics: {e}")
            return {}
    
    def _row_to_score(self, row) -> ReflexScore:
        """Convert database row to ReflexScore object."""
        return ReflexScore(
            cycle_id=row[0],
            success=bool(row[1]),
            strategy=row[2],
            token_id=row[3],
            coherence_delta=row[4],
            volatility_delta=row[5],
            belief_consistency_delta=row[6],
            score=row[7],
            timestamp=row[8],
            cluster_id=row[9],
            affected_facts=json.loads(row[10]) if row[10] else [],
            scoring_notes=row[11]
        )
    
    def _row_to_cycle(self, row) -> ReflexCycle:
        """Convert database row to ReflexCycle object."""
        return ReflexCycle(
            cycle_id=row[0],
            token_id=row[1],
            drift_score=row[2],
            strategy=row[3],
            goal_id=row[4],
            agent_used=row[5],
            start_time=row[6],
            end_time=row[7],
            success=bool(row[8]),
            actions_taken=json.loads(row[9]) if row[9] else [],
            memory_refs=json.loads(row[10]) if row[10] else [],
            error_message=row[11],
            completion_notes=row[12],
            priority=row[13],
            affected_facts=json.loads(row[14]) if row[14] else [],
            cluster_id=row[15]
        )
    
    def format_cycle_display(self, cycle: ReflexCycle) -> str:
        """
        Format a reflex cycle for CLI display.
        
        Args:
            cycle: The reflex cycle to format
            
        Returns:
            Formatted string for display
        """
        start_time_str = datetime.fromtimestamp(cycle.start_time).strftime("%H:%M")
        duration_str = f"{cycle.duration:.2f}s" if cycle.duration > 0 else "N/A"
        
        lines = [
            f"🧠 Reflex Cycle for token {cycle.token_id} [{cycle.strategy}]",
            f"├─ Drift Score: {cycle.drift_score:.2f}",
            f"├─ Agent Used: {cycle.agent_used}",
            f"├─ Goal ID: {cycle.goal_id}",
            f"├─ Executed: {cycle.status_icon} at {start_time_str} (took {duration_str})",
            f"├─ Memory Entries: {len(cycle.memory_refs)} facts"
        ]
        
        # Add reflex score if available
        if cycle.reflex_score:
            lines.append(f"├─ 🧠 Score: {cycle.reflex_score.score:.2f} ({cycle.reflex_score.effectiveness_description})")
        
        # Add strategy optimization info if available
        if hasattr(cycle, 'strategy_metadata') and cycle.strategy_metadata:
            metadata = cycle.strategy_metadata
            if metadata.get('strategy_optimized', False):
                lines.append(f"├─ 🔧 Strategy Optimized: {metadata.get('original_strategy')} → {metadata.get('optimized_strategy')}")
                lines.append(f"├─ 📊 Reason: {metadata.get('strategy_reason', 'N/A')}")
                lines.append(f"├─ 🎯 Drift Type: {metadata.get('drift_type', 'N/A')}")
        
        if cycle.completion_notes:
            lines.append(f"└─ Notes: {cycle.completion_notes}")
        else:
            lines.append("└─ Notes: No completion notes")
        
        return "\n".join(lines)
    
    def format_cycles_summary(self, cycles: List[ReflexCycle]) -> str:
        """
        Format a list of reflex cycles for CLI display.
        
        Args:
            cycles: List of reflex cycles to format
            
        Returns:
            Formatted string for display
        """
        if not cycles:
            return "No reflex cycles found."
        
        lines = [f"🧠 Reflex Cycles Summary ({len(cycles)} cycles)"]
        lines.append("=" * 50)
        
        for i, cycle in enumerate(cycles, 1):
            lines.append(f"\n{i}. {self.format_cycle_display(cycle)}")
        
        return "\n".join(lines)

    def get_cycles_by_token(self, token_id: int, limit: int = 10) -> List[ReflexCycle]:
        """
        Get reflex cycles for a specific token.
        
        Args:
            token_id: Token ID to search for
            limit: Maximum number of cycles to return
            
        Returns:
            List of reflex cycles for the token
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT * FROM reflex_cycles 
                WHERE token_id = ? 
                ORDER BY start_time DESC 
                LIMIT ?
            """, (token_id, limit))
            
            cycles = []
            for row in cursor.fetchall():
                cycle = self._row_to_cycle(row)
                cycles.append(cycle)
            
            conn.close()
            return cycles
            
        except Exception as e:
            logging.error(f"[ReflexLogger] Error getting cycles by token: {e}")
            return []
    
    def get_recent_cycles(self, limit: int = 20) -> List[ReflexCycle]:
        """
        Get recent reflex cycles.
        
        Args:
            limit: Maximum number of cycles to return
            
        Returns:
            List of recent reflex cycles
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT * FROM reflex_cycles 
                ORDER BY start_time DESC 
                LIMIT ?
            """, (limit,))
            
            cycles = []
            for row in cursor.fetchall():
                cycle = self._row_to_cycle(row)
                cycles.append(cycle)
            
            conn.close()
            return cycles
            
        except Exception as e:
            logging.error(f"[ReflexLogger] Error getting recent cycles: {e}")
            return []
    
    def get_cycles_by_strategy(self, strategy: str, limit: int = 10) -> List[ReflexCycle]:
        """
        Get reflex cycles by repair strategy.
        
        Args:
            strategy: Strategy to filter by
            limit: Maximum number of cycles to return
            
        Returns:
            List of reflex cycles for the strategy
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT * FROM reflex_cycles 
                WHERE strategy = ? 
                ORDER BY start_time DESC 
                LIMIT ?
            """, (strategy, limit))
            
            cycles = []
            for row in cursor.fetchall():
                cycle = self._row_to_cycle(row)
                cycles.append(cycle)
            
            conn.close()
            return cycles
            
        except Exception as e:
            logging.error(f"[ReflexLogger] Error getting cycles by strategy: {e}")
            return []
    
    def get_execution_statistics(self) -> Dict[str, Any]:
        """
        Get execution statistics for reflex cycles.
        
        Returns:
            Dictionary with execution statistics
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Total cycles
            cursor.execute("SELECT COUNT(*) FROM reflex_cycles")
            total_cycles = cursor.fetchone()[0]
            
            # Successful cycles
            cursor.execute("SELECT COUNT(*) FROM reflex_cycles WHERE success = 1")
            successful_cycles = cursor.fetchone()[0]
            
            # Failed cycles
            cursor.execute("SELECT COUNT(*) FROM reflex_cycles WHERE success = 0")
            failed_cycles = cursor.fetchone()[0]
            
            # Average duration
            cursor.execute("""
                SELECT AVG(end_time - start_time) 
                FROM reflex_cycles 
                WHERE end_time IS NOT NULL AND start_time IS NOT NULL
            """)
            avg_duration = cursor.fetchone()[0] or 0.0
            
            # Strategy breakdown
            cursor.execute("""
                SELECT strategy, COUNT(*) 
                FROM reflex_cycles 
                GROUP BY strategy
            """)
            strategy_breakdown = dict(cursor.fetchall())
            
            # Success rate by strategy
            cursor.execute("""
                SELECT strategy, 
                       COUNT(*) as total,
                       SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as successful
                FROM reflex_cycles 
                GROUP BY strategy
            """)
            strategy_success = {}
            for row in cursor.fetchall():
                strategy, total, successful = row
                strategy_success[strategy] = {
                    'total': total,
                    'successful': successful,
                    'success_rate': successful / total if total > 0 else 0.0
                }
            
            conn.close()
            
            return {
                'total_cycles': total_cycles,
                'successful_cycles': successful_cycles,
                'failed_cycles': failed_cycles,
                'success_rate': successful_cycles / total_cycles if total_cycles > 0 else 0.0,
                'average_duration': avg_duration,
                'strategy_breakdown': strategy_breakdown,
                'strategy_success': strategy_success
            }
            
        except Exception as e:
            logging.error(f"[ReflexLogger] Error getting statistics: {e}")
            return {}

    def compare_strategies_logically(self, strategy1: str, strategy2: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Compare two strategies using symbolic logic reasoning.
        
        Args:
            strategy1: First strategy to compare
            strategy2: Second strategy to compare
            context: Optional context for comparison (drift type, cluster state, etc.)
            
        Returns:
            Dictionary with comparison results and logical reasoning
        """
        try:
            # Get historical performance data for both strategies
            scores1 = self.get_scores_by_strategy(strategy1, limit=50)
            scores2 = self.get_scores_by_strategy(strategy2, limit=50)
            
            # Calculate performance metrics
            def calculate_metrics(scores: List[ReflexScore]) -> Dict[str, float]:
                if not scores:
                    return {
                        'avg_score': 0.0,
                        'success_rate': 0.0,
                        'avg_coherence_delta': 0.0,
                        'avg_volatility_delta': 0.0,
                        'avg_consistency_delta': 0.0,
                        'sample_count': 0
                    }
                
                return {
                    'avg_score': sum(s.score for s in scores) / len(scores),
                    'success_rate': sum(1 for s in scores if s.success) / len(scores),
                    'avg_coherence_delta': sum(s.coherence_delta for s in scores) / len(scores),
                    'avg_volatility_delta': sum(s.volatility_delta for s in scores) / len(scores),
                    'avg_consistency_delta': sum(s.belief_consistency_delta for s in scores) / len(scores),
                    'sample_count': len(scores)
                }
            
            metrics1 = calculate_metrics(scores1)
            metrics2 = calculate_metrics(scores2)
            
            # Generate symbolic logic rules
            rules = []
            
            # Rule 1: Score-based dominance
            if metrics1['avg_score'] > metrics2['avg_score']:
                rules.append(f"Score({strategy1}) > Score({strategy2})")
            elif metrics2['avg_score'] > metrics1['avg_score']:
                rules.append(f"Score({strategy2}) > Score({strategy1})")
            
            # Rule 2: Success rate dominance
            if metrics1['success_rate'] > metrics2['success_rate']:
                rules.append(f"SuccessRate({strategy1}) > SuccessRate({strategy2})")
            elif metrics2['success_rate'] > metrics1['success_rate']:
                rules.append(f"SuccessRate({strategy2}) > SuccessRate({strategy1})")
            
            # Rule 3: Coherence improvement
            if metrics1['avg_coherence_delta'] > metrics2['avg_coherence_delta']:
                rules.append(f"CoherenceImprovement({strategy1}) > CoherenceImprovement({strategy2})")
            elif metrics2['avg_coherence_delta'] > metrics1['avg_coherence_delta']:
                rules.append(f"CoherenceImprovement({strategy2}) > CoherenceImprovement({strategy1})")
            
            # Rule 4: Volatility reduction
            if metrics1['avg_volatility_delta'] < metrics2['avg_volatility_delta']:
                rules.append(f"VolatilityReduction({strategy1}) > VolatilityReduction({strategy2})")
            elif metrics2['avg_volatility_delta'] < metrics1['avg_volatility_delta']:
                rules.append(f"VolatilityReduction({strategy2}) > VolatilityReduction({strategy1})")
            
            # Rule 5: Consistency improvement
            if metrics1['avg_consistency_delta'] > metrics2['avg_consistency_delta']:
                rules.append(f"ConsistencyImprovement({strategy1}) > ConsistencyImprovement({strategy2})")
            elif metrics2['avg_consistency_delta'] > metrics1['avg_consistency_delta']:
                rules.append(f"ConsistencyImprovement({strategy2}) > ConsistencyImprovement({strategy1})")
            
            # Context-specific rules
            if context:
                drift_type = context.get('drift_type', 'unknown')
                cluster_state = context.get('cluster_state', 'unknown')
                
                # Drift type specific rules
                if drift_type == 'contradiction':
                    if strategy1 == 'belief_clarification':
                        rules.append(f"DriftType({drift_type}) → PreferredStrategy({strategy1})")
                    elif strategy2 == 'belief_clarification':
                        rules.append(f"DriftType({drift_type}) → PreferredStrategy({strategy2})")
                
                elif drift_type == 'volatility':
                    if strategy1 == 'cluster_reassessment':
                        rules.append(f"DriftType({drift_type}) → PreferredStrategy({strategy1})")
                    elif strategy2 == 'cluster_reassessment':
                        rules.append(f"DriftType({drift_type}) → PreferredStrategy({strategy2})")
                
                elif drift_type == 'coherence':
                    if strategy1 == 'fact_consolidation':
                        rules.append(f"DriftType({drift_type}) → PreferredStrategy({strategy1})")
                    elif strategy2 == 'fact_consolidation':
                        rules.append(f"DriftType({drift_type}) → PreferredStrategy({strategy2})")
            
            # Determine overall dominance
            dominance_score1 = 0
            dominance_score2 = 0
            
            for rule in rules:
                if strategy1 in rule and '>' in rule:
                    dominance_score1 += 1
                elif strategy2 in rule and '>' in rule:
                    dominance_score2 += 1
            
            # Generate logical conclusion
            if dominance_score1 > dominance_score2:
                conclusion = f"Strategy({strategy1}) dominates Strategy({strategy2})"
                recommended = strategy1
            elif dominance_score2 > dominance_score1:
                conclusion = f"Strategy({strategy2}) dominates Strategy({strategy1})"
                recommended = strategy2
            else:
                conclusion = f"Strategy({strategy1}) ≈ Strategy({strategy2}) (equivalent)"
                recommended = None
            
            return {
                'strategy1': strategy1,
                'strategy2': strategy2,
                'metrics1': metrics1,
                'metrics2': metrics2,
                'symbolic_rules': rules,
                'dominance_score1': dominance_score1,
                'dominance_score2': dominance_score2,
                'conclusion': conclusion,
                'recommended_strategy': recommended,
                'context': context or {},
                'comparison_timestamp': time.time()
            }
            
        except Exception as e:
            logging.error(f"[ReflexLogger] Error comparing strategies: {e}")
            return {
                'error': str(e),
                'strategy1': strategy1,
                'strategy2': strategy2
            }


# Global reflex logger instance
_reflex_logger_instance = None


def get_reflex_logger() -> ReflexLogger:
    """Get or create the global reflex logger instance."""
    global _reflex_logger_instance
    
    if _reflex_logger_instance is None:
        _reflex_logger_instance = ReflexLogger()
    
    return _reflex_logger_instance


def log_reflex_cycle(cycle: ReflexCycle, before_state: Dict[str, Any] = None, 
                    after_state: Dict[str, Any] = None) -> bool:
    """Convenience function to log a reflex cycle with scoring."""
    return get_reflex_logger().log_reflex_cycle(cycle, before_state, after_state)


def get_cycles_by_token(token_id: int, limit: int = 10) -> List[ReflexCycle]:
    """Convenience function to get cycles by token."""
    return get_reflex_logger().get_cycles_by_token(token_id, limit)


def get_recent_cycles(limit: int = 20) -> List[ReflexCycle]:
    """Convenience function to get recent cycles."""
    return get_reflex_logger().get_recent_cycles(limit)


def get_reflex_scores(limit: int = 20) -> List[ReflexScore]:
    """Convenience function to get recent reflex scores."""
    return get_reflex_logger().get_reflex_scores(limit)


def compare_strategies_logically(strategy1: str, strategy2: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Compare two strategies using symbolic logic reasoning.
    
    Args:
        strategy1: First strategy to compare
        strategy2: Second strategy to compare
        context: Optional context for comparison (drift type, cluster state, etc.)
        
    Returns:
        Dictionary with comparison results and logical reasoning
    """
    return get_reflex_logger().compare_strategies_logically(strategy1, strategy2, context) 