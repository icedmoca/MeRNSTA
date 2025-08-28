#!/usr/bin/env python3
"""
Motivational Drive System for MeRNSTA Phase 7

Implements internal drives (curiosity, coherence, stability) that create autonomous
motivation for cognitive actions. Transforms MeRNSTA from reactive to purpose-driven cognition.
"""

import sqlite3
import logging
import json
import time
import math
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from collections import defaultdict
import numpy as np

from .enhanced_memory_model import EnhancedTripletFact
from .reflex_log import ReflexCycle, ReflexScore
# Goal class definition inline since agents.base doesn't have it
from dataclasses import dataclass
from typing import Optional, Dict, Any

@dataclass
class Goal:
    """Base goal class for autonomous goals."""
    goal_id: str
    description: str
    priority: float = 0.5
    strategy: str = "general"
    token_id: Optional[int] = None
    created_at: float = 0.0
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.created_at == 0.0:
            self.created_at = time.time()
        if self.metadata is None:
            self.metadata = {}

logger = logging.getLogger(__name__)


@dataclass
class DriveSignal:
    """Represents a motivational drive signal for a specific token/cluster."""
    token_id: int
    drive_type: str  # curiosity, coherence, stability, novelty, conflict
    strength: float  # 0.0 to 1.0
    timestamp: float
    cluster_id: Optional[str] = None
    source_facts: List[str] = None  # Fact IDs that contribute to this drive
    decay_rate: float = 0.1  # How quickly drive decays over time
    
    def __post_init__(self):
        if self.source_facts is None:
            self.source_facts = []
        if self.timestamp == 0.0:
            self.timestamp = time.time()
    
    def get_current_strength(self) -> float:
        """Calculate current drive strength accounting for temporal decay."""
        time_elapsed = time.time() - self.timestamp
        decayed_strength = self.strength * math.exp(-self.decay_rate * time_elapsed / 3600)  # Decay per hour
        return max(0.0, min(1.0, decayed_strength))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DriveSignal':
        """Create from dictionary."""
        return cls(**data)


@dataclass
class MotivationalGoal(Goal):
    """Extended Goal class with motivational metadata."""
    driving_motives: Dict[str, float] = None  # Which drives spawned this goal
    tension_score: float = 0.0  # Combined motivational pressure
    autonomy_level: float = 0.8  # How autonomous vs reactive this goal is
    
    def __post_init__(self):
        super().__post_init__()
        if self.driving_motives is None:
            self.driving_motives = {}


class MotivationalDriveSystem:
    """
    Core motivational drive system that monitors internal cognitive tensions
    and spawns autonomous goals based on drive signals.
    
    Drives implemented:
    - Curiosity: Seeks novel information and exploration
    - Coherence: Maintains logical consistency 
    - Stability: Preserves established beliefs
    - Novelty: Attracted to new patterns
    - Conflict: Resolves contradictions and tensions
    """
    
    def __init__(self, db_path: str = "memory.db"):
        self.db_path = db_path
        self.drive_weights = {
            "curiosity": 0.8,
            "coherence": 0.9, 
            "stability": 0.6,
            "novelty": 0.7,
            "conflict": 0.85
        }
        self.tension_threshold = 0.7  # Threshold for spawning autonomous goals
        self.drive_history: List[DriveSignal] = []
        self.active_drives: Dict[str, float] = {}
        
        # Initialize database tables
        self._init_drive_tables()
        
        logger.info(f"[MotivationalDriveSystem] Initialized with weights: {self.drive_weights}")
    
    def _init_drive_tables(self):
        """Initialize database tables for drive system."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Drive signals table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS drive_signals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    token_id INTEGER,
                    drive_type TEXT NOT NULL,
                    strength REAL NOT NULL,
                    timestamp REAL NOT NULL,
                    cluster_id TEXT,
                    source_facts TEXT,  -- JSON array of fact IDs
                    decay_rate REAL DEFAULT 0.1,
                    FOREIGN KEY (token_id) REFERENCES facts(token_id)
                )
            """)
            
            # Motivational goals table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS motivational_goals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    goal_id TEXT UNIQUE NOT NULL,
                    driving_motives TEXT NOT NULL,  -- JSON dict
                    tension_score REAL NOT NULL,
                    autonomy_level REAL NOT NULL,
                    created_at REAL NOT NULL,
                    status TEXT DEFAULT 'pending'
                )
            """)
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Failed to initialize drive tables: {e}")
    
    def evaluate_token_state(self, token_id: int) -> Dict[str, float]:
        """
        Score token on all drive dimensions: curiosity, coherence, volatility, novelty, conflict.
        
        Returns dict mapping drive type to strength (0.0-1.0).
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get token facts and metadata
            cursor.execute("""
                SELECT id, subject, predicate, object, volatility_score, 
                       contradiction, timestamp, access_count, confidence
                FROM facts WHERE token_id = ?
            """, (token_id,))
            facts = cursor.fetchall()
            
            if not facts:
                return {drive: 0.0 for drive in self.drive_weights.keys()}
            
            # Calculate each drive signal
            drives = {}
            
            # CURIOSITY: Low access count + recent timestamp = high curiosity
            avg_access = sum(fact[7] for fact in facts) / len(facts) if facts else 0
            recent_timestamp = max(fact[6] for fact in facts) if facts else 0
            time_since_access = time.time() - recent_timestamp
            curiosity = max(0.0, 1.0 - (avg_access / 10.0)) * math.exp(-time_since_access / 86400)  # Decay over days
            drives["curiosity"] = min(1.0, curiosity)
            
            # COHERENCE: High contradiction score = low coherence (inverted)
            avg_contradiction = sum(1 if fact[5] else 0 for fact in facts) / len(facts)
            coherence = 1.0 - avg_contradiction
            drives["coherence"] = max(0.0, coherence)
            
            # STABILITY: Low volatility = high stability (inverted)
            avg_volatility = sum(fact[4] or 0.0 for fact in facts) / len(facts)
            stability = 1.0 - avg_volatility
            drives["stability"] = max(0.0, stability)
            
            # NOVELTY: Recent facts with low confidence = high novelty
            recent_facts = [f for f in facts if (time.time() - f[6]) < 86400]  # Last 24 hours
            if recent_facts:
                avg_confidence = sum(f[8] for f in recent_facts) / len(recent_facts)
                novelty = 1.0 - avg_confidence
            else:
                novelty = 0.0
            drives["novelty"] = max(0.0, min(1.0, novelty))
            
            # CONFLICT: High contradiction count = high conflict drive
            contradiction_count = sum(1 if fact[5] else 0 for fact in facts)
            conflict = min(1.0, contradiction_count / max(1, len(facts)))
            drives["conflict"] = conflict
            
            conn.close()
            
            # Apply drive weights
            weighted_drives = {
                drive: strength * self.drive_weights.get(drive, 1.0)
                for drive, strength in drives.items()
            }
            
            logger.debug(f"[MotivationalDriveSystem] Token {token_id} drives: {weighted_drives}")
            return weighted_drives
            
        except Exception as e:
            logger.error(f"Failed to evaluate token state for {token_id}: {e}")
            return {drive: 0.0 for drive in self.drive_weights.keys()}
    
    def compute_drive_tension(self, cluster_id: str) -> float:
        """
        Returns combined tension from all drive signals in a cluster.
        
        Tension represents the motivational pressure to act on this cluster.
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get all tokens in this cluster
            cursor.execute("""
                SELECT DISTINCT token_id FROM facts 
                WHERE subject LIKE ? OR object LIKE ?
            """, (f"%{cluster_id}%", f"%{cluster_id}%"))
            token_ids = [row[0] for row in cursor.fetchall()]
            
            if not token_ids:
                return 0.0
            
            # Calculate drive tension for each token
            total_tension = 0.0
            tension_count = 0
            
            for token_id in token_ids:
                drives = self.evaluate_token_state(token_id)
                
                # Tension is weighted sum of conflicting drives
                # High conflict + low coherence + high curiosity = high tension
                tension = (
                    drives.get("conflict", 0.0) * 1.2 +
                    (1.0 - drives.get("coherence", 1.0)) * 1.0 +
                    drives.get("curiosity", 0.0) * 0.8 +
                    drives.get("novelty", 0.0) * 0.6 -
                    drives.get("stability", 0.0) * 0.4  # Stability reduces tension
                )
                
                total_tension += max(0.0, min(1.0, tension))
                tension_count += 1
            
            avg_tension = total_tension / tension_count if tension_count > 0 else 0.0
            
            conn.close()
            
            logger.debug(f"[MotivationalDriveSystem] Cluster {cluster_id} tension: {avg_tension}")
            return avg_tension
            
        except Exception as e:
            logger.error(f"Failed to compute drive tension for cluster {cluster_id}: {e}")
            return 0.0
    
    def rank_tokens_by_drive_pressure(self, limit: int = 50) -> List[Tuple[int, float]]:
        """
        Rank all tokens by motivational urgency (drive pressure).
        
        Returns list of (token_id, pressure_score) tuples sorted by pressure.
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get all active tokens
            cursor.execute("""
                SELECT DISTINCT token_id FROM facts 
                WHERE active = 1 
                ORDER BY timestamp DESC
                LIMIT ?
            """, (limit * 2,))  # Get more to filter later
            token_ids = [row[0] for row in cursor.fetchall()]
            
            # Calculate pressure for each token
            token_pressures = []
            
            for token_id in token_ids:
                drives = self.evaluate_token_state(token_id)
                
                # Drive pressure combines multiple factors
                pressure = (
                    drives.get("conflict", 0.0) * 1.5 +      # Conflicts need resolution
                    drives.get("curiosity", 0.0) * 1.2 +     # Curiosity drives exploration  
                    (1.0 - drives.get("coherence", 1.0)) * 1.3 +  # Incoherence needs fixing
                    drives.get("novelty", 0.0) * 0.9 -       # Novel info is interesting
                    drives.get("stability", 0.0) * 0.3       # Stable tokens need less attention
                )
                
                pressure = max(0.0, min(1.0, pressure))
                
                if pressure > 0.1:  # Only include tokens with meaningful pressure
                    token_pressures.append((token_id, pressure))
            
            # Sort by pressure (highest first)
            token_pressures.sort(key=lambda x: x[1], reverse=True)
            
            conn.close()
            
            result = token_pressures[:limit]
            logger.info(f"[MotivationalDriveSystem] Ranked {len(result)} tokens by drive pressure")
            return result
            
        except Exception as e:
            logger.error(f"Failed to rank tokens by drive pressure: {e}")
            return []
    
    def spawn_goal_if_needed(self, token_id: int, force: bool = False) -> Optional[MotivationalGoal]:
        """
        Create a proactive meta-goal if drive tension crosses threshold.
        
        Returns MotivationalGoal if spawned, None otherwise.
        """
        try:
            drives = self.evaluate_token_state(token_id)
            
            # Calculate overall tension
            tension = sum(drives.values()) / len(drives) if drives else 0.0
            
            # Check if we should spawn a goal
            if not force and tension < self.tension_threshold:
                return None
            
            # Determine goal type based on dominant drive
            dominant_drive = max(drives.items(), key=lambda x: x[1])
            drive_type, drive_strength = dominant_drive
            
            # Generate goal based on drive type
            goal_strategies = {
                "curiosity": "exploration_goal",
                "coherence": "coherence_repair", 
                "stability": "stabilization_goal",
                "novelty": "novelty_investigation",
                "conflict": "conflict_resolution"
            }
            
            strategy = goal_strategies.get(drive_type, "general_improvement")
            
            # Create motivational goal
            goal = MotivationalGoal(
                goal_id=f"drive_{drive_type}_{token_id}_{int(time.time())}",
                description=f"Autonomous {drive_type} goal for token {token_id}",
                priority=min(1.0, tension + 0.2),  # Higher than base tension
                strategy=strategy,
                token_id=token_id,
                driving_motives=drives.copy(),
                tension_score=tension,
                autonomy_level=0.8,  # High autonomy since self-generated
                metadata={
                    "spawn_reason": "drive_tension",
                    "dominant_drive": drive_type,
                    "drive_strength": drive_strength,
                    "all_drives": drives
                }
            )
            
            # Store in database
            self._store_motivational_goal(goal)
            
            logger.info(f"[MotivationalDriveSystem] Spawned {drive_type} goal for token {token_id} "
                       f"with tension {tension:.3f}")
            
            return goal
            
        except Exception as e:
            logger.error(f"Failed to spawn goal for token {token_id}: {e}")
            return None
    
    def _store_motivational_goal(self, goal: MotivationalGoal):
        """Store motivational goal in database."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT OR REPLACE INTO motivational_goals 
                (goal_id, driving_motives, tension_score, autonomy_level, created_at)
                VALUES (?, ?, ?, ?, ?)
            """, (
                goal.goal_id,
                json.dumps(goal.driving_motives),
                goal.tension_score,
                goal.autonomy_level,
                time.time()
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Failed to store motivational goal: {e}")
    
    def get_drive_history(self, hours: int = 24) -> List[DriveSignal]:
        """Get drive signal history for the last N hours."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cutoff_time = time.time() - (hours * 3600)
            
            cursor.execute("""
                SELECT token_id, drive_type, strength, timestamp, cluster_id, 
                       source_facts, decay_rate
                FROM drive_signals 
                WHERE timestamp > ?
                ORDER BY timestamp DESC
            """, (cutoff_time,))
            
            signals = []
            for row in cursor.fetchall():
                signal = DriveSignal(
                    token_id=row[0],
                    drive_type=row[1], 
                    strength=row[2],
                    timestamp=row[3],
                    cluster_id=row[4],
                    source_facts=json.loads(row[5]) if row[5] else [],
                    decay_rate=row[6]
                )
                signals.append(signal)
            
            conn.close()
            return signals
            
        except Exception as e:
            logger.error(f"Failed to get drive history: {e}")
            return []
    
    def update_drive_weights(self, new_weights: Dict[str, float]):
        """Update drive weights based on learning or user preferences."""
        for drive, weight in new_weights.items():
            if drive in self.drive_weights:
                self.drive_weights[drive] = max(0.0, min(1.0, weight))
        
        logger.info(f"[MotivationalDriveSystem] Updated drive weights: {self.drive_weights}")
    
    def get_current_dominant_drives(self) -> Dict[str, float]:
        """Get currently dominant drive signals across all tokens."""
        try:
            recent_tokens = self.rank_tokens_by_drive_pressure(20)
            
            if not recent_tokens:
                return {}
            
            # Aggregate drives across top tokens
            drive_totals = defaultdict(float)
            drive_counts = defaultdict(int)
            
            for token_id, pressure in recent_tokens[:10]:  # Top 10 tokens
                drives = self.evaluate_token_state(token_id)
                for drive, strength in drives.items():
                    drive_totals[drive] += strength * pressure  # Weight by pressure
                    drive_counts[drive] += 1
            
            # Calculate weighted averages
            dominant_drives = {}
            for drive in drive_totals:
                if drive_counts[drive] > 0:
                    dominant_drives[drive] = drive_totals[drive] / drive_counts[drive]
            
            # Update active drives cache
            self.active_drives = dominant_drives.copy()
            
            return dominant_drives
            
        except Exception as e:
            logger.error(f"Failed to get dominant drives: {e}")
            return {}
    
    def analyze_drive_trends(self) -> Dict[str, Any]:
        """Analyze trends in drive signals over time."""
        history = self.get_drive_history(72)  # Last 3 days
        
        if not history:
            return {"error": "No drive history available"}
        
        # Group by drive type and time windows
        drive_trends = defaultdict(list)
        for signal in history:
            drive_trends[signal.drive_type].append({
                "timestamp": signal.timestamp,
                "strength": signal.get_current_strength(),
                "token_id": signal.token_id
            })
        
        # Calculate trend statistics
        analysis = {}
        for drive_type, signals in drive_trends.items():
            if len(signals) < 2:
                continue
                
            strengths = [s["strength"] for s in signals]
            timestamps = [s["timestamp"] for s in signals]
            
            analysis[drive_type] = {
                "avg_strength": np.mean(strengths),
                "trend_direction": "increasing" if strengths[-1] > strengths[0] else "decreasing",
                "peak_strength": max(strengths),
                "signal_count": len(signals),
                "time_span_hours": (max(timestamps) - min(timestamps)) / 3600
            }
        
        return analysis


def get_drive_system(db_path: str = "memory.db") -> MotivationalDriveSystem:
    """Get or create the global drive system instance."""
    if not hasattr(get_drive_system, '_instance'):
        get_drive_system._instance = MotivationalDriveSystem(db_path)
    return get_drive_system._instance