#!/usr/bin/env python3
"""
Autonomous Intent Modeler for MeRNSTA Phase 7

Models and evolves user/system intent from memory patterns and drive signals.
Generates higher-level meta-goals and adjusts priorities based on motivational reasoning.
"""

import sqlite3
import logging
import json
import time
import re
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from collections import defaultdict, Counter
import numpy as np

from .base import Agent

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
from storage.drive_system import MotivationalDriveSystem, MotivationalGoal, DriveSignal
from storage.enhanced_memory_model import EnhancedTripletFact
from storage.reflex_log import ReflexCycle

logger = logging.getLogger(__name__)


@dataclass
class IntentPattern:
    """Represents an abstracted intent pattern discovered from memory."""
    pattern_id: str
    description: str
    confidence: float
    supporting_evidence: List[str]  # Fact IDs or token patterns
    intent_category: str  # exploration, organization, clarification, etc.
    temporal_frequency: float  # How often this intent appears
    context_triggers: List[str]  # What contexts trigger this intent
    created_at: float
    last_seen: float
    
    def __post_init__(self):
        if self.created_at == 0.0:
            self.created_at = time.time()
        if self.last_seen == 0.0:
            self.last_seen = time.time()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)
    
    @classmethod 
    def from_dict(cls, data: Dict[str, Any]) -> 'IntentPattern':
        """Create from dictionary."""
        return cls(**data)


@dataclass
class EvolvedGoal(MotivationalGoal):
    """Goal evolved from intent patterns and drive signals."""
    source_intents: List[str] = field(default_factory=list)  # Intent pattern IDs that spawned this goal
    abstraction_level: int = 0  # How many levels of abstraction from base facts
    goal_lineage: List[str] = field(default_factory=list)  # Chain of goals that led to this one
    adaptive_priority: float = 0.5  # Priority that adapts based on context
    
    def __post_init__(self):
        super().__post_init__()
        if not hasattr(self, 'source_intents'):
            self.source_intents = []
        if not hasattr(self, 'goal_lineage'):
            self.goal_lineage = []


class AutonomousIntentModeler(Agent):
    """
    Models user/system intent from memory patterns and generates autonomous goals.
    
    Capabilities:
    - Abstract intent patterns from memory and token history
    - Evolve higher-level meta-goals from underlying drives
    - Adjust goal priorities based on motivational weights
    - Summarize current dominant motives driving cognition
    """
    
    def __init__(self, memory_system=None, drive_system: MotivationalDriveSystem = None):
        super().__init__(memory_system)
        self.drive_system = drive_system
        self.db_path = drive_system.db_path if drive_system else "memory.db"
        
        # Intent modeling configuration
        self.intent_confidence_threshold = 0.6
        self.max_abstraction_levels = 4
        self.pattern_discovery_window = 168  # Hours to look back for patterns
        
        # Intent pattern storage
        self.discovered_patterns: Dict[str, IntentPattern] = {}
        self.evolved_goals: List[EvolvedGoal] = []
        
        # Initialize database tables
        self._init_intent_tables()
        
        logger.info(f"[AutonomousIntentModeler] Initialized with drive system: {drive_system is not None}")
    
    def get_agent_instructions(self) -> str:
        """Get agent instructions for intent modeling."""
        return """
You are the AutonomousIntentModeler. Your role is to:
1. Model user/system intent from memory patterns and token history
2. Generate autonomous goals from underlying drives
3. Adjust goal priorities based on motivational weights
4. Summarize current dominant motives driving cognition

You analyze patterns in user behavior, memory formation, and system interactions to
understand deeper intentions and generate meaningful autonomous goals.
        """.strip()
    
    def respond(self, input_text: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Generate a response focused on intent modeling and goal generation."""
        try:
            # Analyze intent patterns from current context and memory
            intent_analysis = self.model_intent_from_patterns()
            
            # Generate evolved goals if patterns are discovered
            evolved_goals = []
            if intent_analysis.get("patterns"):
                evolved_goals = self.evolve_goals(intent_analysis)
            
            # Summarize current motivational state
            motivational_summary = self.summarize_current_motives()
            
            response = f"**Intent Analysis Results:**\n\n"
            
            if intent_analysis.get("patterns"):
                response += f"🎯 **Discovered {len(intent_analysis['patterns'])} Intent Patterns:**\n"
                for pattern in intent_analysis["patterns"][:3]:  # Show top 3
                    if isinstance(pattern, dict):
                        response += f"- {pattern.get('description', 'Unknown pattern')} (confidence: {pattern.get('confidence', 0):.2f})\n"
                    else:
                        response += f"- {pattern.description} (confidence: {pattern.confidence:.2f})\n"
                response += "\n"
            
            if evolved_goals:
                response += f"🚀 **Generated {len(evolved_goals)} Evolved Goals:**\n"
                for goal in evolved_goals[:3]:  # Show top 3
                    response += f"- {goal.description} (priority: {goal.priority:.2f})\n"
                response += "\n"
            
            if motivational_summary.get("dominant_drives"):
                response += f"💪 **Current Dominant Drives:**\n"
                for drive, strength in motivational_summary["dominant_drives"].items():
                    response += f"- {drive}: {strength:.2f}\n"
                response += "\n"
            
            response += f"📊 **Analysis Metrics:**\n"
            response += f"- Facts analyzed: {intent_analysis.get('facts_analyzed', 0)}\n"
            response += f"- Patterns discovered: {intent_analysis.get('patterns_discovered', 0)}\n"
            response += f"- Goals evolved: {len(evolved_goals)}\n"
            
            return response
            
        except Exception as e:
            logger.error(f"[AutonomousIntentModeler] Error in respond: {e}")
            return f"Intent modeling encountered an error: {str(e)}"
    
    def _init_intent_tables(self):
        """Initialize database tables for intent modeling."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Intent patterns table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS intent_patterns (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    pattern_id TEXT UNIQUE NOT NULL,
                    description TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    supporting_evidence TEXT,  -- JSON array
                    intent_category TEXT NOT NULL,
                    temporal_frequency REAL DEFAULT 0.0,
                    context_triggers TEXT,  -- JSON array
                    created_at REAL NOT NULL,
                    last_seen REAL NOT NULL
                )
            """)
            
            # Evolved goals table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS evolved_goals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    goal_id TEXT UNIQUE NOT NULL,
                    source_intents TEXT,  -- JSON array of intent pattern IDs
                    abstraction_level INTEGER DEFAULT 1,
                    goal_lineage TEXT,  -- JSON array of parent goal IDs
                    adaptive_priority REAL NOT NULL,
                    created_at REAL NOT NULL,
                    status TEXT DEFAULT 'pending'
                )
            """)
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Failed to initialize intent tables: {e}")
    
    def model_intent_from_patterns(self, token_history: List[str], 
                                  lookback_hours: int = 24) -> Dict[str, Any]:
        """
        Abstract user/system intent from memory and token drift patterns.
        
        Args:
            token_history: List of recent token sequences or fact descriptions
            lookback_hours: How far back to analyze patterns
            
        Returns:
            Dict containing discovered intent patterns and confidence scores
        """
        try:
            # Get recent memory patterns
            cutoff_time = time.time() - (lookback_hours * 3600)
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get recent facts and their contexts
            cursor.execute("""
                SELECT id, subject, predicate, object, timestamp, 
                       confidence, volatility_score, token_id, user_profile_id
                FROM facts 
                WHERE timestamp > ? AND active = 1
                ORDER BY timestamp DESC
                LIMIT 200
            """, (cutoff_time,))
            recent_facts = cursor.fetchall()
            
            # Get recent reflex cycles for behavioral patterns
            cursor.execute("""
                SELECT cycle_id, strategy, success, goal_description, timestamp
                FROM reflex_cycles 
                WHERE timestamp > ?
                ORDER BY timestamp DESC
                LIMIT 50
            """, (cutoff_time,))
            reflex_patterns = cursor.fetchall()
            
            conn.close()
            
            # Analyze patterns in the data
            intent_analysis = self._analyze_intent_patterns(recent_facts, reflex_patterns, token_history)
            
            # Store discovered patterns
            for pattern in intent_analysis.get("patterns", []):
                self._store_intent_pattern(pattern)
            
            return intent_analysis
            
        except Exception as e:
            logger.error(f"Failed to model intent from patterns: {e}")
            return {"error": str(e), "patterns": []}
    
    def _analyze_intent_patterns(self, facts: List[Tuple], reflex_patterns: List[Tuple], 
                                token_history: List[str]) -> Dict[str, Any]:
        """Analyze memory data to discover intent patterns."""
        
        patterns = []
        
        # Pattern 1: Subject clustering (what domains user focuses on)
        subject_counts = Counter()
        predicate_counts = Counter()
        
        for fact in facts:
            subject, predicate, obj = fact[1], fact[2], fact[3]
            subject_counts[subject] += 1
            predicate_counts[predicate] += 1
        
        # Find dominant subjects (exploration intent)
        if subject_counts:
            top_subject, count = subject_counts.most_common(1)[0]
            if count >= 3:  # Threshold for significant pattern
                patterns.append(IntentPattern(
                    pattern_id=f"exploration_{top_subject}_{int(time.time())}",
                    description=f"Strong interest in exploring {top_subject}",
                    confidence=min(1.0, count / 10.0),
                    supporting_evidence=[f"Subject mentioned {count} times"],
                    intent_category="exploration",
                    temporal_frequency=count / max(1, len(facts)),
                    context_triggers=[top_subject]
                ))
        
        # Pattern 2: Strategy preferences from reflex cycles
        if reflex_patterns:
            strategy_success = defaultdict(list)
            for cycle in reflex_patterns:
                strategy, success = cycle[1], cycle[2]
                strategy_success[strategy].append(success)
            
            # Find consistently successful strategies (optimization intent)
            for strategy, successes in strategy_success.items():
                if len(successes) >= 2:
                    success_rate = sum(successes) / len(successes)
                    if success_rate >= 0.7:
                        patterns.append(IntentPattern(
                            pattern_id=f"strategy_pref_{strategy}_{int(time.time())}",
                            description=f"Preference for {strategy} strategy (success rate: {success_rate:.2f})",
                            confidence=success_rate,
                            supporting_evidence=[f"{len(successes)} cycles with {success_rate:.1%} success"],
                            intent_category="optimization",
                            temporal_frequency=len(successes) / len(reflex_patterns),
                            context_triggers=[strategy]
                        ))
        
        # Pattern 3: Volatility response patterns (stabilization intent)
        high_volatility_facts = [f for f in facts if f[6] and f[6] > 0.7]  # volatility_score > 0.7
        if len(high_volatility_facts) >= 3:
            patterns.append(IntentPattern(
                pattern_id=f"stabilization_{int(time.time())}",
                description="Need to stabilize high-volatility information",
                confidence=min(1.0, len(high_volatility_facts) / 10.0),
                supporting_evidence=[f"{len(high_volatility_facts)} high-volatility facts detected"],
                intent_category="stabilization",
                temporal_frequency=len(high_volatility_facts) / len(facts),
                context_triggers=["high_volatility"]
            ))
        
        # Pattern 4: Knowledge gap detection (curiosity intent)
        low_confidence_facts = [f for f in facts if f[5] < 0.5]  # confidence < 0.5
        unique_subjects = set(f[1] for f in low_confidence_facts)
        if len(unique_subjects) >= 3:
            patterns.append(IntentPattern(
                pattern_id=f"curiosity_{int(time.time())}",
                description=f"Curiosity about {len(unique_subjects)} uncertain topics",
                confidence=min(1.0, len(unique_subjects) / 5.0),
                supporting_evidence=[f"Low confidence in {len(unique_subjects)} subject areas"],
                intent_category="curiosity",
                temporal_frequency=len(low_confidence_facts) / len(facts),
                context_triggers=list(unique_subjects)[:3]  # Sample triggers
            ))
        
        # Pattern 5: Token sequence analysis
        if token_history:
            # Look for repetitive patterns that might indicate persistent interests
            token_text = " ".join(token_history)
            
            # Simple keyword extraction for dominant themes
            words = re.findall(r'\b\w+\b', token_text.lower())
            word_counts = Counter(words)
            
            # Filter out common words and find significant themes
            significant_words = [(word, count) for word, count in word_counts.most_common(10) 
                               if len(word) > 3 and count >= 2]
            
            if significant_words:
                top_word, word_count = significant_words[0]
                patterns.append(IntentPattern(
                    pattern_id=f"theme_{top_word}_{int(time.time())}",
                    description=f"Persistent interest in theme: {top_word}",
                    confidence=min(1.0, word_count / len(words)),
                    supporting_evidence=[f"'{top_word}' appeared {word_count} times in recent tokens"],
                    intent_category="thematic_focus",
                    temporal_frequency=word_count / len(words),
                    context_triggers=[top_word]
                ))
        
        return {
            "patterns": patterns,
            "analysis_timestamp": time.time(),
            "facts_analyzed": len(facts),
            "reflex_cycles_analyzed": len(reflex_patterns),
            "patterns_discovered": len(patterns)
        }
    
    def _store_intent_pattern(self, pattern: IntentPattern):
        """Store intent pattern in database."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT OR REPLACE INTO intent_patterns
                (pattern_id, description, confidence, supporting_evidence, 
                 intent_category, temporal_frequency, context_triggers, created_at, last_seen)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                pattern.pattern_id,
                pattern.description,
                pattern.confidence,
                json.dumps(pattern.supporting_evidence),
                pattern.intent_category,
                pattern.temporal_frequency,
                json.dumps(pattern.context_triggers),
                pattern.created_at,
                pattern.last_seen
            ))
            
            conn.commit()
            conn.close()
            
            # Cache in memory
            self.discovered_patterns[pattern.pattern_id] = pattern
            
        except Exception as e:
            logger.error(f"Failed to store intent pattern: {e}")
    
    def evolve_goals(self, base_intent: Dict[str, Any]) -> List[EvolvedGoal]:
        """
        Generate higher-level meta-goals from underlying drives and intent patterns.
        
        Args:
            base_intent: Intent analysis from model_intent_from_patterns()
            
        Returns:
            List of evolved goals with higher abstraction levels
        """
        try:
            evolved_goals = []
            patterns = base_intent.get("patterns", [])
            
            if not patterns:
                logger.info("[AutonomousIntentModeler] No patterns available for goal evolution")
                return []
            
            # Get current drive signals to inform goal evolution
            dominant_drives = self.drive_system.get_current_dominant_drives() if self.drive_system else {}
            
            # Evolve goals based on intent patterns and drives
            for pattern in patterns:
                if isinstance(pattern, dict):
                    pattern = IntentPattern.from_dict(pattern)
                
                if pattern.confidence < self.intent_confidence_threshold:
                    continue
                
                # Generate evolved goals based on pattern type
                evolved_goal = self._evolve_goal_from_pattern(pattern, dominant_drives)
                if evolved_goal:
                    evolved_goals.append(evolved_goal)
            
            # Create meta-goals that combine multiple patterns
            meta_goals = self._create_meta_goals(patterns, dominant_drives)
            evolved_goals.extend(meta_goals)
            
            # Store evolved goals
            for goal in evolved_goals:
                self._store_evolved_goal(goal)
            
            self.evolved_goals.extend(evolved_goals)
            
            logger.info(f"[AutonomousIntentModeler] Evolved {len(evolved_goals)} goals from {len(patterns)} patterns")
            return evolved_goals
            
        except Exception as e:
            logger.error(f"Failed to evolve goals: {e}")
            return []
    
    def _evolve_goal_from_pattern(self, pattern: IntentPattern, 
                                 drives: Dict[str, float]) -> Optional[EvolvedGoal]:
        """Evolve a single goal from an intent pattern."""
        
        # Map intent categories to goal strategies
        strategy_mapping = {
            "exploration": "deep_exploration",
            "optimization": "strategy_refinement", 
            "stabilization": "volatility_reduction",
            "curiosity": "knowledge_acquisition",
            "thematic_focus": "thematic_investigation"
        }
        
        strategy = strategy_mapping.get(pattern.intent_category, "general_improvement")
        
        # Calculate adaptive priority based on pattern strength, drives, and emotions
        drive_boost = 0.0
        if pattern.intent_category == "curiosity" and "curiosity" in drives:
            drive_boost = drives["curiosity"] * 0.3
        elif pattern.intent_category == "stabilization" and "stability" in drives:
            drive_boost = drives["stability"] * 0.3
        elif pattern.intent_category == "optimization" and "coherence" in drives:
            drive_boost = drives["coherence"] * 0.3
        
        # Phase 8: Add emotional and identity influence to goal prioritization
        emotional_boost = self._calculate_emotional_goal_boost(pattern, drives)
        identity_alignment = self._calculate_identity_alignment(pattern, drives)
        
        adaptive_priority = min(1.0, pattern.confidence + drive_boost + emotional_boost + identity_alignment)
        
        # Create evolved goal
        evolved_goal = EvolvedGoal(
            goal_id=f"evolved_{pattern.pattern_id}",
            description=f"Evolved goal: {pattern.description}",
            priority=adaptive_priority,
            strategy=strategy,
            driving_motives=drives.copy(),
            tension_score=sum(drives.values()) / len(drives) if drives else 0.5,
            autonomy_level=0.9,  # High autonomy for evolved goals
            source_intents=[pattern.pattern_id],
            abstraction_level=2,  # Second level abstraction
            adaptive_priority=adaptive_priority,
            metadata={
                "source_pattern": pattern.to_dict(),
                "evolution_reason": "intent_pattern_based",
                "drive_influence": drive_boost,
                "emotional_boost": emotional_boost,
                "identity_alignment": identity_alignment
            }
        )
        
        return evolved_goal
    
    def _create_meta_goals(self, patterns: List[IntentPattern], 
                          drives: Dict[str, float]) -> List[EvolvedGoal]:
        """Create higher-level meta-goals that combine multiple patterns."""
        meta_goals = []
        
        # Group patterns by category
        pattern_groups = defaultdict(list)
        for pattern in patterns:
            if isinstance(pattern, dict):
                pattern = IntentPattern.from_dict(pattern)
            pattern_groups[pattern.intent_category].append(pattern)
        
        # Create meta-goals for categories with multiple patterns
        for category, category_patterns in pattern_groups.items():
            if len(category_patterns) >= 2:
                # Combine confidence scores
                combined_confidence = min(1.0, sum(p.confidence for p in category_patterns) / 2.0)
                
                # Create meta-goal
                meta_goal = EvolvedGoal(
                    goal_id=f"meta_{category}_{int(time.time())}",
                    description=f"Meta-goal: Comprehensive {category} across {len(category_patterns)} areas",
                    priority=combined_confidence,
                    strategy=f"meta_{category}",
                    driving_motives=drives.copy(),
                    tension_score=combined_confidence,
                    autonomy_level=0.95,  # Very high autonomy
                    source_intents=[p.pattern_id for p in category_patterns],
                    abstraction_level=3,  # Third level abstraction
                    adaptive_priority=combined_confidence,
                    metadata={
                        "meta_goal_type": category,
                        "combined_patterns": len(category_patterns),
                        "abstraction_reason": "pattern_synthesis"
                    }
                )
                
                meta_goals.append(meta_goal)
        
        return meta_goals
    
    def _store_evolved_goal(self, goal: EvolvedGoal):
        """Store evolved goal in database."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT OR REPLACE INTO evolved_goals
                (goal_id, source_intents, abstraction_level, goal_lineage, 
                 adaptive_priority, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                goal.goal_id,
                json.dumps(goal.source_intents),
                goal.abstraction_level,
                json.dumps(goal.goal_lineage),
                goal.adaptive_priority,
                time.time()
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Failed to store evolved goal: {e}")
    
    def adjust_priorities(self, goal_list: List[Goal]) -> List[Goal]:
        """
        Reorder goals based on motivational weights and symbolic reasoning.
        
        Integrates drive signals with goal priorities to create adaptive prioritization.
        """
        try:
            if not goal_list:
                return goal_list
            
            # Get current drive state
            dominant_drives = self.drive_system.get_current_dominant_drives() if self.drive_system else {}
            
            # Calculate adjusted priorities for each goal
            adjusted_goals = []
            
            for goal in goal_list:
                adjusted_goal = goal  # Copy goal
                
                # Calculate drive alignment bonus
                drive_bonus = 0.0
                
                if hasattr(goal, 'driving_motives') and goal.driving_motives:
                    # Goal has explicit drive information
                    for drive, strength in goal.driving_motives.items():
                        if drive in dominant_drives:
                            drive_bonus += strength * dominant_drives[drive] * 0.2
                
                elif hasattr(goal, 'strategy'):
                    # Infer drive alignment from strategy
                    strategy_drive_mapping = {
                        "belief_clarification": "coherence",
                        "cluster_reassessment": "coherence", 
                        "fact_consolidation": "stability",
                        "exploration_goal": "curiosity",
                        "deep_exploration": "curiosity",
                        "knowledge_acquisition": "curiosity",
                        "volatility_reduction": "stability",
                        "conflict_resolution": "conflict"
                    }
                    
                    inferred_drive = strategy_drive_mapping.get(goal.strategy)
                    if inferred_drive and inferred_drive in dominant_drives:
                        drive_bonus += dominant_drives[inferred_drive] * 0.15
                
                # Apply priority adjustment
                original_priority = getattr(goal, 'priority', 0.5)
                adjusted_priority = min(1.0, original_priority + drive_bonus)
                
                # Update goal priority
                if hasattr(goal, 'adaptive_priority'):
                    goal.adaptive_priority = adjusted_priority
                else:
                    goal.priority = adjusted_priority
                
                adjusted_goals.append(goal)
            
            # Sort by adjusted priority (highest first)
            adjusted_goals.sort(key=lambda g: getattr(g, 'adaptive_priority', getattr(g, 'priority', 0.0)), 
                              reverse=True)
            
            logger.info(f"[AutonomousIntentModeler] Adjusted priorities for {len(adjusted_goals)} goals")
            return adjusted_goals
            
        except Exception as e:
            logger.error(f"Failed to adjust priorities: {e}")
            return goal_list
    
    def summarize_current_motives(self) -> str:
        """Return current dominant motives driving cognition."""
        try:
            # Get current drive state
            dominant_drives = self.drive_system.get_current_dominant_drives() if self.drive_system else {}
            
            if not dominant_drives:
                return "No active motivational drives detected."
            
            # Sort drives by strength
            sorted_drives = sorted(dominant_drives.items(), key=lambda x: x[1], reverse=True)
            
            # Get recent intent patterns
            recent_patterns = self._get_recent_patterns(hours=24)
            
            # Build summary
            summary_parts = []
            
            # Dominant drives
            summary_parts.append("🧠 **Current Motivational State:**")
            for drive, strength in sorted_drives[:3]:  # Top 3 drives
                intensity = self._get_intensity_label(strength)
                summary_parts.append(f"  • {drive.title()}: {intensity} ({strength:.2f})")
            
            # Active intent patterns
            if recent_patterns:
                summary_parts.append("\n🎯 **Active Intent Patterns:**")
                for pattern in recent_patterns[:3]:  # Top 3 patterns
                    summary_parts.append(f"  • {pattern.description} (confidence: {pattern.confidence:.2f})")
            
            # Current focus areas
            top_drive = sorted_drives[0][0] if sorted_drives else "unknown"
            focus_description = self._get_focus_description(top_drive, recent_patterns)
            summary_parts.append(f"\n🔍 **Current Focus:** {focus_description}")
            
            # Recent goal evolution
            recent_goals = len([g for g in self.evolved_goals if (time.time() - g.created_at) < 3600])  # Last hour
            if recent_goals > 0:
                summary_parts.append(f"\n⚡ **Recent Activity:** {recent_goals} autonomous goals generated in the last hour")
            
            return "\n".join(summary_parts)
            
        except Exception as e:
            logger.error(f"Failed to summarize motives: {e}")
            return f"Error summarizing motives: {e}"
    
    def _get_intensity_label(self, strength: float) -> str:
        """Convert drive strength to human-readable intensity."""
        if strength >= 0.8:
            return "Very High"
        elif strength >= 0.6:
            return "High"
        elif strength >= 0.4:
            return "Moderate"
        elif strength >= 0.2:
            return "Low"
        else:
            return "Very Low"
    
    def _get_focus_description(self, dominant_drive: str, patterns: List[IntentPattern]) -> str:
        """Generate description of current cognitive focus."""
        focus_map = {
            "curiosity": "Actively exploring new information and filling knowledge gaps",
            "coherence": "Working to maintain logical consistency and resolve contradictions",
            "stability": "Focusing on stabilizing volatile information and reinforcing certainties",
            "novelty": "Seeking out new patterns and novel connections",
            "conflict": "Addressing conflicts and tensions in the knowledge base"
        }
        
        base_focus = focus_map.get(dominant_drive, "General cognitive maintenance")
        
        # Add specific context from patterns
        if patterns:
            primary_pattern = patterns[0]
            if primary_pattern.context_triggers:
                trigger = primary_pattern.context_triggers[0]
                base_focus += f", particularly around '{trigger}'"
        
        return base_focus
    
    def _get_recent_patterns(self, hours: int = 24) -> List[IntentPattern]:
        """Get recent intent patterns from database."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cutoff_time = time.time() - (hours * 3600)
            
            cursor.execute("""
                SELECT pattern_id, description, confidence, supporting_evidence,
                       intent_category, temporal_frequency, context_triggers,
                       created_at, last_seen
                FROM intent_patterns
                WHERE last_seen > ?
                ORDER BY confidence DESC, last_seen DESC
                LIMIT 10
            """, (cutoff_time,))
            
            patterns = []
            for row in cursor.fetchall():
                pattern = IntentPattern(
                    pattern_id=row[0],
                    description=row[1],
                    confidence=row[2],
                    supporting_evidence=json.loads(row[3]) if row[3] else [],
                    intent_category=row[4],
                    temporal_frequency=row[5],
                    context_triggers=json.loads(row[6]) if row[6] else [],
                    created_at=row[7],
                    last_seen=row[8]
                )
                patterns.append(pattern)
            
            conn.close()
            return patterns
            
        except Exception as e:
            logger.error(f"Failed to get recent patterns: {e}")
            return []
    
    def _calculate_emotional_goal_boost(self, pattern: IntentPattern, drives: Dict[str, float]) -> float:
        """
        Calculate emotional boost to goal priority based on current emotional state.
        
        Args:
            pattern: Intent pattern being evaluated
            drives: Current drive signals
            
        Returns:
            Emotional boost value (0.0 to 0.3 range)
        """
        try:
            # Import emotion and self-aware models
            from storage.emotion_model import get_emotion_model
            from storage.self_model import get_self_aware_model
            
            emotion_model = get_emotion_model(self.db_path)
            self_model = get_self_aware_model(self.db_path)
            
            # Get current mood
            current_mood = emotion_model.get_current_mood()
            mood_label = current_mood.get("mood_label", "neutral")
            valence = current_mood.get("valence", 0.0)
            arousal = current_mood.get("arousal", 0.3)
            confidence = current_mood.get("confidence", 0.5)
            
            # Get emotional decision influences  
            emotional_influence = self_model.get_emotional_influence_on_decision({
                "goal_prioritization": True,
                "pattern_category": pattern.intent_category
            })
            
            # Calculate emotional alignment with goal type
            emotional_boost = 0.0
            
            # Mood-based goal preferences
            mood_goal_preferences = {
                "curious": {
                    "exploration": 0.25,
                    "curiosity": 0.3,
                    "thematic_focus": 0.2
                },
                "frustrated": {
                    "optimization": 0.2,
                    "stabilization": 0.25,
                    "exploration": -0.1  # Avoid exploration when frustrated
                },
                "calm": {
                    "stabilization": 0.2,
                    "optimization": 0.15,
                    "exploration": 0.1
                },
                "excited": {
                    "exploration": 0.3,
                    "curiosity": 0.25,
                    "thematic_focus": 0.2,
                    "stabilization": -0.05  # Less interest in stabilization when excited
                },
                "content": {
                    "optimization": 0.15,
                    "stabilization": 0.1,
                    "exploration": 0.05
                },
                "tense": {
                    "stabilization": 0.25,
                    "optimization": 0.15,
                    "exploration": -0.15
                }
            }
            
            # Apply mood-specific preferences
            if mood_label in mood_goal_preferences:
                goal_prefs = mood_goal_preferences[mood_label]
                if pattern.intent_category in goal_prefs:
                    mood_boost = goal_prefs[pattern.intent_category]
                    # Weight by mood confidence
                    emotional_boost += mood_boost * confidence
            
            # Apply general emotional characteristics
            novelty_seeking = emotional_influence.get("novelty_seeking", 0.5)
            exploration_bias = emotional_influence.get("exploration_bias", 0.5)
            patience = emotional_influence.get("patience", 0.5)
            
            # Adjust based on pattern type and emotional characteristics
            if pattern.intent_category in ["exploration", "curiosity"]:
                exploration_factor = (novelty_seeking + exploration_bias) / 2.0
                emotional_boost += (exploration_factor - 0.5) * 0.2  # Range: -0.1 to +0.1
            
            elif pattern.intent_category in ["optimization", "stabilization"]:
                patience_factor = patience
                emotional_boost += (patience_factor - 0.5) * 0.2  # Range: -0.1 to +0.1
            
            # Emotional intensity bonus (high arousal can increase urgency)
            if arousal > 0.7 and valence > 0.0:  # High positive arousal
                emotional_boost += 0.05  # Small urgency bonus
            
            # Clamp to reasonable range
            emotional_boost = max(-0.15, min(0.3, emotional_boost))
            
            logger.debug(f"[AutonomousIntentModeler] Emotional boost for {pattern.intent_category}: "
                        f"{emotional_boost:.3f} (mood: {mood_label}, valence: {valence:.2f})")
            
            return emotional_boost
            
        except Exception as e:
            logger.error(f"Failed to calculate emotional goal boost: {e}")
            return 0.0
    
    def _calculate_identity_alignment(self, pattern: IntentPattern, drives: Dict[str, float]) -> float:
        """
        Calculate how well this goal aligns with the system's identity traits.
        
        Args:
            pattern: Intent pattern being evaluated
            drives: Current drive signals
            
        Returns:
            Identity alignment boost (0.0 to 0.2 range)
        """
        try:
            # Import self-aware model
            from storage.self_model import get_self_aware_model
            
            self_model = get_self_aware_model(self.db_path)
            
            # Get current identity traits
            identity_traits = self_model.identity_traits
            
            if not identity_traits:
                return 0.0  # No identity data yet
            
            # Map pattern categories to identity traits
            category_trait_mapping = {
                "exploration": ["curious", "analytical"],
                "curiosity": ["curious", "analytical"],
                "optimization": ["analytical", "resilient"],
                "stabilization": ["resilient", "analytical"],
                "thematic_focus": ["curious", "empathetic"]
            }
            
            # Calculate alignment score
            alignment_score = 0.0
            trait_count = 0
            
            relevant_traits = category_trait_mapping.get(pattern.intent_category, [])
            
            for trait_name in relevant_traits:
                if trait_name in identity_traits:
                    trait = identity_traits[trait_name]
                    # Weight by trait strength and confidence
                    trait_contribution = trait.strength * trait.confidence * 0.1  # Max 0.1 per trait
                    alignment_score += trait_contribution
                    trait_count += 1
            
            # Normalize by number of relevant traits
            if trait_count > 0:
                alignment_score = alignment_score / trait_count
            
            # Apply gentle scaling
            alignment_score = min(0.2, alignment_score)
            
            logger.debug(f"[AutonomousIntentModeler] Identity alignment for {pattern.intent_category}: "
                        f"{alignment_score:.3f} (relevant traits: {relevant_traits})")
            
            return alignment_score
            
        except Exception as e:
            logger.error(f"Failed to calculate identity alignment: {e}")
            return 0.0


def get_intent_modeler(memory_system=None, drive_system=None) -> AutonomousIntentModeler:
    """Get or create the global intent modeler instance."""
    if not hasattr(get_intent_modeler, '_instance'):
        get_intent_modeler._instance = AutonomousIntentModeler(memory_system, drive_system)
    return get_intent_modeler._instance