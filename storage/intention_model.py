#!/usr/bin/env python3
"""
IntentionModel - Causal intention tracking system for MeRNSTA

Records and traces causal chains of motivation, goal derivation,
and intention lineage to enable autonomous self-reflection and
understanding of why actions are taken.
"""

import sqlite3
import json
import logging
import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path

from storage.db_utils import get_conn
from config.settings import get_config

@dataclass
class IntentionRecord:
    """Individual intention record linking goals to their causes"""
    intention_id: str
    goal_id: str
    parent_goal_id: Optional[str]
    triggered_by: str  # what triggered this goal
    drive: str  # underlying drive or motivation
    importance: float  # 0.0 to 1.0
    reflection_note: str  # why this goal was created
    created_at: str
    depth: int = 0  # depth in intention hierarchy
    fulfilled: bool = False
    abandoned: bool = False
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

class IntentionModel:
    """
    Tracks causal chains of motivation and goal derivation.
    
    Capabilities:
    - Record intention lineages
    - Trace causal chains ("why am I doing this?")
    - Identify motivational patterns
    - Support autonomous reflection
    - Link goals across time and context
    """
    
    def __init__(self, db_path: Optional[str] = None):
        self.config = get_config().get('intention_tracking', {})
        self.db_path = db_path or self.config.get('intention_db_path', 'intention_model.db')
        self.max_trace_depth = self.config.get('max_trace_depth', 20)
        self.importance_threshold = self.config.get('importance_threshold', 0.3)
        
        # Initialize database
        self._init_database()
        
        # Cache for performance
        self._intention_cache = {}
        
        logging.info(f"[IntentionModel] Initialized with db: {self.db_path}")
    
    def _init_database(self):
        """Initialize intention tracking database schema"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Main intentions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS intentions (
                intention_id TEXT PRIMARY KEY,
                goal_id TEXT NOT NULL,
                parent_goal_id TEXT,
                triggered_by TEXT NOT NULL,
                drive TEXT NOT NULL,
                importance REAL DEFAULT 0.5,
                reflection_note TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                depth INTEGER DEFAULT 0,
                fulfilled BOOLEAN DEFAULT FALSE,
                abandoned BOOLEAN DEFAULT FALSE,
                user_profile_id TEXT,
                session_id TEXT,
                context_data TEXT,  -- JSON additional context
                metadata TEXT,      -- JSON metadata
                FOREIGN KEY (parent_goal_id) REFERENCES intentions (goal_id)
            )
        """)
        
        # Intention relationships table (for complex causality)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS intention_relationships (
                relationship_id TEXT PRIMARY KEY,
                source_intention_id TEXT NOT NULL,
                target_intention_id TEXT NOT NULL,
                relationship_type TEXT NOT NULL,  -- caused_by, triggered_by, enables, conflicts_with, replaces
                strength REAL DEFAULT 0.5,
                confidence REAL DEFAULT 0.8,
                created_at TEXT NOT NULL,
                active BOOLEAN DEFAULT TRUE,
                metadata TEXT,
                FOREIGN KEY (source_intention_id) REFERENCES intentions (intention_id),
                FOREIGN KEY (target_intention_id) REFERENCES intentions (intention_id)
            )
        """)
        
        # Drive patterns table (for learning motivation patterns)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS drive_patterns (
                pattern_id TEXT PRIMARY KEY,
                drive_category TEXT NOT NULL,
                trigger_pattern TEXT NOT NULL,
                goal_pattern TEXT NOT NULL,
                frequency INTEGER DEFAULT 1,
                success_rate REAL DEFAULT 0.0,
                avg_importance REAL DEFAULT 0.0,
                last_seen TEXT NOT NULL,
                created_at TEXT NOT NULL,
                active BOOLEAN DEFAULT TRUE,
                metadata TEXT
            )
        """)
        
        # Reflection logs table (for self-analysis)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS intention_reflections (
                reflection_id TEXT PRIMARY KEY,
                intention_id TEXT NOT NULL,
                reflection_type TEXT NOT NULL,  -- self_analysis, pattern_discovery, contradiction_found
                reflection_content TEXT NOT NULL,
                insights TEXT,  -- JSON array of insights
                confidence REAL DEFAULT 0.5,
                created_at TEXT NOT NULL,
                user_profile_id TEXT,
                session_id TEXT,
                FOREIGN KEY (intention_id) REFERENCES intentions (intention_id)
            )
        """)
        
        # Create indices for performance
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_intentions_goal ON intentions (goal_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_intentions_parent ON intentions (parent_goal_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_intentions_drive ON intentions (drive)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_intentions_depth ON intentions (depth)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_relationships_source ON intention_relationships (source_intention_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_relationships_target ON intention_relationships (target_intention_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_reflections_intention ON intention_reflections (intention_id)")
        
        conn.commit()
        conn.close()
        
        logging.info("[IntentionModel] Database schema initialized")
    
    def record_intention(self, goal_id: str, triggered_by: str, drive: str, 
                        importance: float = 0.5, reflection_note: str = "",
                        parent_goal_id: Optional[str] = None,
                        context: Optional[Dict[str, Any]] = None) -> str:
        """
        Record a new intention linking a goal to its motivation.
        
        Args:
            goal_id: Unique identifier for the goal
            triggered_by: What triggered this goal (event, need, insight, etc.)
            drive: Underlying drive or motivation category
            importance: Importance score (0.0 to 1.0)
            reflection_note: Human-readable explanation of why this goal exists
            parent_goal_id: Parent goal if this is a subgoal
            context: Additional context data
            
        Returns:
            intention_id of the recorded intention
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            intention_id = str(uuid.uuid4())
            now = datetime.now().isoformat()
            
            # Calculate depth based on parent
            depth = 0
            if parent_goal_id:
                cursor.execute("SELECT depth FROM intentions WHERE goal_id = ?", (parent_goal_id,))
                parent_row = cursor.fetchone()
                if parent_row:
                    depth = parent_row[0] + 1
            
            # Insert intention record
            cursor.execute("""
                INSERT INTO intentions (
                    intention_id, goal_id, parent_goal_id, triggered_by, drive,
                    importance, reflection_note, created_at, updated_at, depth,
                    context_data, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                intention_id, goal_id, parent_goal_id, triggered_by, drive,
                importance, reflection_note, now, now, depth,
                json.dumps(context or {}), json.dumps({})
            ))
            
            # Update drive patterns
            self._update_drive_pattern(drive, triggered_by, goal_id)
            
            conn.commit()
            conn.close()
            
            # Cache the intention
            self._intention_cache[goal_id] = {
                'intention_id': intention_id,
                'triggered_by': triggered_by,
                'drive': drive,
                'importance': importance,
                'depth': depth
            }
            
            logging.info(f"[IntentionModel] Recorded intention for goal: {goal_id}")
            return intention_id
            
        except Exception as e:
            logging.error(f"[IntentionModel] Error recording intention: {e}")
            return ""
    
    def link_goals(self, child_goal_id: str, parent_goal_id: str, 
                   relationship_note: str, relationship_type: str = "caused_by") -> bool:
        """
        Create a relationship link between two goals.
        
        Args:
            child_goal_id: Goal that was derived from parent
            parent_goal_id: Parent goal that caused the child
            relationship_note: Description of the relationship
            relationship_type: Type of relationship
            
        Returns:
            True if linked successfully
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get intention IDs
            cursor.execute("SELECT intention_id FROM intentions WHERE goal_id = ?", (child_goal_id,))
            child_row = cursor.fetchone()
            cursor.execute("SELECT intention_id FROM intentions WHERE goal_id = ?", (parent_goal_id,))
            parent_row = cursor.fetchone()
            
            if not child_row or not parent_row:
                logging.warning(f"[IntentionModel] Could not find intentions for goal linking")
                return False
            
            # Create relationship
            relationship_id = str(uuid.uuid4())
            cursor.execute("""
                INSERT INTO intention_relationships (
                    relationship_id, source_intention_id, target_intention_id,
                    relationship_type, strength, created_at, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                relationship_id, parent_row[0], child_row[0],
                relationship_type, 0.8, datetime.now().isoformat(),
                json.dumps({'note': relationship_note})
            ))
            
            # Update child goal parent reference
            cursor.execute("""
                UPDATE intentions SET parent_goal_id = ? WHERE goal_id = ?
            """, (parent_goal_id, child_goal_id))
            
            conn.commit()
            conn.close()
            
            logging.info(f"[IntentionModel] Linked goals: {child_goal_id} -> {parent_goal_id}")
            return True
            
        except Exception as e:
            logging.error(f"[IntentionModel] Error linking goals: {e}")
            return False
    
    def trace_why(self, goal_id: str) -> List[Dict[str, Any]]:
        """
        Trace the causal chain of why a goal exists.
        
        Args:
            goal_id: Goal to trace backwards from
            
        Returns:
            List of intention records forming the causal chain
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            trace_chain = []
            current_goal_id = goal_id
            depth = 0
            
            while current_goal_id and depth < self.max_trace_depth:
                # Get current intention
                cursor.execute("""
                    SELECT intention_id, goal_id, parent_goal_id, triggered_by, drive,
                           importance, reflection_note, created_at, depth, fulfilled, abandoned
                    FROM intentions 
                    WHERE goal_id = ?
                """, (current_goal_id,))
                
                row = cursor.fetchone()
                if not row:
                    break
                
                intention_record = {
                    'intention_id': row[0],
                    'goal_id': row[1],
                    'parent_goal_id': row[2],
                    'triggered_by': row[3],
                    'drive': row[4],
                    'importance': row[5],
                    'reflection_note': row[6],
                    'created_at': row[7],
                    'depth': row[8],
                    'fulfilled': row[9],
                    'abandoned': row[10],
                    'trace_depth': depth
                }
                
                trace_chain.append(intention_record)
                
                # Move to parent
                current_goal_id = row[2]  # parent_goal_id
                depth += 1
            
            conn.close()
            
            logging.info(f"[IntentionModel] Traced {len(trace_chain)} levels for goal: {goal_id}")
            return trace_chain
            
        except Exception as e:
            logging.error(f"[IntentionModel] Error tracing goal: {e}")
            return []
    
    def trace_why_formatted(self, goal_id: str) -> str:
        """
        Get a human-readable explanation of why a goal exists.
        
        Args:
            goal_id: Goal to explain
            
        Returns:
            Formatted explanation string
        """
        trace_chain = self.trace_why(goal_id)
        
        if not trace_chain:
            return f"No intention record found for goal: {goal_id}"
        
        explanation_parts = []
        
        for i, intention in enumerate(trace_chain):
            prefix = "  " * i + ("└─ " if i > 0 else "🎯 ")
            
            drive_emoji = self._get_drive_emoji(intention['drive'])
            importance_indicator = "!" * min(3, int(intention['importance'] * 3 + 1))
            
            line = f"{prefix}{drive_emoji} {intention['reflection_note']}"
            if intention['triggered_by']:
                line += f" (triggered by: {intention['triggered_by']})"
            
            line += f" [{importance_indicator}]"
            
            if intention['fulfilled']:
                line += " ✅"
            elif intention['abandoned']:
                line += " ❌"
            
            explanation_parts.append(line)
        
        if len(trace_chain) >= self.max_trace_depth:
            explanation_parts.append(f"  {'  ' * len(trace_chain)}⋮ (trace truncated)")
        
        header = f"\n🧠 **Why am I doing this?** (Goal: {goal_id})\n"
        return header + "\n".join(explanation_parts)
    
    def add_reflection(self, intention_id: str, reflection_content: str,
                      reflection_type: str = "self_analysis", 
                      insights: Optional[List[str]] = None) -> str:
        """
        Add a reflection or insight about an intention.
        
        Args:
            intention_id: Intention to reflect on
            reflection_content: The reflection content
            reflection_type: Type of reflection
            insights: Key insights from the reflection
            
        Returns:
            reflection_id
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            reflection_id = str(uuid.uuid4())
            
            cursor.execute("""
                INSERT INTO intention_reflections (
                    reflection_id, intention_id, reflection_type, reflection_content,
                    insights, created_at
                ) VALUES (?, ?, ?, ?, ?, ?)
            """, (
                reflection_id, intention_id, reflection_type, reflection_content,
                json.dumps(insights or []), datetime.now().isoformat()
            ))
            
            conn.commit()
            conn.close()
            
            logging.info(f"[IntentionModel] Added reflection for intention: {intention_id}")
            return reflection_id
            
        except Exception as e:
            logging.error(f"[IntentionModel] Error adding reflection: {e}")
            return ""
    
    def get_motivational_patterns(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Identify patterns in motivation and goal creation.
        
        Args:
            limit: Maximum number of patterns to return
            
        Returns:
            List of discovered motivational patterns
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get drive patterns
            cursor.execute("""
                SELECT drive_category, trigger_pattern, goal_pattern, frequency,
                       success_rate, avg_importance, last_seen
                FROM drive_patterns 
                WHERE active = TRUE
                ORDER BY frequency DESC, success_rate DESC
                LIMIT ?
            """, (limit,))
            
            patterns = []
            for row in cursor.fetchall():
                patterns.append({
                    'drive_category': row[0],
                    'trigger_pattern': row[1],
                    'goal_pattern': row[2],
                    'frequency': row[3],
                    'success_rate': row[4],
                    'avg_importance': row[5],
                    'last_seen': row[6]
                })
            
            # Get intention frequency by drive
            cursor.execute("""
                SELECT drive, COUNT(*) as count, AVG(importance) as avg_importance,
                       SUM(CASE WHEN fulfilled THEN 1 ELSE 0 END) * 1.0 / COUNT(*) as success_rate
                FROM intentions 
                WHERE NOT abandoned
                GROUP BY drive
                ORDER BY count DESC
                LIMIT ?
            """, (limit,))
            
            drive_stats = []
            for row in cursor.fetchall():
                drive_stats.append({
                    'drive': row[0],
                    'frequency': row[1],
                    'avg_importance': row[2],
                    'success_rate': row[3] or 0.0
                })
            
            conn.close()
            
            return {
                'drive_patterns': patterns,
                'drive_statistics': drive_stats
            }
            
        except Exception as e:
            logging.error(f"[IntentionModel] Error getting patterns: {e}")
            return {'drive_patterns': [], 'drive_statistics': []}
    
    def mark_goal_fulfilled(self, goal_id: str) -> bool:
        """Mark a goal as fulfilled"""
        return self._update_goal_status(goal_id, fulfilled=True)
    
    def mark_goal_abandoned(self, goal_id: str) -> bool:
        """Mark a goal as abandoned"""
        return self._update_goal_status(goal_id, abandoned=True)
    
    def get_active_intentions(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get currently active (unfulfilled, non-abandoned) intentions"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT intention_id, goal_id, triggered_by, drive, importance,
                       reflection_note, created_at, depth
                FROM intentions 
                WHERE NOT fulfilled AND NOT abandoned
                ORDER BY importance DESC, created_at DESC
                LIMIT ?
            """, (limit,))
            
            intentions = []
            for row in cursor.fetchall():
                intentions.append({
                    'intention_id': row[0],
                    'goal_id': row[1],
                    'triggered_by': row[2],
                    'drive': row[3],
                    'importance': row[4],
                    'reflection_note': row[5],
                    'created_at': row[6],
                    'depth': row[7]
                })
            
            conn.close()
            return intentions
            
        except Exception as e:
            logging.error(f"[IntentionModel] Error getting active intentions: {e}")
            return []
    
    def get_intention_statistics(self) -> Dict[str, Any]:
        """Get overall intention tracking statistics"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Basic counts
            cursor.execute("SELECT COUNT(*) FROM intentions")
            total_intentions = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM intentions WHERE fulfilled")
            fulfilled_count = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM intentions WHERE abandoned")
            abandoned_count = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM intentions WHERE NOT fulfilled AND NOT abandoned")
            active_count = cursor.fetchone()[0]
            
            # Drive breakdown
            cursor.execute("SELECT drive, COUNT(*) FROM intentions GROUP BY drive ORDER BY COUNT(*) DESC LIMIT 10")
            drive_breakdown = dict(cursor.fetchall())
            
            # Average importance by depth
            cursor.execute("SELECT depth, AVG(importance) FROM intentions GROUP BY depth ORDER BY depth")
            importance_by_depth = dict(cursor.fetchall())
            
            # Relationships count
            cursor.execute("SELECT COUNT(*) FROM intention_relationships WHERE active")
            relationships_count = cursor.fetchone()[0]
            
            conn.close()
            
            return {
                'total_intentions': total_intentions,
                'fulfilled_count': fulfilled_count,
                'abandoned_count': abandoned_count,
                'active_count': active_count,
                'fulfillment_rate': fulfilled_count / max(1, total_intentions),
                'abandonment_rate': abandoned_count / max(1, total_intentions),
                'drive_breakdown': drive_breakdown,
                'importance_by_depth': importance_by_depth,
                'relationships_count': relationships_count
            }
            
        except Exception as e:
            logging.error(f"[IntentionModel] Error getting statistics: {e}")
            return {}
    
    def _update_goal_status(self, goal_id: str, fulfilled: bool = False, abandoned: bool = False) -> bool:
        """Update the status of a goal"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                UPDATE intentions 
                SET fulfilled = ?, abandoned = ?, updated_at = ?
                WHERE goal_id = ?
            """, (fulfilled, abandoned, datetime.now().isoformat(), goal_id))
            
            updated = cursor.rowcount > 0
            conn.commit()
            conn.close()
            
            if updated:
                # Update cache
                if goal_id in self._intention_cache:
                    self._intention_cache[goal_id]['fulfilled'] = fulfilled
                    self._intention_cache[goal_id]['abandoned'] = abandoned
            
            return updated
            
        except Exception as e:
            logging.error(f"[IntentionModel] Error updating goal status: {e}")
            return False
    
    def _update_drive_pattern(self, drive: str, trigger: str, goal: str):
        """Update or create drive pattern record"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Try to find existing pattern
            cursor.execute("""
                SELECT pattern_id, frequency FROM drive_patterns 
                WHERE drive_category = ? AND trigger_pattern = ?
            """, (drive, trigger[:100]))  # Truncate for pattern matching
            
            existing = cursor.fetchone()
            
            if existing:
                # Update existing pattern
                cursor.execute("""
                    UPDATE drive_patterns 
                    SET frequency = frequency + 1, last_seen = ?
                    WHERE pattern_id = ?
                """, (datetime.now().isoformat(), existing[0]))
            else:
                # Create new pattern
                pattern_id = str(uuid.uuid4())
                cursor.execute("""
                    INSERT INTO drive_patterns (
                        pattern_id, drive_category, trigger_pattern, goal_pattern,
                        frequency, last_seen, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    pattern_id, drive, trigger[:100], goal[:100],
                    1, datetime.now().isoformat(), datetime.now().isoformat()
                ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logging.warning(f"[IntentionModel] Could not update drive pattern: {e}")
    
    def _get_drive_emoji(self, drive: str) -> str:
        """Get emoji representation for drive category"""
        drive_emojis = {
            'self_improvement': '📈',
            'problem_solving': '🔧',
            'curiosity': '🔍',
            'efficiency': '⚡',
            'learning': '📚',
            'creativity': '🎨',
            'social': '👥',
            'survival': '🛡️',
            'exploration': '🗺️',
            'maintenance': '🔄',
            'optimization': '⚙️',
            'discovery': '💡',
            'connection': '🔗',
            'understanding': '🧠',
            'achievement': '🏆'
        }
        
        return drive_emojis.get(drive.lower(), '🎯')

# Convenience function for quick intention recording
def record_goal_intention(goal_id: str, why: str, drive: str = "task_completion",
                         importance: float = 0.5, parent_goal_id: Optional[str] = None) -> str:
    """
    Quick function to record an intention for a goal.
    
    Args:
        goal_id: Goal identifier
        why: Explanation of why this goal exists
        drive: Motivational drive category
        importance: Importance score
        parent_goal_id: Parent goal if this is a subgoal
        
    Returns:
        intention_id
    """
    try:
        intention_model = IntentionModel()
        return intention_model.record_intention(
            goal_id=goal_id,
            triggered_by="goal_creation",
            drive=drive,
            importance=importance,
            reflection_note=why,
            parent_goal_id=parent_goal_id
        )
    except Exception as e:
        logging.error(f"Error recording goal intention: {e}")
        return ""

# Convenience function for tracing
def trace_goal_why(goal_id: str) -> str:
    """
    Quick function to get why explanation for a goal.
    
    Args:
        goal_id: Goal to trace
        
    Returns:
        Formatted explanation
    """
    try:
        intention_model = IntentionModel()
        return intention_model.trace_why_formatted(goal_id)
    except Exception as e:
        logging.error(f"Error tracing goal: {e}")
        return f"Could not trace goal {goal_id}: {str(e)}"