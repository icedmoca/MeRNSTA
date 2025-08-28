#!/usr/bin/env python3
"""
SelfRepairLog - Logging and tracking system for self-repair attempts

Manages the history of self-repair attempts, outcomes, and learning from
repair successes and failures to improve future repair strategies.
"""

import sqlite3
import json
import logging
import uuid
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path

from storage.db_utils import get_conn
from config.settings import get_config

@dataclass
class RepairAttempt:
    """Record of a repair attempt"""
    attempt_id: str
    goal: str                    # Repair goal description
    approach: str                # How repair was attempted
    start_time: str              # ISO timestamp
    end_time: Optional[str]      # ISO timestamp when completed
    status: str                  # pending, in_progress, completed, failed, aborted
    result: str                  # Outcome description
    score: float                 # Success score 0.0 to 1.0
    issues_addressed: List[str]  # List of issue IDs that were targeted
    issues_resolved: List[str]   # List of issue IDs that were actually resolved
    side_effects: List[str]      # Any unintended consequences
    metrics_before: Dict[str, Any] = None  # System metrics before repair
    metrics_after: Dict[str, Any] = None   # System metrics after repair
    repair_plan_id: Optional[str] = None   # Associated plan ID if used
    user_profile_id: Optional[str] = None
    session_id: Optional[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metrics_before is None:
            self.metrics_before = {}
        if self.metrics_after is None:
            self.metrics_after = {}
        if self.metadata is None:
            self.metadata = {}

@dataclass
class RepairOutcome:
    """Summary of repair outcomes for analysis"""
    goal_pattern: str            # Pattern of repair goal
    total_attempts: int          # Total attempts for this pattern
    successful_attempts: int     # Number of successful attempts
    average_score: float         # Average success score
    average_duration: float      # Average duration in seconds
    common_approaches: List[str] # Most successful approaches
    frequent_side_effects: List[str] # Common side effects
    success_rate: float          # Success rate percentage

class SelfRepairLog:
    """
    Logging and analysis system for self-repair attempts.
    
    Capabilities:
    - Track repair attempts with detailed metadata
    - Analyze repair success patterns
    - Identify failed repair strategies
    - Generate insights for future repairs
    - Monitor system health improvements
    """
    
    def __init__(self, db_path: Optional[str] = None):
        self.config = get_config().get('self_repair', {})
        self.db_path = db_path or self.config.get('repair_log_db_path', 'self_repair.db')
        self.retention_days = self.config.get('log_retention_days', 90)
        self.analysis_window_days = self.config.get('analysis_window_days', 30)
        
        # Initialize database
        self._init_database()
        
        logging.info(f"[SelfRepairLog] Initialized with db: {self.db_path}")
    
    def _init_database(self):
        """Initialize repair log database schema"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Repair attempts table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS repair_attempts (
                attempt_id TEXT PRIMARY KEY,
                goal TEXT NOT NULL,
                approach TEXT NOT NULL,
                start_time TEXT NOT NULL,
                end_time TEXT,
                status TEXT DEFAULT 'pending',
                result TEXT DEFAULT '',
                score REAL DEFAULT 0.0,
                issues_addressed TEXT,      -- JSON array
                issues_resolved TEXT,       -- JSON array
                side_effects TEXT,          -- JSON array
                metrics_before TEXT,        -- JSON object
                metrics_after TEXT,         -- JSON object
                repair_plan_id TEXT,
                user_profile_id TEXT,
                session_id TEXT,
                metadata TEXT,              -- JSON object
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)
        
        # Repair patterns table (for learning)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS repair_patterns (
                pattern_id TEXT PRIMARY KEY,
                goal_pattern TEXT NOT NULL,
                approach_pattern TEXT NOT NULL,
                success_count INTEGER DEFAULT 0,
                failure_count INTEGER DEFAULT 0,
                average_score REAL DEFAULT 0.0,
                average_duration REAL DEFAULT 0.0,
                last_attempted TEXT,
                last_successful TEXT,
                best_score REAL DEFAULT 0.0,
                worst_score REAL DEFAULT 1.0,
                common_side_effects TEXT,   -- JSON array
                effectiveness_trend TEXT,   -- JSON array of recent scores
                metadata TEXT,              -- JSON object
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)
        
        # System health snapshots table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS health_snapshots (
                snapshot_id TEXT PRIMARY KEY,
                taken_at TEXT NOT NULL,
                health_score REAL NOT NULL,
                total_issues INTEGER DEFAULT 0,
                critical_issues INTEGER DEFAULT 0,
                high_issues INTEGER DEFAULT 0,
                issues_by_category TEXT,    -- JSON object
                patterns_detected TEXT,     -- JSON array
                repair_attempt_id TEXT,     -- Associated repair if taken during repair
                metrics TEXT,               -- JSON object with detailed metrics
                FOREIGN KEY (repair_attempt_id) REFERENCES repair_attempts (attempt_id)
            )
        """)
        
        # Repair insights table (for machine learning from patterns)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS repair_insights (
                insight_id TEXT PRIMARY KEY,
                insight_type TEXT NOT NULL,    -- success_pattern, failure_pattern, side_effect_pattern
                pattern_description TEXT NOT NULL,
                confidence REAL DEFAULT 0.5,
                evidence_count INTEGER DEFAULT 1,
                applicability TEXT,            -- When this insight applies
                recommendation TEXT,           -- What to do based on this insight
                discovered_at TEXT NOT NULL,
                last_validated TEXT,
                validation_count INTEGER DEFAULT 0,
                effectiveness_score REAL DEFAULT 0.0,
                metadata TEXT                  -- JSON object
            )
        """)
        
        # Create indices for performance
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_attempts_goal ON repair_attempts (goal)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_attempts_status ON repair_attempts (status)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_attempts_start_time ON repair_attempts (start_time)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_patterns_goal ON repair_patterns (goal_pattern)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_snapshots_taken_at ON health_snapshots (taken_at)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_insights_type ON repair_insights (insight_type)")
        
        conn.commit()
        conn.close()
        
        logging.info("[SelfRepairLog] Database schema initialized")
    
    def log_repair_attempt(self, goal: str, approach: str, repair_plan_id: Optional[str] = None,
                          issues_addressed: Optional[List[str]] = None,
                          metrics_before: Optional[Dict[str, Any]] = None) -> str:
        """
        Log the start of a repair attempt.
        
        Args:
            goal: Repair goal description
            approach: How the repair will be attempted
            repair_plan_id: Associated plan ID if using planning system
            issues_addressed: List of issue IDs being targeted
            metrics_before: System metrics before repair
            
        Returns:
            attempt_id for tracking this repair
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            attempt_id = str(uuid.uuid4())
            now = datetime.now().isoformat()
            
            cursor.execute("""
                INSERT INTO repair_attempts (
                    attempt_id, goal, approach, start_time, status,
                    issues_addressed, metrics_before, repair_plan_id,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                attempt_id, goal, approach, now, 'in_progress',
                json.dumps(issues_addressed or []),
                json.dumps(metrics_before or {}),
                repair_plan_id, now, now
            ))
            
            conn.commit()
            conn.close()
            
            logging.info(f"[SelfRepairLog] Started repair attempt: {attempt_id}")
            
            return attempt_id
            
        except Exception as e:
            logging.error(f"[SelfRepairLog] Error logging repair attempt: {e}")
            return ""
    
    def update_repair_progress(self, attempt_id: str, status: str, 
                              result: Optional[str] = None,
                              issues_resolved: Optional[List[str]] = None,
                              side_effects: Optional[List[str]] = None) -> bool:
        """
        Update the progress of a repair attempt.
        
        Args:
            attempt_id: ID of the repair attempt
            status: Current status (in_progress, completed, failed, aborted)
            result: Description of current outcome
            issues_resolved: List of issue IDs that have been resolved
            side_effects: Any unintended consequences observed
            
        Returns:
            True if updated successfully
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            update_fields = ["status = ?", "updated_at = ?"]
            update_values = [status, datetime.now().isoformat()]
            
            if result is not None:
                update_fields.append("result = ?")
                update_values.append(result)
            
            if issues_resolved is not None:
                update_fields.append("issues_resolved = ?")
                update_values.append(json.dumps(issues_resolved))
            
            if side_effects is not None:
                update_fields.append("side_effects = ?")
                update_values.append(json.dumps(side_effects))
            
            update_values.append(attempt_id)
            
            cursor.execute(f"""
                UPDATE repair_attempts 
                SET {', '.join(update_fields)}
                WHERE attempt_id = ?
            """, update_values)
            
            updated = cursor.rowcount > 0
            conn.commit()
            conn.close()
            
            if updated:
                logging.info(f"[SelfRepairLog] Updated repair attempt: {attempt_id}")
            
            return updated
            
        except Exception as e:
            logging.error(f"[SelfRepairLog] Error updating repair progress: {e}")
            return False
    
    def complete_repair_attempt(self, attempt_id: str, result: str, score: float,
                               issues_resolved: Optional[List[str]] = None,
                               side_effects: Optional[List[str]] = None,
                               metrics_after: Optional[Dict[str, Any]] = None) -> bool:
        """
        Complete a repair attempt with final results.
        
        Args:
            attempt_id: ID of the repair attempt
            result: Final outcome description
            score: Success score from 0.0 to 1.0
            issues_resolved: List of issue IDs that were resolved
            side_effects: Any unintended consequences
            metrics_after: System metrics after repair
            
        Returns:
            True if completed successfully
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Update the repair attempt
            cursor.execute("""
                UPDATE repair_attempts 
                SET status = ?, end_time = ?, result = ?, score = ?,
                    issues_resolved = ?, side_effects = ?, metrics_after = ?,
                    updated_at = ?
                WHERE attempt_id = ?
            """, (
                'completed' if score >= 0.5 else 'failed',
                datetime.now().isoformat(),
                result,
                score,
                json.dumps(issues_resolved or []),
                json.dumps(side_effects or []),
                json.dumps(metrics_after or {}),
                datetime.now().isoformat(),
                attempt_id
            ))
            
            # Get the attempt details for pattern learning
            cursor.execute("""
                SELECT goal, approach, start_time, end_time 
                FROM repair_attempts 
                WHERE attempt_id = ?
            """, (attempt_id,))
            
            attempt_data = cursor.fetchone()
            
            if attempt_data:
                goal, approach, start_time, end_time = attempt_data
                
                # Calculate duration
                duration = 0.0
                if start_time and end_time:
                    start_dt = datetime.fromisoformat(start_time)
                    end_dt = datetime.fromisoformat(end_time)
                    duration = (end_dt - start_dt).total_seconds()
                
                # Update or create repair pattern
                self._update_repair_pattern(goal, approach, score >= 0.5, score, duration, side_effects or [])
                
                # Generate insights if this is a significant success or failure
                if score >= 0.8 or score <= 0.2:
                    self._generate_insight(goal, approach, score, result, side_effects or [])
            
            conn.commit()
            conn.close()
            
            logging.info(f"[SelfRepairLog] Completed repair attempt: {attempt_id} (score: {score:.2f})")
            
            return True
            
        except Exception as e:
            logging.error(f"[SelfRepairLog] Error completing repair attempt: {e}")
            return False
    
    def get_failed_repairs(self, limit: int = 50, days_back: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get list of failed repair attempts for analysis.
        
        Args:
            limit: Maximum number of failed repairs to return
            days_back: Only include repairs from this many days back
            
        Returns:
            List of failed repair attempt records
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            query = """
                SELECT attempt_id, goal, approach, start_time, end_time, result, score,
                       issues_addressed, issues_resolved, side_effects
                FROM repair_attempts 
                WHERE status = 'failed' OR score < 0.5
            """
            
            params = []
            
            if days_back:
                cutoff_date = (datetime.now() - timedelta(days=days_back)).isoformat()
                query += " AND start_time >= ?"
                params.append(cutoff_date)
            
            query += " ORDER BY start_time DESC LIMIT ?"
            params.append(limit)
            
            cursor.execute(query, params)
            
            failed_repairs = []
            for row in cursor.fetchall():
                repair = {
                    'attempt_id': row[0],
                    'goal': row[1],
                    'approach': row[2],
                    'start_time': row[3],
                    'end_time': row[4],
                    'result': row[5],
                    'score': row[6],
                    'issues_addressed': json.loads(row[7] or '[]'),
                    'issues_resolved': json.loads(row[8] or '[]'),
                    'side_effects': json.loads(row[9] or '[]')
                }
                failed_repairs.append(repair)
            
            conn.close()
            
            logging.info(f"[SelfRepairLog] Retrieved {len(failed_repairs)} failed repairs")
            
            return failed_repairs
            
        except Exception as e:
            logging.error(f"[SelfRepairLog] Error getting failed repairs: {e}")
            return []
    
    def summarize_recent_repairs(self, days_back: int = 30) -> Dict[str, Any]:
        """
        Summarize recent repair attempts and outcomes.
        
        Args:
            days_back: Number of days to include in summary
            
        Returns:
            Summary statistics and insights
        """
        try:
            cutoff_date = (datetime.now() - timedelta(days=days_back)).isoformat()
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Basic statistics
            cursor.execute("""
                SELECT COUNT(*), AVG(score), 
                       SUM(CASE WHEN score >= 0.5 THEN 1 ELSE 0 END),
                       SUM(CASE WHEN score < 0.5 THEN 1 ELSE 0 END)
                FROM repair_attempts 
                WHERE start_time >= ? AND status IN ('completed', 'failed')
            """, (cutoff_date,))
            
            basic_stats = cursor.fetchone()
            total_attempts, avg_score, successful, failed = basic_stats
            
            # Most common goals
            cursor.execute("""
                SELECT goal, COUNT(*), AVG(score)
                FROM repair_attempts 
                WHERE start_time >= ?
                GROUP BY goal
                ORDER BY COUNT(*) DESC
                LIMIT 10
            """, (cutoff_date,))
            
            common_goals = [
                {'goal': row[0], 'attempts': row[1], 'avg_score': row[2]}
                for row in cursor.fetchall()
            ]
            
            # Most successful approaches
            cursor.execute("""
                SELECT approach, COUNT(*), AVG(score)
                FROM repair_attempts 
                WHERE start_time >= ? AND score >= 0.5
                GROUP BY approach
                ORDER BY AVG(score) DESC, COUNT(*) DESC
                LIMIT 10
            """, (cutoff_date,))
            
            successful_approaches = [
                {'approach': row[0], 'attempts': row[1], 'avg_score': row[2]}
                for row in cursor.fetchall()
            ]
            
            # Recent trends
            cursor.execute("""
                SELECT DATE(start_time) as repair_date, COUNT(*), AVG(score)
                FROM repair_attempts 
                WHERE start_time >= ?
                GROUP BY DATE(start_time)
                ORDER BY repair_date DESC
                LIMIT 7
            """, (cutoff_date,))
            
            daily_trends = [
                {'date': row[0], 'attempts': row[1], 'avg_score': row[2]}
                for row in cursor.fetchall()
            ]
            
            # Side effects analysis
            cursor.execute("""
                SELECT side_effects 
                FROM repair_attempts 
                WHERE start_time >= ? AND side_effects != '[]' AND side_effects IS NOT NULL
            """, (cutoff_date,))
            
            all_side_effects = []
            for row in cursor.fetchall():
                effects = json.loads(row[0])
                all_side_effects.extend(effects)
            
            from collections import Counter
            side_effect_counts = Counter(all_side_effects)
            
            conn.close()
            
            summary = {
                'analysis_period': f"{days_back} days",
                'total_attempts': total_attempts or 0,
                'successful_attempts': successful or 0,
                'failed_attempts': failed or 0,
                'success_rate': (successful / max(1, total_attempts)) if total_attempts else 0.0,
                'average_score': avg_score or 0.0,
                'common_goals': common_goals,
                'successful_approaches': successful_approaches,
                'daily_trends': daily_trends,
                'common_side_effects': dict(side_effect_counts.most_common(5))
            }
            
            logging.info(f"[SelfRepairLog] Generated repair summary for {days_back} days")
            
            return summary
            
        except Exception as e:
            logging.error(f"[SelfRepairLog] Error summarizing repairs: {e}")
            return {}
    
    def get_repair_patterns(self, min_attempts: int = 3) -> List[RepairOutcome]:
        """
        Get successful repair patterns for learning.
        
        Args:
            min_attempts: Minimum number of attempts to consider a pattern
            
        Returns:
            List of repair patterns with success metrics
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT goal_pattern, approach_pattern, success_count, failure_count,
                       average_score, average_duration, common_side_effects
                FROM repair_patterns 
                WHERE (success_count + failure_count) >= ?
                ORDER BY average_score DESC, success_count DESC
            """, (min_attempts,))
            
            patterns = []
            for row in cursor.fetchall():
                total_attempts = row[2] + row[3]  # success_count + failure_count
                
                pattern = RepairOutcome(
                    goal_pattern=row[0],
                    total_attempts=total_attempts,
                    successful_attempts=row[2],
                    average_score=row[4],
                    average_duration=row[5],
                    common_approaches=[row[1]],  # Simplified - could be expanded
                    frequent_side_effects=json.loads(row[6] or '[]'),
                    success_rate=(row[2] / total_attempts) if total_attempts > 0 else 0.0
                )
                patterns.append(pattern)
            
            conn.close()
            
            logging.info(f"[SelfRepairLog] Retrieved {len(patterns)} repair patterns")
            
            return patterns
            
        except Exception as e:
            logging.error(f"[SelfRepairLog] Error getting repair patterns: {e}")
            return []
    
    def take_health_snapshot(self, health_score: float, total_issues: int,
                           critical_issues: int, high_issues: int,
                           issues_by_category: Dict[str, int],
                           patterns_detected: List[str],
                           repair_attempt_id: Optional[str] = None,
                           detailed_metrics: Optional[Dict[str, Any]] = None) -> str:
        """
        Take a snapshot of system health for tracking improvements.
        
        Args:
            health_score: Overall health score (0.0 to 1.0)
            total_issues: Total number of issues
            critical_issues: Number of critical issues
            high_issues: Number of high-severity issues
            issues_by_category: Breakdown of issues by category
            patterns_detected: List of detected patterns
            repair_attempt_id: Associated repair attempt if applicable
            detailed_metrics: Additional detailed metrics
            
        Returns:
            snapshot_id
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            snapshot_id = str(uuid.uuid4())
            
            cursor.execute("""
                INSERT INTO health_snapshots (
                    snapshot_id, taken_at, health_score, total_issues,
                    critical_issues, high_issues, issues_by_category,
                    patterns_detected, repair_attempt_id, metrics
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                snapshot_id,
                datetime.now().isoformat(),
                health_score,
                total_issues,
                critical_issues,
                high_issues,
                json.dumps(issues_by_category),
                json.dumps(patterns_detected),
                repair_attempt_id,
                json.dumps(detailed_metrics or {})
            ))
            
            conn.commit()
            conn.close()
            
            logging.info(f"[SelfRepairLog] Took health snapshot: {snapshot_id}")
            
            return snapshot_id
            
        except Exception as e:
            logging.error(f"[SelfRepairLog] Error taking health snapshot: {e}")
            return ""
    
    def get_health_trend(self, days_back: int = 30) -> List[Dict[str, Any]]:
        """
        Get health trend over time.
        
        Args:
            days_back: Number of days to include
            
        Returns:
            List of health snapshots with timestamps
        """
        try:
            cutoff_date = (datetime.now() - timedelta(days=days_back)).isoformat()
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT taken_at, health_score, total_issues, critical_issues, high_issues
                FROM health_snapshots 
                WHERE taken_at >= ?
                ORDER BY taken_at
            """, (cutoff_date,))
            
            trend = []
            for row in cursor.fetchall():
                trend.append({
                    'timestamp': row[0],
                    'health_score': row[1],
                    'total_issues': row[2],
                    'critical_issues': row[3],
                    'high_issues': row[4]
                })
            
            conn.close()
            
            return trend
            
        except Exception as e:
            logging.error(f"[SelfRepairLog] Error getting health trend: {e}")
            return []
    
    def _update_repair_pattern(self, goal: str, approach: str, success: bool,
                              score: float, duration: float, side_effects: List[str]):
        """Update repair pattern statistics for learning"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Normalize goal and approach for pattern matching
            goal_pattern = self._normalize_for_pattern(goal)
            approach_pattern = self._normalize_for_pattern(approach)
            
            pattern_id = f"{goal_pattern}:{approach_pattern}"
            
            # Check if pattern exists
            cursor.execute("""
                SELECT success_count, failure_count, average_score, average_duration,
                       common_side_effects, effectiveness_trend
                FROM repair_patterns 
                WHERE pattern_id = ?
            """, (pattern_id,))
            
            existing = cursor.fetchone()
            
            if existing:
                # Update existing pattern
                success_count, failure_count, avg_score, avg_duration, side_effects_json, trend_json = existing
                
                new_success = success_count + (1 if success else 0)
                new_failure = failure_count + (0 if success else 1)
                total_attempts = new_success + new_failure
                
                # Update averages
                new_avg_score = ((avg_score * (total_attempts - 1)) + score) / total_attempts
                new_avg_duration = ((avg_duration * (total_attempts - 1)) + duration) / total_attempts
                
                # Update side effects
                existing_effects = json.loads(side_effects_json or '[]')
                all_effects = existing_effects + side_effects
                from collections import Counter
                effect_counts = Counter(all_effects)
                common_effects = [effect for effect, count in effect_counts.most_common(5)]
                
                # Update effectiveness trend
                trend = json.loads(trend_json or '[]')
                trend.append(score)
                if len(trend) > 20:  # Keep last 20 scores
                    trend = trend[-20:]
                
                cursor.execute("""
                    UPDATE repair_patterns 
                    SET success_count = ?, failure_count = ?, average_score = ?,
                        average_duration = ?, common_side_effects = ?,
                        effectiveness_trend = ?, last_attempted = ?, updated_at = ?
                    WHERE pattern_id = ?
                """, (
                    new_success, new_failure, new_avg_score, new_avg_duration,
                    json.dumps(common_effects), json.dumps(trend),
                    datetime.now().isoformat(), datetime.now().isoformat(),
                    pattern_id
                ))
                
                if success:
                    cursor.execute("""
                        UPDATE repair_patterns 
                        SET last_successful = ?, best_score = CASE WHEN ? > best_score THEN ? ELSE best_score END
                        WHERE pattern_id = ?
                    """, (datetime.now().isoformat(), score, score, pattern_id))
                else:
                    cursor.execute("""
                        UPDATE repair_patterns 
                        SET worst_score = CASE WHEN ? < worst_score THEN ? ELSE worst_score END
                        WHERE pattern_id = ?
                    """, (score, score, pattern_id))
            
            else:
                # Create new pattern
                cursor.execute("""
                    INSERT INTO repair_patterns (
                        pattern_id, goal_pattern, approach_pattern, success_count,
                        failure_count, average_score, average_duration, last_attempted,
                        last_successful, best_score, worst_score, common_side_effects,
                        effectiveness_trend, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    pattern_id, goal_pattern, approach_pattern,
                    1 if success else 0, 0 if success else 1,
                    score, duration, datetime.now().isoformat(),
                    datetime.now().isoformat() if success else None,
                    score, score, json.dumps(side_effects),
                    json.dumps([score]), datetime.now().isoformat(),
                    datetime.now().isoformat()
                ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logging.warning(f"[SelfRepairLog] Could not update repair pattern: {e}")
    
    def _generate_insight(self, goal: str, approach: str, score: float, 
                         result: str, side_effects: List[str]):
        """Generate insights from significant repair outcomes"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            insight_id = str(uuid.uuid4())
            
            if score >= 0.8:
                # Success pattern insight
                insight_type = "success_pattern"
                pattern_desc = f"Approach '{approach}' highly effective for '{self._normalize_for_pattern(goal)}'"
                recommendation = f"Prioritize '{approach}' for similar repair goals"
                confidence = min(0.9, score)
                
            elif score <= 0.2:
                # Failure pattern insight
                insight_type = "failure_pattern"
                pattern_desc = f"Approach '{approach}' ineffective for '{self._normalize_for_pattern(goal)}'"
                recommendation = f"Avoid '{approach}' for similar repair goals"
                confidence = 1.0 - score
                
            else:
                return  # No insight for moderate scores
            
            # Check for side effect patterns
            if side_effects:
                side_effect_insight_id = str(uuid.uuid4())
                cursor.execute("""
                    INSERT INTO repair_insights (
                        insight_id, insight_type, pattern_description, confidence,
                        evidence_count, recommendation, discovered_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    side_effect_insight_id, "side_effect_pattern",
                    f"Approach '{approach}' may cause: {', '.join(side_effects)}",
                    0.6, 1, f"Monitor for side effects when using '{approach}'",
                    datetime.now().isoformat()
                ))
            
            cursor.execute("""
                INSERT INTO repair_insights (
                    insight_id, insight_type, pattern_description, confidence,
                    evidence_count, recommendation, discovered_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                insight_id, insight_type, pattern_desc, confidence, 1,
                recommendation, datetime.now().isoformat()
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logging.warning(f"[SelfRepairLog] Could not generate insight: {e}")
    
    def _normalize_for_pattern(self, text: str) -> str:
        """Normalize text for pattern matching"""
        # Simple normalization - could be enhanced with NLP
        normalized = text.lower()
        
        # Replace specific component names with generic terms
        patterns = [
            (r'\b[\w_]+\.py\b', 'module'),
            (r'\b[\w_]+_agent\b', 'agent'),
            (r'\b[\w_]+_engine\b', 'engine'),
            (r'\b[\w_]+_system\b', 'system'),
        ]
        
        import re
        for pattern, replacement in patterns:
            normalized = re.sub(pattern, replacement, normalized)
        
        return normalized.strip()