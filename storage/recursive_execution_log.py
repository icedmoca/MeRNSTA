#!/usr/bin/env python3
"""
Recursive Execution Logger for MeRNSTA - Phase 14: Recursive Execution

Enhanced logging and memory system for tracking recursive execution attempts,
patterns, and outcomes. Provides analytics and learning capabilities for
improving autonomous code generation.
"""

import sqlite3
import json
import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass, asdict
from config.settings import get_config


@dataclass
class ExecutionSession:
    """Container for a complete execution session."""
    session_id: str
    goal: str
    initial_code: str
    filename: str
    start_time: datetime
    end_time: Optional[datetime]
    success: bool
    total_attempts: int
    winning_attempt: Optional[int]
    termination_reason: str
    duration: float
    metadata: Dict[str, Any]


@dataclass
class ExecutionAttempt:
    """Container for a single execution attempt within a session."""
    session_id: str
    attempt_number: int
    code_content: str
    write_success: bool
    execution_success: bool
    exit_code: Optional[int]
    output: str
    error: str
    duration: float
    confidence_score: float
    improvement_suggestions: List[str]
    timestamp: datetime


@dataclass
class ExecutionPattern:
    """Container for identified execution patterns."""
    pattern_id: str
    pattern_type: str
    pattern_description: str
    success_rate: float
    frequency: int
    confidence: float
    examples: List[str]
    improvement_suggestions: List[str]
    last_seen: datetime


class RecursiveExecutionLogger:
    """
    Enhanced logger for recursive execution tracking and analysis.
    
    Features:
    - Session and attempt tracking
    - Pattern recognition and learning
    - Success/failure analytics
    - Performance metrics
    - Improvement recommendations
    """
    
    def __init__(self, db_path: str = None):
        self.config = get_config().get('recursive_execution', {})
        self.db_path = db_path or self.config.get('log_db_path', 'recursive_execution.db')
        self.enable_pattern_learning = self.config.get('enable_pattern_learning', True)
        self.max_log_entries = self.config.get('max_log_entries', 10000)
        
        # Initialize database
        self._init_database()
        
        # Initialize logging
        self.logger = logging.getLogger('recursive_execution_log')
        self.logger.setLevel(logging.INFO)
        
        self.logger.info(f"[RecursiveExecutionLogger] Initialized with db_path={self.db_path}")
    
    def _init_database(self):
        """Initialize SQLite database and create tables."""
        try:
            # Ensure database directory exists
            if self.db_path != ':memory:':
                db_dir = Path(self.db_path).parent
                db_dir.mkdir(parents=True, exist_ok=True)
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Execution sessions table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS execution_sessions (
                        session_id TEXT PRIMARY KEY,
                        goal TEXT NOT NULL,
                        initial_code TEXT NOT NULL,
                        filename TEXT NOT NULL,
                        start_time TEXT NOT NULL,
                        end_time TEXT,
                        success BOOLEAN NOT NULL,
                        total_attempts INTEGER NOT NULL,
                        winning_attempt INTEGER,
                        termination_reason TEXT NOT NULL,
                        duration REAL NOT NULL,
                        metadata TEXT
                    )
                ''')
                
                # Execution attempts table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS execution_attempts (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id TEXT NOT NULL,
                        attempt_number INTEGER NOT NULL,
                        code_content TEXT NOT NULL,
                        write_success BOOLEAN NOT NULL,
                        execution_success BOOLEAN NOT NULL,
                        exit_code INTEGER,
                        output TEXT,
                        error TEXT,
                        duration REAL NOT NULL,
                        confidence_score REAL NOT NULL,
                        improvement_suggestions TEXT,
                        timestamp TEXT NOT NULL,
                        FOREIGN KEY (session_id) REFERENCES execution_sessions (session_id)
                    )
                ''')
                
                # Execution patterns table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS execution_patterns (
                        pattern_id TEXT PRIMARY KEY,
                        pattern_type TEXT NOT NULL,
                        pattern_description TEXT NOT NULL,
                        success_rate REAL NOT NULL,
                        frequency INTEGER NOT NULL,
                        confidence REAL NOT NULL,
                        examples TEXT,
                        improvement_suggestions TEXT,
                        last_seen TEXT NOT NULL
                    )
                ''')
                
                # Code improvement tracking table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS code_improvements (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id TEXT NOT NULL,
                        from_attempt INTEGER NOT NULL,
                        to_attempt INTEGER NOT NULL,
                        improvement_type TEXT NOT NULL,
                        improvement_description TEXT NOT NULL,
                        effectiveness_score REAL,
                        timestamp TEXT NOT NULL,
                        FOREIGN KEY (session_id) REFERENCES execution_sessions (session_id)
                    )
                ''')
                
                # Performance metrics table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS performance_metrics (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id TEXT NOT NULL,
                        metric_type TEXT NOT NULL,
                        metric_value REAL NOT NULL,
                        metric_unit TEXT,
                        timestamp TEXT NOT NULL,
                        FOREIGN KEY (session_id) REFERENCES execution_sessions (session_id)
                    )
                ''')
                
                # Create indexes for better query performance
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_sessions_success ON execution_sessions(success)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_sessions_start_time ON execution_sessions(start_time)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_attempts_session ON execution_attempts(session_id)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_attempts_success ON execution_attempts(execution_success)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_patterns_type ON execution_patterns(pattern_type)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_improvements_session ON code_improvements(session_id)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_metrics_session ON performance_metrics(session_id)')
                
                conn.commit()
                
        except Exception as e:
            self.logger.error(f"[RecursiveExecutionLogger] Database initialization error: {e}")
            raise
    
    def log_execution_session(self, session: ExecutionSession):
        """Log a complete execution session."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    INSERT OR REPLACE INTO execution_sessions 
                    (session_id, goal, initial_code, filename, start_time, end_time, 
                     success, total_attempts, winning_attempt, termination_reason, 
                     duration, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    session.session_id,
                    session.goal,
                    session.initial_code,
                    session.filename,
                    session.start_time.isoformat(),
                    session.end_time.isoformat() if session.end_time else None,
                    session.success,
                    session.total_attempts,
                    session.winning_attempt,
                    session.termination_reason,
                    session.duration,
                    json.dumps(session.metadata)
                ))
                
                conn.commit()
                
        except Exception as e:
            self.logger.error(f"[RecursiveExecutionLogger] Error logging session: {e}")
    
    def log_execution_attempt(self, attempt: ExecutionAttempt):
        """Log a single execution attempt."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    INSERT INTO execution_attempts 
                    (session_id, attempt_number, code_content, write_success, 
                     execution_success, exit_code, output, error, duration, 
                     confidence_score, improvement_suggestions, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    attempt.session_id,
                    attempt.attempt_number,
                    attempt.code_content,
                    attempt.write_success,
                    attempt.execution_success,
                    attempt.exit_code,
                    attempt.output,
                    attempt.error,
                    attempt.duration,
                    attempt.confidence_score,
                    json.dumps(attempt.improvement_suggestions),
                    attempt.timestamp.isoformat()
                ))
                
                conn.commit()
                
        except Exception as e:
            self.logger.error(f"[RecursiveExecutionLogger] Error logging attempt: {e}")
    
    def log_code_improvement(self, session_id: str, from_attempt: int, to_attempt: int,
                           improvement_type: str, description: str, effectiveness: float = None):
        """Log a code improvement between attempts."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    INSERT INTO code_improvements 
                    (session_id, from_attempt, to_attempt, improvement_type, 
                     improvement_description, effectiveness_score, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    session_id,
                    from_attempt,
                    to_attempt,
                    improvement_type,
                    description,
                    effectiveness,
                    datetime.now().isoformat()
                ))
                
                conn.commit()
                
        except Exception as e:
            self.logger.error(f"[RecursiveExecutionLogger] Error logging improvement: {e}")
    
    def log_performance_metric(self, session_id: str, metric_type: str, 
                             value: float, unit: str = None):
        """Log a performance metric for a session."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    INSERT INTO performance_metrics 
                    (session_id, metric_type, metric_value, metric_unit, timestamp)
                    VALUES (?, ?, ?, ?, ?)
                ''', (
                    session_id,
                    metric_type,
                    value,
                    unit,
                    datetime.now().isoformat()
                ))
                
                conn.commit()
                
        except Exception as e:
            self.logger.error(f"[RecursiveExecutionLogger] Error logging metric: {e}")
    
    def analyze_execution_patterns(self, days_back: int = 30) -> List[ExecutionPattern]:
        """Analyze execution patterns from recent sessions."""
        try:
            since = (datetime.now() - timedelta(days=days_back)).isoformat()
            patterns = []
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Pattern 1: Success rate by goal type
                cursor.execute('''
                    SELECT 
                        CASE 
                            WHEN LOWER(goal) LIKE '%test%' THEN 'testing'
                            WHEN LOWER(goal) LIKE '%optim%' THEN 'optimization'
                            WHEN LOWER(goal) LIKE '%fix%' OR LOWER(goal) LIKE '%repair%' THEN 'fixing'
                            WHEN LOWER(goal) LIKE '%script%' THEN 'scripting'
                            ELSE 'general'
                        END as goal_type,
                        COUNT(*) as total,
                        SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as successful,
                        AVG(total_attempts) as avg_attempts
                    FROM execution_sessions 
                    WHERE start_time >= ?
                    GROUP BY goal_type
                    HAVING total >= 2
                ''', (since,))
                
                for row in cursor.fetchall():
                    goal_type, total, successful, avg_attempts = row
                    success_rate = successful / total if total > 0 else 0
                    
                    patterns.append(ExecutionPattern(
                        pattern_id=f"goal_type_{goal_type}",
                        pattern_type="goal_type_success",
                        pattern_description=f"Success rate for {goal_type} goals",
                        success_rate=success_rate,
                        frequency=total,
                        confidence=min(0.9, total / 10.0),  # More data = higher confidence
                        examples=[goal_type],
                        improvement_suggestions=self._generate_pattern_suggestions(goal_type, success_rate, avg_attempts),
                        last_seen=datetime.now()
                    ))
                
                # Pattern 2: Common failure reasons
                cursor.execute('''
                    SELECT 
                        termination_reason,
                        COUNT(*) as frequency,
                        AVG(total_attempts) as avg_attempts
                    FROM execution_sessions 
                    WHERE start_time >= ? AND success = 0
                    GROUP BY termination_reason
                    ORDER BY frequency DESC
                    LIMIT 5
                ''', (since,))
                
                for row in cursor.fetchall():
                    reason, frequency, avg_attempts = row
                    
                    patterns.append(ExecutionPattern(
                        pattern_id=f"failure_{reason.replace(' ', '_').lower()}",
                        pattern_type="failure_pattern",
                        pattern_description=f"Common failure: {reason}",
                        success_rate=0.0,
                        frequency=frequency,
                        confidence=min(0.8, frequency / 5.0),
                        examples=[reason],
                        improvement_suggestions=self._generate_failure_suggestions(reason),
                        last_seen=datetime.now()
                    ))
                
                # Pattern 3: Improvement effectiveness
                cursor.execute('''
                    SELECT 
                        improvement_type,
                        COUNT(*) as frequency,
                        AVG(effectiveness_score) as avg_effectiveness
                    FROM code_improvements ci
                    JOIN execution_sessions es ON ci.session_id = es.session_id
                    WHERE es.start_time >= ? AND effectiveness_score IS NOT NULL
                    GROUP BY improvement_type
                    ORDER BY avg_effectiveness DESC
                ''', (since,))
                
                for row in cursor.fetchall():
                    improvement_type, frequency, avg_effectiveness = row
                    
                    patterns.append(ExecutionPattern(
                        pattern_id=f"improvement_{improvement_type}",
                        pattern_type="improvement_effectiveness",
                        pattern_description=f"Effectiveness of {improvement_type} improvements",
                        success_rate=avg_effectiveness or 0.0,
                        frequency=frequency,
                        confidence=min(0.8, frequency / 3.0),
                        examples=[improvement_type],
                        improvement_suggestions=[f"Consider using {improvement_type} improvements more frequently"],
                        last_seen=datetime.now()
                    ))
            
            # Store patterns in database for future reference
            self._store_patterns(patterns)
            
            return patterns
            
        except Exception as e:
            self.logger.error(f"[RecursiveExecutionLogger] Error analyzing patterns: {e}")
            return []
    
    def get_execution_statistics(self, days_back: int = 30) -> Dict[str, Any]:
        """Get comprehensive execution statistics."""
        try:
            since = (datetime.now() - timedelta(days=days_back)).isoformat()
            stats = {
                'period_days': days_back,
                'since': since,
                'total_sessions': 0,
                'successful_sessions': 0,
                'total_attempts': 0,
                'avg_attempts_per_session': 0,
                'success_rate': 0.0,
                'avg_session_duration': 0.0,
                'common_goals': [],
                'top_failure_reasons': [],
                'improvement_trends': []
            }
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Basic statistics
                cursor.execute('''
                    SELECT 
                        COUNT(*) as total,
                        SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as successful,
                        SUM(total_attempts) as attempts,
                        AVG(total_attempts) as avg_attempts,
                        AVG(duration) as avg_duration
                    FROM execution_sessions 
                    WHERE start_time >= ?
                ''', (since,))
                
                row = cursor.fetchone()
                if row:
                    total, successful, attempts, avg_attempts, avg_duration = row
                    stats.update({
                        'total_sessions': total or 0,
                        'successful_sessions': successful or 0,
                        'total_attempts': attempts or 0,
                        'avg_attempts_per_session': avg_attempts or 0,
                        'success_rate': (successful / total) if total > 0 else 0,
                        'avg_session_duration': avg_duration or 0
                    })
                
                # Common goals
                cursor.execute('''
                    SELECT goal, COUNT(*) as frequency
                    FROM execution_sessions 
                    WHERE start_time >= ?
                    GROUP BY goal
                    ORDER BY frequency DESC
                    LIMIT 5
                ''', (since,))
                
                stats['common_goals'] = [{'goal': row[0], 'frequency': row[1]} for row in cursor.fetchall()]
                
                # Top failure reasons
                cursor.execute('''
                    SELECT termination_reason, COUNT(*) as frequency
                    FROM execution_sessions 
                    WHERE start_time >= ? AND success = 0
                    GROUP BY termination_reason
                    ORDER BY frequency DESC
                    LIMIT 5
                ''', (since,))
                
                stats['top_failure_reasons'] = [{'reason': row[0], 'frequency': row[1]} for row in cursor.fetchall()]
                
                # Improvement trends
                cursor.execute('''
                    SELECT 
                        improvement_type,
                        COUNT(*) as frequency,
                        AVG(effectiveness_score) as avg_effectiveness
                    FROM code_improvements ci
                    JOIN execution_sessions es ON ci.session_id = es.session_id
                    WHERE es.start_time >= ?
                    GROUP BY improvement_type
                    ORDER BY frequency DESC
                    LIMIT 5
                ''', (since,))
                
                stats['improvement_trends'] = [{
                    'type': row[0], 
                    'frequency': row[1], 
                    'effectiveness': row[2] or 0
                } for row in cursor.fetchall()]
            
            return stats
            
        except Exception as e:
            self.logger.error(f"[RecursiveExecutionLogger] Error getting statistics: {e}")
            return {}
    
    def get_improvement_recommendations(self, session_id: str = None) -> List[str]:
        """Get improvement recommendations based on historical data."""
        try:
            recommendations = []
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                if session_id:
                    # Session-specific recommendations
                    cursor.execute('''
                        SELECT termination_reason, total_attempts, success
                        FROM execution_sessions 
                        WHERE session_id = ?
                    ''', (session_id,))
                    
                    row = cursor.fetchone()
                    if row:
                        reason, attempts, success = row
                        if not success:
                            recommendations.extend(self._generate_failure_suggestions(reason))
                        if attempts > 3:
                            recommendations.append("Consider breaking down complex goals into smaller steps")
                
                # General recommendations based on patterns
                cursor.execute('''
                    SELECT pattern_type, success_rate, improvement_suggestions
                    FROM execution_patterns
                    WHERE confidence > 0.5
                    ORDER BY last_seen DESC
                    LIMIT 10
                ''', ())
                
                for row in cursor.fetchall():
                    pattern_type, success_rate, suggestions_json = row
                    try:
                        suggestions = json.loads(suggestions_json) if suggestions_json else []
                        if success_rate < 0.6:  # Low success rate patterns
                            recommendations.extend(suggestions)
                    except json.JSONDecodeError:
                        pass
            
            # Remove duplicates and return
            return list(set(recommendations))
            
        except Exception as e:
            self.logger.error(f"[RecursiveExecutionLogger] Error getting recommendations: {e}")
            return []
    
    def _store_patterns(self, patterns: List[ExecutionPattern]):
        """Store identified patterns in database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                for pattern in patterns:
                    cursor.execute('''
                        INSERT OR REPLACE INTO execution_patterns 
                        (pattern_id, pattern_type, pattern_description, success_rate, 
                         frequency, confidence, examples, improvement_suggestions, last_seen)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        pattern.pattern_id,
                        pattern.pattern_type,
                        pattern.pattern_description,
                        pattern.success_rate,
                        pattern.frequency,
                        pattern.confidence,
                        json.dumps(pattern.examples),
                        json.dumps(pattern.improvement_suggestions),
                        pattern.last_seen.isoformat()
                    ))
                
                conn.commit()
                
        except Exception as e:
            self.logger.error(f"[RecursiveExecutionLogger] Error storing patterns: {e}")
    
    def _generate_pattern_suggestions(self, goal_type: str, success_rate: float, avg_attempts: float) -> List[str]:
        """Generate suggestions based on goal type patterns."""
        suggestions = []
        
        if success_rate < 0.5:
            suggestions.append(f"Low success rate for {goal_type} goals - consider simpler initial approaches")
        
        if avg_attempts > 4:
            suggestions.append(f"{goal_type.title()} goals often require multiple attempts - build in more time")
        
        if goal_type == 'testing':
            suggestions.append("Consider using established testing frameworks")
        elif goal_type == 'optimization':
            suggestions.append("Start with profiling before optimization")
        elif goal_type == 'fixing':
            suggestions.append("Analyze error patterns before attempting fixes")
        
        return suggestions
    
    def _generate_failure_suggestions(self, failure_reason: str) -> List[str]:
        """Generate suggestions based on failure reasons."""
        suggestions = []
        reason_lower = failure_reason.lower()
        
        if 'maximum attempts' in reason_lower:
            suggestions.extend([
                "Break complex problems into smaller, testable parts",
                "Start with simpler versions before adding complexity",
                "Consider increasing max attempts for complex goals"
            ])
        elif 'syntax' in reason_lower:
            suggestions.extend([
                "Use syntax checking before execution",
                "Start with basic syntax patterns",
                "Consider language-specific templates"
            ])
        elif 'import' in reason_lower:
            suggestions.extend([
                "Verify required modules are available",
                "Use standard library when possible",
                "Check import paths and names"
            ])
        elif 'timeout' in reason_lower:
            suggestions.extend([
                "Optimize code for faster execution",
                "Increase timeout limits for complex operations",
                "Break long-running tasks into steps"
            ])
        else:
            suggestions.append("Review error patterns and adjust approach accordingly")
        
        return suggestions


# Global logger instance
_recursive_logger = None

def get_recursive_execution_logger() -> RecursiveExecutionLogger:
    """Get or create global recursive execution logger instance."""
    global _recursive_logger
    if _recursive_logger is None:
        _recursive_logger = RecursiveExecutionLogger()
    return _recursive_logger