#!/usr/bin/env python3
"""
Memory Autocleaner (Garbage Collector) for MeRNSTA

This module implements a MemoryCleaner daemon that automatically cleans up:
- Orphaned facts (unlinked, low score, unused)
- Dead contradictions (negated or resolved)
- Duplicates with high cosine similarity
- Compresses or removes them as appropriate
"""

import sqlite3
import logging
import json
import time
import threading
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
import numpy as np
from collections import defaultdict
import hashlib

from .enhanced_memory_model import EnhancedTripletFact


@dataclass
class CleanupAction:
    """Represents a cleanup action taken by the memory cleaner."""
    action_id: str
    action_type: str  # 'remove', 'compress', 'deactivate'
    target_type: str  # 'fact', 'contradiction', 'duplicate'
    target_id: str
    reason: str
    timestamp: float = 0.0
    memory_saved: int = 0  # Bytes saved
    confidence_impact: float = 0.0  # Impact on overall confidence
    
    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CleanupAction':
        """Create from dictionary."""
        return cls(**data)


class MemoryCleaner:
    """
    MemoryCleaner daemon that automatically cleans up memory.
    
    Features:
    - Scans for orphaned facts (unlinked, low score, unused)
    - Removes dead contradictions (negated or resolved)
    - Compresses duplicates with high cosine similarity
    - Logs all cleanup actions
    - Configurable cleanup thresholds
    """
    
    def __init__(self, db_path: str = "enhanced_memory.db", 
                 log_path: str = "memory_clean_log.jsonl"):
        self.db_path = db_path
        self.log_path = log_path
        self.cleanup_interval = 300  # 5 minutes
        self.task_counter = 0
        self.cleanup_threshold = 10  # Cleanup every 10 tasks
        
        # Cleanup thresholds
        self.orphan_score_threshold = 0.3  # Facts below this score are candidates
        self.orphan_age_threshold = 7 * 24 * 3600  # 7 days in seconds
        self.duplicate_similarity_threshold = 0.95  # Facts above this similarity are duplicates
        self.contradiction_resolution_threshold = 0.8  # Contradictions above this are resolved
        
        # Statistics
        self.total_cleanups = 0
        self.total_memory_saved = 0
        self.cleanup_history: List[CleanupAction] = []
        
        # Initialize database
        self._init_database()
        
        # Start daemon thread
        self.running = False
        self.cleanup_thread = None
    
    def _init_database(self):
        """Initialize cleanup log database schema."""
        # Create log file if it doesn't exist
        log_file = Path(self.log_path)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize SQLite table for cleanup tracking
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS cleanup_log (
                action_id TEXT PRIMARY KEY,
                action_type TEXT NOT NULL,
                target_type TEXT NOT NULL,
                target_id TEXT NOT NULL,
                reason TEXT NOT NULL,
                timestamp REAL NOT NULL,
                memory_saved INTEGER DEFAULT 0,
                confidence_impact REAL DEFAULT 0.0,
                INDEX(timestamp),
                INDEX(action_type),
                INDEX(target_type)
            )
        """)
        
        conn.commit()
        conn.close()
    
    def start_daemon(self):
        """Start the cleanup daemon thread."""
        if self.running:
            logging.warning("[MemoryCleaner] Daemon already running")
            return
        
        self.running = True
        self.cleanup_thread = threading.Thread(target=self._daemon_loop, daemon=True)
        self.cleanup_thread.start()
        logging.info("[MemoryCleaner] Daemon started")
    
    def stop_daemon(self):
        """Stop the cleanup daemon thread."""
        self.running = False
        if self.cleanup_thread:
            self.cleanup_thread.join(timeout=5)
        logging.info("[MemoryCleaner] Daemon stopped")
    
    def _daemon_loop(self):
        """Main daemon loop for periodic cleanup."""
        while self.running:
            try:
                time.sleep(self.cleanup_interval)
                if self.running:
                    self.perform_cleanup()
            except Exception as e:
                logging.error(f"[MemoryCleaner] Daemon error: {e}")
    
    def increment_task_counter(self):
        """Increment task counter and trigger cleanup if needed."""
        self.task_counter += 1
        if self.task_counter % self.cleanup_threshold == 0:
            self.perform_cleanup()
    
    def perform_cleanup(self) -> Dict[str, Any]:
        """
        Perform a complete memory cleanup cycle.
        
        Returns:
            Dictionary with cleanup statistics
        """
        logging.info(f"[MemoryCleaner] Starting cleanup cycle (task #{self.task_counter})")
        
        start_time = time.time()
        cleanup_stats = {
            'orphaned_facts_removed': 0,
            'dead_contradictions_removed': 0,
            'duplicates_compressed': 0,
            'memory_saved': 0,
            'confidence_impact': 0.0,
            'duration': 0.0
        }
        
        try:
            # Clean up orphaned facts
            orphaned_stats = self._cleanup_orphaned_facts()
            cleanup_stats.update(orphaned_stats)
            
            # Clean up dead contradictions
            contradiction_stats = self._cleanup_dead_contradictions()
            cleanup_stats.update(contradiction_stats)
            
            # Compress duplicates
            duplicate_stats = self._compress_duplicates()
            cleanup_stats.update(duplicate_stats)
            
            # Update statistics
            self.total_cleanups += 1
            self.total_memory_saved += cleanup_stats['memory_saved']
            
            cleanup_stats['duration'] = time.time() - start_time
            
            # Log cleanup summary
            self._log_cleanup_summary(cleanup_stats)
            
            logging.info(f"[MemoryCleaner] Cleanup completed: {cleanup_stats}")
            
        except Exception as e:
            logging.error(f"[MemoryCleaner] Cleanup error: {e}")
            cleanup_stats['error'] = str(e)
        
        return cleanup_stats
    
    def _cleanup_orphaned_facts(self) -> Dict[str, Any]:
        """Clean up orphaned facts (unlinked, low score, unused)."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Find orphaned facts
        cursor.execute("""
            SELECT id, subject, predicate, object, confidence, timestamp, 
                   contradiction, volatility_score, embedding
            FROM enhanced_facts 
            WHERE active = TRUE 
            AND confidence < ? 
            AND timestamp < ?
            AND (token_ids IS NULL OR token_ids = '')
        """, (self.orphan_score_threshold, time.time() - self.orphan_age_threshold))
        
        orphaned_facts = cursor.fetchall()
        
        removed_count = 0
        memory_saved = 0
        confidence_impact = 0.0
        
        for fact_data in orphaned_facts:
            fact_id, subject, predicate, object_val, confidence, timestamp, contradiction, volatility_score, embedding = fact_data
            
            # Calculate memory saved
            fact_size = len(subject) + len(predicate) + len(object_val)
            if embedding:
                fact_size += len(embedding)
            memory_saved += fact_size
            
            # Calculate confidence impact
            confidence_impact -= confidence * 0.1  # Small negative impact
            
            # Remove fact
            cursor.execute("DELETE FROM enhanced_facts WHERE id = ?", (fact_id,))
            
            # Log action
            action = CleanupAction(
                action_id=f"cleanup_{int(time.time())}_{fact_id}",
                action_type="remove",
                target_type="fact",
                target_id=str(fact_id),
                reason=f"Orphaned fact: low confidence ({confidence:.2f}), old ({timestamp})",
                memory_saved=fact_size,
                confidence_impact=-confidence * 0.1
            )
            self._log_cleanup_action(action)
            
            removed_count += 1
        
        conn.commit()
        conn.close()
        
        return {
            'orphaned_facts_removed': removed_count,
            'memory_saved': memory_saved,
            'confidence_impact': confidence_impact
        }
    
    def _cleanup_dead_contradictions(self) -> Dict[str, Any]:
        """Clean up dead contradictions (negated or resolved)."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Find resolved contradictions
        cursor.execute("""
            SELECT id, subject, predicate, object, confidence, contradiction_score
            FROM enhanced_facts 
            WHERE active = TRUE 
            AND contradiction = TRUE
            AND contradiction_score > ?
        """, (self.contradiction_resolution_threshold,))
        
        resolved_contradictions = cursor.fetchall()
        
        removed_count = 0
        memory_saved = 0
        confidence_impact = 0.0
        
        for fact_data in resolved_contradictions:
            fact_id, subject, predicate, object_val, confidence, contradiction_score = fact_data
            
            # Calculate memory saved
            fact_size = len(subject) + len(predicate) + len(object_val)
            memory_saved += fact_size
            
            # Calculate confidence impact (positive - removing contradictions improves confidence)
            confidence_impact += contradiction_score * 0.2
            
            # Remove contradiction
            cursor.execute("DELETE FROM enhanced_facts WHERE id = ?", (fact_id,))
            
            # Log action
            action = CleanupAction(
                action_id=f"cleanup_{int(time.time())}_{fact_id}",
                action_type="remove",
                target_type="contradiction",
                target_id=str(fact_id),
                reason=f"Resolved contradiction: score {contradiction_score:.2f}",
                memory_saved=fact_size,
                confidence_impact=contradiction_score * 0.2
            )
            self._log_cleanup_action(action)
            
            removed_count += 1
        
        conn.commit()
        conn.close()
        
        return {
            'dead_contradictions_removed': removed_count,
            'memory_saved': memory_saved,
            'confidence_impact': confidence_impact
        }
    
    def _compress_duplicates(self) -> Dict[str, Any]:
        """Compress duplicates with high cosine similarity."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get all facts with embeddings
        cursor.execute("""
            SELECT id, subject, predicate, object, confidence, embedding
            FROM enhanced_facts 
            WHERE active = TRUE 
            AND embedding IS NOT NULL 
            AND embedding != ''
        """)
        
        facts_with_embeddings = cursor.fetchall()
        
        compressed_count = 0
        memory_saved = 0
        confidence_impact = 0.0
        
        # Find duplicate groups
        duplicate_groups = self._find_duplicate_groups(facts_with_embeddings)
        
        for group in duplicate_groups:
            if len(group) < 2:
                continue
            
            # Keep the fact with highest confidence, remove others
            best_fact = max(group, key=lambda x: x[4])  # x[4] is confidence
            duplicates = [f for f in group if f[0] != best_fact[0]]  # All except best
            
            group_memory_saved = 0
            group_confidence_impact = 0.0
            
            for duplicate in duplicates:
                fact_id, subject, predicate, object_val, confidence, embedding = duplicate
                
                # Calculate memory saved
                fact_size = len(subject) + len(predicate) + len(object_val) + len(embedding)
                group_memory_saved += fact_size
                
                # Small confidence impact (removing duplicates slightly improves confidence)
                group_confidence_impact += 0.05
                
                # Remove duplicate
                cursor.execute("DELETE FROM enhanced_facts WHERE id = ?", (fact_id,))
                
                # Log action
                action = CleanupAction(
                    action_id=f"cleanup_{int(time.time())}_{fact_id}",
                    action_type="compress",
                    target_type="duplicate",
                    target_id=str(fact_id),
                    reason=f"Duplicate of fact {best_fact[0]} (similarity > {self.duplicate_similarity_threshold})",
                    memory_saved=fact_size,
                    confidence_impact=0.05
                )
                self._log_cleanup_action(action)
            
            compressed_count += len(duplicates)
            memory_saved += group_memory_saved
            confidence_impact += group_confidence_impact
        
        conn.commit()
        conn.close()
        
        return {
            'duplicates_compressed': compressed_count,
            'memory_saved': memory_saved,
            'confidence_impact': confidence_impact
        }
    
    def _find_duplicate_groups(self, facts: List[Tuple]) -> List[List[Tuple]]:
        """Find groups of duplicate facts based on cosine similarity."""
        groups = []
        used_facts = set()
        
        for i, fact1 in enumerate(facts):
            if i in used_facts:
                continue
            
            # Start a new group
            group = [fact1]
            used_facts.add(i)
            
            # Find similar facts
            for j, fact2 in enumerate(facts):
                if j in used_facts:
                    continue
                
                if self._facts_are_duplicates(fact1, fact2):
                    group.append(fact2)
                    used_facts.add(j)
            
            if len(group) > 1:  # Only add groups with duplicates
                groups.append(group)
        
        return groups
    
    def _facts_are_duplicates(self, fact1: Tuple, fact2: Tuple) -> bool:
        """Check if two facts are duplicates based on cosine similarity."""
        try:
            embedding1 = json.loads(fact1[5])  # fact1[5] is embedding
            embedding2 = json.loads(fact2[5])  # fact2[5] is embedding
            
            similarity = self._cosine_similarity(embedding1, embedding2)
            return similarity >= self.duplicate_similarity_threshold
            
        except (json.JSONDecodeError, ValueError, IndexError):
            return False
    
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
    
    def _log_cleanup_action(self, action: CleanupAction):
        """Log a cleanup action to both database and file."""
        # Log to database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO cleanup_log 
            (action_id, action_type, target_type, target_id, reason, timestamp, 
             memory_saved, confidence_impact)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            action.action_id, action.action_type, action.target_type, action.target_id,
            action.reason, action.timestamp, action.memory_saved, action.confidence_impact
        ))
        
        conn.commit()
        conn.close()
        
        # Log to file
        with open(self.log_path, 'a') as f:
            f.write(json.dumps(action.to_dict()) + '\n')
        
        # Add to history
        self.cleanup_history.append(action)
        
        # Keep only recent history
        if len(self.cleanup_history) > 1000:
            self.cleanup_history = self.cleanup_history[-1000:]
    
    def _log_cleanup_summary(self, stats: Dict[str, Any]):
        """Log a cleanup summary."""
        summary = {
            'timestamp': time.time(),
            'task_counter': self.task_counter,
            'stats': stats,
            'total_cleanups': self.total_cleanups,
            'total_memory_saved': self.total_memory_saved
        }
        
        with open(self.log_path, 'a') as f:
            f.write(json.dumps({'summary': summary}) + '\n')
    
    def get_cleanup_log(self, limit: Optional[int] = None) -> List[CleanupAction]:
        """Get cleanup log entries."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        query = """
            SELECT action_id, action_type, target_type, target_id, reason, timestamp,
                   memory_saved, confidence_impact
            FROM cleanup_log 
            ORDER BY timestamp DESC
        """
        
        if limit:
            query += f" LIMIT {limit}"
        
        cursor.execute(query)
        
        actions = []
        for row in cursor.fetchall():
            action = CleanupAction(
                action_id=row[0],
                action_type=row[1],
                target_type=row[2],
                target_id=row[3],
                reason=row[4],
                timestamp=row[5],
                memory_saved=row[6],
                confidence_impact=row[7]
            )
            actions.append(action)
        
        conn.close()
        return actions
    
    def get_cleanup_statistics(self) -> Dict[str, Any]:
        """Get comprehensive cleanup statistics."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Total cleanup actions
        cursor.execute("SELECT COUNT(*) FROM cleanup_log")
        total_actions = cursor.fetchone()[0]
        
        # Actions by type
        cursor.execute("""
            SELECT action_type, COUNT(*) 
            FROM cleanup_log 
            GROUP BY action_type
        """)
        actions_by_type = dict(cursor.fetchall())
        
        # Total memory saved
        cursor.execute("SELECT SUM(memory_saved) FROM cleanup_log")
        total_memory_saved = cursor.fetchone()[0] or 0
        
        # Total confidence impact
        cursor.execute("SELECT SUM(confidence_impact) FROM cleanup_log")
        total_confidence_impact = cursor.fetchone()[0] or 0.0
        
        # Recent activity (last 24 hours)
        cursor.execute("""
            SELECT COUNT(*) FROM cleanup_log 
            WHERE timestamp > ?
        """, (time.time() - 24 * 3600,))
        recent_actions = cursor.fetchone()[0]
        
        # Most common reasons
        cursor.execute("""
            SELECT reason, COUNT(*) 
            FROM cleanup_log 
            GROUP BY reason 
            ORDER BY COUNT(*) DESC 
            LIMIT 5
        """)
        common_reasons = cursor.fetchall()
        
        conn.close()
        
        return {
            'total_actions': total_actions,
            'actions_by_type': actions_by_type,
            'total_memory_saved': total_memory_saved,
            'total_confidence_impact': total_confidence_impact,
            'recent_actions_24h': recent_actions,
            'common_reasons': [
                {'reason': row[0], 'count': row[1]}
                for row in common_reasons
            ],
            'total_cleanups': self.total_cleanups,
            'cleanup_threshold': self.cleanup_threshold,
            'cleanup_interval': self.cleanup_interval
        }
    
    def get_memory_usage_stats(self) -> Dict[str, Any]:
        """Get current memory usage statistics."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Total facts
        cursor.execute("SELECT COUNT(*) FROM enhanced_facts WHERE active = TRUE")
        total_facts = cursor.fetchone()[0]
        
        # Facts by confidence level
        cursor.execute("""
            SELECT 
                CASE 
                    WHEN confidence >= 0.8 THEN 'high'
                    WHEN confidence >= 0.5 THEN 'medium'
                    ELSE 'low'
                END as confidence_level,
                COUNT(*)
            FROM enhanced_facts 
            WHERE active = TRUE 
            GROUP BY confidence_level
        """)
        facts_by_confidence = dict(cursor.fetchall())
        
        # Contradictions
        cursor.execute("SELECT COUNT(*) FROM enhanced_facts WHERE active = TRUE AND contradiction = TRUE")
        total_contradictions = cursor.fetchone()[0]
        
        # Average confidence
        cursor.execute("SELECT AVG(confidence) FROM enhanced_facts WHERE active = TRUE")
        avg_confidence = cursor.fetchone()[0] or 0.0
        
        # Average volatility
        cursor.execute("SELECT AVG(volatility_score) FROM enhanced_facts WHERE active = TRUE")
        avg_volatility = cursor.fetchone()[0] or 0.0
        
        conn.close()
        
        return {
            'total_facts': total_facts,
            'facts_by_confidence': facts_by_confidence,
            'total_contradictions': total_contradictions,
            'average_confidence': avg_confidence,
            'average_volatility': avg_volatility
        }
    
    def set_cleanup_thresholds(self, orphan_score: Optional[float] = None,
                              orphan_age: Optional[float] = None,
                              duplicate_similarity: Optional[float] = None,
                              contradiction_resolution: Optional[float] = None):
        """Update cleanup thresholds."""
        if orphan_score is not None:
            self.orphan_score_threshold = orphan_score
        if orphan_age is not None:
            self.orphan_age_threshold = orphan_age
        if duplicate_similarity is not None:
            self.duplicate_similarity_threshold = duplicate_similarity
        if contradiction_resolution is not None:
            self.contradiction_resolution_threshold = contradiction_resolution
        
        logging.info(f"[MemoryCleaner] Updated thresholds: orphan_score={self.orphan_score_threshold}, "
                    f"orphan_age={self.orphan_age_threshold}, duplicate_similarity={self.duplicate_similarity_threshold}, "
                    f"contradiction_resolution={self.contradiction_resolution_threshold}")
    
    def set_cleanup_schedule(self, interval: Optional[int] = None, threshold: Optional[int] = None):
        """Update cleanup schedule."""
        if interval is not None:
            self.cleanup_interval = interval
        if threshold is not None:
            self.cleanup_threshold = threshold
        
        logging.info(f"[MemoryCleaner] Updated schedule: interval={self.cleanup_interval}s, "
                    f"threshold={self.cleanup_threshold} tasks") 