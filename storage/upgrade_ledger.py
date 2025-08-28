#!/usr/bin/env python3
"""
Upgrade Ledger for MeRNSTA Self-Upgrading System

Maintains a comprehensive audit trail of all code upgrades, file versions,
and self-modification activities. Provides rollback capabilities and
historical analysis.
"""

import os
import sqlite3
import hashlib
import logging
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from contextlib import contextmanager

from .db_utils import get_db_connection, execute_query


class UpgradeLedger:
    """
    Persistent storage for tracking all upgrade operations and file changes.
    
    Capabilities:
    - Track file versions with checksums
    - Log upgrade proposals and outcomes
    - Store rollback information
    - Maintain upgrade statistics
    - Provide historical analysis
    """
    
    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or "upgrade_ledger.db"
        self.project_root = Path(os.getcwd())
        self._init_database()
    
    def _init_database(self) -> None:
        """Initialize the upgrade ledger database schema."""
        with self._get_connection() as conn:
            # File versions table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS file_versions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    file_path TEXT NOT NULL,
                    content_hash TEXT NOT NULL,
                    file_size INTEGER NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    version_number INTEGER DEFAULT 1,
                    backup_location TEXT,
                    UNIQUE(file_path, content_hash)
                )
            """)
            
            # Upgrade scans table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS upgrade_scans (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    scan_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    target_path TEXT,
                    total_files INTEGER DEFAULT 0,
                    analyzable_files INTEGER DEFAULT 0,
                    total_suggestions INTEGER DEFAULT 0,
                    queued_suggestions INTEGER DEFAULT 0,
                    scan_duration_seconds REAL,
                    analysis_results TEXT,  -- JSON
                    scan_metadata TEXT      -- JSON
                )
            """)
            
            # Upgrade executions table  
            conn.execute("""
                CREATE TABLE IF NOT EXISTS upgrade_executions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    upgrade_id TEXT UNIQUE NOT NULL,
                    suggestion_id TEXT,
                    upgrade_type TEXT NOT NULL,
                    suggestion_data TEXT,      -- JSON
                    execution_result TEXT,     -- JSON
                    started_at TIMESTAMP,
                    completed_at TIMESTAMP,
                    status TEXT NOT NULL,      -- pending, in_progress, completed, failed, rolled_back
                    success BOOLEAN DEFAULT FALSE,
                    errors TEXT,               -- JSON array
                    affected_files TEXT,       -- JSON array
                    backup_location TEXT,
                    risk_level TEXT,
                    priority INTEGER DEFAULT 5
                )
            """)
            
            # File changes table (detailed change tracking)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS file_changes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    upgrade_execution_id INTEGER NOT NULL,
                    file_path TEXT NOT NULL,
                    change_type TEXT NOT NULL,  -- create, modify, delete
                    old_hash TEXT,
                    new_hash TEXT,
                    old_size INTEGER,
                    new_size INTEGER,
                    lines_added INTEGER DEFAULT 0,
                    lines_removed INTEGER DEFAULT 0,
                    backup_path TEXT,
                    change_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(upgrade_execution_id) REFERENCES upgrade_executions(id)
                )
            """)
            
            # Rollback operations table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS rollback_operations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    upgrade_execution_id INTEGER NOT NULL,
                    rollback_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    rollback_reason TEXT,
                    restored_files TEXT,       -- JSON array
                    success BOOLEAN DEFAULT FALSE,
                    rollback_errors TEXT,      -- JSON array
                    FOREIGN KEY(upgrade_execution_id) REFERENCES upgrade_executions(id)
                )
            """)
            
            # Upgrade learning table (for AI improvement)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS upgrade_learning (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    upgrade_type TEXT NOT NULL,
                    pattern_signature TEXT,
                    success_rate REAL DEFAULT 0.0,
                    avg_execution_time REAL DEFAULT 0.0,
                    common_failures TEXT,      -- JSON array
                    success_factors TEXT,      -- JSON array
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    sample_count INTEGER DEFAULT 0
                )
            """)
            
            # Create indexes for performance
            conn.execute("CREATE INDEX IF NOT EXISTS idx_file_versions_path ON file_versions(file_path)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_file_versions_hash ON file_versions(content_hash)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_upgrade_executions_type ON upgrade_executions(upgrade_type)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_upgrade_executions_status ON upgrade_executions(status)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_file_changes_upgrade ON file_changes(upgrade_execution_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_file_changes_path ON file_changes(file_path)")
    
    @contextmanager
    def _get_connection(self):
        """Get a database connection with proper cleanup."""
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()
    
    def log_scan(self, analysis_results: Dict[str, Any], scan_metadata: Dict[str, Any]) -> int:
        """
        Log an architecture scan operation.
        
        Args:
            analysis_results: Results from ArchitectAnalyzer
            scan_metadata: Additional scan information
            
        Returns:
            Scan record ID
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO upgrade_scans (
                    target_path, total_files, analyzable_files, total_suggestions,
                    queued_suggestions, analysis_results, scan_metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                analysis_results.get("analyzed_path"),
                len(analysis_results.get("modules", {})),
                len([m for m in analysis_results.get("modules", {}).values() if m.get("analyzable", True)]),
                len(analysis_results.get("upgrade_suggestions", [])),
                scan_metadata.get("queued_upgrades", 0),
                json.dumps(analysis_results),
                json.dumps(scan_metadata)
            ))
            
            scan_id = cursor.lastrowid
            conn.commit()
            
            logging.info(f"[UpgradeLedger] Logged scan {scan_id}")
            return scan_id
    
    def log_upgrade_execution(self, upgrade_id: str, suggestion: Dict[str, Any], 
                            execution_result: Dict[str, Any]) -> int:
        """
        Log an upgrade execution operation.
        
        Args:
            upgrade_id: Unique identifier for the upgrade
            suggestion: Original upgrade suggestion
            execution_result: Result from CodeRefactorer
            
        Returns:
            Execution record ID
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Insert main execution record
            cursor.execute("""
                INSERT INTO upgrade_executions (
                    upgrade_id, suggestion_id, upgrade_type, suggestion_data,
                    execution_result, started_at, completed_at, status, success,
                    errors, affected_files, backup_location, risk_level, priority
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                upgrade_id,
                suggestion.get("id"),
                suggestion.get("type"),
                json.dumps(suggestion),
                json.dumps(execution_result),
                execution_result.get("started_at"),
                execution_result.get("completed_at"),
                "completed" if execution_result.get("success") else "failed",
                execution_result.get("success", False),
                json.dumps(execution_result.get("errors", [])),
                json.dumps(suggestion.get("affected_modules", [])),
                execution_result.get("backup_location"),
                suggestion.get("risk_level", "medium"),
                suggestion.get("priority", 5)
            ))
            
            execution_id = cursor.lastrowid
            
            # Log individual file changes
            for change in execution_result.get("changes", []):
                self._log_file_change(conn, execution_id, change)
            
            # Update file versions
            self._update_file_versions(conn, execution_result.get("changes", []))
            
            conn.commit()
            
            logging.info(f"[UpgradeLedger] Logged upgrade execution {upgrade_id} (record {execution_id})")
            return execution_id
    
    def log_rollback(self, upgrade_id: str, rollback_result: Dict[str, Any]) -> int:
        """
        Log a rollback operation.
        
        Args:
            upgrade_id: ID of the upgrade being rolled back
            rollback_result: Result of the rollback operation
            
        Returns:
            Rollback record ID
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Find the upgrade execution
            cursor.execute("SELECT id FROM upgrade_executions WHERE upgrade_id = ?", (upgrade_id,))
            execution_row = cursor.fetchone()
            
            if not execution_row:
                raise ValueError(f"Upgrade execution not found: {upgrade_id}")
            
            execution_id = execution_row[0]
            
            # Insert rollback record
            cursor.execute("""
                INSERT INTO rollback_operations (
                    upgrade_execution_id, rollback_reason, restored_files,
                    success, rollback_errors
                ) VALUES (?, ?, ?, ?, ?)
            """, (
                execution_id,
                rollback_result.get("reason", "Manual rollback"),
                json.dumps(rollback_result.get("restored_files", [])),
                rollback_result.get("success", False),
                json.dumps(rollback_result.get("errors", []))
            ))
            
            rollback_id = cursor.lastrowid
            
            # Update execution status
            if rollback_result.get("success"):
                cursor.execute(
                    "UPDATE upgrade_executions SET status = 'rolled_back' WHERE id = ?",
                    (execution_id,)
                )
            
            conn.commit()
            
            logging.info(f"[UpgradeLedger] Logged rollback {rollback_id} for upgrade {upgrade_id}")
            return rollback_id
    
    def get_file_history(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Get version history for a specific file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            List of file version records
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM file_versions 
                WHERE file_path = ? 
                ORDER BY created_at DESC
            """, (file_path,))
            
            return [dict(row) for row in cursor.fetchall()]
    
    def get_upgrade_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Get recent upgrade history.
        
        Args:
            limit: Maximum number of records to return
            
        Returns:
            List of upgrade execution records
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM upgrade_executions 
                ORDER BY started_at DESC 
                LIMIT ?
            """, (limit,))
            
            executions = []
            for row in cursor.fetchall():
                execution = dict(row)
                # Parse JSON fields
                for field in ['suggestion_data', 'execution_result', 'errors', 'affected_files']:
                    if execution[field]:
                        try:
                            execution[field] = json.loads(execution[field])
                        except:
                            pass
                executions.append(execution)
            
            return executions
    
    def get_upgrade_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive upgrade statistics.
        
        Returns:
            Dictionary with various statistics
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            stats = {}
            
            # Overall statistics
            cursor.execute("SELECT COUNT(*) FROM upgrade_executions")
            stats["total_upgrades"] = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM upgrade_executions WHERE success = 1")
            stats["successful_upgrades"] = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM rollback_operations")
            stats["total_rollbacks"] = cursor.fetchone()[0]
            
            # Success rate
            if stats["total_upgrades"] > 0:
                stats["success_rate"] = stats["successful_upgrades"] / stats["total_upgrades"]
            else:
                stats["success_rate"] = 0.0
            
            # By upgrade type
            cursor.execute("""
                SELECT upgrade_type, COUNT(*) as total, 
                       SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as successful
                FROM upgrade_executions 
                GROUP BY upgrade_type
            """)
            
            type_stats = {}
            for row in cursor.fetchall():
                upgrade_type = row[0]
                total = row[1]
                successful = row[2]
                type_stats[upgrade_type] = {
                    "total": total,
                    "successful": successful,
                    "success_rate": successful / total if total > 0 else 0.0
                }
            
            stats["by_type"] = type_stats
            
            # Recent activity (last 30 days)
            cursor.execute("""
                SELECT COUNT(*) FROM upgrade_executions 
                WHERE started_at >= datetime('now', '-30 days')
            """)
            stats["recent_upgrades_30d"] = cursor.fetchone()[0]
            
            # File change statistics
            cursor.execute("SELECT COUNT(*) FROM file_changes")
            stats["total_file_changes"] = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(DISTINCT file_path) FROM file_changes")
            stats["unique_files_changed"] = cursor.fetchone()[0]
            
            return stats
    
    def analyze_failure_patterns(self) -> Dict[str, Any]:
        """
        Analyze patterns in upgrade failures to improve future success.
        
        Returns:
            Analysis of failure patterns and recommendations
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Get failed upgrades
            cursor.execute("""
                SELECT upgrade_type, errors, suggestion_data, execution_result
                FROM upgrade_executions 
                WHERE success = 0
            """)
            
            failures = []
            for row in cursor.fetchall():
                failure = {
                    "type": row[0],
                    "errors": json.loads(row[1]) if row[1] else [],
                    "suggestion": json.loads(row[2]) if row[2] else {},
                    "result": json.loads(row[3]) if row[3] else {}
                }
                failures.append(failure)
            
            analysis = {
                "total_failures": len(failures),
                "failure_by_type": {},
                "common_errors": {},
                "recommendations": []
            }
            
            # Group by type
            for failure in failures:
                upgrade_type = failure["type"]
                if upgrade_type not in analysis["failure_by_type"]:
                    analysis["failure_by_type"][upgrade_type] = []
                analysis["failure_by_type"][upgrade_type].append(failure)
            
            # Find common error patterns
            error_counts = {}
            for failure in failures:
                for error in failure["errors"]:
                    error_type = self._categorize_error(error)
                    error_counts[error_type] = error_counts.get(error_type, 0) + 1
            
            analysis["common_errors"] = dict(sorted(error_counts.items(), 
                                                   key=lambda x: x[1], reverse=True))
            
            # Generate recommendations
            if "syntax_error" in error_counts and error_counts["syntax_error"] > 2:
                analysis["recommendations"].append("Improve syntax validation before executing upgrades")
            
            if "test_failure" in error_counts and error_counts["test_failure"] > 2:
                analysis["recommendations"].append("Enhance test coverage and validation")
            
            high_failure_types = [t for t, failures in analysis["failure_by_type"].items() 
                                 if len(failures) > 3]
            if high_failure_types:
                analysis["recommendations"].append(f"Review upgrade strategies for: {', '.join(high_failure_types)}")
            
            return analysis
    
    def cleanup_old_records(self, days_to_keep: int = 90) -> Dict[str, int]:
        """
        Clean up old records to prevent database bloat.
        
        Args:
            days_to_keep: Number of days of history to keep
            
        Returns:
            Count of records removed by table
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            cutoff_date = f"datetime('now', '-{days_to_keep} days')"
            removed_counts = {}
            
            # Remove old scans
            cursor.execute(f"DELETE FROM upgrade_scans WHERE scan_timestamp < {cutoff_date}")
            removed_counts["scans"] = cursor.rowcount
            
            # Remove old file versions (keep at least 2 versions per file)
            cursor.execute(f"""
                DELETE FROM file_versions 
                WHERE created_at < {cutoff_date}
                AND id NOT IN (
                    SELECT id FROM (
                        SELECT id, ROW_NUMBER() OVER (PARTITION BY file_path ORDER BY created_at DESC) as rn
                        FROM file_versions
                    ) WHERE rn <= 2
                )
            """)
            removed_counts["file_versions"] = cursor.rowcount
            
            # Remove old executions and their related records
            cursor.execute(f"SELECT id FROM upgrade_executions WHERE started_at < {cutoff_date}")
            old_execution_ids = [row[0] for row in cursor.fetchall()]
            
            if old_execution_ids:
                id_list = ','.join(map(str, old_execution_ids))
                
                cursor.execute(f"DELETE FROM file_changes WHERE upgrade_execution_id IN ({id_list})")
                removed_counts["file_changes"] = cursor.rowcount
                
                cursor.execute(f"DELETE FROM rollback_operations WHERE upgrade_execution_id IN ({id_list})")
                removed_counts["rollbacks"] = cursor.rowcount
                
                cursor.execute(f"DELETE FROM upgrade_executions WHERE id IN ({id_list})")
                removed_counts["executions"] = cursor.rowcount
            else:
                removed_counts.update({"file_changes": 0, "rollbacks": 0, "executions": 0})
            
            conn.commit()
            
            total_removed = sum(removed_counts.values())
            logging.info(f"[UpgradeLedger] Cleaned up {total_removed} old records")
            
            return removed_counts
    
    def _log_file_change(self, conn: sqlite3.Connection, execution_id: int, 
                        change: Dict[str, Any]) -> None:
        """Log a single file change."""
        cursor = conn.cursor()
        
        old_hash, new_hash = None, None
        old_size, new_size = None, None
        
        # Calculate hashes and sizes
        if change["type"] == "modify" and "original_path" in change:
            original_path = Path(change["original_path"])
            if original_path.exists():
                with open(original_path, 'rb') as f:
                    old_content = f.read()
                old_hash = hashlib.sha256(old_content).hexdigest()
                old_size = len(old_content)
        
        if "content" in change:
            new_content = change["content"].encode('utf-8')
            new_hash = hashlib.sha256(new_content).hexdigest()
            new_size = len(new_content)
        
        cursor.execute("""
            INSERT INTO file_changes (
                upgrade_execution_id, file_path, change_type, old_hash, new_hash,
                old_size, new_size, backup_path
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            execution_id,
            change.get("path", ""),
            change.get("type", "unknown"),
            old_hash,
            new_hash,
            old_size,
            new_size,
            change.get("backup_path")
        ))
    
    def _update_file_versions(self, conn: sqlite3.Connection, 
                            changes: List[Dict[str, Any]]) -> None:
        """Update file version tracking."""
        cursor = conn.cursor()
        
        for change in changes:
            if "content" in change and change.get("path"):
                file_path = change["path"]
                content = change["content"].encode('utf-8')
                content_hash = hashlib.sha256(content).hexdigest()
                file_size = len(content)
                
                # Check if this version already exists
                cursor.execute(
                    "SELECT id FROM file_versions WHERE file_path = ? AND content_hash = ?",
                    (file_path, content_hash)
                )
                
                if not cursor.fetchone():
                    # Get next version number
                    cursor.execute(
                        "SELECT COALESCE(MAX(version_number), 0) + 1 FROM file_versions WHERE file_path = ?",
                        (file_path,)
                    )
                    version_number = cursor.fetchone()[0]
                    
                    cursor.execute("""
                        INSERT INTO file_versions (
                            file_path, content_hash, file_size, version_number, backup_location
                        ) VALUES (?, ?, ?, ?, ?)
                    """, (
                        file_path,
                        content_hash,
                        file_size,
                        version_number,
                        change.get("backup_path")
                    ))
    
    def _categorize_error(self, error_message: str) -> str:
        """Categorize error messages for pattern analysis."""
        error_lower = error_message.lower()
        
        if "syntax" in error_lower:
            return "syntax_error"
        elif "test" in error_lower and "fail" in error_lower:
            return "test_failure"
        elif "import" in error_lower:
            return "import_error"
        elif "timeout" in error_lower:
            return "timeout_error"
        elif "permission" in error_lower:
            return "permission_error"
        else:
            return "other_error"