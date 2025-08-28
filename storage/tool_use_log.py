#!/usr/bin/env python3
"""
Tool Use Logger for MeRNSTA - Phase 13: Command Routing & Execution

SQLite-based logging system for all command executions, tool usage, and system operations.
Provides comprehensive audit trail with query capabilities.
"""

import sqlite3
import json
import logging
from typing import Dict, Any, List, Optional, Union
from datetime import datetime, timedelta
from pathlib import Path
import threading
from config.settings import get_config


class ToolUseLogger:
    """
    Comprehensive logging system for tool and command usage.
    
    Features:
    - SQLite-based persistent storage
    - Full command execution logging
    - Query and search capabilities
    - Thread-safe operations
    - Automatic table creation
    """
    
    def __init__(self, db_path: str = None):
        """
        Initialize the tool use logger.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.config = get_config().get('tool_use_logger', {})
        self.db_path = db_path or self.config.get('db_path', 'tool_use_log.db')
        self.max_log_entries = self.config.get('max_log_entries', 10000)
        self.auto_cleanup_days = self.config.get('auto_cleanup_days', 30)
        
        # Thread safety
        self._lock = threading.Lock()
        
        # For memory databases, we need to maintain a connection
        self._memory_conn = None
        
        # Initialize logging
        self.logger = logging.getLogger('tool_use_logger')
        self.logger.setLevel(logging.INFO)
        
        # Initialize database
        self._init_database()
        
        self.logger.info(f"[ToolUseLogger] Initialized with db_path={self.db_path}")
    
    def _init_database(self):
        """Initialize SQLite database and create tables."""
        try:
            # Ensure database directory exists for file-based databases
            if self.db_path != ':memory:':
                db_dir = Path(self.db_path).parent
                db_dir.mkdir(parents=True, exist_ok=True)
            
            # For memory databases, maintain a persistent connection
            if self.db_path == ':memory:':
                self._memory_conn = sqlite3.connect(self.db_path, check_same_thread=False)
                conn = self._memory_conn
            else:
                conn = sqlite3.connect(self.db_path)
                
            cursor = conn.cursor()
            
            # Create main command execution log table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS command_executions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    executor TEXT NOT NULL,
                    command TEXT NOT NULL,
                    command_type TEXT,
                    args TEXT,
                    success BOOLEAN NOT NULL,
                    output TEXT,
                    error TEXT,
                    exit_code INTEGER,
                    duration REAL,
                    metadata TEXT
                )
            ''')
            
            # Create tool usage log table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS tool_usage (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    executor TEXT NOT NULL,
                    tool_name TEXT NOT NULL,
                    tool_args TEXT,
                    success BOOLEAN NOT NULL,
                    result TEXT,
                    error TEXT,
                    duration REAL,
                    metadata TEXT
                )
            ''')
            
            # Create agent method calls table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS agent_method_calls (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    executor TEXT NOT NULL,
                    agent_name TEXT NOT NULL,
                    method_name TEXT NOT NULL,
                    arguments TEXT,
                    success BOOLEAN NOT NULL,
                    result TEXT,
                    error TEXT,
                    duration REAL,
                    metadata TEXT
                )
            ''')
            
            # Create system events table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS system_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    description TEXT,
                    data TEXT,
                    metadata TEXT
                )
            ''')
            
            # Create indexes for better query performance
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_cmd_timestamp ON command_executions(timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_cmd_executor ON command_executions(executor)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_cmd_type ON command_executions(command_type)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_tool_timestamp ON tool_usage(timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_tool_name ON tool_usage(tool_name)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_agent_timestamp ON agent_method_calls(timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_agent_name ON agent_method_calls(agent_name)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_event_timestamp ON system_events(timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_event_type ON system_events(event_type)')
            
            conn.commit()
            
            # Close connection for file-based databases (we'll open new ones as needed)
            if self.db_path != ':memory:':
                conn.close()
                
        except Exception as e:
            self.logger.error(f"[ToolUseLogger] Database initialization error: {e}")
            raise
    
    def _get_connection(self):
        """Get appropriate database connection."""
        if self.db_path == ':memory:':
            return self._memory_conn
        else:
            return sqlite3.connect(self.db_path)
    
    def log_command_execution(self, command: str, executor: str, result: Dict[str, Any], 
                            timestamp: datetime = None, metadata: Dict[str, Any] = None):
        """
        Log a command execution.
        
        Args:
            command: The command that was executed
            executor: Who executed the command (user, agent, system)
            result: Execution result dictionary
            timestamp: When the command was executed
            metadata: Additional metadata
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        with self._lock:
            try:
                conn = self._get_connection()
                cursor = conn.cursor()
                
                cursor.execute('''
                    INSERT INTO command_executions 
                    (timestamp, executor, command, command_type, args, success, 
                     output, error, exit_code, duration, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    timestamp.isoformat(),
                    executor,
                    command,
                    self._determine_command_type(command),
                    json.dumps(result.get('args', {})),
                    result.get('success', False),
                    result.get('output', ''),
                    result.get('error', ''),
                    result.get('exit_code'),
                    result.get('duration', 0.0),
                    json.dumps(metadata or {})
                ))
                
                conn.commit()
                
                # Close connection for file-based databases
                if self.db_path != ':memory:':
                    conn.close()
                    
            except Exception as e:
                self.logger.error(f"[ToolUseLogger] Error logging command execution: {e}")
    
    def log_tool_usage(self, tool_name: str, executor: str, tool_args: str,
                      success: bool, result: Any = None, error: str = None,
                      duration: float = 0.0, timestamp: datetime = None,
                      metadata: Dict[str, Any] = None):
        """
        Log tool usage.
        
        Args:
            tool_name: Name of the tool used
            executor: Who used the tool
            tool_args: Arguments passed to the tool
            success: Whether the tool execution was successful
            result: Tool execution result
            error: Error message if any
            duration: Execution duration
            timestamp: When the tool was used
            metadata: Additional metadata
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        with self._lock:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    
                    cursor.execute('''
                        INSERT INTO tool_usage 
                        (timestamp, executor, tool_name, tool_args, success, 
                         result, error, duration, metadata)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        timestamp.isoformat(),
                        executor,
                        tool_name,
                        tool_args,
                        success,
                        str(result) if result is not None else '',
                        error or '',
                        duration,
                        json.dumps(metadata or {})
                    ))
                    
                    conn.commit()
                    
            except Exception as e:
                self.logger.error(f"[ToolUseLogger] Error logging tool usage: {e}")
    
    def log_agent_method_call(self, agent_name: str, method_name: str, executor: str,
                            arguments: str, success: bool, result: Any = None,
                            error: str = None, duration: float = 0.0,
                            timestamp: datetime = None, metadata: Dict[str, Any] = None):
        """
        Log agent method calls.
        
        Args:
            agent_name: Name of the agent
            method_name: Method that was called
            executor: Who called the method
            arguments: Arguments passed to the method
            success: Whether the call was successful
            result: Method call result
            error: Error message if any
            duration: Execution duration
            timestamp: When the method was called
            metadata: Additional metadata
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        with self._lock:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    
                    cursor.execute('''
                        INSERT INTO agent_method_calls 
                        (timestamp, executor, agent_name, method_name, arguments, 
                         success, result, error, duration, metadata)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        timestamp.isoformat(),
                        executor,
                        agent_name,
                        method_name,
                        arguments,
                        success,
                        str(result) if result is not None else '',
                        error or '',
                        duration,
                        json.dumps(metadata or {})
                    ))
                    
                    conn.commit()
                    
            except Exception as e:
                self.logger.error(f"[ToolUseLogger] Error logging agent method call: {e}")
    
    def log_system_event(self, event_type: str, description: str, data: Any = None,
                        timestamp: datetime = None, metadata: Dict[str, Any] = None):
        """
        Log system events.
        
        Args:
            event_type: Type of event (startup, shutdown, error, etc.)
            description: Human-readable description
            data: Event data
            timestamp: When the event occurred
            metadata: Additional metadata
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        with self._lock:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    
                    cursor.execute('''
                        INSERT INTO system_events 
                        (timestamp, event_type, description, data, metadata)
                        VALUES (?, ?, ?, ?, ?)
                    ''', (
                        timestamp.isoformat(),
                        event_type,
                        description,
                        json.dumps(data) if data is not None else '',
                        json.dumps(metadata or {})
                    ))
                    
                    conn.commit()
                    
            except Exception as e:
                self.logger.error(f"[ToolUseLogger] Error logging system event: {e}")
    
    def query_logs(self, query: str = "", limit: int = 50, 
                  log_type: str = "all") -> List[Dict[str, Any]]:
        """
        Query execution logs.
        
        Args:
            query: Search query (searches in command, output, error fields)
            limit: Maximum number of results
            log_type: Type of logs to query (command, tool, agent, event, all)
            
        Returns:
            List of log entries
        """
        results = []
        
        with self._lock:
            try:
                conn = self._get_connection()
                conn.row_factory = sqlite3.Row  # Enable column access by name
                cursor = conn.cursor()
                
                if log_type == "all" or log_type == "command":
                    # Query command executions
                    if query:
                        cursor.execute('''
                            SELECT 'command' as log_type, * FROM command_executions 
                            WHERE command LIKE ? OR output LIKE ? OR error LIKE ?
                            ORDER BY timestamp DESC LIMIT ?
                        ''', (f'%{query}%', f'%{query}%', f'%{query}%', limit))
                    else:
                        cursor.execute('''
                            SELECT 'command' as log_type, * FROM command_executions 
                            ORDER BY timestamp DESC LIMIT ?
                        ''', (limit,))
                    
                    # Convert results and fix boolean fields
                    raw_results = [dict(row) for row in cursor.fetchall()]
                    for result in raw_results:
                        if 'success' in result:
                            result['success'] = bool(result['success'])
                    results.extend(raw_results)
                
                if log_type == "all" or log_type == "tool":
                    # Query tool usage
                    if query:
                        cursor.execute('''
                            SELECT 'tool' as log_type, * FROM tool_usage 
                            WHERE tool_name LIKE ? OR result LIKE ? OR error LIKE ?
                            ORDER BY timestamp DESC LIMIT ?
                        ''', (f'%{query}%', f'%{query}%', f'%{query}%', limit))
                    else:
                        cursor.execute('''
                            SELECT 'tool' as log_type, * FROM tool_usage 
                            ORDER BY timestamp DESC LIMIT ?
                        ''', (limit,))
                    
                    # Convert results and fix boolean fields
                    raw_results = [dict(row) for row in cursor.fetchall()]
                    for result in raw_results:
                        if 'success' in result:
                            result['success'] = bool(result['success'])
                    results.extend(raw_results)
                
                if log_type == "all" or log_type == "agent":
                    # Query agent method calls
                    if query:
                        cursor.execute('''
                            SELECT 'agent' as log_type, * FROM agent_method_calls 
                            WHERE agent_name LIKE ? OR method_name LIKE ? OR result LIKE ?
                            ORDER BY timestamp DESC LIMIT ?
                        ''', (f'%{query}%', f'%{query}%', f'%{query}%', limit))
                    else:
                        cursor.execute('''
                            SELECT 'agent' as log_type, * FROM agent_method_calls 
                            ORDER BY timestamp DESC LIMIT ?
                        ''', (limit,))
                    
                    # Convert results and fix boolean fields
                    raw_results = [dict(row) for row in cursor.fetchall()]
                    for result in raw_results:
                        if 'success' in result:
                            result['success'] = bool(result['success'])
                    results.extend(raw_results)
                
                if log_type == "all" or log_type == "event":
                    # Query system events
                    if query:
                        cursor.execute('''
                            SELECT 'event' as log_type, * FROM system_events 
                            WHERE event_type LIKE ? OR description LIKE ?
                            ORDER BY timestamp DESC LIMIT ?
                        ''', (f'%{query}%', f'%{query}%', limit))
                    else:
                        cursor.execute('''
                            SELECT 'event' as log_type, * FROM system_events 
                            ORDER BY timestamp DESC LIMIT ?
                        ''', (limit,))
                    
                    # Convert results and fix boolean fields
                    raw_results = [dict(row) for row in cursor.fetchall()]
                    for result in raw_results:
                        if 'success' in result:
                            result['success'] = bool(result['success'])
                    results.extend(raw_results)
                
                # Sort results by timestamp
                results.sort(key=lambda x: x['timestamp'], reverse=True)
                
                # Close connection for file-based databases
                if self.db_path != ':memory:':
                    conn.close()
                    
                return results[:limit]
                
            except Exception as e:
                self.logger.error(f"[ToolUseLogger] Error querying logs: {e}")
                return []
    
    def get_execution_stats(self, hours: int = 24) -> Dict[str, Any]:
        """
        Get execution statistics for the specified time period.
        
        Args:
            hours: Time period in hours
            
        Returns:
            Statistics dictionary
        """
        since = datetime.now() - timedelta(hours=hours)
        stats = {
            'period_hours': hours,
            'since': since.isoformat(),
            'command_executions': 0,
            'successful_commands': 0,
            'failed_commands': 0,
            'tool_usage': 0,
            'agent_method_calls': 0,
            'system_events': 0,
            'top_commands': [],
            'top_tools': [],
            'top_agents': []
        }
        
        with self._lock:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    
                    # Command execution stats
                    cursor.execute('''
                        SELECT COUNT(*) FROM command_executions 
                        WHERE timestamp >= ?
                    ''', (since.isoformat(),))
                    stats['command_executions'] = cursor.fetchone()[0]
                    
                    cursor.execute('''
                        SELECT COUNT(*) FROM command_executions 
                        WHERE timestamp >= ? AND success = 1
                    ''', (since.isoformat(),))
                    stats['successful_commands'] = cursor.fetchone()[0]
                    
                    cursor.execute('''
                        SELECT COUNT(*) FROM command_executions 
                        WHERE timestamp >= ? AND success = 0
                    ''', (since.isoformat(),))
                    stats['failed_commands'] = cursor.fetchone()[0]
                    
                    # Tool usage stats
                    cursor.execute('''
                        SELECT COUNT(*) FROM tool_usage 
                        WHERE timestamp >= ?
                    ''', (since.isoformat(),))
                    stats['tool_usage'] = cursor.fetchone()[0]
                    
                    # Agent method call stats
                    cursor.execute('''
                        SELECT COUNT(*) FROM agent_method_calls 
                        WHERE timestamp >= ?
                    ''', (since.isoformat(),))
                    stats['agent_method_calls'] = cursor.fetchone()[0]
                    
                    # System event stats
                    cursor.execute('''
                        SELECT COUNT(*) FROM system_events 
                        WHERE timestamp >= ?
                    ''', (since.isoformat(),))
                    stats['system_events'] = cursor.fetchone()[0]
                    
                    # Top commands
                    cursor.execute('''
                        SELECT command_type, COUNT(*) as count FROM command_executions 
                        WHERE timestamp >= ? AND command_type IS NOT NULL
                        GROUP BY command_type ORDER BY count DESC LIMIT 5
                    ''', (since.isoformat(),))
                    stats['top_commands'] = [{'type': row[0], 'count': row[1]} for row in cursor.fetchall()]
                    
                    # Top tools
                    cursor.execute('''
                        SELECT tool_name, COUNT(*) as count FROM tool_usage 
                        WHERE timestamp >= ?
                        GROUP BY tool_name ORDER BY count DESC LIMIT 5
                    ''', (since.isoformat(),))
                    stats['top_tools'] = [{'name': row[0], 'count': row[1]} for row in cursor.fetchall()]
                    
                    # Top agents
                    cursor.execute('''
                        SELECT agent_name, COUNT(*) as count FROM agent_method_calls 
                        WHERE timestamp >= ?
                        GROUP BY agent_name ORDER BY count DESC LIMIT 5
                    ''', (since.isoformat(),))
                    stats['top_agents'] = [{'name': row[0], 'count': row[1]} for row in cursor.fetchall()]
                
                return stats
                
            except Exception as e:
                self.logger.error(f"[ToolUseLogger] Error getting execution stats: {e}")
                return stats
    
    def cleanup_old_logs(self, days: int = None):
        """
        Clean up old log entries.
        
        Args:
            days: Remove entries older than this many days
        """
        if days is None:
            days = self.auto_cleanup_days
        
        cutoff = datetime.now() - timedelta(days=days)
        
        with self._lock:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    
                    # Clean up command executions
                    cursor.execute('''
                        DELETE FROM command_executions WHERE timestamp < ?
                    ''', (cutoff.isoformat(),))
                    cmd_deleted = cursor.rowcount
                    
                    # Clean up tool usage
                    cursor.execute('''
                        DELETE FROM tool_usage WHERE timestamp < ?
                    ''', (cutoff.isoformat(),))
                    tool_deleted = cursor.rowcount
                    
                    # Clean up agent method calls
                    cursor.execute('''
                        DELETE FROM agent_method_calls WHERE timestamp < ?
                    ''', (cutoff.isoformat(),))
                    agent_deleted = cursor.rowcount
                    
                    # Clean up system events
                    cursor.execute('''
                        DELETE FROM system_events WHERE timestamp < ?
                    ''', (cutoff.isoformat(),))
                    event_deleted = cursor.rowcount
                    
                    conn.commit()
                    
                    self.logger.info(f"[ToolUseLogger] Cleaned up old logs: "
                                   f"commands={cmd_deleted}, tools={tool_deleted}, "
                                   f"agents={agent_deleted}, events={event_deleted}")
                
            except Exception as e:
                self.logger.error(f"[ToolUseLogger] Error cleaning up old logs: {e}")
    
    def _determine_command_type(self, command: str) -> str:
        """Determine the type of command for categorization."""
        if command.startswith('/run_shell'):
            return 'shell'
        elif command.startswith('/pip_install'):
            return 'pip'
        elif command.startswith('/run_tool'):
            return 'tool'
        elif command.startswith('/restart_self'):
            return 'restart'
        elif command.startswith('/agent_status'):
            return 'status'
        elif command.startswith('/tool_log'):
            return 'query'
        else:
            return 'agent_method'


# Global tool use logger instance
_tool_logger = None

def get_tool_logger() -> ToolUseLogger:
    """Get or create global tool use logger instance."""
    global _tool_logger
    if _tool_logger is None:
        _tool_logger = ToolUseLogger()
    return _tool_logger