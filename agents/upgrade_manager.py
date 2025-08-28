#!/usr/bin/env python3
"""
UpgradeManager for MeRNSTA Self-Upgrading System

Orchestrates the self-upgrading process by coordinating between ArchitectAnalyzer
and CodeRefactorer, managing upgrade queues, and handling rollbacks.
"""

import os
import logging
import json
import time
import threading
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

from .base import BaseAgent
from .architect_analyzer import ArchitectAnalyzer
from .code_refactorer import CodeRefactorer


class UpgradeStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    ROLLED_BACK = "rolled_back"


@dataclass
class UpgradeTask:
    """Represents a single upgrade task in the queue."""
    id: str
    suggestion: Dict[str, Any]
    status: UpgradeStatus
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Optional[Dict[str, Any]] = None
    priority: int = 5
    retry_count: int = 0
    max_retries: int = 3


class UpgradeManager(BaseAgent):
    """
    Central manager for the self-upgrading system.
    
    Capabilities:
    - Schedules automatic architecture scans
    - Manages upgrade task queue with priorities
    - Coordinates between analyzer and refactorer
    - Handles rollbacks and recovery
    - Maintains audit trail of all changes
    - Learns from upgrade outcomes
    """
    
    def __init__(self):
        super().__init__("upgrade_manager")
        self.project_root = Path(os.getcwd())
        
        # Initialize component agents
        self.analyzer = ArchitectAnalyzer()
        self.refactorer = CodeRefactorer()
        
        # Upgrade queue and tracking
        self.upgrade_queue: List[UpgradeTask] = []
        self.completed_upgrades: List[UpgradeTask] = []
        self.upgrade_lock = threading.Lock()
        
        # Configuration
        self.config = self._load_upgrade_config()
        self.auto_scan_enabled = self.config.get("auto_scan_enabled", True)
        self.scan_interval_hours = self.config.get("scan_interval_hours", 168)  # Weekly
        self.max_concurrent_upgrades = self.config.get("max_concurrent_upgrades", 1)
        
        # State tracking
        self.last_scan_time = None
        self.active_upgrades = 0
        self.is_running = False
        
        # Learning system
        self.upgrade_patterns = {}
        self.success_metrics = {}
        
        # Import upgrade ledger
        try:
            from storage.upgrade_ledger import UpgradeLedger
            self.ledger = UpgradeLedger()
        except ImportError:
            logging.warning(f"[{self.name}] UpgradeLedger not available, using fallback")
            self.ledger = None
    
    def get_agent_instructions(self) -> str:
        """Return specialized instructions for the upgrade manager agent."""
        return (
            "You are the central upgrade management specialist for the self-upgrading system. "
            "Your role is to coordinate architectural improvements by scheduling scans, managing "
            "upgrade task queues, orchestrating between analyzer and refactorer agents, and ensuring "
            "safe deployment of changes. You handle rollbacks, maintain audit trails, and learn from "
            "upgrade outcomes to improve future operations. Focus on system stability, change management, "
            "and continuous improvement coordination."
        )
    
    def respond(self, message: str, context: Dict[str, Any] = None) -> str:
        """Generate upgrade management responses and coordinate system upgrades."""
        context = context or {}
        
        # Build memory context for upgrade patterns
        memory_context = self.get_memory_context(message)
        
        # Use LLM if available for complex upgrade coordination questions
        if self.llm_fallback:
            prompt = self.build_agent_prompt(message, memory_context)
            try:
                return self.llm_fallback.process(prompt)
            except Exception as e:
                logging.error(f"[{self.name}] LLM processing failed: {e}")
        
        # Handle upgrade management commands
        if "scan" in message.lower() or "analyze" in message.lower():
            try:
                result = self.trigger_scan()
                if result.get("success"):
                    return f"Architecture scan completed! Found {result.get('total_suggestions', 0)} suggestions, queued {result.get('queued_upgrades', 0)} upgrades."
                else:
                    return f"Scan failed: {result.get('error', 'Unknown error')}"
            except Exception as e:
                return f"Scan execution failed: {str(e)}"
        
        if "start" in message.lower() and "manager" in message.lower():
            try:
                self.start_manager()
                return "Upgrade manager started successfully. Automatic scanning is now enabled."
            except Exception as e:
                return f"Failed to start manager: {str(e)}"
        
        if "status" in message.lower() or "queue" in message.lower():
            queue_size = len(self.upgrade_queue)
            completed_count = len(self.completed_upgrades)
            return f"Upgrade queue status: {queue_size} pending tasks, {completed_count} completed upgrades. Manager running: {self.is_running}"
        
        return "I can manage system upgrades, coordinate scans, and handle upgrade queues. Try commands like 'trigger scan', 'start manager', or 'show status'."
    
    def start_manager(self) -> None:
        """Start the upgrade manager with automatic scanning."""
        if self.is_running:
            logging.warning(f"[{self.name}] Manager already running")
            return
        
        self.is_running = True
        logging.info(f"[{self.name}] Starting upgrade manager")
        
        # Start background thread for automatic operations
        if self.auto_scan_enabled:
            self.scheduler_thread = threading.Thread(target=self._scheduler_loop, daemon=True)
            self.scheduler_thread.start()
    
    def stop_manager(self) -> None:
        """Stop the upgrade manager."""
        self.is_running = False
        logging.info(f"[{self.name}] Stopping upgrade manager")
    
    def trigger_scan(self, target_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Manually trigger an architecture scan and queue suggested upgrades.
        
        Args:
            target_path: Specific path to scan, defaults to entire project
            
        Returns:
            Scan results and queued upgrades
        """
        logging.info(f"[{self.name}] Triggering manual scan")
        
        try:
            # Run architecture analysis
            analysis_results = self.analyzer.analyze_codebase(target_path)
            self.last_scan_time = datetime.now()
            
            # Queue upgrade suggestions
            queued_count = 0
            for suggestion in analysis_results.get("upgrade_suggestions", []):
                if self._should_queue_upgrade(suggestion):
                    task = self._create_upgrade_task(suggestion)
                    self._add_to_queue(task)
                    queued_count += 1
            
            scan_result = {
                "success": True,
                "scan_timestamp": self.last_scan_time.isoformat(),
                "total_suggestions": len(analysis_results.get("upgrade_suggestions", [])),
                "queued_upgrades": queued_count,
                "analysis_summary": self._summarize_analysis(analysis_results)
            }
            
            # Log to upgrade ledger
            if self.ledger:
                self.ledger.log_scan(analysis_results, scan_result)
            
            return scan_result
            
        except Exception as e:
            logging.error(f"[{self.name}] Scan failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "scan_timestamp": datetime.now().isoformat()
            }
    
    def execute_upgrade(self, upgrade_id: str) -> Dict[str, Any]:
        """
        Execute a specific upgrade by ID.
        
        Args:
            upgrade_id: ID of the upgrade task to execute
            
        Returns:
            Execution result
        """
        with self.upgrade_lock:
            task = self._find_task_by_id(upgrade_id)
            if not task:
                return {"success": False, "error": "Upgrade task not found"}
            
            if task.status != UpgradeStatus.PENDING:
                return {"success": False, "error": f"Task is not pending (current status: {task.status.value})"}
            
            if self.active_upgrades >= self.max_concurrent_upgrades:
                return {"success": False, "error": "Maximum concurrent upgrades reached"}
        
        return self._execute_upgrade_task(task)
    
    def get_upgrade_status(self) -> Dict[str, Any]:
        """Get current status of all upgrades."""
        with self.upgrade_lock:
            pending_tasks = [task for task in self.upgrade_queue if task.status == UpgradeStatus.PENDING]
            in_progress_tasks = [task for task in self.upgrade_queue if task.status == UpgradeStatus.IN_PROGRESS]
            
            return {
                "queue_length": len(pending_tasks),
                "in_progress": len(in_progress_tasks),
                "completed_today": len([task for task in self.completed_upgrades 
                                       if task.completed_at and 
                                       task.completed_at.date() == datetime.now().date()]),
                "last_scan": self.last_scan_time.isoformat() if self.last_scan_time else None,
                "next_auto_scan": self._calculate_next_scan_time(),
                "pending_tasks": [self._task_to_dict(task) for task in pending_tasks],
                "in_progress_tasks": [self._task_to_dict(task) for task in in_progress_tasks],
                "recent_completed": [self._task_to_dict(task) for task in self.completed_upgrades[-10:]]
            }
    
    def rollback_upgrade(self, upgrade_id: str) -> Dict[str, Any]:
        """
        Rollback a completed upgrade.
        
        Args:
            upgrade_id: ID of the upgrade to rollback
            
        Returns:
            Rollback result
        """
        logging.info(f"[{self.name}] Rolling back upgrade {upgrade_id}")
        
        # Find the completed upgrade
        task = self._find_completed_task_by_id(upgrade_id)
        if not task:
            return {"success": False, "error": "Completed upgrade not found"}
        
        if not task.result or not task.result.get("backup_location"):
            return {"success": False, "error": "No backup available for rollback"}
        
        try:
            # Restore from backup
            backup_path = Path(task.result["backup_location"])
            if not backup_path.exists():
                return {"success": False, "error": "Backup location not found"}
            
            # Restore all files from backup
            import shutil
            restored_files = []
            
            for change in task.result.get("changes", []):
                if change["type"] in ["modify", "create"]:
                    original_path = change.get("original_path")
                    if original_path:
                        backup_file = backup_path / Path(original_path).name
                        if backup_file.exists():
                            shutil.copy2(backup_file, original_path)
                            restored_files.append(original_path)
            
            # Update task status
            task.status = UpgradeStatus.ROLLED_BACK
            
            rollback_result = {
                "success": True,
                "restored_files": restored_files,
                "rollback_timestamp": datetime.now().isoformat()
            }
            
            # Log rollback
            if self.ledger:
                self.ledger.log_rollback(upgrade_id, rollback_result)
            
            return rollback_result
            
        except Exception as e:
            logging.error(f"[{self.name}] Rollback failed: {e}")
            return {"success": False, "error": str(e)}
    
    def get_upgrade_diff(self, upgrade_id: str) -> Dict[str, Any]:
        """
        Get detailed diff of what changed in an upgrade.
        
        Args:
            upgrade_id: ID of the upgrade
            
        Returns:
            Detailed diff information
        """
        task = self._find_task_by_id(upgrade_id) or self._find_completed_task_by_id(upgrade_id)
        if not task or not task.result:
            return {"success": False, "error": "Upgrade not found or no results available"}
        
        changes = task.result.get("changes", [])
        diff_info = {
            "upgrade_id": upgrade_id,
            "upgrade_type": task.suggestion.get("type"),
            "affected_modules": task.suggestion.get("affected_modules", []),
            "changes": []
        }
        
        for change in changes:
            change_info = {
                "type": change["type"],
                "path": change["path"],
                "lines_added": change.get("content", "").count('\n'),
            }
            
            if change["type"] == "modify" and "original_path" in change:
                # Calculate actual diff
                try:
                    original_file = Path(change["original_path"])
                    if original_file.exists():
                        with open(original_file, 'r') as f:
                            original_content = f.read()
                        
                        # Simple diff calculation
                        new_content = change.get("content", "")
                        change_info["diff"] = self._calculate_simple_diff(original_content, new_content)
                except Exception as e:
                    change_info["diff_error"] = str(e)
            
            diff_info["changes"].append(change_info)
        
        return {"success": True, "diff": diff_info}
    
    def learn_from_outcomes(self) -> Dict[str, Any]:
        """Analyze upgrade outcomes to improve future decisions."""
        logging.info(f"[{self.name}] Learning from upgrade outcomes")
        
        learning_data = {
            "total_upgrades": len(self.completed_upgrades),
            "success_rate": 0,
            "failure_patterns": {},
            "success_patterns": {},
            "recommendations": []
        }
        
        if not self.completed_upgrades:
            return learning_data
        
        successful_upgrades = [task for task in self.completed_upgrades 
                              if task.status == UpgradeStatus.COMPLETED]
        failed_upgrades = [task for task in self.completed_upgrades 
                          if task.status == UpgradeStatus.FAILED]
        
        learning_data["success_rate"] = len(successful_upgrades) / len(self.completed_upgrades)
        
        # Analyze failure patterns
        failure_types = {}
        for task in failed_upgrades:
            upgrade_type = task.suggestion.get("type", "unknown")
            if upgrade_type not in failure_types:
                failure_types[upgrade_type] = []
            failure_types[upgrade_type].append(task.result.get("errors", []))
        
        learning_data["failure_patterns"] = failure_types
        
        # Analyze success patterns
        success_types = {}
        for task in successful_upgrades:
            upgrade_type = task.suggestion.get("type", "unknown")
            if upgrade_type not in success_types:
                success_types[upgrade_type] = 0
            success_types[upgrade_type] += 1
        
        learning_data["success_patterns"] = success_types
        
        # Generate recommendations
        if learning_data["success_rate"] < 0.5:
            learning_data["recommendations"].append("Consider more conservative upgrade thresholds")
        
        for upgrade_type, failures in failure_types.items():
            if len(failures) > 2:
                learning_data["recommendations"].append(f"Review {upgrade_type} upgrade strategy - multiple failures detected")
        
        return learning_data
    
    def _scheduler_loop(self) -> None:
        """Background loop for automatic scanning and upgrade execution."""
        while self.is_running:
            try:
                # Check if it's time for automatic scan
                if self._should_run_automatic_scan():
                    self.trigger_scan()
                
                # Process upgrade queue
                self._process_upgrade_queue()
                
                # Sleep for a minute before next check
                time.sleep(60)
                
            except Exception as e:
                logging.error(f"[{self.name}] Scheduler loop error: {e}")
                time.sleep(60)
    
    def _should_run_automatic_scan(self) -> bool:
        """Check if automatic scan should be triggered."""
        if not self.auto_scan_enabled:
            return False
        
        if self.last_scan_time is None:
            return True
        
        next_scan_time = self.last_scan_time + timedelta(hours=self.scan_interval_hours)
        return datetime.now() >= next_scan_time
    
    def _process_upgrade_queue(self) -> None:
        """Process pending upgrades in the queue."""
        with self.upgrade_lock:
            if self.active_upgrades >= self.max_concurrent_upgrades:
                return
            
            # Find highest priority pending task
            pending_tasks = [task for task in self.upgrade_queue if task.status == UpgradeStatus.PENDING]
            if not pending_tasks:
                return
            
            # Sort by priority and creation time
            pending_tasks.sort(key=lambda t: (-t.priority, t.created_at))
            next_task = pending_tasks[0]
        
        # Execute the upgrade in a separate thread
        upgrade_thread = threading.Thread(
            target=self._execute_upgrade_task,
            args=(next_task,),
            daemon=True
        )
        upgrade_thread.start()
    
    def _execute_upgrade_task(self, task: UpgradeTask) -> Dict[str, Any]:
        """Execute a single upgrade task."""
        logging.info(f"[{self.name}] Executing upgrade task {task.id}")
        
        with self.upgrade_lock:
            task.status = UpgradeStatus.IN_PROGRESS
            task.started_at = datetime.now()
            self.active_upgrades += 1
        
        try:
            # Execute the refactoring
            refactor_result = self.refactorer.execute_refactor(task.suggestion)
            
            # Update task with results
            task.result = refactor_result
            task.completed_at = datetime.now()
            
            if refactor_result.get("success", False):
                task.status = UpgradeStatus.COMPLETED
                logging.info(f"[{self.name}] Upgrade task {task.id} completed successfully")
            else:
                task.status = UpgradeStatus.FAILED
                logging.warning(f"[{self.name}] Upgrade task {task.id} failed")
                
                # Consider retry if within limits
                if task.retry_count < task.max_retries:
                    task.retry_count += 1
                    task.status = UpgradeStatus.PENDING
                    task.started_at = None
                    logging.info(f"[{self.name}] Retrying upgrade task {task.id} (attempt {task.retry_count + 1})")
            
            # Log to upgrade ledger
            if self.ledger:
                self.ledger.log_upgrade_execution(task.id, task.suggestion, refactor_result)
            
            # Move to completed if final status
            if task.status in [UpgradeStatus.COMPLETED, UpgradeStatus.FAILED]:
                with self.upgrade_lock:
                    if task in self.upgrade_queue:
                        self.upgrade_queue.remove(task)
                    self.completed_upgrades.append(task)
            
            return {"success": task.status == UpgradeStatus.COMPLETED, "result": refactor_result}
            
        except Exception as e:
            logging.error(f"[{self.name}] Upgrade task {task.id} execution failed: {e}")
            task.status = UpgradeStatus.FAILED
            task.completed_at = datetime.now()
            task.result = {"success": False, "errors": [str(e)]}
            
            with self.upgrade_lock:
                if task in self.upgrade_queue:
                    self.upgrade_queue.remove(task)
                self.completed_upgrades.append(task)
            
            return {"success": False, "error": str(e)}
            
        finally:
            with self.upgrade_lock:
                self.active_upgrades -= 1
    
    def _should_queue_upgrade(self, suggestion: Dict[str, Any]) -> bool:
        """Determine if an upgrade suggestion should be queued."""
        # Filter based on configuration and learning
        risk_level = suggestion.get("risk_level", "medium")
        priority = suggestion.get("priority", 5)
        
        # Don't queue high-risk upgrades automatically
        if risk_level == "high" and not self.config.get("allow_high_risk_auto", False):
            return False
        
        # Don't queue if priority is too low
        min_priority = self.config.get("min_auto_priority", 3)
        if priority < min_priority:
            return False
        
        # Check if similar upgrade failed recently
        upgrade_type = suggestion.get("type")
        recent_failures = [task for task in self.completed_upgrades[-10:] 
                          if task.suggestion.get("type") == upgrade_type and 
                          task.status == UpgradeStatus.FAILED]
        
        if len(recent_failures) >= 2:
            logging.info(f"[{self.name}] Skipping {upgrade_type} upgrade due to recent failures")
            return False
        
        return True
    
    def _create_upgrade_task(self, suggestion: Dict[str, Any]) -> UpgradeTask:
        """Create an upgrade task from a suggestion."""
        task_id = f"upgrade_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{suggestion.get('id', 'unknown')}"
        
        return UpgradeTask(
            id=task_id,
            suggestion=suggestion,
            status=UpgradeStatus.PENDING,
            created_at=datetime.now(),
            priority=suggestion.get("priority", 5)
        )
    
    def _add_to_queue(self, task: UpgradeTask) -> None:
        """Add a task to the upgrade queue."""
        with self.upgrade_lock:
            self.upgrade_queue.append(task)
            # Sort queue by priority
            self.upgrade_queue.sort(key=lambda t: (-t.priority, t.created_at))
    
    def _find_task_by_id(self, task_id: str) -> Optional[UpgradeTask]:
        """Find a task in the current queue by ID."""
        for task in self.upgrade_queue:
            if task.id == task_id:
                return task
        return None
    
    def _find_completed_task_by_id(self, task_id: str) -> Optional[UpgradeTask]:
        """Find a completed task by ID."""
        for task in self.completed_upgrades:
            if task.id == task_id:
                return task
        return None
    
    def _task_to_dict(self, task: UpgradeTask) -> Dict[str, Any]:
        """Convert an UpgradeTask to a dictionary for serialization."""
        return {
            "id": task.id,
            "type": task.suggestion.get("type"),
            "title": task.suggestion.get("title"),
            "status": task.status.value,
            "priority": task.priority,
            "created_at": task.created_at.isoformat(),
            "started_at": task.started_at.isoformat() if task.started_at else None,
            "completed_at": task.completed_at.isoformat() if task.completed_at else None,
            "retry_count": task.retry_count,
            "affected_modules": task.suggestion.get("affected_modules", [])
        }
    
    def _summarize_analysis(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Create a summary of analysis results."""
        return {
            "total_modules": len(analysis_results.get("modules", {})),
            "analyzable_modules": len([m for m in analysis_results.get("modules", {}).values() 
                                     if m.get("analyzable", True)]),
            "circular_imports": len(analysis_results.get("global_issues", {}).get("circular_imports", [])),
            "architectural_violations": len(analysis_results.get("global_issues", {}).get("architectural_violations", [])),
            "duplicate_patterns": len(analysis_results.get("global_issues", {}).get("duplicate_patterns", [])),
            "total_suggestions": len(analysis_results.get("upgrade_suggestions", []))
        }
    
    def _calculate_next_scan_time(self) -> Optional[str]:
        """Calculate when the next automatic scan will run."""
        if not self.auto_scan_enabled or not self.last_scan_time:
            return None
        
        next_scan = self.last_scan_time + timedelta(hours=self.scan_interval_hours)
        return next_scan.isoformat()
    
    def _calculate_simple_diff(self, original: str, new: str) -> Dict[str, Any]:
        """Calculate a simple diff between two strings."""
        original_lines = original.split('\n')
        new_lines = new.split('\n')
        
        return {
            "lines_added": len(new_lines) - len(original_lines),
            "lines_changed": sum(1 for i, line in enumerate(original_lines) 
                               if i < len(new_lines) and line != new_lines[i])
        }
    
    def _load_upgrade_config(self) -> Dict[str, Any]:
        """Load upgrade manager configuration."""
        config_file = self.project_root / "config" / "upgrade_config.json"
        
        default_config = {
            "auto_scan_enabled": True,
            "scan_interval_hours": 168,  # Weekly
            "max_concurrent_upgrades": 1,
            "allow_high_risk_auto": False,
            "min_auto_priority": 3
        }
        
        if config_file.exists():
            try:
                with open(config_file, 'r') as f:
                    user_config = json.load(f)
                default_config.update(user_config)
            except Exception as e:
                logging.warning(f"[{self.name}] Could not load config: {e}")
        
        return default_config