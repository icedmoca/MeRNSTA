#!/usr/bin/env python3
"""
Mesh Synchronization Engine for MeRNSTA Phase 32

Implements distributed memory synchronization and task sharing across mesh nodes.
Handles conflict resolution, data integrity, and efficient synchronization protocols.

Key Features:
- Multi-backend sync (SQLite, Redis, gRPC)
- Conflict resolution strategies
- Memory compression and optimization
- Task distribution with context preservation
- Fault tolerance and recovery
"""

import asyncio
import json
import logging
import time
import zlib
import hashlib
import threading
from typing import Dict, List, Any, Optional, Set, Tuple, Callable, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from collections import defaultdict, deque
from enum import Enum
import sqlite3

from config.settings import get_config
from .mesh_manager import MemorySync, SharedTask, MeshNode, NodeType, TaskPriority


class SyncStrategy(Enum):
    """Synchronization strategies for different data types."""
    EVENTUAL_CONSISTENCY = "eventual_consistency"
    STRONG_CONSISTENCY = "strong_consistency"
    MASTER_SLAVE = "master_slave"
    PEER_TO_PEER = "peer_to_peer"


class ConflictResolution(Enum):
    """Conflict resolution strategies."""
    LATEST_WINS = "latest_wins"
    MERGE_COMPATIBLE = "merge_compatible"
    MANUAL_RESOLUTION = "manual_resolution"
    SOURCE_PRIORITY = "source_priority"
    VERSION_VECTOR = "version_vector"


@dataclass
class SyncOperation:
    """Represents a synchronization operation."""
    operation_id: str
    operation_type: str  # 'push', 'pull', 'merge', 'delete'
    source_node: str
    target_nodes: List[str]
    data_type: str
    data_payload: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)
    priority: int = 5
    retry_count: int = 0
    max_retries: int = 3
    status: str = "pending"  # pending, in_progress, completed, failed
    conflict_detected: bool = False
    resolution_strategy: ConflictResolution = ConflictResolution.LATEST_WINS


@dataclass
class TaskDistribution:
    """Task distribution metadata."""
    task_id: str
    distribution_algorithm: str
    target_node_preferences: List[str]
    load_balancing_score: float
    affinity_score: float
    distribution_timestamp: datetime = field(default_factory=datetime.now)
    estimated_completion_time: Optional[float] = None
    resource_requirements: Dict[str, Any] = field(default_factory=dict)


class MeshSyncEngine:
    """
    Core synchronization engine for distributed mesh operations.
    
    Handles memory synchronization, task distribution, and conflict resolution
    across multiple mesh nodes with different communication backends.
    """
    
    def __init__(self, mesh_manager=None):
        self.mesh_manager = mesh_manager
        
        # Load configuration
        self.config = get_config()
        self.mesh_config = self.config.get('agent_mesh', {})
        self.sync_config = self.mesh_config.get('memory_sync', {})
        self.task_config = self.mesh_config.get('task_distribution', {})
        
        # Synchronization settings
        self.sync_enabled = self.sync_config.get('enabled', True)
        self.sync_priority_threshold = self.sync_config.get('sync_priority_threshold', 5)
        self.conflict_resolution_strategy = ConflictResolution(
            self.sync_config.get('conflict_resolution_strategy', 'latest_wins')
        )
        self.max_sync_batch_size = self.sync_config.get('max_sync_batch_size', 50)
        self.compression_enabled = self.sync_config.get('compression_enabled', True)
        
        # Task distribution settings
        self.load_balancing_algorithm = self.task_config.get('load_balancing_algorithm', 'least_loaded')
        self.retry_failed_tasks = self.task_config.get('retry_failed_tasks', True)
        self.max_task_retries = self.task_config.get('max_task_retries', 3)
        self.task_affinity_enabled = self.task_config.get('task_affinity_enabled', True)
        
        # Internal state
        self.sync_operations: Dict[str, SyncOperation] = {}
        self.task_distributions: Dict[str, TaskDistribution] = {}
        self.conflict_queue: deque = deque(maxlen=100)
        self.sync_metrics: Dict[str, Any] = defaultdict(int)
        
        # Node affinity tracking
        self.node_affinities: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self.task_success_rates: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        
        # Background processing
        self._running = False
        self._sync_thread = None
        
        logging.info(f"[MeshSyncEngine] Initialized with strategy: {self.conflict_resolution_strategy.value}")
    
    def start_sync_engine(self):
        """Start the background synchronization engine."""
        if self._running:
            return
        
        self._running = True
        self._sync_thread = threading.Thread(target=self._sync_engine_loop, daemon=True)
        self._sync_thread.start()
        
        logging.info("[MeshSyncEngine] Background sync engine started")
    
    def stop_sync_engine(self):
        """Stop the background synchronization engine."""
        self._running = False
        if self._sync_thread and self._sync_thread.is_alive():
            self._sync_thread.join(timeout=5)
        
        logging.info("[MeshSyncEngine] Background sync engine stopped")
    
    def _sync_engine_loop(self):
        """Main synchronization engine loop."""
        while self._running:
            try:
                # Process pending sync operations
                self._process_sync_operations()
                
                # Process task distributions
                self._process_task_distributions()
                
                # Handle conflict resolution
                self._process_conflict_queue()
                
                # Update metrics and cleanup
                self._update_sync_metrics()
                self._cleanup_completed_operations()
                
                time.sleep(1)  # Process every second
                
            except Exception as e:
                logging.error(f"[MeshSyncEngine] Error in sync loop: {e}")
                time.sleep(5)  # Wait longer on error
    
    def sync_memory_item(self, memory_sync: MemorySync, target_nodes: Optional[List[str]] = None) -> str:
        """
        Synchronize a memory item across specified nodes.
        
        Args:
            memory_sync: Memory synchronization item
            target_nodes: Specific nodes to sync to (None for all nodes)
            
        Returns:
            Operation ID for tracking
        """
        if not self.sync_enabled:
            raise RuntimeError("Memory synchronization is disabled")
        
        operation_id = f"sync_{int(time.time())}_{hash(memory_sync.sync_id) % 10000}"
        
        # Determine target nodes
        if target_nodes is None:
            target_nodes = self._get_all_active_nodes()
        
        # Create sync operation
        sync_op = SyncOperation(
            operation_id=operation_id,
            operation_type='push',
            source_node=memory_sync.source_node,
            target_nodes=target_nodes,
            data_type=memory_sync.memory_type,
            data_payload=memory_sync.to_dict(),
            priority=memory_sync.sync_priority,
            resolution_strategy=self.conflict_resolution_strategy
        )
        
        self.sync_operations[operation_id] = sync_op
        
        # Process immediately if high priority
        if memory_sync.sync_priority >= 8:
            self._execute_sync_operation(sync_op)
        
        logging.info(f"[MeshSyncEngine] Queued memory sync {operation_id} for {len(target_nodes)} nodes")
        return operation_id
    
    def distribute_task(self, shared_task: SharedTask) -> str:
        """
        Distribute a task to the most appropriate node.
        
        Args:
            shared_task: Task to distribute
            
        Returns:
            Distribution ID for tracking
        """
        distribution_id = f"dist_{int(time.time())}_{hash(shared_task.task_id) % 10000}"
        
        # Find optimal node for task
        target_node = self._find_optimal_node_for_task(shared_task)
        
        if not target_node:
            raise RuntimeError("No suitable node found for task distribution")
        
        # Calculate distribution metadata
        distribution = TaskDistribution(
            task_id=shared_task.task_id,
            distribution_algorithm=self.load_balancing_algorithm,
            target_node_preferences=[target_node],
            load_balancing_score=self._calculate_load_balance_score(target_node),
            affinity_score=self._calculate_affinity_score(target_node, shared_task)
        )
        
        self.task_distributions[distribution_id] = distribution
        
        # Assign task to target node
        shared_task.assigned_to = target_node
        shared_task.assigned_at = datetime.now()
        
        # Send task to target node
        self._send_task_to_node(shared_task, target_node)
        
        logging.info(f"[MeshSyncEngine] Distributed task {shared_task.task_id} to node {target_node}")
        return distribution_id
    
    def handle_conflict(self, conflict_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle synchronization conflicts based on resolution strategy.
        
        Args:
            conflict_data: Conflict information and competing versions
            
        Returns:
            Resolution result
        """
        strategy = conflict_data.get('resolution_strategy', self.conflict_resolution_strategy)
        
        if strategy == ConflictResolution.LATEST_WINS:
            return self._resolve_latest_wins(conflict_data)
        elif strategy == ConflictResolution.MERGE_COMPATIBLE:
            return self._resolve_merge_compatible(conflict_data)
        elif strategy == ConflictResolution.SOURCE_PRIORITY:
            return self._resolve_source_priority(conflict_data)
        elif strategy == ConflictResolution.VERSION_VECTOR:
            return self._resolve_version_vector(conflict_data)
        else:
            # Manual resolution - queue for human intervention
            self.conflict_queue.append(conflict_data)
            return {"status": "queued_for_manual_resolution", "queue_position": len(self.conflict_queue)}
    
    def _get_all_active_nodes(self) -> List[str]:
        """Get list of all active nodes in the mesh."""
        if not self.mesh_manager:
            return []
        
        active_nodes = []
        
        # Add local node if active
        if self.mesh_manager.local_node and self.mesh_manager.local_node.status.value == 'active':
            active_nodes.append(self.mesh_manager.local_node.node_id)
        
        # Add known active nodes
        for node in self.mesh_manager.known_nodes.values():
            if node.status.value == 'active':
                active_nodes.append(node.node_id)
        
        return active_nodes
    
    def _find_optimal_node_for_task(self, task: SharedTask) -> Optional[str]:
        """Find the most suitable node for a given task."""
        if not self.mesh_manager:
            return None
        
        candidate_nodes = []
        
        # Get all available nodes
        all_nodes = []
        if self.mesh_manager.local_node:
            all_nodes.append(self.mesh_manager.local_node)
        all_nodes.extend(self.mesh_manager.known_nodes.values())
        
        # Filter by capabilities and node type preferences
        for node in all_nodes:
            if node.status.value not in ['active', 'busy']:
                continue
            
            # Check capability requirements
            if task.required_capabilities:
                if not all(cap in node.capabilities for cap in task.required_capabilities):
                    continue
            
            # Check node type preferences
            if task.preferred_node_types:
                if node.node_type not in task.preferred_node_types:
                    continue
            
            candidate_nodes.append(node)
        
        if not candidate_nodes:
            return None
        
        # Apply load balancing algorithm
        if self.load_balancing_algorithm == 'least_loaded':
            return min(candidate_nodes, key=lambda n: n.current_load).node_id
        elif self.load_balancing_algorithm == 'round_robin':
            # Simple round-robin based on task count
            return min(candidate_nodes, key=lambda n: n.total_tasks_completed).node_id
        elif self.load_balancing_algorithm == 'capability_based':
            # Score based on capability match
            best_node = max(candidate_nodes, 
                          key=lambda n: len(set(n.capabilities) & set(task.required_capabilities)))
            return best_node.node_id
        elif self.load_balancing_algorithm == 'random':
            import random
            return random.choice(candidate_nodes).node_id
        else:
            # Default to least loaded
            return min(candidate_nodes, key=lambda n: n.current_load).node_id
    
    def _calculate_load_balance_score(self, node_id: str) -> float:
        """Calculate load balancing score for a node."""
        if not self.mesh_manager:
            return 0.0
        
        node = None
        if self.mesh_manager.local_node and self.mesh_manager.local_node.node_id == node_id:
            node = self.mesh_manager.local_node
        else:
            node = self.mesh_manager.known_nodes.get(node_id)
        
        if not node:
            return 0.0
        
        # Score based on inverse of current load
        return 1.0 - node.current_load
    
    def _calculate_affinity_score(self, node_id: str, task: SharedTask) -> float:
        """Calculate task affinity score for a node."""
        if not self.task_affinity_enabled:
            return 0.5  # Neutral score
        
        # Get historical success rate for this task type on this node
        task_type = task.task_type
        success_rate = self.task_success_rates[node_id].get(task_type, 0.5)
        
        # Get general affinity score
        affinity = self.node_affinities[node_id].get(task_type, 0.5)
        
        # Combine scores
        return (success_rate * 0.7) + (affinity * 0.3)
    
    def _send_task_to_node(self, task: SharedTask, target_node: str):
        """Send task to target node for execution."""
        try:
            # Use SQLite backend for local task queue
            if self.mesh_manager and self.mesh_manager._sqlite_backend:
                cursor = self.mesh_manager._sqlite_backend.cursor()
                cursor.execute(
                    '''INSERT OR REPLACE INTO shared_tasks 
                       (task_id, task_data, status, priority) VALUES (?, ?, ?, ?)''',
                    (task.task_id, json.dumps(task.to_dict()), 'assigned', task.priority.value)
                )
                self.mesh_manager._sqlite_backend.commit()
            
            # Use Redis backend for distributed task queue
            if self.mesh_manager and self.mesh_manager._redis_backend:
                self.mesh_manager._redis_backend.hset(
                    f'node_tasks:{target_node}',
                    task.task_id,
                    json.dumps(task.to_dict())
                )
            
            logging.info(f"[MeshSyncEngine] Sent task {task.task_id} to node {target_node}")
            
        except Exception as e:
            logging.error(f"[MeshSyncEngine] Failed to send task to node {target_node}: {e}")
    
    def _execute_sync_operation(self, sync_op: SyncOperation):
        """Execute a synchronization operation."""
        try:
            sync_op.status = "in_progress"
            
            # Compress data if enabled
            payload = sync_op.data_payload
            if self.compression_enabled and len(json.dumps(payload)) > 1024:
                compressed_data = zlib.compress(json.dumps(payload).encode())
                payload = {
                    'compressed': True,
                    'data': compressed_data.hex(),
                    'original_size': len(json.dumps(sync_op.data_payload))
                }
            
            # Send to target nodes
            success_count = 0
            for target_node in sync_op.target_nodes:
                if self._send_sync_to_node(payload, target_node, sync_op):
                    success_count += 1
            
            # Update operation status
            if success_count == len(sync_op.target_nodes):
                sync_op.status = "completed"
            elif success_count > 0:
                sync_op.status = "partial_success"
            else:
                sync_op.status = "failed"
            
            # Update metrics
            self.sync_metrics['operations_completed'] += 1
            self.sync_metrics['nodes_synced'] += success_count
            
            logging.info(f"[MeshSyncEngine] Sync operation {sync_op.operation_id}: {sync_op.status}")
            
        except Exception as e:
            sync_op.status = "failed"
            sync_op.retry_count += 1
            logging.error(f"[MeshSyncEngine] Sync operation {sync_op.operation_id} failed: {e}")
    
    def _send_sync_to_node(self, payload: Dict[str, Any], target_node: str, sync_op: SyncOperation) -> bool:
        """Send synchronization data to a specific node."""
        try:
            # Use SQLite backend for local sync
            if self.mesh_manager and self.mesh_manager._sqlite_backend:
                cursor = self.mesh_manager._sqlite_backend.cursor()
                cursor.execute(
                    '''INSERT OR REPLACE INTO memory_sync 
                       (sync_id, memory_type, sync_data, source_node, version, checksum) 
                       VALUES (?, ?, ?, ?, ?, ?)''',
                    (
                        sync_op.operation_id,
                        sync_op.data_type,
                        json.dumps(payload),
                        sync_op.source_node,
                        1,
                        hashlib.sha256(json.dumps(payload).encode()).hexdigest()[:16]
                    )
                )
                self.mesh_manager._sqlite_backend.commit()
            
            # Use Redis backend for distributed sync
            if self.mesh_manager and self.mesh_manager._redis_backend:
                self.mesh_manager._redis_backend.hset(
                    f'memory_sync:{target_node}',
                    sync_op.operation_id,
                    json.dumps(payload)
                )
            
            return True
            
        except Exception as e:
            logging.error(f"[MeshSyncEngine] Failed to send sync to node {target_node}: {e}")
            return False
    
    def _resolve_latest_wins(self, conflict_data: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve conflict using latest timestamp wins strategy."""
        versions = conflict_data.get('versions', [])
        if not versions:
            return {"status": "no_versions_to_resolve"}
        
        # Find version with latest timestamp
        latest_version = max(versions, key=lambda v: v.get('timestamp', ''))
        
        return {
            "status": "resolved_latest_wins",
            "winning_version": latest_version,
            "resolution_timestamp": datetime.now().isoformat()
        }
    
    def _resolve_merge_compatible(self, conflict_data: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve conflict by merging compatible changes."""
        versions = conflict_data.get('versions', [])
        if len(versions) < 2:
            return {"status": "insufficient_versions_for_merge"}
        
        # Simple merge strategy - combine non-conflicting fields
        merged_data = {}
        
        for version in versions:
            data = version.get('data', {})
            for key, value in data.items():
                if key not in merged_data:
                    merged_data[key] = value
                elif merged_data[key] != value:
                    # Conflict detected - use latest
                    if version.get('timestamp', '') > merged_data.get('_last_timestamp', ''):
                        merged_data[key] = value
                        merged_data['_last_timestamp'] = version.get('timestamp', '')
        
        return {
            "status": "resolved_merge_compatible",
            "merged_data": merged_data,
            "resolution_timestamp": datetime.now().isoformat()
        }
    
    def _resolve_source_priority(self, conflict_data: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve conflict based on source node priority."""
        # Simple implementation - could be enhanced with node priority configuration
        versions = conflict_data.get('versions', [])
        if not versions:
            return {"status": "no_versions_to_resolve"}
        
        # For now, prefer coordination hub nodes, then general nodes
        priority_order = ['coordination_hub', 'planning_only', 'general', 'language_expert', 'sensor_connected']
        
        best_version = None
        best_priority = len(priority_order)
        
        for version in versions:
            source_type = version.get('source_node_type', 'general')
            try:
                priority = priority_order.index(source_type)
                if priority < best_priority:
                    best_priority = priority
                    best_version = version
            except ValueError:
                continue
        
        return {
            "status": "resolved_source_priority",
            "winning_version": best_version or versions[0],
            "resolution_timestamp": datetime.now().isoformat()
        }
    
    def _resolve_version_vector(self, conflict_data: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve conflict using version vectors (simplified implementation)."""
        # This would require a more sophisticated version vector implementation
        # For now, fall back to latest wins
        return self._resolve_latest_wins(conflict_data)
    
    def _process_sync_operations(self):
        """Process pending synchronization operations."""
        pending_ops = [op for op in self.sync_operations.values() if op.status == "pending"]
        
        # Sort by priority
        pending_ops.sort(key=lambda op: op.priority, reverse=True)
        
        # Process up to max batch size
        for op in pending_ops[:self.max_sync_batch_size]:
            self._execute_sync_operation(op)
    
    def _process_task_distributions(self):
        """Process pending task distributions."""
        # This would handle ongoing task distribution monitoring
        pass
    
    def _process_conflict_queue(self):
        """Process conflicts in the conflict queue."""
        while self.conflict_queue:
            conflict = self.conflict_queue.popleft()
            try:
                resolution = self.handle_conflict(conflict)
                logging.info(f"[MeshSyncEngine] Resolved conflict: {resolution['status']}")
            except Exception as e:
                logging.error(f"[MeshSyncEngine] Failed to resolve conflict: {e}")
    
    def _update_sync_metrics(self):
        """Update synchronization metrics."""
        self.sync_metrics['total_operations'] = len(self.sync_operations)
        self.sync_metrics['pending_operations'] = len([op for op in self.sync_operations.values() if op.status == "pending"])
        self.sync_metrics['active_distributions'] = len(self.task_distributions)
        self.sync_metrics['conflicts_queued'] = len(self.conflict_queue)
    
    def _cleanup_completed_operations(self):
        """Clean up completed synchronization operations."""
        # Remove operations completed more than 1 hour ago
        cutoff_time = datetime.now() - timedelta(hours=1)
        
        completed_ops = [
            op_id for op_id, op in self.sync_operations.items()
            if op.status in ["completed", "failed"] and op.timestamp < cutoff_time
        ]
        
        for op_id in completed_ops:
            del self.sync_operations[op_id]
        
        if completed_ops:
            logging.debug(f"[MeshSyncEngine] Cleaned up {len(completed_ops)} completed operations")
    
    def get_sync_status(self) -> Dict[str, Any]:
        """Get comprehensive synchronization status."""
        return {
            "sync_enabled": self.sync_enabled,
            "conflict_resolution_strategy": self.conflict_resolution_strategy.value,
            "load_balancing_algorithm": self.load_balancing_algorithm,
            "metrics": dict(self.sync_metrics),
            "active_operations": len([op for op in self.sync_operations.values() if op.status == "in_progress"]),
            "pending_operations": len([op for op in self.sync_operations.values() if op.status == "pending"]),
            "failed_operations": len([op for op in self.sync_operations.values() if op.status == "failed"]),
            "conflicts_queued": len(self.conflict_queue),
            "task_distributions_active": len(self.task_distributions),
            "compression_enabled": self.compression_enabled,
            "task_affinity_enabled": self.task_affinity_enabled
        }