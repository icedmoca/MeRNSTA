#!/usr/bin/env python3
"""
Celery task queue for MeRNSTA enterprise deployment.
Replaces threading-based background tasks with robust queue management.
"""

from celery import Celery
from celery.schedules import crontab
from celery.utils.log import get_task_logger
import time
import traceback
from typing import Dict, Any, List, Optional
from config.environment import get_settings
from monitoring.logger import get_logger, log_background_task
from monitoring.metrics import track_background_task

# Import settings
settings = get_settings()

# Configure Celery
app = Celery(
    'mernsta_tasks',
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend
)

# Alias for compatibility
celery_app = app

# Auto-discover tasks in this module
app.autodiscover_tasks(['tasks'])

# Celery configuration
app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    task_track_started=True,
    task_time_limit=30 * 60,  # 30 minutes
    task_soft_time_limit=25 * 60,  # 25 minutes
    worker_prefetch_multiplier=1,
    worker_max_tasks_per_child=1000,
    worker_max_memory_per_child=200000,  # 200MB
    result_expires=3600,  # 1 hour
    beat_schedule={
        'auto-reconciliation': {
            'task': 'tasks.task_queue.auto_reconciliation_task',
            'schedule': crontab(minute=f'*/{settings.reconciliation_interval // 60}'),
        },
        'memory-compression': {
            'task': 'tasks.task_queue.memory_compression_task',
            'schedule': crontab(minute=0, hour=f'*/{settings.compression_interval // 3600}'),
        },
        'memory-health-check': {
            'task': 'tasks.task_queue.memory_health_check_task',
            'schedule': crontab(minute='*/5'),  # Every 5 minutes
        },
        'system-cleanup': {
            'task': 'tasks.task_queue.system_cleanup_task',
            'schedule': crontab(hour=2, minute=0),  # Daily at 2 AM
        },
        'meta-self-health-check': {
            'task': 'tasks.task_queue.meta_self_health_check_task',
            'schedule': crontab(minute='*/30'),  # Every 30 minutes
        },
        'meta-self-deep-analysis': {
            'task': 'tasks.task_queue.meta_self_deep_analysis_task',
            'schedule': crontab(minute=0, hour='*/6'),  # Every 6 hours
        },
    }
)

logger = get_logger("celery")


@track_background_task("auto_reconciliation")
@app.task(bind=True, max_retries=3, default_retry_delay=60)
def auto_reconciliation_task(self):
    """Auto-reconciliation background task with retry logic."""
    start_time = time.time()
    
    try:
        from storage.auto_reconciliation import AutoReconciliationEngine
        from storage.memory_log import MemoryLog
        
        memory_log = MemoryLog()
        reconciliation_engine = AutoReconciliationEngine(memory_log)
        
        # Perform reconciliation
        contradictions_found = reconciliation_engine._check_and_resolve_contradictions()
        
        duration = time.time() - start_time
        log_background_task(
            "auto_reconciliation",
            "success",
            duration,
            contradictions_found=contradictions_found
        )
        
        return {
            "status": "success",
            "contradictions_found": contradictions_found,
            "duration": duration
        }
        
    except Exception as exc:
        duration = time.time() - start_time
        log_background_task(
            "auto_reconciliation",
            "error",
            duration,
            error=str(exc)
        )
        
        # Retry logic
        if self.request.retries < self.max_retries:
            logger.warning(
                f"Auto-reconciliation task failed, retrying {self.request.retries + 1}/{self.max_retries}",
                error=str(exc)
            )
            raise self.retry(exc=exc, countdown=60 * (2 ** self.request.retries))
        else:
            logger.error(
                "Auto-reconciliation task failed permanently",
                error=str(exc),
                traceback=traceback.format_exc()
            )
            raise


@track_background_task("memory_compression")
@app.task(bind=True, max_retries=3, default_retry_delay=120)
def memory_compression_task(self):
    """Memory compression background task with retry logic."""
    start_time = time.time()
    
    try:
        from storage.memory_compression import MemoryCompressionEngine
        from storage.memory_log import MemoryLog
        
        memory_log = MemoryLog()
        compression_engine = MemoryCompressionEngine(memory_log)
        
        # Perform compression
        clusters_compressed = compression_engine._check_and_compress_memory()
        
        duration = time.time() - start_time
        log_background_task(
            "memory_compression",
            "success",
            duration,
            clusters_compressed=clusters_compressed
        )
        
        return {
            "status": "success",
            "clusters_compressed": clusters_compressed,
            "duration": duration
        }
        
    except Exception as exc:
        duration = time.time() - start_time
        log_background_task(
            "memory_compression",
            "error",
            duration,
            error=str(exc)
        )
        
        # Retry logic
        if self.request.retries < self.max_retries:
            logger.warning(
                f"Memory compression task failed, retrying {self.request.retries + 1}/{self.max_retries}",
                error=str(exc)
            )
            raise self.retry(exc=exc, countdown=120 * (2 ** self.request.retries))
        else:
            logger.error(
                "Memory compression task failed permanently",
                error=str(exc),
                traceback=traceback.format_exc()
            )
            raise


@track_background_task("memory_health_check")
@app.task(bind=True, max_retries=2, default_retry_delay=30)
def memory_health_check_task(self):
    """Memory health check background task."""
    start_time = time.time()
    
    try:
        from storage.memory_log import MemoryLog
        from monitoring.metrics import update_memory_metrics
        
        memory_log = MemoryLog()
        stats = memory_log.get_memory_stats()
        
        # Update metrics
        update_memory_metrics(stats)
        
        duration = time.time() - start_time
        log_background_task(
            "memory_health_check",
            "success",
            duration,
            total_facts=stats.get('total_facts', 0)
        )
        
        return {
            "status": "success",
            "total_facts": stats.get('total_facts', 0),
            "duration": duration
        }
        
    except Exception as exc:
        duration = time.time() - start_time
        log_background_task(
            "memory_health_check",
            "error",
            duration,
            error=str(exc)
        )
        
        if self.request.retries < self.max_retries:
            raise self.retry(exc=exc, countdown=30 * (2 ** self.request.retries))
        else:
            logger.error(
                "Memory health check task failed permanently",
                error=str(exc)
            )
            raise


@track_background_task("system_cleanup")
@app.task(bind=True, max_retries=1, default_retry_delay=60)
def system_cleanup_task(self):
    """System cleanup background task."""
    start_time = time.time()
    
    try:
        from storage.memory_log import MemoryLog
        
        memory_log = MemoryLog()
        
        # Perform cleanup operations
        cleanup_results = {
            "pruned_facts": 0,
            "cleaned_contradictions": 0,
            "optimized_clusters": 0
        }
        
        # Prune old facts
        if settings.enable_compression:
            cleanup_results["pruned_facts"] = memory_log.prune_memory(
                confidence_threshold=0.3,
                age_threshold_days=30
            )
        
        # Clean resolved contradictions
        contradictions = memory_log.get_contradictions(resolved=True)
        for contradiction in contradictions[:100]:  # Limit to prevent long-running task
            memory_log.resolve_contradiction(contradiction['id'], "Auto-cleanup")
            cleanup_results["cleaned_contradictions"] += 1
        
        duration = time.time() - start_time
        log_background_task(
            "system_cleanup",
            "success",
            duration,
            **cleanup_results
        )
        
        return {
            "status": "success",
            "cleanup_results": cleanup_results,
            "duration": duration
        }
        
    except Exception as exc:
        duration = time.time() - start_time
        log_background_task(
            "system_cleanup",
            "error",
            duration,
            error=str(exc)
        )
        
        if self.request.retries < self.max_retries:
            raise self.retry(exc=exc, countdown=60)
        else:
            logger.error(
                "System cleanup task failed permanently",
                error=str(exc)
            )
            raise


@track_background_task("manual_reconciliation")
@app.task(bind=True, max_retries=2, default_retry_delay=30)
def manual_reconciliation_task(self, subject: str = None):
    """Manual reconciliation task triggered by user."""
    start_time = time.time()
    
    try:
        from storage.auto_reconciliation import AutoReconciliationEngine
        from storage.memory_log import MemoryLog
        
        memory_log = MemoryLog()
        reconciliation_engine = AutoReconciliationEngine(memory_log)
        
        if subject:
            # Reconcile specific subject
            contradictions_found = reconciliation_engine.reconcile_subject(subject)
        else:
            # Full reconciliation
            contradictions_found = reconciliation_engine._check_and_resolve_contradictions()
        
        duration = time.time() - start_time
        log_background_task(
            "manual_reconciliation",
            "success",
            duration,
            subject=subject,
            contradictions_found=contradictions_found
        )
        
        return {
            "status": "success",
            "subject": subject,
            "contradictions_found": contradictions_found,
            "duration": duration
        }
        
    except Exception as exc:
        duration = time.time() - start_time
        log_background_task(
            "manual_reconciliation",
            "error",
            duration,
            subject=subject,
            error=str(exc)
        )
        
        if self.request.retries < self.max_retries:
            raise self.retry(exc=exc, countdown=30 * (2 ** self.request.retries))
        else:
            logger.error(
                "Manual reconciliation task failed permanently",
                error=str(exc),
                subject=subject
            )
            raise


@track_background_task("manual_compression")
@app.task(bind=True, max_retries=2, default_retry_delay=60)
def manual_compression_task(self, subject: str = None):
    """Manual compression task triggered by user."""
    start_time = time.time()
    
    try:
        from storage.memory_compression import MemoryCompressionEngine
        from storage.memory_log import MemoryLog
        
        memory_log = MemoryLog()
        compression_engine = MemoryCompressionEngine(memory_log)
        
        if subject:
            # Compress specific subject
            result = compression_engine.compress_cluster(subject)
            clusters_compressed = 1 if result else 0
        else:
            # Full compression
            clusters_compressed = compression_engine._check_and_compress_memory()
        
        duration = time.time() - start_time
        log_background_task(
            "manual_compression",
            "success",
            duration,
            subject=subject,
            clusters_compressed=clusters_compressed
        )
        
        return {
            "status": "success",
            "subject": subject,
            "clusters_compressed": clusters_compressed,
            "duration": duration
        }
        
    except Exception as exc:
        duration = time.time() - start_time
        log_background_task(
            "manual_compression",
            "error",
            duration,
            subject=subject,
            error=str(exc)
        )
        
        if self.request.retries < self.max_retries:
            raise self.retry(exc=exc, countdown=60 * (2 ** self.request.retries))
        else:
            logger.error(
                "Manual compression task failed permanently",
                error=str(exc),
                subject=subject
            )
            raise


@track_background_task("meta_self_health_check")
@app.task(bind=True, max_retries=2, default_retry_delay=30)
def meta_self_health_check_task(self):
    """MetaSelfAgent health check background task."""
    start_time = time.time()
    
    try:
        from agents.meta_self_agent import get_meta_self_agent
        
        # Get MetaSelfAgent instance
        meta_agent = get_meta_self_agent()
        
        # Check if health check is needed
        if not meta_agent.should_run_health_check():
            return {
                "status": "skipped",
                "reason": "Health check not due yet",
                "duration": time.time() - start_time
            }
        
        # Perform health check
        health_metrics = meta_agent.perform_health_check()
        
        duration = time.time() - start_time
        log_background_task(
            "meta_self_health_check",
            "success",
            duration,
            overall_health_score=health_metrics.overall_health_score,
            critical_issues=len(health_metrics.critical_issues),
            anomalies_detected=len(health_metrics.anomalies_detected)
        )
        
        return {
            "status": "success",
            "overall_health_score": health_metrics.overall_health_score,
            "critical_issues": len(health_metrics.critical_issues),
            "anomalies_detected": len(health_metrics.anomalies_detected),
            "goals_generated": len(meta_agent.generated_goals),
            "duration": duration
        }
        
    except Exception as exc:
        duration = time.time() - start_time
        log_background_task(
            "meta_self_health_check",
            "error",
            duration,
            error=str(exc)
        )
        
        if self.request.retries < self.max_retries:
            raise self.retry(exc=exc, countdown=30 * (2 ** self.request.retries))
        else:
            logger.error(
                "Meta-self health check task failed permanently",
                error=str(exc)
            )
            raise


@track_background_task("meta_self_deep_analysis")
@app.task(bind=True, max_retries=1, default_retry_delay=60)
def meta_self_deep_analysis_task(self):
    """MetaSelfAgent deep introspective analysis background task."""
    start_time = time.time()
    
    try:
        from agents.meta_self_agent import get_meta_self_agent
        
        # Get MetaSelfAgent instance
        meta_agent = get_meta_self_agent()
        
        # Check if deep analysis is needed
        if not meta_agent.should_run_deep_analysis():
            return {
                "status": "skipped",
                "reason": "Deep analysis not due yet",
                "duration": time.time() - start_time
            }
        
        # Perform deep introspective analysis
        analysis_results = meta_agent.trigger_introspective_analysis()
        
        duration = time.time() - start_time
        log_background_task(
            "meta_self_deep_analysis",
            "success",
            duration,
            health_score=analysis_results.get('health_metrics', {}).get('overall_health_score', 0),
            patterns_detected=len(analysis_results.get('pattern_detection', {})),
            recommendations=len(analysis_results.get('recommendations', []))
        )
        
        return {
            "status": "success",
            "analysis_timestamp": analysis_results.get('timestamp'),
            "health_score": analysis_results.get('health_metrics', {}).get('overall_health_score', 0),
            "patterns_detected": len(analysis_results.get('pattern_detection', {})),
            "recommendations": len(analysis_results.get('recommendations', [])),
            "insights": len(analysis_results.get('system_insights', [])),
            "duration": duration
        }
        
    except Exception as exc:
        duration = time.time() - start_time
        log_background_task(
            "meta_self_deep_analysis",
            "error",
            duration,
            error=str(exc)
        )
        
        if self.request.retries < self.max_retries:
            raise self.retry(exc=exc, countdown=60)
        else:
            logger.error(
                "Meta-self deep analysis task failed permanently",
                error=str(exc)
            )
            raise


def get_task_status(task_id: str) -> Dict[str, Any]:
    """Get the status of a specific task."""
    try:
        result = app.AsyncResult(task_id)
        return {
            "task_id": task_id,
            "status": result.status,
            "result": result.result if result.ready() else None,
            "info": result.info if hasattr(result, 'info') else None
        }
    except Exception as e:
        logger.error(f"Error getting task status for {task_id}", error=str(e))
        return {
            "task_id": task_id,
            "status": "error",
            "error": str(e)
        }


def get_active_tasks() -> List[Dict[str, Any]]:
    """Get list of active tasks."""
    try:
        inspector = app.control.inspect()
        active = inspector.active()
        reserved = inspector.reserved()
        
        tasks = []
        for worker, worker_tasks in active.items():
            for task in worker_tasks:
                tasks.append({
                    "worker": worker,
                    "task_id": task['id'],
                    "task_name": task['name'],
                    "status": "active",
                    "start_time": task.get('time_start', 0)
                })
        
        for worker, worker_tasks in reserved.items():
            for task in worker_tasks:
                tasks.append({
                    "worker": worker,
                    "task_id": task['id'],
                    "task_name": task['name'],
                    "status": "reserved",
                    "start_time": task.get('time_start', 0)
                })
        
        return tasks
    except Exception as e:
        logger.error("Error getting active tasks", error=str(e))
        return []


def revoke_task(task_id: str) -> bool:
    """Revoke a running task."""
    try:
        app.control.revoke(task_id, terminate=True)
        logger.info(f"Task {task_id} revoked")
        return True
    except Exception as e:
        logger.error(f"Error revoking task {task_id}", error=str(e))
        return False


# Task queue management functions
def start_worker():
    """Start Celery worker."""
    app.worker_main(['worker', '--loglevel=info'])


def start_beat():
    """Start Celery beat scheduler."""
    app.start(['beat', '--loglevel=info'])


def purge_queue():
    """Purge all tasks from the queue."""
    app.control.purge()
    logger.info("Task queue purged")


def get_queue_stats() -> Dict[str, Any]:
    """Get queue statistics."""
    try:
        inspector = app.control.inspect()
        stats = inspector.stats()
        active = inspector.active()
        reserved = inspector.reserved()
        
        total_active = sum(len(tasks) for tasks in active.values()) if active else 0
        total_reserved = sum(len(tasks) for tasks in reserved.values()) if reserved else 0
        
        return {
            "workers": len(stats) if stats else 0,
            "active_tasks": total_active,
            "reserved_tasks": total_reserved,
            "total_tasks": total_active + total_reserved
        }
    except Exception as e:
        logger.error("Error getting queue stats", error=str(e))
        return {
            "workers": 0,
            "active_tasks": 0,
            "reserved_tasks": 0,
            "total_tasks": 0,
            "error": str(e)
        } 