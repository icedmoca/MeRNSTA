"""
Tasks package for MeRNSTA Enterprise.
Provides Celery task queue functionality.
"""

__version__ = "1.0.0"

# Import main celery app and all tasks for proper registration
from .task_queue import (
    app as celery_app,
    auto_reconciliation_task,
    memory_compression_task,
    memory_health_check_task,
    system_cleanup_task,
    manual_reconciliation_task,
    manual_compression_task
)

__all__ = [
    "celery_app",
    "auto_reconciliation_task",
    "memory_compression_task", 
    "memory_health_check_task",
    "system_cleanup_task",
    "manual_reconciliation_task",
    "manual_compression_task"
] 