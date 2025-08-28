#!/usr/bin/env python3
"""
Structured logging system for MeRNSTA enterprise deployment.
Uses structlog for JSON-formatted logs with correlation IDs and context.
"""

import structlog
import logging
import sys
import os
from typing import Any, Dict, Optional
from datetime import datetime
from pathlib import Path
from config.environment import get_settings

# Import settings
settings = get_settings()

# Create specialized loggers for audit events including causal links
audit_logger = structlog.get_logger("audit")


def setup_structured_logging():
    """Configure structured logging with JSON output and correlation IDs."""
    
    # Create logs directory if it doesn't exist
    Path("logs").mkdir(exist_ok=True)
    
    # Configure structlog
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer()
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    
    # Configure standard library logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, settings.log_level.upper())
    )
    
    # Add file handlers
    file_handler = logging.FileHandler(settings.log_file)
    file_handler.setFormatter(logging.Formatter("%(message)s"))
    
    background_file_handler = logging.FileHandler(settings.background_log_file)
    background_file_handler.setFormatter(logging.Formatter("%(message)s"))
    
    # Get root logger and add handlers
    root_logger = logging.getLogger()
    root_logger.addHandler(file_handler)
    
    # Create background logger
    background_logger = logging.getLogger('background')
    background_logger.addHandler(background_file_handler)
    background_logger.propagate = False


def get_logger(name: str = None) -> structlog.BoundLogger:
    """Get a structured logger instance."""
    return structlog.get_logger(name)


class CorrelationContext:
    """Context manager for correlation IDs in logs."""
    
    def __init__(self, correlation_id: str = None):
        self.correlation_id = correlation_id or f"req_{int(datetime.now().timestamp() * 1000)}"
    
    def __enter__(self):
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(correlation_id=self.correlation_id)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        structlog.contextvars.clear_contextvars()


def log_memory_operation(operation: str, **kwargs):
    """Log memory operations with structured data."""
    logger = get_logger("memory")
    logger.info(
        "memory_operation",
        operation=operation,
        timestamp=datetime.now().isoformat(),
        **kwargs
    )


def log_api_request(method: str, path: str, status_code: int, duration: float, **kwargs):
    """Log API requests with structured data."""
    logger = get_logger("api")
    logger.info(
        "api_request",
        method=method,
        path=path,
        status_code=status_code,
        duration_ms=duration * 1000,
        timestamp=datetime.now().isoformat(),
        **kwargs
    )


def log_background_task(task_name: str, status: str, duration: float = None, **kwargs):
    """Log background task execution."""
    logger = get_logger("background")
    log_data = {
        "task_name": task_name,
        "status": status,
        "timestamp": datetime.now().isoformat(),
        **kwargs
    }
    
    if duration is not None:
        log_data["duration_ms"] = duration * 1000
    
    logger.info("background_task", **log_data)


def log_error(error: Exception, context: Dict[str, Any] = None):
    """Log errors with structured context."""
    logger = get_logger("error")
    logger.error(
        "error_occurred",
        error_type=type(error).__name__,
        error_message=str(error),
        timestamp=datetime.now().isoformat(),
        context=context or {}
    )


def log_performance_metric(metric_name: str, value: float, unit: str = None, **kwargs):
    """Log performance metrics."""
    logger = get_logger("performance")
    log_data = {
        "metric_name": metric_name,
        "value": value,
        "timestamp": datetime.now().isoformat(),
        **kwargs
    }
    
    if unit:
        log_data["unit"] = unit
    
    logger.info("performance_metric", **log_data)


def log_security_event(event_type: str, severity: str, **kwargs):
    """Log security events."""
    logger = get_logger("security")
    logger.warning(
        "security_event",
        event_type=event_type,
        severity=severity,
        timestamp=datetime.now().isoformat(),
        **kwargs
    )


def log_config_change(config_key: str, old_value: Any, new_value: Any):
    """Log configuration changes."""
    logger = get_logger("config")
    logger.info(
        "config_changed",
        config_key=config_key,
        old_value=str(old_value),
        new_value=str(new_value),
        timestamp=datetime.now().isoformat()
    )


# Initialize logging on import
setup_structured_logging()

# Export commonly used loggers
memory_logger = get_logger("memory")
api_logger = get_logger("api")
background_logger = get_logger("background")
error_logger = get_logger("error")


def log_causal_link_created(fact_id: str, cause: str, strength: float, 
                           user_id: str = None, session_id: str = None):
    """
    Audit log for causal link creation events.
    
    Logs structured data about causal link creation for compliance and debugging.
    """
    audit_logger.info(
        "causal_link_created",
        fact_id=fact_id,
        cause=cause,
        strength=round(strength, 3),
        user_id=user_id,
        session_id=session_id,
        timestamp=datetime.now().isoformat()
    )


def log_causal_analysis_performed(fact_count: int, links_created: int, 
                                analysis_duration: float, user_id: str = None):
    """
    Audit log for causal analysis sessions.
    
    Records aggregate statistics about causal analysis performance.
    """
    audit_logger.info(
        "causal_analysis_performed",
        fact_count=fact_count,
        links_created=links_created,
        analysis_duration_seconds=round(analysis_duration, 3),
        user_id=user_id,
        timestamp=datetime.now().isoformat()
    )


def log_causal_failure(reason: str, details: str = None, 
                      user_id: str = None, session_id: str = None):
    """
    Audit log for causal analysis failures.
    
    Records failures in causal analysis for monitoring and debugging.
    """
    audit_logger.warning(
        "causal_analysis_failure",
        reason=reason,
        details=details,
        user_id=user_id,
        session_id=session_id,
        timestamp=datetime.now().isoformat()
    )
performance_logger = get_logger("performance")
security_logger = get_logger("security")
config_logger = get_logger("config") 