#!/usr/bin/env python3
"""
Prometheus metrics for MeRNSTA enterprise monitoring.
Provides comprehensive metrics for memory operations, API performance, and system health.
"""

from prometheus_client import Counter, Histogram, Gauge, Summary, generate_latest, CONTENT_TYPE_LATEST
import time
import psutil
from typing import Dict, Any
from config.environment import get_settings
from storage.memory_log import MemoryLog
from fastapi import APIRouter
from storage.db_utils import get_connection_pool
from monitoring.logger import memory_logger

router = APIRouter()

@router.get("/dashboard/metrics")
async def get_dashboard_metrics():
    try:
        with get_connection_pool().get_connection() as conn:
            fact_count = conn.execute("SELECT COUNT(*) FROM facts").fetchone()[0]
            contradiction_count = conn.execute("SELECT COUNT(*) FROM facts WHERE contradiction_score > 0").fetchone()[0]
            memory_usage_mb = conn.execute("SELECT SUM(length(embedding)) / 1048576 FROM facts").fetchone()[0] or 0
            metrics = {
                "fact_count": int(fact_count),
                "contradiction_rate": float(contradiction_count / fact_count if fact_count > 0 else 0),
                "memory_usage_mb": float(memory_usage_mb)
            }
        memory_logger.debug(f"Metrics fetched: {metrics}")
        return metrics
    except Exception as e:
        memory_logger.error(f"Metrics error: {str(e)}")
        raise


# Import settings
settings = get_settings()

# Memory Operation Metrics
memory_operations_total = Counter(
    'memory_operations_total',
    'Total memory operations',
    ['operation', 'status']
)

memory_operation_duration_seconds = Histogram(
    'memory_operation_duration_seconds',
    'Memory operation latency',
    ['operation'],
    buckets=[0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
)

# Fact Management Metrics
active_facts_total = Gauge('active_facts_total', 'Total active facts in memory')
contradiction_score = Gauge('contradiction_score', 'Average contradiction score')
compression_ratio = Gauge('compression_ratio', 'Memory compression ratio')
volatility_score = Gauge('volatility_score', 'Average volatility score')

# API Metrics
api_requests_total = Counter(
    'api_requests_total',
    'Total API requests',
    ['method', 'endpoint', 'status_code']
)

api_request_duration_seconds = Histogram(
    'api_request_duration_seconds',
    'API request latency',
    ['method', 'endpoint'],
    buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
)

# Background Task Metrics
background_tasks_total = Counter(
    'background_tasks_total',
    'Total background tasks executed',
    ['task_name', 'status']
)

background_task_duration_seconds = Histogram(
    'background_task_duration_seconds',
    'Background task execution time',
    ['task_name'],
    buckets=[1.0, 5.0, 10.0, 30.0, 60.0, 300.0, 600.0]
)

# Causal Linkage Metrics
cognitive_causal_links_total = Counter(
    'cognitive_causal_links_total',
    'Total causal links created',
    ['status']  # created, failed, skipped
)

cognitive_causal_link_strength = Histogram(
    'cognitive_causal_link_strength',
    'Distribution of causal link strength scores',
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
)

cognitive_causal_analysis_duration_seconds = Histogram(
    'cognitive_causal_analysis_duration_seconds',
    'Time spent analyzing causal relationships',
    buckets=[0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0]
)

cognitive_temporal_proximity_score = Histogram(
    'cognitive_temporal_proximity_score',
    'Distribution of temporal proximity scores',
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
)

cognitive_semantic_similarity_score = Histogram(
    'cognitive_semantic_similarity_score', 
    'Distribution of semantic similarity scores',
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
)

cognitive_logical_consistency_score = Histogram(
    'cognitive_logical_consistency_score',
    'Distribution of logical consistency scores', 
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
)

cognitive_causal_failures_total = Counter(
    'cognitive_causal_failures_total',
    'Total causal analysis failures',
    ['reason']  # missing_timestamps, low_similarity, calculation_error
)

# System Metrics
system_memory_bytes = Gauge('system_memory_bytes', 'System memory usage in bytes')
system_cpu_percent = Gauge('system_cpu_percent', 'System CPU usage percentage')
database_connections = Gauge('database_connections', 'Active database connections')
cache_hit_ratio = Gauge('cache_hit_ratio', 'Cache hit ratio')

# Error Metrics
errors_total = Counter(
    'errors_total',
    'Total errors',
    ['error_type', 'component']
)

# Performance Metrics
embedding_operations_total = Counter(
    'embedding_operations_total',
    'Total embedding operations',
    ['operation', 'status']
)

embedding_operation_duration_seconds = Histogram(
    'embedding_operation_duration_seconds',
    'Embedding operation latency',
    ['operation'],
    buckets=[0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0]
)

# Clustering Metrics
clustering_operations_total = Counter(
    'clustering_operations_total',
    'Total clustering operations',
    ['operation', 'status']
)

clusters_total = Gauge('clusters_total', 'Total number of clusters')
cluster_size_average = Gauge('cluster_size_average', 'Average cluster size')

# Security Metrics
security_events_total = Counter(
    'security_events_total',
    'Total security events',
    ['event_type', 'severity']
)

rate_limit_hits_total = Counter(
    'rate_limit_hits_total',
    'Total rate limit hits',
    ['client_ip']
)


def track_memory_operation(operation: str):
    """Decorator to track memory operations with metrics."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                memory_operations_total.labels(operation=operation, status="success").inc()
                return result
            except Exception as e:
                memory_operations_total.labels(operation=operation, status="error").inc()
                errors_total.labels(error_type=type(e).__name__, component="memory").inc()
                raise
            finally:
                duration = time.time() - start_time
                memory_operation_duration_seconds.labels(operation=operation).observe(duration)
        return wrapper
    return decorator


def track_api_request(method: str, endpoint: str):
    """Decorator to track API requests with metrics."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                api_requests_total.labels(method=method, endpoint=endpoint, status_code=200).inc()
                return result
            except Exception as e:
                api_requests_total.labels(method=method, endpoint=endpoint, status_code=500).inc()
                errors_total.labels(error_type=type(e).__name__, component="api").inc()
                raise
            finally:
                duration = time.time() - start_time
                api_request_duration_seconds.labels(method=method, endpoint=endpoint).observe(duration)
        return wrapper
    return decorator


def track_background_task(task_name: str):
    """Decorator to track background tasks with metrics."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                background_tasks_total.labels(task_name=task_name, status="success").inc()
                return result
            except Exception as e:
                background_tasks_total.labels(task_name=task_name, status="error").inc()
                errors_total.labels(error_type=type(e).__name__, component="background").inc()
                raise
            finally:
                duration = time.time() - start_time
                background_task_duration_seconds.labels(task_name=task_name).observe(duration)
        return wrapper
    return decorator


def update_memory_metrics(memory_stats: Dict[str, Any]):
    """Update memory-related metrics."""
    active_facts_total.set(memory_stats.get('total_facts', 0))
    
    if 'avg_contradiction_score' in memory_stats:
        contradiction_score.set(memory_stats['avg_contradiction_score'])
    
    if 'compression_ratio' in memory_stats:
        compression_ratio.set(memory_stats['compression_ratio'])
    
    if 'avg_volatility_score' in memory_stats:
        volatility_score.set(memory_stats['avg_volatility_score'])


def update_system_metrics():
    """Update system-level metrics."""
    try:
        # Memory usage
        memory = psutil.virtual_memory()
        system_memory_bytes.set(memory.used)
        
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=1)
        system_cpu_percent.set(cpu_percent)
        
    except Exception as e:
        # Don't let metrics collection break the application
        pass


def update_clustering_metrics(cluster_stats: Dict[str, Any]):
    """Update clustering-related metrics."""
    clusters_total.set(cluster_stats.get('total_clusters', 0))
    cluster_size_average.set(cluster_stats.get('average_cluster_size', 0))


def record_security_event(event_type: str, severity: str):
    """Record security events."""
    security_events_total.labels(event_type=event_type, severity=severity).inc()


def record_rate_limit_hit(client_ip: str):
    """Record rate limit hits."""
    rate_limit_hits_total.labels(client_ip=client_ip).inc()


def record_embedding_operation(operation: str, duration: float, success: bool = True):
    """Record embedding operation metrics."""
    status = "success" if success else "error"
    embedding_operations_total.labels(operation=operation, status=status).inc()
    embedding_operation_duration_seconds.labels(operation=operation).observe(duration)


def record_cache_metrics(hits: int, misses: int):
    """Record cache performance metrics."""
    total = hits + misses
    if total > 0:
        hit_ratio = hits / total
        cache_hit_ratio.set(hit_ratio)


def get_metrics_response():
    """Get Prometheus metrics response."""
    return generate_latest()


def get_metrics_content_type():
    """Get Prometheus metrics content type."""
    return CONTENT_TYPE_LATEST


def get_dashboard_metrics_dict():
    """Return a dictionary of key dashboard metrics for real-time display."""
    memory_log = MemoryLog()
    facts = memory_log.get_all_facts()
    contradictions = memory_log.get_contradictions(resolved=False)
    clusters = memory_log.list_clusters()
    return {
        "fact_count": len(facts),
        "contradiction_count": len(contradictions),
        "cluster_count": len(clusters),
        "contradiction_rate": (len(contradictions) / len(facts)) if facts else 0.0,
    }


# Metrics collection interval (in seconds)
METRICS_UPDATE_INTERVAL = 60

# Start periodic metrics updates
def start_metrics_collection():
    """Start periodic collection of system metrics."""
    import threading
    import time
    
    def update_metrics():
        while True:
            try:
                update_system_metrics()
                time.sleep(METRICS_UPDATE_INTERVAL)
            except Exception as e:
                # Log error but don't stop collection
                print(f"Error updating metrics: {e}")
                time.sleep(METRICS_UPDATE_INTERVAL)
    
    metrics_thread = threading.Thread(target=update_metrics, daemon=True)
    metrics_thread.start()
    return metrics_thread

# Export aliases for compatibility
memory_operations_counter = memory_operations_total
api_latency_histogram = api_request_duration_seconds 