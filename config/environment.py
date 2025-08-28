#!/usr/bin/env python3
"""
Environment-driven configuration for MeRNSTA enterprise deployment.
Uses Pydantic settings for validation and type safety.
"""

import os
from pathlib import Path
from typing import List, Optional

from pydantic import Field, field_validator
from pydantic import ConfigDict
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Centralized configuration for MeRNSTA enterprise deployment."""

    # Environment
    environment: str = Field(default="development", json_schema_extra={"env": "ENVIRONMENT"})
    debug: bool = Field(default=False, json_schema_extra={"env": "DEBUG"})

    # Database Configuration
    database_url: str = Field(default="sqlite:///memory.db", json_schema_extra={"env": "DATABASE_URL"})
    max_connections: int = Field(default=10, json_schema_extra={"env": "MAX_CONNECTIONS"})
    database_timeout: float = Field(default=30.0, json_schema_extra={"env": "DATABASE_TIMEOUT"})

    # Memory System Configuration
    max_facts: int = Field(default=1000000, json_schema_extra={"env": "MAX_FACTS"})
    compression_threshold: float = Field(default=0.8, json_schema_extra={"env": "COMPRESSION_THRESHOLD"})
    min_cluster_size: int = Field(default=3, json_schema_extra={"env": "MIN_CLUSTER_SIZE"})
    similarity_threshold: float = Field(default=0.7, json_schema_extra={"env": "SIMILARITY_THRESHOLD"})

    # Background Tasks Configuration
    reconciliation_interval: int = Field(default=300, json_schema_extra={"env": "RECONCILIATION_INTERVAL"})
    compression_interval: int = Field(default=3600, json_schema_extra={"env": "COMPRESSION_INTERVAL"})
    celery_broker_url: str = Field(
        default="redis://localhost:6379/0", json_schema_extra={"env": "CELERY_BROKER_URL"}
    )
    celery_result_backend: str = Field(
        default="redis://localhost:6379/0", json_schema_extra={"env": "CELERY_RESULT_BACKEND"}
    )
    max_concurrent_tasks: int = Field(default=10, json_schema_extra={"env": "MAX_CONCURRENT_TASKS"})

    # Monitoring Configuration
    metrics_port: int = Field(default=9090, json_schema_extra={"env": "METRICS_PORT"})
    log_level: str = Field(default="INFO", json_schema_extra={"env": "LOG_LEVEL"})
    enable_tracing: bool = Field(default=True, json_schema_extra={"env": "ENABLE_TRACING"})
    prometheus_enabled: bool = Field(default=True, json_schema_extra={"env": "PROMETHEUS_ENABLED"})

    # Security Configuration
    api_security_token: str = Field(default="test_token_for_testing", json_schema_extra={"env": "API_SECURITY_TOKEN"})
    rate_limit: int = Field(default=100, json_schema_extra={"env": "RATE_LIMIT"})
    rate_limit_window: int = Field(default=60, json_schema_extra={"env": "RATE_LIMIT_WINDOW"})
    disable_rate_limit: bool = Field(default=False, json_schema_extra={"env": "DISABLE_RATE_LIMIT"})

    # Cache Configuration
    redis_url: str = Field(default="redis://localhost:6379/0", json_schema_extra={"env": "REDIS_URL"})
    cache_ttl: int = Field(default=3600, json_schema_extra={"env": "CACHE_TTL"})
    enable_caching: bool = Field(default=True, json_schema_extra={"env": "ENABLE_CACHING"})

    # LLM Configuration
    ollama_base_url: str = Field(
        default="http://localhost:11434", json_schema_extra={"env": "OLLAMA_BASE_URL"}
    )
    default_model: str = Field(default="mistral", json_schema_extra={"env": "DEFAULT_MODEL"})

    # Feature Flags
    enable_compression: bool = Field(default=True, json_schema_extra={"env": "ENABLE_COMPRESSION"})
    enable_auto_reconciliation: bool = Field(
        default=True, json_schema_extra={"env": "ENABLE_AUTO_RECONCILIATION"}
    )
    enable_personality_biasing: bool = Field(
        default=True, json_schema_extra={"env": "ENABLE_PERSONALITY_BIASING"}
    )
    enable_emotion_analysis: bool = Field(default=True, json_schema_extra={"env": "ENABLE_EMOTION_ANALYSIS"})

    # Performance Configuration
    batch_size: int = Field(default=1000, json_schema_extra={"env": "BATCH_SIZE"})
    embedding_cache_size: int = Field(default=10000, json_schema_extra={"env": "EMBEDDING_CACHE_SIZE"})

    # Sentiment Analysis Configuration
    sentiment_love_intensity: float = Field(default=0.8, json_schema_extra={"env": "SENTIMENT_LOVE_INTENSITY"})
    sentiment_like_intensity: float = Field(default=0.6, json_schema_extra={"env": "SENTIMENT_LIKE_INTENSITY"})
    sentiment_dislike_intensity: float = Field(default=-0.6, json_schema_extra={"env": "SENTIMENT_DISLIKE_INTENSITY"})
    sentiment_hate_intensity: float = Field(default=-0.8, json_schema_extra={"env": "SENTIMENT_HATE_INTENSITY"})

    # Logging Configuration
    log_format: str = Field(default="json", json_schema_extra={"env": "LOG_FORMAT"})
    log_file: str = Field(default="logs/mernsta.log", json_schema_extra={"env": "LOG_FILE"})
    background_log_file: str = Field(
        default="logs/background.log", json_schema_extra={"env": "BACKGROUND_LOG_FILE"}
    )

    # Health Check Configuration
    health_check_timeout: int = Field(default=30, json_schema_extra={"env": "HEALTH_CHECK_TIMEOUT"})
    health_check_interval: int = Field(default=60, json_schema_extra={"env": "HEALTH_CHECK_INTERVAL"})

    # Network Configuration Defaults
    websocket_port: int = Field(default=8002, json_schema_extra={"env": "WEBSOCKET_PORT"})
    websocket_channel: str = Field(default="memory_updates", json_schema_extra={"env": "WEBSOCKET_CHANNEL"})
    
    # Database Configuration Defaults
    database_path: str = Field(default="memory.db", json_schema_extra={"env": "DATABASE_PATH"})
    database_journal_mode: str = Field(default="WAL", json_schema_extra={"env": "DATABASE_JOURNAL_MODE"})
    
    # Behavior Configuration Defaults
    cross_session_search_enabled: bool = Field(default=True, json_schema_extra={"env": "CROSS_SESSION_SEARCH_ENABLED"})
    auto_reconcile: bool = Field(default=True, json_schema_extra={"env": "AUTO_RECONCILE"})
    semantic_drift_threshold: float = Field(default=0.35, json_schema_extra={"env": "SEMANTIC_DRIFT_THRESHOLD"})
    max_facts_summary: int = Field(default=100, json_schema_extra={"env": "MAX_FACTS_SUMMARY"})
    multimodal_similarity_threshold: float = Field(default=0.7, json_schema_extra={"env": "MULTIMODAL_SIMILARITY_THRESHOLD"})
    default_personality: str = Field(default="neutral", json_schema_extra={"env": "DEFAULT_PERSONALITY"})
    
    # Causal linkage configuration
    causal_link_threshold: float = Field(default=0.35, json_schema_extra={"env": "CAUSAL_LINK_THRESHOLD"})
    temporal_decay_lambda: float = Field(default=0.1, json_schema_extra={"env": "TEMPORAL_DECAY_LAMBDA"})

    @field_validator("compression_threshold", "similarity_threshold")
    @classmethod
    def validate_thresholds(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError("Thresholds must be between 0.0 and 1.0")
        return v

    @field_validator("sentiment_love_intensity", "sentiment_like_intensity", "sentiment_dislike_intensity", "sentiment_hate_intensity")
    @classmethod
    def validate_sentiment_intensities(cls, v):
        if not -1.0 <= v <= 1.0:
            raise ValueError("Sentiment intensities must be between -1.0 and 1.0")
        return v

    @field_validator("max_facts", "max_connections", "batch_size")
    @classmethod
    def validate_positive_integers(cls, v):
        if v <= 0:
            raise ValueError("Values must be positive integers")
        return v

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v):
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"Log level must be one of: {valid_levels}")
        return v.upper()

    model_config = ConfigDict(env_file=".env", env_file_encoding="utf-8", case_sensitive=False)


# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    """Get the global settings instance."""
    return settings


def reload_settings() -> Settings:
    """Reload settings from environment."""
    global settings
    settings = Settings()
    return settings
