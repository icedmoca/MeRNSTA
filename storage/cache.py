#!/usr/bin/env python3
"""
Redis-based caching system for MeRNSTA enterprise deployment.
Provides caching for embeddings, cluster centroids, and frequently accessed data.
"""

import hashlib
import json
import time
from typing import Any, Dict, List, Optional

import numpy as np
import redis

from config.environment import get_settings
from monitoring.logger import get_logger
from monitoring.metrics import record_cache_metrics

# Import settings
settings = get_settings()
logger = get_logger("cache")

# Cache statistics
_cache_hits = 0
_cache_misses = 0


class MemoryCache:
    """Redis-based caching system for MeRNSTA memory operations."""

    def __init__(self, redis_url: str = None):
        self.redis_url = redis_url or settings.redis_url
        self.default_ttl = settings.cache_ttl
        self.enable_caching = settings.enable_caching

        if self.enable_caching:
            try:
                self.redis = redis.from_url(self.redis_url)
                # Test connection
                self.redis.ping()
                logger.info("Cache system initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize cache: {e}")
                self.enable_caching = False
        else:
            self.redis = None
            logger.info("Caching disabled")

    def _get_cache_key(self, prefix: str, key: str) -> str:
        """Generate a cache key with prefix."""
        return f"mernsta:{prefix}:{key}"

    def _serialize_embedding(self, embedding: np.ndarray) -> str:
        """Serialize numpy array to string for caching."""
        return embedding.tobytes().hex()

    def _deserialize_embedding(self, data: str) -> np.ndarray:
        """Deserialize string back to numpy array."""
        return np.frombuffer(bytes.fromhex(data), dtype=np.float32)

    def get_embedding(self, text: str) -> Optional[np.ndarray]:
        """Get cached embedding for text."""
        global _cache_hits, _cache_misses

        if not self.enable_caching or not self.redis:
            _cache_misses += 1
            return None

        try:
            # Create hash of text for consistent key
            text_hash = hashlib.md5(text.encode()).hexdigest()
            cache_key = self._get_cache_key("embedding", text_hash)

            cached_data = self.redis.get(cache_key)
            if cached_data:
                _cache_hits += 1
                embedding = self._deserialize_embedding(cached_data.decode())
                logger.debug(f"Cache hit for embedding: {text[:50]}...")
                return embedding
            else:
                _cache_misses += 1
                logger.debug(f"Cache miss for embedding: {text[:50]}...")
                return None

        except Exception as e:
            logger.error(f"Error getting embedding from cache: {e}")
            _cache_misses += 1
            return None

    def set_embedding(self, text: str, embedding: np.ndarray, ttl: int = None):
        """Cache embedding for text."""
        if not self.enable_caching or not self.redis:
            return

        try:
            text_hash = hashlib.md5(text.encode()).hexdigest()
            cache_key = self._get_cache_key("embedding", text_hash)
            serialized_embedding = self._serialize_embedding(embedding)

            ttl = ttl or self.default_ttl
            self.redis.setex(cache_key, ttl, serialized_embedding)
            logger.debug(f"Cached embedding: {text[:50]}...")

        except Exception as e:
            logger.error(f"Error caching embedding: {e}")

    def get_cluster_centroid(self, subject: str) -> Optional[np.ndarray]:
        """Get cached cluster centroid for subject."""
        global _cache_hits, _cache_misses

        if not self.enable_caching or not self.redis:
            _cache_misses += 1
            return None

        try:
            cache_key = self._get_cache_key("centroid", subject.lower())
            cached_data = self.redis.get(cache_key)

            if cached_data:
                _cache_hits += 1
                centroid = self._deserialize_embedding(cached_data.decode())
                logger.debug(f"Cache hit for centroid: {subject}")
                return centroid
            else:
                _cache_misses += 1
                logger.debug(f"Cache miss for centroid: {subject}")
                return None

        except Exception as e:
            logger.error(f"Error getting centroid from cache: {e}")
            _cache_misses += 1
            return None

    def set_cluster_centroid(self, subject: str, centroid: np.ndarray, ttl: int = None):
        """Cache cluster centroid for subject."""
        if not self.enable_caching or not self.redis:
            return

        try:
            cache_key = self._get_cache_key("centroid", subject.lower())
            serialized_centroid = self._serialize_embedding(centroid)

            ttl = ttl or self.default_ttl
            self.redis.setex(cache_key, ttl, serialized_centroid)
            logger.debug(f"Cached centroid: {subject}")

        except Exception as e:
            logger.error(f"Error caching centroid: {e}")

    def get_memory_stats(self) -> Optional[Dict[str, Any]]:
        """Get cached memory statistics."""
        global _cache_hits, _cache_misses

        if not self.enable_caching or not self.redis:
            _cache_misses += 1
            return None

        try:
            cache_key = self._get_cache_key("stats", "memory")
            cached_data = self.redis.get(cache_key)

            if cached_data:
                _cache_hits += 1
                stats = json.loads(cached_data.decode())
                logger.debug("Cache hit for memory stats")
                return stats
            else:
                _cache_misses += 1
                logger.debug("Cache miss for memory stats")
                return None

        except Exception as e:
            logger.error(f"Error getting memory stats from cache: {e}")
            _cache_misses += 1
            return None

    def set_memory_stats(self, stats: Dict[str, Any], ttl: int = None):
        """Cache memory statistics."""
        if not self.enable_caching or not self.redis:
            return

        try:
            cache_key = self._get_cache_key("stats", "memory")
            serialized_stats = json.dumps(stats)

            ttl = ttl or 300  # 5 minutes for stats
            self.redis.setex(cache_key, ttl, serialized_stats)
            logger.debug("Cached memory stats")

        except Exception as e:
            logger.error(f"Error caching memory stats: {e}")

    def get_semantic_search_results(
        self, query: str, topk: int
    ) -> Optional[List[Dict[str, Any]]]:
        """Get cached semantic search results."""
        global _cache_hits, _cache_misses

        if not self.enable_caching or not self.redis:
            _cache_misses += 1
            return None

        try:
            query_hash = hashlib.md5(f"{query}:{topk}".encode()).hexdigest()
            cache_key = self._get_cache_key("search", query_hash)
            cached_data = self.redis.get(cache_key)

            if cached_data:
                _cache_hits += 1
                results = json.loads(cached_data.decode())
                logger.debug(f"Cache hit for search: {query[:50]}...")
                return results
            else:
                _cache_misses += 1
                logger.debug(f"Cache miss for search: {query[:50]}...")
                return None

        except Exception as e:
            logger.error(f"Error getting search results from cache: {e}")
            _cache_misses += 1
            return None

    def set_semantic_search_results(
        self, query: str, topk: int, results: List[Dict[str, Any]], ttl: int = None
    ):
        """Cache semantic search results."""
        if not self.enable_caching or not self.redis:
            return

        try:
            query_hash = hashlib.md5(f"{query}:{topk}".encode()).hexdigest()
            cache_key = self._get_cache_key("search", query_hash)
            serialized_results = json.dumps(results)

            ttl = ttl or 600  # 10 minutes for search results
            self.redis.setex(cache_key, ttl, serialized_results)
            logger.debug(f"Cached search results: {query[:50]}...")

        except Exception as e:
            logger.error(f"Error caching search results: {e}")

    def get_cluster_info(self, subject: str) -> Optional[Dict[str, Any]]:
        """Get cached cluster information."""
        global _cache_hits, _cache_misses

        if not self.enable_caching or not self.redis:
            _cache_misses += 1
            return None

        try:
            cache_key = self._get_cache_key("cluster", subject.lower())
            cached_data = self.redis.get(cache_key)

            if cached_data:
                _cache_hits += 1
                cluster_info = json.loads(cached_data.decode())
                logger.debug(f"Cache hit for cluster info: {subject}")
                return cluster_info
            else:
                _cache_misses += 1
                logger.debug(f"Cache miss for cluster info: {subject}")
                return None

        except Exception as e:
            logger.error(f"Error getting cluster info from cache: {e}")
            _cache_misses += 1
            return None

    def set_cluster_info(
        self, subject: str, cluster_info: Dict[str, Any], ttl: int = None
    ):
        """Cache cluster information."""
        if not self.enable_caching or not self.redis:
            return

        try:
            cache_key = self._get_cache_key("cluster", subject.lower())
            serialized_info = json.dumps(cluster_info)

            ttl = ttl or self.default_ttl
            self.redis.setex(cache_key, ttl, serialized_info)
            logger.debug(f"Cached cluster info: {subject}")

        except Exception as e:
            logger.error(f"Error caching cluster info: {e}")

    def invalidate_embeddings(self, pattern: str = "*"):
        """Invalidate cached embeddings matching pattern."""
        if not self.enable_caching or not self.redis:
            return

        try:
            cache_key_pattern = self._get_cache_key("embedding", pattern)
            keys = self.redis.keys(cache_key_pattern)
            if keys:
                self.redis.delete(*keys)
                logger.info(f"Invalidated {len(keys)} embedding cache entries")
        except Exception as e:
            logger.error(f"Error invalidating embeddings: {e}")

    def invalidate_clusters(self, subject: str = None):
        """Invalidate cached cluster data."""
        if not self.enable_caching or not self.redis:
            return

        try:
            if subject:
                cache_key = self._get_cache_key("cluster", subject.lower())
                self.redis.delete(cache_key)
                logger.info(f"Invalidated cluster cache for: {subject}")
            else:
                cache_key_pattern = self._get_cache_key("cluster", "*")
                keys = self.redis.keys(cache_key_pattern)
                if keys:
                    self.redis.delete(*keys)
                    logger.info(f"Invalidated {len(keys)} cluster cache entries")
        except Exception as e:
            logger.error(f"Error invalidating clusters: {e}")

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics."""
        global _cache_hits, _cache_misses

        total = _cache_hits + _cache_misses
        hit_ratio = _cache_hits / total if total > 0 else 0.0

        # Update metrics
        record_cache_metrics(_cache_hits, _cache_misses)

        return {
            "hits": _cache_hits,
            "misses": _cache_misses,
            "total": total,
            "hit_ratio": hit_ratio,
            "enabled": self.enable_caching,
            "redis_connected": self.redis is not None and self.enable_caching,
        }

    def clear_cache(self):
        """Clear all cached data."""
        if not self.enable_caching or not self.redis:
            return

        try:
            keys = self.redis.keys("mernsta:*")
            if keys:
                self.redis.delete(*keys)
                logger.info(f"Cleared {len(keys)} cache entries")
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")

    def get_cache_size(self) -> Dict[str, int]:
        """Get cache size information."""
        if not self.enable_caching or not self.redis:
            return {"total_keys": 0, "memory_usage": 0}

        try:
            keys = self.redis.keys("mernsta:*")
            memory_info = self.redis.info("memory")

            return {
                "total_keys": len(keys),
                "memory_usage": memory_info.get("used_memory", 0),
            }
        except Exception as e:
            logger.error(f"Error getting cache size: {e}")
            return {"total_keys": 0, "memory_usage": 0}


# Global cache instance
_cache_instance = None


def get_cache() -> MemoryCache:
    """Get the global cache instance."""
    global _cache_instance
    if _cache_instance is None:
        _cache_instance = MemoryCache()
    return _cache_instance


def clear_cache():
    """Clear the global cache."""
    global _cache_instance
    if _cache_instance:
        _cache_instance.clear_cache()


def get_cache_stats() -> Dict[str, Any]:
    """Get cache statistics."""
    cache = get_cache()
    return cache.get_cache_stats()
