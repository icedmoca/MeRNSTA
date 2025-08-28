"""
Vector Memory Adapters for MeRNSTA

This package provides adapters for different memory backends:
- HRRFormer: High-dimensional symbolic compression
- VecSymR: Human-like analogical mapping  
- Default: Standard FAISS semantic search (current system)
"""

from .hrrformer_adapter import hrrformer_vectorize  # noqa: F401
from .vecsymr_adapter import vecsymr_vectorize  # noqa: F401
from .hlb_adapter import hlb_vectorize  # noqa: F401
from .hybrid_memory import HybridVectorMemory  # noqa: F401
from .memory_source_logger import memory_source_logger  # noqa: F401

__all__ = ['hrrformer_vectorize', 'vecsymr_vectorize', 'hlb_vectorize', 'get_vectorizer', 'HybridVectorMemory', 'memory_source_logger']

def get_vectorizer(model='default'):
    """
    Get vectorizer function based on model name.
    
    Args:
        model: 'default', 'hrrformer', 'vecsymr', or 'hlb'
        
    Returns:
        Vectorization function that takes text and returns list of floats
    """
    if model == 'hrrformer':
        return hrrformer_vectorize
    elif model == 'vecsymr':
        return vecsymr_vectorize
    elif model == 'hlb':
        return hlb_vectorize
    else:
        # Default fallback vectorizer
        def default_vectorize(text: str) -> list[float]:
            """Default fallback vectorizer - returns zero vector"""
            import logging
            logging.warning("Using fallback vectorizer - no embeddings generated")
            return [0.0] * 384  # Standard embedding dimension
        return default_vectorize