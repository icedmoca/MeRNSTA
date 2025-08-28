# vector_memory/hlb_adapter.py

"""
HLB (Hadamard-derived Linear Binding) Adapter for MeRNSTA

This adapter integrates HLB's symbolic vector encoding capabilities
for enhanced symbolic reasoning and binding operations.

Based on: "A Walsh Hadamard Derived Linear Vector Symbolic Architecture"
(Alam et al., 2024) - https://arxiv.org/abs/2410.22669
"""

import logging
import torch
import numpy as np
from typing import List, Optional
import hashlib
import sys
import os
from math import sqrt

# Add the HLB external directory to path for importing
HLB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "external", "hlb", "Classical VSA Tasks")
if HLB_PATH not in sys.path:
    sys.path.append(HLB_PATH)

try:
    from vsa_models import HLBTensor
    HLB_AVAILABLE = True
except ImportError as e:
    logging.warning(f"HLB not available: {e}")
    HLB_AVAILABLE = False
    HLBTensor = None


class HLBVectorizer:
    """
    Vectorizer using Hadamard-derived Linear Binding (HLB) for symbolic text encoding.
    
    HLB provides:
    - Element-wise multiplication for binding operations  
    - Element-wise addition for bundling operations
    - Invertible operations for symbolic reasoning
    - 1024-dimensional vectors for rich representation
    """
    
    def __init__(self, vector_dim: int = 1024, device: Optional[str] = None):
        """
        Initialize HLB vectorizer.
        
        Args:
            vector_dim: Dimensionality of output vectors (default: 1024)
            device: PyTorch device ('cpu', 'cuda', etc.)
        """
        if not HLB_AVAILABLE:
            raise ImportError("HLB dependencies not available. Please check external/hlb installation.")
            
        self.vector_dim = vector_dim
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Cache for word vectors to ensure deterministic encoding
        self._word_vector_cache = {}
        
        logging.info(f"[HLB] Initialized with {vector_dim}D vectors on {self.device}")
    
    def encode(self, text: str) -> torch.Tensor:
        """
        Encode text into HLB vector representation.
        
        Args:
            text: Input text to encode
            
        Returns:
            torch.Tensor: 1024-dimensional HLB vector
        """
        if not HLB_AVAILABLE:
            raise RuntimeError("HLB not available")
            
        try:
            return self._hlb_encode_text(text)
        except Exception as e:
            logging.error(f"[HLB] Encoding failed for text '{text[:50]}...': {e}")
            # Return zero vector as fallback
            return torch.zeros(self.vector_dim, device=self.device, dtype=torch.float32)
    
    def _hlb_encode_text(self, text: str) -> torch.Tensor:
        """
        Core HLB encoding implementing symbolic vector operations.
        
        Process:
        1. Tokenize text into words
        2. Generate deterministic HLB vectors for each word
        3. Create position vectors for sequential binding
        4. Bind each word with its position using HLB operations
        5. Bundle all bound vectors into final representation
        """
        words = text.lower().strip().split()
        if not words:
            return torch.zeros(self.vector_dim, device=self.device, dtype=torch.float32)
        
        # Generate word vectors
        word_vectors = []
        for word in words:
            if word not in self._word_vector_cache:
                self._word_vector_cache[word] = self._generate_word_vector(word)
            word_vectors.append(self._word_vector_cache[word])
        
        # Generate position vectors
        position_vectors = []
        for i in range(len(words)):
            pos_vector = self._generate_position_vector(i)
            position_vectors.append(pos_vector)
        
        # Bind words with positions using HLB operations
        bound_vectors = []
        for word_vec, pos_vec in zip(word_vectors, position_vectors):
            # HLB binding is element-wise multiplication
            bound = word_vec.bind(pos_vec)
            bound_vectors.append(bound)
        
        # Bundle all bound vectors (element-wise addition)
        if len(bound_vectors) == 1:
            result = bound_vectors[0]
        else:
            # Stack and use multibundle
            stacked = torch.stack([bv.squeeze() for bv in bound_vectors])
            result = HLBTensor(stacked).multibundle().squeeze()
        
        # Normalize to prevent overflow
        norm = torch.norm(result)
        if norm > 0:
            result = result / norm
            
        return result.to(device=self.device)
    
    def _generate_word_vector(self, word: str) -> HLBTensor:
        """Generate deterministic HLB vector for a word."""
        # Use word hash as seed for deterministic generation
        seed = int(hashlib.md5(word.encode()).hexdigest()[:8], 16)
        
        # Set seed for reproducibility
        generator = torch.Generator()
        generator.manual_seed(seed)
        
        # Generate HLB random vector using the same distribution as in vsa_models.py
        uniform = torch.rand(self.vector_dim, generator=generator)
        n1 = torch.normal(-1, 1 / sqrt(self.vector_dim), (self.vector_dim,), generator=generator)
        n2 = torch.normal(1, 1 / sqrt(self.vector_dim), (self.vector_dim,), generator=generator)
        vector = torch.where(uniform > 0.5, n1, n2)
        
        return HLBTensor(vector.unsqueeze(0))
    
    def _generate_position_vector(self, position: int) -> HLBTensor:
        """Generate deterministic HLB vector for a position."""
        # Use position hash as seed
        pos_str = f"pos_{position}"
        seed = int(hashlib.md5(pos_str.encode()).hexdigest()[:8], 16)
        
        generator = torch.Generator()
        generator.manual_seed(seed)
        
        # Generate position vector with same distribution
        uniform = torch.rand(self.vector_dim, generator=generator)
        n1 = torch.normal(-1, 1 / sqrt(self.vector_dim), (self.vector_dim,), generator=generator)
        n2 = torch.normal(1, 1 / sqrt(self.vector_dim), (self.vector_dim,), generator=generator)
        vector = torch.where(uniform > 0.5, n1, n2)
        
        return HLBTensor(vector.unsqueeze(0))
    
    def get_info(self) -> dict:
        """Get information about HLB configuration."""
        return {
            "backend": "HLB",
            "description": "Hadamard-derived Linear Binding for symbolic vector encoding",
            "vector_size": self.vector_dim,
            "device": self.device,
            "available": HLB_AVAILABLE,
            "features": [
                "Element-wise multiplication binding",
                "Element-wise addition bundling", 
                "Invertible symbolic operations",
                "Deterministic text encoding",
                "1024-dimensional representation"
            ]
        }


def hlb_vectorize(text: str) -> List[float]:
    """
    Convenience function for HLB vectorization compatible with MeRNSTA interface.
    
    Args:
        text: Input text to vectorize
        
    Returns:
        List of floats representing the HLB vector
    """
    try:
        if not HLB_AVAILABLE:
            logging.error("[HLB] HLB not available, returning zero vector")
            return [0.0] * 1024
            
        vectorizer = HLBVectorizer()
        vector = vectorizer.encode(text)
        logging.info(f"[HLB] Generated {len(vector)} dimensional vector for: '{text[:50]}...'")
        return vector.cpu().tolist()
        
    except Exception as e:
        logging.error(f"[HLB] Vectorization failed: {e}")
        return [0.0] * 1024


def check_hlb_dependencies() -> dict:
    """
    Check if HLB dependencies are available.
    
    Returns:
        Dictionary with status information
    """
    status = {
        "hlb_available": HLB_AVAILABLE,
        "pytorch_available": True,  # Assumed since we're importing torch
        "external_path_exists": os.path.exists(HLB_PATH),
        "status": "not_ready"
    }
    
    try:
        import torch
        status["pytorch_available"] = True
    except ImportError:
        status["pytorch_available"] = False
    
    if HLB_AVAILABLE and status["pytorch_available"] and status["external_path_exists"]:
        status["status"] = "ready"
    elif not status["external_path_exists"]:
        status["status"] = "missing_external_repo"
    elif not status["pytorch_available"]:
        status["status"] = "missing_pytorch"
    else:
        status["status"] = "missing_hlb_import"
    
    return status


def get_hlb_info() -> dict:
    """Get information about HLB configuration."""
    deps = check_hlb_dependencies()
    vectorizer_info = {}
    
    if HLB_AVAILABLE:
        try:
            vectorizer = HLBVectorizer()
            vectorizer_info = vectorizer.get_info()
        except Exception as e:
            vectorizer_info = {"error": str(e)}
    
    return {
        **vectorizer_info,
        "dependencies": deps,
        "note": "Requires external HLB repository and PyTorch"
    }