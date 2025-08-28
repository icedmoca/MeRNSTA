# tests/test_hlb_adapter.py

"""
Unit tests for HLB (Hadamard-derived Linear Binding) adapter.

Tests vector shape, determinism, and binding invertibility properties.
"""

import pytest
import torch
import numpy as np
import sys
import os
from unittest.mock import patch, Mock

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Test imports
try:
    from vector_memory.hlb_adapter import (
        HLBVectorizer,
        hlb_vectorize,
        check_hlb_dependencies,
        get_hlb_info,
        HLB_AVAILABLE
    )
    HLB_IMPORT_SUCCESS = True
except ImportError as e:
    HLB_IMPORT_SUCCESS = False
    import_error = str(e)


class TestHLBDependencies:
    """Test HLB dependency checking."""
    
    def test_check_hlb_dependencies(self):
        """Test dependency checking function."""
        deps = check_hlb_dependencies()
        
        assert isinstance(deps, dict)
        assert 'hlb_available' in deps
        assert 'pytorch_available' in deps
        assert 'external_path_exists' in deps
        assert 'status' in deps
        
        # Status should be one of expected values
        assert deps['status'] in ['ready', 'not_ready', 'missing_external_repo', 'missing_pytorch', 'missing_hlb_import']
    
    def test_get_hlb_info(self):
        """Test HLB info function."""
        info = get_hlb_info()
        
        assert isinstance(info, dict)
        assert 'dependencies' in info
        assert 'note' in info
        
        if HLB_AVAILABLE:
            assert 'backend' in info
            assert 'description' in info
            assert 'vector_size' in info


@pytest.mark.skipif(not HLB_IMPORT_SUCCESS, reason="HLB adapter import failed")
class TestHLBVectorizer:
    """Test HLBVectorizer class functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        if not HLB_AVAILABLE:
            pytest.skip("HLB not available")
        self.vectorizer = HLBVectorizer(vector_dim=128)  # Smaller dim for faster tests
    
    def test_vectorizer_initialization(self):
        """Test vectorizer initialization."""
        vectorizer = HLBVectorizer(vector_dim=256, device='cpu')
        
        assert vectorizer.vector_dim == 256
        assert vectorizer.device == 'cpu'
        assert hasattr(vectorizer, '_word_vector_cache')
        assert isinstance(vectorizer._word_vector_cache, dict)
    
    def test_encode_basic_functionality(self):
        """Test basic encoding functionality."""
        text = "hello world"
        vector = self.vectorizer.encode(text)
        
        # Check return type and shape
        assert isinstance(vector, torch.Tensor)
        assert vector.shape == (self.vectorizer.vector_dim,)
        assert vector.dtype == torch.float32
    
    def test_encode_empty_text(self):
        """Test encoding empty text."""
        vector = self.vectorizer.encode("")
        
        assert isinstance(vector, torch.Tensor)
        assert vector.shape == (self.vectorizer.vector_dim,)
        assert torch.all(vector == 0.0)  # Should be zero vector
    
    def test_encode_determinism(self):
        """Test that encoding is deterministic."""
        text = "artificial intelligence"
        
        # Encode the same text multiple times
        vector1 = self.vectorizer.encode(text)
        vector2 = self.vectorizer.encode(text)
        vector3 = self.vectorizer.encode(text)
        
        # All vectors should be identical
        assert torch.allclose(vector1, vector2, atol=1e-6)
        assert torch.allclose(vector2, vector3, atol=1e-6)
    
    def test_encode_different_texts_different_vectors(self):
        """Test that different texts produce different vectors."""
        text1 = "machine learning"
        text2 = "deep learning"
        
        vector1 = self.vectorizer.encode(text1)
        vector2 = self.vectorizer.encode(text2)
        
        # Vectors should be different
        assert not torch.allclose(vector1, vector2, atol=1e-3)
    
    def test_word_caching(self):
        """Test that word vectors are cached correctly."""
        # Clear cache
        self.vectorizer._word_vector_cache.clear()
        
        text = "test caching"
        self.vectorizer.encode(text)
        
        # Cache should now contain word vectors
        assert len(self.vectorizer._word_vector_cache) > 0
        assert "test" in self.vectorizer._word_vector_cache
        assert "caching" in self.vectorizer._word_vector_cache
    
    def test_vector_properties(self):
        """Test mathematical properties of generated vectors."""
        text = "vector properties test"
        vector = self.vectorizer.encode(text)
        
        # Vector should be normalized (since we normalize in the implementation)
        norm = torch.norm(vector)
        assert abs(norm.item() - 1.0) < 1e-3  # Should be approximately unit norm
        
        # Vector should not be all zeros (except for empty input)
        assert not torch.all(vector == 0.0)
    
    def test_hlb_binding_properties(self):
        """Test HLB binding operation properties."""
        if not HLB_AVAILABLE:
            pytest.skip("HLB not available")
            
        # Test that we can access HLBTensor binding operations
        word_vector = self.vectorizer._generate_word_vector("test")
        pos_vector = self.vectorizer._generate_position_vector(0)
        
        # Test binding
        bound = word_vector.bind(pos_vector)
        assert bound.shape == word_vector.shape
        
        # Test that binding is different from original vectors
        assert not torch.allclose(bound.squeeze(), word_vector.squeeze(), atol=1e-3)
        assert not torch.allclose(bound.squeeze(), pos_vector.squeeze(), atol=1e-3)


@pytest.mark.skipif(not HLB_IMPORT_SUCCESS, reason="HLB adapter import failed")
class TestHLBBindingInvertibility:
    """Test HLB binding invertibility properties."""
    
    def setup_method(self):
        """Set up test fixtures."""
        if not HLB_AVAILABLE:
            pytest.skip("HLB not available")
        self.vectorizer = HLBVectorizer(vector_dim=128)
    
    def test_binding_inverse_property(self):
        """Test that HLB binding has proper inverse properties."""
        # Generate test vectors
        word_vec = self.vectorizer._generate_word_vector("test")
        pos_vec = self.vectorizer._generate_position_vector(0)
        
        # Bind vectors
        bound = word_vec.bind(pos_vec)
        
        # Test inverse property: bound.bind(pos_vec.inverse()) should approximate word_vec
        pos_inverse = pos_vec.inverse()
        recovered = bound.bind(pos_inverse)
        
        # Should be similar to original word vector (allowing for some numerical error)
        similarity = torch.cosine_similarity(word_vec.squeeze(), recovered.squeeze(), dim=0)
        assert similarity > 0.8  # Should have high similarity
    
    def test_bundling_distributive_property(self):
        """Test that bundling distributes over binding."""
        # Create test vectors
        a = self.vectorizer._generate_word_vector("a")
        b = self.vectorizer._generate_word_vector("b") 
        c = self.vectorizer._generate_word_vector("c")
        
        # Test distributive property: (a + b) * c ≈ a * c + b * c
        left_side = a.bundle(b).bind(c)
        right_side = a.bind(c).bundle(b.bind(c))
        
        # Should be approximately equal
        similarity = torch.cosine_similarity(left_side.squeeze(), right_side.squeeze(), dim=0)
        assert similarity > 0.8


@pytest.mark.skipif(not HLB_IMPORT_SUCCESS, reason="HLB adapter import failed")
class TestHLBConvenienceFunction:
    """Test the hlb_vectorize convenience function."""
    
    def test_hlb_vectorize_basic(self):
        """Test basic hlb_vectorize function."""
        if not HLB_AVAILABLE:
            # When HLB not available, should return zero vector
            result = hlb_vectorize("test")
            assert isinstance(result, list)
            assert len(result) == 1024  # Default dimension
            assert all(v == 0.0 for v in result)
        else:
            result = hlb_vectorize("test")
            assert isinstance(result, list)
            assert len(result) == 1024  # Default dimension
            assert not all(v == 0.0 for v in result)  # Should not be all zeros
    
    def test_hlb_vectorize_determinism(self):
        """Test determinism of convenience function."""
        text = "determinism test"
        
        result1 = hlb_vectorize(text)
        result2 = hlb_vectorize(text)
        
        # Should be identical
        assert result1 == result2
    
    @patch('vector_memory.hlb_adapter.HLBVectorizer')
    def test_hlb_vectorize_error_handling(self, mock_vectorizer_class):
        """Test error handling in hlb_vectorize."""
        # Mock vectorizer to raise an exception
        mock_vectorizer = Mock()
        mock_vectorizer.encode.side_effect = Exception("Test error")
        mock_vectorizer_class.return_value = mock_vectorizer
        
        result = hlb_vectorize("test")
        
        # Should return zero vector on error
        assert isinstance(result, list)
        assert len(result) == 1024
        assert all(v == 0.0 for v in result)


class TestHLBWithoutDependencies:
    """Test HLB adapter behavior when dependencies are not available."""
    
    @patch('vector_memory.hlb_adapter.HLB_AVAILABLE', False)
    def test_vectorizer_init_without_hlb(self):
        """Test vectorizer initialization when HLB is not available."""
        with pytest.raises(ImportError, match="HLB dependencies not available"):
            HLBVectorizer()
    
    @patch('vector_memory.hlb_adapter.HLB_AVAILABLE', False)
    def test_hlb_vectorize_without_hlb(self):
        """Test convenience function when HLB is not available."""
        result = hlb_vectorize("test")
        
        assert isinstance(result, list)
        assert len(result) == 1024
        assert all(v == 0.0 for v in result)


class TestHLBIntegration:
    """Integration tests for HLB adapter."""
    
    def test_vector_memory_integration(self):
        """Test integration with vector_memory module."""
        try:
            from vector_memory import get_vectorizer
            
            # Should be able to get HLB vectorizer
            vectorizer = get_vectorizer('hlb')
            assert callable(vectorizer)
            
            # Test vectorization
            result = vectorizer("integration test")
            assert isinstance(result, list)
            assert len(result) > 0
            
        except ImportError:
            pytest.skip("vector_memory module not available")
    
    def test_config_integration(self):
        """Test integration with config module."""
        try:
            from vector_memory.config import get_vectorizer_info
            
            info = get_vectorizer_info()
            assert isinstance(info, dict)
            
        except ImportError:
            pytest.skip("vector_memory.config module not available")


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v"])