import pytest
from unittest.mock import patch, MagicMock
from storage.spacy_extractor import extract_triplets
from storage.memory_log import MemoryLog
from storage.db_utils import reset_pool
from config.settings import DATABASE_CONFIG
import os
import numpy as np

@pytest.fixture(scope="function")
def mock_embedders():
    """Mock all embedder instances to avoid connection issues"""
    
    def smart_embed(text):
        """Return different but similar embeddings for related terms"""
        text = text.lower()
        if "function" in text or "my function" in text:
            return [0.1, 0.2, 0.3, 0.4, 0.5]  # Base function embedding
        elif "sum" in text:
            return [0.15, 0.25, 0.35, 0.4, 0.5]  # Similar to function but slightly different
        elif "name" in text:
            return [0.2, 0.1, 0.4, 0.3, 0.5]   # Different pattern
        elif "alice" in text:
            return [0.3, 0.2, 0.1, 0.4, 0.5]   # Different pattern  
        else:
            return [0.1, 0.2, 0.3, 0.4, 0.5]   # Default
    
    with patch('storage.spacy_extractor.embedder') as mock_spacy_embedder, \
         patch('storage.spacy_extractor.OllamaEmbedder') as mock_ollama_class:
        
        # Mock the global embedder instance in spacy_extractor
        mock_spacy_embedder.embed.side_effect = smart_embed
        
        # Mock the OllamaEmbedder class for memory_log instances
        mock_ollama_instance = MagicMock()
        mock_ollama_instance.embed.side_effect = smart_embed
        mock_ollama_class.return_value = mock_ollama_instance
        
        yield {
            'spacy_embedder': mock_spacy_embedder,
            'ollama_class': mock_ollama_class,
            'ollama_instance': mock_ollama_instance
        }

@pytest.fixture(scope="function")
def memory_log(mock_embedders):
    # Reset connection pool before test to ensure clean state
    reset_pool()
    
    # Use in-memory SQLite database for complete isolation
    db_path = ":memory:"
    # Patch DATABASE_CONFIG for this test
    orig_path = DATABASE_CONFIG.get("path", "memory.db")
    DATABASE_CONFIG["path"] = db_path
    
    try:
        # Create MemoryLog with in-memory database and mocks in place
        log = MemoryLog(db_path)
        yield log
    finally:
        # Cleanup: reset pool and restore original config
        reset_pool()
        DATABASE_CONFIG["path"] = orig_path

def test_clustering(memory_log, mock_embedders):
    triplets1 = extract_triplets("my name is Alice", message_id=1)
    memory_log.store_triplets(triplets1)
    triplets2 = extract_triplets("user name is Alice", message_id=2)
    memory_log.store_triplets(triplets2)
    # Both should cluster to 'name'
    assert triplets1[0][0] == triplets2[0][0] or triplets2[0][0] == "name"
    
    # Use the same user_profile_id that was stored with the triplets
    user_profile_id = '32bcd336456e'  # This should match what's stored  
    results = memory_log.semantic_search("what's my name", user_profile_id=user_profile_id)
    # Handle None objects in results
    valid_results = [r for r in results if r.get("object") is not None]
    assert any(r["object"].lower() == "alice" for r in valid_results)

def test_code_handling(memory_log, mock_embedders):
    code = "def sum(a, b): return a + b"
    triplets = extract_triplets(f"my function calculates sum with {code}", message_id=3)
    memory_log.store_triplets(triplets)
    
    # Search for function-related content using question format
    # Use the same user_profile_id that was stored with the triplets  
    user_profile_id = '32bcd336456e'  # This should match what's stored
    results = memory_log.semantic_search("what is my function", user_profile_id=user_profile_id)
    
    # Also try searching for sum-related content  
    sum_results = memory_log.semantic_search("what calculates sum", user_profile_id=user_profile_id)
    
    assert len(results) > 0 or len(sum_results) > 0 