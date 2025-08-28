#!/usr/bin/env python3
"""
Test suite for MeRNSTA memory system fixes.
Tests fact retrieval, contradiction detection, summarization, and personality biasing.
"""

import pytest
import sqlite3
import tempfile
import os
from unittest.mock import patch, MagicMock
from datetime import datetime

from storage.memory_log import MemoryLog
from storage.memory_search import MemorySearchEngine  
from storage.fact_manager import FactManager
from cortex.memory_ops import process_user_input, publish_memory_update
from cortex.personality_ops import apply_personality, load_personality_profiles
from config.environment import Settings


class TestMemoryFixes:
    """Test suite for memory system fixes."""
    
    @pytest.fixture
    def memory_log(self):
        """Create a test memory log with in-memory database."""
        # Create temporary database file
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
            db_path = tmp.name
        
        # Initialize memory log
        memory_log = MemoryLog(db_path)
        
        yield memory_log
        
        # Cleanup
        memory_log.shutdown()
        if os.path.exists(db_path):
            os.unlink(db_path)
    
    @pytest.fixture
    def search_engine(self, memory_log):
        """Create a test search engine."""
        return MemorySearchEngine(memory_log)
    
    @pytest.fixture
    def fact_manager(self, memory_log):
        """Create a test fact manager."""
        return FactManager(memory_log)
    
    def test_fact_storage_and_retrieval(self, memory_log, search_engine):
        """Test that facts are stored and retrieved correctly."""
        
        # Store some test facts
        triplets = [
            ("user", "dislike", "weird people", 0.8),
            ("user", "hate", "racist people", 0.9),
            ("user", "like", "red", 0.85)
        ]
        
        memory_log.store_triplets(triplets, user_profile_id="test_user")
        
        # Test fact retrieval for "what people do i not like"
        result = search_engine.search_facts(
            "what people do i not like",
            user_profile_id="test_user",
            personality="neutral"
        )
        
        assert result["success"] is True
        assert len(result["facts"]) > 0
        assert "Found" in result["response"]
        
        # Check that dislike/hate facts are found
        fact_texts = [f"{f['subject']} {f['predicate']} {f['object']}" for f in result["facts"]]
        assert any("weird people" in text for text in fact_texts)
        assert any("racist people" in text for text in fact_texts)
    
    def test_color_preference_retrieval(self, memory_log, search_engine):
        """Test retrieval of color preferences."""
        
        # Store color preference
        triplets = [("user", "like", "red", 0.85)]
        memory_log.store_triplets(triplets, user_profile_id="test_user")
        
        # Test retrieval
        result = search_engine.search_facts(
            "what color is my fav",
            user_profile_id="test_user", 
            personality="neutral"
        )
        
        assert result["success"] is True
        assert len(result["facts"]) > 0
        assert "red" in result["response"]
        assert "Found" in result["response"]
    
    def test_contradiction_detection(self, fact_manager, memory_log):
        """Test that contradictions are properly detected and logged."""
        
        # Store initial fact
        triplets1 = [("user", "like", "yellow cats", 0.8)]
        result1 = fact_manager.store_facts_with_validation(
            triplets1, 
            user_profile_id="test_user"
        )
        assert result1["success"] is True
        assert len(result1["contradictions"]) == 0
        
        # Store contradictory fact
        triplets2 = [("user", "hate", "yellow cats", 0.9)]
        result2 = fact_manager.store_facts_with_validation(
            triplets2,
            user_profile_id="test_user"
        )
        
        assert result2["success"] is True
        assert len(result2["contradictions"]) > 0
        
        # Check contradiction details
        contradiction = result2["contradictions"][0]
        assert "yellow cats" in contradiction["new_fact"]
        assert "yellow cats" in contradiction["existing_fact"]
        assert contradiction["score"] > 0.5  # High contradiction score
    
    def test_summarization_functionality(self, memory_log, search_engine):
        """Test that summarization generates proper summaries."""
        
        # Store various facts
        triplets = [
            ("user", "like", "red", 0.85),
            ("user", "dislike", "weird people", 0.8), 
            ("user", "hate", "racist people", 0.9),
            ("user", "have", "a dog", 0.7)
        ]
        memory_log.store_triplets(triplets, user_profile_id="test_user")
        
        # Test summarization
        result = search_engine.search_facts(
            "summarize our conversation so far",
            user_profile_id="test_user",
            personality="neutral"
        )
        
        assert result["success"] is True
        assert result["type"] == "summarization"
        
        # Check that summary contains expected elements
        response = result["response"].lower()
        assert "mentioned" in response or "like" in response or "dislike" in response
        
        # Ensure the command itself wasn't stored as a fact
        all_facts = memory_log.get_all_facts()
        command_facts = [f for f in all_facts if "summarize" in f.subject.lower() or "summarize" in f.object.lower()]
        assert len(command_facts) == 0  # No command should be stored as fact
    
    def test_personality_biasing_skeptical(self, search_engine, memory_log):
        """Test that skeptical personality reduces confidence scores."""
        
        # Store a fact
        triplets = [("user", "like", "blue", 1.0)]
        memory_log.store_triplets(triplets, user_profile_id="test_user")
        
        # Search with neutral personality
        result_neutral = search_engine.search_facts(
            "what do i like",
            user_profile_id="test_user",
            personality="neutral"
        )
        
        # Search with skeptical personality
        result_skeptical = search_engine.search_facts(
            "what do i like", 
            user_profile_id="test_user",
            personality="skeptical"
        )
        
        assert result_neutral["success"] is True
        assert result_skeptical["success"] is True
        
        # Get confidence scores
        neutral_confidence = result_neutral["facts"][0]["confidence"] if result_neutral["facts"] else 1.0
        skeptical_confidence = result_skeptical["facts"][0]["confidence"] if result_skeptical["facts"] else 1.0
        
        # Skeptical should have lower confidence (1.0 / 1.5 = 0.67)
        assert skeptical_confidence < neutral_confidence
        assert abs(skeptical_confidence - (1.0 / 1.5)) < 0.1  # Should be around 0.67
    
    def test_personality_loading(self):
        """Test that personality profiles are loaded correctly from config."""
        
        profiles = load_personality_profiles()
        
        # Check required personalities exist
        assert "neutral" in profiles
        assert "skeptical" in profiles
        assert "enthusiastic" in profiles
        
        # Check skeptical personality has correct multiplier
        skeptical = profiles["skeptical"]
        assert skeptical["multiplier"] == 1.5
        assert skeptical["confidence_adjustment"] == -0.05
    
    def test_apply_personality_function(self):
        """Test the apply_personality function directly."""
        
        # Test facts
        facts = [
            {"id": 1, "subject": "user", "predicate": "like", "object": "red", "confidence": 0.8},
            {"id": 2, "subject": "user", "predicate": "hate", "object": "blue", "confidence": 0.9}
        ]
        
        # Apply skeptical personality
        adjusted_facts = apply_personality(facts, "skeptical")
        
        assert len(adjusted_facts) == 2
        
        # Check confidence adjustments
        for i, fact in enumerate(adjusted_facts):
            original_confidence = facts[i]["confidence"]
            adjusted_confidence = fact["confidence"]
            
            # Should be original / 1.5 - 0.05
            expected = (original_confidence / 1.5) - 0.05
            expected = max(0.0, min(1.0, expected))  # Clamp to [0,1]
            
            assert abs(adjusted_confidence - expected) < 0.01
            assert fact["personality_applied"] == "skeptical"
            assert "original_confidence" in fact
    
    @patch('cortex.memory_ops.publish_memory_update')
    def test_websocket_updates(self, mock_publish, memory_log, search_engine):
        """Test that WebSocket updates are published correctly."""
        
        # Test query WebSocket update
        search_engine.search_facts(
            "what do i like",
            user_profile_id="test_user",
            personality="neutral"
        )
        
        # Should not call publish directly from search_engine, but from higher level
        # Let's test the process_user_input function instead
        with patch('cortex.memory_ops.get_database_path') as mock_path:
            mock_path.return_value = ":memory:"  
            
            response, tokens, personality, metadata = process_user_input(
                "what do i like",
                current_personality="neutral",
                session_id="test_session",
                request_context={"client_ip": "127.0.0.1"}
            )
            
            # Check that publish_memory_update was called
            assert mock_publish.called
            call_args = mock_publish.call_args[0][0]  # First argument of first call
            assert call_args["type"] == "query_result"
            assert "test_session" in call_args["session_id"] or call_args["session_id"] is not None
    
    def test_user_profile_isolation(self, memory_log, search_engine):
        """Test that facts are properly isolated by user_profile_id."""
        
        # Store facts for different users
        triplets_user1 = [("user", "like", "coffee", 0.8)]
        triplets_user2 = [("user", "like", "tea", 0.9)]
        
        memory_log.store_triplets(triplets_user1, user_profile_id="user1")
        memory_log.store_triplets(triplets_user2, user_profile_id="user2")
        
        # Query as user1
        result1 = search_engine.search_facts(
            "what do i like",
            user_profile_id="user1",
            personality="neutral"
        )
        
        # Query as user2
        result2 = search_engine.search_facts(
            "what do i like", 
            user_profile_id="user2",
            personality="neutral"
        )
        
        # Each user should only see their own facts
        assert result1["success"] is True
        assert result2["success"] is True
        
        result1_text = result1["response"].lower()
        result2_text = result2["response"].lower()
        
        # User1 should see coffee, not tea
        if "coffee" in result1_text:
            assert "tea" not in result1_text
        
        # User2 should see tea, not coffee
        if "tea" in result2_text:
            assert "coffee" not in result2_text
    
    def test_database_schema(self, memory_log):
        """Test that database tables have correct schema."""
        
        with memory_log._connection_pool.get_connection() as conn:
            # Check facts table
            cursor = conn.execute("PRAGMA table_info(facts)")
            facts_columns = {row[1]: row[2] for row in cursor.fetchall()}
            
            required_facts_columns = [
                "id", "subject", "predicate", "object", "confidence", 
                "timestamp", "user_profile_id", "session_id"
            ]
            
            for col in required_facts_columns:
                assert col in facts_columns
            
            # Check contradictions table exists
            cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='contradictions'")
            assert cursor.fetchone() is not None
            
            # Check episodes table exists  
            cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='episodes'")
            assert cursor.fetchone() is not None
    
    def test_full_memory_pipeline(self, memory_log):
        """Integration test of the full memory pipeline."""
        
        # Test the complete flow: store facts, detect contradictions, retrieve, summarize
        
        # 1. Store initial facts
        triplets1 = [
            ("user", "like", "red", 0.85),
            ("user", "dislike", "weird people", 0.8)
        ]
        memory_log.store_triplets(triplets1, user_profile_id="test_user", session_id="session1")
        
        # 2. Store contradictory fact
        triplets2 = [("user", "hate", "red", 0.9)]  # Contradicts liking red
        fact_manager = FactManager(memory_log)
        result = fact_manager.store_facts_with_validation(triplets2, user_profile_id="test_user")
        
        # Should detect contradiction
        assert len(result["contradictions"]) > 0
        
        # 3. Test retrieval
        search_engine = MemorySearchEngine(memory_log)
        
        # Query about dislikes
        dislike_result = search_engine.search_facts(
            "what people do i not like",
            user_profile_id="test_user",
            personality="neutral"
        )
        assert dislike_result["success"] is True
        assert "weird people" in dislike_result["response"]
        
        # Query about colors with skeptical personality
        color_result = search_engine.search_facts(
            "what color do i like",
            user_profile_id="test_user", 
            personality="skeptical"
        )
        assert color_result["success"] is True
        
        # 4. Test summarization
        summary_result = search_engine.search_facts(
            "summarize our conversation",
            user_profile_id="test_user",
            personality="neutral"
        )
        assert summary_result["success"] is True
        assert summary_result["type"] == "summarization"
        
        # Verify that all components work together
        all_facts = memory_log.get_all_facts()
        user_facts = [f for f in all_facts if getattr(f, 'user_profile_id', None) == "test_user"]
        assert len(user_facts) >= 3  # Should have stored the facts


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 