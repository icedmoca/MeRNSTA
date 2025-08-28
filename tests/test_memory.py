#!/usr/bin/env python3
"""
Test script for MeRNSTA's persistent memory system
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import sqlite3
import tempfile
import unittest
from storage.memory_log import MemoryLog, MemoryEntry, Fact
from storage.memory_utils import TripletFact

class TestMemorySystem(unittest.TestCase):
    """Test cases for the persistent memory system"""
    
    def setUp(self):
        """Set up test fixtures"""
        import tempfile
        import os
        
        # Use shared in-memory database to avoid connection pool issues
        self.db_path = "file:memdb1?mode=memory&cache=shared"
        
        # Initialize memory log with shared in-memory database
        self.memory_log = MemoryLog(self.db_path)
        
        # Clear any existing data to ensure clean test state
        try:
            from tests.utils.db_cleanup import safe_cleanup_database
            safe_cleanup_database(self.db_path)
        except Exception as e:
            # Skip cleanup if there are issues with in-memory database
            print(f"Warning: Could not clean database: {e}")
            pass
    
    def tearDown(self):
        """Clean up test database"""
        # No cleanup needed for in-memory database
        pass
    
    def test_memory_logging(self):
        """Test basic memory logging functionality"""
        # Log user message
        user_id = self.memory_log.log_memory("user", "My name is Alice", tags=["introduction"])
        self.assertIsInstance(user_id, int)
        self.assertGreater(user_id, 0)
        
        # Log assistant response
        assistant_id = self.memory_log.log_memory("assistant", "Nice to meet you Alice!", tags=["greeting"])
        self.assertIsInstance(assistant_id, int)
        self.assertGreater(assistant_id, 0)
        
        # Check memory stats
        stats = self.memory_log.get_memory_stats()
        # The database might have existing messages, so check that we have at least our 2 messages
        self.assertGreaterEqual(stats['total_messages'], 2)
        self.assertGreaterEqual(stats['user_messages'], 1)
        self.assertGreaterEqual(stats['assistant_messages'], 1)
    
    def test_context_recall(self):
        """Test contextual recall functionality"""
        # Add some test messages
        self.memory_log.log_memory("user", "I like pizza", tags=["preference"])
        self.memory_log.log_memory("assistant", "Pizza is great!", tags=["response"])
        self.memory_log.log_memory("user", "What's my favorite food?", tags=["question"])
        
        # Test recent context
        recent = self.memory_log.fetch_recent_context(3)
        self.assertEqual(len(recent), 3)
        self.assertEqual(recent[0].role, "user")
        self.assertEqual(recent[0].content, "I like pizza")
        
        # Test semantic context
        semantic = self.memory_log.fetch_semantic_context("food", 2)
        self.assertGreater(len(semantic), 0)
    
    def test_fact_extraction(self):
        """Test fact extraction from text"""
        message_id = self.memory_log.log_memory("user", "My name is Bob and I love coffee")
        facts = self.memory_log.extract_facts("My name is Bob and I love coffee")
        self.assertIsInstance(facts, list)
        # Should extract at least one fact
        self.assertGreater(len(facts), 0)
        
        # Test that facts have the expected structure
        if facts:
            fact = facts[0]
            self.assertIsInstance(fact, TripletFact)
            self.assertIsInstance(fact.subject, str)
            self.assertIsInstance(fact.predicate, str)
            self.assertIsInstance(fact.object, str)
    
    def test_memory_formatting(self):
        """Test memory formatting for LLM context"""
        # Add test messages
        self.memory_log.log_memory("user", "Hello", tags=["greeting"])
        self.memory_log.log_memory("assistant", "Hi there!", tags=["response"])
        
        # Get entries and format
        entries = self.memory_log.fetch_recent_context(2)
        formatted = self.memory_log.format_context_for_llm(entries)
        
        self.assertIn("user: Hello", formatted)
        self.assertIn("assistant: Hi there!", formatted)
    
    def test_database_structure(self):
        """Test database table structure"""
        with self.memory_log._connection_pool.get_connection() as conn:
            # Check memory table
            cursor = conn.execute("PRAGMA table_info(memory)")
            columns = [row[1] for row in cursor.fetchall()]
            expected_columns = ['id', 'timestamp', 'role', 'content', 'embedding', 'tags']
            for col in expected_columns:
                self.assertIn(col, columns)

            # Check facts table
            cursor = conn.execute("PRAGMA table_info(facts)")
            columns = [row[1] for row in cursor.fetchall()]
            expected_columns = ['id', 'subject', 'predicate', 'object', 'source_message_id', 'timestamp', 'frequency', 'contradiction_score', 'volatility_score', 'confidence', 'last_reinforced', 'episode_id', 'emotion_score', 'context']
            for col in expected_columns:
                self.assertIn(col, columns)

# Note: This file uses unittest.TestCase, which does not support pytest fixtures directly.
# To use isolated_db, a refactor to pytest-style tests is needed. Leaving as is for now.
def run_demo():
    """Run a demonstration of the memory system"""
    print("🧠 MeRNSTA Memory System Demo")
    print("=" * 50)
    
    # Create memory log
    memory_log = MemoryLog("demo_memory.db")
    
    # Simulate a conversation
    print("\n1. Logging conversation...")
    user_id1 = memory_log.log_memory("user", "My name is Sarah", tags=["introduction"])
    assistant_id1 = memory_log.log_memory("assistant", "Nice to meet you Sarah!", tags=["greeting"])
    
    user_id2 = memory_log.log_memory("user", "I love reading science fiction", tags=["preference"])
    assistant_id2 = memory_log.log_memory("assistant", "Science fiction is fascinating! What's your favorite book?", tags=["question"])
    
    user_id3 = memory_log.log_memory("user", "Dune is my favorite", tags=["preference"])
    
    # Extract facts
    print("\n2. Extracting facts...")
    facts1 = memory_log.extract_facts("My name is Sarah")
    facts2 = memory_log.extract_facts("I love reading science fiction")
    facts3 = memory_log.extract_facts("Dune is my favorite")
    
    all_facts = facts1 + facts2 + facts3
    memory_log.store_facts(all_facts)
    
    # Show memory stats
    print("\n3. Memory Statistics:")
    stats = memory_log.get_memory_stats()
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    # Show recent context
    print("\n4. Recent Conversation:")
    recent = memory_log.fetch_recent_context(5)
    for entry in recent:
        print(f"   {entry.role}: {entry.content}")
    
    # Show extracted facts
    print("\n5. Extracted Facts:")
    user_facts = memory_log.get_facts_about("user_name")
    preference_facts = memory_log.get_facts_about("preference")
    
    for fact in user_facts + preference_facts:
        print(f"   {fact.entity}: {fact.value} (confidence: {fact.confidence:.2f})")
    
    # Test semantic search
    print("\n6. Semantic Search for 'books':")
    semantic = memory_log.fetch_semantic_context("books", 3)
    for entry in semantic:
        print(f"   {entry.role}: {entry.content}")
    
    print("\n✅ Demo completed! Check 'demo_memory.db' for the database.")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "demo":
        run_demo()
    else:
        unittest.main() 