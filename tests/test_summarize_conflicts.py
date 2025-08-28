#!/usr/bin/env python3
"""
Test script for the summarize_conflicts command
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from storage.memory_log import MemoryLog

def test_summarize_conflicts(isolated_db):
    """Test the summarize_conflicts command"""
    print("🧪 Testing summarize_conflicts command")
    print("=" * 50)
    
    # Create a memory log for testing
    memory_log = isolated_db
    
    # Add contradictory facts about pizza
    test_messages = [
        "I love pizza",
        "Pizza makes me sick",
        "I eat pizza every day",
        "I avoid pizza because it's unhealthy",
        "My favorite food is pizza",
        "I never eat pizza anymore"
    ]
    
    print("Adding contradictory facts about pizza:")
    for message in test_messages:
        message_id = memory_log.log_memory("user", message)
        triplets = memory_log.extract_triplets(message)
        if triplets:
            memory_log.store_triplets(triplets, message_id)
            print(f"  ✓ {message}")
    
    print(f"\n📊 Total facts: {len(memory_log.get_all_facts())}")
    
    # Test summarize_conflicts for pizza
    print(f"\n🔍 Testing summarize_conflicts for 'pizza':")
    print("-" * 40)
    
    try:
        summary = memory_log.summarize_conflicts_llm("pizza")
        print(summary)
        print("\n✅ summarize_conflicts test completed successfully!")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    # Clean up
    if os.path.exists("test_conflicts.db"):
        os.remove("test_conflicts.db")

if __name__ == "__main__":
    # Create an isolated database for testing
    isolated_db = MemoryLog("test_conflicts.db")
    test_summarize_conflicts(isolated_db) 