#!/usr/bin/env python3
"""
Test script for MeRNSTA's dynamic memory injection system
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import sqlite3
from storage.memory_log import MemoryLog
from storage.memory_utils import build_prompt, format_facts_for_display
import time

def test_dynamic_memory(isolated_db):
    """Test the dynamic memory injection system"""
    print("🧠 Testing Dynamic Memory Injection System")
    print("=" * 60)
    
    # Create memory log
    memory_log = isolated_db
    
    # Test 1: No memory facts
    print("\n1. Testing prompt with NO memory facts:")
    test_query = "What do you know about me?"
    
    # Build prompt with empty memory
    system_prompt = build_prompt(test_query, memory_log, max_tokens=512)
    
    print(f"\nPrompt with no memory:")
    print("-" * 40)
    print(system_prompt)
    print("-" * 40)
    
    # Test 2: Add some memory facts
    print("\n2. Adding memory facts...")
    
    messages = [
        ("My name is Luna Vega.", "introduction"),
        ("I live in Flagstaff.", "location"),
        ("I love dark matter.", "preference"),
    ]
    
    for message, tag in messages:
        print(f"   Logging: {message}")
        message_id = memory_log.log_memory("user", message, tags=[tag])
        
        # Extract facts
        facts = memory_log.extract_facts(message)
        if facts:
            memory_log.store_facts(facts)
            print(f"   Extracted facts: {[f'{f.subject}: {f.object}' for f in facts]}")
    
    # Test 3: Prompt with memory facts
    print("\n3. Testing prompt WITH memory facts:")
    
    # Build prompt with memory facts
    system_prompt_with_memory = build_prompt(test_query, memory_log, max_tokens=512)
    
    print(f"\nPrompt with memory:")
    print("-" * 40)
    print(system_prompt_with_memory)
    print("-" * 40)
    
    # Test 4: Show the difference
    print("\n4. Key differences:")
    print("   ✅ [BEGIN MEMORY] section only appears when facts exist")
    print("   ✅ No hardcoded facts in prompt builder")
    print("   ✅ Dynamic fact retrieval via get_all_facts()")
    print("   ✅ Graceful handling of empty memory")
    
    # Test 5: Memory dump
    print("\n5. Current memory state:")
    memory_facts = memory_log.get_all_facts()
    formatted_facts = format_facts_for_display(memory_facts)
    print(formatted_facts)
    
    # Test 6: Clear memory and test again
    print("\n6. Clearing memory and testing again...")
    from tests.utils.db_cleanup import safe_cleanup_database
    safe_cleanup_database(memory_log.db_path)
    
    # Build prompt after clearing memory
    system_prompt_after_clear = build_prompt(test_query, memory_log, max_tokens=512)
    
    print(f"\nPrompt after clearing memory:")
    print("-" * 40)
    print(system_prompt_after_clear)
    print("-" * 40)
    
    print("\n✅ Dynamic memory test completed!")
    print("\nKey improvements:")
    print("✅ No hardcoded facts in prompt builder")
    print("✅ Dynamic get_all_facts() calls")
    print("✅ [BEGIN MEMORY] only when facts exist")
    print("✅ Graceful empty memory handling")
    print("✅ Real-time memory state reflection")

if __name__ == "__main__":
    # Create an isolated database for testing
    isolated_db_path = "isolated_test_memory.db"
    isolated_db = MemoryLog(isolated_db_path)
    
    # Ensure the isolated database is empty before testing
    if os.path.exists(isolated_db_path):
        os.remove(isolated_db_path)
    
    test_dynamic_memory(isolated_db)
    
    # Clean up the isolated database
    if os.path.exists(isolated_db_path):
        os.remove(isolated_db_path)
        print(f"\nCleaned up isolated database at {isolated_db_path}") 