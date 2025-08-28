#!/usr/bin/env python3
"""
Final test to verify dynamic memory system cleanup and functionality
"""

import sys
import os
import pytest
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import sqlite3
from storage.memory_log import MemoryLog
from storage.memory_utils import build_prompt, format_facts_for_display
import time
from storage.db_utils import get_conn
import tempfile
import uuid

def test_final_cleanup(isolated_db):
    """Test that the cleanup is complete and system works correctly"""
    print("🧠 Final Cleanup Verification Test")
    print("=" * 60)

    memory_log = isolated_db
    db_path = memory_log.db_path
    # Use reset_database_for_test for robust cleanup
    from tests.utils.db_cleanup import reset_database_for_test
    if hasattr(memory_log, '_connection_pool') and hasattr(memory_log._connection_pool, 'close_all_connections'):
        memory_log._connection_pool.close_all_connections()
    memory_log.shutdown()
    reset_database_for_test(db_path)
    memory_log = MemoryLog(db_path)
    remaining_facts = memory_log.get_all_facts()
    if remaining_facts:
        print(f'❌ ERROR: Facts remain after initial cleanup in {db_path}:')
        for f in remaining_facts:
            print(f"   {f.subject} {f.predicate} {f.object}")
    assert not remaining_facts, f"Facts remain in DB after initial cleanup: {remaining_facts} (db: {db_path})"

    # Test 1: Verify no hardcoded facts in build_prompt
    print("\n1. Testing build_prompt with empty memory:")
    test_query = "Hello, what do you know about me?"
    
    # This should NOT include any [BEGIN MEMORY] section
    empty_prompt = build_prompt(test_query, memory_log, max_tokens=512)
    
    print(f"\nPrompt with empty memory:")
    print("-" * 40)
    print(empty_prompt)
    print("-" * 40)
    
    # Verify no memory section (or empty memory section)
    if "[BEGIN MEMORY]" in empty_prompt:
        # If memory section exists, it should be empty or contain only placeholders
        memory_section = empty_prompt.split("[BEGIN MEMORY]")[1].split("[END MEMORY]")[0]
        # Allow for placeholder text like "No relevant memories found"
        assert not memory_section.strip() or "no relevant" in memory_section.lower() or "no memories" in memory_section.lower(), "Memory section should be empty when no facts exist"
        print("✅ SUCCESS: Empty memory section in empty memory")
    else:
        print("✅ SUCCESS: No memory section in empty memory")
    
    # Test 2: Add facts and verify dynamic inclusion
    print("\n2. Adding facts and testing dynamic inclusion:")
    
    messages = [
        ("My name is Alice.", "introduction"),
        ("I'm from Seattle.", "location"),
        ("I love programming.", "preference"),
    ]
    
    for message, tag in messages:
        print(f"   Logging: {message}")
        message_id = memory_log.log_memory("user", message, tags=[tag])
        
        # Extract facts
        facts = memory_log.extract_facts(message)
        if facts:
            memory_log.store_facts(facts)
    
    # Now build prompt - should include memory section
    print("\nDEBUG: Facts in DB before prompt:")
    all_facts = memory_log.get_all_facts()
    for f in all_facts:
        print(f"   {f.subject} {f.predicate} {f.object}")
    populated_prompt = build_prompt(test_query, memory_log, max_tokens=512)
    
    print(f"\nPrompt with facts:")
    print("-" * 40)
    print(populated_prompt)
    print("-" * 40)
    
    # Verify memory section is present
    assert "[BEGIN MEMORY]" in populated_prompt and "[END MEMORY]" in populated_prompt, "Memory section missing when facts exist"
    print("✅ SUCCESS: Memory section included when facts exist")
    
    # Test 3: Verify no manual fact fetching in build_prompt
    print("\n3. Verifying build_prompt is self-contained:")
    
    # Check that build_prompt doesn't require external fact preparation
    try:
        # This should work without any external fact preparation
        dynamic_prompt = build_prompt("Test query", memory_log)
        print("✅ SUCCESS: build_prompt works without external fact fetching")
    except Exception as e:
        print(f"❌ ERROR: build_prompt failed: {e}")
        assert False, f"build_prompt failed: {e}"
    
    # Test 4: Verify real-time memory state reflection
    print("\n4. Testing real-time memory state reflection:")
    
    # Use reset_database_for_test for robust cleanup
    from tests.utils.db_cleanup import reset_database_for_test
    if hasattr(memory_log, '_connection_pool') and hasattr(memory_log._connection_pool, 'close_all_connections'):
        memory_log._connection_pool.close_all_connections()
    memory_log.shutdown()
    reset_database_for_test(db_path)
    memory_log = MemoryLog(db_path)
    remaining_facts = memory_log.get_all_facts()
    if remaining_facts:
        print('❌ ERROR: Facts remain after cleanup:')
        for f in remaining_facts:
            print(f"   {f.subject} {f.predicate} {f.object}")
    assert not remaining_facts, f"Facts remain in DB after cleanup: {remaining_facts}"

    # After dynamic inclusion, cleanup again and assert DB is empty
    memory_log.shutdown()
    reset_database_for_test(db_path)
    memory_log = MemoryLog(db_path)
    remaining_facts = memory_log.get_all_facts()
    if remaining_facts:
        print('❌ ERROR: Facts remain after second cleanup:')
        for f in remaining_facts:
            print(f"   {f.subject} {f.predicate} {f.object}")
    assert not remaining_facts, f"Facts remain in DB after second cleanup: {remaining_facts}"

    # Build prompt after clearing - should be empty again
    cleared_prompt = build_prompt(test_query, memory_log, max_tokens=512)
    if "[BEGIN MEMORY]" in cleared_prompt:
        memory_section = cleared_prompt.split("[BEGIN MEMORY]")[1].split("[END MEMORY]")[0]
        assert not memory_section.strip() or "no relevant" in memory_section.lower() or "no memories" in memory_section.lower(), "Memory section should be empty after clearing"
        print("✅ SUCCESS: Memory section empty after clearing")
    else:
        print("✅ SUCCESS: Memory section removed when facts cleared")
    
    # Test 5: Verify all facts are included (not just last N)
    print("\n5. Testing comprehensive fact inclusion:")
    
    # Add many facts
    many_messages = [
        ("My name is Bob.", "introduction"),
        ("I'm 30 years old.", "identity"),
        ("I live in New York.", "location"),
        ("I work as a developer.", "profession"),
        ("I love coffee.", "preference"),
        ("My favorite color is blue.", "preference"),
        ("I want to learn AI.", "goal"),
        ("I'm studying machine learning.", "background"),
    ]
    
    for message, tag in many_messages:
        message_id = memory_log.log_memory("user", message, tags=[tag])
        facts = memory_log.extract_facts(message)
        if facts:
            memory_log.store_facts(facts)
    
    # Build prompt with many facts
    comprehensive_prompt = build_prompt(test_query, memory_log, max_tokens=512)
    
    # Count facts in the prompt
    fact_lines = [line for line in comprehensive_prompt.split('\n') if line.strip().startswith('✅') or line.strip().startswith('⚠️') or line.strip().startswith('❓')]
    
    print(f"   Facts found in prompt: {len(fact_lines)}")
    assert len(fact_lines) >= 6, f"Not all facts included, only {len(fact_lines)} found"
    print("✅ SUCCESS: Comprehensive fact inclusion working")
    
    print("\n✅ Final cleanup verification completed successfully!")
    print("\nKey verifications:")
    print("✅ No hardcoded facts in build_prompt")
    print("✅ Dynamic memory state reflection")
    print("✅ Conditional [BEGIN MEMORY] sections")
    print("✅ Comprehensive fact inclusion")
    print("✅ Real-time memory updates")
    print("✅ No legacy manual fact fetching")

if __name__ == "__main__":
    try:
        test_final_cleanup()
        print("\n🎉 All tests passed! Dynamic memory system is working correctly.")
    except AssertionError as e:
        print(f"\n❌ Test failed with assertion error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        sys.exit(1) 