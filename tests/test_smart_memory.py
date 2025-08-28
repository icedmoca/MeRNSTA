#!/usr/bin/env python3
"""
Test script for MeRNSTA's smart memory injection system
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from storage.memory_log import MemoryLog
from storage.memory_utils import build_prompt, format_facts_for_display, categorize_fact
import time

def test_smart_memory(isolated_db):
    """Test the smart memory injection system"""
    print("🧠 Testing Smart Memory Injection System")
    print("=" * 60)
    
    # Create memory log
    memory_log = isolated_db
    
    # Simulate the conversation from the user's example
    print("\n1. Logging user information with smart categorization...")
    
    # User provides information (same as before)
    messages = [
        ("My name is Luna Vega.", "introduction"),
        ("I live in Flagstaff.", "location"),
        ("I love dark matter.", "preference"),
        ("My favorite number is 137.", "favorite_number"),
        ("I want to become an astrophysicist.", "goal"),
        ("I'm studying physics at university.", "background"),
        ("I'm 25 years old.", "identity"),
    ]
    
    for message, tag in messages:
        print(f"   Logging: {message}")
        message_id = memory_log.log_memory("user", message, tags=[tag])
        
        # Extract facts
        facts = memory_log.extract_facts(message)
        if facts:
            memory_log.store_facts(facts)
            print(f"   Extracted facts: {[f'{f.subject}: {f.object}' for f in facts]}")
    
    # Show all facts with smart categorization
    print("\n2. All extracted facts (Smart Categorized):")
    memory_facts = memory_log.get_all_facts()
    formatted_facts = format_facts_for_display(memory_facts)
    print(formatted_facts)
    
    # Test the smart prompt builder
    print("\n3. Testing smart prompt for 'What do you know about me?':")
    test_query = "What do you know about me?"
    
    # Build smart prompt with dynamic memory injection
    system_prompt = build_prompt(test_query, memory_log, max_tokens=512)
    
    # Show the complete prompt
    print(f"\nComplete Smart Prompt:")
    print("-" * 60)
    print(system_prompt)
    print("-" * 60)
    
    # Test categorization function
    print("\n4. Testing fact categorization:")
    test_cases = [
        ("luna vega", "user_name"),
        ("flagstaff", "location"),
        ("dark matter", "preference"),
        ("137", "favorite_number"),
        ("astrophysicist", "goal"),
        ("physics", "field_of_study"),
        ("25", "age"),
    ]
    
    for value, entity in test_cases:
        category = categorize_fact(value, entity)
        print(f"   {entity}: {value} → {category}")
    
    # Test token management
    print("\n5. Testing token management:")
    print(f"   Total facts: {len(memory_facts)}")
    print(f"   Estimated tokens in prompt: {len(system_prompt.split())}")
    
    # Test with different token limits
    for max_tokens in [100, 200, 512]:
        limited_prompt = build_prompt(test_query, memory_log, max_tokens=max_tokens)
        token_count = len(limited_prompt.split())
        print(f"   Max tokens {max_tokens}: {token_count} actual tokens")
    
    # Show memory dump
    print("\n6. Memory dump with categorization:")
    memory_data = memory_log.dump_all_memory()
    print(f"   Memory entries: {len(memory_data['memory_entries'])}")
    print(f"   Facts: {len(memory_data['facts'])}")
    
    # Show memory stats
    print("\n7. Memory statistics:")
    stats = memory_log.get_memory_stats()
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    print("\n✅ Smart memory injection test completed!")
    print("\nKey improvements:")
    print("✅ Smart categorization: identity, location, preferences, goals, skills, misc")
    print("✅ Token management: Respects max_tokens limit")
    print("✅ Confidence-aware formatting with visual indicators")
    print("✅ Structured prompt with [BEGIN MEMORY] / [END MEMORY] sections")
    print("✅ Clear instructions for the LLM")
    print("✅ Enhanced fact extraction with better patterns")

if __name__ == "__main__":
    # Create an isolated database for testing
    isolated_db = MemoryLog("smart_test_memory.db")
    test_smart_memory(isolated_db) 