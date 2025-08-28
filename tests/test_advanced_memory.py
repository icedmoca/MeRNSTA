#!/usr/bin/env python3
"""
Test script for advanced memory features:
- Subject reference normalization
- Conflict resolution
- Volatility tracking
"""

import sqlite3
import time
from storage.memory_log import MemoryLog
from storage.memory_utils import normalize_subject_references, detect_contradictions, calculate_volatility_score

def test_subject_normalization():
    """Test subject reference normalization"""
    print("🧪 Testing Subject Reference Normalization")
    print("=" * 50)
    
    test_cases = [
        "I like pizza",
        "My favorite color is blue",
        "I'm going to the store",
        "I've been working on this project",
        "Myself and my friends went hiking"
    ]
    
    for text in test_cases:
        normalized = normalize_subject_references(text)
        print(f"Original: {text}")
        print(f"Normalized: {normalized}")
        print()

def test_conflict_detection():
    """Test conflict detection between facts"""
    print("🧪 Testing Conflict Detection")
    print("=" * 50)
    
    # Create test facts
    from storage.memory_utils import TripletFact
    
    existing_facts = [
        TripletFact(1, "you", "like", "pizza", 1, "2024-01-01 10:00:00", 1, 0.0, 0.0),
        TripletFact(2, "you", "hate", "spaghetti", 2, "2024-01-01 11:00:00", 1, 0.0, 0.0),
    ]
    
    # Add a new fact that contradicts the hate fact
    new_fact = TripletFact(3, "you", "like", "spaghetti", 3, "2024-01-01 12:00:00", 1, 0.0, 0.0)
    
    # Combine all facts and detect contradictions
    all_facts = existing_facts + [new_fact]
    contradictions = detect_contradictions(all_facts)
    
    print(f"New fact: {new_fact.subject} {new_fact.predicate} {new_fact.object}")
    print("Existing facts:")
    for fact in existing_facts:
        print(f"  {fact.subject} {fact.predicate} {fact.object}")
    
    print(f"\nDetected contradictions: {len(contradictions)}")
    for fact1, fact2, score in contradictions:
        print(f"  Contradiction score {score:.2f}: {fact1.subject} {fact1.predicate} {fact1.object} vs {fact2.subject} {fact2.predicate} {fact2.object}")

def test_volatility_tracking():
    """Test volatility score calculation"""
    print("🧪 Testing Volatility Tracking")
    print("=" * 50)
    
    # Simulate fact history with frequent changes
    fact_history = [
        {"timestamp": "2024-01-01 10:00:00", "value": "pizza"},
        {"timestamp": "2024-01-01 12:00:00", "value": "sushi"},
        {"timestamp": "2024-01-01 14:00:00", "value": "pizza"},
        {"timestamp": "2024-01-01 16:00:00", "value": "burgers"},
    ]
    
    volatility = calculate_volatility_score(fact_history)
    print(f"Fact history with {len(fact_history)} changes:")
    for record in fact_history:
        print(f"  {record['timestamp']}: {record['value']}")
    
    print(f"\nVolatility score: {volatility:.3f}")
    if volatility > 0.5:
        print("🔥 High volatility detected!")
    elif volatility > 0.2:
        print("⚡ Medium volatility detected")
    else:
        print("✅ Stable fact")

def test_integrated_memory_system(isolated_db):
    """Test the integrated memory system with all features"""
    print("🧪 Testing Integrated Memory System")
    print("=" * 50)
    
    memory_log = isolated_db
    
    # Test messages that should trigger subject normalization and conflicts
    test_messages = [
        "I like pizza",
        "My favorite color is blue", 
        "I hate pizza",  # Contradicts first message
        "I love pizza",  # Contradicts hate message
        "My favorite color is red",  # Contradicts blue
    ]
    
    print("Processing test messages:")
    for i, message in enumerate(test_messages, 1):
        print(f"\n{i}. User: {message}")
        
        # Log the message
        message_id = memory_log.log_memory("user", message)
        
        # Extract and store triplets
        triplets = memory_log.extract_triplets(message)
        if triplets:
            # Convert 4-tuples to 3-tuples for store_triplets
            clean_triplets = []
            for triplet in triplets:
                if len(triplet) == 4:
                    # Extract just the subject, predicate, object (skip confidence)
                    clean_triplets.append((triplet[0], triplet[1], triplet[2]))
                else:
                    clean_triplets.append(triplet)
            
            print(f"   Extracted triplets: {clean_triplets}")
            memory_log.store_triplets(clean_triplets, message_id)
        else:
            print("   No triplets extracted")
        
        # Show current facts
        facts = memory_log.get_all_facts()
        if facts:
            print("   Current facts:")
            for fact in facts:
                print(f"     {fact.subject} {fact.predicate} {fact.object} (freq: {fact.frequency}, contradiction: {fact.contradiction_score:.2f}, volatility: {fact.volatility_score:.2f})")
    
    # Clean up
    import os
    if os.path.exists(isolated_db.db_path):
        os.remove(isolated_db.db_path)
    print(f"\n✅ Test completed. Database cleaned up.")

if __name__ == "__main__":
    test_subject_normalization()
    print("\n" + "="*60 + "\n")
    
    test_conflict_detection()
    print("\n" + "="*60 + "\n")
    
    test_volatility_tracking()
    print("\n" + "="*60 + "\n")
    
    # Create a test database
    test_db_path = "test_advanced_memory.db"
    isolated_db = MemoryLog(test_db_path)
    test_integrated_memory_system(isolated_db) 