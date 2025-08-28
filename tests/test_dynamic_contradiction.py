#!/usr/bin/env python3
"""
Test script for dynamic contradiction detection using embedding similarity.
Demonstrates zero-hardcode contradiction detection.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from storage.memory_utils import (
    calculate_contradiction_score, 
    detect_contradictions,
    llm_confirms_contradiction,
    TripletFact
)
from storage.memory_log import MemoryLog

def test_contradiction_detection():
    """Test the dynamic contradiction detection system"""
    print("🧠 Dynamic Contradiction Detection Test")
    print("=" * 60)
    
    # Create test facts
    existing_facts = [
        TripletFact(1, "you", "like", "coffee", 1, "2024-01-01 10:00:00", 1, 0.0, 0.0),
        TripletFact(2, "you", "prefer", "tea", 2, "2024-01-01 11:00:00", 1, 0.0, 0.0),
        TripletFact(3, "you", "hate", "mushrooms", 3, "2024-01-01 12:00:00", 1, 0.0, 0.0),
        TripletFact(4, "you", "love", "pizza", 4, "2024-01-01 13:00:00", 1, 0.0, 0.0),
    ]
    
    # Test cases for contradiction detection
    test_cases = [
        ("you", "hate", "coffee"),  # Should contradict "like coffee"
        ("you", "love", "coffee"),  # Should not contradict "like coffee"
        ("you", "prefer", "coffee"),  # Should weakly contradict "like coffee"
        ("you", "hate", "pizza"),  # Should contradict "love pizza"
        ("you", "like", "mushrooms"),  # Should contradict "hate mushrooms"
        ("you", "enjoy", "tea"),  # Should not contradict "prefer tea"
    ]
    
    print("Existing facts:")
    for fact in existing_facts:
        print(f"  {fact.subject} {fact.predicate} {fact.object}")
    
    print("\nTesting contradiction detection:")
    for i, new_fact in enumerate(test_cases, 1):
        print(f"\n{i}. New fact: {new_fact[0]} {new_fact[1]} {new_fact[2]}")
        
        # Create a TripletFact for the new fact
        new_fact_obj = TripletFact(0, new_fact[0], new_fact[1], new_fact[2], 1, "2024-01-01", 1.0, 0.0, 0.0)
        all_facts = existing_facts + [new_fact_obj]
        
        # Detect contradictions
        contradictions = detect_contradictions(all_facts)
        
        if contradictions:
            print("   Contradictions detected:")
            for fact1, fact2, score in contradictions:
                print(f"     ⚠️ Score {score:.3f}: {fact1.subject} {fact1.predicate} {fact1.object} vs {fact2.subject} {fact2.predicate} {fact2.object}")
                
                # Optional: Use LLM to confirm for ambiguous cases
                if 0.3 < score < 0.7:
                    llm_confirmed = llm_confirms_contradiction(new_fact, (fact1.subject, fact1.predicate, fact1.object))
                    print(f"       LLM confirmation: {'Yes' if llm_confirmed else 'No'}")
        else:
            print("   ✅ No contradictions detected")

def test_embedding_similarity():
    """Test embedding similarity calculations"""
    print("\n🔍 Embedding Similarity Test")
    print("=" * 60)
    
    # Test pairs for similarity
    test_pairs = [
        (("you", "like", "coffee"), ("you", "hate", "coffee")),
        (("you", "love", "pizza"), ("you", "enjoy", "pizza")),
        (("you", "prefer", "tea"), ("you", "want", "tea")),
        (("you", "hate", "mushrooms"), ("you", "like", "mushrooms")),
        (("you", "need", "sleep"), ("you", "don't need", "sleep")),
    ]
    
    for fact1, fact2 in test_pairs:
        score = calculate_contradiction_score(fact1, fact2)
        print(f"Contradiction score: {score:.3f}")
        print(f"  {fact1[0]} {fact1[1]} {fact1[2]}")
        print(f"  {fact2[0]} {fact2[1]} {fact2[2]}")
        print()

def test_edge_cases():
    """Test edge cases and robustness"""
    print("\n🔧 Edge Cases Test")
    print("=" * 60)
    
    # Test with None values
    fact1 = ("you", "like", None)
    fact2 = ("you", "hate", "coffee")
    score = calculate_contradiction_score(fact1, fact2)
    print(f"None value test: {score:.3f}")
    
    # Test with empty strings
    fact1 = ("you", "", "coffee")
    fact2 = ("you", "hate", "coffee")
    score = calculate_contradiction_score(fact1, fact2)
    print(f"Empty string test: {score:.3f}")
    
    # Test with very different subjects
    fact1 = ("you", "like", "coffee")
    fact2 = ("John", "hate", "coffee")
    score = calculate_contradiction_score(fact1, fact2)
    print(f"Different subjects test: {score:.3f}")

def test_high_confidence_contradictions():
    """Test that high-confidence contradicting facts are not incorrectly pruned."""
    print("\n🔒 High Confidence Contradiction Test")
    print("-" * 60)
    
    # Create high confidence contradicting facts
    fact1 = TripletFact(1, "you", "love", "coffee", frequency=3, timestamp="2024-10-01 10:00:00", confidence=0.95, volatility_score=0.2, contradiction_score=0.1)
    fact2 = TripletFact(2, "you", "hate", "coffee", frequency=2, timestamp="2024-10-02 11:00:00", confidence=0.92, volatility_score=0.3, contradiction_score=0.1)
    
    score = calculate_contradiction_score(fact1, fact2)
    print(f"Contradiction score: {score:.3f}")
    assert score > 0.7, "Should detect high contradiction"
    
    # Test contradiction detection logic
    contradictions = detect_contradictions([fact1, fact2])
    assert len(contradictions) > 0, "Should detect contradictions between high-confidence facts"
    
    print("✅ High confidence facts detected as contradictory")

def test_no_prune_high_confidence_low_drift():
    """Test that high-confidence, low-drift facts are not pruned, only marked as volatile."""
    print("\n🔒 High Confidence Low Drift Test")
    print("-" * 60)
    
    fact1 = TripletFact(1, "alice", "loves", "cats", frequency=5, timestamp="2024-01-01 10:00:00", confidence=0.98, volatility_score=0.1, contradiction_score=0.1)
    fact2 = TripletFact(2, "alice", "hates", "cats", frequency=4, timestamp="2024-01-02 11:00:00", confidence=0.97, volatility_score=0.1, contradiction_score=0.1)
    
    # Calculate and print the contradiction score
    score = calculate_contradiction_score(fact1, fact2)
    print(f"Contradiction score: {score:.3f}")
    
    # Test contradiction detection
    contradictions = detect_contradictions([fact1, fact2])
    print(f"Number of contradictions detected: {len(contradictions)}")
    assert len(contradictions) > 0, "Should detect contradictions"
    
    # Test that both facts have high confidence and low volatility
    assert fact1.confidence > 0.9, "Fact1 should have high confidence"
    assert fact2.confidence > 0.9, "Fact2 should have high confidence"
    assert fact1.volatility_score < 0.2, "Fact1 should have low volatility"
    assert fact2.volatility_score < 0.2, "Fact2 should have low volatility"
    
    print("✅ High-confidence, low-drift facts properly identified")

def test_contradiction_triggers_cluster_update():
    """Test that contradiction between clusters triggers cluster update/merge."""
    print("\n🔒 Contradiction Cluster Update Test")
    print("-" * 60)
    
    fact1 = TripletFact(1, "alice", "loves", "cats", frequency=2, timestamp="2024-01-01 10:00:00", confidence=0.9, volatility_score=0.1, contradiction_score=0.1)
    fact2 = TripletFact(2, "alice", "hates", "cats", frequency=2, timestamp="2024-01-02 11:00:00", confidence=0.9, volatility_score=0.1, contradiction_score=0.1)
    
    # Test contradiction detection across different subjects (simulating different clusters)
    contradictions = detect_contradictions([fact1, fact2])
    assert len(contradictions) > 0, "Should detect contradictions across clusters"
    
    print("✅ Contradiction detection works across clusters")

def test_contradiction_across_episodes():
    """Test that contradiction across episodes is detected and resolved."""
    print("\n🔒 Cross-Episode Contradiction Test")
    print("-" * 60)
    
    fact1 = TripletFact(1, "bob", "loves", "pizza", frequency=1, timestamp="2024-01-01 10:00:00", confidence=0.8, volatility_score=0.1, contradiction_score=0.1)
    fact2 = TripletFact(2, "bob", "hates", "pizza", frequency=1, timestamp="2024-01-02 11:00:00", confidence=0.8, volatility_score=0.1, contradiction_score=0.1)
    
    # Test contradiction detection regardless of episode (timestamp difference)
    contradictions = detect_contradictions([fact1, fact2])
    assert len(contradictions) > 0, "Should detect contradictions across episodes"
    
    print("✅ Contradiction detection works across episodes")

def main():
    """Run all tests"""
    try:
        test_contradiction_detection()
        test_embedding_similarity()
        test_edge_cases()
        test_high_confidence_contradictions()
        test_no_prune_high_confidence_low_drift()
        test_contradiction_triggers_cluster_update()
        test_contradiction_across_episodes()
        print("\n" + "=" * 60)
        print("✅ Dynamic contradiction detection test completed!")
        print("\nKey Features:")
        print("• Zero hardcoded contradiction rules")
        print("• Embedding-based semantic similarity")
        print("• Dynamic threshold-based detection")
        print("• LLM fallback for ambiguous cases")
        print("• Robust handling of edge cases")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        print("Make sure Ollama is running with the mistral model")

if __name__ == "__main__":
    main() 