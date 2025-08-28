#!/usr/bin/env python3
"""
Test script for the new rank_facts function that implements weighted recency + confidence ranking.
"""

import sys
import os
from datetime import datetime, timedelta
from storage.memory_utils import rank_facts, TripletFact

def test_rank_facts():
    """Test the rank_facts function with sample data."""
    
    # Create test facts with different timestamps and confidence levels
    now = datetime.now()
    
    # Fact 1: High confidence, old (should rank lower)
    fact1 = TripletFact(
        id=1,
        subject="user",
        predicate="likes",
        object="pizza",
        frequency=1,
        timestamp=(now - timedelta(days=30)).isoformat(),
        confidence=0.9
    )
    
    # Fact 2: Medium confidence, recent (should rank higher due to recency)
    fact2 = TripletFact(
        id=2,
        subject="user",
        predicate="likes",
        object="sushi",
        frequency=1,
        timestamp=(now - timedelta(days=1)).isoformat(),
        confidence=0.6
    )
    
    # Fact 3: Low confidence, very recent (should rank highest due to recency)
    fact3 = TripletFact(
        id=3,
        subject="user",
        predicate="likes",
        object="tacos",
        frequency=1,
        timestamp=now.isoformat(),
        confidence=0.3
    )
    
    # Fact 4: High confidence, recent (should rank highest overall)
    fact4 = TripletFact(
        id=4,
        subject="user",
        predicate="likes",
        object="burgers",
        frequency=1,
        timestamp=(now - timedelta(hours=6)).isoformat(),
        confidence=0.8
    )
    
    # Test facts list
    test_facts = [fact1, fact2, fact3, fact4]
    
    print("Original order (by ID):")
    for i, fact in enumerate(test_facts):
        print(f"{i+1}. {fact.subject} {fact.predicate} {fact.object} (conf: {fact.confidence:.1f}, time: {fact.timestamp})")
    
    print("\nRanked order (by weighted recency + confidence):")
    ranked_facts = rank_facts(test_facts)
    
    for i, fact in enumerate(ranked_facts):
        print(f"{i+1}. {fact.subject} {fact.predicate} {fact.object} (conf: {fact.confidence:.1f}, time: {fact.timestamp})")
    
    # Verify that the ranking makes sense
    print("\nVerification:")
    
    # The most recent high-confidence fact should be first
    if ranked_facts[0].id == 4:  # fact4 (high conf, recent)
        print("✅ Most recent high-confidence fact ranked first")
    else:
        print("❌ Most recent high-confidence fact not ranked first")
    
    # The oldest fact should not be first
    if ranked_facts[0].id != 1:  # fact1 (old)
        print("✅ Oldest fact not ranked first")
    else:
        print("❌ Oldest fact incorrectly ranked first")
    
    # The very recent low-confidence fact should rank higher than the old high-confidence fact
    fact3_rank = next(i for i, f in enumerate(ranked_facts) if f.id == 3)
    fact1_rank = next(i for i, f in enumerate(ranked_facts) if f.id == 1)
    
    if fact3_rank < fact1_rank:
        print("✅ Recent low-confidence fact ranked higher than old high-confidence fact")
    else:
        print("❌ Recent low-confidence fact not ranked higher than old high-confidence fact")
    
    print(f"\nRanking summary:")
    print(f"- Fact 4 (high conf, recent): rank {next(i for i, f in enumerate(ranked_facts) if f.id == 4) + 1}")
    print(f"- Fact 3 (low conf, very recent): rank {fact3_rank + 1}")
    print(f"- Fact 2 (medium conf, recent): rank {next(i for i, f in enumerate(ranked_facts) if f.id == 2) + 1}")
    print(f"- Fact 1 (high conf, old): rank {fact1_rank + 1}")

if __name__ == "__main__":
    test_rank_facts() 