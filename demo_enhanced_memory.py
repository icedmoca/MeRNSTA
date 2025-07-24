#!/usr/bin/env python3
"""
Demo script for the enhanced MeRNSTA memory system.
Tests all major features: dynamic extraction, contradiction handling,
volatility tracking, semantic search, and summarization.
"""

import logging
import sys
from datetime import datetime, timedelta

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Import the enhanced memory system
try:
    from storage.enhanced_memory_system import EnhancedMemorySystem
    print("✅ Enhanced memory system loaded successfully!")
except ImportError as e:
    print(f"❌ Failed to load enhanced memory system: {e}")
    sys.exit(1)


def print_separator(title=""):
    """Print a nice separator"""
    if title:
        print(f"\n{'='*20} {title} {'='*20}")
    else:
        print("="*60)


def demo_basic_extraction():
    """Test basic fact extraction with confidence detection"""
    print_separator("BASIC FACT EXTRACTION")
    
    memory = EnhancedMemorySystem(db_path=":memory:")  # In-memory for demo
    
    test_inputs = [
        # Standard facts
        "I love pizza",
        "My favorite color is blue",
        "I have a cat named Whiskers",
        
        # With hedge words (lower confidence)
        "I think I like sushi",
        "Maybe I prefer coffee over tea",
        "I guess my hometown is Boston",
        
        # With intensifiers (higher confidence)
        "I absolutely love hiking",
        "I definitely hate spiders",
        "I always enjoy reading",
        
        # Negations
        "I don't like mushrooms",
        "I am not a morning person",
    ]
    
    for text in test_inputs:
        print(f"\n📝 Input: '{text}'")
        result = memory.process_input(text, user_profile_id="demo_user", session_id="demo_session")
        
        for fact in result["extracted_facts"]:
            conf_bar = "█" * int(fact.confidence * 5)
            hedge = " [hedge]" if fact.hedge_detected else ""
            intense = " [intensifier]" if fact.intensifier_detected else ""
            neg = " [negated]" if fact.negation else ""
            
            print(f"   → {fact.subject} {fact.predicate} {fact.object}")
            print(f"     Confidence: [{conf_bar}] {fact.confidence:.2f}{hedge}{intense}{neg}")


def demo_contradiction_handling():
    """Test contradiction detection and resolution"""
    print_separator("CONTRADICTION HANDLING")
    
    memory = EnhancedMemorySystem(db_path=":memory:")
    user_id = "demo_user"
    session_id = "demo_session"
    
    # Create contradictory statements
    contradictory_inputs = [
        ("I love sushi", "Initial preference"),
        ("I hate sushi", "Changed mind"),
        ("My favorite food is pizza", "First favorite"),
        ("My favorite food is pasta", "Changed favorite"),
        ("I live in New York", "Location 1"),
        ("I live in California", "Location 2"),
    ]
    
    for text, comment in contradictory_inputs:
        print(f"\n📝 Input: '{text}' ({comment})")
        result = memory.process_input(text, user_profile_id=user_id, session_id=session_id)
        
        if result["contradictions"]:
            print(f"   ⚠️ Contradictions detected: {len(result['contradictions'])}")
            
        print(f"   Response: {result['response']}")
    
    # Show contradictions
    print("\n" + "="*60)
    result = memory._handle_command("/show_contradictions", user_id, session_id)
    print(result["response"])


def demo_volatility_tracking():
    """Test volatility tracking with opinion changes"""
    print_separator("VOLATILITY TRACKING")
    
    memory = EnhancedMemorySystem(db_path=":memory:")
    user_id = "volatile_user"
    session_id = "volatile_session"
    
    # Simulate frequent opinion changes
    volatile_statements = [
        "I like coffee",
        "I hate coffee",
        "I love coffee",
        "I don't like coffee anymore",
        "Actually, coffee is okay",
        "I prefer tea over coffee",
        "No wait, coffee is better than tea",
    ]
    
    print("Simulating volatile opinions about coffee...")
    for i, statement in enumerate(volatile_statements):
        print(f"\nDay {i+1}: '{statement}'")
        result = memory.process_input(statement, user_profile_id=user_id, session_id=session_id)
    
    # Generate volatility report
    print("\n" + "="*60)
    result = memory._handle_command("/volatility_report", user_id, session_id)
    print(result["response"])
    
    # Generate meta-goals for clarification
    print("\n" + "="*60)
    result = memory._handle_command("/generate_meta_goals", user_id, session_id)
    print(result["response"])


def demo_semantic_search():
    """Test semantic memory queries"""
    print_separator("SEMANTIC MEMORY SEARCH")
    
    memory = EnhancedMemorySystem(db_path=":memory:")
    user_id = "search_user"
    session_id = "search_session"
    
    # Store various facts
    facts_to_store = [
        "I work as a software engineer",
        "My office is in downtown Seattle",
        "I enjoy programming in Python",
        "I have been coding for 10 years",
        "My favorite framework is Django",
        "I also know JavaScript and React",
        "I graduated from MIT",
        "I studied computer science",
        "I have a dog named Max",
        "Max is a golden retriever",
        "I take Max to the park every morning",
    ]
    
    print("Storing facts...")
    for fact in facts_to_store:
        memory.process_input(fact, user_profile_id=user_id, session_id=session_id)
    
    # Test various queries
    queries = [
        "What do I do for work?",
        "Where is my office?",
        "What programming languages do I know?",
        "Tell me about my pet",
        "What did I study in college?",
        "Do I like coding?",
    ]
    
    print("\n\nTesting semantic queries:")
    for query in queries:
        print(f"\n❓ Query: '{query}'")
        result = memory.process_input(query, user_profile_id=user_id, session_id=session_id)
        print(f"   💡 Response: {result['response']}")


def demo_summarization():
    """Test fact summarization"""
    print_separator("SUMMARIZATION")
    
    memory = EnhancedMemorySystem(db_path=":memory:")
    user_id = "summary_user"
    session_id = "summary_session"
    
    # Store a mix of facts
    diverse_facts = [
        # Preferences
        "I love Italian food",
        "I enjoy hiking on weekends",
        "I prefer mountains over beaches",
        "I like reading science fiction",
        
        # Personal info
        "I am 30 years old",
        "I have two siblings",
        "My birthday is in July",
        
        # Contradictory/volatile
        "I love running",
        "Actually, I hate running",
        "Well, running is okay sometimes",
        
        # With varying confidence
        "I definitely want to visit Japan",
        "I think I might enjoy skydiving",
        "Maybe I should learn Spanish",
    ]
    
    print("Storing diverse facts...")
    for fact in diverse_facts:
        memory.process_input(fact, user_profile_id=user_id, session_id=session_id)
        
    # Get summary
    print("\n" + "="*60)
    result = memory._handle_command("/summarize", user_id, session_id)
    print(result["response"])
    
    # List facts
    print("\n" + "="*60)
    result = memory._handle_command("/list_facts", user_id, session_id)
    print(result["response"])


def main():
    """Run all demos"""
    print("\n🚀 ENHANCED MeRNSTA MEMORY SYSTEM DEMO")
    print("="*60)
    
    demos = [
        ("Basic Extraction", demo_basic_extraction),
        ("Contradiction Handling", demo_contradiction_handling),
        ("Volatility Tracking", demo_volatility_tracking),
        ("Semantic Search", demo_semantic_search),
        ("Summarization", demo_summarization),
    ]
    
    for name, demo_func in demos:
        try:
            demo_func()
        except Exception as e:
            print(f"\n❌ Error in {name} demo: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n✅ Demo completed!")
    print("="*60)


if __name__ == "__main__":
    main() 
