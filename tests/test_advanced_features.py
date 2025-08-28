#!/usr/bin/env python3
"""
Test script for MeRNSTA v0.6.0 Advanced Memory Features
Demonstrates: Episodic Memory, Active Forgetting, Personality Engine
"""

import sys
import os
import time
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from storage import db
from storage.memory_log import MemoryLog
from config.settings import MEMORY_ROUTING_MODES, DEFAULT_MEMORY_MODE, PERSONALITY_PROFILES, DEFAULT_PERSONALITY

def test_episodic_memory(isolated_db):
    """Test episodic memory chaining functionality"""
    print("🧪 Testing Episodic Memory...")
    
    # Use isolated_db fixture
    memory_log = isolated_db
    
    # Add facts to create episodes
    test_facts = [
        ("user", "likes", "pizza"),
        ("user", "lives in", "New York"),
        ("user", "works as", "developer"),
    ]
    
    # Store facts (should create episode 1)
    for subject, predicate, object_val in test_facts:
        triplets = [(subject, predicate, object_val)]
        memory_log.store_triplets(triplets, 1)
    
    # Simulate time gap by waiting
    print("   Waiting 2 seconds to simulate time gap...")
    time.sleep(2)
    
    # Add more facts (should create episode 2)
    more_facts = [
        ("user", "favorite color", "blue"),
        ("user", "has", "2 cats"),
        ("user", "studies", "computer science"),
    ]
    
    for subject, predicate, object_val in more_facts:
        triplets = [(subject, predicate, object_val)]
        memory_log.store_triplets(triplets, 2)
    
    # List episodes
    episodes = memory_log.list_episodes()
    print(f"✅ Created {len(episodes)} episodes")
    assert len(episodes) > 0, "No episodes were created"
    
    for episode in episodes:
        print(f"   Episode {episode['id']}: {episode['summary']} ({episode['fact_count']} facts)")
    
    # Show episode details
    if episodes:
        episode_data = memory_log.show_episode(episodes[0]['id'])
        if episode_data:
            print(f"   Episode {episode_data['episode']['id']} has {len(episode_data['facts'])} facts")
            assert episode_data['episode']['id'] == episodes[0]['id'], "Episode ID mismatch"

def test_active_forgetting(isolated_db):
    """Test active forgetting functionality"""
    print("\n🧪 Testing Active Forgetting...")
    
    memory_log = isolated_db
    
    # Add facts with varying confidence levels
    test_facts = [
        ("user", "likes", "coffee"),
        ("user", "dislikes", "tea"),
        ("user", "works at", "Google"),
        ("user", "studied", "engineering"),
        ("user", "has", "a dog"),
    ]
    
    for subject, predicate, object_val in test_facts:
        triplets = [(subject, predicate, object_val)]
        memory_log.store_triplets(triplets, 1)
    
    # Test pruning memory
    print("   Testing memory pruning...")
    result = memory_log.prune_memory(confidence_threshold=0.5)
    print(f"   Pruned {result['pruned_facts']} facts out of {result['total_facts']} total facts")
    assert isinstance(result, dict), "Prune memory should return a dictionary"
    assert 'pruned_facts' in result, "Result should contain pruned_facts key"
    
    # Test forgetting a subject
    print("   Testing subject forgetting...")
    result = memory_log.forget_subject("user")
    print(f"   Forgot {result['deleted_count']} facts about 'user'")
    assert isinstance(result, dict), "Forget subject should return a dictionary"
    assert 'deleted_count' in result, "Result should contain deleted_count key"

def test_personality_engine(isolated_db):
    """Test personality-based memory biasing"""
    print("\n🧪 Testing Personality Engine...")
    
    memory_log = isolated_db
    
    # Add facts for testing
    test_facts = [
        ("you", "are", "my friend"),
        ("friend", "likes", "pizza"),
        ("user", "hates", "mushrooms"),
        ("user", "loves", "chocolate"),
    ]
    
    for subject, predicate, object_val in test_facts:
        triplets = [(subject, predicate, object_val)]
        memory_log.store_triplets(triplets, 1)
    
    # Test different personalities
    personalities = ["neutral", "loyal", "skeptical", "emotional", "analytical"]
    
    for personality in personalities:
        print(f"   Testing {personality} personality...")
        facts = memory_log.get_facts_with_personality_decay(personality=personality)
        assert isinstance(facts, list), f"get_facts_with_personality_decay should return a list for {personality}"
        
        # Show confidence scores for each personality
        for fact in facts[:2]:  # Show first 2 facts
            conf = getattr(fact, 'decayed_confidence', 1.0)
            print(f"     {fact.subject} {fact.predicate} {fact.object}: confidence {conf:.3f}")
            assert 0.0 <= conf <= 1.0, f"Confidence should be between 0 and 1, got {conf}"

def test_memory_routing_modes(isolated_db):
    """Test different memory routing modes with personality"""
    print("\n🧪 Testing Memory Routing Modes with Personality...")
    
    memory_log = isolated_db
    
    # Add some test facts
    test_facts = [
        ("user", "name", "Alice"),
        ("user", "age", "30"),
        ("user", "profession", "engineer"),
        ("user", "location", "San Francisco"),
        ("user", "hobby", "photography")
    ]
    
    for subject, predicate, object_val in test_facts:
        triplets = [(subject, predicate, object_val)]
        memory_log.store_triplets(triplets, 1)
    
    # Test each routing mode with different personalities
    from storage.memory_utils import build_prompt
    
    for mode in ["MAC", "MAG", "MEL"]:
        for personality in ["neutral", "loyal"]:
            print(f"\n📝 Testing {mode} mode with {personality} personality:")
            prompt = build_prompt("Tell me about yourself", memory_log, memory_mode=mode, personality=personality)
            assert isinstance(prompt, str), f"build_prompt should return a string for {mode} mode"
            assert len(prompt) > 0, f"Prompt should not be empty for {mode} mode"
            
            # Show first 200 characters of prompt
            preview = prompt[:200] + "..." if len(prompt) > 200 else prompt
            print(f"   Prompt preview: {preview}")
            
            # Count facts in prompt
            fact_count = prompt.count("confidence:")
            print(f"   Facts included: {fact_count}")

def test_advanced_features(isolated_db):
    """Test the new polish features"""
    print("\n🧪 Testing Advanced Features...")
    
    memory_log = isolated_db
    
    # Add diverse facts for testing
    diverse_facts = [
        ("user", "likes", "coffee"),
        ("user", "dislikes", "tea"),
        ("user", "likes", "tea"),  # Contradiction
        ("user", "works at", "Google"),
        ("user", "works at", "Microsoft"),  # Contradiction
        ("user", "studied", "computer science"),
        ("user", "speaks", "English"),
        ("user", "speaks", "Spanish"),
        ("user", "has", "a dog"),
        ("user", "has", "a cat")
    ]
    
    for subject, predicate, object_val in diverse_facts:
        triplets = [(subject, predicate, object_val)]
        memory_log.store_triplets(triplets, 1)
    
    # Test memory leaders
    print("🏆 Memory Leaders:")
    leaders = memory_log.get_memory_leaders(top_n=5)
    assert isinstance(leaders, list), "get_memory_leaders should return a list"
    for i, (subject, count, avg_conf) in enumerate(leaders, 1):
        print(f"   {i}. {subject}: {count} facts (avg confidence: {avg_conf:.2f})")
        assert isinstance(subject, str), "Subject should be a string"
        assert isinstance(count, int), "Count should be an integer"
        assert isinstance(avg_conf, (int, float)), "Average confidence should be a number"
    
    # Test conflict highlighting
    print("\n⚠️ Conflicts:")
    conflicts = memory_log.highlight_conflicts()
    assert isinstance(conflicts, dict), "highlight_conflicts should return a dictionary"
    for subject, conflict_data in conflicts.items():
        print(f"   🔥 {subject}: {conflict_data['fact_count']} conflicting facts")
        assert 'fact_count' in conflict_data, "Conflict data should contain fact_count"
    
    # Test memory export
    print("\n📤 Memory Export:")
    success = memory_log.export_memory_summary("test_export_v2.jsonl")
    if success:
        print("   ✅ Export successful")
    assert isinstance(success, bool), "export_memory_summary should return a boolean"
    
    # Test episode management
    print("\n📚 Episodes:")
    episodes = memory_log.list_episodes()
    assert isinstance(episodes, list), "list_episodes should return a list"
    for episode in episodes:
        print(f"   Episode {episode['id']}: {episode['fact_count']} facts")
        assert 'id' in episode, "Episode should have an id"
        assert 'fact_count' in episode, "Episode should have a fact_count"

def main():
    """Run all tests"""
    print("🧠 MeRNSTA v0.6.0 Advanced Features Test Suite")
    print("=" * 60)
    
    try:
        # Test episodic memory
        test_episodic_memory()
        
        # Test active forgetting
        test_active_forgetting()
        
        # Test personality engine
        test_personality_engine()
        
        # Test memory routing modes with personality
        test_memory_routing_modes()
        
        # Test advanced features
        test_advanced_features()
        
        print("\n✅ All tests completed successfully!")
        print("\n🎉 New Features Summary:")
        print("   • Episodic Memory: Group facts by conversation sessions")
        print("   • Active Forgetting: Prune weak facts and forget subjects")
        print("   • Personality Engine: Adjust memory behavior based on personality")
        print("   • Memory Routing: MAC (context), MAG (generation), MEL (everything)")
        print("   • Volatility Index: Facts that flip often decay faster")
        print("   • Similarity Reinforcement: Similar facts reinforce instead of duplicate")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 