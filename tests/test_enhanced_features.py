#!/usr/bin/env python3
"""
Comprehensive tests for enhanced contradiction pipeline features
"""

import sys
import os
import unittest
import numpy as np

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from storage.memory_utils import (
    TripletFact, calculate_contradiction_score, detect_contradictions,
    calculate_agreement_score, _are_synonyms, _are_antonyms,
    get_contradiction_summary_by_clusters, get_embedding_cache_stats,
    clear_embedding_cache, get_cached_embedding, group_contradictions_by_subject,
    analyze_contradiction_clusters
)

class TestEnhancedFeatures(unittest.TestCase):
    """Test suite for enhanced contradiction pipeline features"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Clear cache before each test
        clear_embedding_cache()
        
        # Create test facts
        self.test_facts = [
            TripletFact(1, "you", "love", "spiders", 1, "2024-01-01", 1),
            TripletFact(2, "you", "hate", "spiders", 1, "2024-01-01", 1),
            TripletFact(3, "you", "like", "pizza", 1, "2024-01-01", 1),
            TripletFact(4, "you", "dislike", "pizza", 1, "2024-01-01", 1),
            TripletFact(5, "you", "love", "cats", 1, "2024-01-01", 1),
            TripletFact(6, "you", "hate", "dogs", 1, "2024-01-01", 1),
            TripletFact(7, "you", "adore", "spiders", 1, "2024-01-01", 1),  # Synonym of love
        ]
    
    def tearDown(self):
        """Clean up after each test"""
        clear_embedding_cache()
    
    def test_calculate_agreement_score(self):
        """Test agreement score calculation"""
        print("\n🧪 Testing calculate_agreement_score")
        
        # Test perfect agreement
        fact1 = TripletFact(1, "you", "love", "pizza", 1, "2024-01-01", 1)
        fact2 = TripletFact(2, "you", "love", "pizza", 1, "2024-01-01", 1)
        score = calculate_agreement_score(fact1, fact2)
        self.assertAlmostEqual(score, 1.0, places=2, msg="Perfect agreement should score 1.0")
        
        # Test synonym predicates
        fact1 = TripletFact(3, "you", "love", "cats", 1, "2024-01-01", 1)
        fact2 = TripletFact(4, "you", "adore", "cats", 1, "2024-01-01", 1)
        score = calculate_agreement_score(fact1, fact2)
        self.assertGreaterEqual(score, 0.5, msg="Synonym predicates should have moderate agreement")
        
        # Test different subjects (no agreement)
        fact1 = TripletFact(5, "you", "love", "pizza", 1, "2024-01-01", 1)
        fact2 = TripletFact(6, "me", "love", "pizza", 1, "2024-01-01", 1)
        score = calculate_agreement_score(fact1, fact2)
        self.assertAlmostEqual(score, 0.0, places=2, msg="Different subjects should have no agreement")
        
        # Test contradiction (should have low agreement)
        fact1 = TripletFact(7, "you", "love", "spiders", 1, "2024-01-01", 1)
        fact2 = TripletFact(8, "you", "hate", "spiders", 1, "2024-01-01", 1)
        score = calculate_agreement_score(fact1, fact2)
        self.assertLess(score, 0.5, msg="Contradictory facts should have low agreement")
        
        print("✅ calculate_agreement_score tests passed")
    
    def test_get_cached_embedding(self):
        """Test embedding cache functionality"""
        print("\n🧪 Testing get_cached_embedding")
        
        # Test initial cache state
        initial_stats = get_embedding_cache_stats()
        self.assertEqual(initial_stats['cache_size'], 0, "Cache should be empty initially")
        self.assertEqual(initial_stats['cache_hits'], 0, "No cache hits initially")
        self.assertEqual(initial_stats['cache_misses'], 0, "No cache misses initially")
        
        # Test first embedding (cache miss)
        text1 = "pizza"
        embedding1 = get_cached_embedding(text1)
        self.assertIsInstance(embedding1, np.ndarray, "Should return numpy array")
        self.assertEqual(len(embedding1), 384, "Should return 384-dimensional vector")
        
        # Test second embedding (cache miss)
        text2 = "spiders"
        embedding2 = get_cached_embedding(text2)
        self.assertIsInstance(embedding2, np.ndarray, "Should return numpy array")
        self.assertEqual(len(embedding2), 384, "Should return 384-dimensional vector")
        
        # Test repeated embedding (cache hit)
        embedding1_again = get_cached_embedding(text1)
        np.testing.assert_array_equal(embedding1, embedding1_again, "Cached embedding should be identical")
        
        # Check cache statistics
        stats = get_embedding_cache_stats()
        self.assertEqual(stats['cache_size'], 2, "Cache should have 2 entries")
        self.assertEqual(stats['cache_hits'], 1, "Should have 1 cache hit")
        self.assertEqual(stats['cache_misses'], 2, "Should have 2 cache misses")
        self.assertGreater(stats['hit_rate'], 0.0, "Hit rate should be positive")
        
        # Test empty text handling
        empty_embedding = get_cached_embedding("")
        self.assertIsInstance(empty_embedding, np.ndarray, "Empty text should return numpy array")
        self.assertTrue(np.all(empty_embedding == 0), "Empty text should return zero vector")
        
        print("✅ get_cached_embedding tests passed")
    
    def test_contradiction_clusters(self):
        """Test contradiction clustering functionality"""
        print("\n🧪 Testing contradiction_clusters")
        
        # Detect contradictions
        contradictions = detect_contradictions(self.test_facts)
        self.assertGreater(len(contradictions), 0, "Should detect some contradictions")
        
        # Test grouping by subject
        grouped = group_contradictions_by_subject(contradictions)
        self.assertIn("you", grouped, "Should group contradictions by subject")
        self.assertGreater(len(grouped["you"]), 0, "Should have contradictions for 'you' subject")
        
        # Test cluster analysis
        analysis = analyze_contradiction_clusters(contradictions)
        self.assertIn("you", analysis, "Should analyze contradictions for 'you' subject")
        
        cluster_data = analysis["you"]
        self.assertIn("contradiction_count", cluster_data, "Should have contradiction count")
        self.assertIn("max_score", cluster_data, "Should have max score")
        self.assertIn("severity", cluster_data, "Should have severity level")
        self.assertIn("patterns", cluster_data, "Should have patterns")
        
        # Test severity levels
        self.assertIn(cluster_data["severity"], ["CRITICAL", "HIGH", "MEDIUM", "LOW"], 
                     "Severity should be one of the expected levels")
        
        # Test summary generation
        summary = get_contradiction_summary_by_clusters(contradictions)
        self.assertIsInstance(summary, str, "Should return string summary")
        self.assertIn("CONTRADICTION CLUSTER ANALYSIS", summary, "Should contain analysis header")
        self.assertIn("you", summary, "Should mention the subject")
        
        # Test with no contradictions
        no_contradictions = []
        empty_summary = get_contradiction_summary_by_clusters(no_contradictions)
        self.assertIn("No contradictions detected", empty_summary, 
                     "Should handle empty contradiction list")
        
        print("✅ contradiction_clusters tests passed")
    
    def test_synonym_detection(self):
        """Test synonym detection functionality"""
        print("\n🧪 Testing synonym detection")
        
        # Test known synonyms
        self.assertTrue(_are_synonyms("love", "adore"), "love and adore should be synonyms")
        self.assertTrue(_are_synonyms("hate", "dislike"), "hate and dislike should be synonyms")
        self.assertTrue(_are_synonyms("good", "excellent"), "good and excellent should be synonyms")
        
        # Test antonyms (should not be synonyms)
        self.assertFalse(_are_synonyms("love", "hate"), "love and hate should not be synonyms")
        self.assertFalse(_are_synonyms("good", "bad"), "good and bad should not be synonyms")
        
        # Test unrelated words
        self.assertFalse(_are_synonyms("pizza", "spiders"), "pizza and spiders should not be synonyms")
        
        # Test case insensitivity
        self.assertTrue(_are_synonyms("LOVE", "adore"), "Should be case insensitive")
        self.assertTrue(_are_synonyms("love", "ADORE"), "Should be case insensitive")
        
        print("✅ synonym detection tests passed")
    
    def test_antonym_detection(self):
        """Test antonym detection functionality"""
        print("\n🧪 Testing antonym detection")
        
        # Test known antonyms
        self.assertTrue(_are_antonyms("love", "hate"), "love and hate should be antonyms")
        self.assertTrue(_are_antonyms("like", "dislike"), "like and dislike should be antonyms")
        self.assertTrue(_are_antonyms("good", "bad"), "good and bad should be antonyms")
        
        # Test synonyms (should not be antonyms)
        self.assertFalse(_are_antonyms("love", "adore"), "love and adore should not be antonyms")
        self.assertFalse(_are_antonyms("good", "excellent"), "good and excellent should not be antonyms")
        
        # Test unrelated words
        self.assertFalse(_are_antonyms("pizza", "spiders"), "pizza and spiders should not be antonyms")
        
        print("✅ antonym detection tests passed")
    
    def test_cache_management(self):
        """Test cache management functionality"""
        print("\n🧪 Testing cache management")
        
        # Test cache clearing
        get_cached_embedding("test")
        stats_before = get_embedding_cache_stats()
        self.assertGreater(stats_before['cache_size'], 0, "Cache should have entries")
        
        clear_embedding_cache()
        stats_after = get_embedding_cache_stats()
        self.assertEqual(stats_after['cache_size'], 0, "Cache should be empty after clearing")
        self.assertEqual(stats_after['cache_hits'], 0, "Cache hits should be reset")
        self.assertEqual(stats_after['cache_misses'], 0, "Cache misses should be reset")
        
        print("✅ cache management tests passed")
    
    def test_enhanced_contradiction_detection(self):
        """Test enhanced contradiction detection with caching"""
        print("\n🧪 Testing enhanced contradiction detection")
        
        # Clear cache for clean test
        clear_embedding_cache()
        
        # Test high contradiction cases
        fact1 = TripletFact(1, "you", "love", "spiders", 1, "2024-01-01", 1)
        fact2 = TripletFact(2, "you", "hate", "spiders", 1, "2024-01-01", 1)
        score = calculate_contradiction_score(fact1, fact2)
        self.assertGreaterEqual(score, 0.5, "Love vs hate should have moderate contradiction score")
        
        # Test no contradiction cases
        fact1 = TripletFact(3, "you", "love", "pizza", 1, "2024-01-01", 1)
        fact2 = TripletFact(4, "you", "hate", "pasta", 1, "2024-01-01", 1)
        score = calculate_contradiction_score(fact1, fact2)
        self.assertLess(score, 0.5, "Different subjects should have low contradiction score")
        
        # Check cache performance
        cache_stats = get_embedding_cache_stats()
        self.assertGreater(cache_stats['cache_size'], 0, "Should have cached embeddings")
        
        print("✅ enhanced contradiction detection tests passed")

def run_performance_test():
    """Run performance test for embedding cache"""
    print("\n🚀 Performance Test: Embedding Cache")
    print("=" * 60)
    
    clear_embedding_cache()
    
    # Test repeated embeddings
    test_texts = ["pizza", "spiders", "love", "hate", "good", "bad"] * 10  # 60 total, 6 unique
    
    import time
    start_time = time.time()
    
    for text in test_texts:
        embedding = get_cached_embedding(text)
    
    end_time = time.time()
    total_time = end_time - start_time
    
    stats = get_embedding_cache_stats()
    
    print(f"Total embeddings requested: {len(test_texts)}")
    print(f"Unique embeddings: {stats['cache_size']}")
    print(f"Cache hits: {stats['cache_hits']}")
    print(f"Cache misses: {stats['cache_misses']}")
    print(f"Hit rate: {stats['hit_rate']:.2%}")
    print(f"Total time: {total_time:.3f} seconds")
    print(f"Average time per embedding: {total_time/len(test_texts):.4f} seconds")
    
    # Performance assertions
    assert stats['cache_hits'] > 0, "Should have cache hits"
    assert stats['hit_rate'] > 0.5, "Hit rate should be high with repeated texts"
    assert total_time < 10.0, "Should complete within reasonable time"
    
    print("✅ Performance test passed")

if __name__ == "__main__":
    # Run unit tests
    print("🧪 Running Enhanced Features Test Suite")
    print("=" * 80)
    
    # Create test suite
    test_suite = unittest.TestLoader().loadTestsFromTestCase(TestEnhancedFeatures)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Run performance test
    try:
        run_performance_test()
    except Exception as e:
        print(f"❌ Performance test failed: {e}")
    
    # Summary
    print(f"\n📊 Test Summary")
    print("=" * 80)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.wasSuccessful():
        print("🎉 All tests passed!")
    else:
        print("⚠️ Some tests failed")
        for test, traceback in result.failures:
            print(f"❌ FAILED: {test}")
        for test, traceback in result.errors:
            print(f"❌ ERROR: {test}")
    
    sys.exit(0 if result.wasSuccessful() else 1) 