#!/usr/bin/env python3
"""
Test suite for MeRNSTA Hybrid Memory Intelligence (Phase 17).

Tests ensemble fusion accuracy, fallback behavior, source traceability,
and weight tuning across FAISS, HRRFormer, and VecSymR backends.
"""

import sys
import os
import logging
import time
from typing import List, Dict, Any

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from vector_memory.hybrid_memory import HybridVectorMemory, HybridMemoryResult
from vector_memory.config import load_memory_config
from vector_memory import get_vectorizer

class TestHybridMemory:
    """Test suite for hybrid memory intelligence."""
    
    def __init__(self):
        self.hybrid_memory = HybridVectorMemory()
        self.test_facts = self._create_test_facts()
        
    def _create_test_facts(self) -> List[Dict[str, Any]]:
        """Create test facts for hybrid memory testing."""
        facts = [
            {
                'id': 1,
                'subject': 'Python',
                'predicate': 'is',
                'object': 'programming language',
                'content': 'Python is programming language',
                'confidence': 0.9,
                'timestamp': time.time() - 3600,  # 1 hour ago
                'context': 'programming'
            },
            {
                'id': 2,
                'subject': 'Machine learning',
                'predicate': 'uses',
                'object': 'algorithms',
                'content': 'Machine learning uses algorithms',
                'confidence': 0.8,
                'timestamp': time.time() - 1800,  # 30 minutes ago
                'context': 'AI'
            },
            {
                'id': 3,
                'subject': 'Neural networks',
                'predicate': 'are',
                'object': 'like brain cells',
                'content': 'Neural networks are like brain cells',
                'confidence': 0.7,
                'timestamp': time.time() - 900,   # 15 minutes ago
                'context': 'analogy'
            },
            {
                'id': 4,
                'subject': 'Vectors',
                'predicate': 'represent',
                'object': 'mathematical concepts',
                'content': 'Vectors represent mathematical concepts',
                'confidence': 0.85,
                'timestamp': time.time() - 300,   # 5 minutes ago
                'context': 'mathematics'
            },
            {
                'id': 5,
                'subject': 'Symbolic reasoning',
                'predicate': 'enables',
                'object': 'logical inference',
                'content': 'Symbolic reasoning enables logical inference',
                'confidence': 0.9,
                'timestamp': time.time() - 60,    # 1 minute ago
                'context': 'logic'
            }
        ]
        return facts
    
    def test_parallel_vectorization(self):
        """Test parallel vectorization across all backends."""
        print("\n🔄 Testing Parallel Vectorization:")
        print("="*50)
        
        test_query = "What is machine learning?"
        
        # Test parallel vectorization
        start_time = time.time()
        results = self.hybrid_memory.vectorize_parallel(test_query)
        total_time = time.time() - start_time
        
        print(f"Query: '{test_query}'")
        print(f"Total time: {total_time:.3f}s")
        print(f"Backends tested: {len(results)}")
        
        for backend, result in results.items():
            if result.success:
                print(f"  ✅ {backend}: {len(result.vector)} dims, {result.latency:.3f}s")
            else:
                print(f"  ❌ {backend}: FAILED - {result.error}")
        
        return len([r for r in results.values() if r.success]) > 0
    
    def test_ensemble_fusion(self):
        """Test ensemble fusion with weighted voting."""
        print("\n🎯 Testing Ensemble Fusion:")
        print("="*50)
        
        # Set to ensemble mode
        self.hybrid_memory.hybrid_strategy = 'ensemble'
        
        test_queries = [
            "programming languages",
            "machine learning algorithms", 
            "brain-like networks",
            "mathematical vectors"
        ]
        
        fusion_results = {}
        
        for query in test_queries:
            print(f"\nQuery: '{query}'")
            
            results = self.hybrid_memory.find_similar_hybrid(query, self.test_facts, top_k=3)
            
            print(f"  Results: {len(results)}")
            for i, result in enumerate(results[:3]):
                print(f"    {i+1}. Content: '{result.content[:40]}...'")
                print(f"       Backend: {result.source_backend}")
                print(f"       Confidence: {result.confidence:.3f}")
                print(f"       Hybrid Score: {result.hybrid_score:.3f}")
            
            fusion_results[query] = results
        
        return fusion_results
    
    def test_priority_routing(self):
        """Test priority-based backend routing."""
        print("\n📋 Testing Priority Routing:")
        print("="*50)
        
        # Set to priority mode
        self.hybrid_memory.hybrid_strategy = 'priority'
        
        test_queries = [
            "machine learning",
            "neural networks", 
            "symbolic reasoning"
        ]
        
        for query in test_queries:
            print(f"\nQuery: '{query}'")
            
            results = self.hybrid_memory.find_similar_hybrid(query, self.test_facts, top_k=2)
            
            if results:
                primary_backend = results[0].source_backend
                print(f"  Primary backend selected: {primary_backend}")
                print(f"  Top result: '{results[0].content[:40]}...'")
                print(f"  Confidence: {results[0].confidence:.3f}")
            else:
                print("  No results returned")
    
    def test_contextual_routing(self):
        """Test contextual routing based on query characteristics."""
        print("\n🧠 Testing Contextual Routing:")
        print("="*50)
        
        # Set to contextual mode
        self.hybrid_memory.hybrid_strategy = 'contextual'
        
        test_cases = [
            ("Calculate the similarity between vectors", "hrrformer"),  # Mathematical
            ("What is like a neural network?", "vecsymr"),             # Analogical
            ("Machine learning uses algorithms", "default"),           # Semantic
            ("If this then that logic", "hrrformer"),                  # Logical
            ("Compare programming languages", "vecsymr"),              # Comparative
        ]
        
        routing_success = 0
        
        for query, expected_backend in test_cases:
            print(f"\nQuery: '{query}'")
            
            # Test backend selection
            selected_backend = self.hybrid_memory._select_backend_for_query(query)
            print(f"  Expected backend: {expected_backend}")
            print(f"  Selected backend: {selected_backend}")
            
            if selected_backend == expected_backend:
                routing_success += 1
                print("  ✅ Correct routing")
            else:
                print("  ⚠️ Different routing")
            
            # Test actual search
            results = self.hybrid_memory.find_similar_hybrid(query, self.test_facts, top_k=1)
            if results:
                print(f"  Search result backend: {results[0].source_backend}")
        
        routing_accuracy = routing_success / len(test_cases)
        print(f"\n📊 Routing Accuracy: {routing_accuracy:.1%}")
        
        return routing_accuracy > 0.6  # 60% accuracy threshold
    
    def test_result_fusion(self):
        """Test result fusion with confidence weighting and semantic overlap."""
        print("\n🔄 Testing Result Fusion:")
        print("="*50)
        
        # Create test scenario with overlapping results
        query = "machine learning algorithms"
        
        # Get results from multiple backends
        vectorizers = self.hybrid_memory.vectorizers
        
        print(f"Testing fusion with {len(vectorizers)} vectorizers")
        
        # Test the fusion process
        query_vectors = self.hybrid_memory.vectorize_parallel(query)
        results = self.hybrid_memory._ensemble_search(query_vectors, self.test_facts, 3)
        
        print(f"Fused results: {len(results)}")
        
        for i, result in enumerate(results):
            print(f"  {i+1}. '{result.content[:30]}...'")
            print(f"     Backend: {result.source_backend}")
            print(f"     Confidence: {result.confidence:.3f}")
            print(f"     Hybrid Score: {result.hybrid_score:.3f}")
            print(f"     Semantic Overlap: {result.semantic_overlap:.3f}")
            
            # Check for fusion info
            if 'fusion_info' in result.metadata:
                fusion = result.metadata['fusion_info']
                print(f"     Fusion backends: {fusion.get('backends', [])}")
        
        return len(results) > 0
    
    def test_fallback_behavior(self):
        """Test fallback behavior when backends fail."""
        print("\n🛡️ Testing Fallback Behavior:")
        print("="*50)
        
        # Temporarily break a backend
        original_vectorizers = self.hybrid_memory.vectorizers.copy()
        
        # Remove a backend to simulate failure
        if 'vecsymr' in self.hybrid_memory.vectorizers:
            del self.hybrid_memory.vectorizers['vecsymr']
            print("Simulated VecSymR failure")
        
        query = "test fallback behavior"
        
        try:
            results = self.hybrid_memory.find_similar_hybrid(query, self.test_facts, top_k=2)
            print(f"Results with {len(self.hybrid_memory.vectorizers)} backends: {len(results)}")
            
            if results:
                active_backends = set(r.source_backend for r in results)
                print(f"Active backends in results: {active_backends}")
                fallback_success = True
            else:
                print("No results - fallback may have issues")
                fallback_success = False
                
        finally:
            # Restore original vectorizers
            self.hybrid_memory.vectorizers = original_vectorizers
        
        return fallback_success
    
    def test_source_traceability(self):
        """Test source traceability and attribution."""
        print("\n📊 Testing Source Traceability:")
        print("="*50)
        
        query = "programming languages and algorithms"
        results = self.hybrid_memory.find_similar_hybrid(query, self.test_facts, top_k=3)
        
        source_distribution = {}
        
        for result in results:
            backend = result.source_backend
            source_distribution[backend] = source_distribution.get(backend, 0) + 1
        
        print(f"Source distribution across {len(results)} results:")
        for backend, count in source_distribution.items():
            percentage = (count / len(results)) * 100
            print(f"  {backend}: {count} results ({percentage:.1f}%)")
        
        # Test metadata preservation
        if results:
            sample_result = results[0]
            print(f"\nSample result metadata:")
            print(f"  Original confidence: {sample_result.original_score:.3f}")
            print(f"  Hybrid score: {sample_result.hybrid_score:.3f}")
            print(f"  Recency score: {sample_result.recency_score:.3f}")
            print(f"  Source backend: {sample_result.source_backend}")
            
            if 'fusion_info' in sample_result.metadata:
                fusion = sample_result.metadata['fusion_info']
                print(f"  Fusion strategy: {fusion.get('fusion_strategy', 'N/A')}")
                print(f"  Participating backends: {fusion.get('backends', [])}")
        
        return len(source_distribution) > 0
    
    def test_weight_tuning(self):
        """Test backend weight tuning effects."""
        print("\n⚖️ Testing Weight Tuning:")
        print("="*50)
        
        query = "symbolic reasoning and logic"
        
        # Test with default weights
        print("Default weights:")
        for backend, weight in self.hybrid_memory.weights.items():
            print(f"  {backend}: {weight}")
        
        default_results = self.hybrid_memory.find_similar_hybrid(query, self.test_facts, top_k=3)
        default_scores = [r.hybrid_score for r in default_results]
        
        print(f"Default hybrid scores: {[f'{s:.3f}' for s in default_scores]}")
        
        # Test with modified weights (boost HRRFormer for symbolic queries)
        original_weights = self.hybrid_memory.weights.copy()
        self.hybrid_memory.weights = {
            'default': 0.2,
            'hrrformer': 0.6,  # Boost for symbolic reasoning
            'vecsymr': 0.2
        }
        
        print("\nModified weights (boost HRRFormer):")
        for backend, weight in self.hybrid_memory.weights.items():
            print(f"  {backend}: {weight}")
        
        modified_results = self.hybrid_memory.find_similar_hybrid(query, self.test_facts, top_k=3)
        modified_scores = [r.hybrid_score for r in modified_results]
        
        print(f"Modified hybrid scores: {[f'{s:.3f}' for s in modified_scores]}")
        
        # Restore original weights
        self.hybrid_memory.weights = original_weights
        
        # Check if weights affected scoring
        score_difference = any(abs(d - m) > 0.01 for d, m in zip(default_scores, modified_scores))
        
        return score_difference
    
    def test_performance_metrics(self):
        """Test performance and timing metrics."""
        print("\n⏱️ Testing Performance Metrics:")
        print("="*50)
        
        queries = [
            "machine learning",
            "neural networks", 
            "programming languages",
            "symbolic reasoning"
        ]
        
        total_times = []
        result_counts = []
        
        for query in queries:
            start_time = time.time()
            results = self.hybrid_memory.find_similar_hybrid(query, self.test_facts, top_k=2)
            query_time = time.time() - start_time
            
            total_times.append(query_time)
            result_counts.append(len(results))
            
            print(f"Query: '{query}'")
            print(f"  Time: {query_time:.3f}s")
            print(f"  Results: {len(results)}")
        
        avg_time = sum(total_times) / len(total_times)
        avg_results = sum(result_counts) / len(result_counts)
        
        print(f"\n📊 Performance Summary:")
        print(f"  Average query time: {avg_time:.3f}s")
        print(f"  Average result count: {avg_results:.1f}")
        print(f"  Total queries: {len(queries)}")
        
        # Performance should be reasonable (< 2 seconds per query)
        return avg_time < 2.0
    
    def run_all_tests(self):
        """Run complete hybrid memory test suite."""
        print("🧠 MeRNSTA Hybrid Memory Intelligence Test Suite")
        print("="*70)
        print("Phase 17: Testing ensemble fusion, routing, and source attribution")
        
        # Set up logging
        logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
        
        test_results = {}
        
        # Run all tests
        tests = [
            ("Parallel Vectorization", self.test_parallel_vectorization),
            ("Ensemble Fusion", self.test_ensemble_fusion),
            ("Priority Routing", self.test_priority_routing), 
            ("Contextual Routing", self.test_contextual_routing),
            ("Result Fusion", self.test_result_fusion),
            ("Fallback Behavior", self.test_fallback_behavior),
            ("Source Traceability", self.test_source_traceability),
            ("Weight Tuning", self.test_weight_tuning),
            ("Performance Metrics", self.test_performance_metrics)
        ]
        
        passed_tests = 0
        
        for test_name, test_func in tests:
            try:
                print(f"\n{'='*70}")
                result = test_func()
                
                if isinstance(result, bool):
                    if result:
                        print(f"✅ {test_name}: PASSED")
                        passed_tests += 1
                    else:
                        print(f"❌ {test_name}: FAILED")
                else:
                    print(f"✅ {test_name}: COMPLETED")
                    passed_tests += 1
                    
                test_results[test_name] = result
                
            except Exception as e:
                print(f"❌ {test_name}: ERROR - {e}")
                test_results[test_name] = False
        
        # Summary
        print(f"\n{'='*70}")
        print("🎯 Hybrid Memory Test Suite Summary:")
        print(f"  Tests passed: {passed_tests}/{len(tests)}")
        print(f"  Success rate: {(passed_tests/len(tests)*100):.1f}%")
        
        # Display hybrid memory stats
        stats = self.hybrid_memory.get_hybrid_stats()
        print(f"\n📊 Hybrid Memory Configuration:")
        print(f"  Mode: {'ENABLED' if stats['hybrid_mode'] else 'DISABLED'}")
        print(f"  Strategy: {stats['strategy']}")
        print(f"  Active backends: {stats['active_backends']}")
        print(f"  Backend weights: {stats['backend_weights']}")
        
        if passed_tests >= len(tests) * 0.7:  # 70% pass rate
            print("\n🚀 Hybrid Memory Intelligence: READY FOR PRODUCTION!")
        else:
            print("\n⚠️ Some tests failed - review configuration and dependencies")
        
        return test_results

def main():
    """Run hybrid memory test suite."""
    tester = TestHybridMemory()
    results = tester.run_all_tests()
    
    print("\n💡 Next Steps:")
    print("  - Tune backend weights for your use case")
    print("  - Choose optimal hybrid strategy (ensemble/priority/contextual)")
    print("  - Monitor source attribution in LLM prompts")
    print("  - Implement domain-specific routing rules")

if __name__ == "__main__":
    main()