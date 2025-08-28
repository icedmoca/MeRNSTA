#!/usr/bin/env python3
"""
Test suite for MeRNSTA sentiment analysis and preference modeling.
Tests dynamic sentiment scoring, trajectory analysis, and volatility detection.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
import numpy as np
from datetime import datetime, timedelta
from storage.memory_utils import (
    get_sentiment_score, 
    get_volatility_score, 
    get_sentiment_trajectory,
)
# Define reference words inline since constants were removed
positive_reference_words = ["love", "like", "adore", "enjoy", "prefer", "good", "great", "excellent", "wonderful", "amazing"]
negative_reference_words = ["hate", "dislike", "loathe", "despise", "abhor", "terrible", "awful", "horrible", "dreadful"]
from storage.memory_log import TripletFact
from storage.memory_utils import compute_decay_weighted_confidence

class TestSentimentAnalysis(unittest.TestCase):
    """Test suite for sentiment analysis functionality"""
    
    def setUp(self):
        """Set up test data"""
        # Create test facts with different timestamps
        base_date = datetime.now()
        
        self.positive_facts = [
            TripletFact(1, "user", "love", "pizza", 1, (base_date - timedelta(days=10)).strftime("%Y-%m-%d %H:%M:%S"), 1),
            TripletFact(2, "user", "like", "pizza", 1, (base_date - timedelta(days=8)).strftime("%Y-%m-%d %H:%M:%S"), 1),
            TripletFact(3, "user", "enjoy", "pizza", 1, (base_date - timedelta(days=5)).strftime("%Y-%m-%d %H:%M:%S"), 1),
            TripletFact(4, "user", "adore", "pizza", 1, (base_date - timedelta(days=2)).strftime("%Y-%m-%d %H:%M:%S"), 1),
        ]
        
        self.negative_facts = [
            TripletFact(5, "user", "hate", "spiders", 1, (base_date - timedelta(days=10)).strftime("%Y-%m-%d %H:%M:%S"), 1),
            TripletFact(6, "user", "despise", "spiders", 1, (base_date - timedelta(days=8)).strftime("%Y-%m-%d %H:%M:%S"), 1),
            TripletFact(7, "user", "loathe", "spiders", 1, (base_date - timedelta(days=5)).strftime("%Y-%m-%d %H:%M:%S"), 1),
            TripletFact(8, "user", "detest", "spiders", 1, (base_date - timedelta(days=2)).strftime("%Y-%m-%d %H:%M:%S"), 1),
        ]
        
        self.volatile_facts = [
            TripletFact(9, "user", "love", "bikes", 1, (base_date - timedelta(days=10)).strftime("%Y-%m-%d %H:%M:%S"), 1),
            TripletFact(10, "user", "hate", "bikes", 1, (base_date - timedelta(days=8)).strftime("%Y-%m-%d %H:%M:%S"), 1),
            TripletFact(11, "user", "like", "bikes", 1, (base_date - timedelta(days=5)).strftime("%Y-%m-%d %H:%M:%S"), 1),
            TripletFact(12, "user", "despise", "bikes", 1, (base_date - timedelta(days=2)).strftime("%Y-%m-%d %H:%M:%S"), 1),
        ]
        
        self.improving_facts = [
            TripletFact(13, "user", "hate", "coffee", 1, (base_date - timedelta(days=10)).strftime("%Y-%m-%d %H:%M:%S"), 1),
            TripletFact(14, "user", "dislike", "coffee", 1, (base_date - timedelta(days=8)).strftime("%Y-%m-%d %H:%M:%S"), 1),
            TripletFact(15, "user", "like", "coffee", 1, (base_date - timedelta(days=5)).strftime("%Y-%m-%d %H:%M:%S"), 1),
            TripletFact(16, "user", "love", "coffee", 1, (base_date - timedelta(days=2)).strftime("%Y-%m-%d %H:%M:%S"), 1),
        ]

    def test_sentiment_score_polarity(self):
        """Test that sentiment scores correctly map polarity"""
        print("\n🧪 Testing sentiment score polarity mapping...")
        
        # Test positive predicates
        for predicate in positive_reference_words[:4]:  # Test first 4
            score = get_sentiment_score(predicate)
            print(f"   {predicate}: {score:.3f}")
            self.assertGreater(score, 0.0, f"Positive predicate '{predicate}' should have positive score")
        
        # Test negative predicates
        for predicate in negative_reference_words[:4]:  # Test first 4
            score = get_sentiment_score(predicate)
            print(f"   {predicate}: {score:.3f}")
            self.assertLess(score, 0.0, f"Negative predicate '{predicate}' should have negative score")
        
        # Test neutral predicate
        neutral_score = get_sentiment_score("is")
        print(f"   is: {neutral_score:.3f}")
        self.assertAlmostEqual(neutral_score, 0.0, delta=0.5, msg="Neutral predicate should have near-zero score")

    def test_sentiment_score_relative_ordering(self):
        """Test that sentiment scores maintain relative ordering"""
        print("\n🧪 Testing sentiment score relative ordering...")
        
        # Test that "love" > "like" > "neutral" > "dislike" > "hate"
        love_score = get_sentiment_score("love")
        like_score = get_sentiment_score("like")
        dislike_score = get_sentiment_score("dislike")
        hate_score = get_sentiment_score("hate")
        
        print(f"   love: {love_score:.3f}")
        print(f"   like: {like_score:.3f}")
        print(f"   dislike: {dislike_score:.3f}")
        print(f"   hate: {hate_score:.3f}")
        
        self.assertGreater(love_score, like_score, "love should score higher than like")
        self.assertGreater(like_score, dislike_score, "like should score higher than dislike")
        self.assertGreater(dislike_score, hate_score, "dislike should score higher than hate")

    def test_volatility_score_detection(self):
        """Test volatility score detection for different patterns"""
        print("\n🧪 Testing volatility score detection...")
        
        # Test stable positive facts
        stable_positive_vol = get_volatility_score(self.positive_facts)
        print(f"   Stable positive facts: {stable_positive_vol:.3f}")
        self.assertLess(stable_positive_vol, 0.2, "Stable positive facts should have low volatility")
        
        # Test stable negative facts
        stable_negative_vol = get_volatility_score(self.negative_facts)
        print(f"   Stable negative facts: {stable_negative_vol:.3f}")
        self.assertLess(stable_negative_vol, 0.2, "Stable negative facts should have low volatility")
        
        # Test volatile facts
        volatile_vol = get_volatility_score(self.volatile_facts)
        print(f"   Volatile facts: {volatile_vol:.3f}")
        self.assertGreater(volatile_vol, 0.5, "Volatile facts should have high volatility")
        
        # Test improving facts
        improving_vol = get_volatility_score(self.improving_facts)
        print(f"   Improving facts: {improving_vol:.3f}")
        self.assertGreater(improving_vol, 0.3, "Improving facts should have moderate volatility")

    def test_sentiment_trajectory_analysis(self):
        """Test sentiment trajectory analysis"""
        print("\n🧪 Testing sentiment trajectory analysis...")
        
        # Test positive trajectory
        positive_traj = get_sentiment_trajectory(self.positive_facts)
        print(f"   Positive trajectory slope: {positive_traj['slope']:.3f}")
        self.assertGreater(positive_traj['slope'], -0.1, "Positive facts should have near-zero or positive slope")
        self.assertGreater(positive_traj['recent_sentiment'], 0.0, "Positive facts should have positive recent sentiment")
        
        # Test negative trajectory
        negative_traj = get_sentiment_trajectory(self.negative_facts)
        print(f"   Negative trajectory slope: {negative_traj['slope']:.3f}")
        self.assertLess(negative_traj['slope'], 0.1, "Negative facts should have near-zero or negative slope")
        self.assertLess(negative_traj['recent_sentiment'], 0.0, "Negative facts should have negative recent sentiment")
        
        # Test improving trajectory
        improving_traj = get_sentiment_trajectory(self.improving_facts)
        print(f"   Improving trajectory slope: {improving_traj['slope']:.3f}")
        self.assertGreater(improving_traj['slope'], 0.1, "Improving facts should have positive slope")
        self.assertGreater(improving_traj['recent_sentiment'], improving_traj['intercept'], "Recent sentiment should be higher than baseline")

    def test_decay_weighted_confidence(self):
        """Test recency-weighted confidence calculation"""
        print("\n🧪 Testing decay-weighted confidence...")
        
        # Test recent fact (should have high confidence)
        recent_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        recent_confidence = compute_decay_weighted_confidence(1.0, recent_timestamp)
        print(f"   Recent fact confidence: {recent_confidence:.3f}")
        self.assertGreater(recent_confidence, 0.95, "Recent fact should maintain high confidence")
        
        # Test old fact (should have decayed confidence)
        old_timestamp = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d %H:%M:%S")
        old_confidence = compute_decay_weighted_confidence(1.0, old_timestamp)
        print(f"   Old fact confidence: {old_confidence:.3f}")
        self.assertLess(old_confidence, 0.8, "Old fact should have decayed confidence")
        
        # Test that recent > old
        self.assertGreater(recent_confidence, old_confidence, "Recent fact should have higher confidence than old fact")

    def test_edge_cases(self):
        """Test edge cases and error handling"""
        print("\n🧪 Testing edge cases...")
        
        # Test empty facts list
        empty_vol = get_volatility_score([])
        self.assertEqual(empty_vol, 0.0, "Empty facts list should return 0 volatility")
        
        # Test single fact
        single_vol = get_volatility_score([self.positive_facts[0]])
        self.assertEqual(single_vol, 0.0, "Single fact should return 0 volatility")
        
        # Test invalid predicate
        invalid_score = get_sentiment_score("")
        self.assertEqual(invalid_score, 0.0, "Empty predicate should return 0 score")
        
        # Test invalid timestamp
        invalid_confidence = compute_decay_weighted_confidence(1.0, "invalid_timestamp")
        self.assertEqual(invalid_confidence, 1.0, "Invalid timestamp should return original confidence")

def run_sentiment_analysis_tests():
    """Run all sentiment analysis tests"""
    print("🧪 Running MeRNSTA Sentiment Analysis Tests")
    print("=" * 60)
    
    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestSentimentAnalysis)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Summary
    print("\n📊 Test Summary")
    print("=" * 60)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.failures:
        print("\n❌ Failures:")
        for test, traceback in result.failures:
            print(f"   {test}: {traceback}")
    
    if result.errors:
        print("\n❌ Errors:")
        for test, traceback in result.errors:
            print(f"   {test}: {traceback}")
    
    if result.wasSuccessful():
        print("\n🎉 All tests passed! Sentiment analysis is working correctly.")
    else:
        print("\n⚠️ Some tests failed. Check the results above.")
    
    return result.wasSuccessful()

if __name__ == "__main__":
    run_sentiment_analysis_tests() 