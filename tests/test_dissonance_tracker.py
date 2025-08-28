#!/usr/bin/env python3
"""
Test suite for DissonanceTracker (Phase 26: Cognitive Dissonance Modeling)

Tests contradiction scoring, volatility tracking, integration with memory and reflection,
and overall dissonance system functionality.
"""

import json
import os
import sys
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Add the project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.dissonance_tracker import DissonanceTracker, DissonanceRegion, DissonanceEvent


class TestDissonanceTracker(unittest.TestCase):
    """Test suite for the DissonanceTracker class"""
    
    def setUp(self):
        """Set up test environment before each test"""
        # Create temporary directory for test storage
        self.temp_dir = tempfile.mkdtemp()
        self.storage_path = Path(self.temp_dir) / "test_dissonance.jsonl"
        
        # Mock configuration
        self.mock_config = {
            'dissonance_tracking': {
                'contradiction_threshold': 0.6,
                'pressure_threshold': 0.7,
                'urgency_threshold': 0.8,
                'volatility_threshold': 0.5,
                'resolution_timeout_hours': 24,
                'scoring_weights': {
                    'frequency': 0.3,
                    'semantic_distance': 0.25,
                    'causality': 0.2,
                    'duration': 0.25
                }
            }
        }
        
        # Create DissonanceTracker with mocked config and storage path
        with patch('agents.dissonance_tracker.get_config', return_value=self.mock_config):
            with patch.object(DissonanceTracker, '_load_persistent_state'):
                self.tracker = DissonanceTracker()
                self.tracker.storage_path = self.storage_path
                # Mock the reflection trigger to prevent database issues
                self.tracker._trigger_dissonance_reflection = Mock()
    
    def tearDown(self):
        """Clean up test environment after each test"""
        # Clean up temporary files
        if self.storage_path.exists():
            self.storage_path.unlink()
        os.rmdir(self.temp_dir)
    
    def test_initialization(self):
        """Test DissonanceTracker initialization"""
        self.assertEqual(self.tracker.name, "dissonance_tracker")
        self.assertEqual(self.tracker.contradiction_threshold, 0.6)
        self.assertEqual(self.tracker.pressure_threshold, 0.7)
        self.assertEqual(self.tracker.urgency_threshold, 0.8)
        self.assertEqual(len(self.tracker.dissonance_regions), 0)
        self.assertEqual(len(self.tracker.event_history), 0)
    
    def test_contradiction_processing_basic(self):
        """Test basic contradiction processing"""
        contradiction_data = {
            'belief_id': 'test_belief_1',
            'source_belief': 'I like coffee',
            'target_belief': 'I hate coffee',
            'semantic_distance': 0.8,
            'confidence': 0.9
        }
        
        result = self.tracker.process_contradiction(contradiction_data)
        
        self.assertEqual(result, 'test_belief_1')
        self.assertIn('test_belief_1', self.tracker.dissonance_regions)
        
        region = self.tracker.dissonance_regions['test_belief_1']
        self.assertEqual(region.belief_id, 'test_belief_1')
        self.assertEqual(region.contradiction_frequency, 1.0)
        self.assertEqual(region.semantic_distance, 0.8)
        self.assertIn('I like coffee', region.conflict_sources)
    
    def test_contradiction_processing_low_confidence(self):
        """Test that low-confidence contradictions are ignored"""
        contradiction_data = {
            'belief_id': 'test_belief_2',
            'source_belief': 'I like tea',
            'target_belief': 'I prefer coffee',
            'semantic_distance': 0.3,
            'confidence': 0.4  # Below threshold of 0.6
        }
        
        result = self.tracker.process_contradiction(contradiction_data)
        
        self.assertIsNone(result)
        self.assertNotIn('test_belief_2', self.tracker.dissonance_regions)
    
    def test_semantic_clustering(self):
        """Test semantic clustering functionality"""
        # Test common words clustering
        cluster1 = self.tracker._determine_semantic_cluster("I like coffee", "I prefer coffee")
        self.assertIn("coffee", cluster1)
        
        # Test topic-based clustering
        cluster2 = self.tracker._determine_semantic_cluster("I like pizza", "I hate burgers")
        self.assertEqual(cluster2, "preferences")
        
        # Test fallback clustering - this case should find "random" as common word
        cluster3 = self.tracker._determine_semantic_cluster("Random statement", "Another random thing")
        self.assertEqual(cluster3, "random")
        
        # Test true fallback clustering with no common words
        cluster4 = self.tracker._determine_semantic_cluster("Unique statement", "Different concept")
        self.assertEqual(cluster4, "general")
    
    def test_causality_strength_calculation(self):
        """Test causality strength calculation"""
        # Test with causal words
        strength1 = self.tracker._calculate_causality_strength(
            "I hate coffee because it makes me jittery",
            "I avoid caffeine"
        )
        self.assertGreater(strength1, 0.2)  # Should detect "because"
        
        # Test with shared concepts
        strength2 = self.tracker._calculate_causality_strength(
            "I love chocolate cake",
            "Chocolate is my favorite"
        )
        self.assertGreater(strength2, 0.0)  # Should detect "chocolate"
        
        # Test with no relation
        strength3 = self.tracker._calculate_causality_strength(
            "I like dogs",
            "Mathematics is complex"
        )
        self.assertLessEqual(strength3, 0.1)
    
    def test_pressure_vector_updates(self):
        """Test pressure vector calculations"""
        # Create a test region
        region = DissonanceRegion(
            belief_id='test_belief',
            semantic_cluster='preferences',
            conflict_sources=['source1', 'source2'],
            contradiction_frequency=3.0,
            semantic_distance=0.7,
            causality_strength=0.5,
            duration=12.0,  # 12 hours
            pressure_score=0.0,
            urgency=0.0,
            confidence_erosion=0.0,
            emotional_volatility=0.0,
            last_updated=datetime.now()
        )
        
        self.tracker._update_pressure_vectors(region)
        
        # Check that pressure vectors were calculated
        self.assertGreater(region.pressure_score, 0.0)
        self.assertGreater(region.urgency, 0.0)
        self.assertGreater(region.confidence_erosion, 0.0)
        self.assertLessEqual(region.pressure_score, 1.0)
        self.assertLessEqual(region.urgency, 1.0)
    
    def test_dissonance_report_generation(self):
        """Test comprehensive dissonance report generation"""
        # Add some test regions
        self.tracker.dissonance_regions['test1'] = DissonanceRegion(
            belief_id='test1',
            semantic_cluster='preferences',
            conflict_sources=['source1'],
            contradiction_frequency=2.0,
            semantic_distance=0.8,
            causality_strength=0.6,
            duration=5.0,
            pressure_score=0.75,
            urgency=0.6,
            confidence_erosion=0.4,
            emotional_volatility=0.3,
            last_updated=datetime.now()
        )
        
        self.tracker.dissonance_regions['test2'] = DissonanceRegion(
            belief_id='test2',
            semantic_cluster='abilities',
            conflict_sources=['source2'],
            contradiction_frequency=1.0,
            semantic_distance=0.5,
            causality_strength=0.3,
            duration=2.0,
            pressure_score=0.4,
            urgency=0.2,
            confidence_erosion=0.1,
            emotional_volatility=0.1,
            last_updated=datetime.now()
        )
        
        report = self.tracker.get_dissonance_report()
        
        # Check report structure
        self.assertIn('summary', report)
        self.assertIn('top_stress_regions', report)
        self.assertIn('recent_events', report)
        
        # Check summary data
        summary = report['summary']
        self.assertEqual(summary['total_regions'], 2)
        self.assertGreater(summary['total_pressure'], 1.0)
        self.assertGreater(summary['average_pressure'], 0.5)
        
        # Check top stress regions (should be sorted by pressure)
        top_regions = report['top_stress_regions']
        self.assertEqual(len(top_regions), 2)
        self.assertEqual(top_regions[0]['belief_id'], 'test1')  # Higher pressure first
        self.assertEqual(top_regions[1]['belief_id'], 'test2')
    
    def test_dissonance_resolution(self):
        """Test dissonance resolution functionality"""
        # Create a high-pressure region
        self.tracker.dissonance_regions['high_pressure'] = DissonanceRegion(
            belief_id='high_pressure',
            semantic_cluster='preferences',
            conflict_sources=['source1'],
            contradiction_frequency=5.0,
            semantic_distance=0.9,
            causality_strength=0.8,
            duration=20.0,
            pressure_score=0.85,
            urgency=0.9,
            confidence_erosion=0.7,
            emotional_volatility=0.6,
            last_updated=datetime.now()
        )
        
        # Mock reflection orchestrator by setting it directly
        mock_orchestrator = Mock()
        self.tracker._reflection_orchestrator = mock_orchestrator
        mock_result = Mock()
        mock_result.request_id = 'test_reflection_123'
        mock_orchestrator.trigger_reflection.return_value = mock_result
        
        result = self.tracker.resolve_dissonance('high_pressure')
        
        # Check resolution results
        self.assertEqual(result['status'], 'attempted')
        self.assertEqual(result['belief_id'], 'high_pressure')
        self.assertLess(result['pressure_after'], result['pressure_before'])
        self.assertIn('results', result)
    
    def test_dissonance_history(self):
        """Test dissonance history tracking"""
        # Add some events to history
        now = datetime.now()
        
        event1 = DissonanceEvent(
            timestamp=now - timedelta(minutes=30),
            belief_id='test_belief',
            event_type='contradiction_detected',
            source_belief='I like X',
            target_belief='I hate X',
            intensity=0.8
        )
        
        event2 = DissonanceEvent(
            timestamp=now - timedelta(minutes=15),
            belief_id='test_belief',
            event_type='pressure_increase',
            source_belief='system',
            target_belief='test_belief',
            intensity=0.6
        )
        
        self.tracker.event_history.extend([event1, event2])
        
        # Get history for last hour
        history = self.tracker.get_dissonance_history(hours=1)
        
        # Check history structure
        self.assertIn('total_events', history)
        self.assertIn('event_counts', history)
        self.assertIn('pressure_trend', history)
        self.assertIn('events', history)
        
        # Check event data
        self.assertEqual(history['total_events'], 2)
        self.assertIn('contradiction_detected', history['event_counts'])
        self.assertIn('pressure_increase', history['event_counts'])
        
        # Check events are properly serialized
        events = history['events']
        self.assertEqual(len(events), 2)
        self.assertIn('timestamp', events[0])
        self.assertIn('belief_id', events[0])
    
    def test_pressure_trend_calculation(self):
        """Test pressure trend analysis"""
        # Add some regions for trend calculation
        self.tracker.dissonance_regions['trend_test'] = DissonanceRegion(
            belief_id='trend_test',
            semantic_cluster='test',
            conflict_sources=['source'],
            contradiction_frequency=1.0,
            semantic_distance=0.5,
            causality_strength=0.3,
            duration=1.0,
            pressure_score=0.6,
            urgency=0.4,
            confidence_erosion=0.2,
            emotional_volatility=0.1,
            last_updated=datetime.now()
        )
        
        # Add recent contradiction event
        recent_event = DissonanceEvent(
            timestamp=datetime.now() - timedelta(minutes=30),
            belief_id='trend_test',
            event_type='contradiction_detected',
            source_belief='test',
            target_belief='test',
            intensity=0.7
        )
        self.tracker.event_history.append(recent_event)
        
        trend = self.tracker._calculate_pressure_trend(2)  # 2 hours
        
        self.assertIn('current_pressure', trend)
        self.assertIn('estimated_baseline', trend)
        self.assertIn('pressure_change', trend)
        self.assertIn('contradiction_rate', trend)
        
        self.assertGreater(trend['current_pressure'], 0.0)
        self.assertGreater(trend['contradiction_rate'], 0.0)
    
    def test_persistent_storage(self):
        """Test saving and loading persistent state"""
        # Create test region
        region = DissonanceRegion(
            belief_id='persistent_test',
            semantic_cluster='test_cluster',
            conflict_sources=['source1', 'source2'],
            contradiction_frequency=2.0,
            semantic_distance=0.7,
            causality_strength=0.5,
            duration=10.0,
            pressure_score=0.65,
            urgency=0.55,
            confidence_erosion=0.3,
            emotional_volatility=0.2,
            last_updated=datetime.now()
        )
        
        self.tracker.dissonance_regions['persistent_test'] = region
        
        # Save state
        self.tracker._save_persistent_state()
        
        # Verify file was created and contains data
        self.assertTrue(self.storage_path.exists())
        
        with open(self.storage_path, 'r') as f:
            lines = f.readlines()
            self.assertGreater(len(lines), 0)
            
            # Check that region data was saved
            region_found = False
            for line in lines:
                data = json.loads(line.strip())
                if data.get('type') == 'dissonance_region':
                    if data['data']['belief_id'] == 'persistent_test':
                        region_found = True
                        self.assertEqual(data['data']['semantic_cluster'], 'test_cluster')
                        self.assertEqual(data['data']['contradiction_frequency'], 2.0)
                        break
            
            self.assertTrue(region_found, "Region data not found in storage")
    
    def test_old_region_cleanup(self):
        """Test cleanup of old resolved dissonance regions"""
        # Create an old, low-pressure region
        old_time = datetime.now() - timedelta(hours=200)  # 200 hours ago
        
        old_region = DissonanceRegion(
            belief_id='old_region',
            semantic_cluster='test',
            conflict_sources=['source'],
            contradiction_frequency=1.0,
            semantic_distance=0.3,
            causality_strength=0.2,
            duration=200.0,
            pressure_score=0.05,  # Very low pressure
            urgency=0.02,  # Very low urgency
            confidence_erosion=0.01,
            emotional_volatility=0.01,
            last_updated=old_time
        )
        
        # Create a recent region that should not be cleaned up
        recent_region = DissonanceRegion(
            belief_id='recent_region',
            semantic_cluster='test',
            conflict_sources=['source'],
            contradiction_frequency=1.0,
            semantic_distance=0.5,
            causality_strength=0.3,
            duration=1.0,
            pressure_score=0.5,
            urgency=0.4,
            confidence_erosion=0.2,
            emotional_volatility=0.1,
            last_updated=datetime.now()
        )
        
        self.tracker.dissonance_regions['old_region'] = old_region
        self.tracker.dissonance_regions['recent_region'] = recent_region
        
        # Run cleanup
        self.tracker.cleanup_old_regions(max_age_hours=168)  # 1 week
        
        # Check that old region was removed but recent region remains
        self.assertNotIn('old_region', self.tracker.dissonance_regions)
        self.assertIn('recent_region', self.tracker.dissonance_regions)
    
    def test_integration_hooks(self):
        """Test integration with memory system and reflection orchestrator"""
        # Test memory system integration
        mock_memory = Mock()
        self.tracker._memory_system = mock_memory
        result = self.tracker.integrate_with_memory_system()
        self.assertTrue(result)
        
        # Test reflection orchestrator integration
        mock_reflection = Mock()
        self.tracker._reflection_orchestrator = mock_reflection
        mock_result = Mock()
        mock_result.request_id = 'test_123'
        mock_reflection.trigger_reflection.return_value = mock_result
        
        # Create a high-pressure region to trigger reflection
        region = DissonanceRegion(
            belief_id='trigger_test',
            semantic_cluster='test',
            conflict_sources=['source'],
            contradiction_frequency=1.0,
            semantic_distance=0.8,
            causality_strength=0.6,
            duration=1.0,
            pressure_score=0.85,  # Above threshold
            urgency=0.8,
            confidence_erosion=0.5,
            emotional_volatility=0.4,
            last_updated=datetime.now()
        )
        
        # Restore the original method for this test
        from agents.dissonance_tracker import DissonanceTracker
        self.tracker._trigger_dissonance_reflection = DissonanceTracker._trigger_dissonance_reflection.__get__(self.tracker, DissonanceTracker)
        
        # Trigger reflection
        self.tracker._trigger_dissonance_reflection(region)
        
        # Verify reflection was triggered
        mock_reflection.trigger_reflection.assert_called_once()
    
    def test_multiple_contradictions_same_belief(self):
        """Test handling multiple contradictions for the same belief"""
        # Process first contradiction
        contradiction1 = {
            'belief_id': 'multi_test',
            'source_belief': 'I like coffee',
            'target_belief': 'I hate coffee',
            'semantic_distance': 0.8,
            'confidence': 0.9
        }
        
        self.tracker.process_contradiction(contradiction1)
        
        # Process second contradiction for same belief
        contradiction2 = {
            'belief_id': 'multi_test',
            'source_belief': 'Coffee is good',
            'target_belief': 'Coffee is terrible',
            'semantic_distance': 0.7,
            'confidence': 0.85
        }
        
        self.tracker.process_contradiction(contradiction2)
        
        # Check that frequency increased and distance updated
        region = self.tracker.dissonance_regions['multi_test']
        self.assertEqual(region.contradiction_frequency, 2.0)
        self.assertEqual(region.semantic_distance, 0.8)  # Should be max of the two
        self.assertEqual(len(region.conflict_sources), 3)  # Should have source + both targets
        # Should contain the initial source and both target beliefs
        self.assertIn('I like coffee', region.conflict_sources)  # Initial source
        self.assertIn('I hate coffee', region.conflict_sources)  # First target
        self.assertIn('Coffee is terrible', region.conflict_sources)  # Second target


class TestDissonanceTrackerIntegration(unittest.TestCase):
    """Integration tests for DissonanceTracker with other components"""
    
    def setUp(self):
        """Set up integration test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.storage_path = Path(self.temp_dir) / "integration_test.jsonl"
        
        # Mock comprehensive config
        self.mock_config = {
            'dissonance_tracking': {
                'enabled': True,
                'contradiction_threshold': 0.6,
                'pressure_threshold': 0.7,
                'urgency_threshold': 0.8,
                'volatility_threshold': 0.5,
                'resolution_timeout_hours': 24,
                'scoring_weights': {
                    'frequency': 0.3,
                    'semantic_distance': 0.25,
                    'causality': 0.2,
                    'duration': 0.25
                },
                'integration': {
                    'memory_system': True,
                    'reflection_orchestrator': True,
                    'auto_resolution': True,
                    'belief_evolution': False
                }
            }
        }
    
    def tearDown(self):
        """Clean up integration test environment"""
        if self.storage_path.exists():
            self.storage_path.unlink()
        os.rmdir(self.temp_dir)
    
    def test_singleton_instance(self):
        """Test that get_dissonance_tracker returns singleton instance"""
        from agents.dissonance_tracker import get_dissonance_tracker
        
        with patch('agents.dissonance_tracker.get_config', return_value=self.mock_config):
            with patch.object(DissonanceTracker, '_load_persistent_state'):
                tracker1 = get_dissonance_tracker()
                tracker2 = get_dissonance_tracker()
                
                # Should be the same instance
                self.assertIs(tracker1, tracker2)
    
    def test_end_to_end_dissonance_cycle(self):
        """Test complete dissonance cycle from detection to resolution"""
        with patch('agents.dissonance_tracker.get_config', return_value=self.mock_config):
            with patch.object(DissonanceTracker, '_load_persistent_state'):
                tracker = DissonanceTracker()
                tracker.storage_path = self.storage_path
                
                # 1. Process contradiction
                contradiction = {
                    'belief_id': 'e2e_test',
                    'source_belief': 'I love programming',
                    'target_belief': 'I hate programming',
                    'semantic_distance': 0.9,
                    'confidence': 0.95
                }
                
                result = tracker.process_contradiction(contradiction)
                self.assertEqual(result, 'e2e_test')
                
                # 2. Check that region was created
                self.assertIn('e2e_test', tracker.dissonance_regions)
                region = tracker.dissonance_regions['e2e_test']
                self.assertGreater(region.pressure_score, 0.0)
                
                # 3. Generate report
                report = tracker.get_dissonance_report()
                self.assertGreater(report['summary']['total_regions'], 0)
                
                # 4. Get history
                history = tracker.get_dissonance_history()
                self.assertGreater(history['total_events'], 0)
                
                # 5. Attempt resolution
                mock_orchestrator = Mock()
                tracker._reflection_orchestrator = mock_orchestrator
                mock_result = Mock()
                mock_result.request_id = 'test_reflection'
                mock_orchestrator.trigger_reflection.return_value = mock_result
                
                resolution_result = tracker.resolve_dissonance('e2e_test')
                self.assertEqual(resolution_result['status'], 'attempted')


if __name__ == '__main__':
    # Set up test environment
    import tempfile
    import shutil
    
    # Create temporary test directory
    test_temp_dir = tempfile.mkdtemp(prefix='dissonance_test_')
    
    try:
        # Run tests
        unittest.main(verbosity=2)
    finally:
        # Clean up
        shutil.rmtree(test_temp_dir, ignore_errors=True)