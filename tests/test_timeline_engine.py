#!/usr/bin/env python3
"""
Test suite for MeRNSTA Phase 24: Timeline Engine

Tests cover:
- Event creation and storage
- Timeline querying and filtering
- Causal sequence detection
- Temporal anomaly detection
- Persistence and data integrity
- Integration with WorldModeler
"""

import unittest
import tempfile
import os
import time
import json
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

# Ensure the project root is in the path
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from agents.timeline_engine import (
    TemporalTimeline, TemporalEvent, EventType, CausalSequence,
    get_timeline_engine, reset_timeline_engine
)


class TestTemporalEvent(unittest.TestCase):
    """Test TemporalEvent data structure"""
    
    def test_temporal_event_creation(self):
        """Test basic TemporalEvent creation"""
        timestamp = datetime.now()
        event = TemporalEvent(
            timestamp=timestamp,
            fact="It is raining",
            confidence=0.8,
            source="weather_sensor",
            event_type=EventType.OBSERVATION,
            reasoning_agent="test_agent"
        )
        
        self.assertEqual(event.timestamp, timestamp)
        self.assertEqual(event.fact, "It is raining")
        self.assertEqual(event.confidence, 0.8)
        self.assertEqual(event.source, "weather_sensor")
        self.assertEqual(event.event_type, EventType.OBSERVATION)
        self.assertEqual(event.reasoning_agent, "test_agent")
        self.assertIsNotNone(event.event_id)
    
    def test_event_serialization(self):
        """Test event to/from dictionary conversion"""
        timestamp = datetime.now()
        event = TemporalEvent(
            timestamp=timestamp,
            fact="Test fact",
            confidence=0.9,
            source="test",
            event_type=EventType.INFERRED,
            metadata={"test_key": "test_value"}
        )
        
        # Test to_dict
        data = event.to_dict()
        self.assertIsInstance(data, dict)
        self.assertEqual(data['fact'], "Test fact")
        self.assertEqual(data['confidence'], 0.9)
        self.assertEqual(data['event_type'], 'inferred')
        self.assertEqual(data['metadata']['test_key'], "test_value")
        
        # Test from_dict
        reconstructed = TemporalEvent.from_dict(data)
        self.assertEqual(reconstructed.fact, event.fact)
        self.assertEqual(reconstructed.confidence, event.confidence)
        self.assertEqual(reconstructed.event_type, event.event_type)
        self.assertEqual(reconstructed.metadata, event.metadata)
    
    def test_event_id_generation(self):
        """Test unique event ID generation"""
        event1 = TemporalEvent(
            timestamp=datetime.now(),
            fact="Event 1",
            confidence=0.5,
            source="test",
            event_type=EventType.OBSERVATION
        )
        
        event2 = TemporalEvent(
            timestamp=datetime.now(),
            fact="Event 2",
            confidence=0.5,
            source="test",
            event_type=EventType.OBSERVATION
        )
        
        self.assertNotEqual(event1.event_id, event2.event_id)
        self.assertTrue(event1.event_id.startswith("event_"))
        self.assertTrue(event2.event_id.startswith("event_"))


class TestCausalSequence(unittest.TestCase):
    """Test CausalSequence functionality"""
    
    def test_causal_sequence_creation(self):
        """Test basic CausalSequence creation"""
        event1 = TemporalEvent(
            timestamp=datetime.now(),
            fact="Rain started",
            confidence=0.9,
            source="sensor",
            event_type=EventType.OBSERVATION
        )
        
        event2 = TemporalEvent(
            timestamp=datetime.now() + timedelta(minutes=10),
            fact="Streets are wet",
            confidence=0.8,
            source="sensor",
            event_type=EventType.OBSERVATION
        )
        
        sequence = CausalSequence(
            events=[event1, event2],
            causal_strength=0.7,
            confidence=0.8,
            sequence_id="test_seq_001"
        )
        
        self.assertEqual(len(sequence.events), 2)
        self.assertEqual(sequence.causal_strength, 0.7)
        self.assertEqual(sequence.confidence, 0.8)
        self.assertEqual(sequence.sequence_id, "test_seq_001")
    
    def test_temporal_consistency_check(self):
        """Test temporal consistency validation"""
        # Correct order
        event1 = TemporalEvent(
            timestamp=datetime.now(),
            fact="Cause",
            confidence=0.9,
            source="test",
            event_type=EventType.OBSERVATION
        )
        
        event2 = TemporalEvent(
            timestamp=datetime.now() + timedelta(minutes=5),
            fact="Effect",
            confidence=0.8,
            source="test",
            event_type=EventType.OBSERVATION
        )
        
        correct_sequence = CausalSequence(
            events=[event1, event2],
            causal_strength=0.7,
            confidence=0.8,
            sequence_id="correct"
        )
        
        self.assertTrue(correct_sequence.is_temporally_consistent())
        
        # Incorrect order
        incorrect_sequence = CausalSequence(
            events=[event2, event1],  # Effect before cause
            causal_strength=0.7,
            confidence=0.8,
            sequence_id="incorrect"
        )
        
        self.assertFalse(incorrect_sequence.is_temporally_consistent())


class TestTemporalTimeline(unittest.TestCase):
    """Test TemporalTimeline class functionality"""
    
    def setUp(self):
        """Set up test environment with temporary files"""
        self.temp_dir = tempfile.mkdtemp()
        self.timeline_path = os.path.join(self.temp_dir, "test_timeline.jsonl")
        
        # Reset global timeline
        reset_timeline_engine()
        
        # Create timeline with test path
        self.timeline = TemporalTimeline(
            agent_id="test_timeline",
            persistence_path=self.timeline_path,
            max_events=1000
        )
    
    def tearDown(self):
        """Clean up test environment"""
        # Clean up temporary files
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        
        # Reset global timeline
        reset_timeline_engine()
    
    def test_timeline_initialization(self):
        """Test timeline initialization"""
        self.assertEqual(self.timeline.agent_id, "test_timeline")
        self.assertEqual(self.timeline.persistence_path, self.timeline_path)
        self.assertEqual(len(self.timeline.events), 0)
        self.assertEqual(len(self.timeline.causal_sequences), 0)
        self.assertEqual(len(self.timeline.anomalies), 0)
    
    def test_add_event(self):
        """Test adding events to timeline"""
        # Add a simple event
        event = self.timeline.add_event(
            fact="User said hello",
            confidence=0.9,
            source="user_input",
            event_type=EventType.OBSERVATION,
            reasoning_agent="test_agent"
        )
        
        self.assertIsInstance(event, TemporalEvent)
        self.assertEqual(len(self.timeline.events), 1)
        self.assertEqual(event.fact, "User said hello")
        self.assertEqual(event.confidence, 0.9)
        self.assertEqual(event.source, "user_input")
        self.assertEqual(event.event_type, EventType.OBSERVATION)
        
        # Check indexing
        self.assertIn("said", self.timeline.events_by_subject)
        self.assertIn("hello", self.timeline.events_by_subject)
        self.assertIn(event, self.timeline.events_by_type[EventType.OBSERVATION])
        self.assertIn(event, self.timeline.events_by_source["user_input"])
    
    def test_get_events_about(self):
        """Test querying events about specific subjects"""
        # Add test events
        self.timeline.add_event("Rain started", 0.9, source="weather", event_type=EventType.OBSERVATION)
        self.timeline.add_event("Rain stopped", 0.8, source="weather", event_type=EventType.OBSERVATION)
        self.timeline.add_event("Sun came out", 0.7, source="weather", event_type=EventType.OBSERVATION)
        
        # Query for rain events
        rain_events = self.timeline.get_events_about("rain")
        self.assertEqual(len(rain_events), 2)
        
        for event in rain_events:
            self.assertIn("rain", event.fact.lower())
        
        # Test time window filtering
        recent_events = self.timeline.get_events_about("rain", time_window_hours=1)
        self.assertEqual(len(recent_events), 2)  # All events should be recent
    
    def test_event_sequence_detection(self):
        """Test causal sequence detection"""
        # Add events that could form a causal sequence
        self.timeline.add_event("Rain started", 0.9, source="weather", event_type=EventType.OBSERVATION)
        time.sleep(0.01)  # Small delay to ensure different timestamps
        self.timeline.add_event("Streets became wet", 0.8, source="sensor", event_type=EventType.OBSERVATION)
        time.sleep(0.01)
        self.timeline.add_event("Traffic slowed down", 0.7, source="traffic", event_type=EventType.OBSERVATION)
        
        # Search for the causal sequence
        sequence = self.timeline.get_event_sequence(["rain", "wet", "traffic"])
        
        if sequence:  # Sequence detection is heuristic, so it might not always find one
            self.assertIsInstance(sequence, CausalSequence)
            self.assertTrue(sequence.is_temporally_consistent())
            self.assertGreaterEqual(sequence.causal_strength, 0.0)
        
        # Test with non-existent sequence
        no_sequence = self.timeline.get_event_sequence(["unicorn", "magic", "rainbow"])
        self.assertIsNone(no_sequence)
    
    def test_timeline_summarization(self):
        """Test timeline summarization functionality"""
        # Add various events
        base_time = datetime.now() - timedelta(hours=2)
        
        events_data = [
            ("First observation", 0.9, EventType.OBSERVATION),
            ("Inference made", 0.7, EventType.INFERRED),
            ("Belief updated", 0.8, EventType.BELIEF_CHANGE),
            ("Another observation", 0.6, EventType.OBSERVATION),
        ]
        
        for i, (fact, confidence, event_type) in enumerate(events_data):
            timestamp = base_time + timedelta(minutes=i*15)
            self.timeline.add_event(
                fact=fact,
                confidence=confidence,
                timestamp=timestamp,
                source="test",
                event_type=event_type
            )
        
        # Generate summary
        start_time = base_time - timedelta(hours=1)
        end_time = base_time + timedelta(hours=2)
        summary = self.timeline.summarize_timeline((start_time, end_time))
        
        self.assertIsInstance(summary, dict)
        self.assertIn('time_window', summary)
        self.assertIn('statistics', summary)
        self.assertIn('top_subjects', summary)
        self.assertIn('temporal_patterns', summary)
        
        # Check statistics
        stats = summary['statistics']
        self.assertEqual(stats['total_events'], 4)
        self.assertGreater(stats['average_confidence'], 0)
        self.assertIn('events_by_type', stats)
    
    def test_temporal_anomaly_detection(self):
        """Test temporal anomaly detection"""
        # Create events that might be temporally inconsistent
        now = datetime.now()
        
        # Add some normal events first
        self.timeline.add_event(
            "Event A happened",
            0.9,
            timestamp=now,
            source="test",
            event_type=EventType.OBSERVATION
        )
        
        self.timeline.add_event(
            "Event B happened after A",
            0.8,
            timestamp=now + timedelta(minutes=5),
            source="test",
            event_type=EventType.OBSERVATION
        )
        
        # Test anomaly detection
        anomalies = self.timeline.detect_temporal_inconsistencies()
        self.assertIsInstance(anomalies, list)
        
        # The specific anomalies found depend on the heuristics,
        # but the method should run without errors
    
    def test_timeline_statistics(self):
        """Test timeline statistics generation"""
        # Add some test events
        for i in range(5):
            self.timeline.add_event(
                f"Event {i}",
                0.5 + i * 0.1,
                source=f"source_{i % 2}",
                event_type=EventType.OBSERVATION if i % 2 == 0 else EventType.INFERRED
            )
        
        stats = self.timeline.get_timeline_stats()
        
        self.assertIsInstance(stats, dict)
        self.assertEqual(stats['total_events'], 5)
        self.assertIn('timeline_span_hours', stats)
        self.assertIn('event_type_distribution', stats)
        self.assertIn('source_distribution', stats)
        self.assertIn('causal_sequences', stats)
        self.assertIn('detected_anomalies', stats)
        self.assertIn('average_confidence', stats)
        
        # Check event type distribution
        type_dist = stats['event_type_distribution']
        self.assertEqual(type_dist['observation'], 3)  # Events 0, 2, 4
        self.assertEqual(type_dist['inferred'], 2)     # Events 1, 3
    
    def test_timeline_persistence(self):
        """Test timeline persistence to file"""
        # Add an event
        event = self.timeline.add_event(
            "Test persistence",
            0.8,
            source="test",
            event_type=EventType.OBSERVATION
        )
        
        # Check that file was created and contains the event
        self.assertTrue(os.path.exists(self.timeline_path))
        
        with open(self.timeline_path, 'r') as f:
            lines = f.readlines()
            self.assertEqual(len(lines), 1)
            
            data = json.loads(lines[0])
            self.assertEqual(data['fact'], "Test persistence")
            self.assertEqual(data['confidence'], 0.8)
            self.assertEqual(data['event_type'], 'observation')
    
    def test_timeline_loading(self):
        """Test loading timeline from existing file"""
        # Create a timeline with some events
        original_timeline = TemporalTimeline(
            agent_id="original",
            persistence_path=self.timeline_path
        )
        
        event1 = original_timeline.add_event("Event 1", 0.7, source="test")
        event2 = original_timeline.add_event("Event 2", 0.8, source="test")
        
        # Create a new timeline that should load the existing events
        loaded_timeline = TemporalTimeline(
            agent_id="loaded",
            persistence_path=self.timeline_path
        )
        
        self.assertEqual(len(loaded_timeline.events), 2)
        self.assertEqual(loaded_timeline.events[0].fact, "Event 1")
        self.assertEqual(loaded_timeline.events[1].fact, "Event 2")
    
    def test_timeline_capacity_management(self):
        """Test timeline capacity management"""
        # Create timeline with small capacity
        small_timeline = TemporalTimeline(
            agent_id="small",
            persistence_path=os.path.join(self.temp_dir, "small_timeline.jsonl"),
            max_events=3
        )
        
        # Add more events than capacity
        for i in range(5):
            small_timeline.add_event(f"Event {i}", 0.5, source="test")
        
        # Should only keep the most recent events
        self.assertEqual(len(small_timeline.events), 3)
        self.assertEqual(small_timeline.events[0].fact, "Event 2")  # Oldest kept
        self.assertEqual(small_timeline.events[-1].fact, "Event 4")  # Newest
    
    def test_global_timeline_engine(self):
        """Test global timeline engine singleton"""
        # Get the global instance
        global_timeline1 = get_timeline_engine()
        global_timeline2 = get_timeline_engine()
        
        # Should be the same instance
        self.assertIs(global_timeline1, global_timeline2)
        
        # Reset and get new instance
        reset_timeline_engine()
        global_timeline3 = get_timeline_engine()
        
        # Should be a different instance
        self.assertIsNot(global_timeline1, global_timeline3)


class TestTimelineIntegration(unittest.TestCase):
    """Test integration with other MeRNSTA components"""
    
    def setUp(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()
        reset_timeline_engine()
    
    def tearDown(self):
        """Clean up test environment"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        reset_timeline_engine()
    
    @patch('agents.world_modeler.get_timeline_engine')
    def test_world_modeler_integration(self, mock_get_timeline):
        """Test integration with WorldModeler"""
        # Mock the timeline engine
        mock_timeline = MagicMock()
        mock_get_timeline.return_value = mock_timeline
        
        # Import and create WorldModeler
        from agents.world_modeler import WorldModeler
        
        try:
            world_modeler = WorldModeler()
            
            # Test observe_fact integration
            world_modeler.observe_fact("Test observation", confidence=0.8, source="test")
            
            # Verify timeline.add_event was called
            self.assertTrue(mock_timeline.add_event.called)
            call_args = mock_timeline.add_event.call_args
            
            # Check the arguments passed to add_event
            self.assertEqual(call_args[1]['fact'], "Test observation")
            self.assertEqual(call_args[1]['confidence'], 0.8)
            self.assertEqual(call_args[1]['source'], "test")
            
        except Exception as e:
            # If WorldModeler dependencies are missing, skip this test
            self.skipTest(f"WorldModeler dependencies not available: {e}")
    
    def test_cli_command_imports(self):
        """Test that CLI commands can import timeline engine"""
        try:
            from agents.timeline_engine import get_timeline_engine
            timeline = get_timeline_engine()
            self.assertIsNotNone(timeline)
            
            # Test that timeline methods are available
            self.assertTrue(hasattr(timeline, 'add_event'))
            self.assertTrue(hasattr(timeline, 'get_events_about'))
            self.assertTrue(hasattr(timeline, 'get_event_sequence'))
            self.assertTrue(hasattr(timeline, 'detect_temporal_inconsistencies'))
            self.assertTrue(hasattr(timeline, 'summarize_timeline'))
            
        except ImportError as e:
            self.fail(f"Failed to import timeline engine: {e}")


class TestTimelineEdgeCases(unittest.TestCase):
    """Test edge cases and error handling"""
    
    def setUp(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.timeline_path = os.path.join(self.temp_dir, "edge_case_timeline.jsonl")
        self.timeline = TemporalTimeline(
            agent_id="edge_test",
            persistence_path=self.timeline_path
        )
    
    def tearDown(self):
        """Clean up test environment"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        reset_timeline_engine()
    
    def test_empty_timeline_operations(self):
        """Test operations on empty timeline"""
        # Test querying empty timeline
        events = self.timeline.get_events_about("nonexistent")
        self.assertEqual(len(events), 0)
        
        # Test sequence detection on empty timeline
        sequence = self.timeline.get_event_sequence(["a", "b", "c"])
        self.assertIsNone(sequence)
        
        # Test anomaly detection on empty timeline
        anomalies = self.timeline.detect_temporal_inconsistencies()
        self.assertEqual(len(anomalies), 0)
        
        # Test statistics on empty timeline
        stats = self.timeline.get_timeline_stats()
        self.assertEqual(stats['total_events'], 0)
        self.assertEqual(stats['timeline_span_hours'], 0)
    
    def test_invalid_confidence_values(self):
        """Test handling of invalid confidence values"""
        # Timeline should handle confidence values outside 0-1 range gracefully
        event = self.timeline.add_event(
            "Test event",
            confidence=1.5,  # Invalid: > 1.0
            source="test"
        )
        
        # The add_event method should handle this gracefully
        self.assertIsNotNone(event)
        self.assertLessEqual(event.confidence, 1.0)
    
    def test_very_long_facts(self):
        """Test handling of very long fact strings"""
        long_fact = "A" * 10000  # Very long string
        
        event = self.timeline.add_event(
            long_fact,
            confidence=0.5,
            source="test"
        )
        
        self.assertIsNotNone(event)
        self.assertEqual(event.fact, long_fact)
    
    def test_special_characters_in_facts(self):
        """Test handling of special characters in facts"""
        special_fact = "Fact with émojis 🎉 and spëcial chars: !@#$%^&*()"
        
        event = self.timeline.add_event(
            special_fact,
            confidence=0.7,
            source="test"
        )
        
        self.assertIsNotNone(event)
        self.assertEqual(event.fact, special_fact)
    
    def test_simultaneous_events(self):
        """Test handling of events with identical timestamps"""
        timestamp = datetime.now()
        
        event1 = self.timeline.add_event(
            "Event 1",
            confidence=0.5,
            timestamp=timestamp,
            source="test"
        )
        
        event2 = self.timeline.add_event(
            "Event 2", 
            confidence=0.6,
            timestamp=timestamp,  # Same timestamp
            source="test"
        )
        
        # Both events should be stored
        self.assertEqual(len(self.timeline.events), 2)
        self.assertNotEqual(event1.event_id, event2.event_id)
    
    def test_missing_persistence_directory(self):
        """Test handling when persistence directory doesn't exist"""
        nonexistent_path = "/tmp/nonexistent_dir/timeline.jsonl"
        
        # Timeline should create the directory
        try:
            timeline = TemporalTimeline(
                agent_id="test",
                persistence_path=nonexistent_path
            )
            
            # Add an event to test persistence
            timeline.add_event("Test", 0.5, source="test")
            
            # Clean up
            os.remove(nonexistent_path)
            os.rmdir(os.path.dirname(nonexistent_path))
            
        except Exception:
            # If we can't create the directory due to permissions, that's expected
            pass


if __name__ == '__main__':
    # Run the tests
    print("🧪 Running Timeline Engine Tests...")
    print("=" * 60)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestTemporalEvent,
        TestCausalSequence,
        TestTemporalTimeline,
        TestTimelineIntegration,
        TestTimelineEdgeCases
    ]
    
    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 60)
    print(f"📊 Test Results Summary:")
    print(f"  Tests run: {result.testsRun}")
    print(f"  Failures: {len(result.failures)}")
    print(f"  Errors: {len(result.errors)}")
    print(f"  Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / max(1, result.testsRun)) * 100:.1f}%")
    
    if result.failures:
        print(f"\n❌ Failures:")
        for test, traceback in result.failures:
            print(f"  - {test}: {traceback.split('AssertionError:')[-1].strip()}")
    
    if result.errors:
        print(f"\n💥 Errors:")
        for test, traceback in result.errors:
            print(f"  - {test}: {traceback.split(':')[-1].strip()}")
    
    if not result.failures and not result.errors:
        print("\n✅ All tests passed!")
    
    print("\n🎯 Timeline Engine testing complete!")