#!/usr/bin/env python3
"""
Tests for Phase 25 SelfModel - Reflective Self-Awareness & Identity Continuity

Tests the reflective self-awareness capabilities including:
- Core identity tracking and evolution
- Personality sync with drift detection
- Self-reflection journal system
- Identity continuity across sessions
"""

import unittest
import tempfile
import json
import os
import time
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

# Import the SelfModel classes
from agents.self_model import SelfModel, CoreSelf, EvolvingSelf, ReflectionEntry, get_self_model


class TestSelfModel(unittest.TestCase):
    """Test cases for the Phase 25 SelfModel."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create temporary output directory for testing
        self.temp_dir = tempfile.mkdtemp()
        self.temp_output_dir = Path(self.temp_dir) / "output"
        self.temp_output_dir.mkdir(exist_ok=True)
        
        # Create SelfModel with custom output directory
        self.self_model = SelfModel("TestSelfModel")
        self.self_model.output_dir = self.temp_output_dir
        self.self_model.self_model_file = self.temp_output_dir / "self_model.json"
    
    def tearDown(self):
        """Clean up test fixtures."""
        # Clean up temporary files
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
        
        # Reset global instance
        if hasattr(get_self_model, '_instance'):
            delattr(get_self_model, '_instance')
    
    def test_initialization(self):
        """Test that the SelfModel initializes correctly."""
        self.assertIsNotNone(self.self_model)
        self.assertEqual(self.self_model.name, "TestSelfModel")
        
        # Check core identity
        self.assertEqual(self.self_model.core_self.agent_name, "MeRNSTA")
        self.assertEqual(self.self_model.core_self.version, "0.7.0")
        self.assertIn("contradiction-aware", self.self_model.core_self.purpose)
        
        # Check evolving state
        self.assertIsInstance(self.self_model.evolving_self.trait_vector, dict)
        self.assertIn("empathetic", self.self_model.evolving_self.trait_vector)
        self.assertIn("analytical", self.self_model.evolving_self.trait_vector)
        self.assertGreaterEqual(self.self_model.evolving_self.confidence_level, 0.0)
        self.assertLessEqual(self.self_model.evolving_self.confidence_level, 1.0)
        
        # Check journal
        self.assertIsInstance(self.self_model.journal, list)
        self.assertEqual(len(self.self_model.journal), 0)  # Should start empty
    
    def test_core_self_dataclass(self):
        """Test CoreSelf dataclass functionality."""
        core = CoreSelf(
            agent_name="TestAgent",
            version="1.0.0",
            created_at="2025-01-01T00:00:00",
            purpose="Test purpose"
        )
        
        # Test serialization
        core_dict = core.to_dict()
        self.assertIsInstance(core_dict, dict)
        self.assertEqual(core_dict['agent_name'], "TestAgent")
        
        # Test deserialization
        reconstructed = CoreSelf.from_dict(core_dict)
        self.assertEqual(core.agent_name, reconstructed.agent_name)
        self.assertEqual(core.version, reconstructed.version)
    
    def test_evolving_self_dataclass(self):
        """Test EvolvingSelf dataclass functionality."""
        evolving = EvolvingSelf(
            dominant_tone="analytical",
            emotion_state="focused",
            confidence_level=0.75,
            trait_vector={"curious": 0.8, "analytical": 0.9}
        )
        
        # Test serialization
        evolving_dict = evolving.to_dict()
        self.assertIsInstance(evolving_dict, dict)
        self.assertEqual(evolving_dict['dominant_tone'], "analytical")
        
        # Test deserialization
        reconstructed = EvolvingSelf.from_dict(evolving_dict)
        self.assertEqual(evolving.dominant_tone, reconstructed.dominant_tone)
        self.assertEqual(evolving.trait_vector, reconstructed.trait_vector)
    
    def test_reflection_entry_dataclass(self):
        """Test ReflectionEntry dataclass functionality."""
        entry = ReflectionEntry(
            timestamp="2025-01-01T12:00:00",
            summary="Test reflection",
            trigger="test_trigger",
            changes={"trait": ["old", "new"]}
        )
        
        # Test serialization
        entry_dict = entry.to_dict()
        self.assertIsInstance(entry_dict, dict)
        self.assertEqual(entry_dict['summary'], "Test reflection")
        
        # Test deserialization
        reconstructed = ReflectionEntry.from_dict(entry_dict)
        self.assertEqual(entry.summary, reconstructed.summary)
        self.assertEqual(entry.changes, reconstructed.changes)
    
    def test_generate_self_summary(self):
        """Test generating self-summary."""
        summary = self.self_model.generate_self_summary()
        
        self.assertIsInstance(summary, str)
        self.assertIn("MeRNSTA Identity Snapshot", summary)
        self.assertIn("Core Identity:", summary)
        self.assertIn("Current State:", summary)
        self.assertIn("Primary Traits:", summary)
        self.assertIn(self.self_model.core_self.purpose, summary)
        self.assertIn(self.self_model.evolving_self.dominant_tone, summary)
    
    def test_write_reflection_entry(self):
        """Test writing reflection entries."""
        initial_count = len(self.self_model.journal)
        
        # Write a reflection entry
        self.self_model.write_reflection_entry(
            trigger="test_trigger",
            summary="Test reflection summary",
            changes={"trait": ["old_value", "new_value"]}
        )
        
        # Check that entry was added
        self.assertEqual(len(self.self_model.journal), initial_count + 1)
        
        entry = self.self_model.journal[-1]
        self.assertEqual(entry.trigger, "test_trigger")
        self.assertEqual(entry.summary, "Test reflection summary")
        self.assertIn("trait", entry.changes)
    
    def test_journal_size_limit(self):
        """Test that journal respects size limits."""
        # Set a small limit for testing
        original_limit = self.self_model.max_journal_entries
        self.self_model.max_journal_entries = 3
        
        # Write more entries than the limit
        for i in range(5):
            self.self_model.write_reflection_entry(
                trigger=f"test_trigger_{i}",
                summary=f"Test summary {i}",
                changes={"test": i}
            )
        
        # Check that only the limit number of entries are kept
        self.assertEqual(len(self.self_model.journal), 3)
        
        # Check that the most recent entries are kept
        self.assertEqual(self.self_model.journal[-1].summary, "Test summary 4")
        self.assertEqual(self.self_model.journal[0].summary, "Test summary 2")
        
        # Restore original limit
        self.self_model.max_journal_entries = original_limit
    
    def test_detect_drift(self):
        """Test drift detection functionality."""
        # Create a previous state
        prev_state = EvolvingSelf(
            dominant_tone="neutral",
            emotion_state="calm",
            confidence_level=0.8,
            trait_vector={"empathetic": 0.5, "analytical": 0.7}
        )
        
        # Modify current state to trigger drift detection
        self.self_model.evolving_self.dominant_tone = "assertive"
        self.self_model.evolving_self.trait_vector["empathetic"] = 0.1  # Significant change
        
        # Detect drift
        drift_detected = self.self_model.detect_drift(prev_state)
        
        self.assertTrue(drift_detected)
        self.assertGreater(len(self.self_model.journal), 0)
        
        # Check that a reflection entry was created
        last_entry = self.self_model.journal[-1]
        self.assertIn("drift", last_entry.trigger)
        self.assertIn("tone", last_entry.changes)
        self.assertIn("traits", last_entry.changes)
    
    def test_no_drift_detection(self):
        """Test that small changes don't trigger drift detection."""
        # Create a previous state similar to current
        prev_state = EvolvingSelf(**self.self_model.evolving_self.to_dict())
        
        # Make only small changes
        self.self_model.evolving_self.confidence_level += 0.1  # Small change
        
        initial_journal_size = len(self.self_model.journal)
        drift_detected = self.self_model.detect_drift(prev_state)
        
        self.assertFalse(drift_detected)
        self.assertEqual(len(self.self_model.journal), initial_journal_size)
    
    def test_manual_reflection(self):
        """Test manual reflection trigger."""
        initial_journal_size = len(self.self_model.journal)
        
        result = self.self_model.manual_reflection()
        
        self.assertIsInstance(result, str)
        self.assertIn("Reflection completed", result)
        self.assertGreater(len(self.self_model.journal), initial_journal_size)
        
        # Check that a reflection entry was created
        last_entry = self.self_model.journal[-1]
        self.assertIn("reflection", last_entry.trigger)
    
    def test_get_self_journal(self):
        """Test getting the self journal."""
        # Add some test entries
        for i in range(3):
            self.self_model.write_reflection_entry(
                trigger=f"test_{i}",
                summary=f"Summary {i}",
                changes={"value": i}
            )
        
        journal = self.self_model.get_self_journal()
        
        self.assertIsInstance(journal, list)
        self.assertEqual(len(journal), 3)
        
        # Check that entries are properly serialized
        for i, entry in enumerate(journal):
            self.assertIsInstance(entry, dict)
            self.assertEqual(entry['trigger'], f"test_{i}")
            self.assertEqual(entry['summary'], f"Summary {i}")
    
    def test_get_recent_changes(self):
        """Test getting recent changes summary."""
        # Test with no changes
        changes = self.self_model.get_recent_changes()
        self.assertEqual(changes['status'], 'stable')
        self.assertEqual(len(changes['recent_changes']), 0)
        
        # Add some reflection entries
        self.self_model.write_reflection_entry(
            trigger="test_change",
            summary="Test change occurred",
            changes={"trait": ["old", "new"]}
        )
        
        changes = self.self_model.get_recent_changes()
        self.assertEqual(changes['status'], 'stable')  # Only one change
        self.assertEqual(len(changes['recent_changes']), 1)
        
        # Add another entry
        self.self_model.write_reflection_entry(
            trigger="another_change",
            summary="Another change occurred",
            changes={"emotion": ["calm", "excited"]}
        )
        
        changes = self.self_model.get_recent_changes()
        self.assertEqual(changes['status'], 'evolving')  # Multiple changes
        self.assertEqual(len(changes['recent_changes']), 2)
    
    def test_state_persistence(self):
        """Test saving and loading state."""
        # Modify the model state
        self.self_model.evolving_self.dominant_tone = "playful"
        self.self_model.evolving_self.confidence_level = 0.95
        self.self_model.write_reflection_entry(
            trigger="persistence_test",
            summary="Testing state persistence",
            changes={"test": "value"}
        )
        
        # Save state
        self.self_model.save_state()
        
        # Verify file was created
        self.assertTrue(self.self_model.self_model_file.exists())
        
        # Create new model instance and load state
        new_model = SelfModel("TestSelfModel2")
        new_model.output_dir = self.temp_output_dir
        new_model.self_model_file = self.temp_output_dir / "self_model.json"
        new_model.load_state()
        
        # Verify state was loaded correctly
        self.assertEqual(new_model.evolving_self.dominant_tone, "playful")
        self.assertEqual(new_model.evolving_self.confidence_level, 0.95)
        self.assertEqual(len(new_model.journal), 1)
        self.assertEqual(new_model.journal[0].trigger, "persistence_test")
    
    def test_drift_summary_generation(self):
        """Test drift summary generation."""
        # Test various change types
        changes = {
            "tone": ["neutral", "assertive"],
            "confidence": [0.8, 0.6],
            "traits": {
                "empathetic": [0.7, 0.4],
                "analytical": [0.6, 0.9]
            }
        }
        
        summary = self.self_model._generate_drift_summary(changes)
        
        self.assertIsInstance(summary, str)
        self.assertIn("Tone shifted", summary)
        self.assertIn("Confidence decreased", summary)
        self.assertIn("Empathetic decreased", summary)
        self.assertIn("Analytical increased", summary)
    
    @patch('agents.self_model.logger')
    def test_error_handling(self, mock_logger):
        """Test error handling in various methods."""
        # Test error handling in drift summary generation
        invalid_changes = {"invalid": None}
        summary = self.self_model._generate_drift_summary(invalid_changes)
        self.assertIsInstance(summary, str)
        
        # Test error handling in recent changes by causing an exception
        with patch.object(self.self_model, 'get_recent_journal_entries', side_effect=Exception("Test error")):
            changes = self.self_model.get_recent_changes()
            self.assertEqual(changes['status'], 'unknown')

    @patch('agents.base.BaseAgent.personality')
    def test_sync_from_personality_engine(self, mock_personality):
        """Test syncing from personality engine."""
        # Mock personality engine components
        mock_tone_profile = Mock()
        mock_tone_profile.mode.value = "empathetic"
        
        mock_personality_state = Mock()
        mock_personality_state.tone = "empathetic"
        mock_personality_state.emotional_state = "compassionate"
        mock_personality_state.core_traits = {
            "empathetic": 0.9,
            "analytical": 0.6,
            "playful": 0.7
        }
        mock_personality_state.stability_factor = 0.8
        
        mock_engine = Mock()
        mock_engine.get_current_tone_profile.return_value = mock_tone_profile
        mock_engine.personality_state = mock_personality_state
        
        # Test successful sync
        success = self.self_model.sync_from_personality_engine(mock_engine)
        
        self.assertTrue(success)
        self.assertEqual(self.self_model.evolving_self.dominant_tone, "empathetic")
        self.assertEqual(self.self_model.evolving_self.emotion_state, "compassionate")
        self.assertEqual(self.self_model.evolving_self.trait_vector["empathetic"], 0.9)
        
        # Verify that confidence was calculated from stability
        expected_confidence = min(0.95, 0.5 + (0.8 * 0.45))
        self.assertAlmostEqual(self.self_model.evolving_self.confidence_level, expected_confidence, places=2)
    
    @patch('agents.base.BaseAgent.personality', None)
    def test_sync_without_personality_engine(self):
        """Test sync behavior when personality engine is not available."""
        success = self.self_model.sync_from_personality_engine(None)
        self.assertFalse(success)
    
    def test_response_generation(self):
        """Test that the agent can generate responses."""
        with patch.object(self.self_model, 'generate_llm_response') as mock_llm:
            mock_llm.return_value = "Test response"
            
            response = self.self_model.respond("Tell me about yourself")
            
            self.assertIsInstance(response, str)
            mock_llm.assert_called_once()
    
    def test_agent_instructions(self):
        """Test that agent instructions are properly defined."""
        instructions = self.self_model.get_agent_instructions()
        
        self.assertIsInstance(instructions, str)
        self.assertIn("SelfModel agent", instructions)
        self.assertIn("self-awareness", instructions)
        self.assertIn("identity tracking", instructions)


class TestGlobalSelfModel(unittest.TestCase):
    """Test the global SelfModel instance functionality."""
    
    def tearDown(self):
        """Clean up global instance."""
        if hasattr(get_self_model, '_instance'):
            delattr(get_self_model, '_instance')
    
    def test_singleton_behavior(self):
        """Test that get_self_model returns the same instance."""
        model1 = get_self_model()
        model2 = get_self_model()
        
        self.assertIs(model1, model2)
        self.assertEqual(model1.name, "SelfModel")
    
    def test_custom_name(self):
        """Test creating SelfModel with custom name."""
        model = get_self_model("CustomSelfModel")
        self.assertEqual(model.name, "CustomSelfModel")


class TestIntegrationScenarios(unittest.TestCase):
    """Test realistic integration scenarios."""
    
    def setUp(self):
        """Set up for integration tests."""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_output_dir = Path(self.temp_dir) / "output"
        self.temp_output_dir.mkdir(exist_ok=True)
        
        self.self_model = SelfModel("IntegrationTest")
        self.self_model.output_dir = self.temp_output_dir
        self.self_model.self_model_file = self.temp_output_dir / "self_model.json"
    
    def tearDown(self):
        """Clean up integration test fixtures."""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_personality_evolution_scenario(self):
        """Test a realistic personality evolution scenario."""
        # Initial state
        initial_summary = self.self_model.generate_self_summary()
        self.assertIn("stable", initial_summary)
        
        # Simulate personality changes over time
        states = [
            {"tone": "analytical", "emotion": "focused", "empathetic": 0.4, "analytical": 0.9},
            {"tone": "empathetic", "emotion": "caring", "empathetic": 0.8, "analytical": 0.7},
            {"tone": "playful", "emotion": "cheerful", "empathetic": 0.6, "analytical": 0.5}
        ]
        
        for i, state in enumerate(states):
            # Update state
            self.self_model.evolving_self.dominant_tone = state["tone"]
            self.self_model.evolving_self.emotion_state = state["emotion"]
            self.self_model.evolving_self.trait_vector["empathetic"] = state["empathetic"]
            self.self_model.evolving_self.trait_vector["analytical"] = state["analytical"]
            
            # Simulate drift detection
            if i > 0:
                prev_state = EvolvingSelf(
                    dominant_tone=states[i-1]["tone"],
                    emotion_state=states[i-1]["emotion"],
                    confidence_level=0.8,
                    trait_vector={
                        "empathetic": states[i-1]["empathetic"],
                        "analytical": states[i-1]["analytical"],
                        "playful": 0.4,
                        "assertive": 0.5,
                        "reflective": 0.9
                    }
                )
                
                drift_detected = self.self_model.detect_drift(prev_state)
                self.assertTrue(drift_detected)
        
        # Check that multiple journal entries were created
        self.assertGreaterEqual(len(self.self_model.journal), 2)
        
        # Check that evolution is reflected in status
        recent_changes = self.self_model.get_recent_changes()
        self.assertEqual(recent_changes['status'], 'evolving')
        
        # Final summary should reflect changes
        final_summary = self.self_model.generate_self_summary()
        self.assertIn("evolving", final_summary)
    
    def test_session_continuity(self):
        """Test identity continuity across session restarts."""
        # Create initial state with some evolution
        self.self_model.evolving_self.dominant_tone = "assertive"
        self.self_model.write_reflection_entry(
            trigger="session_test",
            summary="Testing session continuity",
            changes={"tone": ["neutral", "assertive"]}
        )
        
        # Save state
        self.self_model.save_state()
        initial_journal_size = len(self.self_model.journal)
        
        # Simulate session restart
        new_model = SelfModel("SessionTest")
        new_model.output_dir = self.temp_output_dir
        new_model.self_model_file = self.temp_output_dir / "self_model.json"
        new_model.load_state()
        
        # Verify continuity
        self.assertEqual(new_model.evolving_self.dominant_tone, "assertive")
        self.assertEqual(len(new_model.journal), initial_journal_size)
        self.assertEqual(new_model.journal[-1].trigger, "session_test")
        
        # Add new reflection and verify it builds on previous state
        new_model.manual_reflection()
        self.assertGreater(len(new_model.journal), initial_journal_size)


if __name__ == '__main__':
    unittest.main()