#!/usr/bin/env python3
"""
Test suite for Phase 8 Emotion Model
"""

import unittest
import tempfile
import os
import time
from unittest.mock import patch, MagicMock

from storage.emotion_model import EmotionModel, EmotionState, MoodSignature, get_emotion_model


class TestEmotionState(unittest.TestCase):
    """Test cases for EmotionState dataclass."""
    
    def test_emotion_state_creation(self):
        """Test creating EmotionState with valid values."""
        emotion = EmotionState(valence=0.5, arousal=0.7)
        
        self.assertEqual(emotion.valence, 0.5)
        self.assertEqual(emotion.arousal, 0.7)
        self.assertIsNotNone(emotion.timestamp)
        self.assertIsNone(emotion.source_token_id)
        self.assertIsNone(emotion.event_type)
    
    def test_emotion_state_clamping(self):
        """Test that emotion values are clamped to valid ranges."""
        # Test valence clamping
        emotion1 = EmotionState(valence=2.0, arousal=0.5)
        self.assertEqual(emotion1.valence, 1.0)
        
        emotion2 = EmotionState(valence=-2.0, arousal=0.5)
        self.assertEqual(emotion2.valence, -1.0)
        
        # Test arousal clamping
        emotion3 = EmotionState(valence=0.0, arousal=-0.5)
        self.assertEqual(emotion3.arousal, 0.0)
        
        emotion4 = EmotionState(valence=0.0, arousal=2.0)
        self.assertEqual(emotion4.arousal, 1.0)
    
    def test_emotion_state_serialization(self):
        """Test emotion state to_dict and from_dict."""
        emotion = EmotionState(
            valence=0.3,
            arousal=0.6,
            source_token_id="test_token",
            event_type="curiosity"
        )
        
        data = emotion.to_dict()
        self.assertIn('valence', data)
        self.assertIn('arousal', data)
        self.assertIn('source_token_id', data)
        self.assertIn('event_type', data)
        
        restored_emotion = EmotionState.from_dict(data)
        self.assertEqual(restored_emotion.valence, emotion.valence)
        self.assertEqual(restored_emotion.arousal, emotion.arousal)
        self.assertEqual(restored_emotion.source_token_id, emotion.source_token_id)
        self.assertEqual(restored_emotion.event_type, emotion.event_type)


class TestEmotionModel(unittest.TestCase):
    """Test cases for EmotionModel."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        self.test_db.close()
        self.db_path = self.test_db.name
        
        self.emotion_model = EmotionModel(self.db_path)
    
    def tearDown(self):
        """Clean up test environment."""
        if os.path.exists(self.db_path):
            os.unlink(self.db_path)
    
    def test_emotion_model_initialization(self):
        """Test emotion model initialization."""
        self.assertIsNotNone(self.emotion_model.current_emotion_state)
        self.assertEqual(len(self.emotion_model.emotion_history), 0)
        self.assertEqual(len(self.emotion_model.token_emotions), 0)
        self.assertIsInstance(self.emotion_model.emotion_event_mappings, dict)
    
    def test_update_emotion_from_event(self):
        """Test updating emotion from cognitive events."""
        # Test contradiction event
        emotion_state = self.emotion_model.update_emotion_from_event(
            token_id="test_token",
            event_type="contradiction",
            strength=1.0
        )
        
        self.assertIsNotNone(emotion_state)
        self.assertLess(emotion_state.valence, 0)  # Should be negative for contradiction
        self.assertGreater(emotion_state.arousal, 0)  # Should have some arousal
        self.assertEqual(emotion_state.event_type, "contradiction")
        self.assertEqual(emotion_state.source_token_id, "test_token")
        
        # Check that emotion was stored in history and token emotions
        self.assertGreater(len(self.emotion_model.emotion_history), 0)
        self.assertIn("test_token", self.emotion_model.token_emotions)
    
    def test_emotion_event_types(self):
        """Test different emotion event types."""
        test_events = [
            ("novelty", 0.3, 0.6),  # Expected to be positive valence
            ("resolution", 0.7, 0.4),  # Expected to be positive valence
            ("confusion", -0.3, 0.8),  # Expected to be negative valence
            ("discovery", 0.8, 0.7)  # Expected to be positive valence
        ]
        
        for event_type, expected_valence_sign, expected_arousal_min in test_events:
            emotion_state = self.emotion_model.update_emotion_from_event(
                token_id=None,
                event_type=event_type,
                strength=1.0
            )
            
            if expected_valence_sign > 0:
                self.assertGreater(emotion_state.valence, 0, 
                                 f"Expected positive valence for {event_type}")
            else:
                self.assertLess(emotion_state.valence, 0,
                               f"Expected negative valence for {event_type}")
            
            self.assertGreaterEqual(emotion_state.arousal, 0,
                                   f"Expected positive arousal for {event_type}")
    
    def test_get_current_mood(self):
        """Test mood calculation from recent emotions."""
        # Add some emotions to history
        self.emotion_model.update_emotion_from_event(None, "curiosity", 1.0)
        self.emotion_model.update_emotion_from_event(None, "excitement", 1.0)
        self.emotion_model.update_emotion_from_event(None, "satisfaction", 0.8)
        
        mood_info = self.emotion_model.get_current_mood()
        
        self.assertIn("mood_label", mood_info)
        self.assertIn("valence", mood_info)
        self.assertIn("arousal", mood_info)
        self.assertIn("confidence", mood_info)
        self.assertIn("contributing_events", mood_info)
        
        # Mood should be positive given the positive events
        self.assertGreater(mood_info["valence"], 0)
    
    def test_mood_classification(self):
        """Test mood classification logic."""
        # Test specific mood classifications
        test_cases = [
            (0.7, 0.8, "excited"),  # High positive valence, high arousal
            (0.3, 0.1, "calm"),     # Moderate positive valence, low arousal
            (-0.7, 0.8, "angry"),   # High negative valence, high arousal
            (-0.3, 0.1, "sad"),     # Moderate negative valence, low arousal
            (0.0, 0.5, "alert")     # Neutral valence, medium arousal
        ]
        
        for valence, arousal, expected_mood in test_cases:
            classified_mood = self.emotion_model._classify_mood(valence, arousal)
            self.assertEqual(classified_mood, expected_mood,
                           f"Expected {expected_mood} for valence={valence}, arousal={arousal}")
    
    def test_get_token_emotion(self):
        """Test getting emotion for specific tokens."""
        # Test token without emotion
        valence, arousal = self.emotion_model.get_token_emotion("unknown_token")
        self.assertEqual(valence, 0.0)
        self.assertEqual(arousal, 0.3)
        
        # Add emotion for a token
        self.emotion_model.update_emotion_from_event("test_token", "curiosity", 1.0)
        
        valence, arousal = self.emotion_model.get_token_emotion("test_token")
        self.assertGreater(valence, 0)  # Curiosity should be positive
        self.assertGreater(arousal, 0)
    
    def test_mood_signature_generation(self):
        """Test generating mood signature strings."""
        # Add some emotions
        self.emotion_model.update_emotion_from_event(None, "curiosity", 1.0)
        time.sleep(0.1)  # Small delay for duration calculation
        
        signature = self.emotion_model.get_mood_signature()
        self.assertIsInstance(signature, str)
        self.assertGreater(len(signature), 0)
        
        # Should contain mood information
        self.assertNotEqual(signature, "neutral")
    
    def test_emotion_override(self):
        """Test manual emotion override functionality."""
        # Set emotion override
        self.emotion_model.set_emotion_override(valence=-0.8, arousal=0.9, duration=60.0)
        
        # Current state should reflect override
        current_mood = self.emotion_model.get_current_mood()
        self.assertLess(current_mood["valence"], -0.5)  # Should be significantly negative
        self.assertGreater(current_mood["arousal"], 0.7)  # Should be high arousal
    
    def test_emotional_history_retrieval(self):
        """Test retrieving emotional history."""
        # Add some emotions
        self.emotion_model.update_emotion_from_event("token1", "curiosity", 1.0)
        self.emotion_model.update_emotion_from_event("token2", "frustration", 0.8)
        self.emotion_model.update_emotion_from_event(None, "satisfaction", 0.6)
        
        # Get global history
        global_history = self.emotion_model.get_emotional_history()
        self.assertGreater(len(global_history), 0)
        
        # Get token-specific history
        token_history = self.emotion_model.get_emotional_history("token1")
        self.assertGreater(len(token_history), 0)
        
        # Verify token-specific history only contains relevant entries
        for record in token_history:
            if record.get('token_id'):
                self.assertEqual(record['token_id'], "token1")
    
    def test_emotion_decay_and_blending(self):
        """Test that emotions decay and blend over time."""
        initial_state = self.emotion_model.current_emotion_state
        initial_valence = initial_state.valence
        
        # Add a strong positive emotion
        self.emotion_model.update_emotion_from_event(None, "excitement", 1.0)
        
        after_positive_valence = self.emotion_model.current_emotion_state.valence
        self.assertGreater(after_positive_valence, initial_valence)
        
        # Add a negative emotion
        self.emotion_model.update_emotion_from_event(None, "frustration", 1.0)
        
        after_negative_valence = self.emotion_model.current_emotion_state.valence
        # Should be less positive than after the excitement
        self.assertLess(after_negative_valence, after_positive_valence)


class TestEmotionModelIntegration(unittest.TestCase):
    """Integration tests for emotion model."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        self.test_db.close()
        self.db_path = self.test_db.name
    
    def tearDown(self):
        """Clean up test environment."""
        if os.path.exists(self.db_path):
            os.unlink(self.db_path)
    
    def test_get_emotion_model_singleton(self):
        """Test that get_emotion_model returns singleton."""
        model1 = get_emotion_model(self.db_path)
        model2 = get_emotion_model(self.db_path)
        
        # Should be the same instance
        self.assertIs(model1, model2)
    
    def test_database_persistence(self):
        """Test that emotional data persists across model instances."""
        # Create model and add some emotions
        model1 = EmotionModel(self.db_path)
        model1.update_emotion_from_event("test_token", "curiosity", 1.0)
        
        # Create new model instance with same database
        model2 = EmotionModel(self.db_path)
        
        # Should be able to retrieve emotional history
        history = model2.get_emotional_history()
        self.assertGreater(len(history), 0)
        
        # Should find the emotion we added
        found_emotion = False
        for record in history:
            if (record.get('token_id') == "test_token" and 
                record.get('event_type') == "curiosity"):
                found_emotion = True
                break
        
        self.assertTrue(found_emotion, "Should find persisted emotion in database")
    
    def test_mood_duration_calculation(self):
        """Test mood duration tracking."""
        model = EmotionModel(self.db_path)
        
        # Add emotions to establish a mood
        model.update_emotion_from_event(None, "excitement", 1.0)
        model.update_emotion_from_event(None, "curiosity", 0.8)
        
        # Get mood info
        mood_info = model.get_current_mood()
        initial_duration = mood_info.get('duration', 0.0)
        
        # Wait a short time
        time.sleep(0.1)
        
        # Add similar emotion to maintain mood
        model.update_emotion_from_event(None, "excitement", 0.9)
        
        # Duration should have increased
        mood_info2 = model.get_current_mood()
        new_duration = mood_info2.get('duration', 0.0)
        
        # Duration should be greater (though small due to short test time)
        self.assertGreaterEqual(new_duration, initial_duration)


if __name__ == '__main__':
    unittest.main()